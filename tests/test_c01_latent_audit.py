import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from viral_cnn.eval.latent import (
    EmbeddingArtifact,
    LinearProbeClassifier,
    evaluate_cosine_knn,
    evaluate_genome_disjoint_knn,
    extract_latent_embeddings,
    load_embedding_artifact,
    save_embedding_artifact,
    evaluate_silhouette_with_audit,
)
from viral_cnn.notebook_runtime import preflight_latent_inputs
from viral_cnn.data.parquet_dataset import ParquetBatchIterableDataset


def artifact(split, rows, *, variant="tiny", verified=True):
    rng = np.random.default_rng(4)
    return EmbeddingArtifact(
        rng.normal(size=(len(rows), 4)).astype(np.float32),
        np.arange(len(rows), dtype=np.int64) % 2,
        np.asarray(rows, dtype=str),
        {"verified": verified, "split": split, "variant": variant,
         "partitions": [f"{split}.parquet"], "partition_sha256": [f"partition-hash-{split}"],
         "checkpoint": "fixture.pt", "checkpoint_sha256": "checkpoint-hash",
         "class_order": [0, 1], "encoding": {"channels": ["A", "C", "G", "T"]},
         "sampling": {"rule": "all", "seed": 42}, "mode": "synthetic",
         "extractor_code_sha256": "code-hash"},
    )


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(3, 2)

    def forward(self, x):
        return self.fc(x)


class LatentArtifactTests(unittest.TestCase):
    def test_extraction_preserves_rows_and_rejects_partition_alias(self):
        model = TinyModel()
        x = torch.tensor([[1., 2., 3.], [4., 5., 6.]])
        batch = [(x[:1], torch.tensor([0]), ["part-a:0"]), (x[1:], torch.tensor([1]), ["part-b:0"])]
        feats, labels, ids = extract_latent_embeddings(model, batch, return_identities=True)
        np.testing.assert_allclose(feats, x.numpy())
        np.testing.assert_array_equal(labels, [0, 1])
        np.testing.assert_array_equal(ids, ["part-a:0", "part-b:0"])
        with tempfile.TemporaryDirectory() as tmp:
            import pyarrow as pa
            import pyarrow.parquet as pq
            parquet_path = Path(tmp) / "rows.parquet"
            pq.write_table(pa.table({"sequence_padded": ["ACGT", "TGCA"], "label": [0, 1]}), parquet_path)
            dataset = ParquetBatchIterableDataset(parquet_path, maxlen=4, batch_size=2, include_identity=True)
            _, _, parquet_ids = next(iter(dataset))
            import hashlib
            partition_hash = hashlib.sha256(parquet_path.read_bytes()).hexdigest()
            self.assertEqual(parquet_ids, [f"{partition_hash}:0", f"{partition_hash}:1"])
            partition = Path(tmp) / "same.parquet"
            checkpoint = Path(tmp) / "model.pt"
            partition.write_bytes(b"fixture")
            checkpoint.write_bytes(b"checkpoint")
            taxonomy = __import__("pandas").DataFrame({"model_label": [0], "Genus": ["g"], "Order": ["o"]})
            with self.assertRaisesRegex(ValueError, "Partition alias"):
                preflight_latent_inputs(
                    {"train": [partition], "validation": [partition], "test": [partition]},
                    taxonomy, {"tiny": checkpoint}, ["tiny"],
                )

    def test_cache_identity_rejects_legacy_and_stale_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            item = artifact("train", ["p:0", "p:1"])
            path = save_embedding_artifact(item, Path(tmp) / "features")
            expected = dict(item.manifest)
            expected.pop("verified")
            loaded = load_embedding_artifact(path, expected)
            self.assertEqual(len(loaded.features), 2)
            with self.assertRaisesRegex(ValueError, "split expected"):
                load_embedding_artifact(path, dict(expected, split="test"))
            legacy = Path(tmp) / "legacy.parquet"
            legacy.write_bytes(path.read_bytes())
            with self.assertRaisesRegex(ValueError, "manifest missing"):
                load_embedding_artifact(legacy, expected)
            corrupted = Path(tmp) / "corrupted.parquet"
            corrupted.write_bytes(path.read_bytes() + b"corruption")
            corrupted.with_suffix(".json").write_text(path.with_suffix(".json").read_text())
            with self.assertRaisesRegex(ValueError, "checksum"):
                load_embedding_artifact(corrupted, expected)

    def test_sampling_provenance_is_recorded(self):
        notebook_path = Path(__file__).resolve().parents[1] / "notebooks/reproduction/short/06_ictv_latent_space_and_hierarchy.ipynb"
        notebook = json.loads(notebook_path.read_text())
        source = "\n".join("".join(c.get("source", [])) for c in notebook["cells"])
        self.assertIn('"population_available": dataset.total_samples', source)
        self.assertIn('"extracted_count": count', source)
        self.assertIn('"rule": "none", "seed": ctx.seed', source)


class ProbeProtocolTests(unittest.TestCase):
    def test_test_artifact_cannot_be_fitted_even_with_train_label(self):
        model = LinearProbeClassifier(4, 2)
        before = {k: v.clone() for k, v in model.state_dict().items()}
        bad = artifact("test", ["t:0", "t:1"])
        train, validation = artifact("train", ["a:0", "a:1"]), artifact("validation", ["b:0", "b:1"])
        with self.assertRaisesRegex(ValueError, "requires train and validation"):
            model.fit_select_and_seal(bad, validation, epochs=1)
        for key, value in model.state_dict().items():
            torch.testing.assert_close(value, before[key])
        with self.assertRaisesRegex(ValueError, "cannot be trained on test"):
            model.fit(np.zeros((2, 4), np.float32), np.zeros(2), split_origin="test")

    def test_validation_selection_and_seal_are_irreversible(self):
        train = artifact("train", [f"tr:{i}" for i in range(20)])
        validation = artifact("validation", [f"va:{i}" for i in range(8)])
        probe = LinearProbeClassifier(4, 2)
        selection = probe.fit_select_and_seal(train, validation, epochs=3, batch_size=4)
        self.assertRegex(selection["weights_sha256"], r"^[0-9a-f]{64}$")
        self.assertEqual(selection["selection_rule"], "maximum validation Accuracy; earliest epoch wins ties")
        self.assertEqual(selection["selected_epoch"], min(x["epoch"] for x in selection["selection_history"] if x["validation_accuracy"] == max(y["validation_accuracy"] for y in selection["selection_history"])))
        test = artifact("test", [f"te:{i}" for i in range(4)])
        result = probe.evaluate_artifact(test)
        self.assertEqual(result["n_test_samples"], 4)
        self.assertEqual(result["row_ids"], test.row_ids.tolist())
        self.assertEqual(result["weights_sha256"], selection["weights_sha256"])
        with self.assertRaisesRegex(RuntimeError, "only once"):
            probe.evaluate_artifact(test)
        with self.assertRaisesRegex(RuntimeError, "cannot be fitted"):
            probe.fit_select_and_seal(train, validation, epochs=1)
        notebook_path = Path(__file__).resolve().parents[1] / "notebooks/reproduction/short/06_ictv_latent_space_and_hierarchy.ipynb"
        notebook = json.loads(notebook_path.read_text())
        cell = next("".join(c["source"]) for c in notebook["cells"] if c.get("cell_type") == "code" and "fit_select_and_seal" in "".join(c.get("source", [])))
        self.assertLess(cell.index("fit_select_and_seal"), cell.index('split="test"'))


class KNNAuditTests(unittest.TestCase):
    def test_knn_records_disjoint_identity_and_denominators(self):
        labels = np.repeat(np.arange(3), 10)
        features = np.eye(3, dtype=np.float32)[labels] + np.random.default_rng(2).normal(0, .01, (30, 3))
        ids = np.array([f"row-{i}" for i in range(len(labels))])
        result = evaluate_cosine_knn(features, labels, row_ids=ids)
        self.assertTrue(result["available"])
        self.assertEqual(result["k_requested"], 5)
        self.assertEqual(result["identity_overlap_count"], 0)
        self.assertEqual(result["self_neighbor_count"], 0)
        self.assertEqual(result["n_query"], len(result["query_ids"]))
        self.assertIn("true label appears", result["top5_definition"])

    def test_degenerate_support_is_unavailable(self):
        labels = np.arange(8)
        result = evaluate_cosine_knn(np.eye(8, dtype=np.float32), labels)
        self.assertFalse(result["available"])
        self.assertEqual(result["unavailable_reason"], "no_stratified_queries")
        self.assertEqual(result["k"], 0)
        singleton = evaluate_cosine_knn(np.ones((1, 2), np.float32), np.zeros(1, dtype=np.int64))
        self.assertFalse(singleton["available"])
        self.assertEqual(singleton["n_query"], 0)
        four_reference = evaluate_cosine_knn(np.ones((4, 2), np.float32), np.zeros(4, dtype=np.int64))
        self.assertFalse(four_reference["available"])
        self.assertEqual(four_reference["unavailable_reason"], "fewer_than_requested_neighbors")

    def test_taxonomic_supports_remain_independent(self):
        labels = np.repeat(np.arange(2), 5)
        result = evaluate_cosine_knn(np.eye(2, dtype=np.float32)[labels], labels)
        self.assertEqual(result["n_classes_total"], 2)
        self.assertEqual(result["n_classes_with_query"], 2)
        self.assertEqual(result["n_classes_reference_only"], 0)
        sil_labels = np.array([0, 0, 0, 1, 1, 1, 2])
        silhouette = evaluate_silhouette_with_audit(np.eye(3, dtype=np.float32)[sil_labels], sil_labels)
        self.assertEqual(silhouette["valid_classes_count"], 2)
        self.assertEqual(silhouette["excluded_classes_count"], 1)
        self.assertEqual(silhouette["retained_sample_count"], 6)
        self.assertEqual(silhouette["retained_sample_rate"], 6 / 7)
        self.assertEqual(silhouette["valid_class_rate"], 2 / 3)

    def test_genome_independence_requires_explicit_ids(self):
        # Operational row IDs are deliberately not interpreted as genome accessions.
        item = artifact("test", ["assembly:0", "assembly:1"])
        self.assertNotIn("genome_ids", item.manifest)
        labels = np.array([0, 0, 0, 1, 1, 1, 2])
        genome_ids = np.array(["a", "b", "c", "d", "e", "f", "g"])
        features = np.eye(3, dtype=np.float32)[labels]
        result = evaluate_genome_disjoint_knn(features, labels, genome_ids, k=1)
        self.assertEqual(result["excluded_species_labels"], [2])
        self.assertEqual(result["n_single_genome_species_excluded"], 1)
        self.assertTrue(result["zero_accession_overlap"])


class NotebookContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = Path(__file__).resolve().parents[1] / "notebooks/reproduction/short/06_ictv_latent_space_and_hierarchy.ipynb"
        cls.notebook = json.loads(path.read_text())
        cls.source = "\n".join("".join(cell.get("source", [])) for cell in cls.notebook["cells"])

    def test_all_five_variants_and_shared_test_identities(self):
        for name in ("resnet_clean_species", "resnet_noise_species", "convnext_zero_species", "convnext_transfer_species", "convnext_noise_species"):
            self.assertIn(name, self.source)
        self.assertIn("fit_select_and_seal", self.source)
        self.assertIn("evaluate_artifact(test", self.source)
        self.assertIn("Test_Row_IDs", self.source)
        self.assertIn("native_test_predictions", self.source)
        self.assertNotIn("native_test_logits", self.source)
        self.assertIn('"Linear_Probe_Test_Acc"', self.source)
        self.assertIn('"Linear_Probe_Macro_F1"', self.source)
        self.assertIn('"N"', self.source)
        self.assertIn('"Supported_Classes"', self.source)
        cell = next("".join(c["source"]) for c in self.notebook["cells"] if c.get("cell_type") == "code" and "fit_select_and_seal" in "".join(c.get("source", [])))
        self.assertIn("test_ids", cell)
        self.assertIn('"Test_Row_IDs_SHA256"', self.source)

    def test_notebook_preflight_and_run_manifest_contract(self):
        self.assertIn("preflight_latent_inputs", self.source)
        self.assertIn("reserve_run_directory", self.source)
        self.assertIn("run_id=ctx.run_id", self.source)
        self.assertIn("train", self.source)
        self.assertIn("validation", self.source)

    def test_smoke_and_empirical_inputs_are_separated(self):
        self.assertIn('"mode": "smoke" if ctx.is_smoke_test else "empirical"', self.source)
        self.assertIn('if not ctx.is_smoke_test:', self.source)
        self.assertIn('"genome_independence": "unknown: genome IDs unavailable for test fragments"', self.source)

    def test_evidence_boundary_is_documented(self):
        self.assertIn("linear_probe_audit.csv", self.source)
        self.assertIn("scientific_reproduction_certified", Path(__file__).resolve().parents[1].joinpath("src/viral_cnn/notebook_runtime.py").read_text())


if __name__ == "__main__":
    unittest.main()
