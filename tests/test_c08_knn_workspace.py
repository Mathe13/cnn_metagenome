"""Contracts for exact tiled KNN and compact latent-space audit (C08)."""

import ast
import contextlib
import hashlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pyarrow.parquet as pq
import torch

from viral_cnn.eval.latent import (
    EmbeddingArtifact,
    evaluate_cosine_knn,
    evaluate_silhouette_with_audit,
    save_embedding_artifact,
    save_knn_identity_audit,
)


NOTEBOOK = Path(__file__).resolve().parents[1] / "notebooks/reproduction/short/06_ictv_latent_space_and_hierarchy.ipynb"


def notebook_code():
    notebook = json.loads(NOTEBOOK.read_text())
    return "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"] if cell.get("cell_type") == "code")


def fixture(n=80, d=8):
    labels = np.arange(n, dtype=np.int64) % 4
    features = np.random.default_rng(42).normal(size=(n, d)).astype(np.float32)
    features += np.eye(4, d, dtype=np.float32)[labels] * 2
    ids = np.asarray([f"row:{i:04d}" for i in range(n)], dtype="U256")
    return features, labels, ids


def dense_neighbors(features, query_indices, reference_indices, k=5):
    x = features.astype(np.float64)
    x /= np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-30)
    similarity = x[query_indices] @ x[reference_indices].T
    return np.asarray([
        reference_indices[np.lexsort((reference_indices, -row))[:k]]
        for row in similarity
    ])


class GuardedFeatures:
    def __init__(self, features, maximum):
        self.values = features
        self.shape = features.shape
        self.maximum = maximum
        self.gathers = []

    def __len__(self):
        return len(self.values)

    def __array__(self, dtype=None):
        raise AssertionError("full feature array was materialized")

    def __getitem__(self, indices):
        result = self.values[indices]
        self.gathers.append(len(result))
        if len(result) > self.maximum:
            raise AssertionError(f"feature gather exceeded {self.maximum} rows")
        return result


class KNNWorkspaceTests(unittest.TestCase):
    smoke_namespace = None

    @classmethod
    def run_smoke_notebook(cls):
        if cls.smoke_namespace is None:
            namespace = {"__name__": "__main__"}
            notebook = json.loads(NOTEBOOK.read_text())
            with patch.dict(os.environ, {"REMOTE_SERVER": "False", "SMOKE_TEST": "1", "MPLCONFIGDIR": "/tmp"}):
                with contextlib.redirect_stdout(io.StringIO()):
                    for cell in notebook["cells"]:
                        if cell.get("cell_type") == "code":
                            exec("".join(cell["source"]), namespace)
            cls.smoke_namespace = namespace
        return cls.smoke_namespace

    def test_notebook_knn_defaults_overrides_and_device(self):
        code = notebook_code()
        tree = ast.parse(code)
        names = {"latent_knn_workspace_mb": 1024, "latent_knn_query_batch_size": 256}
        for name, expected in names.items():
            assignment = next(node for node in ast.walk(tree) if isinstance(node, ast.Assign)
                              and any(isinstance(target, ast.Name) and target.id == name for target in node.targets))
            namespace = {"os": os}
            with patch.dict(os.environ, {}, clear=True):
                exec(compile(ast.Module(body=[assignment], type_ignores=[]), "notebook", "exec"), namespace)
            self.assertEqual(namespace[name], expected)
        self.assertIn("workspace_mb=latent_knn_workspace_mb", code)
        self.assertIn("query_batch_size=latent_knn_query_batch_size", code)
        self.assertIn("device=ctx.device", code)
        self.assertIn('"latent_knn_device": str(ctx.device)', code)
        knn_call = next(node for node in ast.walk(tree) if isinstance(node, ast.Call)
                        and isinstance(node.func, ast.Name) and node.func.id == "evaluate_cosine_knn")
        device_argument = next(keyword.value for keyword in knn_call.keywords if keyword.arg == "device")
        self.assertEqual(ast.unparse(device_argument), "ctx.device")
        with patch.dict(os.environ, {"LATENT_KNN_WORKSPACE_MB": "64", "LATENT_KNN_QUERY_BATCH_SIZE": "17"}, clear=True):
            for name, expected in (("latent_knn_workspace_mb", 64), ("latent_knn_query_batch_size", 17)):
                assignment = next(node for node in ast.walk(tree) if isinstance(node, ast.Assign)
                                  and any(isinstance(target, ast.Name) and target.id == name for target in node.targets))
                namespace = {"os": os}
                exec(compile(ast.Module(body=[assignment], type_ignores=[]), "notebook", "exec"), namespace)
                self.assertEqual(namespace[name], expected)
        smoke = self.run_smoke_notebook()
        manifest = json.loads((smoke["output_dir"] / "manifest.json").read_text())
        configuration = manifest["configuration"]
        self.assertEqual(configuration["latent_knn_workspace_mb"], 1024)
        self.assertEqual(configuration["latent_knn_query_batch_size"], 256)
        self.assertEqual(configuration["latent_knn_device"], "cpu")
        self.assertEqual(len(configuration["knn_effective_blocks"]), 15)

    def test_invalid_workspace_and_too_small_budget_fail_before_matmul(self):
        features, labels, ids = fixture()
        with patch.object(torch, "mm", wraps=torch.mm) as mm:
            for workspace, batch in ((0, 4), (-1, 4), (1.5, 4), (1, 0), (1, -1)):
                with self.assertRaises(ValueError):
                    evaluate_cosine_knn(features, labels, row_ids=ids, workspace_mb=workspace, query_batch_size=batch)
            self.assertEqual(mm.call_count, 0)
        wide_features, wide_labels, wide_ids = fixture(40, 100000)
        with patch.object(torch, "mm", wraps=torch.mm) as mm:
            with self.assertRaisesRegex(RuntimeError, "requested.*minimum"):
                evaluate_cosine_knn(wide_features, wide_labels, row_ids=wide_ids, workspace_mb=1,
                                    query_batch_size=1)
            self.assertEqual(mm.call_count, 0)

    def test_tiled_knn_keeps_scientific_protocol(self):
        features, labels, ids = fixture(100)
        result = evaluate_cosine_knn(features, labels, row_ids=ids, k=5, test_size=0.2,
                                     random_state=42, reference_batch_size=7, query_batch_size=6,
                                     workspace_mb=1)
        self.assertEqual((result["n_reference"], result["n_query"]), (80, 20))
        self.assertEqual(result["k"], 5)
        self.assertEqual(result["identity_overlap_count"], 0)
        self.assertEqual(result["self_neighbor_count"], 0)
        self.assertEqual(result["protocol"], "c08_tiled_cosine_v1")
        np.testing.assert_array_equal(result["neighbor_indices"],
                                      dense_neighbors(features, result["query_indices"], result["reference_indices"]))
        rng = np.random.default_rng(42)
        expected_reference, expected_query = [], []
        for cls in np.unique(labels):
            positions = np.flatnonzero(labels == cls)
            rng.shuffle(positions)
            n_query = max(1, int(round(len(positions) * 0.2)))
            expected_query.extend(positions[:n_query])
            expected_reference.extend(positions[n_query:])
        np.testing.assert_array_equal(result["reference_indices"], expected_reference)
        np.testing.assert_array_equal(result["query_indices"], expected_query)
        neighbors = result["neighbor_indices"]
        expected_vote = np.asarray([np.bincount(labels[row], minlength=4).argmax() for row in neighbors])
        np.testing.assert_array_equal(result["y_pred"], expected_vote)

    def test_workspace_bounds_matmul_and_never_gathers_full_features(self):
        features, labels, ids = fixture(1031, 64)
        guarded = GuardedFeatures(features, 31)
        products = []
        original_mm = torch.mm

        def observed_mm(left, right):
            products.append((left.shape, right.shape))
            return original_mm(left, right)

        with patch.object(torch, "mm", side_effect=observed_mm):
            result = evaluate_cosine_knn(guarded, labels, row_ids=ids, workspace_mb=1,
                                         query_batch_size=17, reference_batch_size=31, compact=True)
        self.assertGreater(len(products), 1)
        self.assertLessEqual(max(guarded.gathers), 31)
        self.assertLess(max(left[0] * right[1] * 4 for left, right in products), 1024 * 1024)
        for left, right in products:
            query_rows, dimension = left
            reference_rows = right[1]
            estimated = 12 * dimension * (query_rows + reference_rows) + 64 * query_rows * reference_rows + 64 * query_rows * 5
            self.assertLessEqual(estimated, 1024 * 1024)
        self.assertLessEqual(result["workspace_peak_estimated_bytes"], 1024 * 1024)
        self.assertEqual(result["n_reference"] + result["n_query"], 1031)
        self.assertEqual(result["query_batch_size_effective"], 17)
        self.assertNotEqual(result["n_query"] % 17, 0)
        self.assertIn(result["n_query"] % 17, [left[0] for left, _ in products])

    def test_tiled_topk_matches_dense_and_breaks_ties_by_original_index(self):
        labels = np.repeat(np.arange(2), 10)
        features = np.tile(np.array([[1, 0], [0, 1]], dtype=np.float32), (10, 1))
        ids = np.asarray([f"tie:{i}" for i in range(20)])
        tiled = evaluate_cosine_knn(features, labels, row_ids=ids, reference_batch_size=3,
                                    query_batch_size=2, workspace_mb=1)
        one = evaluate_cosine_knn(features, labels, row_ids=ids, reference_batch_size=20,
                                  query_batch_size=2, workspace_mb=1)
        expected = dense_neighbors(features, tiled["query_indices"], tiled["reference_indices"])
        np.testing.assert_array_equal(tiled["neighbor_indices"], expected)
        np.testing.assert_array_equal(one["neighbor_indices"], expected)

    def test_tiled_knn_global_metrics_match_dense_reference(self):
        features, labels, ids = fixture(101)
        result = evaluate_cosine_knn(features, labels, row_ids=ids, reference_batch_size=9,
                                     query_batch_size=7, workspace_mb=1)
        neighbors = dense_neighbors(features, result["query_indices"], result["reference_indices"])
        reference_labels = labels[neighbors]
        predictions = np.asarray([np.bincount(row, minlength=4).argmax() for row in reference_labels])
        true = labels[result["query_indices"]]
        self.assertEqual(result["n_query"], len(true))
        self.assertAlmostEqual(result["global_knn_accuracy"], np.mean(predictions == true))
        self.assertAlmostEqual(result["global_knn_top5_accuracy"], np.mean((reference_labels == true[:, None]).any(axis=1)))
        expected_counts = np.bincount(true, minlength=4)
        actual = result["per_cluster_metrics"].set_index("class_label")["test_samples"]
        for cls, count in enumerate(expected_counts):
            self.assertEqual(int(actual.loc[cls]), int(count))
            class_mask = true == cls
            class_metrics = result["per_cluster_metrics"].set_index("class_label").loc[cls]
            self.assertAlmostEqual(class_metrics["knn_accuracy"], np.mean(predictions[class_mask] == true[class_mask]))
            self.assertAlmostEqual(class_metrics["knn_top5_accuracy"],
                                   np.mean((reference_labels[class_mask] == true[class_mask, None]).any(axis=1)))
        scores = []
        original_mm = torch.mm
        def capture_scores(left, right):
            product = original_mm(left, right)
            scores.append(product.detach().cpu().numpy().copy())
            return product
        with patch.object(torch, "mm", side_effect=capture_scores):
            single_tile = evaluate_cosine_knn(features, labels, row_ids=ids, reference_batch_size=101,
                                              query_batch_size=101, workspace_mb=1)
        self.assertEqual(len(scores), 1)
        x = features.astype(np.float64)
        x /= np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-30)
        expected_scores = x[single_tile["query_indices"]] @ x[single_tile["reference_indices"]].T
        np.testing.assert_allclose(scores[0], expected_scores, atol=1e-5, rtol=0)

    def test_rank_selection_and_silhouette_sample_before_feature_gather(self):
        features, labels, ids = fixture(400, 16)
        guarded = GuardedFeatures(features, 30)
        selected = np.arange(0, 400, 2, dtype=np.intp)
        mapped = labels[selected] // 2
        knn = evaluate_cosine_knn(guarded, mapped, row_ids=ids, selected_indices=selected,
                                  reference_batch_size=17, query_batch_size=11, workspace_mb=1)
        self.assertTrue(set(knn["query_indices"]).issubset(set(selected)))
        sil = evaluate_silhouette_with_audit(guarded, mapped, selected_indices=selected,
                                             max_samples=20, random_state=42)
        self.assertEqual(sil["actual_sample_size"], 20)
        self.assertLessEqual(max(guarded.gathers), 30)
        for divisor, rank in ((1, "Species"), (2, "Genus"), (4, "Order")):
            rank_labels = labels[selected] // divisor
            rank_result = evaluate_cosine_knn(guarded, rank_labels, row_ids=ids, selected_indices=selected,
                                              taxonomic_rank=rank, reference_batch_size=17,
                                              query_batch_size=11, workspace_mb=1)
            lookup = dict(zip(selected, rank_labels))
            np.testing.assert_array_equal(rank_result["y_true"],
                                          [lookup[index] for index in rank_result["query_indices"]])
            self.assertEqual(rank_result["taxonomic_rank"], rank)

    def test_compact_knn_audit_roundtrip_and_artifact_binding(self):
        features, labels, ids = fixture(80)
        with tempfile.TemporaryDirectory() as tmp:
            artifact = EmbeddingArtifact(features, labels, ids,
                                         {"verified": True, "variant": "synthetic", "split": "test"})
            artifact_path = save_embedding_artifact(artifact, Path(tmp) / "features", batch_size=7)
            artifact_hash = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
            self.assertEqual(json.loads(artifact_path.with_suffix(".json").read_text())["artifact_sha256"], artifact_hash)
            parquet_path = Path(tmp) / "neighbors.parquet"
            result = evaluate_cosine_knn(features, labels, row_ids=ids, compact=True,
                                         audit_output_path=parquet_path, embedding_artifact_sha256=artifact_hash,
                                         reference_batch_size=7, query_batch_size=6, workspace_mb=1)
            json_path = Path(tmp) / "knn_identity_audit.json"
            save_knn_identity_audit([result["audit_summary"]], json_path)
            document = json.loads(json_path.read_text())
            self.assertEqual(document["schema_version"], "c08_knn_audit_v1")
            self.assertNotIn('"reference_ids":', json_path.read_text())
            self.assertNotIn('"neighbor_ids":', json_path.read_text())
            self.assertEqual(document["evaluations"][0]["embedding_artifact_sha256"], artifact_hash)
            table = pq.read_table(parquet_path)
            self.assertEqual(table.num_rows, result["n_query"])
            self.assertEqual(table.schema.field("neighbor_indices").type.list_size, 5)
            np.testing.assert_array_equal(table.column("query_index").to_numpy(), result["query_indices"])
            restored = np.asarray(table.column("neighbor_indices").to_pylist(), dtype=np.int64)
            np.testing.assert_array_equal(restored, dense_neighbors(features, result["query_indices"], result["reference_indices"]))
            saved_ids = np.asarray(pq.read_table(artifact_path, columns=["row_id"]).column("row_id").to_pylist(), dtype=ids.dtype)
            self.assertEqual(document["evaluations"][0]["neighbor_ids_sha256"],
                             hashlib.sha256(saved_ids[restored].tobytes()).hexdigest())
            self.assertEqual(document["evaluations"][0]["neighbor_parquet_sha256"], hashlib.sha256(parquet_path.read_bytes()).hexdigest())

    def test_knn_audit_metadata_hashes_and_counts(self):
        features, labels, ids = fixture(80)
        with tempfile.TemporaryDirectory() as tmp:
            result = evaluate_cosine_knn(features, labels, row_ids=ids, compact=True,
                                         audit_output_path=Path(tmp) / "neighbors.parquet",
                                         embedding_artifact_sha256="abc", reference_batch_size=7,
                                         query_batch_size=6, workspace_mb=1)
            summary = result["audit_summary"]
            self.assertEqual(summary["reference_ids_sha256"], hashlib.sha256(ids[result["reference_indices"]].tobytes()).hexdigest())
            self.assertEqual(summary["query_ids_sha256"], hashlib.sha256(ids[result["query_indices"]].tobytes()).hexdigest())
            neighbors = dense_neighbors(features, result["query_indices"], result["reference_indices"])
            self.assertEqual(summary["neighbor_ids_sha256"], hashlib.sha256(ids[neighbors].tobytes()).hexdigest())
            self.assertEqual(summary["n_reference"], result["n_reference"])
            self.assertEqual(summary["n_query"], result["n_query"])
            self.assertEqual(summary["self_neighbor_count"], 0)
            self.assertEqual(summary["device"], "cpu")
            self.assertEqual(summary["protocol"], "c08_tiled_cosine_v1")
            self.assertEqual(summary["query_batch_size_effective"], 6)
            self.assertEqual(summary["reference_batch_size_effective"], 7)

    def test_audit_write_failure_is_atomic(self):
        features, labels, ids = fixture(40)
        with tempfile.TemporaryDirectory() as tmp:
            parquet_path = Path(tmp) / "neighbors.parquet"
            with patch("viral_cnn.eval.latent.os.replace", side_effect=OSError("disk full")):
                with self.assertRaisesRegex(OSError, "disk full"):
                    evaluate_cosine_knn(features, labels, row_ids=ids, compact=True,
                                        audit_output_path=parquet_path, embedding_artifact_sha256="abc")
            self.assertFalse(parquet_path.exists())
            self.assertEqual(list(Path(tmp).glob("*.tmp")), [])
            json_path = Path(tmp) / "knn_identity_audit.json"
            with patch("viral_cnn.eval.latent.os.replace", side_effect=OSError("disk full")):
                with self.assertRaisesRegex(OSError, "disk full"):
                    save_knn_identity_audit([{"protocol": "c08_tiled_cosine_v1"}], json_path)
            self.assertFalse(json_path.exists())
        namespace = {"__name__": "__main__"}
        notebook = json.loads(NOTEBOOK.read_text())
        with patch.dict(os.environ, {"REMOTE_SERVER": "False", "SMOKE_TEST": "1", "MPLCONFIGDIR": "/tmp"}):
            with patch("viral_cnn.eval.latent.save_knn_identity_audit", side_effect=OSError("disk full")):
                with contextlib.redirect_stdout(io.StringIO()):
                    with self.assertRaisesRegex(OSError, "disk full"):
                        for cell in notebook["cells"]:
                            if cell.get("cell_type") == "code":
                                exec("".join(cell["source"]), namespace)
        self.assertFalse((namespace["output_dir"] / "knn_identity_audit.json").exists())
        self.assertFalse((namespace["output_dir"] / "manifest.json").exists())
        source = notebook_code()
        self.assertLess(source.index("save_knn_identity_audit(knn_audits, knn_audit_path)"),
                        source.index("manifest = write_notebook_manifest("))

    def test_legacy_knn_details_and_notebook_compact_mode(self):
        features, labels, ids = fixture(40)
        result = evaluate_cosine_knn(features, labels, row_ids=ids, reference_batch_size=7, workspace_mb=1)
        self.assertEqual(result["n_reference"], len(result["reference_ids"]))
        self.assertEqual(result["n_query"], len(result["query_ids"]))
        self.assertEqual(result["neighbor_indices"].shape, (result["n_query"], 5))
        np.testing.assert_array_equal(result["neighbor_ids"], ids[result["neighbor_indices"]])
        source = notebook_code()
        self.assertIn("compact=True", source)
        self.assertNotIn('"reference_ids": knn_res["reference_ids"].tolist()', source)
        self.assertNotIn('"neighbor_ids": knn_res.get("neighbor_ids"', source)

    def test_notebook_silhouette_umap_limits_and_manifest(self):
        source = notebook_code()
        self.assertIn("200 if ctx.is_smoke_test else 5000", source)
        self.assertIn("umap_feats[:100]", source)
        self.assertIn('"silhouette_sample_sizes"', source)
        self.assertIn('"silhouette_excluded_classes"', source)
        self.assertIn('"umap_sample_size"', source)
        smoke = self.run_smoke_notebook()
        manifest = json.loads((smoke["output_dir"] / "manifest.json").read_text())
        configuration = manifest["configuration"]
        self.assertEqual(len(configuration["silhouette_sample_sizes"]), 15)
        self.assertEqual(len(configuration["silhouette_retained_classes"]), 15)
        self.assertEqual(len(configuration["silhouette_excluded_classes"]), 15)
        for key, sample_size in configuration["silhouette_sample_sizes"].items():
            self.assertLessEqual(sample_size, 200)
            self.assertEqual(sample_size, smoke["table5_df"].set_index(["Model", "Rank"])
                             .loc[(next(m["display"] for m in smoke["models_to_evaluate"] if m["name"] == key.split(":")[0]),
                                   key.split(":")[1]), "Silhouette_Actual_Sample_Size"])
            self.assertGreaterEqual(configuration["silhouette_retained_classes"][key], 0)
            self.assertGreaterEqual(configuration["silhouette_excluded_classes"][key], 0)
        self.assertLessEqual(configuration["umap_sample_size"], 100)


if __name__ == "__main__":
    unittest.main()
