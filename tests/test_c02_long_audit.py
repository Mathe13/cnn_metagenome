import json
import hashlib
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from viral_cnn.inference.audit_long import (
    audit_long_inputs,
    build_diagnostic_finding,
    calibrate_mean_logits,
    compare_query_and_taxon_accuracy,
    diagnose_window_logits,
    map_labels_to_canonical,
    quantify_training_overlap_detailed,
    sample_long_queries,
    shared_noise_conditions,
    stratify_long_errors,
    verify_window_logit_equivalence,
)
from viral_cnn.inference.voter import LongSequenceVoter
from viral_cnn.baselines.long_eval import C02_LONG_VARIANTS, build_c02_table12, validate_c02_prediction_artifacts


class TinyModel(torch.nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.fc = torch.nn.Linear(4, num_classes)

    def forward(self, x):
        if x.ndim == 2:
            oh = torch.nn.functional.one_hot(x.clamp(max=3), num_classes=4).float()
            oh[x >= 4] = 0
            x = oh.transpose(1, 2)
        return self.fc(x.mean(-1))


def paired_artifacts():
    truth = pd.DataFrame({"query_id": ["q1", "q2"], "truth_label": [0, 1]})
    artifacts = []
    for i, variant in enumerate(C02_LONG_VARIANTS):
        frame = pd.DataFrame({
            "query_id": ["q2", "q1"], "truth_label": [1, 0],
            "pred_mean_logits": [1, 0], "pred_majority_vote": [1, 2],
            "variant": variant, "long_sequence_protocol": "c02_audit_v1", "noise_rate": [0.0, 0.0], "handle_short": ["pad", "pad"],
        })
        artifacts.append({"variant": variant, "predictions": frame})
    return truth, artifacts


class TestLongInputAudit(unittest.TestCase):
    def test_audit_reports_observed_support_and_unknown_genomes(self):
        frame = pd.DataFrame({"query_id": ["a", "b", "c"], "sequence": ["A"*600, "N"*10001, "ACGT"*130], "truth_label": [1, 1, 2], "origin_len": [10000, 100000, 12000]})
        report = audit_long_inputs(frame, genome_id_col="accession")
        self.assertEqual(report["eligible_count"], 3)
        self.assertEqual(report["length"]["max"], 10001)
        self.assertEqual(report["taxon_counts"], {"1": 2, "2": 1})
        self.assertEqual(report["genome_identity"], {"status": "unknown", "genome_count": None})
        self.assertEqual(report["historical_sampling_rule"]["large_ge_100000"], 500)
        with tempfile.TemporaryDirectory() as tmp:
            source=Path(tmp)/"queries.csv"; source.write_bytes(b"query_id,sequence\na,ACGT\n")
            hashed=audit_long_inputs(frame,source_path=str(source),data_kind="contigs")
            self.assertEqual(len(hashed["source_sha256"]),64)
            self.assertEqual(hashed["source_files"][0]["sha256"],hashlib.sha256(source.read_bytes()).hexdigest())

    def test_sampling_and_overlap_provenance(self):
        frame = pd.DataFrame({"query_id": [f"q{i}" for i in range(8)], "sequence": ["ACGT"*30]*8})
        sample, meta = sample_long_queries(frame, limit=3, seed=42)
        self.assertEqual(meta["realized_count"], 3)
        self.assertEqual(meta["seed"], 42)
        self.assertEqual(meta["selected_query_ids"], sample.query_id.tolist())
        overlap = quantify_training_overlap_detailed(["NNNN", "ACGTACGT"], ["ACGTACGT"], k=5)
        self.assertEqual(overlap["query_denominator"], 2)
        self.assertEqual(overlap["queries_without_valid_kmers"], 1)
        self.assertIn("canonicalized", overlap["canonicalization"])
        from viral_cnn.notebook_runtime import read_parquet_hash_sample
        with tempfile.TemporaryDirectory() as tmp:
            parquet=Path(tmp)/"train.parquet"
            pd.DataFrame({"sequence_padded":["ACGT"*10 for _ in range(50)]}).to_parquet(parquet,index=False)
            sample_a,meta_a=read_parquet_hash_sample(parquet,columns=["sequence_padded"],sequence_column="sequence_padded",limit=7,seed=42)
            sample_b,meta_b=read_parquet_hash_sample(parquet,columns=["sequence_padded"],sequence_column="sequence_padded",limit=7,seed=42)
            self.assertEqual(sample_a.query_id.tolist(),sample_b.query_id.tolist())
            self.assertEqual(meta_a["population_count"],50)
            self.assertEqual(meta_a["realized_count"],7)
            self.assertEqual(meta_a["criterion"],meta_b["criterion"])
        contigs=[("q", "A"*4096, 0, "species-a")]
        noisy_a=shared_noise_conditions(contigs,[.05],seed=42)
        noisy_b=shared_noise_conditions(contigs,[.05],seed=42)
        self.assertEqual(noisy_a[.05],noisy_b[.05])
        self.assertNotEqual(noisy_a[.05][0][1],contigs[0][1])


class TestLabelMapping(unittest.TestCase):
    def test_mapping_requires_unique_identified_canonical_order(self):
        mapping = pd.DataFrame({"local_id": [17, 23], "species_id": ["s2", "s1"]})
        labels, source = map_labels_to_canonical([17, 23], mapping, source_id_col="local_id", canonical_id_col="species_id", canonical_order=["s1", "s2"])
        self.assertEqual(labels, [1, 0])
        self.assertEqual(source, [17, 23])
        with self.assertRaisesRegex(ValueError, "ambiguous"):
            map_labels_to_canonical([17], pd.concat([mapping, mapping.iloc[[0]]]), source_id_col="local_id", canonical_id_col="species_id", canonical_order=["s1", "s2"])
        with self.assertRaisesRegex(ValueError, "unmapped"):
            map_labels_to_canonical([99], mapping, source_id_col="local_id", canonical_id_col="species_id", canonical_order=["s1", "s2"])


class TestPairedInference(unittest.TestCase):
    def test_comparison_rejects_incomplete_or_misaligned_variants(self):
        truth, artifacts = paired_artifacts()
        valid = validate_c02_prediction_artifacts(artifacts, truth)
        self.assertEqual(set(valid), set(C02_LONG_VARIANTS))
        with self.assertRaisesRegex(ValueError, "missing variants"):
            validate_c02_prediction_artifacts(artifacts[:-1], truth)
        bad = [dict(x) for x in artifacts]
        bad[0]["predictions"] = pd.concat([bad[0]["predictions"], bad[0]["predictions"].iloc[[0]]])
        with self.assertRaisesRegex(ValueError, "duplicate query_id"):
            validate_c02_prediction_artifacts(bad, truth)
        cases=[]
        bad=[dict(x,predictions=x["predictions"].copy()) for x in artifacts]; bad[0]["predictions"]=bad[0]["predictions"].iloc[:1]; cases.append((bad,"query IDs missing"))
        bad=[dict(x,predictions=x["predictions"].copy()) for x in artifacts]; bad[0]["predictions"].loc[0,"truth_label"]=99; cases.append((bad,"truth mismatch"))
        bad=[dict(x,predictions=x["predictions"].copy()) for x in artifacts]; bad[0]["predictions"]["noise_rate"]=0.05; cases.append((bad,"condition mismatch"))
        bad=[dict(x,predictions=x["predictions"].copy()) for x in artifacts]; bad[0]["predictions"]["variant"]="convnext_noise_species"; cases.append((bad,"artifact variant identity mismatch"))
        bad=[dict(x) for x in artifacts]; bad[1]["variant"]=bad[0]["variant"]; cases.append((bad,"duplicate variant"))
        for invalid, message in cases:
            with self.subTest(message=message), self.assertRaisesRegex(ValueError,message):
                validate_c02_prediction_artifacts(invalid,truth)

    def test_table_metrics_use_observed_denominator_and_skip_is_error(self):
        truth, artifacts = paired_artifacts()
        artifacts[0]["predictions"].loc[0, "pred_mean_logits"] = np.nan
        table = build_c02_table12(artifacts, truth)
        row = table[(table.variant == C02_LONG_VARIANTS[0]) & (table.aggregation == "mean_logits")].iloc[0]
        self.assertEqual(row.N, 2)
        self.assertEqual(row.covered, 1)
        self.assertEqual(row.correct, 1)
        self.assertEqual(row.accuracy, 0.5)
        self.assertEqual(row.conditional_accuracy, 1.0)

    def test_prediction_artifact_contract_and_checkpoint_hashes(self):
        from viral_cnn.notebook_runtime import file_identity
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "checkpoint.pt"
            path.write_bytes(b"weights")
            identity = file_identity(path)
            self.assertEqual(len(identity["sha256"]), 64)
            _, artifacts = paired_artifacts()
            self.assertEqual(set(a["variant"] for a in artifacts), set(C02_LONG_VARIANTS))
            self.assertTrue(all(set(("query_id", "variant", "noise_rate", "handle_short", "long_sequence_protocol")) <= set(a["predictions"].columns) for a in artifacts))


class TestWindowDiagnostics(unittest.TestCase):
    def test_window_metadata_and_singleton_equivalence(self):
        model = TinyModel()
        res = LongSequenceVoter(model, window_size=512, stride=256, handle_short="pad").predict_sequence("ACGT"*128,include_window_logits=True)
        self.assertEqual(res.num_windows, 1)
        self.assertEqual(len(res.window_logits), 1)
        diff = verify_window_logit_equivalence(model, "ACGT"*128)
        self.assertLessEqual(diff, 1e-5)

    def test_strata_are_complete_and_empty_is_unavailable(self):
        df = pd.DataFrame({"length": [400, 512, 2000, 5000, 10001], "ambiguity_ratio": [0, .01, .05, .051, None], "truth_label": [0]*5, "pred_mean_logits": [0]*5, "pred_majority_vote": [0]*5, "taxon_id": [None]*5})
        audit = stratify_long_errors(df, length_bins=[(0, 512, "short"), (512, 2000, "middle"), (2000, 5000, "two_to_five_k"), (5000, 10001, "long"), (10001, 10002, "over_10k"), (10002, float("inf"), "empty")])
        self.assertEqual(sum(x["count"] for x in audit["by_length"].values()), 5)
        self.assertEqual(audit["by_length"]["empty"]["count"], 0)
        self.assertIsNone(audit["by_length"]["empty"]["accuracy_mean_logits"])
        self.assertEqual(audit["by_length"]["over_10k"]["count"], 1)
        self.assertEqual(audit["by_ambiguity"]["ambig_0_to_5_pct"]["count"], 2)
        self.assertEqual(audit["by_ambiguity"]["ambig_gt_5_pct"]["count"], 1)
        self.assertEqual(audit["by_ambiguity"]["ambig_unknown"]["count"], 1)
        self.assertEqual(audit["by_taxon"]["unknown"]["count"], 5)

    def test_composition_reports_query_and_taxon_weighting(self):
        frame = pd.DataFrame({"query_id": ["a", "b", "c", "d"], "truth_label": [0, 0, 0, 1], "pred_mean_logits": [0, 1, 1, 1], "taxon_id": ["large", "large", "large", "small"]})
        result = compare_query_and_taxon_accuracy(frame)
        self.assertEqual(result["n"], 4)
        self.assertAlmostEqual(result["query_weighted_accuracy"], .5)
        self.assertAlmostEqual(result["taxon_macro_accuracy"], 2/3)
        self.assertAlmostEqual(result["difference_pp_taxon_minus_query"], 100/6)
        self.assertEqual(result["genome_attribution"], "not assessed; genome IDs are not inferred")

    def test_logit_influence_requires_and_measures_window_logits(self):
        dominant = diagnose_window_logits([[0, 50], [4, 0], [4, 0]], truth_label=0)
        self.assertTrue(dominant["aggregation_disagrees"])
        self.assertGreaterEqual(dominant["leave_one_window_out_flips"], 1)
        self.assertEqual(diagnose_window_logits(None)["status"], "INCONCLUSIVE")
        smoke = diagnose_window_logits([[0, 1], [1, 0]], is_smoke_test=True)
        self.assertEqual(smoke["status"], "INCONCLUSIVE")

    def test_calibration_uses_mean_logits_and_fifteen_bins(self):
        result = calibrate_mean_logits([[[3, 0], [3, 0]], [[0, 3], [0, 3]]], [0, 1])
        self.assertEqual(result["bins"], 15)
        self.assertEqual(result["n"], 2)
        self.assertEqual(len(result["bin_metrics"]), 15)
        self.assertEqual(result["probability_source"], "softmax of mean pre-softmax window logits")

    def test_evidence_status_rejects_synthetic_support(self):
        with self.assertRaisesRegex(ValueError, "synthetic"):
            build_diagnostic_finding("claim", "test", 0, 2, [], [], status="SUPPORTED", is_smoke_test=True)
        finding = build_diagnostic_finding("claim", "test", None, 0, [], [], status="INCONCLUSIVE")
        self.assertEqual(finding["status"], "INCONCLUSIVE")


class TestNotebookContracts(unittest.TestCase):
    def test_notebooks_use_c00_run_manifests_and_artifact_logging(self):
        from viral_cnn.notebook_runtime import reserve_c02_long_run
        with tempfile.TemporaryDirectory() as tmp:
            run_id, run = reserve_c02_long_run(tmp, "long/01", "c02-check")
            self.assertEqual(run_id, "c02-check")
            self.assertIn("runs/c02-check/long/01", run.as_posix())
            with self.assertRaises(FileExistsError):
                reserve_c02_long_run(tmp, "long/01", "c02-check")
        for name in ("01_ncbi_download_and_prepare_database.ipynb", "02_cnn_long_voter_inference.ipynb", "03_cnn_long_voter_compare.ipynb"):
            doc = json.loads((Path("notebooks/reproduction/long") / name).read_text())
            source = "\n".join("".join(c.get("source", [])) for c in doc["cells"])
            self.assertIn("reserve_c02_long_run", source)
            self.assertIn("write_notebook_manifest", source)
            if "02_" in name or "03_" in name: self.assertIn("log_artifact_path", source)

    def test_smoke_and_empirical_evidence_are_separated(self):
        from viral_cnn.notebook_runtime import empirical_manifest_gate, preflight_notebook
        self.assertFalse(empirical_manifest_gate({"schema_version": 1, "run_id": "r", "notebook_id": "long/01", "git_commit": "abc", "is_smoke_test": True, "inputs": [], "outputs": [], "evidence": {"status": "synthetic"}})["accepted"])
        with tempfile.TemporaryDirectory() as tmp:
            report=preflight_notebook("long/02",tmp,smoke_test=False)
            self.assertEqual(report["status"],"BLOCKED")
            self.assertTrue(any(item["status"]=="BLOCKED" for item in report["requirements"]))

    def test_evidence_boundary_and_ad010_revision_are_documented(self):
        decision = Path("docs/adr/AD-020-long-sequence-evidence-boundary.md")
        report = Path("docs/long_sequence_audit.md")
        self.assertTrue(decision.exists())
        self.assertTrue(report.exists())
        text = decision.read_text() + report.read_text()
        self.assertIn("AD-010", text)
        self.assertIn("No C02 remote empirical execution", report.read_text())
        self.assertIn("historical", report.read_text().lower())
        self.assertIn("unknown", text.lower())


if __name__ == "__main__":
    unittest.main()
