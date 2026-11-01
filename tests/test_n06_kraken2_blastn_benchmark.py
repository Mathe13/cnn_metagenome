import json
import shutil
import subprocess
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from viral_cnn.baselines.overlap import (
    BenchmarkArtifacts,
    load_prepared_baseline_benchmark,
)


class TestN06Kraken2BlastnBenchmark(unittest.TestCase):
    """Contract tests verifying N06 checks C1 to C10 for Kraken2, BLASTN, and CNN benchmark reproduction."""

    def test_c1_notebook_09_imports_and_decoupling(self):
        """C1: 09_benchmark_cnns_kraken2_blast_overlap.ipynb has valid JSON, zero short/utils/* imports, exclusively viral_cnn.*."""
        nb_path = Path("notebooks/reproduction/short/09_benchmark_cnns_kraken2_blast_overlap.ipynb")
        self.assertTrue(nb_path.exists(), f"Notebook {nb_path} does not exist")

        with open(nb_path, "r", encoding="utf-8") as f:
            nb = json.load(f)

        self.assertIn("cells", nb)
        self.assertGreater(len(nb["cells"]), 0)

        for idx, cell in enumerate(nb["cells"]):
            if cell.get("cell_type") == "code":
                code = "".join(cell.get("source", []))
                self.assertNotIn("short.utils", code, f"Cell {idx} imports short.utils")
                self.assertNotIn("short/utils", code, f"Cell {idx} imports short/utils")

    def test_c2_notebook_09_execution_smoke_test(self):
        """C2: 09_benchmark_cnns_kraken2_blast_overlap.ipynb executes under SMOKE_TEST=1 in under 60 seconds."""
        nb_path = Path("notebooks/reproduction/short/09_benchmark_cnns_kraken2_blast_overlap.ipynb")
        self.assertTrue(nb_path.exists())

        with open(nb_path, "r", encoding="utf-8") as f:
            nb = json.load(f)

        start_time = time.time()
        global_ns = {"__name__": "__main__"}
        for idx, cell in enumerate(nb["cells"]):
            if cell.get("cell_type") == "code":
                code = "".join(cell.get("source", []))
                try:
                    exec(code, global_ns)
                except Exception as e:
                    self.fail(f"Cell {idx} of 09_benchmark_cnns_kraken2_blast_overlap.ipynb raised: {e}")

        duration = time.time() - start_time
        self.assertLess(duration, 60.0, f"Notebook 09 took too long: {duration:.2f}s")

    def test_c3_abstention_accounting_and_coverage(self):
        """C3: Baseline predictions count abstentions (y_pred == -1) as errors in total accuracy while reporting coverage."""
        from viral_cnn.baselines.paired import evaluate_predictions

        y_true = np.array([0, 1, 2, 3])
        # Model abstains on query 1 and 3 (y_pred = -1), correctly predicts 0, wrong on 2
        y_pred = np.array([0, -1, 1, -1])

        metrics = evaluate_predictions(y_true, y_pred)
        # Global accuracy should be 1 / 4 = 0.25 (abstentions count in denominator)
        self.assertEqual(metrics["accuracy"], 0.25)
        # Coverage should be 2 / 4 = 0.50
        self.assertEqual(metrics["coverage"], 0.50)
        # Conditional accuracy should be 1 / 2 = 0.50
        self.assertEqual(metrics["conditional_accuracy"], 0.50)

    def test_c4_overlap_tripartite_stratification(self):
        """C4: Exact sequence overlap audit partitions queries into all, overlap, and no_exact_overlap."""
        audit_csv = Path("results_reproduction/baselines/exact_overlap_audit.csv")
        self.assertTrue(audit_csv.exists(), "exact_overlap_audit.csv not found")

        df = pd.read_csv(audit_csv)
        self.assertIn("query_id", df.columns)
        self.assertIn("train_count", df.columns)
        self.assertIn("has_overlap", df.columns)

        has_overlap = df["has_overlap"].values
        self.assertTrue(has_overlap.any(), "Should have some overlap queries")
        self.assertTrue((~has_overlap).any(), "Should have some non-overlap queries")

    def test_c5_stratified_metrics_export(self):
        """C5: Stratified metrics exported to benchmark_metrics_stratified.csv with Top-1, Wilson CI, Coverage."""
        metrics_csv = Path("results_reproduction/baselines/benchmark_metrics_stratified.csv")
        self.assertTrue(metrics_csv.exists(), "benchmark_metrics_stratified.csv not found")

        df = pd.read_csv(metrics_csv)
        expected_cols = {
            "method", "noise_rate", "stratum", "n", "accuracy",
            "accuracy_ci_low", "accuracy_ci_high", "coverage",
            "conditional_accuracy", "macro_f1"
        }
        self.assertTrue(expected_cols.issubset(df.columns), f"Missing columns in {df.columns}")

        strata = set(df["stratum"].unique())
        self.assertEqual(strata, {"all", "overlap", "no_exact_overlap"})

    def test_c6_historical_benchmark_notebook_preserved(self):
        """C6: Historical notebook short/benchmark_cnns_kraken2_blast_overlap.ipynb remains strictly unmodified."""
        res = subprocess.run(
            ["git", "status", "--porcelain", "short/benchmark_cnns_kraken2_blast_overlap.ipynb"],
            capture_output=True,
            text=True,
            check=True,
        )
        self.assertEqual(res.stdout.strip(), "", "Historical notebook short/benchmark_cnns_kraken2_blast_overlap.ipynb was modified!")

    def test_c7_paired_discordance_on_same_queries(self):
        """C7: Paired comparisons compute paired discordance counts on identical query IDs and export paired_discordance.csv."""
        disc_csv = Path("results_reproduction/baselines/paired_discordance.csv")
        self.assertTrue(disc_csv.exists(), "paired_discordance.csv not found")

        df = pd.read_csv(disc_csv)
        self.assertIn("method_a", df.columns)
        self.assertIn("method_b", df.columns)
        self.assertIn("both_correct", df.columns)
        self.assertIn("only_a_correct", df.columns)
        self.assertIn("only_b_correct", df.columns)
        self.assertIn("both_wrong", df.columns)
        self.assertIn("discordant_pct", df.columns)

    def test_c8_species_cluster_bootstrap(self):
        """C8: Paired difference confidence intervals estimated via cluster bootstrap (2,000 repeats, seed 42)."""
        boot_csv = Path("results_reproduction/baselines/species_cluster_bootstrap.csv")
        self.assertTrue(boot_csv.exists(), "species_cluster_bootstrap.csv not found")

        df = pd.read_csv(boot_csv)
        self.assertIn("method_a", df.columns)
        self.assertIn("method_b", df.columns)
        self.assertIn("diff_mean", df.columns)
        self.assertIn("ci_lower", df.columns)
        self.assertIn("ci_upper", df.columns)

    def test_c9_kmer_centroid_baseline(self):
        """C9: Optional k-mer baseline evaluated via KmerCentroidClassifier and recorded across overlap strata."""
        kmer_csv = Path("results_reproduction/baselines/kmer_baseline_metrics.csv")
        self.assertTrue(kmer_csv.exists(), "kmer_baseline_metrics.csv not found")

        df = pd.read_csv(kmer_csv)
        self.assertIn("method", df.columns)
        self.assertIn("stratum", df.columns)
        self.assertIn("accuracy", df.columns)

        # Accuracy on overlap should be higher than or equal to no_exact_overlap
        acc_overlap = df[df["stratum"] == "overlap"]["accuracy"].values[0]
        acc_no_overlap = df[df["stratum"] == "no_exact_overlap"]["accuracy"].values[0]
        self.assertGreaterEqual(acc_overlap, acc_no_overlap)

    def test_c10_figure_and_manifest_generated(self):
        """C10: Comparison figure and manifest_09_baselines.json generated under results_reproduction/baselines/."""
        fig_path = Path("results_reproduction/baselines/figure_benchmark_overlap.png")
        manifest_path = Path("results_reproduction/baselines/manifest_09_baselines.json")

        self.assertTrue(fig_path.exists(), "figure_benchmark_overlap.png not found")
        self.assertTrue(manifest_path.exists(), "manifest_09_baselines.json not found")

        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        self.assertIn("total_queries", manifest)
        self.assertIn("overlap_queries", manifest)
        self.assertIn("methods_evaluated", manifest)
        self.assertEqual(manifest.get("bootstrap_repeats"), 2000)


    def test_r06_load_prepared_baseline_benchmark_success(self):
        """R06 C1: load_prepared_baseline_benchmark loads valid truth and prediction artifacts."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            comp_path = tmp_path / "comparison_results"
            comp_path.mkdir(parents=True)

            qids = [f"q_{i:04d}" for i in range(20)]
            labels = [i % 5 for i in range(20)]
            train_counts = [2 if i < 15 else 0 for i in range(20)]
            exact_overlap_df = pd.DataFrame({
                "query_id": qids,
                "label": labels,
                "train_count": train_counts,
                "has_overlap": [c > 0 for c in train_counts],
            })
            exact_overlap_df.to_csv(tmp_path / "exact_overlap.csv", index=False)

            methods = [
                "resnet50_species", "resnet_noise_species",
                "convnext_zero_species", "convnext_transfer_species", "convnext_noise_species",
                "Kraken2-flat", "BLASTN"
            ]
            noise_levels = [0.0, 0.02, 0.05, 0.10, 0.15]

            for m in methods:
                for noise in noise_levels:
                    tag = f"noise_{round(noise * 100):02d}"
                    pred_df = pd.DataFrame({
                        "query_id": qids,
                        "prediction": labels,
                        "top1": labels,
                        "top2": [(l + 1) % 5 for l in labels],
                        "top3": [(l + 2) % 5 for l in labels],
                        "top4": [(l + 3) % 5 for l in labels],
                        "top5": [(l + 4) % 5 for l in labels],
                    })
                    pred_df.to_parquet(comp_path / f"predictions_{m}_{tag}.parquet", index=False)

            runtime_records = [{"method": "resnet50_species", "noise": 0.0, "seconds": 1.23, "n": 20}]
            with (comp_path / "runtime.jsonl").open("w", encoding="utf-8") as f:
                f.write(json.dumps(runtime_records[0]) + "\n")

            artifacts = load_prepared_baseline_benchmark(tmp_path)
            self.assertIsInstance(artifacts, BenchmarkArtifacts)
            self.assertEqual(len(artifacts.audit_df), 20)
            self.assertEqual(int(artifacts.strata_masks["overlap"].sum()), 15)
            self.assertEqual(int(artifacts.strata_masks["no_exact_overlap"].sum()), 5)
            self.assertEqual(set(artifacts.predictions.keys()), set(methods))
            for m in methods:
                for noise in noise_levels:
                    self.assertEqual(len(artifacts.predictions[m][noise]), 20)
                    self.assertIsNotNone(artifacts.top_candidates[m][noise])
                    self.assertEqual(artifacts.top_candidates[m][noise].shape, (20, 5))
            self.assertIsNotNone(artifacts.runtime)
            self.assertEqual(len(artifacts.runtime), 1)

    def test_r06_load_prepared_baseline_benchmark_validation_errors(self):
        """R06 C2: load_prepared_baseline_benchmark detects duplicate or mismatched query_ids."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            comp_path = tmp_path / "comparison_results"
            comp_path.mkdir(parents=True)

            # Duplicate query_id in truth table
            dup_truth_df = pd.DataFrame({
                "query_id": ["q_01", "q_01"],
                "label": [0, 1],
                "train_count": [1, 0],
            })
            dup_truth_df.to_csv(tmp_path / "exact_overlap.csv", index=False)

            with self.assertRaises(ValueError):
                load_prepared_baseline_benchmark(tmp_path)

            # Valid truth table
            valid_truth_df = pd.DataFrame({
                "query_id": ["q_01", "q_02"],
                "label": [0, 1],
                "train_count": [1, 0],
            })
            valid_truth_df.to_csv(tmp_path / "exact_overlap.csv", index=False)

            # Duplicate query_id in prediction file
            dup_pred_df = pd.DataFrame({
                "query_id": ["q_01", "q_01"],
                "prediction": [0, 1],
            })
            dup_pred_df.to_parquet(comp_path / "predictions_resnet50_species_noise_00.parquet", index=False)

            with self.assertRaises(ValueError):
                load_prepared_baseline_benchmark(tmp_path, methods=["resnet50_species"], noise_levels=[0.0])

            # Mismatched query_ids
            mismatch_pred_df = pd.DataFrame({
                "query_id": ["q_01", "q_99"],
                "prediction": [0, 1],
            })
            mismatch_pred_df.to_parquet(comp_path / "predictions_resnet50_species_noise_00.parquet", index=False)

            with self.assertRaises(ValueError):
                load_prepared_baseline_benchmark(tmp_path, methods=["resnet50_species"], noise_levels=[0.0])

    def test_r06_load_prepared_baseline_benchmark_missing_dir(self):
        """R06 C3: load_prepared_baseline_benchmark raises FileNotFoundError when directory or artifacts are missing."""
        non_existent = Path("/non_existent_benchmark_directory_r06_xyz")
        with self.assertRaises(FileNotFoundError):
            load_prepared_baseline_benchmark(non_existent)

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            # Empty directory without exact_overlap.csv
            with self.assertRaises(FileNotFoundError):
                load_prepared_baseline_benchmark(tmp_path)

            # Directory with truth but missing predictions
            valid_truth_df = pd.DataFrame({
                "query_id": ["q_01"],
                "label": [0],
                "has_overlap": [True],
            })
            valid_truth_df.to_csv(tmp_path / "exact_overlap.csv", index=False)

            with self.assertRaises(FileNotFoundError):
                load_prepared_baseline_benchmark(tmp_path, methods=["resnet50_species"], noise_levels=[0.0])

    def test_r06_notebook_09_fail_fast_outside_smoke(self):
        """R06 C4: Notebook 09 raises FileNotFoundError when is_smoke_test=False and kraken_blast_dir does not exist."""
        nb_path = Path("notebooks/reproduction/short/09_benchmark_cnns_kraken2_blast_overlap.ipynb")
        with open(nb_path, "r", encoding="utf-8") as f:
            nb = json.load(f)

        # Execute Cell 1 and Cell 3 with is_smoke_test = False and a non-existent kraken_blast_dir
        c1_code = "".join(nb["cells"][1]["source"])
        c3_code = "".join(nb["cells"][3]["source"])

        global_ns = {"__name__": "__main__"}
        exec(c1_code, global_ns)
        # Override runtime context
        global_ns["ctx"].is_smoke_test = False
        global_ns["ctx"].topology.paths["short"]["kraken_blast_dir"] = "/non_existent_cluster_kraken_blast_dir"

        with self.assertRaises(FileNotFoundError):
            exec(c3_code, global_ns)

    def test_r06_notebook_09_no_synthetic_fabrication_outside_smoke(self):
        """R06 C5: Notebook 09 does not execute random prediction generation when is_smoke_test=False."""
        nb_path = Path("notebooks/reproduction/short/09_benchmark_cnns_kraken2_blast_overlap.ipynb")
        with open(nb_path, "r", encoding="utf-8") as f:
            nb = json.load(f)

        c5_code = "".join(nb["cells"][5]["source"])
        # Check that acc_target and rng.uniform are strictly inside ctx.is_smoke_test branch
        self.assertIn("if ctx.is_smoke_test:", c5_code)
        self.assertIn("benchmark_artifacts.predictions", c5_code)

        parts = c5_code.split("\nelse:")
        self.assertGreater(len(parts), 1, "Cell 5 must contain an outer else branch for non-smoke execution")
        non_smoke_branch = parts[1]
        self.assertNotIn("acc_target", non_smoke_branch)
        self.assertNotIn("rng.uniform", non_smoke_branch)

    def test_r06_notebook_09_smoke_execution_and_exports(self):
        """R06 C6: Notebook 09 executes end-to-end in smoke mode in <60s, producing 40 queries and all exports."""
        nb_path = Path("notebooks/reproduction/short/09_benchmark_cnns_kraken2_blast_overlap.ipynb")
        with open(nb_path, "r", encoding="utf-8") as f:
            nb = json.load(f)

        start_time = time.time()
        global_ns = {"__name__": "__main__"}
        for idx, cell in enumerate(nb["cells"]):
            if cell.get("cell_type") == "code":
                code = "".join(cell.get("source", []))
                try:
                    exec(code, global_ns)
                except Exception as e:
                    self.fail(f"Cell {idx} raised exception in smoke test: {e}")

        duration = time.time() - start_time
        self.assertLess(duration, 60.0, f"Notebook 09 took {duration:.2f}s (>60s)")

        # Verify exports
        base_dir = Path("results_reproduction/baselines")
        self.assertTrue((base_dir / "exact_overlap_audit.csv").exists())
        self.assertTrue((base_dir / "benchmark_metrics_stratified.csv").exists())
        self.assertTrue((base_dir / "paired_discordance.csv").exists())
        self.assertTrue((base_dir / "species_cluster_bootstrap.csv").exists())
        self.assertTrue((base_dir / "figure_benchmark_overlap.png").exists())
        self.assertTrue((base_dir / "manifest_09_baselines.json").exists())

        audit_df = pd.read_csv(base_dir / "exact_overlap_audit.csv")
        self.assertEqual(len(audit_df), 40)
        self.assertEqual(int(audit_df["has_overlap"].sum()), 20)

        with open(base_dir / "manifest_09_baselines.json") as f:
            manifest = json.load(f)
        self.assertEqual(manifest["total_queries"], 40)
        self.assertEqual(manifest["overlap_queries"], 20)
        self.assertEqual(manifest["bootstrap_repeats"], 2000)


if __name__ == "__main__":
    unittest.main()

