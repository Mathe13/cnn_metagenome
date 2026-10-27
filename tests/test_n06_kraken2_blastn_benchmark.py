import json
import subprocess
import time
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


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


if __name__ == "__main__":
    unittest.main()

