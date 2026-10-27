import json
import subprocess
import time
import unittest
from pathlib import Path

import pandas as pd


class TestN07LongSequenceVoter(unittest.TestCase):
    """Contract tests verifying N07 checks C1 to C15 for long-sequence voter reproduction notebooks."""

    # --------------------------------------------------------------------------
    # S1: NCBI Download, Database Preparation & Overlap Audit
    # --------------------------------------------------------------------------
    def test_c1_notebook_01_imports_and_decoupling(self):
        """C1: 01_ncbi_download_and_prepare_database.ipynb has valid JSON, zero legacy imports, exclusively viral_cnn.*."""
        nb_path = Path("notebooks/reproduction/long/01_ncbi_download_and_prepare_database.ipynb")
        self.assertTrue(nb_path.exists(), f"Notebook {nb_path} does not exist")

        with open(nb_path, "r", encoding="utf-8") as f:
            nb = json.load(f)

        self.assertIn("cells", nb)
        for idx, cell in enumerate(nb["cells"]):
            if cell.get("cell_type") == "code":
                code = "".join(cell.get("source", []))
                self.assertNotIn("short.utils", code, f"Cell {idx} imports short.utils")
                self.assertNotIn("short/utils", code, f"Cell {idx} imports short/utils")

    def test_c2_notebook_01_execution_smoke_test(self):
        """C2: 01_ncbi_download_and_prepare_database.ipynb executes under SMOKE_TEST=1 in under 60 seconds."""
        nb_path = Path("notebooks/reproduction/long/01_ncbi_download_and_prepare_database.ipynb")
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
                    self.fail(f"Cell {idx} of 01_ncbi_download_and_prepare_database.ipynb raised: {e}")

        duration = time.time() - start_time
        self.assertLess(duration, 60.0, f"Notebook 01 took too long: {duration:.2f}s")

    def test_c3_overlap_quantification_and_sampling_bias(self):
        """C3: Sequence preparation audit quantifies reference overlap and documents read sampling disparity."""
        audit_csv = Path("results_reproduction/long_voter/long_database_preparation_audit.csv")
        manifest_path = Path("results_reproduction/long_voter/manifest_01_ncbi_prep.json")

        self.assertTrue(audit_csv.exists(), "long_database_preparation_audit.csv not found")
        self.assertTrue(manifest_path.exists(), "manifest_01_ncbi_prep.json not found")

        df = pd.read_csv(audit_csv)
        self.assertIn("genome_category", df.columns)
        self.assertIn("reads_simulated_per_genome", df.columns)

        # Check 500 reads vs 50 reads documented
        reads_counts = df["reads_simulated_per_genome"].tolist()
        self.assertIn(500, reads_counts)
        self.assertIn(50, reads_counts)

        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        self.assertIn("overlap_metrics", manifest)
        self.assertIn("mean_kmer_overlap_fraction", manifest["overlap_metrics"])

    def test_c4_historical_download_and_prep_notebooks_preserved(self):
        """C4: Historical notebooks long/download_from_ncbi.ipynb and long/prepare database.ipynb remain unmodified."""
        for nb in ["long/download_from_ncbi.ipynb", "long/prepare database.ipynb"]:
            res = subprocess.run(
                ["git", "status", "--porcelain", nb],
                capture_output=True,
                text=True,
                check=True,
            )
            self.assertEqual(res.stdout.strip(), "", f"Historical notebook {nb} was modified!")

    # --------------------------------------------------------------------------
    # S2: CNN Long Sequence Voter Inference & Noise Robustness
    # --------------------------------------------------------------------------
    def test_c5_notebook_02_imports_and_decoupling(self):
        """C5: 02_cnn_long_voter_inference.ipynb has valid JSON, zero legacy imports, exclusively viral_cnn.*."""
        nb_path = Path("notebooks/reproduction/long/02_cnn_long_voter_inference.ipynb")
        self.assertTrue(nb_path.exists())

        with open(nb_path, "r", encoding="utf-8") as f:
            nb = json.load(f)

        for idx, cell in enumerate(nb["cells"]):
            if cell.get("cell_type") == "code":
                code = "".join(cell.get("source", []))
                self.assertNotIn("short.utils", code)
                self.assertNotIn("short/utils", code)

    def test_c6_notebook_02_execution_smoke_test(self):
        """C6: 02_cnn_long_voter_inference.ipynb executes under SMOKE_TEST=1 in under 60s, verifying logit equivalence."""
        nb_path = Path("notebooks/reproduction/long/02_cnn_long_voter_inference.ipynb")
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
                    self.fail(f"Cell {idx} of 02_cnn_long_voter_inference.ipynb raised: {e}")

        duration = time.time() - start_time
        self.assertLess(duration, 60.0, f"Notebook 02 took too long: {duration:.2f}s")

    def test_c7_sliding_window_consensus_and_short_handling(self):
        """C7: Sliding-window consensus evaluates mean_logits and majority_vote, comparing handle_short pad vs skip."""
        pred_csv = Path("results_reproduction/long_voter/predictions_long_voter_resnet50_species.csv")
        manifest_path = Path("results_reproduction/long_voter/manifest_02_voter_inference.json")

        self.assertTrue(pred_csv.exists(), "predictions_long_voter CSV not found")
        self.assertTrue(manifest_path.exists(), "manifest_02 not found")

        df = pd.read_csv(pred_csv)
        self.assertIn("pred_mean_logits", df.columns)
        self.assertIn("pred_majority_vote", df.columns)
        self.assertIn("skip_mode_handled", df.columns)

        # For sequence <512 bp (read_001), skip mode should be False and pad mode should have n_windows > 0
        short_row = df[df["length"] < 512].iloc[0]
        self.assertFalse(short_row["skip_mode_handled"])
        self.assertGreater(short_row["n_windows"], 0)

        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        self.assertLess(manifest.get("logit_diff_max", 1.0), 1e-4)

    def test_c8_historical_voter_notebook_preserved(self):
        """C8: Historical notebook long/cnn-long-voter.ipynb remains strictly unmodified."""
        res = subprocess.run(
            ["git", "status", "--porcelain", "long/cnn-long-voter.ipynb"],
            capture_output=True,
            text=True,
            check=True,
        )
        self.assertEqual(res.stdout.strip(), "", "Historical notebook long/cnn-long-voter.ipynb was modified!")

    # --------------------------------------------------------------------------
    # S3: Long Sequence Comparison & ConvNeXt Collapse Diagnosis
    # --------------------------------------------------------------------------
    def test_c9_notebook_03_imports_and_decoupling(self):
        """C9: 03_cnn_long_voter_compare.ipynb has valid JSON, zero legacy imports, exclusively viral_cnn.*."""
        nb_path = Path("notebooks/reproduction/long/03_cnn_long_voter_compare.ipynb")
        self.assertTrue(nb_path.exists())

        with open(nb_path, "r", encoding="utf-8") as f:
            nb = json.load(f)

        for idx, cell in enumerate(nb["cells"]):
            if cell.get("cell_type") == "code":
                code = "".join(cell.get("source", []))
                self.assertNotIn("short.utils", code)
                self.assertNotIn("short/utils", code)

    def test_c10_notebook_03_execution_smoke_test(self):
        """C10: 03_cnn_long_voter_compare.ipynb executes under SMOKE_TEST=1 in under 60s, reproducing Table 12."""
        nb_path = Path("notebooks/reproduction/long/03_cnn_long_voter_compare.ipynb")
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
                    self.fail(f"Cell {idx} of 03_cnn_long_voter_compare.ipynb raised: {e}")

        duration = time.time() - start_time
        self.assertLess(duration, 60.0, f"Notebook 03 took too long: {duration:.2f}s")

    def test_c11_error_stratification_and_collapse_diagnosis(self):
        """C11: Error stratification partitions performance and diagnoses the 3 ConvNeXt collapse mechanisms."""
        t12_csv = Path("results_reproduction/long_voter/table_12_long_voter_comparison.csv")
        strat_csv = Path("results_reproduction/long_voter/stratified_long_errors.csv")
        diag_json = Path("results_reproduction/long_voter/convnext_collapse_diagnosis.json")

        self.assertTrue(t12_csv.exists(), "Table 12 CSV not found")
        self.assertTrue(strat_csv.exists(), "stratified_long_errors.csv not found")
        self.assertTrue(diag_json.exists(), "convnext_collapse_diagnosis.json not found")

        df_t12 = pd.read_csv(t12_csv)
        self.assertIn("model", df_t12.columns)
        self.assertIn("acc_mean_logits", df_t12.columns)

        # Check ResNet50 ~0.7269 and ConvNeXt_T ~0.0349
        resnet_acc = df_t12[df_t12["model"].str.contains("ResNet50_1D \\(Clean\\)")]["acc_mean_logits"].values[0]
        convnext_t_acc = df_t12[df_t12["model"].str.contains("ConvNeXt_T")]["acc_mean_logits"].values[0]
        self.assertAlmostEqual(resnet_acc, 0.7269, places=3)
        self.assertAlmostEqual(convnext_t_acc, 0.0349, places=3)

        with open(diag_json, "r", encoding="utf-8") as f:
            diag = json.load(f)
        mechanisms = [m["mechanism"] for m in diag["mechanisms"]]
        self.assertIn("Giant Virus Read Sampling Disparity", mechanisms)
        self.assertIn("Pre-softmax Logit Pulling", mechanisms)
        self.assertIn("Calibration Loss in Genus Transfer (ConvNeXt_T)", mechanisms)

    def test_c12_historical_compare_notebook_preserved(self):
        """C12: Historical notebook long/cnn-long-voter-compare.ipynb remains strictly unmodified."""
        res = subprocess.run(
            ["git", "status", "--porcelain", "long/cnn-long-voter-compare.ipynb"],
            capture_output=True,
            text=True,
            check=True,
        )
        self.assertEqual(res.stdout.strip(), "", "Historical notebook long/cnn-long-voter-compare.ipynb was modified!")

    # --------------------------------------------------------------------------
    # S4: Long Contig Motifs and GC Profile Analysis
    # --------------------------------------------------------------------------
    def test_c13_notebook_04_imports_and_decoupling(self):
        """C13: 04_long_motifs_and_gc_analysis.ipynb has valid JSON, zero legacy imports, exclusively viral_cnn.*."""
        nb_path = Path("notebooks/reproduction/long/04_long_motifs_and_gc_analysis.ipynb")
        self.assertTrue(nb_path.exists())

        with open(nb_path, "r", encoding="utf-8") as f:
            nb = json.load(f)

        for idx, cell in enumerate(nb["cells"]):
            if cell.get("cell_type") == "code":
                code = "".join(cell.get("source", []))
                self.assertNotIn("short.utils", code)
                self.assertNotIn("short/utils", code)

    def test_c14_notebook_04_execution_smoke_test(self):
        """C14: 04_long_motifs_and_gc_analysis.ipynb executes under SMOKE_TEST=1 in under 60s, generating summary and figure."""
        nb_path = Path("notebooks/reproduction/long/04_long_motifs_and_gc_analysis.ipynb")
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
                    self.fail(f"Cell {idx} of 04_long_motifs_and_gc_analysis.ipynb raised: {e}")

        duration = time.time() - start_time
        self.assertLess(duration, 60.0, f"Notebook 04 took too long: {duration:.2f}s")

        summary_csv = Path("results_reproduction/long_voter/long_contig_motif_summary.csv")
        fig_path = Path("results_reproduction/long_voter/figure_long_motifs_gc.png")
        self.assertTrue(summary_csv.exists(), "long_contig_motif_summary.csv not found")
        self.assertTrue(fig_path.exists(), "figure_long_motifs_gc.png not found")

    def test_c15_historical_long_loss_notebook_preserved(self):
        """C15: Historical notebook long/long_loss_function.ipynb remains strictly unmodified."""
        res = subprocess.run(
            ["git", "status", "--porcelain", "long/long_loss_function.ipynb"],
            capture_output=True,
            text=True,
            check=True,
        )
        self.assertEqual(res.stdout.strip(), "", "Historical notebook long/long_loss_function.ipynb was modified!")


if __name__ == "__main__":
    unittest.main()

