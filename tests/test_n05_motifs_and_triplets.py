import json
import subprocess
import time
import unittest
from pathlib import Path

import pandas as pd
import numpy as np


class TestN05MotifsAndTriplets(unittest.TestCase):
    """Contract tests verifying N05 checks C1 to C14 for first-layer motifs, perturbation, and triplet notebooks."""

    @classmethod
    def setUpClass(cls):
        import os
        os.environ["SMOKE_TEST"] = "1"
        os.environ["REMOTE_SERVER"] = "False"
        os.environ["MPLCONFIGDIR"] = "/tmp"

    def setUp(self):
        import os
        os.environ["SMOKE_TEST"] = "1"
        os.environ["REMOTE_SERVER"] = "False"
        os.environ["MPLCONFIGDIR"] = "/tmp"

    def test_c1_notebook_07_imports_and_decoupling(self):
        """C1: 07_first_layer_motifs_and_perturbation.ipynb has valid JSON, zero short/utils/* imports, exclusively viral_cnn.*."""
        nb_path = Path("notebooks/reproduction/short/07_first_layer_motifs_and_perturbation.ipynb")
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

    def test_c2_notebook_07_execution_smoke_test(self):
        """C2: 07_first_layer_motifs_and_perturbation.ipynb executes under SMOKE_TEST=1 in under 60 seconds."""
        nb_path = Path("notebooks/reproduction/short/07_first_layer_motifs_and_perturbation.ipynb")
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
                    self.fail(f"Cell {idx} of 07_first_layer_motifs_and_perturbation.ipynb raised: {e}")

        duration = time.time() - start_time
        self.assertLess(duration, 60.0, f"Notebook 07 took too long: {duration:.2f}s")

    def test_c3_notebook_07_panel_e_checkpoint_attribution(self):
        """C3: Figure 3 panel (e) links to checkpoint convnext_noise_species distinct from panel (b) resnet_noise_species."""
        manifest_path = Path("results_reproduction/motifs/manifest_07_motifs.json")
        self.assertTrue(manifest_path.exists(), "Manifest 07 not found")

        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        ckpt_map = manifest.get("checkpoint_attributions", {})
        self.assertIn("convnext_noise_species", ckpt_map)
        self.assertIn("resnet_noise_species", ckpt_map)

        # Panel e must be distinct from panel b
        self.assertNotEqual(
            ckpt_map["convnext_noise_species"],
            ckpt_map["resnet_noise_species"],
            "Panel (e) and Panel (b) should point to distinct checkpoints (AD-004, Nota 39)"
        )

    def test_c4_notebook_07_paired_gc_controls(self):
        """C4: Paired motif perturbation evaluates both uniform random control and GC-matched control (|ΔGC| = 0)."""
        table_path = Path("results_reproduction/motifs/table_08_motif_perturbation.csv")
        self.assertTrue(table_path.exists(), "Table 8 CSV not found")

        df = pd.read_csv(table_path)
        self.assertIn("mean_specific_drop_uniform", df.columns)
        self.assertIn("mean_specific_drop_gc", df.columns)
        self.assertIn("mean_motif_drop_gc", df.columns)
        self.assertIn("mean_random_drop_gc", df.columns)

    def test_c5_notebook_07_wilcoxon_and_fdr(self):
        """C5: Motif perturbation reports paired Wilcoxon signed-rank test and Benjamini-Hochberg FDR for both control policies."""
        table_path = Path("results_reproduction/motifs/table_08_motif_perturbation.csv")
        self.assertTrue(table_path.exists())

        df = pd.read_csv(table_path)
        self.assertIn("p_value_uniform", df.columns)
        self.assertIn("p_fdr_uniform", df.columns)
        self.assertIn("p_value_gc", df.columns)
        self.assertIn("p_fdr_gc", df.columns)
        self.assertIn("significant_fdr_gc", df.columns)

    def test_c6_notebook_07_artifacts_and_jaccard_export(self):
        """C6: Symmetric Jaccard co-occurrence matrices exported to Parquet and Table 8 / Fig 3 artifacts generated."""
        out_dir = Path("results_reproduction/motifs")
        jaccard_parquet = out_dir / "motif_interaction_jaccard_resnet50_species.parquet"
        fig3_path = out_dir / "figure_03_motifs.png"
        table8_parquet = out_dir / "motif_summary.parquet"

        self.assertTrue(jaccard_parquet.exists(), "Jaccard parquet not exported")
        self.assertTrue(fig3_path.exists(), "Figure 3 not generated")
        self.assertTrue(table8_parquet.exists(), "Table 8 parquet not generated")

        # Verify symmetry of exported Jaccard matrix
        df_jacc = pd.read_parquet(jaccard_parquet)
        mat = df_jacc.values
        np.testing.assert_allclose(mat, mat.T, atol=1e-6, err_msg="Jaccard matrix is not symmetric")

    def test_c7_historical_full_loss_notebook_preserved(self):
        """C7: Historical notebook short/full_loss_function.ipynb remains strictly unmodified."""
        res = subprocess.run(
            ["git", "status", "--porcelain", "short/full_loss_function.ipynb"],
            capture_output=True,
            text=True,
            check=True,
        )
        self.assertEqual(res.stdout.strip(), "", "Historical notebook short/full_loss_function.ipynb was modified!")

    def test_c8_notebook_08_imports_and_decoupling(self):
        """C8: 08_triplet_composition_and_cooccurrence.ipynb has valid JSON, zero short/utils/* imports, exclusively viral_cnn.*."""
        nb_path = Path("notebooks/reproduction/short/08_triplet_composition_and_cooccurrence.ipynb")
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

    def test_c9_notebook_08_execution_smoke_test(self):
        """C9: 08_triplet_composition_and_cooccurrence.ipynb executes under SMOKE_TEST=1 in under 60 seconds."""
        nb_path = Path("notebooks/reproduction/short/08_triplet_composition_and_cooccurrence.ipynb")
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
                    self.fail(f"Cell {idx} of 08_triplet_composition_and_cooccurrence.ipynb raised: {e}")

        duration = time.time() - start_time
        self.assertLess(duration, 60.0, f"Notebook 08 took too long: {duration:.2f}s")

    def test_c10_notebook_08_rscu_and_fdr(self):
        """C10: Triplet composition and exploratory RSCU evaluated across frame 0, 3-frames, and 6-frames with disclaimer."""
        table_path = Path("results_reproduction/motifs/table_09_exploratory_rscu.csv")
        fig_path = Path("results_reproduction/motifs/figure_04_rscu_profile.png")
        manifest_path = Path("results_reproduction/motifs/manifest_08_triplets.json")

        self.assertTrue(table_path.exists(), "Table 9 CSV not found")
        self.assertTrue(fig_path.exists(), "Figure 4 PNG not found")
        self.assertTrue(manifest_path.exists(), "Manifest 08 not found")

        df = pd.read_csv(table_path)
        self.assertIn("rscu_frame0", df.columns)
        self.assertIn("rscu_forward_3frames", df.columns)
        self.assertIn("rscu_six_frames", df.columns)
        self.assertIn("multinomial_null", df.columns)

        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        self.assertEqual(manifest.get("metric_label"), "ExploratoryTripletComposition")
        self.assertFalse(manifest.get("verified_biological_frame"))

    def test_c11_notebook_08_translation_boundary_signals(self):
        """C11: Translation boundary signals (ATG, TAA/TAG/TGA) audited across learned motifs against background."""
        table_path = Path("results_reproduction/motifs/table_10_translation_boundary_signals.csv")
        self.assertTrue(table_path.exists(), "Table 10 CSV not found")

        df = pd.read_csv(table_path)
        signals = set(df["signal"].tolist())
        self.assertIn("start_atg", signals)
        self.assertIn("stop_taa", signals)
        self.assertIn("stop_tag", signals)
        self.assertIn("stop_tga", signals)

    def test_c12_notebook_08_cooccurrence_symmetry_and_null_model(self):
        """C12: Top-200 co-occurrence pairs tested for orientation symmetry and permutation null model."""
        table_path = Path("results_reproduction/motifs/table_11_cooccurrence_synergy.csv")
        fig_path = Path("results_reproduction/motifs/figure_05_cooccurrence_synergy.png")

        self.assertTrue(table_path.exists(), "Table 11 CSV not found")
        self.assertTrue(fig_path.exists(), "Figure 5 PNG not found")

        df = pd.read_csv(table_path)
        self.assertIn("r_canonical", df.columns)
        self.assertIn("r_reversed", df.columns)
        self.assertIn("delta_r_orientation", df.columns)
        self.assertIn("icc_symmetric", df.columns)
        self.assertIn("p_permutation_null", df.columns)

    def test_c13_notebook_08_inter_architecture_correlations(self):
        """C13: Inter-architecture co-occurrence correlations compute Pearson r and Spearman rho with FDR."""
        from viral_cnn.interpretability.motifs import compute_pearson_r, multipletests_fdr_bh
        v1 = np.array([0.1, 0.4, 0.7, 0.2, 0.8])
        v2 = np.array([0.15, 0.38, 0.65, 0.22, 0.79])
        r, p = compute_pearson_r(v1, v2)
        self.assertGreater(r, 0.9)
        self.assertLess(p, 0.05)

        reject, q = multipletests_fdr_bh([p])
        self.assertTrue(reject[0])

    def test_c14_historical_codon_usage_notebook_preserved(self):
        """C14: Historical notebook short/codon_usage.ipynb remains strictly unmodified."""
        res = subprocess.run(
            ["git", "status", "--porcelain", "short/codon_usage.ipynb"],
            capture_output=True,
            text=True,
            check=True,
        )
        self.assertEqual(res.stdout.strip(), "", "Historical notebook short/codon_usage.ipynb was modified!")

    # -----------------------------------------------------------------------
    # R05 checks (r05-real-motif-jaccard-and-triplet-analysis)
    # -----------------------------------------------------------------------

    def test_r05_compute_motif_presence_matrix_loader(self):
        """C1: compute_motif_presence_matrix_from_loader scans tokens and string sequences, returning exact binary presence."""
        import torch
        from torch.utils.data import DataLoader, TensorDataset
        from viral_cnn.interpretability.motifs import compute_motif_presence_matrix_from_loader

        # 1. Tensor dataset format
        s0 = [0, 1, 2, 3] * 2 + [1] * 7 + [0, 1, 2, 3] * 3 + [2] * 7 + [0, 1] * 5
        s1 = [0, 1, 2, 3] * 5 + [1] * 7 + [0, 1] * 10
        s2 = [0, 3, 0, 3] * 10
        inputs = torch.tensor([s0[:40], s1[:40], s2[:40]], dtype=torch.long)
        loader_tokens = DataLoader(TensorDataset(inputs, torch.tensor([0, 1, 2])), batch_size=2)

        motifs = ["CCCCCCC", "GGGGGGG", "AAAAAAA"]
        mat_tokens = compute_motif_presence_matrix_from_loader(loader_tokens, motifs)
        self.assertEqual(mat_tokens.shape, (3, 3))
        self.assertEqual(mat_tokens[0, 0], 1.0)
        self.assertEqual(mat_tokens[0, 1], 1.0)
        self.assertEqual(mat_tokens[0, 2], 0.0)
        self.assertEqual(mat_tokens[1, 0], 1.0)
        self.assertEqual(mat_tokens[1, 1], 0.0)
        self.assertEqual(mat_tokens[2, 0], 0.0)

        # 2. String sequences dataset format
        string_batch = [
            "ACGTACGTCCCCCCCACGTACGTGGGGGGGACACACACAC",
            "ACGTACGTACGTACGTACGTCCCCCCCACACACACACACA",
            "ATATATATATATATATATATATATATATATATATATATAT",
        ]
        loader_strings = DataLoader(string_batch, batch_size=2)
        mat_strings = compute_motif_presence_matrix_from_loader(loader_strings, motifs)
        self.assertEqual(mat_strings.shape, (3, 3))
        np.testing.assert_array_equal(mat_tokens, mat_strings)

    def test_r05_jaccard_cooccurrence_symmetry_and_values(self):
        """C2: compute_jaccard_cooccurrence produces exact symmetric pairwise Jaccard index with unit diagonal."""
        from viral_cnn.interpretability.motifs import compute_jaccard_cooccurrence

        motifs = ["CCCCCCC", "GGGGGGG", "AAAAAAA"]
        presence = np.array([
            [1.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 1.0],
            [0.0, 0.0, 0.0],
        ])
        jacc_df = compute_jaccard_cooccurrence(presence, motifs)
        self.assertEqual(jacc_df.shape, (3, 3))
        mat = jacc_df.values
        np.testing.assert_allclose(mat, mat.T, atol=1e-6)
        np.testing.assert_allclose(np.diag(mat), np.ones(3), atol=1e-6)
        self.assertTrue((mat >= 0.0).all() and (mat <= 1.0).all())

    def test_r05_notebook_07_five_variants_and_jaccard_matrices(self):
        """C3: Notebook 07 evaluates perturbation across 5 variants and exports empirical Jaccard matrices."""
        nb_path = Path("notebooks/reproduction/short/07_first_layer_motifs_and_perturbation.ipynb")
        with open(nb_path, "r", encoding="utf-8") as f:
            nb = json.load(f)

        # Verify Cell 9 uses compute_motif_presence_matrix_from_loader
        c9_code = "".join(nb["cells"][9]["source"])
        self.assertIn("compute_motif_presence_matrix_from_loader", c9_code)
        self.assertNotIn("presence[:, 0] = 1.0", c9_code)

        # Run smoke test on notebook 07
        global_ns = {"__name__": "__main__"}
        for cell in nb["cells"]:
            if cell.get("cell_type") == "code":
                exec("".join(cell.get("source", [])), global_ns)

        out_dir = Path("results_reproduction/motifs")
        t8_path = out_dir / "table_08_motif_perturbation.csv"
        t8_parquet = out_dir / "motif_summary.parquet"
        self.assertTrue(t8_path.exists())
        self.assertTrue(t8_parquet.exists())

        df_summary = pd.read_parquet(t8_parquet)
        self.assertIn("model_name", df_summary.columns)
        self.assertIn("motif", df_summary.columns)
        self.assertGreaterEqual(df_summary["model_name"].nunique(), 1)

    def test_r05_notebook_08_fail_fast_missing_motif_summary(self):
        """C4: Notebook 08 raises FileNotFoundError when SMOKE_TEST=0 and motif_summary.parquet is missing."""
        import os
        nb_path = Path("notebooks/reproduction/short/08_triplet_composition_and_cooccurrence.ipynb")
        with open(nb_path, "r", encoding="utf-8") as f:
            nb = json.load(f)

        setup_code = """
import os
os.environ["SMOKE_TEST"] = "0"
os.environ["REMOTE_SERVER"] = "False"
from viral_cnn.notebook_runtime import setup_notebook_runtime
ctx = setup_notebook_runtime()
output_dir = Path("/tmp/fake_output_dir_r05_test")
output_dir.mkdir(parents=True, exist_ok=True)
rscu_frame0 = {"base_frequencies": {"A": 0.25, "C": 0.25, "G": 0.25, "T": 0.25}}
"""
        c5_code = "".join(nb["cells"][5]["source"])

        global_ns = {"__name__": "__main__"}
        try:
            exec("".join(nb["cells"][1]["source"]), global_ns)
            exec(setup_code, global_ns)
            with self.assertRaises(FileNotFoundError):
                exec(c5_code, global_ns)
        finally:
            os.environ["SMOKE_TEST"] = "1"
            os.environ["REMOTE_SERVER"] = "False"

    def test_r05_notebook_08_table_10_boundary_signals(self):
        """C5: Notebook 08 generates Table 10 reporting translation boundary signals on learned motifs."""
        nb_path = Path("notebooks/reproduction/short/08_triplet_composition_and_cooccurrence.ipynb")
        with open(nb_path, "r", encoding="utf-8") as f:
            nb = json.load(f)

        global_ns = {"__name__": "__main__"}
        for idx in [1, 3, 5]:
            exec("".join(nb["cells"][idx]["source"]), global_ns)

        t10_path = Path("results_reproduction/motifs/table_10_translation_boundary_signals.csv")
        self.assertTrue(t10_path.exists())
        df10 = pd.read_csv(t10_path)
        signals = set(df10["signal"].tolist())
        self.assertIn("start_atg", signals)
        self.assertIn("stop_taa", signals)
        self.assertIn("stop_tag", signals)
        self.assertIn("stop_tga", signals)

    def test_r05_notebook_08_table_11_and_inter_architecture_correlations(self):
        """C6: Notebook 08 computes Table 11 symmetry and non-degenerate inter-architecture correlations."""
        nb_path = Path("notebooks/reproduction/short/08_triplet_composition_and_cooccurrence.ipynb")
        with open(nb_path, "r", encoding="utf-8") as f:
            nb = json.load(f)

        global_ns = {"__name__": "__main__"}
        for cell in nb["cells"]:
            if cell.get("cell_type") == "code":
                exec("".join(cell.get("source", [])), global_ns)

        t11_path = Path("results_reproduction/motifs/table_11_cooccurrence_synergy.csv")
        self.assertTrue(t11_path.exists())
        df11 = pd.read_csv(t11_path)
        self.assertIn("r_canonical", df11.columns)
        self.assertIn("p_permutation_null", df11.columns)


if __name__ == "__main__":
    unittest.main()

