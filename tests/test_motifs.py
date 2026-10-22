"""Unit and contract tests for motif extraction, compositional controls, and triplet analysis."""

import os
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from viral_cnn.interpretability.motifs import (
    extract_first_layer_motifs,
    scan_motif_occurrences,
    sample_matched_control_region,
    perturb_sequence_mask,
    evaluate_motif_perturbation,
    summarize_motif_control,
    audit_motif_composition_robustness,
    compute_jaccard_cooccurrence,
    analyze_cooccurrence_pairs,
    compute_exploratory_triplet_rscu,
    audit_translation_boundary_signals,
    generate_motif_audit_figures,
)


class Dummy1DCNN(nn.Module):
    """Synthetic 1D CNN with controllable first-layer weights for deterministic testing."""

    def __init__(self, num_filters=4, kernel_size=7, num_classes=5):
        super().__init__()
        self.conv1 = nn.Conv1d(4, num_filters, kernel_size=kernel_size, bias=False)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(num_filters, num_classes)

    def forward(self, x):
        # x: (B, 4, L) or (B, L)
        if x.dim() == 2:
            # integer indices (B, L) -> one-hot (B, 4, L)
            B, L = x.shape
            oh = torch.zeros((B, 4, L), device=x.device, dtype=torch.float32)
            valid = (x >= 0) & (x < 4)
            for b in range(B):
                for l in range(L):
                    idx = x[b, l].item()
                    if 0 <= idx < 4:
                        oh[b, idx, l] = 1.0
            x = oh
        feat = torch.relu(self.conv1(x))
        pooled = self.pool(feat).squeeze(-1)
        return self.fc(pooled)


class TestMotifAnalysis(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        torch.manual_seed(42)
        np.random.seed(42)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_extract_first_layer_motifs_max_positive(self):
        """C1: extract_first_layer_motifs extracts consensus using max_positive rule."""
        model = Dummy1DCNN(num_filters=2, kernel_size=7)
        # Filter 0 weights: set highest positive weight to C (idx 1) for all 7 positions
        with torch.no_grad():
            model.conv1.weight.zero_()
            # Channel order: A=0, C=1, G=2, T=3
            # Filter 0: C has +2.0, others -1.0
            model.conv1.weight[0, 1, :] = 2.0
            model.conv1.weight[0, 0, :] = -1.0
            # Filter 1: A has +1.5 for first 3, G has +2.5 for remaining 4
            model.conv1.weight[1, 0, :3] = 1.5
            model.conv1.weight[1, 2, 3:] = 2.5

        motifs = extract_first_layer_motifs(model, consensus_rule="max_positive")
        self.assertIn("CCCCCCC", motifs)
        self.assertIn("AAAGGGG", motifs)
        self.assertEqual(motifs["CCCCCCC"], 1)
        self.assertEqual(motifs["AAAGGGG"], 1)

    def test_extract_first_layer_motifs_max_abs(self):
        """C2: extract_first_layer_motifs extracts consensus using max_abs rule."""
        model = Dummy1DCNN(num_filters=1, kernel_size=7)
        # Position 0: positive A (+1.0) vs negative T (-5.0) -> max_abs should pick T (idx 3)
        with torch.no_grad():
            model.conv1.weight.zero_()
            model.conv1.weight[0, 0, 0] = 1.0
            model.conv1.weight[0, 3, 0] = -5.0  # larger absolute magnitude
            # positions 1-6: G has +3.0
            model.conv1.weight[0, 2, 1:] = 3.0

        motifs_pos = extract_first_layer_motifs(model, consensus_rule="max_positive")
        motifs_abs = extract_first_layer_motifs(model, consensus_rule="max_abs")

        # max_positive picks A for pos 0 -> AGGGGGG
        self.assertIn("AGGGGGG", motifs_pos)
        # max_abs picks T for pos 0 -> TGGGGGG
        self.assertIn("TGGGGGG", motifs_abs)

    def test_scan_motif_occurrences_exact_coordinates(self):
        """C3: scan_motif_occurrences locates exact 0-based match coordinates via 1D conv."""
        # Sequence: "ACGT" + "CCCCCCC" + "ACGT" + "CCCCCCC"
        # Length: 4 + 7 + 4 + 7 = 22
        seq_str = "ACGTCCCCCCCACGTCCCCCCC"
        base_map = {"A": 0, "C": 1, "G": 2, "T": 3}
        seq_ints = torch.tensor([[base_map[b] for b in seq_str]], dtype=torch.long)

        matches = scan_motif_occurrences(seq_ints, ["CCCCCCC"])
        self.assertIn("CCCCCCC", matches)
        # Expect matches at index 4 and index 15
        self.assertEqual(matches["CCCCCCC"][0], [4, 15])

    def test_scan_motif_occurrences_padding_exclusion(self):
        """C4: scan_motif_occurrences excludes positions with padding/ambiguous bases."""
        # Sequence with padding (-1) right in the middle of CCCCCCC
        # "CCCC" + -1 + "CC" -> not a valid occurrence
        seq_ints = torch.tensor([[1, 1, 1, 1, -1, 1, 1, 0, 1, 2, 3]], dtype=torch.long)
        matches = scan_motif_occurrences(seq_ints, ["CCCCCCC"])
        self.assertEqual(len(matches["CCCCCCC"][0]), 0)

    def test_perturb_sequence_mask_different_nucleotides(self):
        """C5: perturb_sequence_mask substitutes target positions with different bases."""
        seq = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3], dtype=torch.long)  # A C G T A C G T
        mask = torch.tensor([True, True, True, True, False, False, False, False], dtype=torch.bool)
        generator = torch.Generator().manual_seed(42)

        perturbed = perturb_sequence_mask(seq.unsqueeze(0), mask.unsqueeze(0), generator=generator).squeeze(0)

        # Masked positions must change and remain valid [0, 3]
        for i in range(4):
            self.assertNotEqual(seq[i].item(), perturbed[i].item())
            self.assertTrue(0 <= perturbed[i].item() < 4)
        # Unmasked positions must remain identical
        for i in range(4, 8):
            self.assertEqual(seq[i].item(), perturbed[i].item())

    def test_sample_matched_control_uniform_random_non_overlapping(self):
        """C6: sample_matched_control_region with uniform_random samples valid non-overlapping window."""
        # Sequence of length 64, motif at 10..16
        seq = torch.randint(0, 4, (64,), dtype=torch.long)
        motif_starts = [10]
        rng = np.random.default_rng(42)

        start = sample_matched_control_region(
            seq,
            motif_starts=motif_starts,
            k=7,
            query_motif="CCCCCCC",
            control_policy="uniform_random",
            rng=rng,
        )
        self.assertIsNotNone(start)
        self.assertTrue(0 <= start <= 64 - 7)
        # Must not overlap [10, 16]
        self.assertTrue(start + 7 <= 10 or start >= 17)

    def test_evaluate_motif_perturbation_paired_repeats(self):
        """C7: evaluate_motif_perturbation computes paired probability drops across repeats."""
        model = Dummy1DCNN(num_filters=2, kernel_size=7, num_classes=3)
        # Create synthetic batch with known motif occurrence
        # Sequence: 10 bases random + CCCCCCC + 20 bases random
        motif = "CCCCCCC"
        base_map = {"A": 0, "C": 1, "G": 2, "T": 3}
        seq = [0, 1, 2, 3, 0, 1, 2, 3, 0, 1] + [1] * 7 + [2] * 20
        inputs = torch.tensor([seq], dtype=torch.long)
        labels = torch.tensor([1], dtype=torch.long)
        dataset = torch.utils.data.TensorDataset(inputs, labels)
        loader = torch.utils.data.DataLoader(dataset, batch_size=1)

        records = evaluate_motif_perturbation(
            model=model,
            loader=loader,
            motifs=[motif],
            model_name="test_cnn",
            control_policy="uniform_random",
            n_repeats=5,
            seed=42,
        )
        self.assertFalse(records.empty)
        self.assertIn("motif_probability_drop", records.columns)
        self.assertIn("random_probability_drop", records.columns)
        self.assertIn("specific_probability_drop", records.columns)
        self.assertEqual(records["n_repeats"].iloc[0], 5)

    def test_summarize_motif_control_wilcoxon_and_fdr(self):
        """C8: summarize_motif_control performs paired Wilcoxon test and Benjamini-Hochberg FDR."""
        # Synthetic DataFrame with 10 sequences for motif CCCCCCC having positive specific drop
        records = []
        for i in range(10):
            records.append({
                "model_name": "test_cnn",
                "sequence_id": i,
                "motif": "CCCCCCC",
                "n_occurrences": 1,
                "baseline_probability": 0.8,
                "motif_probability_drop": 0.35 + 0.01 * i,
                "random_probability_drop": 0.05 + 0.01 * i,
                "specific_probability_drop": 0.30,
                "n_repeats": 5,
            })
        df = pd.DataFrame(records)
        summary = summarize_motif_control(df, alpha=0.05)

        self.assertEqual(len(summary), 1)
        self.assertTrue(summary["significant_fdr"].iloc[0])
        self.assertLess(summary["p_value"].iloc[0], 0.05)
        self.assertLess(summary["p_fdr"].iloc[0], 0.05)
        self.assertGreater(summary["mean_specific_drop"].iloc[0], 0.0)

    def test_sample_matched_control_gc_matched(self):
        """C9: sample_matched_control_region with gc_matched samples window with identical GC count."""
        # Query motif has 7 bases: CCCCCCC -> GC count = 7
        # Sequence contains a 70% GC region and a 100% GC region
        base_map = {"A": 0, "C": 1, "G": 2, "T": 3}
        # 10 bases AT + 7 bases GGGGGGG (GC=7) + 10 bases AT + CCCCCCC (motif at pos 27)
        seq_str = "ATATATATAT" + "GGGGGGG" + "ATATATATAT" + "CCCCCCC"
        seq = torch.tensor([base_map[b] for b in seq_str], dtype=torch.long)
        motif_starts = [27]
        rng = np.random.default_rng(42)

        start = sample_matched_control_region(
            seq,
            motif_starts=motif_starts,
            k=7,
            query_motif="CCCCCCC",
            control_policy="gc_matched",
            rng=rng,
        )
        self.assertIsNotNone(start)
        # Control window must be GGGGGGG (pos 10), which has GC=7
        self.assertEqual(start, 10)

    def test_sample_matched_control_complexity_matched(self):
        """C10: sample_matched_control_region with complexity_matched matches homopolymer run length."""
        base_map = {"A": 0, "C": 1, "G": 2, "T": 3}
        # Query motif: CCCCCCC has homopolymer length 7
        # Seq has: 10 bases mixed + 7 bases TTTTTTT (homopolymer length 7) + 10 mixed + CCCCCCC at 27
        seq_str = "ACGTACGTAC" + "TTTTTTT" + "ACGTACGTAC" + "CCCCCCC"
        seq = torch.tensor([base_map[b] for b in seq_str], dtype=torch.long)
        motif_starts = [27]
        rng = np.random.default_rng(42)

        start = sample_matched_control_region(
            seq,
            motif_starts=motif_starts,
            k=7,
            query_motif="CCCCCCC",
            control_policy="complexity_matched",
            rng=rng,
        )
        self.assertIsNotNone(start)
        self.assertEqual(start, 10)

    def test_evaluate_motif_perturbation_gc_matched(self):
        """C11: evaluate_motif_perturbation under gc_matched computes GC-controlled drop."""
        model = Dummy1DCNN(num_filters=2, kernel_size=7, num_classes=3)
        # Sequence containing both CCCCCCC and GGGGGGG
        seq = [0, 1, 2, 3] * 5 + [1] * 7 + [0, 1, 2, 3] * 5 + [2] * 7
        inputs = torch.tensor([seq], dtype=torch.long)
        labels = torch.tensor([0], dtype=torch.long)
        loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(inputs, labels), batch_size=1)

        records = evaluate_motif_perturbation(
            model=model,
            loader=loader,
            motifs=["CCCCCCC"],
            model_name="test_cnn",
            control_policy="gc_matched",
            n_repeats=5,
            seed=42,
        )
        self.assertFalse(records.empty)
        self.assertIn("specific_probability_drop", records.columns)

    def test_audit_motif_composition_flags_confounded_homopolymers(self):
        """C12: audit_motif_composition_robustness flags homopolymers confounded by GC composition."""
        # Simulated summary where CCCCCCC is significant under uniform random but not under GC-matched
        df_uniform = pd.DataFrame([{
            "model_name": "ConvNext_finetuned",
            "motif": "CCCCCCC",
            "mean_specific_drop": 0.08,
            "significant_fdr": True,
            "p_fdr": 0.001,
            "n_sequences": 100,
        }])
        df_gc = pd.DataFrame([{
            "model_name": "ConvNext_finetuned",
            "motif": "CCCCCCC",
            "mean_specific_drop": -0.005,
            "significant_fdr": False,
            "p_fdr": 0.75,
            "n_sequences": 100,
        }])

        audit = audit_motif_composition_robustness(df_uniform, df_gc)
        self.assertEqual(len(audit), 1)
        self.assertTrue(audit["is_confounded"].iloc[0])
        self.assertEqual(audit["verdict"].iloc[0], "compositionally_confounded")

    def test_compute_jaccard_cooccurrence_symmetric_matrix(self):
        """C13: compute_jaccard_cooccurrence builds exact symmetric pairwise Jaccard matrix."""
        # 3 sequences, 2 motifs:
        # seq 0 has motif A and B
        # seq 1 has motif A
        # seq 2 has motif B
        presence = torch.tensor([[1.0, 1.0], [1.0, 0.0], [0.0, 1.0]])
        motifs = ["MOTIF_A", "MOTIF_B"]
        # Intersection = 1, Union = 3 -> Jaccard = 1/3 = 0.3333
        jaccard = compute_jaccard_cooccurrence(presence, motifs)

        self.assertAlmostEqual(jaccard.loc["MOTIF_A", "MOTIF_A"], 1.0)
        self.assertAlmostEqual(jaccard.loc["MOTIF_B", "MOTIF_B"], 1.0)
        self.assertAlmostEqual(jaccard.loc["MOTIF_A", "MOTIF_B"], 1.0 / 3.0, places=4)
        self.assertAlmostEqual(jaccard.loc["MOTIF_B", "MOTIF_A"], 1.0 / 3.0, places=4)
        # Symmetry check
        np.testing.assert_allclose(jaccard.values, jaccard.values.T)

    def test_analyze_cooccurrence_pairs_orientation_dependency(self):
        """C14: analyze_cooccurrence_pairs computes orientation dependency between (u, v) and (v, u)."""
        motifs = [f"M_{i}" for i in range(10)]
        np.random.seed(42)
        # Asymmetric values for RSCU to observe orientation effects
        rscu_map = {m: 0.5 + 0.1 * i for i, m in enumerate(motifs)}
        mat = np.random.uniform(0.1, 0.9, size=(10, 10))
        mat = (mat + mat.T) / 2.0
        np.fill_diagonal(mat, 1.0)
        jaccard_df = pd.DataFrame(mat, index=motifs, columns=motifs)

        res = analyze_cooccurrence_pairs(jaccard_df, rscu_map, top_k=15, n_permutations=100)
        self.assertIn("r_canonical", res)
        self.assertIn("r_reversed", res)
        self.assertIn("delta_r_orientation", res)
        self.assertGreaterEqual(res["delta_r_orientation"], 0.0)

    def test_analyze_cooccurrence_pairs_symmetric_intraclass_correlation(self):
        """C15: analyze_cooccurrence_pairs computes symmetric intraclass and bidirectional correlation."""
        motifs = [f"M_{i}" for i in range(8)]
        rscu_map = {m: 1.0 + 0.2 * i for i, m in enumerate(motifs)}
        mat = np.eye(8)
        mat[0, 1] = mat[1, 0] = 0.8
        mat[2, 3] = mat[3, 2] = 0.7
        jaccard_df = pd.DataFrame(mat, index=motifs, columns=motifs)

        res = analyze_cooccurrence_pairs(jaccard_df, rscu_map, top_k=2, n_permutations=100)
        self.assertIn("icc_symmetric", res)
        self.assertIn("r_bidirectional", res)
        self.assertTrue(-1.0 <= res["icc_symmetric"] <= 1.0)
        self.assertTrue(-1.0 <= res["r_bidirectional"] <= 1.0)

    def test_analyze_cooccurrence_pairs_node_permutation_null(self):
        """C16: analyze_cooccurrence_pairs evaluates significance via node-permutation null model."""
        motifs = [f"M_{i}" for i in range(12)]
        rscu_map = {m: float(i % 3) for i, m in enumerate(motifs)}
        mat = np.random.uniform(0, 1, size=(12, 12))
        mat = (mat + mat.T) / 2.0
        np.fill_diagonal(mat, 1.0)
        jaccard_df = pd.DataFrame(mat, index=motifs, columns=motifs)

        res = analyze_cooccurrence_pairs(jaccard_df, rscu_map, top_k=20, n_permutations=200, seed=42)
        self.assertIn("p_perm", res)
        self.assertTrue(0.0 <= res["p_perm"] <= 1.0)

    def test_compute_exploratory_triplet_rscu_multiframe(self):
        """C17: compute_exploratory_triplet_rscu calculates triplet RSCU across frame modes."""
        # Known sequence: 12 bases "ATG GCC GCA GCG" (ATG=Met, GCC/GCA/GCG=Ala)
        seqs = ["ATGGCCGCAGCG"]
        rscu_f0 = compute_exploratory_triplet_rscu(seqs, frame_mode="frame0")
        rscu_3f = compute_exploratory_triplet_rscu(seqs, frame_mode="forward_3frames")
        rscu_6f = compute_exploratory_triplet_rscu(seqs, frame_mode="six_frames")

        self.assertIn("ATG", rscu_f0["rscu"])
        self.assertIn("GCC", rscu_f0["rscu"])
        # For single codon family (Met), RSCU is always 1.0
        self.assertAlmostEqual(rscu_f0["rscu"]["ATG"], 1.0)
        # All frame modes must return complete dictionary of 64 codons
        self.assertEqual(len(rscu_f0["rscu"]), 64)
        self.assertEqual(len(rscu_3f["rscu"]), 64)
        self.assertEqual(len(rscu_6f["rscu"]), 64)

    def test_compute_exploratory_triplet_multinomial_null(self):
        """C18: compute_exploratory_triplet_rscu computes 0th-order multinomial base composition null."""
        # 100 bases purely A and C (50% A, 50% C)
        seqs = ["AC" * 50]
        res = compute_exploratory_triplet_rscu(seqs, frame_mode="frame0")

        null_probs = res["multinomial_null_probabilities"]
        # P(AAA) = 0.5 * 0.5 * 0.5 = 0.125
        self.assertAlmostEqual(null_probs["AAA"], 0.125, places=3)
        # P(TTT) = 0.0
        self.assertAlmostEqual(null_probs["TTT"], 0.0, places=3)

    def test_audit_translation_boundary_signals_against_null(self):
        """C19: audit_translation_boundary_signals tests ATG/stop frequencies against base composition."""
        # Motifs with artificial enrichment of ATG and TAA
        motifs = ["ATGAAAA", "TAAGGGG", "CCCCCCC", "GGGGGGG"]
        base_freqs = {"A": 0.25, "C": 0.25, "G": 0.25, "T": 0.25}

        report = audit_translation_boundary_signals(motifs, base_freqs)
        self.assertIn("start_atg", report)
        self.assertIn("stop_taa", report)
        self.assertIn("stop_tag", report)
        self.assertIn("stop_tga", report)
        self.assertIn("observed_count", report["start_atg"])
        self.assertIn("expected_count", report["start_atg"])

    def test_compute_exploratory_triplet_demarcates_unverified_frames(self):
        """C20: compute_exploratory_triplet_rscu designates output as exploratory and warns."""
        seqs = ["ACGT" * 10]
        with self.assertWarns(UserWarning):
            res = compute_exploratory_triplet_rscu(seqs, frame_mode="frame0")

        self.assertEqual(res["metric_label"], "ExploratoryTripletComposition")
        self.assertFalse(res["verified_biological_frame"])

    def test_run_motif_analysis_cli_exit_zero_and_artifacts(self):
        """C21: CLI scripts/run_motif_analysis.py exits 0 and produces artifacts."""
        out_dir = os.path.join(self.temp_dir, "cli_test_out")
        cmd = (
            f".venv/bin/python3 scripts/run_motif_analysis.py "
            f"--synthetic-test "
            f"--output-dir {out_dir} "
            f"--n-repeats 5 "
            f"--seed 42"
        )
        exit_code = os.system(cmd)
        self.assertEqual(exit_code, 0)
        self.assertTrue(os.path.exists(os.path.join(out_dir, "motif_summary.parquet")))
        self.assertTrue(os.path.exists(os.path.join(out_dir, "cooccurrence_synergy.json")))
        self.assertTrue(os.path.exists(os.path.join(out_dir, "run_manifest.json")))

    def test_run_motif_analysis_cli_invalid_args_exit_2(self):
        """C22: CLI scripts/run_motif_analysis.py exits 2 on invalid arguments."""
        cmd = ".venv/bin/python3 scripts/run_motif_analysis.py --invalid-flag"
        exit_code = os.system(cmd)
        # Exit code 2 in python argparse is 512 in os.system (exit_code >> 8 == 2)
        self.assertEqual(exit_code >> 8, 2)

    def test_generate_motif_audit_figures_checkpoint_attribution(self):
        """C23: generate_motif_audit_figures renders multi-panel figures attributed to checkpoints."""
        out_path = os.path.join(self.temp_dir, "audit_panel.png")
        checkpoint_attributions = {
            "ConvNext_finetuned": "checkpoints/best_convnext_species_finetuned.pt",
            "ResNet50": "checkpoints/best_resnet_classifier_fast_phase4.pt",
        }
        df_summary = pd.DataFrame([{
            "model_name": "ConvNext_finetuned",
            "motif": "CCCCCCC",
            "mean_specific_drop_uniform": 0.08,
            "mean_specific_drop_gc": -0.002,
            "significant_fdr": False,
        }])

        generate_motif_audit_figures(df_summary, checkpoint_attributions, out_path)
        self.assertTrue(os.path.exists(out_path))
        self.assertGreater(os.path.getsize(out_path), 1000)

    def test_motif_audit_doc_documents_controls_and_recommendations(self):
        """C24: docs/motifs_and_triplet_composition_audit.md exists and documents controls."""
        doc_path = "docs/motifs_and_triplet_composition_audit.md"
        self.assertTrue(os.path.exists(doc_path), f"Missing {doc_path}")
        with open(doc_path, "r", encoding="utf-8") as f:
            content = f.read()

        self.assertIn("GCMatchedControl", content)
        self.assertIn("HomopolymerComplexityControl", content)
        self.assertIn("SymmetricCooccurrenceCorrelation", content)
        self.assertIn("NodePermutationNullModel", content)
        self.assertIn("ExploratoryTripletComposition", content)

    def test_motif_pipeline_synthetic_deterministic_execution(self):
        """C25: Motif analysis pipeline executes deterministically on synthetic fixtures."""
        model = Dummy1DCNN(num_filters=2, kernel_size=7, num_classes=2)
        motifs = extract_first_layer_motifs(model, consensus_rule="max_positive")
        self.assertEqual(len(motifs), 2)
        # Check reproducibility with fixed seed
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)
        seq = torch.randint(0, 4, (64,), dtype=torch.long)
        s1 = sample_matched_control_region(seq, [10], 7, "CCCCCCC", "uniform_random", rng1)
        s2 = sample_matched_control_region(seq, [10], 7, "CCCCCCC", "uniform_random", rng2)
        self.assertEqual(s1, s2)


if __name__ == "__main__":
    unittest.main()
