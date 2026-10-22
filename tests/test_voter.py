"""
Unit and contract tests for S06: Long-Sequence Inference Audit and Voter Extraction.
"""

from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from viral_cnn.inference.voter import (
    LongSequenceVoter,
    LongConsensusResult,
    slice_sequence,
    aggregate_window_logits,
    DNA_MAP,
)
from viral_cnn.inference.audit_long import (
    verify_window_logit_equivalence,
    quantify_training_overlap,
    stratify_long_errors,
)
from viral_cnn.inference.engine import predict_sequence
from viral_cnn.models.builder import build_model
from viral_cnn.data.encoding import OneHotEmbedding


class TinyMockModel(nn.Module):
    """Deterministic tiny model for unit testing voter logic."""

    def __init__(self, num_classes: int = 13683):
        super().__init__()
        self.one_hot_layer = OneHotEmbedding(num_features=4)
        self.conv = nn.Conv1d(4, 8, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Linear(8, num_classes)
        # Deterministic weight init
        torch.manual_seed(42)
        nn.init.normal_(self.conv.weight, mean=0.0, std=0.1)
        nn.init.normal_(self.fc.weight, mean=0.0, std=0.1)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.one_hot_layer(x)
        x = self.conv(x)
        x = self.pool(x)
        x = x.squeeze(-1)
        return self.fc(x)


class TestVoterAndLongSequenceAudit(unittest.TestCase):

    # --- S1: Long-Sequence Slicing and Modular Voter Engine (C1 - C5) ---

    def test_slice_sequence_window_stride_and_token_normalization(self):
        """C1: slice_sequence extracts 512 bp windows with stride 256 bp and normalizes characters."""
        # 1024 bp sequence: should produce exactly 3 windows (0:512, 256:768, 512:1024)
        seq_1024 = "A" * 512 + "C" * 256 + "g" * 256  # includes lowercase 'g'
        windows = slice_sequence(seq_1024, window_size=512, stride=256, handle_short="pad")
        self.assertEqual(len(windows), 3, "1024 bp sequence must yield exactly 3 windows with stride 256")
        self.assertEqual(len(windows[0]), 512)
        self.assertEqual(len(windows[1]), 512)
        self.assertEqual(len(windows[2]), 512)

        # Check normalization: 'A' -> 0, 'C' -> 1, lowercase 'g' -> uppercase 'G' -> 2
        self.assertTrue(all(tok == 0 for tok in windows[0]))
        self.assertTrue(all(tok == 1 for tok in windows[2][:256]))
        self.assertTrue(all(tok == 2 for tok in windows[2][256:]), "Lowercase 'g' must normalize to 2")

    def test_slice_sequence_handle_short_skip(self):
        """C2: slice_sequence with handle_short='skip' returns 0 windows for sequences < 512 bp."""
        seq_short = "ACGT" * 50  # 200 bp < 512 bp
        windows = slice_sequence(seq_short, window_size=512, stride=256, handle_short="skip")
        self.assertEqual(len(windows), 0, "handle_short='skip' must return empty list for sequences < 512 bp")

    def test_slice_sequence_handle_short_pad(self):
        """C3: slice_sequence with handle_short='pad' pads contigs < 512 bp to 512 bp with token 4."""
        seq_short = "ACGT" * 50  # 200 bp
        windows = slice_sequence(seq_short, window_size=512, stride=256, handle_short="pad")
        self.assertEqual(len(windows), 1, "handle_short='pad' must return exactly 1 window for contigs < 512 bp")
        self.assertEqual(len(windows[0]), 512)
        # First 200 bases should be ACGT tokens [0, 1, 2, 3]
        expected_first = [0, 1, 2, 3] * 50
        self.assertEqual(windows[0][:200], expected_first)
        # Remaining 312 bases should be token 4 (null vector padding)
        self.assertTrue(all(tok == 4 for tok in windows[0][200:]), "Padding tokens must be 4")

    def test_aggregate_window_logits_mean_and_majority(self):
        """C4: aggregate_window_logits computes pre-softmax mean logits and hard window majority voting."""
        # Mock logits for 3 windows across 5 classes
        # Window 0: predicts class 1
        # Window 1: predicts class 2
        # Window 2: predicts class 2
        logits = torch.tensor([
            [1.0, 10.0, 2.0, 0.0, 0.0],
            [1.0, 2.0, 8.0, 0.0, 0.0],
            [1.0, 2.0, 8.0, 0.0, 0.0],
        ], dtype=torch.float32)

        res = aggregate_window_logits(logits)
        # Majority winners are [1, 2, 2] -> mode is 2
        self.assertEqual(res.pred_majority_vote, 2)
        # Mean logits:
        # Class 0: (1+1+1)/3 = 1.0
        # Class 1: (10+2+2)/3 = 14/3 = 4.67
        # Class 2: (2+8+8)/3 = 18/3 = 6.0 -> highest mean logit is class 2
        self.assertEqual(res.pred_mean_logits, 2)
        self.assertEqual(res.num_windows, 3)

    def test_aggregate_window_logits_deterministic_tie_break(self):
        """C5: aggregate_window_logits resolves ties deterministically by selecting smallest class index."""
        # 2 windows predicting class 3 and class 1 respectively (exact tie in count: 1 vs 1)
        logits = torch.tensor([
            [0.0, 0.0, 0.0, 10.0],  # winner: 3
            [0.0, 10.0, 0.0, 0.0],  # winner: 1
        ], dtype=torch.float32)

        res = aggregate_window_logits(logits, tie_break="min_index")
        # torch.mode selects smallest element among ties: min(3, 1) = 1
        self.assertEqual(res.pred_majority_vote, 1, "Tie break must select smallest class index")

    # --- S2: Logit Equivalence and Invariant Integrity Verification (C6 - C9) ---

    def test_voter_window_logit_equivalence_with_engine(self):
        """C6: A single 512 bp window produces identical logits between engine.py and voter.py."""
        model = TinyMockModel(num_classes=100)
        seq_512 = ("ACGT" * 128)  # Exactly 512 bp

        max_diff = verify_window_logit_equivalence(model, seq_512)
        self.assertLess(max_diff, 1e-5, f"Logits between engine and voter must be identical (<1e-5), got {max_diff}")

    def test_voter_label_offset_one_based_tax_id(self):
        """C7: Predicted class indices [0..N-1] convert to 1-based tax IDs via model_label + 1."""
        model = TinyMockModel(num_classes=13683)
        voter = LongSequenceVoter(model=model, window_size=512, stride=256, handle_short="pad")
        seq = "A" * 512
        res = voter.predict_sequence(seq)
        self.assertIsNotNone(res)

        # Check 0-based range
        self.assertGreaterEqual(res.pred_mean_logits, 0)
        self.assertLess(res.pred_mean_logits, 13683)

        # Check 1-based offset
        tax_id = res.pred_mean_logits + 1
        self.assertGreaterEqual(tax_id, 1)
        self.assertLessEqual(tax_id, 13683)

    def test_voter_iupac_ambiguity_produces_null_one_hot(self):
        """C8: IUPAC ambiguous characters map to token 4 and produce null one-hot vectors."""
        # Construct sequence containing canonical bases and IUPAC ambiguous bases
        seq_ambig = "ACGT" + "NRYWSKMBDHV" + "ACGT"
        windows = slice_sequence(seq_ambig, window_size=512, stride=256, handle_short="pad")
        tokens = torch.tensor(windows[0], dtype=torch.long)

        # In slice_sequence, IUPAC characters map to token 4
        # Positions 4 to 14 are the IUPAC characters
        for pos in range(4, 15):
            self.assertEqual(tokens[pos].item(), 4, f"IUPAC base at position {pos} must be token 4")

        # Passing through OneHotEmbedding
        embed = OneHotEmbedding(num_features=4)
        one_hot = embed(tokens.unsqueeze(0))  # shape: (1, 4, 512)
        # For positions 4 to 14, vector must be all zeros [0, 0, 0, 0]
        null_vectors = one_hot[0, :, 4:15]
        self.assertTrue(
            torch.all(null_vectors == 0.0),
            "IUPAC positions must produce strictly zero vector [0, 0, 0, 0]",
        )

    def test_voter_rejects_class_dimension_mismatch(self):
        """C9: LongSequenceVoter rejects model with incompatible head dimension with ValueError."""
        # Model with 2702 classes (Genus) passed to voter expecting 13683 classes (Species)
        genus_model = TinyMockModel(num_classes=2702)
        with self.assertRaises(ValueError):
            LongSequenceVoter(
                model=genus_model,
                expected_classes=13683,
            )

    # --- S3: Training Genome Overlap, Error Stratification, and Audit Report (C10 - C13) ---

    def test_audit_report_scientific_claims_and_evidence(self):
        """C10: docs/long_sequence_inference_audit.md documents causes of ConvNeXt drop and ResNet reference overlap."""
        report_path = PROJECT_ROOT / "docs" / "long_sequence_inference_audit.md"
        self.assertTrue(report_path.exists(), "docs/long_sequence_inference_audit.md must exist")
        text = report_path.read_text(encoding="utf-8")

        self.assertIn("72,69%", text, "Must cite ResNet50 72.69% accuracy")
        self.assertIn("calculate_n_reads", text, "Must document read simulation bias function")
        self.assertIn("500", text, "Must cite 500 reads for giant genomes")
        self.assertIn("50", text, "Must cite 50 reads for small genomes")
        self.assertIn("Bracoviriform", text, "Must identify dominant false-positive class")
        self.assertIn("Pandoravirus", text, "Must identify giant virus false-positive")
        self.assertIn("generalização", text.lower(), "Must address generalization limitation")

    def test_quantify_training_overlap_kmers_and_exact_match(self):
        """C11: quantify_training_overlap computes exact matches and canonical k-mer sharing."""
        train_seqs = [
            "ACGTACGTACGTACGT",  # length 16
            "TGCATGCATGCATGCA",
        ]
        test_seqs = [
            "ACGTACGTACGTACGT",  # exact match
            "ACGTACGTACGTAAAA",  # partial 12-mer match with k=8
            "GGGGGGGGGGGGGGGG",  # zero overlap
        ]

        metrics = quantify_training_overlap(test_seqs, train_seqs, k=8)
        self.assertIn("exact_match_fraction", metrics)
        self.assertIn("mean_kmer_overlap_fraction", metrics)
        self.assertIn("fraction_with_any_overlap", metrics)

        # Exactly 1 out of 3 matches train sequence
        self.assertAlmostEqual(metrics["exact_match_fraction"], 1.0 / 3.0, places=4)
        # 2 out of 3 share at least one 8-mer
        self.assertAlmostEqual(metrics["fraction_with_any_overlap"], 2.0 / 3.0, places=4)

    def test_stratify_long_errors_by_length_and_ambiguity(self):
        """C12: stratify_long_errors stratifies predictions across length and ambiguity bins."""
        df_dummy = pd.DataFrame({
            "length": [600, 1500, 3000, 7000],
            "ambiguity_ratio": [0.0, 0.02, 0.0, 0.10],
            "truth_label": [1, 2, 3, 4],
            "pred_mean_logits": [1, 2, 99, 4],      # 3 correct, 1 error (at 3000)
            "pred_majority_vote": [1, 99, 99, 4],   # 2 correct, 2 error
        })

        strata = stratify_long_errors(df_dummy)
        self.assertIn("overall", strata)
        self.assertIn("by_length", strata)
        self.assertIn("by_ambiguity", strata)

        # Check length bins
        self.assertIn("len_512_to_2000", strata["by_length"])
        self.assertIn("len_2000_to_5000", strata["by_length"])
        self.assertIn("len_5000_to_10000", strata["by_length"])
        self.assertEqual(strata["by_length"]["len_512_to_2000"]["count"], 2)
        self.assertEqual(strata["by_length"]["len_512_to_2000"]["accuracy_mean_logits"], 1.0)

        # Check ambiguity bins
        self.assertIn("ambig_0_pct", strata["by_ambiguity"])
        self.assertIn("ambig_0_to_5_pct", strata["by_ambiguity"])
        self.assertIn("ambig_gt_5_pct", strata["by_ambiguity"])
        self.assertEqual(strata["by_ambiguity"]["ambig_0_pct"]["count"], 2)

    def test_run_long_voter_cli_exit_zero_and_prediction_output(self):
        """C13: CLI scripts/run_long_voter.py executes long inference and exits with code 0."""
        with tempfile.TemporaryDirectory() as tmpdir:
            out_csv = Path(tmpdir) / "voter_out.csv"
            input_csv = PROJECT_ROOT / "tests" / "fixtures" / "sample_genomes.csv"

            cmd = [
                sys.executable,
                str(PROJECT_ROOT / "scripts" / "run_long_voter.py"),
                "--model-variant", "resnet_clean_species",
                "--input", str(input_csv),
                "--handle-short", "pad",
                "--output", str(out_csv),
                "--limit", "4",
            ]
            res = subprocess.run(cmd, capture_output=True, text=True)
            self.assertEqual(
                res.returncode,
                0,
                f"CLI execution failed with returncode {res.returncode}. Stderr: {res.stderr}",
            )
            self.assertTrue(out_csv.exists(), "Output prediction file must be generated")
            df_out = pd.read_csv(out_csv)
            self.assertIn("pred_mean_logits", df_out.columns)
            self.assertIn("pred_majority_vote", df_out.columns)
            self.assertIn("num_windows", df_out.columns)
            self.assertEqual(len(df_out), 4)


if __name__ == "__main__":
    unittest.main()
