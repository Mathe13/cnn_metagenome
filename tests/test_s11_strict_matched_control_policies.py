"""Unit and regression tests for S11 strict matched-control policies (C1-C10 / S14)."""
from __future__ import annotations

import sys
from pathlib import Path
import unittest

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from viral_cnn.interpretability.motifs import (
    MatchedControl,
    sample_matched_control_region,
    evaluate_motif_perturbation,
    summarize_matched_control_coverage,
    SyntheticAuditCNN,
)


class TestS11StrictMatchedControlPolicies(unittest.TestCase):
    """Regression test suite for matched control policies and coverage (C1-C10)."""

    def test_gc_exact_control_has_identical_gc_count(self):
        """C1: Returns MatchedControl with quality='exact' and distance=0 when exact GC match exists."""
        # Seq: 10 bases AT (0% GC) + CCCCCCC (100% GC at 10..17) + 10 bases AT + GGGGGGG (100% GC at 27..34)
        seq = torch.tensor([0, 3] * 5 + [1] * 7 + [0, 3] * 5 + [2] * 7, dtype=torch.long)
        motif_starts = [10]
        rng = np.random.default_rng(42)

        ctrl = sample_matched_control_region(
            sequence=seq,
            motif_starts=motif_starts,
            k=7,
            query_motif="CCCCCCC",
            control_policy="gc_matched",
            rng=rng,
        )
        self.assertIsInstance(ctrl, MatchedControl)
        self.assertEqual(ctrl.quality, "exact")
        self.assertEqual(ctrl.distance, 0)
        self.assertIsNotNone(ctrl.start)
        # GC count in control must be 7
        ctrl_bases = seq[ctrl.start : ctrl.start + 7].tolist()
        gc_count = sum(1 for b in ctrl_bases if b in (1, 2))
        self.assertEqual(gc_count, 7)

    def test_gc_relaxed_control_is_labeled_relaxed(self):
        """C2: Returns MatchedControl with quality='relaxed_1' and distance=1 when only +-1 GC exists."""
        # Query motif: CCCCCCC (GC=7).
        # Sequence has motif at 0..7. The only other non-overlapping region has GC=6 (e.g. CCCCCCA).
        seq = torch.tensor([1] * 7 + [1] * 6 + [0], dtype=torch.long)
        motif_starts = [0]
        rng = np.random.default_rng(42)

        ctrl = sample_matched_control_region(
            sequence=seq,
            motif_starts=motif_starts,
            k=7,
            query_motif="CCCCCCC",
            control_policy="gc_matched",
            rng=rng,
        )
        self.assertIsInstance(ctrl, MatchedControl)
        self.assertEqual(ctrl.quality, "relaxed_1")
        self.assertEqual(ctrl.distance, 1)
        self.assertEqual(ctrl.start, 7)

    def test_gc_unavailable_returns_unavailable(self):
        """C3: Returns MatchedControl with start=None, quality='unavailable', distance=-1 when no candidate satisfies GC <= 1."""
        # Query motif: CCCCCCC (GC=7).
        # Sequence has motif at 0..7. The rest of the sequence is all AT (GC=0, distance=7 > 1).
        seq = torch.tensor([1] * 7 + [0, 3] * 10, dtype=torch.long)
        motif_starts = [0]
        rng = np.random.default_rng(42)

        ctrl = sample_matched_control_region(
            sequence=seq,
            motif_starts=motif_starts,
            k=7,
            query_motif="CCCCCCC",
            control_policy="gc_matched",
            rng=rng,
        )
        self.assertIsInstance(ctrl, MatchedControl)
        self.assertIsNone(ctrl.start)
        self.assertEqual(ctrl.quality, "unavailable")
        self.assertEqual(ctrl.distance, -1)

    def test_complexity_exact_control_matches_run_length(self):
        """C4: Returns MatchedControl with quality='exact' and distance=0 when homopolymer run matches."""
        # Query motif: CCCCCCC has run length 7.
        # Sequence has motif at 0..7, alternating spacer at 7..14, and TTTTTTT (run length 7) at 15..21.
        seq = torch.tensor([1] * 7 + [0, 1, 0, 1, 0, 1, 0, 1] + [3] * 7, dtype=torch.long)
        motif_starts = [0]
        rng = np.random.default_rng(42)

        ctrl = sample_matched_control_region(
            sequence=seq,
            motif_starts=motif_starts,
            k=7,
            query_motif="CCCCCCC",
            control_policy="complexity_matched",
            rng=rng,
        )
        self.assertIsInstance(ctrl, MatchedControl)
        self.assertEqual(ctrl.quality, "exact")
        self.assertEqual(ctrl.distance, 0)
        self.assertEqual(ctrl.start, 15)

    def test_complexity_fallback_is_not_labeled_exact(self):
        """C5: Does not label fallback candidate as 'exact' (labels relaxed_1 or unmatched_fallback)."""
        # Query motif: CCCCCCC (run length 7).
        # Sequence has motif at 0..7. The only other region has run length 1 (ACGTACG).
        # Distance = |7 - 1| = 6 > 1 -> unmatched_fallback
        seq = torch.tensor([1] * 7 + [0, 1, 2, 3, 0, 1, 2], dtype=torch.long)
        motif_starts = [0]
        rng = np.random.default_rng(42)

        ctrl = sample_matched_control_region(
            sequence=seq,
            motif_starts=motif_starts,
            k=7,
            query_motif="CCCCCCC",
            control_policy="complexity_matched",
            rng=rng,
        )
        self.assertIsInstance(ctrl, MatchedControl)
        self.assertNotEqual(ctrl.quality, "exact")
        self.assertIn(ctrl.quality, ("relaxed_1", "unmatched_fallback"))
        self.assertGreater(ctrl.distance, 0)

    def test_control_never_overlaps_target_motif(self):
        """C6: Candidate window never overlaps any target motif in motif_starts."""
        seq = torch.tensor([0, 1, 2, 3] * 10, dtype=torch.long)
        motif_starts = [8, 20]
        k = 7
        rng = np.random.default_rng(42)

        for _ in range(20):
            ctrl = sample_matched_control_region(
                sequence=seq,
                motif_starts=motif_starts,
                k=k,
                query_motif="ACGTACG",
                control_policy="uniform_random",
                rng=rng,
            )
            self.assertIsNotNone(ctrl.start)
            for m in motif_starts:
                overlap = max(ctrl.start, m) < min(ctrl.start + k, m + k)
                self.assertFalse(overlap, f"Control at {ctrl.start} overlaps motif at {m}")

    def test_ambiguous_bases_are_not_selected_as_control(self):
        """C7: Rejects candidate windows containing invalid bases or padding (< 0 or >= 4)."""
        # Sequence where non-overlapping region has negative bases (padding) or 4 (unknown)
        seq = torch.tensor([1] * 7 + [0, -1, 4, 1, -1, 0, 2] + [0, 1, 2, 3] * 3, dtype=torch.long)
        motif_starts = [0]
        k = 7
        rng = np.random.default_rng(42)

        ctrl = sample_matched_control_region(
            sequence=seq,
            motif_starts=motif_starts,
            k=k,
            query_motif="CCCCCCC",
            control_policy="uniform_random",
            rng=rng,
        )
        self.assertIsNotNone(ctrl.start)
        # Window [ctrl.start : ctrl.start + k] must contain only 0..3
        window = seq[ctrl.start : ctrl.start + k]
        self.assertTrue(bool(((window >= 0) & (window < 4)).all().item()))

    def test_motif_control_artifact_records_match_quality(self):
        """C8: evaluate_motif_perturbation records control_match_quality and control_match_distance."""
        model = SyntheticAuditCNN(num_filters=4, kernel_size=7, num_classes=2)
        seq = torch.full((1, 512), 0, dtype=torch.long)
        seq[0, 10:17] = torch.tensor([1, 1, 1, 1, 1, 1, 1])  # CCCCCCC
        seq[0, 50:57] = torch.tensor([2, 2, 2, 2, 2, 2, 2])  # GGGGGGG (exact GC=7 match)

        dataset = TensorDataset(seq, torch.tensor([0], dtype=torch.long))
        loader = DataLoader(dataset, batch_size=1)

        df = evaluate_motif_perturbation(
            model=model,
            loader=loader,
            motifs=["CCCCCCC"],
            control_policy="gc_matched",
            n_repeats=2,
            seed=42,
        )
        self.assertFalse(df.empty)
        self.assertIn("control_match_quality", df.columns)
        self.assertIn("control_match_distance", df.columns)
        self.assertIn("control_policy_requested", df.columns)
        self.assertEqual(df["control_match_quality"].iloc[0], "exact")
        self.assertEqual(df["control_match_distance"].iloc[0], 0)

    def test_strict_matching_skips_non_exact_controls(self):
        """C9: When strict_matching=True, non-exact controls are skipped."""
        model = SyntheticAuditCNN(num_filters=4, kernel_size=7, num_classes=2)
        # Sequence where query CCCCCCC (GC=7) only has a relaxed_1 control (GC=6)
        seq = torch.full((1, 512), 0, dtype=torch.long)  # All AT (GC=0)
        seq[0, 10:17] = torch.tensor([1, 1, 1, 1, 1, 1, 1])  # CCCCCCC at 10..17
        seq[0, 50:57] = torch.tensor([1, 1, 1, 1, 1, 1, 0])  # CCCCCCA (GC=6, distance=1) at 50..57

        dataset = TensorDataset(seq, torch.tensor([0], dtype=torch.long))
        loader = DataLoader(dataset, batch_size=1)

        # Without strict_matching: accepts relaxed_1
        df_relaxed = evaluate_motif_perturbation(
            model=model,
            loader=loader,
            motifs=["CCCCCCC"],
            control_policy="gc_matched",
            strict_matching=False,
            n_repeats=2,
            seed=42,
        )
        self.assertEqual(len(df_relaxed), 1)
        self.assertEqual(df_relaxed["control_match_quality"].iloc[0], "relaxed_1")

        # With strict_matching=True: skips non-exact control
        df_strict = evaluate_motif_perturbation(
            model=model,
            loader=loader,
            motifs=["CCCCCCC"],
            control_policy="gc_matched",
            strict_matching=True,
            n_repeats=2,
            seed=42,
        )
        self.assertEqual(len(df_strict), 0)

    def test_summarize_matched_control_coverage_metrics(self):
        """C10: summarize_matched_control_coverage computes exact_match_rate and quality breakdown."""
        df = pd.DataFrame([
            {"model_name": "m1", "motif": "AAA", "control_match_quality": "exact"},
            {"model_name": "m1", "motif": "AAA", "control_match_quality": "exact"},
            {"model_name": "m1", "motif": "AAA", "control_match_quality": "relaxed_1"},
            {"model_name": "m1", "motif": "AAA", "control_match_quality": "unavailable"},
        ])
        cov = summarize_matched_control_coverage(df)
        self.assertFalse(cov.empty)
        row = cov.iloc[0]
        self.assertEqual(row["n_requested"], 4)
        self.assertEqual(row["n_exact"], 2)
        self.assertEqual(row["n_relaxed"], 1)
        self.assertEqual(row["n_unavailable"], 1)
        self.assertAlmostEqual(row["exact_match_rate"], 0.5)


if __name__ == "__main__":
    unittest.main()
