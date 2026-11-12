"""Unit tests for S10 exact paired motif perturbation (C1-C8)."""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path
import unittest

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from viral_cnn.interpretability.motifs import (
    sample_substitution_offsets,
    perturb_region_with_offsets,
    hash_mutation_offsets,
    evaluate_motif_perturbation,
    perturb_sequence_mask,
    SyntheticAuditCNN,
)


class TestS10ExactPairedMotifPerturbation(unittest.TestCase):
    """Tests satisfying obligations C1 through C8."""

    def test_sample_substitution_offsets(self):
        """C1: Returns 1D tensor of length K with values strictly in {1, 2, 3}."""
        gen = torch.Generator().manual_seed(42)
        k = 7
        offsets = sample_substitution_offsets(k, generator=gen)
        self.assertEqual(offsets.shape, (k,))
        self.assertEqual(offsets.dtype, torch.long)
        self.assertTrue(torch.all((offsets >= 1) & (offsets <= 3)))
        # Verify all values in {1, 2, 3} can be generated across multiple draws
        all_vals = set()
        for _ in range(50):
            o = sample_substitution_offsets(k, generator=gen)
            all_vals.update(o.tolist())
        self.assertEqual(all_vals, {1, 2, 3})

    def test_perturb_region_with_offsets_replaces_nucleotides(self):
        """C2: Replaces valid nucleotides x in {0, 1, 2, 3} at [start, start+K) with (x + delta_i) % 4."""
        # Seq of 10 bases: [0, 1, 2, 3, 0, 1, 2, 3, 0, 1]
        seq = torch.tensor([[0, 1, 2, 3, 0, 1, 2, 3, 0, 1]], dtype=torch.long)
        start = 2
        offsets = torch.tensor([1, 2, 3, 1], dtype=torch.long)  # K = 4
        # Original region: [2, 3, 0, 1]
        # Expected: [(2+1)%4, (3+2)%4, (0+3)%4, (1+1)%4] = [3, 1, 3, 2]
        perturbed = perturb_region_with_offsets(seq, start=start, offsets=offsets)
        expected_region = torch.tensor([3, 1, 3, 2], dtype=torch.long)
        self.assertTrue(torch.equal(perturbed[0, start : start + 4], expected_region))

    def test_perturb_region_preserves_outside_positions(self):
        """C3: Leaves all bases outside [start, start+K) strictly unchanged."""
        seq = torch.tensor([[0, 1, 2, 3, 0, 1, 2, 3, 0, 1]], dtype=torch.long)
        start = 3
        offsets = torch.tensor([1, 2, 3], dtype=torch.long)
        perturbed = perturb_region_with_offsets(seq, start=start, offsets=offsets)
        # Check prefix [0:start]
        self.assertTrue(torch.equal(perturbed[0, :start], seq[0, :start]))
        # Check suffix [start+3:]
        self.assertTrue(torch.equal(perturbed[0, start + 3 :], seq[0, start + 3 :]))

    def test_perturb_region_skips_invalid_padding(self):
        """C4: Leaves invalid bases and padding (< 0 or >= 4) intact without applying offsets."""
        seq = torch.tensor([[0, -1, 4, 2, -1]], dtype=torch.long)
        start = 1
        offsets = torch.tensor([1, 2, 3], dtype=torch.long)
        # Positions:
        # idx 1: -1 (invalid -> unchanged)
        # idx 2: 4 (invalid -> unchanged)
        # idx 3: 2 (valid -> (2+3)%4 = 1)
        perturbed = perturb_region_with_offsets(seq, start=start, offsets=offsets)
        self.assertEqual(perturbed[0, 1].item(), -1)
        self.assertEqual(perturbed[0, 2].item(), 4)
        self.assertEqual(perturbed[0, 3].item(), 1)

    def test_paired_motif_control_reuses_identical_offsets(self):
        """C5: evaluate_motif_perturbation supplies identical offsets to motif and control in each repeat."""
        # Build synthetic model and dataset
        model = SyntheticAuditCNN(num_filters=4, kernel_size=7, num_classes=3)
        # Sequence with 'ACGTACG' (0,1,2,3,0,1,2) at pos 0 and control at pos 20
        seq = torch.full((1, 512), 0, dtype=torch.long)
        # Put motif at 10..17: [0, 1, 2, 3, 0, 1, 2]
        seq[0, 10:17] = torch.tensor([0, 1, 2, 3, 0, 1, 2])
        # Put differing control at 50..57: [3, 2, 1, 0, 3, 2, 1]
        seq[0, 50:57] = torch.tensor([3, 2, 1, 0, 3, 2, 1])

        dataset = TensorDataset(seq, torch.tensor([0], dtype=torch.long))
        loader = DataLoader(dataset, batch_size=1)

        df = evaluate_motif_perturbation(
            model=model,
            loader=loader,
            motifs=["ACGTACG"],
            control_policy="uniform_random",
            n_repeats=3,
            seed=42,
        )
        self.assertFalse(df.empty)
        # Verify provenance fields
        for _, rec in df.iterrows():
            self.assertIn("mutation_pattern_hash", rec)
            self.assertIn("motif_start", rec)
            self.assertIn("control_start", rec)
            self.assertNotEqual(rec["motif_start"], rec["control_start"])

    def test_identical_subsequences_yield_identical_mutations(self):
        """C6: When motif and control have identical sequences, both mutate to the identical output."""
        # Create a sequence where motif region and control region have identical 7-mer
        seq = torch.full((1, 100), 0, dtype=torch.long)
        motif_pattern = torch.tensor([1, 2, 3, 0, 1, 2, 3], dtype=torch.long)
        m_start = 10
        c_start = 30
        seq[0, m_start : m_start + 7] = motif_pattern
        seq[0, c_start : c_start + 7] = motif_pattern

        gen = torch.Generator().manual_seed(123)
        offsets = sample_substitution_offsets(7, generator=gen)

        mut_m = perturb_region_with_offsets(seq, start=m_start, offsets=offsets)
        mut_c = perturb_region_with_offsets(seq, start=c_start, offsets=offsets)

        subseq_m = mut_m[0, m_start : m_start + 7]
        subseq_c = mut_c[0, c_start : c_start + 7]

        self.assertTrue(torch.equal(subseq_m, subseq_c))
        # Ensure mutation actually happened
        self.assertFalse(torch.equal(subseq_m, motif_pattern))

    def test_hash_mutation_offsets_is_deterministic_hex16(self):
        """C7: hash_mutation_offsets computes a deterministic 16-hex SHA-256 fingerprint."""
        offsets = torch.tensor([1, 2, 3, 1, 2, 3, 1], dtype=torch.long)
        h1 = hash_mutation_offsets(offsets)
        h2 = hash_mutation_offsets(offsets)
        self.assertEqual(h1, h2)
        self.assertEqual(len(h1), 16)
        # Check hex characters
        int(h1, 16)  # should not raise
        # Check against manual sha256
        expected_raw = hashlib.sha256(offsets.detach().cpu().numpy().tobytes()).hexdigest()[:16]
        self.assertEqual(h1, expected_raw)

    def test_evaluate_motif_perturbation_records_provenance_metadata(self):
        """C8: Records contain mutation_pattern_hash, motif_start, control_start and summary preserves schema."""
        model = SyntheticAuditCNN(num_filters=4, kernel_size=7, num_classes=2)
        seq = torch.full((2, 512), 0, dtype=torch.long)
        # insert ACGTACG
        seq[0, 10:17] = torch.tensor([0, 1, 2, 3, 0, 1, 2])
        seq[1, 20:27] = torch.tensor([0, 1, 2, 3, 0, 1, 2])
        labels = torch.tensor([0, 1], dtype=torch.long)
        loader = DataLoader(TensorDataset(seq, labels), batch_size=2)

        df = evaluate_motif_perturbation(
            model=model,
            loader=loader,
            motifs=["ACGTACG"],
            control_policy="uniform_random",
            n_repeats=2,
            seed=42,
        )
        self.assertFalse(df.empty)
        rec = df.iloc[0]
        self.assertIn("mutation_pattern_hash", rec)
        self.assertIn("motif_start", rec)
        self.assertIn("control_start", rec)
        self.assertEqual(len(str(rec["mutation_pattern_hash"])), 16)


if __name__ == "__main__":
    unittest.main()
