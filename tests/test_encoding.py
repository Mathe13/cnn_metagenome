"""
Contract tests for 4-channel spatial one-hot encoding and parity with historical code (C5, C6).
"""

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
SHORT_DIR = PROJECT_ROOT / "short"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(SHORT_DIR) not in sys.path:
    sys.path.insert(0, str(SHORT_DIR))

import torch
from viral_cnn.data.encoding import (
    NUCLEOTIDES,
    NUCLEOTIDE_MAPPING,
    encode_sequence_tokens,
    OneHotEmbedding,
    encode_spatial_one_hot,
)


class TestEncodingInvariants(unittest.TestCase):
    def test_spatial_one_hot_strict_channels_and_null_pad(self):
        """C5: Verifies strict A/C/G/T channels (0,1,2,3), 512 length, and null vectors for pad/ambiguity."""
        self.assertEqual(NUCLEOTIDES, ["A", "C", "G", "T"])
        self.assertEqual(NUCLEOTIDE_MAPPING, {"A": 0, "C": 1, "G": 2, "T": 3})

        # Test sequence with all 4 canonical bases + ambiguous N + padding _ + gap -
        test_seq = "ACGTN_-"
        tensor = encode_spatial_one_hot(test_seq, max_len=512)

        # Shape must be (1, 4, 512)
        self.assertEqual(tensor.shape, (1, 4, 512))

        # Position 0: 'A' -> [1, 0, 0, 0]
        self.assertTrue(torch.equal(tensor[0, :, 0], torch.tensor([1.0, 0.0, 0.0, 0.0])))
        # Position 1: 'C' -> [0, 1, 0, 0]
        self.assertTrue(torch.equal(tensor[0, :, 1], torch.tensor([0.0, 1.0, 0.0, 0.0])))
        # Position 2: 'G' -> [0, 0, 1, 0]
        self.assertTrue(torch.equal(tensor[0, :, 2], torch.tensor([0.0, 0.0, 1.0, 0.0])))
        # Position 3: 'T' -> [0, 0, 0, 1]
        self.assertTrue(torch.equal(tensor[0, :, 3], torch.tensor([0.0, 0.0, 0.0, 1.0])))

        # Position 4: 'N' -> null vector [0, 0, 0, 0]
        self.assertTrue(torch.equal(tensor[0, :, 4], torch.tensor([0.0, 0.0, 0.0, 0.0])))
        # Position 5: '_' -> null vector [0, 0, 0, 0]
        self.assertTrue(torch.equal(tensor[0, :, 5], torch.tensor([0.0, 0.0, 0.0, 0.0])))
        # Position 6: '-' -> null vector [0, 0, 0, 0]
        self.assertTrue(torch.equal(tensor[0, :, 6], torch.tensor([0.0, 0.0, 0.0, 0.0])))

        # Padded positions [7..511] must all be null vectors
        for pos in [7, 100, 255, 511]:
            self.assertTrue(
                torch.equal(tensor[0, :, pos], torch.tensor([0.0, 0.0, 0.0, 0.0])),
                f"Position {pos} is not null vector",
            )

    def test_parity_with_historical_one_hot_embedding(self):
        """C6: Verifies numerical identity with historical short/utils/models.py:OneHotEmbedding."""
        from utils.models import OneHotEmbedding as HistoricalOneHotEmbedding

        hist_emb = HistoricalOneHotEmbedding(num_features=4)
        new_emb = OneHotEmbedding(num_features=4)

        # Test on varied token tensor: values in [0, 1, 2, 3, 4, 99, -1]
        test_tokens = torch.tensor(
            [
                [0, 1, 2, 3, 4, 4, 4, 0],
                [3, 2, 1, 0, 99, -1, 4, 2],
            ],
            dtype=torch.long,
        )

        with torch.no_grad():
            hist_output = hist_emb(test_tokens)
            new_output = new_emb(test_tokens)

        self.assertEqual(hist_output.shape, new_output.shape)
        self.assertTrue(
            torch.equal(hist_output, new_output),
            "Extracted OneHotEmbedding output differs from historical implementation!",
        )


if __name__ == "__main__":
    unittest.main()

