"""
Unit tests for model forward pass on ResNet50-1D and ConvNeXt-1D.
"""

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import torch
from viral_cnn.models.builder import build_model
from viral_cnn.data.encoding import encode_sequence_tokens


class TestModelForward(unittest.TestCase):
    def test_resnet50_forward_pass(self):
        torch.manual_seed(42)
        model = build_model("resnet50_1d", num_classes=10, max_seq_len=512, dropout_rate=0.0)
        model.eval()

        tokens = encode_sequence_tokens("ACGT" * 32, max_len=512)
        x = torch.tensor([tokens], dtype=torch.long)

        with torch.no_grad():
            logits = model(x)

        self.assertEqual(logits.shape, (1, 10))
        self.assertFalse(torch.isnan(logits).any())
        self.assertFalse(torch.isinf(logits).any())

    def test_convnext_tiny_forward_pass(self):
        torch.manual_seed(42)
        model = build_model("convnext_tiny_1d", num_classes=10, max_seq_len=512, dropout_rate=0.0)
        model.eval()

        tokens = encode_sequence_tokens("TGCAT" * 25, max_len=512)
        x = torch.tensor([tokens], dtype=torch.long)

        with torch.no_grad():
            logits = model(x)

        self.assertEqual(logits.shape, (1, 10))
        self.assertFalse(torch.isnan(logits).any())
        self.assertFalse(torch.isinf(logits).any())


if __name__ == "__main__":
    unittest.main()

