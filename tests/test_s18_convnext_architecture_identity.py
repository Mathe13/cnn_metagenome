"""
Tests for S18: ConvNeXt architecture identity and metadata contract.
Verifies HybridConvNeXt1D canonical identity, backward-compatible aliases,
structured metadata schema, builder resolution, and parameter counts.
"""

import unittest
import torch

from viral_cnn.models.convnext import (
    HybridConvNeXt1D,
    ConvNeXtInspired1D,
    ConvNeXt_Tiny_1D,
)
from viral_cnn.models.builder import build_model
from viral_cnn.training.profiler import profile_inference_model


class TestS18ConvNeXtArchitectureIdentity(unittest.TestCase):
    """Unit tests for ConvNeXt 1D architecture identity and metadata (S18)."""

    def test_hybrid_convnext_metadata_schema(self):
        """C1: HybridConvNeXt1D exposes ARCHITECTURE_METADATA and get_architecture_metadata() matching required schema."""
        expected_meta = {
            "architecture_family": "convnext-inspired",
            "implementation": "HybridConvNeXt1D",
            "stage_depths": [3, 4, 6, 3],
            "stem": "resnet-style",
            "pooling": "global-max",
        }
        # Class-level verification
        self.assertEqual(HybridConvNeXt1D.ARCHITECTURE_METADATA, expected_meta)
        self.assertEqual(HybridConvNeXt1D.get_architecture_metadata(), expected_meta)

        # Instance-level verification
        model = HybridConvNeXt1D(num_classes=10)
        self.assertEqual(model.ARCHITECTURE_METADATA, expected_meta)
        self.assertEqual(model.get_architecture_metadata(), expected_meta)

    def test_backward_compatible_aliases(self):
        """C2: ConvNeXt_Tiny_1D and ConvNeXtInspired1D resolve directly to HybridConvNeXt1D preserving identity."""
        self.assertIs(ConvNeXtInspired1D, HybridConvNeXt1D)
        self.assertIs(ConvNeXt_Tiny_1D, HybridConvNeXt1D)

        model_legacy = ConvNeXt_Tiny_1D(num_classes=10)
        self.assertIsInstance(model_legacy, HybridConvNeXt1D)
        self.assertEqual(
            model_legacy.get_architecture_metadata()["implementation"],
            "HybridConvNeXt1D",
        )

    def test_builder_resolves_all_aliases(self):
        """C3: build_model successfully instantiates HybridConvNeXt1D for all alias identifiers."""
        identifiers = [
            "hybrid_convnext_1d",
            "convnext_inspired_1d",
            "convnext_tiny_1d",
            "convnext",
        ]
        for name in identifiers:
            model = build_model(name, num_classes=10)
            self.assertIsInstance(model, HybridConvNeXt1D, f"Failed for identifier '{name}'")
            self.assertEqual(model.num_classes, 10)

    def test_state_dict_and_forward_parity(self):
        """C4: State dict weights load symmetrically with strict=True and produce identical logits."""
        torch.manual_seed(42)
        model_a = ConvNeXt_Tiny_1D(num_classes=10)
        model_b = HybridConvNeXt1D(num_classes=10)

        # Transfer state dict with strict=True
        model_b.load_state_dict(model_a.state_dict(), strict=True)
        model_a.eval()
        model_b.eval()

        dummy_seq = torch.randint(0, 4, (2, 512))
        with torch.no_grad():
            logits_a = model_a(dummy_seq)
            logits_b = model_b(dummy_seq)

        self.assertTrue(torch.allclose(logits_a, logits_b, atol=1e-7))

    def test_parameter_profile_decomposition(self):
        """C5: profile_inference_model decomposes parameters matching exact expected counts."""
        # 1. Species level (13,683 classes)
        model_sp = HybridConvNeXt1D(num_classes=13683)
        prof_sp = profile_inference_model(model_sp, num_warmup=1, num_runs=1)
        self.assertEqual(prof_sp["backbone_params"], 135977344)
        self.assertEqual(prof_sp["head_params"], 28036467)
        self.assertEqual(prof_sp["total_params"], 164013811)

        # 2. Genus level (2,702 classes)
        model_gen = HybridConvNeXt1D(num_classes=2702)
        prof_gen = profile_inference_model(model_gen, num_warmup=1, num_runs=1)
        self.assertEqual(prof_gen["backbone_params"], 135977344)
        self.assertEqual(prof_gen["head_params"], 5536398)
        self.assertEqual(prof_gen["total_params"], 141513742)


if __name__ == "__main__":
    unittest.main()
