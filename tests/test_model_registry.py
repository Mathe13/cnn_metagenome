"""
Unit and contract tests for S04: Model Registry, Checkpoint Loading, and Training Manifest.
"""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import torch
import yaml

from viral_cnn.models.registry import (
    REGISTERED_VARIANTS,
    ModelVariantSpec,
    list_model_variants,
    get_model_spec,
    load_model_checkpoint,
    build_registered_model,
)
from viral_cnn.models.resnet import ResNet50_1D
from viral_cnn.models.convnext import ConvNeXt_Tiny_1D

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class TestModelRegistryAndTrainingManifest(unittest.TestCase):

    # --- S1: Documentation and Manifest Audits (C1 - C4) ---

    def test_manifest_documents_all_seven_model_variants(self):
        """C1: docs/model_registry.md documents all 7 model variants with architecture and checkpoint."""
        doc_path = PROJECT_ROOT / "docs" / "model_registry.md"
        self.assertTrue(doc_path.exists(), "docs/model_registry.md must exist")
        content = doc_path.read_text(encoding="utf-8")

        expected_variants = [
            "resnet_clean_species",
            "resnet_noise_species",
            "convnext_zero_species",
            "convnext_transfer_species",
            "convnext_noise_species",
            "resnet_genus",
            "convnext_genus",
        ]
        for variant in expected_variants:
            self.assertIn(variant, content, f"Variant '{variant}' missing from docs/model_registry.md")

        self.assertIn("13.683", content, "Species class count 13,683 must be documented")
        self.assertIn("2.702", content, "Genus class count 2,702 must be documented")

    def test_manifest_documents_encoding_and_label_offset(self):
        """C2: docs/model_registry.md documents 4-channel encoding and 1-based to 0-based offset."""
        doc_path = PROJECT_ROOT / "docs" / "model_registry.md"
        content = doc_path.read_text(encoding="utf-8")

        self.assertIn("A/C/G/T", content)
        self.assertIn("model_label", content)
        self.assertIn("csv_id", content)
        self.assertIn("vetor nulo", content.lower())

    def test_manifest_records_verified_and_unverified_hyperparameters(self):
        """C3: docs/model_registry.md documents verified hyperparameters and explicitly labels unverified gaps."""
        doc_path = PROJECT_ROOT / "docs" / "model_registry.md"
        content = doc_path.read_text(encoding="utf-8")

        self.assertIn("AdamW", content)
        self.assertIn("CosineAnnealingLR", content)
        self.assertIn("label_smoothing", content)
        self.assertIn("GradScaler", content)
        self.assertIn("unverified", content.lower(), "Must explicitly label gaps as unverified")

    def test_manifest_distinguishes_best_validation_from_last_checkpoint(self):
        """C4: docs/model_registry.md explicitly distinguishes best validation checkpoint from last epoch."""
        doc_path = PROJECT_ROOT / "docs" / "model_registry.md"
        content = doc_path.read_text(encoding="utf-8")

        self.assertIn("finetune_noise_14.pt", content)
        self.assertIn("best_resnet_classifier__fast_phase4.pt", content)
        self.assertIn("última época", content.lower())
        self.assertIn("melhor", content.lower())

    # --- S2: Unified Model Registry and Checkpoint Loader Engine (C5 - C8) ---

    def test_checkpoint_loader_handles_bare_and_nested_dict(self):
        """C5: load_model_checkpoint loads weights identically from bare state_dict or nested dict."""
        model1 = ResNet50_1D(num_classes=10)
        model2 = ResNet50_1D(num_classes=10)

        # Generate synthetic state_dict handling floating point and non-floating buffers (e.g. num_batches_tracked)
        torch.manual_seed(123)
        dummy_state = {
            k: (torch.randn_like(v) if v.is_floating_point() else v.clone())
            for k, v in model1.state_dict().items()
        }

        # 1. Bare state_dict
        load_model_checkpoint(model1, dummy_state, expected_classes=10)

        # 2. Nested model_state_dict with metadata
        nested_checkpoint = {
            "epoch": 14,
            "metrics": {"val_loss": 0.123},
            "model_state_dict": dummy_state,
        }
        metadata = load_model_checkpoint(model2, nested_checkpoint, expected_classes=10)
        self.assertEqual(metadata.get("epoch"), 14)
        self.assertEqual(metadata.get("metrics"), {"val_loss": 0.123})

        # Verify exact weight identity
        for k in model1.state_dict():
            self.assertTrue(
                torch.equal(model1.state_dict()[k], model2.state_dict()[k]),
                f"Weights for key {k} must match identically between bare and nested dict loading",
            )

    def test_checkpoint_loader_strips_orig_mod_compile_prefix(self):
        """C6: load_model_checkpoint strips _orig_mod. compile prefixes without missing key errors."""
        model = ConvNeXt_Tiny_1D(num_classes=10)
        torch.manual_seed(456)
        clean_state = {
            k: (torch.randn_like(v) if v.is_floating_point() else v.clone())
            for k, v in model.state_dict().items()
        }

        # Emulate torch.compile prefix
        compiled_state = {f"_orig_mod.{k}": v for k, v in clean_state.items()}

        # Should load cleanly without error
        load_model_checkpoint(model, compiled_state, expected_classes=10, strict=True)

        for k in model.state_dict():
            self.assertTrue(torch.equal(model.state_dict()[k], clean_state[k]))

    def test_checkpoint_loader_rejects_class_count_mismatch(self):
        """C7: load_model_checkpoint raises ValueError when checkpoint head shape disagrees with model/expected."""
        model = ResNet50_1D(num_classes=13683)

        # Create checkpoint with 2702 classes (genus head)
        genus_model = ResNet50_1D(num_classes=2702)
        genus_state = genus_model.state_dict()

        with self.assertRaises(ValueError) as ctx:
            load_model_checkpoint(model, genus_state, expected_classes=13683)

        err_msg = str(ctx.exception)
        self.assertIn("2702", err_msg)
        self.assertIn("13683", err_msg)
        self.assertIn("Classification head class count mismatch", err_msg)

    def test_model_registry_api_queries_and_instantiation(self):
        """C8: Programmatic registry APIs query specifications and instantiate all 7 variants."""
        variants = list_model_variants()
        self.assertEqual(len(variants), 7)

        for var_name in variants:
            spec = get_model_spec(var_name)
            self.assertIsInstance(spec, ModelVariantSpec)
            self.assertEqual(spec.name, var_name)
            self.assertIn(spec.rank, ["species", "genus"])
            if spec.rank == "species":
                self.assertEqual(spec.num_classes, 13683)
            else:
                self.assertEqual(spec.num_classes, 2702)

            # Test instantiation
            model = build_registered_model(var_name, device="cpu")
            self.assertIsInstance(model, torch.nn.Module)
            self.assertEqual(model.num_classes, spec.num_classes)

    # --- S3: Declarative Configuration and CLI Integration (C9 - C11) ---

    def test_models_yaml_declares_all_variants_without_personal_paths(self):
        """C9: configs/models.yaml defines all 7 model variants with relative paths and no personal paths."""
        yaml_path = PROJECT_ROOT / "configs" / "models.yaml"
        self.assertTrue(yaml_path.exists(), "configs/models.yaml must exist")

        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        models_data = data.get("models", {})
        self.assertEqual(len(models_data), 7)

        for var_name in list_model_variants():
            self.assertIn(var_name, models_data)
            entry = models_data[var_name]
            ckpt_path = entry.get("checkpoint_rel_path", "")
            self.assertFalse(ckpt_path.startswith("/"), f"Path {ckpt_path} must be relative, not absolute")
            self.assertNotIn("/home/", ckpt_path, f"Path {ckpt_path} must not contain personal paths")

    def test_run_inference_cli_with_model_variant_flag(self):
        """C10: scripts/run_inference.py --model-variant instantiates variant and runs inference."""
        cli_script = PROJECT_ROOT / "scripts" / "run_inference.py"
        cmd = [
            sys.executable,
            str(cli_script),
            "--model-variant",
            "resnet_clean_species",
            "--sequence",
            "ACGTACGTACGT",
            "--seed",
            "42",
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, cwd=str(PROJECT_ROOT))
        self.assertEqual(res.returncode, 0, f"CLI exited with error: {res.stderr}")

        out_json = json.loads(res.stdout)
        self.assertEqual(out_json["status"], "success")
        self.assertEqual(out_json["model_type"], "resnet50_1d")
        self.assertEqual(out_json["num_classes"], 13683)
        self.assertEqual(len(out_json["top5_classes"]), 5)

        # Test invalid model-variant returns exit code 2
        bad_cmd = [
            sys.executable,
            str(cli_script),
            "--model-variant",
            "invalid_variant_name",
            "--sequence",
            "ACGTACGT",
        ]
        bad_res = subprocess.run(bad_cmd, capture_output=True, text=True, cwd=str(PROJECT_ROOT))
        self.assertEqual(bad_res.returncode, 2)
        self.assertIn("Error: Unknown model variant", bad_res.stderr)

    def test_model_registry_deterministic_synthetic_execution(self):
        """C11: Model registry and checkpoint loading execute deterministically without external weights."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_file = Path(tmpdir) / "synthetic_model.pt"

            # Create synthetic checkpoint
            torch.manual_seed(999)
            base_model = ConvNeXt_Tiny_1D(num_classes=10)
            synthetic_weights = base_model.state_dict()
            torch.save({"model_state_dict": synthetic_weights, "epoch": 5}, ckpt_file)

            # Load into fresh model
            loaded_model = ConvNeXt_Tiny_1D(num_classes=10)
            metadata = load_model_checkpoint(loaded_model, ckpt_file, expected_classes=10)
            self.assertEqual(metadata.get("epoch"), 5)

            # Test deterministic forward pass
            dummy_tokens = torch.tensor([[0, 1, 2, 3, 0, 1, 2, 3]], dtype=torch.long)
            base_model.eval()
            loaded_model.eval()
            with torch.no_grad():
                out_base = base_model(dummy_tokens)
                out_loaded = loaded_model(dummy_tokens)

            self.assertTrue(torch.allclose(out_base, out_loaded, atol=1e-6))


if __name__ == "__main__":
    unittest.main()
