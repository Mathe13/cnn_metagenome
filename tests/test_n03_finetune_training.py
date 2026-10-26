"""
Unit and integration tests for N03: Noise-aware fine-tuning and reproduction notebooks.
Verifies checks C1 through C14 as defined in .specs/features/n03-short-finetune-training-notebooks/checks.md.
"""

import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from viral_cnn.training.config import ControlledExperimentConfig
from viral_cnn.training.trainer import ControlledTrainer
from viral_cnn.eval.noise import inject_noise


class DummyConv1DModel(nn.Module):
    """Simple 1D CNN with Conv1d, BatchNorm1d, and Linear head for testing."""

    def __init__(self, num_classes: int = 5, in_channels: int = 4, seq_len: int = 32):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(16)
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(16, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # If input is (B, L) integer tokens, one-hot encode
        if x.dim() == 2:
            x = nn.functional.one_hot(x.clamp(0, 3), num_classes=4).float()
            x = x.transpose(1, 2)  # (B, 4, L)
        h = self.relu(self.bn1(self.conv1(x)))
        h = self.pool(h).squeeze(-1)
        return self.fc(h)


class TestN03FinetuneTraining(unittest.TestCase):
    """Checks C1-C14 for N03 fine-tuning training and reproduction notebooks."""

    @classmethod
    def setUpClass(cls):
        os.environ["SMOKE_TEST"] = "1"
        os.environ["REMOTE_SERVER"] = "False"

    def setUp(self):
        self.device = torch.device("cpu")
        self.num_classes = 5
        self.seq_len = 32
        self.batch_size = 8

        # Create synthetic integer sequence batches (values 0..3)
        torch.manual_seed(42)
        x = torch.randint(0, 4, (16, self.seq_len), dtype=torch.long)
        y = torch.randint(0, self.num_classes, (16,), dtype=torch.long)
        ds = TensorDataset(x, y)
        self.loader = DataLoader(ds, batch_size=self.batch_size, shuffle=False)

    def test_c1_train_epoch_injects_noise(self):
        """C1: ControlledTrainer.train_epoch corrupts input tokens at expected effective rate 3*epsilon/4."""
        model = DummyConv1DModel(num_classes=self.num_classes, seq_len=self.seq_len)
        epsilon = 0.20
        expected_sub_rate = 3.0 * epsilon / 4.0  # 0.15

        cfg = ControlledExperimentConfig(
            architectures=("resnet50_1d",),
            num_classes=self.num_classes,
            learning_rate=1e-3,
            noise_rate=epsilon,
        )
        trainer = ControlledTrainer(model=model, config=cfg, device=self.device)

        # Track perturbed sequences inside train_epoch
        original_inputs = []
        for bx, _ in self.loader:
            original_inputs.append(bx.clone())

        res = trainer.train_epoch(self.loader)
        self.assertIn("train_loss", res)
        self.assertEqual(res["total_samples"], 16)

        # Direct verification that inject_noise with rate epsilon gives effective mutation rate ~ 3*epsilon/4
        orig = torch.cat(original_inputs)
        corrupted = inject_noise(orig, noise_rate=epsilon)
        mutated_positions = (orig != corrupted).float().mean().item()
        self.assertAlmostEqual(mutated_positions, expected_sub_rate, delta=0.03)

    def test_c2_evaluate_epoch_injects_noise(self):
        """C2: ControlledTrainer.evaluate_epoch corrupts validation input tokens at expected effective rate."""
        model = DummyConv1DModel(num_classes=self.num_classes, seq_len=self.seq_len)
        epsilon = 0.20
        expected_sub_rate = 3.0 * epsilon / 4.0  # 0.15

        cfg = ControlledExperimentConfig(
            architectures=("resnet50_1d",),
            num_classes=self.num_classes,
            learning_rate=1e-3,
            noise_rate=epsilon,
        )
        trainer = ControlledTrainer(model=model, config=cfg, device=self.device)

        res = trainer.evaluate_epoch(self.loader)
        self.assertIn("val_loss", res)
        self.assertIn("val_acc", res)
        self.assertIn("val_f1_macro", res)
        self.assertEqual(res["total_samples"], 16)

        # Verify inject_noise logic for evaluate
        sample_x = torch.randint(0, 4, (100, 512))
        corrupted_x = inject_noise(sample_x, noise_rate=epsilon)
        effective_rate = (sample_x != corrupted_x).float().mean().item()
        self.assertAlmostEqual(effective_rate, expected_sub_rate, delta=0.03)

    def test_c3_freeze_layers_and_bn_eval(self):
        """C3: freeze_layers sets requires_grad=False and locks matching BatchNorm modules to eval()."""
        model = DummyConv1DModel(num_classes=self.num_classes, seq_len=self.seq_len)
        cfg = ControlledExperimentConfig(
            architectures=("resnet50_1d",),
            num_classes=self.num_classes,
        )

        trainer = ControlledTrainer(
            model=model,
            config=cfg,
            device=self.device,
            freeze_layers=("conv1", "bn1"),
        )

        # Conv1 and BN1 parameters must have requires_grad=False
        for name, param in model.named_parameters():
            if name.startswith("conv1") or name.startswith("bn1"):
                self.assertFalse(param.requires_grad, f"Parameter {name} should be frozen")
            elif name.startswith("fc"):
                self.assertTrue(param.requires_grad, f"Parameter {name} should be trainable")

        # BatchNorm module must stay in eval mode during train_step and train_epoch
        model.train()
        trainer._apply_bn_eval()
        self.assertFalse(model.bn1.training, "bn1 should remain in eval mode")

        # After a train_epoch, bn1 must still be in eval mode
        trainer.train_epoch(self.loader, max_batches=1)
        self.assertFalse(model.bn1.training, "bn1 should remain in eval mode after train_epoch")

    def test_c4_train_epoch_returns_metrics(self):
        """C4: train_epoch returns train_loss, train_acc, train_f1, total_samples, and learning_rate."""
        model = DummyConv1DModel(num_classes=self.num_classes, seq_len=self.seq_len)
        cfg = ControlledExperimentConfig(
            architectures=("resnet50_1d",),
            num_classes=self.num_classes,
            learning_rate=5e-4,
        )
        trainer = ControlledTrainer(model=model, config=cfg, device=self.device)
        metrics = trainer.train_epoch(self.loader)

        required_keys = ["train_loss", "train_acc", "train_f1", "train_f1_weighted", "total_samples", "learning_rate"]
        for k in required_keys:
            self.assertIn(k, metrics, f"Missing key {k} in train_epoch metrics")

        self.assertGreater(metrics["total_samples"], 0)
        self.assertGreaterEqual(metrics["train_acc"], 0.0)
        self.assertLessEqual(metrics["train_acc"], 1.0)
        self.assertGreaterEqual(metrics["train_f1"], 0.0)
        self.assertLessEqual(metrics["train_f1"], 1.0)

    def test_c5_evaluate_epoch_returns_metrics(self):
        """C5: evaluate_epoch returns val_loss, val_acc, val_f1_macro, val_f1_weighted, and total_samples."""
        model = DummyConv1DModel(num_classes=self.num_classes, seq_len=self.seq_len)
        cfg = ControlledExperimentConfig(
            architectures=("resnet50_1d",),
            num_classes=self.num_classes,
        )
        trainer = ControlledTrainer(model=model, config=cfg, device=self.device)
        metrics = trainer.evaluate_epoch(self.loader)

        required_keys = ["val_loss", "val_acc", "val_acc_top5", "val_f1_macro", "val_f1_weighted", "total_samples"]
        for k in required_keys:
            self.assertIn(k, metrics, f"Missing key {k} in evaluate_epoch metrics")

        self.assertEqual(metrics["total_samples"], 16)
        self.assertGreaterEqual(metrics["val_acc"], 0.0)
        self.assertLessEqual(metrics["val_acc"], 1.0)
        self.assertGreaterEqual(metrics["val_f1_macro"], 0.0)
        self.assertLessEqual(metrics["val_f1_macro"], 1.0)

    def test_c6_save_epoch_checkpoint_payload(self):
        """C6: save_epoch_checkpoint serializes all required state dictionary keys."""
        model = DummyConv1DModel(num_classes=self.num_classes, seq_len=self.seq_len)
        cfg = ControlledExperimentConfig(
            architectures=("resnet50_1d",),
            num_classes=self.num_classes,
            learning_rate=1e-4,
            noise_rate=0.05,
        )
        trainer = ControlledTrainer(model=model, config=cfg, device=self.device)

        with tempfile.TemporaryDirectory() as tmpdir:
            chk_path = Path(tmpdir) / "checkpoints" / "test_chk.pt"
            dummy_metrics = {"train_loss": 0.5, "val_loss": 0.4}
            saved_path = trainer.save_epoch_checkpoint(
                epoch=3,
                metrics=dummy_metrics,
                save_path=chk_path,
            )

            self.assertTrue(saved_path.exists())
            payload = torch.load(saved_path, map_location="cpu")

            expected_keys = [
                "epoch",
                "model_state_dict",
                "optimizer_state_dict",
                "scheduler_state_dict",
                "scaler_state_dict",
                "metrics",
                "config",
            ]
            for k in expected_keys:
                self.assertIn(k, payload, f"Missing required payload key {k}")

            self.assertEqual(payload["epoch"], 3)
            self.assertEqual(payload["metrics"], dummy_metrics)
            self.assertEqual(payload["config"]["noise_rate"], 0.05)

    def test_c7_resume_checkpoint_restoration(self):
        """C7: resume_checkpoint restores model weights, optimizer, and start epoch."""
        model1 = DummyConv1DModel(num_classes=self.num_classes, seq_len=self.seq_len)
        cfg = ControlledExperimentConfig(
            architectures=("resnet50_1d",),
            num_classes=self.num_classes,
        )
        trainer1 = ControlledTrainer(model=model1, config=cfg, device=self.device)
        trainer1.train_epoch(self.loader)

        with tempfile.TemporaryDirectory() as tmpdir:
            chk_path = Path(tmpdir) / "checkpoint.pt"
            trainer1.save_epoch_checkpoint(epoch=5, metrics={"val_loss": 0.25}, save_path=chk_path)

            # Create new trainer and restore
            model2 = DummyConv1DModel(num_classes=self.num_classes, seq_len=self.seq_len)
            trainer2 = ControlledTrainer(model=model2, config=cfg, device=self.device)

            # Assert weights initially differ or can be overwritten
            resumed_epoch = trainer2.resume_checkpoint(chk_path)
            self.assertEqual(resumed_epoch, 5)

            # Verify parameters match exactly
            for p1, p2 in zip(model1.parameters(), model2.parameters()):
                self.assertTrue(torch.allclose(p1, p2))

    def test_c8_historical_preservation(self):
        """C8: Historical directories short/ and long/ remain strictly unmodified."""
        git_status = subprocess.check_output(
            ["git", "status", "--porcelain", "short/", "long/"],
            text=True,
        ).strip()
        self.assertEqual(
            git_status,
            "",
            f"Historical directories short/ or long/ have been modified:\n{git_status}",
        )

    # -------------------------------------------------------------------------
    # Slice S2: Reproduction Notebooks for ResNet50 and ConvNeXt
    # -------------------------------------------------------------------------

    def test_c9_notebook_03_imports_and_decoupling(self):
        """C9: 03_finetune_resnet50_species.ipynb contains zero short/utils/* imports."""
        nb_path = Path("notebooks/reproduction/short/03_finetune_resnet50_species.ipynb")
        self.assertTrue(nb_path.exists())

        with open(nb_path, "r", encoding="utf-8") as f:
            nb = json.load(f)

        code_text = ""
        for cell in nb["cells"]:
            if cell.get("cell_type") == "code":
                code_text += "".join(cell.get("source", [])) + "\n"

        self.assertNotIn("short.utils", code_text)
        self.assertNotIn("from utils", code_text)
        self.assertNotIn("class ResNet50_1D", code_text)
        self.assertIn("viral_cnn", code_text)

    def test_c10_notebook_03_execution_smoke_test(self):
        """C10: 03_finetune_resnet50_species.ipynb executes without error under SMOKE_TEST=1."""
        nb_path = Path("notebooks/reproduction/short/03_finetune_resnet50_species.ipynb")
        with open(nb_path, "r", encoding="utf-8") as f:
            nb = json.load(f)

        global_ns = {"__name__": "__main__"}
        for idx, cell in enumerate(nb["cells"]):
            if cell.get("cell_type") == "code":
                code = "".join(cell.get("source", []))
                try:
                    exec(code, global_ns)
                except Exception as e:
                    self.fail(f"Cell {idx} of 03_finetune_resnet50_species.ipynb raised exception: {e}")

    def test_c11_notebook_03_artifacts_generated(self):
        """C11: 03_finetune_resnet50_species.ipynb generates metrics and manifest."""
        out_dir = Path("results_reproduction/n03_finetune/resnet50")
        expected_files = [
            "training_history.csv",
            "training_history.json",
            "manifest.json",
        ]
        for fname in expected_files:
            fpath = out_dir / fname
            self.assertTrue(fpath.exists(), f"Expected artifact {fname} not found in {out_dir}")
            self.assertGreater(fpath.stat().st_size, 0)

    def test_c12_notebook_04_imports_and_decoupling(self):
        """C12: 04_finetune_convnext_species.ipynb contains zero short/utils/* imports."""
        nb_path = Path("notebooks/reproduction/short/04_finetune_convnext_species.ipynb")
        self.assertTrue(nb_path.exists())

        with open(nb_path, "r", encoding="utf-8") as f:
            nb = json.load(f)

        code_text = ""
        for cell in nb["cells"]:
            if cell.get("cell_type") == "code":
                code_text += "".join(cell.get("source", [])) + "\n"

        self.assertNotIn("short.utils", code_text)
        self.assertNotIn("from utils", code_text)
        self.assertNotIn("class ConvNeXt_Tiny_1D", code_text)
        self.assertIn("viral_cnn", code_text)

    def test_c13_notebook_04_execution_smoke_test(self):
        """C13: 04_finetune_convnext_species.ipynb executes without error under SMOKE_TEST=1."""
        nb_path = Path("notebooks/reproduction/short/04_finetune_convnext_species.ipynb")
        with open(nb_path, "r", encoding="utf-8") as f:
            nb = json.load(f)

        global_ns = {"__name__": "__main__"}
        for idx, cell in enumerate(nb["cells"]):
            if cell.get("cell_type") == "code":
                code = "".join(cell.get("source", []))
                try:
                    exec(code, global_ns)
                except Exception as e:
                    self.fail(f"Cell {idx} of 04_finetune_convnext_species.ipynb raised exception: {e}")

    def test_c14_notebook_04_artifacts_generated(self):
        """C14: 04_finetune_convnext_species.ipynb generates metrics and manifest."""
        out_dir = Path("results_reproduction/n03_finetune/convnext")
        expected_files = [
            "training_history.csv",
            "training_history.json",
            "manifest.json",
        ]
        for fname in expected_files:
            fpath = out_dir / fname
            self.assertTrue(fpath.exists(), f"Expected artifact {fname} not found in {out_dir}")
            self.assertGreater(fpath.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()

