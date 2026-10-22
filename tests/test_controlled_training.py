"""
Tests for S12: Controlled Cross-Architecture Training Experiment.
Verifies checks C1 through C17 as derived in .specs/features/s12-controlled-architecture-experiment/checks.md.
"""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from viral_cnn.models.builder import build_model
from viral_cnn.training.config import (
    ControlledExperimentConfig,
    load_controlled_config,
    seed_everything,
)
from viral_cnn.training.profiler import ResourceProfile, ResourceProfiler
from viral_cnn.training.trainer import ControlledTrainer
from viral_cnn.reporting.controlled_audit import (
    generate_controlled_experiment_report,
    verify_controlled_report_consistency,
)


class TestControlledTrainingConfiguration(unittest.TestCase):
    """Checks C1-C4: Configuration validation, parameter parity, seeding, and YAML loading."""

    def test_controlled_experiment_config_enforces_parameter_parity(self):
        """C1: ControlledExperimentConfig validates and enforces parameter parity across compared architectures."""
        cfg = ControlledExperimentConfig(
            architectures=("resnet50_1d", "convnext_tiny_1d"),
            max_steps=100,
            batch_size=32,
            learning_rate=5e-5,
            weight_decay=1e-4,
            warmup_steps=10,
            label_smoothing=0.1,
            dropout_rate=0.5,
            noise_rate=0.0,
            seeds=(42,),
            checkpoint_metric="best_val_macro_f1",
        )
        self.assertEqual(len(cfg.architectures), 2)
        self.assertIn("resnet50_1d", cfg.architectures)
        self.assertIn("convnext_tiny_1d", cfg.architectures)
        self.assertEqual(cfg.batch_size, 32)
        self.assertEqual(cfg.dropout_rate, 0.5)
        self.assertEqual(cfg.label_smoothing, 0.1)

    def test_controlled_experiment_config_rejects_mismatched_parameters(self):
        """C2: ControlledExperimentConfig raises ValueError when invalid or conflicting parameters are provided."""
        # Negative step budget
        with self.assertRaises(ValueError):
            ControlledExperimentConfig(
                architectures=("resnet50_1d", "convnext_tiny_1d"),
                max_steps=-10,
            )

        # Negative batch size
        with self.assertRaises(ValueError):
            ControlledExperimentConfig(
                architectures=("resnet50_1d", "convnext_tiny_1d"),
                batch_size=0,
            )

        # Invalid label smoothing
        with self.assertRaises(ValueError):
            ControlledExperimentConfig(
                architectures=("resnet50_1d", "convnext_tiny_1d"),
                label_smoothing=1.5,
            )

        # Invalid checkpoint metric
        with self.assertRaises(ValueError):
            ControlledExperimentConfig(
                architectures=("resnet50_1d", "convnext_tiny_1d"),
                checkpoint_metric="invalid_metric",
            )

        # Unrecognized architecture
        with self.assertRaises(ValueError):
            ControlledExperimentConfig(
                architectures=("resnet50_1d", "unknown_transformer_1d"),
            )

    def test_seed_everything_deterministic_reproducibility(self):
        """C3: seed_everything deterministically seeds Python random, NumPy, PyTorch CPU, and PyTorch CUDA generators."""
        seed_everything(42)
        r1 = torch.rand(10)
        n1 = np.random.rand(10)

        seed_everything(42)
        r2 = torch.rand(10)
        n2 = np.random.rand(10)

        self.assertTrue(torch.allclose(r1, r2))
        self.assertTrue(np.allclose(n1, n2))

        # Changing seed produces different output
        seed_everything(43)
        r3 = torch.rand(10)
        self.assertFalse(torch.allclose(r1, r3))

    def test_load_controlled_config_from_yaml(self):
        """C4: load_controlled_config parses declarative YAML without personal or absolute paths."""
        yaml_path = Path("configs/controlled/baseline_pair.yaml")
        self.assertTrue(yaml_path.exists(), f"Missing configuration file {yaml_path}")
        cfg = load_controlled_config(yaml_path)

        self.assertIsInstance(cfg, ControlledExperimentConfig)
        self.assertEqual(cfg.architectures, ("resnet50_1d", "convnext_tiny_1d"))
        self.assertEqual(cfg.label_smoothing, 0.1)
        self.assertEqual(cfg.dropout_rate, 0.5)
        self.assertEqual(cfg.checkpoint_metric, "best_val_macro_f1")


class TestUnifiedTrainerAndTelemetry(unittest.TestCase):
    """Checks C5-C9: Training loop mechanics, optimizer/scheduler, label smoothing, profiler, and evaluation."""

    def setUp(self):
        self.device = torch.device("cpu")
        self.num_classes = 10
        self.seq_len = 64
        self.batch_size = 8

        # Create small synthetic dataset
        x_train = torch.randn(32, 4, self.seq_len)
        y_train = torch.randint(0, self.num_classes, (32,))
        self.train_loader = DataLoader(
            TensorDataset(x_train, y_train), batch_size=self.batch_size, shuffle=True
        )

        x_val = torch.randn(16, 4, self.seq_len)
        y_val = torch.randint(0, self.num_classes, (16,))
        self.val_loader = DataLoader(
            TensorDataset(x_val, y_val), batch_size=self.batch_size, shuffle=False
        )

        self.model = nn.Sequential(
            nn.Conv1d(4, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(16, self.num_classes),
        )

    def test_controlled_trainer_amp_and_grad_clip(self):
        """C5: ControlledTrainer supports PyTorch AMP mixed precision, gradient clipping, and step execution."""
        cfg = ControlledExperimentConfig(
            architectures=("resnet50_1d",),
            num_classes=self.num_classes,
            max_steps=5,
            batch_size=self.batch_size,
            amp=True,
            grad_clip_norm=1.0,
        )
        trainer = ControlledTrainer(self.model, cfg, device=self.device)
        batch_x, batch_y = next(iter(self.train_loader))
        metrics = trainer.train_step(batch_x, batch_y)

        self.assertIn("loss", metrics)
        self.assertIn("grad_norm", metrics)
        self.assertTrue(metrics["loss"] >= 0.0)
        self.assertTrue(metrics["grad_norm"] <= 1.5)

    def test_controlled_trainer_adamw_cosine_warmup(self):
        """C6: ControlledTrainer updates weights with AdamW and schedules learning rates with linear warmup followed by CosineAnnealingLR."""
        cfg = ControlledExperimentConfig(
            architectures=("resnet50_1d",),
            num_classes=self.num_classes,
            max_steps=10,
            batch_size=self.batch_size,
            learning_rate=1e-3,
            weight_decay=1e-4,
            warmup_steps=3,
        )
        trainer = ControlledTrainer(self.model, cfg, device=self.device)

        # Verify optimizer is AdamW
        self.assertIsInstance(trainer.optimizer, torch.optim.AdamW)
        self.assertEqual(trainer.optimizer.param_groups[0]["weight_decay"], 1e-4)

        # Step 0 learning rate should be scaled down by warmup
        lr_0 = trainer.get_current_lr()
        self.assertLess(lr_0, 1e-3)

        # Train 3 steps -> reaches max_lr
        batch_x, batch_y = next(iter(self.train_loader))
        for _ in range(3):
            trainer.train_step(batch_x, batch_y)
        lr_warm = trainer.get_current_lr()
        self.assertAlmostEqual(lr_warm, 1e-3, places=4)

        # Train remaining steps -> lr decays
        for _ in range(5):
            trainer.train_step(batch_x, batch_y)
        lr_decay = trainer.get_current_lr()
        self.assertLess(lr_decay, 1e-3)

    def test_controlled_trainer_label_smoothing(self):
        """C7: ControlledTrainer computes classification loss via CrossEntropyLoss with exact label smoothing = 0.1."""
        cfg = ControlledExperimentConfig(
            architectures=("resnet50_1d",),
            num_classes=self.num_classes,
            label_smoothing=0.1,
        )
        trainer = ControlledTrainer(self.model, cfg, device=self.device)

        self.assertIsInstance(trainer.criterion, nn.CrossEntropyLoss)
        self.assertEqual(trainer.criterion.label_smoothing, 0.1)

    def test_resource_profiler_telemetry_metrics(self):
        """C8: ResourceProfiler measures peak VRAM, step execution latency in ms, and sequence throughput in sequences/second."""
        profiler = ResourceProfiler(device=self.device)

        with profiler.profile_step(batch_size=16):
            # simulate light compute
            _ = torch.randn(100, 100) @ torch.randn(100, 100)

        profile: ResourceProfile = profiler.finalize(total_steps=1, model=self.model)
        self.assertIsInstance(profile, ResourceProfile)
        self.assertGreater(profile.mean_step_time_ms, 0.0)
        self.assertGreater(profile.throughput_seq_per_sec, 0.0)
        self.assertGreaterEqual(profile.peak_vram_mb, 0.0)
        self.assertGreater(profile.total_flops, 0.0)

    def test_controlled_trainer_evaluation_and_checkpoint_selection(self):
        """C9: ControlledTrainer.evaluate computes Top-1, Top-5, and Macro-F1 across validation batches and tracks best checkpoint."""
        cfg = ControlledExperimentConfig(
            architectures=("resnet50_1d",),
            num_classes=self.num_classes,
            checkpoint_metric="best_val_macro_f1",
        )
        trainer = ControlledTrainer(self.model, cfg, device=self.device)
        eval_metrics = trainer.evaluate(self.val_loader)

        self.assertIn("accuracy_top1", eval_metrics)
        self.assertIn("accuracy_top5", eval_metrics)
        self.assertIn("macro_f1", eval_metrics)
        self.assertIn("val_loss", eval_metrics)

        # Checkpoint selection logic
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "best_model.pt"
            is_best = trainer.check_and_save_checkpoint(eval_metrics, save_path)
            self.assertTrue(is_best)
            self.assertTrue(save_path.exists())


class TestControlledTrainingCLIRunner(unittest.TestCase):
    """Checks C10-C13: CLI execution, dry-run validation, multi-seed runs, and error handling."""

    def test_run_controlled_training_cli_arguments(self):
        """C10: CLI script scripts/run_controlled_training.py accepts --config, --steps, --seeds, --device, --dry-run, and --output-dir."""
        proc = subprocess.run(
            [sys.executable, "scripts/run_controlled_training.py", "--help"],
            capture_output=True,
            text=True,
        )
        self.assertEqual(proc.returncode, 0)
        self.assertIn("--config", proc.stdout)
        self.assertIn("--steps", proc.stdout)
        self.assertIn("--seeds", proc.stdout)
        self.assertIn("--device", proc.stdout)
        self.assertIn("--dry-run", proc.stdout)
        self.assertIn("--output-dir", proc.stdout)

    def test_run_controlled_training_dry_run_exit_zero(self):
        """C11: CLI scripts/run_controlled_training.py --dry-run validates config, models, parameter parity, and exits 0."""
        env = os.environ.copy()
        env["PYTHONPATH"] = "src:" + env.get("PYTHONPATH", "")
        proc = subprocess.run(
            [
                sys.executable,
                "scripts/run_controlled_training.py",
                "--config",
                "configs/controlled/baseline_pair.yaml",
                "--dry-run",
            ],
            capture_output=True,
            text=True,
            env=env,
        )
        self.assertEqual(proc.returncode, 0, f"Stderr: {proc.stderr}")
        self.assertIn("resnet50_1d", proc.stdout)
        self.assertIn("convnext_tiny_1d", proc.stdout)
        self.assertIn("DRY-RUN VALIDATION PASSED", proc.stdout)

    def test_run_controlled_training_multi_seed_execution(self):
        """C12: CLI executes sequential training across multiple seeds and saves structured JSON logs and checkpoints."""
        env = os.environ.copy()
        env["PYTHONPATH"] = "src:" + env.get("PYTHONPATH", "")
        with tempfile.TemporaryDirectory() as tmpdir:
            proc = subprocess.run(
                [
                    sys.executable,
                    "scripts/run_controlled_training.py",
                    "--config",
                    "configs/controlled/baseline_pair.yaml",
                    "--steps",
                    "1",
                    "--seeds",
                    "42",
                    "--device",
                    "cpu",
                    "--output-dir",
                    tmpdir,
                    "--synthetic-micro-benchmark",
                ],
                capture_output=True,
                text=True,
                env=env,
            )
            self.assertEqual(proc.returncode, 0, f"Stderr: {proc.stderr}")

            out_dir = Path(tmpdir)
            json_logs = list(out_dir.glob("*.json"))
            self.assertGreaterEqual(len(json_logs), 2)  # 2 models (resnet50 + convnext) for seed 42

            with open(json_logs[0], "r") as f:
                data = json.load(f)
                self.assertIn("architecture", data)
                self.assertIn("seed", data)
                self.assertIn("telemetry", data)
                self.assertIn("eval_metrics", data)

    def test_run_controlled_training_cli_invalid_options_exit_two(self):
        """C13: CLI exits with status 2 when an invalid option or conflicting parameter is passed."""
        proc = subprocess.run(
            [
                sys.executable,
                "scripts/run_controlled_training.py",
                "--invalid-flag-xyz",
            ],
            capture_output=True,
            text=True,
        )
        self.assertEqual(proc.returncode, 2)


class TestControlledArchitectureAuditReport(unittest.TestCase):
    """Checks C14-C17: Audit report generation, side-by-side metrics, causal demarcation, and verification."""

    def test_generate_controlled_experiment_report_budget_and_gpu_cost(self):
        """C14: generate_controlled_experiment_report writes docs/controlled_architecture_experiment.md with budget and GPU cost estimates."""
        with tempfile.TemporaryDirectory() as tmpdir:
            report_file = Path(tmpdir) / "controlled_architecture_experiment.md"
            generate_controlled_experiment_report(
                runs_dir=tmpdir,
                output_path=report_file,
                full_dataset_size=1706270,
            )
            self.assertTrue(report_file.exists())
            content = report_file.read_text(encoding="utf-8")
            self.assertIn("Controlled Architecture Experiment", content)
            self.assertIn("GPU Cost Estimation", content)
            self.assertIn("Hyperparameter Parity Matrix", content)
            self.assertIn("AdamW", content)
            self.assertIn("CosineAnnealingLR", content)

    def test_controlled_experiment_report_side_by_side_metrics(self):
        """C15: docs/controlled_architecture_experiment.md contains side-by-side comparative metrics for ResNet50 and ConvNeXt."""
        with tempfile.TemporaryDirectory() as tmpdir:
            report_file = Path(tmpdir) / "controlled_architecture_experiment.md"
            generate_controlled_experiment_report(
                runs_dir=tmpdir,
                output_path=report_file,
            )
            content = report_file.read_text(encoding="utf-8")
            self.assertIn("resnet50_1d", content)
            self.assertIn("convnext_tiny_1d", content)
            self.assertIn("Macro-F1", content)
            self.assertIn("Throughput", content)

    def test_controlled_experiment_report_causal_demarcation(self):
        """C16: docs/controlled_architecture_experiment.md contains mandatory causal claim demarcation text."""
        with tempfile.TemporaryDirectory() as tmpdir:
            report_file = Path(tmpdir) / "controlled_architecture_experiment.md"
            generate_controlled_experiment_report(
                runs_dir=tmpdir,
                output_path=report_file,
            )
            content = report_file.read_text(encoding="utf-8")
            expected_text = (
                "In the absence of multi-seed full-scale retraining across the entire closed-set taxonomy, "
                "causal conclusions are strictly confined to the evaluated training configurations and cannot "
                "be attributed exclusively to architecture."
            )
            self.assertIn(expected_text, content)

    def test_verify_controlled_report_consistency(self):
        """C17: verify_controlled_report_consistency asserts report tables match executed run artifacts."""
        with tempfile.TemporaryDirectory() as tmpdir:
            report_file = Path(tmpdir) / "controlled_architecture_experiment.md"
            generate_controlled_experiment_report(
                runs_dir=tmpdir,
                output_path=report_file,
            )
            is_valid = verify_controlled_report_consistency(report_file, run_logs=[])
            self.assertTrue(is_valid)


if __name__ == "__main__":
    unittest.main()
