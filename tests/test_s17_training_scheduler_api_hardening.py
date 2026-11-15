"""
Tests for S17: Training scheduler API hardening and ownership contract.
Verifies explicit scheduler_mode declarations, conflict detection, fail-fast validations,
and learning rate stepping dynamics.
"""

import unittest
from unittest.mock import MagicMock
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from viral_cnn.training.config import ControlledExperimentConfig
from viral_cnn.training.trainer import ControlledTrainer


class DummyModel(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        if x.dim() == 2:
            x = x.float()
        return self.fc(x)


class TestS17TrainingSchedulerApiHardening(unittest.TestCase):
    """Unit tests for training scheduler ownership and API hardening (S17)."""

    def setUp(self):
        torch.manual_seed(42)
        self.device = torch.device("cpu")
        self.num_classes = 10
        self.model = DummyModel(num_classes=self.num_classes)

        # Create dummy dataloader
        x = torch.randn(8, 512)
        y = torch.randint(0, self.num_classes, (8,))
        self.loader = DataLoader(TensorDataset(x, y), batch_size=4)

    def test_valid_scheduler_modes_accepted(self):
        """C1: ControlledExperimentConfig stores valid scheduler_mode values without error."""
        modes = ["internal_step", "external_batch", "external_epoch", "constant"]
        for mode in modes:
            warmup = 10 if mode == "internal_step" else 0
            cfg = ControlledExperimentConfig(
                architectures=("resnet50_1d",),
                num_classes=self.num_classes,
                warmup_steps=warmup,
                scheduler_mode=mode,
            )
            self.assertEqual(cfg.scheduler_mode, mode)

    def test_invalid_scheduler_mode_raises_value_error(self):
        """C2: ControlledExperimentConfig raises ValueError on unrecognized scheduler_mode."""
        with self.assertRaises(ValueError) as ctx:
            ControlledExperimentConfig(
                architectures=("resnet50_1d",),
                num_classes=self.num_classes,
                scheduler_mode="unsupported_schedule",  # type: ignore
            )
        self.assertIn("Unsupported scheduler_mode", str(ctx.exception))
        self.assertIn("internal_step", str(ctx.exception))

    def test_warmup_constant_conflict_raises_value_error(self):
        """C3: ControlledExperimentConfig raises ValueError when warmup_steps > 0 with constant mode."""
        with self.assertRaises(ValueError) as ctx:
            ControlledExperimentConfig(
                architectures=("resnet50_1d",),
                num_classes=self.num_classes,
                warmup_steps=20,
                scheduler_mode="constant",
                mode="controlled_parity",
            )
        self.assertIn("warmup_steps (20) > 0 requires a scheduling owner", str(ctx.exception))

    def test_conflicting_external_scheduler_raises_value_error(self):
        """C4: ControlledTrainer.train_epoch raises ValueError on conflicting external scheduler."""
        mock_scheduler = MagicMock()

        # 1. Conflict with internal_step
        cfg_internal = ControlledExperimentConfig(
            architectures=("resnet50_1d",),
            num_classes=self.num_classes,
            warmup_steps=10,
            scheduler_mode="internal_step",
        )
        trainer_internal = ControlledTrainer(model=self.model, config=cfg_internal, device=self.device)
        with self.assertRaises(ValueError) as ctx1:
            trainer_internal.train_epoch(self.loader, scheduler=mock_scheduler)
        self.assertIn("Conflicting schedulers", str(ctx1.exception))
        self.assertIn("internal_step", str(ctx1.exception))

        # 2. Conflict with constant
        cfg_const = ControlledExperimentConfig(
            architectures=("resnet50_1d",),
            num_classes=self.num_classes,
            warmup_steps=0,
            scheduler_mode="constant",
        )
        trainer_const = ControlledTrainer(model=self.model, config=cfg_const, device=self.device)
        with self.assertRaises(ValueError) as ctx2:
            trainer_const.train_epoch(self.loader, scheduler=mock_scheduler)
        self.assertIn("Conflicting schedulers", str(ctx2.exception))
        self.assertIn("constant", str(ctx2.exception))

    def test_missing_external_scheduler_raises_value_error(self):
        """C5: ControlledTrainer.train_epoch raises ValueError on missing external scheduler."""
        # 1. external_epoch missing scheduler
        cfg_epoch = ControlledExperimentConfig(
            architectures=("resnet50_1d",),
            num_classes=self.num_classes,
            warmup_steps=0,
            scheduler_mode="external_epoch",
        )
        trainer_epoch = ControlledTrainer(model=self.model, config=cfg_epoch, device=self.device)
        with self.assertRaises(ValueError) as ctx1:
            trainer_epoch.train_epoch(self.loader, scheduler=None)
        self.assertIn("Missing external scheduler", str(ctx1.exception))
        self.assertIn("external_epoch", str(ctx1.exception))

        # 2. external_batch missing scheduler
        cfg_batch = ControlledExperimentConfig(
            architectures=("resnet50_1d",),
            num_classes=self.num_classes,
            warmup_steps=0,
            scheduler_mode="external_batch",
        )
        trainer_batch = ControlledTrainer(model=self.model, config=cfg_batch, device=self.device)
        with self.assertRaises(ValueError) as ctx2:
            trainer_batch.train_epoch(self.loader, scheduler=None)
        self.assertIn("Missing external scheduler", str(ctx2.exception))
        self.assertIn("external_batch", str(ctx2.exception))

    def test_train_step_internal_step_schedules_learning_rate(self):
        """C6: ControlledTrainer.train_step under internal_step dynamically updates learning rate."""
        cfg = ControlledExperimentConfig(
            architectures=("resnet50_1d",),
            num_classes=self.num_classes,
            learning_rate=1e-3,
            warmup_steps=10,
            max_steps=50,
            scheduler_mode="internal_step",
        )
        trainer = ControlledTrainer(model=self.model, config=cfg, device=self.device)
        batch_x = torch.randn(4, 512)
        batch_y = torch.randint(0, self.num_classes, (4,))

        # Step 0: warmup step 1 / 10 * 1e-3 = 1e-4
        res0 = trainer.train_step(batch_x, batch_y)
        self.assertAlmostEqual(res0["learning_rate"], 1e-4, places=6)

        # Advance steps through warmup: 4 steps -> current_step reaches 4
        for _ in range(4):
            res = trainer.train_step(batch_x, batch_y)
        # At step 4, warmup step (4 + 1) / 10 * 1e-3 = 5e-4
        self.assertAlmostEqual(res["learning_rate"], 5e-4, places=6)

    def test_train_step_constant_mode_preserves_learning_rate(self):
        """C7: ControlledTrainer.train_step under constant mode preserves initial learning rate."""
        base_lr = 5e-4
        cfg = ControlledExperimentConfig(
            architectures=("resnet50_1d",),
            num_classes=self.num_classes,
            learning_rate=base_lr,
            warmup_steps=0,
            scheduler_mode="constant",
        )
        trainer = ControlledTrainer(model=self.model, config=cfg, device=self.device)
        batch_x = torch.randn(4, 512)
        batch_y = torch.randint(0, self.num_classes, (4,))

        for _ in range(5):
            res = trainer.train_step(batch_x, batch_y)
            self.assertEqual(res["learning_rate"], base_lr)
            self.assertEqual(trainer.get_current_lr(), base_lr)


if __name__ == "__main__":
    unittest.main()
