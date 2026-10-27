"""
Proof tests for feature n09-notebook-throughput-and-gpu-pipeline-optimization (C1..C11).
"""

from __future__ import annotations

import inspect
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

_SRC = str(Path(__file__).resolve().parents[1] / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from viral_cnn.data.encoding import OneHotEmbedding
from viral_cnn.data.retraining_audit import audit_retraining_need
from viral_cnn.eval.calibration import TaxonomyEntropyAnalyzer
from viral_cnn.eval.confusion import get_confused_class_samples
from viral_cnn.notebook_runtime import create_optimized_dataloader
from viral_cnn.training.config import ControlledExperimentConfig
from viral_cnn.training.trainer import ControlledTrainer


class _TinySeqClassifier(nn.Module):
    def __init__(self, num_classes: int = 6):
        super().__init__()
        self.num_classes = num_classes
        self.embed = OneHotEmbedding(num_features=4)
        self.fc = nn.Linear(4 * 16, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.embed(x).flatten(1)
        return self.fc(h)


class TestN09PipelineOptimization(unittest.TestCase):
    """Proof suite for C1..C11 of n09-notebook-throughput-and-gpu-pipeline-optimization."""

    @classmethod
    def setUpClass(cls) -> None:
        os.environ["SMOKE_TEST"] = "1"
        os.environ["REMOTE_SERVER"] = "False"

    def test_c1_trainer_gpu_resident_epoch_metrics(self) -> None:
        """C1: train_epoch and evaluate_epoch accumulate metrics via torch.bincount without per-batch .cpu().numpy()."""
        src_train = inspect.getsource(ControlledTrainer.train_epoch)
        src_eval = inspect.getsource(ControlledTrainer.evaluate_epoch)
        self.assertIn("torch.bincount", src_train)
        self.assertIn("torch.bincount", src_eval)
        self.assertNotIn(".detach().cpu().numpy()", src_train)
        self.assertNotIn("preds.cpu().numpy()", src_eval)

        torch.manual_seed(42)
        num_classes = 6
        model = _TinySeqClassifier(num_classes=num_classes)
        cfg = ControlledExperimentConfig(
            architectures=("resnet50_1d",),
            num_classes=num_classes,
            learning_rate=1e-3,
            noise_rate=0.0,
            amp=False,
        )
        trainer = ControlledTrainer(model=model, config=cfg, device=torch.device("cpu"))

        x = torch.randint(0, 4, (24, 16), dtype=torch.long)
        y = torch.tensor([i % num_classes for i in range(24)], dtype=torch.long)
        loader = DataLoader(TensorDataset(x, y), batch_size=8, shuffle=False)

        val_res = trainer.evaluate_epoch(loader)
        self.assertEqual(val_res["total_samples"], 24)
        self.assertTrue(0.0 <= val_res["val_acc"] <= 1.0)
        self.assertTrue(0.0 <= val_res["val_acc_top5"] <= 1.0)
        self.assertTrue(0.0 <= val_res["val_f1_macro"] <= 1.0)
        self.assertTrue(0.0 <= val_res["val_f1_weighted"] <= 1.0)

        train_res = trainer.train_epoch(loader)
        self.assertEqual(train_res["total_samples"], 24)
        self.assertTrue(0.0 <= train_res["train_acc"] <= 1.0)
        self.assertTrue(0.0 <= train_res["train_f1"] <= 1.0)
        self.assertTrue(0.0 <= train_res["train_f1_weighted"] <= 1.0)

    def test_c2_trainer_zero_grad_set_to_none(self) -> None:
        """C2: ControlledTrainer.train_epoch calls optimizer.zero_grad(set_to_none=True)."""
        model = _TinySeqClassifier(num_classes=4)
        cfg = ControlledExperimentConfig(
            architectures=("resnet50_1d",),
            num_classes=4,
            noise_rate=0.0,
            amp=False,
        )
        trainer = ControlledTrainer(model=model, config=cfg, device=torch.device("cpu"))
        x = torch.randint(0, 4, (8, 16), dtype=torch.long)
        y = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3], dtype=torch.long)
        loader = DataLoader(TensorDataset(x, y), batch_size=4)

        calls: list[bool] = []
        orig_zero_grad = trainer.optimizer.zero_grad

        def _spy_zero_grad(set_to_none: bool = True) -> None:
            calls.append(set_to_none)
            return orig_zero_grad(set_to_none=set_to_none)

        with patch.object(trainer.optimizer, "zero_grad", side_effect=_spy_zero_grad):
            trainer.train_epoch(loader)

        self.assertEqual(len(calls), 2)
        self.assertTrue(all(flag is True for flag in calls))

    def test_c3_trainer_device_first_noise_and_generator_compat(self) -> None:
        """C3: Device-first noise injection when generator is None, and deterministic parity when CPU generator is passed."""
        model = _TinySeqClassifier(num_classes=4)
        cfg = ControlledExperimentConfig(
            architectures=("resnet50_1d",),
            num_classes=4,
            noise_rate=0.2,
            amp=False,
        )
        trainer = ControlledTrainer(model=model, config=cfg, device=torch.device("cpu"))
        x = torch.randint(0, 4, (12, 16), dtype=torch.long)
        y = torch.tensor([i % 4 for i in range(12)], dtype=torch.long)
        loader = DataLoader(TensorDataset(x, y), batch_size=6, shuffle=False)

        g1 = torch.Generator(device="cpu").manual_seed(1234)
        res1 = trainer.evaluate_epoch(loader, generator=g1)

        g2 = torch.Generator(device="cpu").manual_seed(1234)
        res2 = trainer.evaluate_epoch(loader, generator=g2)

        self.assertAlmostEqual(res1["val_loss"], res2["val_loss"], places=6)
        self.assertAlmostEqual(res1["val_acc"], res2["val_acc"], places=6)

    def test_c4_onehot_embedding_clamp_and_padding(self) -> None:
        """C4: OneHotEmbedding bounds indices via torch.clamp without .clone() and maps out-of-bounds (-1, 4, 5, 99) to zero vectors."""
        src = inspect.getsource(OneHotEmbedding.forward)
        self.assertIn("torch.clamp", src)
        self.assertNotIn(".clone()", src)

        layer = OneHotEmbedding(num_features=4)
        tokens = torch.tensor([[0, 1, 2, 3, 4, -1, 5, 99]], dtype=torch.long)
        out = layer(tokens)
        self.assertEqual(tuple(out.shape), (1, 4, 8))

        for idx in range(4):
            expected = torch.zeros(4)
            expected[idx] = 1.0
            self.assertTrue(torch.allclose(out[0, :, idx], expected))

        for pad_pos in (4, 5, 6, 7):
            self.assertTrue(torch.allclose(out[0, :, pad_pos], torch.zeros(4)))

    def test_c5_dataloader_persistent_workers_and_cap(self) -> None:
        """C5: create_optimized_dataloader sets persistent_workers=True when workers > 0 and caps default workers at 8."""
        src = inspect.getsource(create_optimized_dataloader)
        self.assertIn('"persistent_workers"] = True', src)
        self.assertIn("min(8,", src)

        ds = TensorDataset(torch.zeros((4, 8), dtype=torch.long))
        ds.file_paths = [Path(f"part.{i}.parquet") for i in range(16)]  # type: ignore[attr-defined]

        with patch("os.cpu_count", return_value=32):
            loader = create_optimized_dataloader(ds, is_smoke_test=False, device="cuda:0")
            self.assertEqual(loader.num_workers, 8)
            self.assertTrue(loader.persistent_workers)

    def test_c6_activation_entropy_fp16_no_nan(self) -> None:
        """C6: TaxonomyEntropyAnalyzer.calculate_activation_entropy casts FP16/BF16 to FP32, preventing NaN overflow and eps underflow."""
        model = _TinySeqClassifier(num_classes=4)
        s2g = torch.tensor([0, 0, 1, 1], dtype=torch.long)
        analyzer = TaxonomyEntropyAnalyzer(model, s2g, device=torch.device("cpu"))
        analyzer.remove_hook()

        act_fp16 = torch.zeros((4, 2048), dtype=torch.float16)
        act_fp16[:, :1024] = 80.0  # 1024 * 80 = 81920 > 65504 (FP16 max)

        h_act = analyzer.calculate_activation_entropy(act_fp16, normalize=True)
        self.assertEqual(h_act.dtype, torch.float32)
        self.assertEqual(tuple(h_act.shape), (4,))
        self.assertFalse(bool(torch.isnan(h_act).any()))
        self.assertFalse(bool(torch.isinf(h_act).any()))
        self.assertTrue(bool(torch.all((h_act >= 0.0) & (h_act <= 1.0))))

    def test_c7_vectorized_confused_class_samples(self) -> None:
        """C7: get_confused_class_samples uses torch.isin and respects max_batches and n_examples_per_class."""
        src = inspect.getsource(get_confused_class_samples)
        self.assertIn("torch.isin", src)

        batches = [
            (torch.full((4, 16), 1, dtype=torch.long), torch.tensor([0, 2, 0, 3], dtype=torch.long)),
            (torch.full((4, 16), 2, dtype=torch.long), torch.tensor([2, 1, 2, 0], dtype=torch.long)),
            (torch.full((4, 16), 3, dtype=torch.long), torch.tensor([0, 2, 1, 1], dtype=torch.long)),
        ]
        pairs = [(0.45, 0, 1), (0.30, 2, 3)]

        res_bounded = get_confused_class_samples(
            confused_pairs_info=pairs,
            test_loader=batches,  # type: ignore[arg-type]
            device=torch.device("cpu"),
            n_examples_per_class=2,
            max_batches=1,
        )
        self.assertEqual(len(res_bounded), 2)
        ex_0 = [r for r in res_bounded if r["class_true"]["id"] == 0][0]["class_true"]["examples"]
        ex_2 = [r for r in res_bounded if r["class_true"]["id"] == 2][0]["class_true"]["examples"]
        self.assertEqual(len(ex_0), 2)
        self.assertEqual(len(ex_2), 1)

    def test_c8_batched_sqlite_retraining_audit(self) -> None:
        """C8: audit_retraining_need batches SQLite hash lookups in chunks up to 900 and computes exact overlap counts."""
        src = inspect.getsource(audit_retraining_need)
        self.assertIn("WHERE hash IN", src)
        self.assertIn("900", src)

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            train_dir = tmp_path / "train"
            test_dir = tmp_path / "test"
            train_dir.mkdir()
            test_dir.mkdir()

            s1 = "ACGT" * 25
            s2 = "TGCA" * 25
            s3 = "AAAA" * 25

            pd.DataFrame({
                "sequence": [s1, s2],
                "species_id": [1, 2],
                "genome_id": ["G1", "G2"],
            }).to_parquet(train_dir / "part.0.parquet", index=False)

            pd.DataFrame({
                "sequence": [s1, s3],
                "species_id": [1, 2],
                "genome_id": ["G1", "G3"],
            }).to_parquet(test_dir / "part.0.parquet", index=False)

            out_dir = tmp_path / "out"
            report = audit_retraining_need(train_dir, test_dir, out_dir)
            self.assertEqual(report["train_rows"], 2)
            self.assertEqual(report["test_rows"], 2)
            self.assertEqual(report["exact_test_matches_train"], 1)
            self.assertAlmostEqual(report["exact_overlap_pct"], 50.0, places=4)

    def test_c9_finetune_notebooks_batch_size_and_dynamic_compile(self) -> None:
        """C9: Notebooks 03 and 04 configure batch_size=4096 (non-smoke) and torch.compile(model, dynamic=True)."""
        root = Path(__file__).resolve().parents[1]
        for nb_name in ("03_finetune_resnet50_species.ipynb", "04_finetune_convnext_species.ipynb"):
            nb_path = root / "notebooks" / "reproduction" / "short" / nb_name
            nb_data = json.loads(nb_path.read_text(encoding="utf-8"))
            full_src = "\n".join("".join(c.get("source", [])) for c in nb_data["cells"])
            self.assertIn("batch_size=32 if ctx.is_smoke_test else 4096", full_src)
            self.assertIn("torch.compile(model, dynamic=True)", full_src)

    def test_c10_finetune_notebooks_single_save_and_copyfile(self) -> None:
        """C10: Notebooks 03 and 04 call save_epoch_checkpoint once per epoch and use shutil.copyfile for numbered/best."""
        root = Path(__file__).resolve().parents[1]
        for nb_name in ("03_finetune_resnet50_species.ipynb", "04_finetune_convnext_species.ipynb"):
            nb_path = root / "notebooks" / "reproduction" / "short" / nb_name
            nb_data = json.loads(nb_path.read_text(encoding="utf-8"))
            full_src = "\n".join("".join(c.get("source", [])) for c in nb_data["cells"])
            self.assertEqual(full_src.count("trainer.save_epoch_checkpoint("), 1)
            self.assertIn("shutil.copyfile(last_ckpt,", full_src)
            self.assertIn("epoch_seconds", full_src)

    def test_c11_calibration_notebook_bounded_confused_samples(self) -> None:
        """C11: Notebook 02 passes max_batches=2 if ctx.is_smoke_test else 50 to get_confused_class_samples."""
        root = Path(__file__).resolve().parents[1]
        nb_path = root / "notebooks" / "reproduction" / "short" / "02_analyze_confidence_and_calibration.ipynb"
        nb_data = json.loads(nb_path.read_text(encoding="utf-8"))
        full_src = "\n".join("".join(c.get("source", [])) for c in nb_data["cells"])
        self.assertIn("max_batches=2 if ctx.is_smoke_test else 50", full_src)


if __name__ == "__main__":
    unittest.main()
