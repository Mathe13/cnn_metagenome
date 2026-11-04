"""Focused contracts for the minimum C05 fine-tuning audit trail."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
import pandas as pd
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from viral_cnn.training.audit import (
    protocol_audit,
    require_empirical_inputs,
    select_checkpoint_epoch,
    sha256_file,
    taxonomy_order_sha256,
    write_c05_manifest,
)
from viral_cnn.training.config import ControlledExperimentConfig
from viral_cnn.training.trainer import ControlledTrainer
from viral_cnn.eval.noise import inject_noise
from viral_cnn.data.parquet_dataset import ParquetBatchIterableDataset
from viral_cnn.reporting.controlled_audit import generate_controlled_experiment_report


class TinyClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 3)

    def forward(self, x):
        return self.linear(x)


class TestC05Preflight(unittest.TestCase):
    def test_empirical_preflight_requires_all_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaisesRegex(FileNotFoundError, "before model construction"):
                require_empirical_inputs([], [], [], root / "taxonomy.csv", root / "base.pt")

    def test_parquet_loader_yields_without_identity_tracking(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "partition.parquet"
            pd.DataFrame({"sequence_padded": ["ACGT", "TGCA"], "label": [0, 1]}).to_parquet(path, index=False)
            dataset = ParquetBatchIterableDataset(path, maxlen=4, batch_size=2, include_identity=False)
            batches = list(dataset)
            self.assertEqual(len(batches), 1)
            sequences, labels = batches[0]
            self.assertEqual(sequences.tolist(), [[0, 1, 2, 3], [3, 2, 1, 0]])
            self.assertEqual(labels.tolist(), [0, 1])


class TestC05Manifest(unittest.TestCase):
    def test_manifest_records_lineage_and_checkpoint_roles(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = root / "base.pt"
            checkpoint.write_bytes(b"weights")
            taxonomy = root / "species.csv"
            taxonomy.write_text("species,id\nSpecies A,1\nSpecies B,2\n", encoding="utf-8")
            order_sha, class_count = taxonomy_order_sha256(taxonomy)
            self.assertEqual(class_count, 2)
            protocol = protocol_audit(
                base_variant="resnet_clean_species",
                base_checkpoint_sha256=sha256_file(checkpoint),
                rank="species",
                num_classes=3,
                class_order_sha256=order_sha,
                effective_dropout=0.5,
                dropout_source="model config",
                config={"noise_rate": 0.05},
            )
            destination = root / "manifest.json"
            manifest = write_c05_manifest(
                destination,
                protocol=protocol,
                status="COMPLETE",
                is_smoke_test=False,
                inputs={"train": [{"sha256": "trainhash"}]},
                outputs=[checkpoint],
                completed_epochs=2,
                planned_epochs=2,
                selected_checkpoint={"role": "selected", "rule": "last", "epoch": 2, "sha256": "last-hash"},
                checkpoint_roles={"last": {"sha256": "last-hash"}, "best": {"sha256": "best-hash"}},
                metrics={"test_support": 0},
            )
            loaded = json.loads(destination.read_text())
            self.assertEqual(loaded["protocol"], "c05_finetune_audit_v1")
            self.assertEqual(loaded["base_variant"], "resnet_clean_species")
            self.assertEqual(loaded["base_checkpoint_sha256"], sha256_file(checkpoint))
            self.assertEqual(loaded["class_order_sha256"], order_sha)
            self.assertEqual(loaded["rank"], "species")
            self.assertEqual(loaded["effective_dropout"], 0.5)
            self.assertEqual(loaded["selected_checkpoint"]["role"], "selected")
            self.assertIn("last", loaded["checkpoint_roles"])
            self.assertIn("best", loaded["checkpoint_roles"])
            self.assertEqual(manifest["status"], "COMPLETE")

    def test_protocol_audit_reports_historical_divergences(self):
        record = protocol_audit(
            base_variant="convnext_transfer_species",
            base_checkpoint_sha256="h",
            rank="species",
            num_classes=3,
            class_order_sha256="c",
            effective_dropout=None,
            dropout_source="unknown",
            config={"scheduler_step_unit": "epoch"},
        )
        history = record["historical_comparison"]
        self.assertEqual(history["scheduler_step_unit"], "epoch")
        self.assertEqual(history["convnext_dropout_registry"], 0.7)
        self.assertEqual(history["convnext_dropout_historical_notebook"], 0.5)
        self.assertEqual(history["published_checkpoint_equivalence"], "unknown without original run configuration/logs")

    def test_manifest_never_promotes_smoke_or_incomplete_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            destination = Path(tmp) / "manifest.json"
            smoke = write_c05_manifest(
                destination, protocol={"protocol": "c05_finetune_audit_v1"}, status="COMPLETE",
                is_smoke_test=True, inputs={}, outputs=[], completed_epochs=1, planned_epochs=1,
                selected_checkpoint=None, metrics={},
            )
            self.assertEqual(smoke["status"], "SYNTHETIC")
            incomplete = write_c05_manifest(
                destination, protocol={"protocol": "c05_finetune_audit_v1"}, status="COMPLETE",
                is_smoke_test=False, inputs={}, outputs=[], completed_epochs=1, planned_epochs=2,
                selected_checkpoint=None, metrics={},
            )
            self.assertEqual(incomplete["status"], "PARTIAL")
            unselected = write_c05_manifest(
                destination, protocol={"protocol": "c05_finetune_audit_v1"}, status="COMPLETE",
                is_smoke_test=False, inputs={}, outputs=[], completed_epochs=2, planned_epochs=2,
                selected_checkpoint=None, metrics={},
            )
            self.assertEqual(unselected["status"], "PARTIAL")


class TestC05Training(unittest.TestCase):
    def _trainer(self):
        config = ControlledExperimentConfig(
            num_classes=3, noise_rate=0.05, amp=False, max_steps=4,
            warmup_steps=0, learning_rate=1e-3, mode="historical_reproduction",
        )
        return ControlledTrainer(TinyClassifier(), config, torch.device("cpu"))

    def test_train_and_validation_noise_are_audited(self):
        import viral_cnn.training.trainer as trainer_module

        trainer = self._trainer()
        x = torch.eye(4)[:3]
        y = torch.tensor([0, 1, 2])
        loader = DataLoader(TensorDataset(x, y), batch_size=3)
        original = trainer_module.inject_noise
        observed = []

        def record_noise(values, noise_rate, generator=None):
            observed.append(float(noise_rate))
            return original(values, noise_rate=noise_rate, generator=generator)

        with patch.object(trainer_module, "inject_noise", side_effect=record_noise):
            train_metrics = trainer.train_epoch(loader, generator=torch.Generator().manual_seed(42))
            val_metrics = trainer.evaluate_epoch(loader, generator=torch.Generator().manual_seed(43))
        self.assertEqual(observed, [0.05, 0.05])
        self.assertEqual(train_metrics["total_samples"], 3)
        self.assertEqual(val_metrics["total_samples"], 3)
        self.assertEqual(train_metrics["steps_attempted"], 1)
        self.assertEqual(train_metrics["steps_effective"], 1)
        self.assertEqual(train_metrics["train_acc_top5"], 1.0)
        resampled = inject_noise(torch.zeros(4096, dtype=torch.long), 1.0, generator=torch.Generator().manual_seed(42))
        self.assertTrue(torch.all((resampled >= 0) & (resampled < 4)))
        self.assertGreater(int((resampled == 0).sum()), 0)  # resampling may retain the original base
        self.assertGreater(int((resampled != 0).sum()), 0)

    def test_pretraining_evaluation_labels_support_and_noise(self):
        # The notebook contract requires separate clean/noisy labels and support reporting.
        for path in (
            "notebooks/reproduction/short/03_finetune_resnet50_species.ipynb",
            "notebooks/reproduction/short/04_finetune_convnext_species.ipynb",
        ):
            data = json.loads(Path(path).read_text())
            code = "\n".join("".join(cell.get("source", [])) for cell in data["cells"])
            self.assertIn("val_clean_pre", code)
            self.assertIn("val_noise_pre", code)
            self.assertIn("total_samples", code)
            self.assertIn("epsilon=0.05", code)

    def test_checkpoint_selection_last_best_and_zero_tie(self):
        rows = [
            {"epoch": 1, "val_f1_macro": 0.0},
            {"epoch": 2, "val_f1_macro": 0.0},
            {"epoch": 3, "val_f1_macro": 0.4},
            {"epoch": 4, "val_f1_macro": 0.4},
        ]
        self.assertEqual(select_checkpoint_epoch(rows, "last"), 4)
        self.assertEqual(select_checkpoint_epoch(rows[:2], "best_val_macro_f1"), 1)
        self.assertEqual(select_checkpoint_epoch(rows, "best_val_macro_f1"), 3)


class TestC05Resume(unittest.TestCase):
    def _trainer(self):
        config = ControlledExperimentConfig(num_classes=3, amp=False, max_steps=4, warmup_steps=0)
        return ControlledTrainer(TinyClassifier(), config, torch.device("cpu"))

    def test_checkpoint_contains_full_versioned_state_and_rng(self):
        with tempfile.TemporaryDirectory() as tmp:
            trainer = self._trainer()
            scheduler = torch.optim.lr_scheduler.StepLR(trainer.optimizer, step_size=1)
            noise_generator = torch.Generator().manual_seed(42)
            path = trainer.save_epoch_checkpoint(
                1, {"val_f1_macro": 0.25}, Path(tmp) / "run.pt", scheduler,
                [{"epoch": 1, "val_f1_macro": 0.25}], {"data_sha256": "d"}, noise_generator,
                checkpoint_role="last",
            )
            saved = torch.load(path, map_location="cpu", weights_only=False)
            self.assertEqual(saved["checkpoint_schema"], "c05_finetune_checkpoint_v1")
            self.assertTrue({"model_state_dict", "optimizer_state_dict", "scheduler_state_dict", "scaler_state_dict", "rng_state", "training_history"}.issubset(saved))
            self.assertEqual(set(("python", "numpy", "torch_cpu", "torch_cuda", "noise_generator")) - set(saved["rng_state"]), set())
            self.assertEqual(saved["run_identity"], {"data_sha256": "d"})
            self.assertEqual(saved["checkpoint_role"], "last")
            self.assertEqual(saved["steps_attempted"], trainer.steps_attempted)
            self.assertEqual(saved["steps_effective"], trainer.steps_effective)

    def test_resume_validates_identity_and_restores_required_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            trainer1 = self._trainer()
            scheduler1 = torch.optim.lr_scheduler.StepLR(trainer1.optimizer, step_size=1)
            noise1 = torch.Generator().manual_seed(77)
            trainer1.train_epoch(DataLoader(TensorDataset(torch.eye(4)[:3], torch.tensor([0, 1, 2])), batch_size=3))
            scheduler1.step()
            trainer1.best_metric_value = 0.6
            model_state = {key: value.clone() for key, value in trainer1.model.state_dict().items()}
            optimizer_state = trainer1.optimizer.state_dict()
            scheduler_state = scheduler1.state_dict()
            identity = {"variant": "resnet_clean_species", "data_sha256": "data"}
            path = trainer1.save_epoch_checkpoint(
                3, {}, Path(tmp) / "resume.pt", scheduler1,
                [{"epoch": 1}, {"epoch": 2}, {"epoch": 3}], identity, noise1,
                checkpoint_role="last",
            )
            trainer2 = self._trainer()
            scheduler2 = torch.optim.lr_scheduler.StepLR(trainer2.optimizer, step_size=1)
            noise2 = torch.Generator().manual_seed(1)
            next_epoch, history = trainer2.load_resume_checkpoint(
                path, scheduler2, expected_identity=identity,
                noise_generator=noise2, require_complete_state=True,
            )
            self.assertEqual(next_epoch, 4)
            self.assertEqual(len(history), 3)
            self.assertEqual(trainer2.best_metric_value, 0.6)
            self.assertTrue(torch.equal(noise1.get_state(), noise2.get_state()))
            for key, value in model_state.items():
                self.assertTrue(torch.equal(value, trainer2.model.state_dict()[key]))
            restored_optimizer = trainer2.optimizer.state_dict()
            self.assertEqual(restored_optimizer["param_groups"], optimizer_state["param_groups"])
            for param_id, state in optimizer_state["state"].items():
                for name, value in state.items():
                    self.assertTrue(torch.equal(value, restored_optimizer["state"][param_id][name]))
            self.assertEqual(scheduler2.state_dict(), scheduler_state)
            self.assertEqual(trainer2.scaler.state_dict(), trainer1.scaler.state_dict())

    def test_resume_rejects_incompatible_or_missing_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            trainer = self._trainer()
            path = trainer.save_epoch_checkpoint(1, {}, Path(tmp) / "resume.pt", run_identity={"x": 1})
            target = self._trainer()
            with self.assertRaisesRegex(ValueError, "identity mismatch"):
                target.load_resume_checkpoint(path, expected_identity={"x": 2}, require_complete_state=True)
            corrupt = torch.load(path, map_location="cpu", weights_only=False)
            del corrupt["optimizer_state_dict"]
            torch.save(corrupt, Path(tmp) / "missing.pt")
            with self.assertRaisesRegex(ValueError, "missing required state"):
                target.load_resume_checkpoint(Path(tmp) / "missing.pt", expected_identity={"x": 1}, require_complete_state=True)

    def test_atomic_checkpoint_preserves_previous_file_on_replace_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            trainer = self._trainer()
            destination = Path(tmp) / "last.pt"
            destination.write_bytes(b"previous-complete-checkpoint")
            with patch("viral_cnn.training.trainer.os.replace", side_effect=OSError("simulated interruption")):
                with self.assertRaisesRegex(OSError, "simulated interruption"):
                    trainer.save_epoch_checkpoint(1, {}, destination)
            self.assertEqual(destination.read_bytes(), b"previous-complete-checkpoint")


class TestC05Notebooks(unittest.TestCase):
    def test_notebooks_isolate_runs_and_require_explicit_resume(self):
        for path in (
            "notebooks/reproduction/short/03_finetune_resnet50_species.ipynb",
            "notebooks/reproduction/short/04_finetune_convnext_species.ipynb",
        ):
            data = json.loads(Path(path).read_text())
            code = "\n".join("".join(cell.get("source", [])) for cell in data["cells"])
            self.assertIn("reserve_run_directory", code)
            self.assertIn("RESUME_CHECKPOINT", code)
            self.assertIn("sha256", code)
            self.assertIn("checkpoint_path=None if ctx.is_smoke_test else base_checkpoint", code)
            self.assertNotIn("if last_ckpt.exists()", code)


class TestC05Report(unittest.TestCase):
    def test_generated_report_never_invents_controlled_results(self):
        with tempfile.TemporaryDirectory() as tmp:
            destination = Path(tmp) / "controlled_architecture_experiment.md"
            generate_controlled_experiment_report(Path(tmp) / "runs", destination)
            report = destination.read_text()
            self.assertIn("NOT_RUN", report)
            self.assertIn("NOT_MEASURED", report)
            self.assertIn("resnet50_1d", report)
            self.assertIn("convnext_tiny_1d", report)
            self.assertIn("Macro-F1", report)
            self.assertNotIn("84.88", report)
            self.assertNotIn("17,650", report)

    def test_report_marks_optional_comparison_not_run_and_limits_claims(self):
        report = Path("docs/controlled_architecture_experiment.md").read_text().lower()
        self.assertIn("not_run", report)
        self.assertIn("conclusão causal indisponível", report)
        self.assertIn("uma execução por configuração", report)

    def test_report_separates_test_evaluation_from_selection_and_architecture_claims(self):
        report = Path("docs/controlled_architecture_experiment.md").read_text().lower()
        self.assertIn("split de teste identificado", report)
        self.assertIn("não seleciona checkpoints pelo teste", report)
        self.assertIn("origens distintas", report)
        self.assertIn("não atribui diferenças exclusivamente à arquitetura", report)
        for notebook in (
            "notebooks/reproduction/short/03_finetune_resnet50_species.ipynb",
            "notebooks/reproduction/short/04_finetune_convnext_species.ipynb",
        ):
            data = json.loads(Path(notebook).read_text())
            code = "\n".join("".join(cell.get("source", [])) for cell in data["cells"])
            self.assertLess(code.index("selected_payload = torch.load"), code.index("# Test rows and loader are instantiated only after selection is frozen."))
            self.assertLess(code.index("selected_payload = torch.load"), code.index("test_ds = ParquetBatchIterableDataset"))
            self.assertIn("test_loader", code)
            self.assertIn("selection_uses_test", code)


if __name__ == "__main__":
    unittest.main()
