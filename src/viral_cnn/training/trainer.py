"""
Unified training loop and evaluation for controlled architecture experiments.
"""

from pathlib import Path
from typing import Any, Iterator, Optional, Sequence
import math
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from viral_cnn.training.config import ControlledExperimentConfig
from viral_cnn.training.profiler import ResourceProfiler, ResourceProfile
from viral_cnn.eval.metrics import compute_f1_scores
from viral_cnn.eval.noise import inject_noise


class ControlledTrainer:
    """
    Executes controlled optimization steps with strict parameter parity, AMP, and step telemetry.
    Supports both step-based training and epoch-level fine-tuning with noise corruption and layer freezing.
    """

    def __init__(
        self,
        model: nn.Module,
        config: ControlledExperimentConfig,
        device: torch.device,
        freeze_layers: Optional[Sequence[str]] = None,
    ):
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.current_step = 0

        # Layer freezing policy
        if freeze_layers is not None:
            self.freeze_layers = tuple(freeze_layers)
        else:
            self.freeze_layers = tuple(getattr(config, "freeze_layers", ()))

        self._frozen_bn_modules: list[nn.Module] = []
        if self.freeze_layers:
            for name, param in self.model.named_parameters():
                if any(name.startswith(prefix) for prefix in self.freeze_layers):
                    param.requires_grad = False
            for name, module in self.model.named_modules():
                if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                    if any(name.startswith(prefix) for prefix in self.freeze_layers):
                        self._frozen_bn_modules.append(module)

        # Optimizer: AdamW on trainable parameters
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        initial_lr = self._compute_scheduled_lr(0)
        self.optimizer = torch.optim.AdamW(
            trainable_params if trainable_params else self.model.parameters(),
            lr=initial_lr,
            weight_decay=self.config.weight_decay,
        )

        # Loss function with label smoothing
        self.criterion = nn.CrossEntropyLoss(
            label_smoothing=self.config.label_smoothing
        )

        # Mixed precision setup
        self.use_amp = self.config.amp and (self.device.type in ("cuda", "cpu"))
        scaler_enabled = self.config.amp and (self.device.type == "cuda" and torch.cuda.is_available())
        self.scaler = torch.amp.GradScaler("cuda", enabled=scaler_enabled)

        # Checkpoint tracking
        self.best_metric_value: Optional[float] = None
        self.profiler = ResourceProfiler(device=self.device)

    def _apply_bn_eval(self) -> None:
        """
        Ensures that BatchNorm layers associated with frozen modules stay in eval mode during training.
        """
        for bn in self._frozen_bn_modules:
            bn.eval()

    def _compute_scheduled_lr(self, step: int) -> float:
        """
        Calculates learning rate with linear warmup and cosine annealing down to 1e-7.
        """
        warmup = self.config.warmup_steps
        max_steps = self.config.max_steps
        base_lr = self.config.learning_rate
        min_lr = 1e-7

        if step < warmup and warmup > 0:
            return base_lr * float(step + 1) / float(warmup)

        # Cosine decay phase
        progress = float(step - warmup) / float(max(1, max_steps - warmup))
        progress = min(max(progress, 0.0), 1.0)
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr + (base_lr - min_lr) * cosine_decay

    def get_current_lr(self) -> float:
        """
        Returns the learning rate of the first parameter group.
        """
        for param_group in self.optimizer.param_groups:
            return float(param_group["lr"])
        return float(self.config.learning_rate)

    def train_step(self, batch_x: torch.Tensor, batch_y: torch.Tensor) -> dict[str, float]:
        """
        Executes a single forward, backward, and optimization step.
        """
        self.model.train()
        self._apply_bn_eval()

        batch_x = batch_x.to(self.device)
        batch_y = batch_y.to(self.device)

        # Update learning rate per schedule
        lr = self._compute_scheduled_lr(self.current_step)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

        self.optimizer.zero_grad()

        with self.profiler.profile_step(batch_size=batch_x.size(0)):
            # Forward pass under autocast
            autocast_device = "cuda" if self.device.type == "cuda" and torch.cuda.is_available() else "cpu"
            with torch.autocast(device_type=autocast_device, enabled=self.use_amp):
                logits = self.model(batch_x)
                loss = self.criterion(logits, batch_y)

            # Backward pass with scaler
            if self.scaler.is_enabled():
                self.scaler.scale(loss).backward()
                if self.config.grad_clip_norm is not None:
                    self.scaler.unscale_(self.optimizer)
                    grad_norm = nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.grad_clip_norm
                    )
                else:
                    grad_norm = 0.0
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                if self.config.grad_clip_norm is not None:
                    grad_norm = nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.grad_clip_norm
                    )
                else:
                    grad_norm = 0.0
                self.optimizer.step()

        self.current_step += 1

        grad_val = float(grad_norm.item() if hasattr(grad_norm, "item") else grad_norm)
        return {
            "step": self.current_step,
            "loss": float(loss.item()),
            "learning_rate": lr,
            "grad_norm": grad_val,
        }

    def train_epoch(
        self,
        train_loader: DataLoader,
        max_batches: Optional[int] = None,
        generator: Optional[torch.Generator] = None,
        scheduler: Optional[Any] = None,
    ) -> dict[str, float]:
        """
        Executes a full training epoch with optional noise corruption and metric accumulation.
        """
        self.model.train()
        self._apply_bn_eval()

        running_loss = 0.0
        total_samples = 0
        correct_top1 = 0

        all_preds: list[np.ndarray] = []
        all_targets: list[np.ndarray] = []

        batch_count = 0
        autocast_device = "cuda" if self.device.type == "cuda" and torch.cuda.is_available() else "cpu"

        for batch_x, batch_y in train_loader:
            if max_batches is not None and batch_count >= max_batches:
                break

            # Apply noise perturbation if configured
            if self.config.noise_rate > 0.0:
                batch_x = inject_noise(
                    batch_x,
                    noise_rate=self.config.noise_rate,
                    generator=generator,
                )

            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)
            b_size = batch_x.size(0)

            self.optimizer.zero_grad()

            with torch.autocast(device_type=autocast_device, enabled=self.use_amp):
                logits = self.model(batch_x)
                loss = self.criterion(logits, batch_y)

            if self.scaler.is_enabled():
                self.scaler.scale(loss).backward()
                if self.config.grad_clip_norm is not None:
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.grad_clip_norm
                    )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                if self.config.grad_clip_norm is not None:
                    nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.grad_clip_norm
                    )
                self.optimizer.step()

            if scheduler is not None:
                scheduler.step()

            running_loss += float(loss.item()) * b_size
            total_samples += b_size

            preds = torch.argmax(logits, dim=1)
            correct_top1 += int((preds == batch_y).sum().item())

            all_preds.append(preds.detach().cpu().numpy())
            all_targets.append(batch_y.detach().cpu().numpy())

            batch_count += 1
            self.current_step += 1

        if total_samples == 0:
            return {
                "train_loss": 0.0,
                "train_acc": 0.0,
                "train_f1": 0.0,
                "train_f1_weighted": 0.0,
                "total_samples": 0,
                "learning_rate": self.get_current_lr(),
            }

        y_pred_arr = np.concatenate(all_preds)
        y_true_arr = np.concatenate(all_targets)

        _, macro_f1, weighted_f1 = compute_f1_scores(
            y_true=y_true_arr,
            y_pred=y_pred_arr,
            num_classes=self.config.num_classes,
        )

        return {
            "train_loss": float(running_loss / total_samples),
            "train_acc": float(correct_top1 / total_samples),
            "train_f1": float(macro_f1),
            "train_f1_weighted": float(weighted_f1),
            "total_samples": total_samples,
            "learning_rate": self.get_current_lr(),
        }

    def evaluate_epoch(
        self,
        val_loader: DataLoader,
        max_batches: Optional[int] = None,
        generator: Optional[torch.Generator] = None,
    ) -> dict[str, float]:
        """
        Evaluates model over validation dataloader with optional noise corruption.
        """
        self.model.eval()
        running_loss = 0.0
        total_samples = 0
        correct_top1 = 0
        correct_top5 = 0

        all_preds: list[np.ndarray] = []
        all_targets: list[np.ndarray] = []

        batch_count = 0
        autocast_device = "cuda" if self.device.type == "cuda" and torch.cuda.is_available() else "cpu"

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                if max_batches is not None and batch_count >= max_batches:
                    break

                if self.config.noise_rate > 0.0:
                    batch_x = inject_noise(
                        batch_x,
                        noise_rate=self.config.noise_rate,
                        generator=generator,
                    )

                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                b_size = batch_x.size(0)

                with torch.autocast(device_type=autocast_device, enabled=self.use_amp):
                    logits = self.model(batch_x)
                    loss = self.criterion(logits, batch_y)

                running_loss += float(loss.item()) * b_size
                total_samples += b_size

                preds = torch.argmax(logits, dim=1)
                correct_top1 += int((preds == batch_y).sum().item())

                k = min(5, logits.size(1))
                if k > 1:
                    _, top5_indices = torch.topk(logits, k, dim=1)
                    target_expanded = batch_y.unsqueeze(1).expand_as(top5_indices)
                    correct_top5 += int((top5_indices == target_expanded).any(dim=1).sum().item())
                else:
                    correct_top5 = correct_top1

                all_preds.append(preds.cpu().numpy())
                all_targets.append(batch_y.cpu().numpy())
                batch_count += 1

        if total_samples == 0:
            return {
                "val_loss": 0.0,
                "val_acc": 0.0,
                "val_acc_top5": 0.0,
                "val_f1_macro": 0.0,
                "val_f1_weighted": 0.0,
                "total_samples": 0,
            }

        y_pred_arr = np.concatenate(all_preds)
        y_true_arr = np.concatenate(all_targets)

        _, macro_f1, weighted_f1 = compute_f1_scores(
            y_true=y_true_arr,
            y_pred=y_pred_arr,
            num_classes=self.config.num_classes,
        )

        return {
            "val_loss": float(running_loss / total_samples),
            "val_acc": float(correct_top1 / total_samples),
            "val_acc_top5": float(correct_top5 / total_samples),
            "val_f1_macro": float(macro_f1),
            "val_f1_weighted": float(weighted_f1),
            "total_samples": total_samples,
        }

    def evaluate(self, val_loader: DataLoader) -> dict[str, float]:
        """
        Computes Top-1 accuracy, Top-5 accuracy, and Macro-F1 across validation data.
        """
        metrics = self.evaluate_epoch(val_loader)
        return {
            "val_loss": metrics["val_loss"],
            "accuracy_top1": metrics["val_acc"],
            "accuracy_top5": metrics.get("val_acc_top5", metrics["val_acc"]),
            "macro_f1": metrics["val_f1_macro"],
            "n_samples": metrics["total_samples"],
        }

    def check_and_save_checkpoint(
        self, eval_metrics: dict[str, float], save_path: Path | str
    ) -> bool:
        """
        Evaluates checkpoint improvement and saves if metric improved or rule is last_step.
        """
        metric_name = self.config.checkpoint_metric
        is_best = False

        if metric_name == "last_step":
            is_best = True
        elif metric_name == "best_val_loss":
            current_val = eval_metrics.get("val_loss", float("inf"))
            if self.best_metric_value is None or current_val < self.best_metric_value:
                self.best_metric_value = current_val
                is_best = True
        elif metric_name == "best_val_macro_f1":
            current_val = eval_metrics.get("macro_f1", eval_metrics.get("val_f1_macro", 0.0))
            if self.best_metric_value is None or current_val > self.best_metric_value:
                self.best_metric_value = current_val
                is_best = True

        if is_best:
            path = Path(save_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "step": self.current_step,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "eval_metrics": eval_metrics,
                "checkpoint_metric": metric_name,
            }
            torch.save(payload, path)

        return is_best

    def save_epoch_checkpoint(
        self,
        epoch: int,
        metrics: dict[str, Any],
        save_path: Path | str,
        scheduler: Optional[Any] = None,
    ) -> Path:
        """
        Serializes full training state dictionary compatible with historical resume keys.
        """
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        sched_state = {}
        if scheduler is not None and hasattr(scheduler, "state_dict"):
            sched_state = scheduler.state_dict()

        scaler_state = {}
        if self.scaler is not None and hasattr(self.scaler, "state_dict"):
            scaler_state = self.scaler.state_dict()

        payload = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": sched_state,
            "scaler_state_dict": scaler_state,
            "metrics": metrics,
            "config": {
                "num_classes": self.config.num_classes,
                "noise_rate": self.config.noise_rate,
                "learning_rate": self.config.learning_rate,
                "weight_decay": self.config.weight_decay,
                "dropout_rate": self.config.dropout_rate,
                "label_smoothing": self.config.label_smoothing,
                "freeze_layers": list(self.freeze_layers),
            },
        }
        torch.save(payload, path)
        return path

    def resume_checkpoint(
        self,
        checkpoint_path: Path | str,
        scheduler: Optional[Any] = None,
    ) -> int:
        """
        Restores model weights, optimizer, scheduler, and scaler from checkpoint file.
        Returns the epoch index to resume from.
        """
        path = Path(checkpoint_path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found at '{path}'")

        checkpoint = torch.load(path, map_location=self.device)
        if not isinstance(checkpoint, dict):
            raise ValueError(f"Invalid checkpoint format in '{path}': expected dict, got {type(checkpoint)}")

        # Clean potential _orig_mod. prefix from torch.compile
        raw_state = checkpoint.get("model_state_dict", checkpoint)
        cleaned_state = {k.replace("_orig_mod.", ""): v for k, v in raw_state.items()}
        self.model.load_state_dict(cleaned_state)

        if "optimizer_state_dict" in checkpoint and self.optimizer is not None:
            try:
                self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            except Exception:
                pass

        if "scheduler_state_dict" in checkpoint and scheduler is not None and hasattr(scheduler, "load_state_dict"):
            try:
                scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            except Exception:
                pass

        if "scaler_state_dict" in checkpoint and self.scaler is not None and hasattr(self.scaler, "load_state_dict"):
            try:
                self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
            except Exception:
                pass

        start_epoch = int(checkpoint.get("epoch", 0))
        return start_epoch
