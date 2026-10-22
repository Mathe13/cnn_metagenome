"""
Unified training loop and evaluation for controlled architecture experiments.
"""

from pathlib import Path
from typing import Any, Iterator, Optional
import math
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from viral_cnn.training.config import ControlledExperimentConfig
from viral_cnn.training.profiler import ResourceProfiler, ResourceProfile
from viral_cnn.eval.metrics import compute_f1_scores


class ControlledTrainer:
    """
    Executes controlled optimization steps with strict parameter parity, AMP, and step telemetry.
    """

    def __init__(
        self,
        model: nn.Module,
        config: ControlledExperimentConfig,
        device: torch.device,
    ):
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.current_step = 0

        # Optimizer: AdamW
        initial_lr = self._compute_scheduled_lr(0)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=initial_lr,
            weight_decay=self.config.weight_decay,
        )

        # Loss function with label smoothing
        self.criterion = nn.CrossEntropyLoss(
            label_smoothing=self.config.label_smoothing
        )

        # Mixed precision setup
        self.use_amp = self.config.amp and (self.device.type in ("cuda", "cpu"))
        # GradScaler only operates on cuda in older PyTorch, or cuda/cpu in newer
        scaler_enabled = self.config.amp and (self.device.type == "cuda" and torch.cuda.is_available())
        self.scaler = torch.amp.GradScaler("cuda", enabled=scaler_enabled)

        # Checkpoint tracking
        self.best_metric_value: Optional[float] = None
        self.profiler = ResourceProfiler(device=self.device)

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

    def evaluate(self, val_loader: DataLoader) -> dict[str, float]:
        """
        Computes Top-1 accuracy, Top-5 accuracy, and Macro-F1 across validation data.
        """
        self.model.eval()
        total_loss = 0.0
        total_samples = 0
        correct_top1 = 0
        correct_top5 = 0

        all_preds: list[np.ndarray] = []
        all_targets: list[np.ndarray] = []

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                b_size = batch_x.size(0)

                logits = self.model(batch_x)
                loss = self.criterion(logits, batch_y)
                total_loss += float(loss.item()) * b_size
                total_samples += b_size

                # Top-1 accuracy
                preds = torch.argmax(logits, dim=1)
                correct_top1 += int((preds == batch_y).sum().item())

                # Top-5 accuracy
                k = min(5, logits.size(1))
                if k > 1:
                    _, top5_indices = torch.topk(logits, k, dim=1)
                    target_expanded = batch_y.unsqueeze(1).expand_as(top5_indices)
                    correct_top5 += int((top5_indices == target_expanded).any(dim=1).sum().item())
                else:
                    correct_top5 = correct_top1

                all_preds.append(preds.cpu().numpy())
                all_targets.append(batch_y.cpu().numpy())

        if total_samples == 0:
            return {
                "val_loss": 0.0,
                "accuracy_top1": 0.0,
                "accuracy_top5": 0.0,
                "macro_f1": 0.0,
                "n_samples": 0,
            }

        y_pred_arr = np.concatenate(all_preds)
        y_true_arr = np.concatenate(all_targets)

        _, macro_f1, _ = compute_f1_scores(
            y_true=y_true_arr,
            y_pred=y_pred_arr,
            num_classes=self.config.num_classes,
        )

        return {
            "val_loss": float(total_loss / total_samples),
            "accuracy_top1": float(correct_top1 / total_samples),
            "accuracy_top5": float(correct_top5 / total_samples),
            "macro_f1": float(macro_f1),
            "n_samples": total_samples,
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
            current_val = eval_metrics.get("macro_f1", 0.0)
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
