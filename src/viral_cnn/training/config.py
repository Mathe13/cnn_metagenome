"""
Configuration and parameter parity enforcement for controlled architecture experiments.
"""

from dataclasses import dataclass, field
from pathlib import Path
import random
from typing import Any, Literal, Sequence

import numpy as np
import torch
import yaml

SUPPORTED_ARCHITECTURES = ("resnet50_1d", "convnext_tiny_1d")
VALID_CHECKPOINT_METRICS = ("best_val_loss", "best_val_macro_f1", "last_step")


def seed_everything(seed: int) -> None:
    """
    Deterministically seeds Python random, NumPy, PyTorch CPU, and PyTorch CUDA generators.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


@dataclass
class ControlledExperimentConfig:
    """
    Declarative configuration enforcing parameter parity across architectures.
    """

    architectures: Sequence[str] = ("resnet50_1d", "convnext_tiny_1d")
    num_classes: int = 13683
    window_size: int = 512
    channels: Sequence[str] = ("A", "C", "G", "T")
    max_steps: int = 200
    batch_size: int = 32
    learning_rate: float = 5e-5
    weight_decay: float = 1e-4
    warmup_steps: int = 20
    label_smoothing: float = 0.1
    dropout_rate: float = 0.5
    noise_rate: float = 0.0
    seeds: Sequence[int] = (42,)
    checkpoint_metric: str = "best_val_macro_f1"
    eval_interval_steps: int = 50
    amp: bool = True
    grad_clip_norm: float | None = 1.0
    freeze_layers: Sequence[str] = ()
    mode: Literal["historical_reproduction", "controlled_parity"] = "controlled_parity"
    scheduler_step_unit: Literal["epoch", "batch"] = "epoch"

    def __post_init__(self):
        # Normalize sequences to tuples
        self.architectures = tuple(self.architectures)
        self.channels = tuple(self.channels)
        self.seeds = tuple(self.seeds)
        self.freeze_layers = tuple(self.freeze_layers)

        if self.mode not in ("historical_reproduction", "controlled_parity"):
            raise ValueError(f"Unsupported mode '{self.mode}'. Expected 'historical_reproduction' or 'controlled_parity'")

        if self.scheduler_step_unit not in ("epoch", "batch"):
            raise ValueError(f"Unsupported scheduler_step_unit '{self.scheduler_step_unit}'. Expected 'epoch' or 'batch'")

        # Validate architectures
        for arch in self.architectures:
            if arch not in SUPPORTED_ARCHITECTURES:
                raise ValueError(
                    f"Unsupported architecture '{arch}'. Supported: {SUPPORTED_ARCHITECTURES}"
                )

        if self.max_steps <= 0:
            raise ValueError(f"max_steps must be > 0, got {self.max_steps}")

        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be > 0, got {self.batch_size}")

        if not (0.0 <= self.label_smoothing < 1.0):
            raise ValueError(
                f"label_smoothing must be in [0.0, 1.0), got {self.label_smoothing}"
            )

        if not (0.0 <= self.noise_rate <= 1.0):
            raise ValueError(f"noise_rate must be in [0.0, 1.0], got {self.noise_rate}")

        if not (0.0 <= self.dropout_rate < 1.0):
            raise ValueError(
                f"dropout_rate must be in [0.0, 1.0), got {self.dropout_rate}"
            )

        if self.checkpoint_metric not in VALID_CHECKPOINT_METRICS:
            raise ValueError(
                f"checkpoint_metric '{self.checkpoint_metric}' not in {VALID_CHECKPOINT_METRICS}"
            )

        if self.warmup_steps < 0:
            raise ValueError(f"warmup_steps must be >= 0, got {self.warmup_steps}")
        if self.warmup_steps > self.max_steps:
            self.warmup_steps = self.max_steps


def load_controlled_config(config_path: str | Path) -> ControlledExperimentConfig:
    """
    Parses a declarative YAML configuration file into a validated ControlledExperimentConfig.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data: dict[str, Any] = yaml.safe_load(f)

    # Extract controlled section or root
    params = data.get("controlled_experiment", data)

    # Cast to schema
    return ControlledExperimentConfig(
        architectures=params.get(
            "architectures", ("resnet50_1d", "convnext_tiny_1d")
        ),
        num_classes=int(params.get("num_classes", 13683)),
        window_size=int(params.get("window_size", 512)),
        channels=params.get("channels", ("A", "C", "G", "T")),
        max_steps=int(params.get("max_steps", 200)),
        batch_size=int(params.get("batch_size", 32)),
        learning_rate=float(params.get("learning_rate", 5e-5)),
        weight_decay=float(params.get("weight_decay", 1e-4)),
        warmup_steps=int(params.get("warmup_steps", 20)),
        label_smoothing=float(params.get("label_smoothing", 0.1)),
        dropout_rate=float(params.get("dropout_rate", 0.5)),
        noise_rate=float(params.get("noise_rate", 0.0)),
        seeds=params.get("seeds", (42,)),
        checkpoint_metric=params.get("checkpoint_metric", "best_val_macro_f1"),
        eval_interval_steps=int(params.get("eval_interval_steps", 50)),
        amp=bool(params.get("amp", True)),
        grad_clip_norm=(
            float(params["grad_clip_norm"])
            if params.get("grad_clip_norm") is not None
            else None
        ),
        mode=params.get("mode", "historical_reproduction"),
        scheduler_step_unit=params.get("scheduler_step_unit", "epoch"),
    )
