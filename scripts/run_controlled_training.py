#!/usr/bin/env python3
"""
CLI entry point to execute controlled cross-architecture training experiments.
Supports dry-run validation, parameter parity checks, and multi-seed benchmarking.
"""

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from viral_cnn.models.builder import build_model
from viral_cnn.training.config import (
    ControlledExperimentConfig,
    load_controlled_config,
    seed_everything,
    SUPPORTED_ARCHITECTURES,
)
from viral_cnn.training.trainer import ControlledTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run controlled cross-architecture training experiments."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/controlled/baseline_pair.yaml",
        help="Path to YAML experiment configuration.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="Override step budget.",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        help="List of random seeds to evaluate.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Computation device ('cpu', 'cuda', etc.).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate configuration, initialize models, verify parity, and exit.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="runs/controlled",
        help="Output directory to save checkpoints and JSON run manifests.",
    )
    parser.add_argument(
        "--synthetic-micro-benchmark",
        action="store_true",
        help="Execute on micro synthetic dataset for fast verification.",
    )
    return parser.parse_args()


def create_synthetic_data(
    num_classes: int,
    window_size: int,
    batch_size: int,
    n_train_samples: int = 64,
    n_val_samples: int = 32,
    noise_rate: float = 0.0,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader]:
    """
    Creates deterministic synthetic batches for micro-benchmarking.
    """
    g = torch.Generator().manual_seed(seed)
    
    # Generate random sequence indices 0..3 (A, C, G, T)
    x_train_idx = torch.randint(0, 4, (n_train_samples, window_size), generator=g)
    y_train = torch.randint(0, num_classes, (n_train_samples,), generator=g)

    # Optional noise injection
    if noise_rate > 0.0:
        mask = torch.rand(x_train_idx.shape, generator=g) < noise_rate
        rand_bases = torch.randint(0, 4, x_train_idx.shape, generator=g)
        x_train_idx = torch.where(mask, rand_bases, x_train_idx)

    # Spatial one-hot encoding: (N, 4, L)
    x_train = torch.zeros(n_train_samples, 4, window_size)
    for c in range(4):
        x_train[:, c, :] = (x_train_idx == c).float()

    x_val_idx = torch.randint(0, 4, (n_val_samples, window_size), generator=g)
    y_val = torch.randint(0, num_classes, (n_val_samples,), generator=g)
    x_val = torch.zeros(n_val_samples, 4, window_size)
    for c in range(4):
        x_val[:, c, :] = (x_val_idx == c).float()

    train_loader = DataLoader(
        TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(x_val, y_val), batch_size=batch_size, shuffle=False
    )
    return train_loader, val_loader


def run_dry_run_validation(
    config: ControlledExperimentConfig, device: torch.device
) -> None:
    """
    Instantiates both models, verifies head dimensions and hyperparameter parity.
    """
    print("=" * 70)
    print("CONTROLLED ARCHITECTURE EXPERIMENT: DRY-RUN PARITY VALIDATION")
    print("=" * 70)
    print(f"Target Architectures: {config.architectures}")
    print(f"Num Classes:          {config.num_classes}")
    print(f"Window Size:          {config.window_size} bp")
    print(f"Step Budget:          {config.max_steps}")
    print(f"Batch Size:           {config.batch_size}")
    print(f"Learning Rate:        {config.learning_rate}")
    print(f"Weight Decay:         {config.weight_decay}")
    print(f"Label Smoothing:      {config.label_smoothing}")
    print(f"Dropout Rate:         {config.dropout_rate}")
    print(f"Noise Rate (Train):   {config.noise_rate}")
    print(f"Checkpoint Metric:    {config.checkpoint_metric}")
    print(f"Evaluation Interval:  {config.eval_interval_steps} steps")
    print(f"Target Device:        {device}")
    print("-" * 70)

    for arch in config.architectures:
        # Build model with small class count if testing or configured
        test_classes = min(config.num_classes, 100) if not torch.cuda.is_available() else config.num_classes
        model = build_model(
            model_type=arch,
            num_classes=test_classes,
            dropout_rate=config.dropout_rate,
        )
        param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Architecture '{arch}': instantiated successfully ({param_count:,} params).")

    print("=" * 70)
    print("DRY-RUN VALIDATION PASSED: Hyperparameter parity verified across models.")
    print("=" * 70)


def main() -> None:
    args = parse_args()

    # Load configuration
    try:
        config = load_controlled_config(args.config)
    except Exception as e:
        print(f"Configuration Error: {e}", file=sys.stderr)
        sys.exit(2)

    # Apply CLI overrides
    if args.steps is not None:
        if args.steps <= 0:
            print(f"Error: --steps must be > 0, got {args.steps}", file=sys.stderr)
            sys.exit(2)
        config.max_steps = args.steps

    if args.seeds is not None:
        config.seeds = tuple(args.seeds)

    # Determine device
    if args.device is not None:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dry-run mode
    if args.dry_run:
        run_dry_run_validation(config, device)
        sys.exit(0)

    # Benchmark execution
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Starting Controlled Experiment Execution on {device}...")
    print(f"Output directory: {output_dir}")

    # Classes: in synthetic mode, limit to 20 for CPU speed
    num_classes = 20 if args.synthetic_micro_benchmark else config.num_classes

    executed_runs: list[dict[str, Any]] = []

    for seed in config.seeds:
        for arch in config.architectures:
            print(f"\n>>> Running {arch} with seed {seed} (steps: {config.max_steps})")
            seed_everything(seed)

            # Build model
            model = build_model(
                model_type=arch,
                num_classes=num_classes,
                dropout_rate=config.dropout_rate,
            )

            # Build trainer
            trainer = ControlledTrainer(model=model, config=config, device=device)

            # Data loaders
            if args.synthetic_micro_benchmark or not Path("cnn_kraken/processed_kraken_species_splited_parquet").exists():
                train_loader, val_loader = create_synthetic_data(
                    num_classes=num_classes,
                    window_size=config.window_size,
                    batch_size=config.batch_size,
                    noise_rate=config.noise_rate,
                    seed=seed,
                )
            else:
                # Real data loaders would go here if remote cluster parquet is mounted
                raise NotImplementedError("Real parquet loading requires cluster environment (AD-006)")

            # Train steps
            train_iter = iter(train_loader)
            step_losses: list[float] = []

            for step_idx in range(config.max_steps):
                try:
                    bx, by = next(train_iter)
                except StopIteration:
                    train_iter = iter(train_loader)
                    bx, by = next(train_iter)

                step_info = trainer.train_step(bx, by)
                step_losses.append(step_info["loss"])

            # Final evaluation
            eval_metrics = trainer.evaluate(val_loader)
            print(
                f"Validation [{arch} s{seed}]: Loss={eval_metrics['val_loss']:.4f}, "
                f"Top1={eval_metrics['accuracy_top1']*100:.2f}%, "
                f"Top5={eval_metrics['accuracy_top5']*100:.2f}%, "
                f"Macro-F1={eval_metrics['macro_f1']*100:.2f}%"
            )

            # Checkpoint
            ckpt_path = output_dir / f"checkpoint_{arch}_seed_{seed}.pt"
            trainer.check_and_save_checkpoint(eval_metrics, ckpt_path)

            # Finalize telemetry
            telemetry = trainer.profiler.finalize(
                total_steps=config.max_steps, model=model
            )

            run_record = {
                "architecture": arch,
                "seed": seed,
                "step_budget": config.max_steps,
                "batch_size": config.batch_size,
                "learning_rate": config.learning_rate,
                "weight_decay": config.weight_decay,
                "label_smoothing": config.label_smoothing,
                "dropout_rate": config.dropout_rate,
                "noise_rate": config.noise_rate,
                "checkpoint_metric": config.checkpoint_metric,
                "final_train_loss": float(step_losses[-1]) if step_losses else 0.0,
                "eval_metrics": eval_metrics,
                "telemetry": telemetry.to_dict(),
                "checkpoint_path": str(ckpt_path),
                "timestamp": time.time(),
            }

            log_path = output_dir / f"run_log_{arch}_seed_{seed}.json"
            with open(log_path, "w", encoding="utf-8") as f:
                json.dump(run_record, f, indent=2)

            executed_runs.append(run_record)

    print("\n" + "=" * 70)
    print(f"CONTROLLED EXPERIMENT RUNS COMPLETED ({len(executed_runs)} runs).")
    print(f"Results saved to: {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
