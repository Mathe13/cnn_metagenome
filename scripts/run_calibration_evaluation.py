#!/usr/bin/env python3
"""
CLI entrypoint for calibration and confidence evaluation of 1D-CNN viral models across noise levels.
"""

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Optional

# Ensure src/ is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

try:
    from viral_cnn.models.registry import (
        list_model_variants,
        get_model_spec,
        build_registered_model,
    )
    from viral_cnn.eval.calibration import (
        compute_calibration_metrics,
        compute_reliability_diagram,
        save_reliability_diagram,
        TemperatureScaler,
        audit_model_calibration_by_noise,
    )
    from viral_cnn.eval.noise import inject_noise
    from viral_cnn.data.encoding import encode_sequence_tokens
except ImportError as e:
    sys.stderr.write(f"ImportError: Failed to import viral_cnn modules: {e}\n")
    sys.exit(2)

SUPPORTED_NOISE_RATES = [0.0, 0.02, 0.05, 0.10, 0.15]
SUPPORTED_BINNING_STRATEGIES = ["equal_width", "equal_mass"]


def load_input_data(
    input_path: Path,
    seq_col: Optional[str] = None,
    label_col: Optional[str] = None,
    id_col: Optional[str] = None,
) -> tuple[list[str], list[int], list[str]]:
    """Loads sequences, labels, and query IDs from CSV or Parquet file."""
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    if input_path.suffix.lower() == ".parquet":
        df = pd.read_parquet(input_path)
    else:
        df = pd.read_csv(input_path)

    # Resolve sequence column
    sequence_col = seq_col
    if sequence_col is None:
        for candidate in ["sequence", "sequence_padded", "seq", "fragment"]:
            if candidate in df.columns:
                sequence_col = candidate
                break
        if sequence_col is None:
            raise ValueError(
                f"Cannot identify sequence column. Available columns: {list(df.columns)}"
            )

    # Resolve label column
    target_col = label_col
    if target_col is None:
        for candidate in ["label", "species_id", "genus_id", "taxid", "class"]:
            if candidate in df.columns:
                target_col = candidate
                break
        if target_col is None:
            raise ValueError(
                f"Cannot identify label column. Available columns: {list(df.columns)}"
            )

    # Resolve query ID column
    query_id_col = id_col
    if query_id_col is None:
        for candidate in ["query_id", "id", "frag_id", "accession"]:
            if candidate in df.columns:
                query_id_col = candidate
                break

    seqs = df[sequence_col].astype(str).tolist()
    labels = df[target_col].astype(int).tolist()
    ids = (
        df[query_id_col].astype(str).tolist()
        if query_id_col is not None
        else [f"q_{i:07d}" for i in range(len(seqs))]
    )

    return seqs, labels, ids


def main():
    parser = argparse.ArgumentParser(
        description="Run probabilistic calibration evaluation on viral 1D-CNN models across noise."
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to evaluation test dataset file (.csv or .parquet).",
    )
    parser.add_argument(
        "--val-input",
        type=str,
        default=None,
        help="Path to validation dataset file for temperature scaling (.csv or .parquet).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="resnet_clean_species",
        help="Registered model variant name.",
    )
    parser.add_argument(
        "--noise-rates",
        type=float,
        nargs="+",
        default=[0.0],
        help="List of nominal noise rates to evaluate (e.g. 0.0 0.02 0.05 0.10 0.15).",
    )
    parser.add_argument(
        "--n-bins",
        type=int,
        default=15,
        help="Number of confidence bins for ECE and reliability diagrams (default: 15).",
    )
    parser.add_argument(
        "--binning",
        type=str,
        default="equal_width",
        choices=["equal_width", "equal_mass"],
        help="Binning strategy: 'equal_width' or 'equal_mass'.",
    )
    parser.add_argument(
        "--apply-temperature-scaling",
        action="store_true",
        help="Whether to learn temperature scaling on validation set and evaluate on test set.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/calibration",
        help="Directory where calibration outputs and diagrams will be written.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for noise generation.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to run inference on ('cpu' or 'cuda').",
    )

    args = parser.parse_args()

    input_file = Path(args.input)
    if not input_file.exists():
        sys.stderr.write(f"Error: input file '{args.input}' does not exist.\n")
        sys.exit(2)

    if args.binning not in SUPPORTED_BINNING_STRATEGIES:
        sys.stderr.write(f"Error: unsupported binning '{args.binning}'.\n")
        sys.exit(2)

    for r in args.noise_rates:
        if r not in SUPPORTED_NOISE_RATES:
            sys.stderr.write(
                f"Error: noise rate {r} not in supported rates {SUPPORTED_NOISE_RATES}.\n"
            )
            sys.exit(2)

    if args.apply_temperature_scaling and args.val_input is None:
        sys.stderr.write(
            "Error: --apply-temperature-scaling requires --val-input validation dataset.\n"
        )
        sys.exit(2)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load data
    try:
        seqs, labels, ids = load_input_data(input_file)
    except Exception as e:
        sys.stderr.write(f"Error reading input dataset: {e}\n")
        sys.exit(2)

    # 2. Build model
    try:
        model = build_registered_model(args.model, device=args.device)
    except Exception as e:
        sys.stderr.write(f"Error building model variant '{args.model}': {e}\n")
        sys.exit(2)

    model.eval()
    model.to(args.device)

    # 3. Optional Temperature Scaling on validation
    scaler = None
    temperature_val = 1.0
    if args.apply_temperature_scaling:
        val_file = Path(args.val_input)
        if not val_file.exists():
            sys.stderr.write(f"Error: val_input file '{args.val_input}' not found.\n")
            sys.exit(2)
        val_seqs, val_labels, _ = load_input_data(val_file)

        # Batch forward pass on validation to get logits
        val_tokens = torch.tensor(
            [encode_sequence_tokens(s, max_len=512) for s in val_seqs],
            dtype=torch.long,
        )
        with torch.no_grad():
            val_logits = model(val_tokens.to(args.device)).cpu()

        scaler = TemperatureScaler()
        scaler.fit(val_logits, val_labels, split_name="validation")
        temperature_val = scaler.temperature
        print(f"Fitted Temperature on validation split: T = {temperature_val:.4f}")

    # 4. Evaluate across noise rates
    tokens_tensor = torch.tensor(
        [encode_sequence_tokens(s, max_len=512) for s in seqs],
        dtype=torch.long,
    )
    labels_tensor = torch.tensor(labels, dtype=torch.long)
    n_samples = len(seqs)

    summary_records = {}

    for rate in args.noise_rates:
        rate_key = f"noise_{rate}"
        gen = torch.Generator(device="cpu").manual_seed(args.seed)

        all_logits_list = []
        with torch.no_grad():
            for start_idx in range(0, n_samples, 256):
                end_idx = min(start_idx + 256, n_samples)
                batch_tokens = tokens_tensor[start_idx:end_idx]

                if rate > 0.0:
                    batch_tokens = inject_noise(
                        sequences=batch_tokens,
                        noise_rate=rate,
                        generator=gen,
                    )

                batch_tokens = batch_tokens.to(args.device)
                logits = model(batch_tokens)
                all_logits_list.append(logits.cpu())

        all_logits = torch.cat(all_logits_list, dim=0)
        uncal_probs = F.softmax(all_logits, dim=-1).numpy()

        uncal_metrics = compute_calibration_metrics(
            probs=uncal_probs,
            y_true=labels,
            n_bins=args.n_bins,
            binning_strategy=args.binning,
        )

        uncal_bins = compute_reliability_diagram(
            probs=uncal_probs,
            y_true=labels,
            n_bins=args.n_bins,
            binning_strategy=args.binning,
        )

        save_reliability_diagram(
            bins=uncal_bins,
            output_dir=output_dir,
            title=f"Reliability Diagram ({args.model}, noise={rate})",
            filename_prefix=f"reliability_{args.model}_{rate_key}",
        )

        rate_record = {
            "uncalibrated": uncal_metrics.to_dict(),
        }

        if scaler is not None:
            scaled_res = scaler.transform(all_logits.numpy(), labels, n_bins=args.n_bins)
            rate_record["calibrated"] = scaled_res["calibrated_metrics"]
            rate_record["temperature"] = scaled_res["temperature"]

            cal_bins = compute_reliability_diagram(
                probs=scaled_res["calibrated_probs"],
                y_true=labels,
                n_bins=args.n_bins,
                binning_strategy=args.binning,
            )
            save_reliability_diagram(
                bins=cal_bins,
                output_dir=output_dir,
                title=f"Calibrated Reliability Diagram ({args.model}, noise={rate}, T={temperature_val:.2f})",
                filename_prefix=f"reliability_calibrated_{args.model}_{rate_key}",
            )

        summary_records[rate_key] = rate_record

    # 5. Save calibration summary JSON
    summary_file = output_dir / "calibration_summary.json"
    summary_file.write_text(json.dumps(summary_records, indent=2), encoding="utf-8")

    # 6. Save execution manifest
    manifest_file = output_dir / "run_manifest.json"
    manifest = {
        "model_name": args.model,
        "input_dataset": str(input_file),
        "val_dataset": str(args.val_input) if args.val_input else None,
        "noise_rates": args.noise_rates,
        "n_bins": args.n_bins,
        "binning_strategy": args.binning,
        "seed": args.seed,
        "n_samples": n_samples,
        "temperature_fitted": temperature_val if scaler else None,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    manifest_file.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Calibration evaluation successfully completed. Artifacts saved to {output_dir}")
    sys.exit(0)


if __name__ == "__main__":
    main()
