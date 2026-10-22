#!/usr/bin/env python3
"""
CLI entrypoint for clean and controlled noise evaluation of 1D-CNN viral models.
"""

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Optional

# Ensure src/ is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import pandas as pd
import torch

try:
    from viral_cnn.models.registry import (
        list_model_variants,
        get_model_spec,
        build_registered_model,
    )
    from viral_cnn.eval.runner import evaluate_model, save_evaluation_artifacts
except ImportError as e:
    sys.stderr.write(f"ImportError: Failed to import viral_cnn modules: {e}\n")
    sys.exit(2)

SUPPORTED_NOISE_RATES = [0.0, 0.02, 0.05, 0.10, 0.15]


def load_input_data(
    input_path: Path,
    seq_col: Optional[str] = None,
    label_col: Optional[str] = None,
    id_col: Optional[str] = None,
) -> tuple[list[str], list[int], list[str]]:
    """Loads sequences, labels, and query IDs from CSV or Parquet file."""
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
        description="Run clean or controlled noise evaluation on viral 1D-CNN models."
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to evaluation dataset file (.csv or .parquet).",
    )
    parser.add_argument(
        "--model-variant",
        type=str,
        required=True,
        help=f"Registered model variant name. Supported: {list_model_variants()}",
    )
    parser.add_argument(
        "--noise",
        type=float,
        default=0.0,
        help=f"Nominal Bernoulli noise rate. Supported: {SUPPORTED_NOISE_RATES} (default: 0.0)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to save evaluation artifacts (metrics, predictions, manifest).",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Optional path to model weights checkpoint (.pt file).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Evaluation batch size (default: 256).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Inference device: 'cpu' or 'cuda' (default: 'cpu').",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for noise generation and model initialization (default: 42).",
    )
    parser.add_argument("--seq-col", type=str, default=None, help="Sequence column name.")
    parser.add_argument("--label-col", type=str, default=None, help="Label column name.")
    parser.add_argument("--id-col", type=str, default=None, help="Query ID column name.")

    args = parser.parse_args()

    # 1. Validate noise rate
    matched_noise = None
    for r in SUPPORTED_NOISE_RATES:
        if abs(args.noise - r) < 1e-5:
            matched_noise = r
            break
    if matched_noise is None:
        sys.stderr.write(
            f"Error: Unsupported noise rate {args.noise}. Supported nominal rates: {SUPPORTED_NOISE_RATES}\n"
        )
        sys.exit(2)

    # 2. Validate input file
    input_path = Path(args.input)
    if not input_path.exists():
        sys.stderr.write(f"Error: Input file not found: {input_path}\n")
        sys.exit(2)

    # 3. Validate model variant
    try:
        spec = get_model_spec(args.model_variant)
    except ValueError as e:
        sys.stderr.write(f"Error: {e}\n")
        sys.exit(2)

    # Load dataset
    try:
        seqs, labels, ids = load_input_data(
            input_path=input_path,
            seq_col=args.seq_col,
            label_col=args.label_col,
            id_col=args.id_col,
        )
    except Exception as e:
        sys.stderr.write(f"Error loading input dataset: {e}\n")
        sys.exit(2)

    # Device selection
    device_str = args.device
    if device_str == "cuda" and not torch.cuda.is_available():
        sys.stderr.write("Warning: CUDA requested but not available. Falling back to CPU.\n")
        device_str = "cpu"

    # Instantiate model
    torch.manual_seed(args.seed)
    model = build_registered_model(
        variant_name=args.model_variant,
        checkpoint_path=args.checkpoint,
        device=device_str,
    )

    # Execute evaluation
    result = evaluate_model(
        model=model,
        sequences=seqs,
        labels=labels,
        query_ids=ids,
        noise_rate=matched_noise,
        model_name=args.model_variant,
        device=device_str,
        batch_size=args.batch_size,
        num_classes=spec.num_classes,
        seed=args.seed,
    )

    # Save artifacts
    saved_files = save_evaluation_artifacts(result=result, output_dir=args.output_dir)

    # Output summary to stdout
    print(json.dumps(result.summary_dict(), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
