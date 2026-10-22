#!/usr/bin/env python3
"""
CLI entrypoint for long-sequence sliding-window consensus inference (Long Voter).
"""

import argparse
import os
import sys
from pathlib import Path
import pandas as pd
import torch

# Ensure src/ is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

try:
    from viral_cnn.models.builder import build_model
    from viral_cnn.models.registry import list_model_variants, get_model_spec
    from viral_cnn.inference.voter import (
        LongSequenceVoter,
        DEFAULT_WINDOW_SIZE,
        DEFAULT_STRIDE,
        DEFAULT_BATCH_SIZE,
    )
except ImportError as e:
    sys.stderr.write(f"ImportError: Failed to import viral_cnn modules: {e}\n")
    sys.exit(2)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Run sliding-window consensus inference (Long Voter) on variable-length viral sequences."
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to input CSV or Parquet file containing sequences.",
    )
    parser.add_argument(
        "--model-variant",
        type=str,
        required=True,
        help=f"Registered model variant name. Supported: {list_model_variants()}",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to output Parquet or CSV file for prediction records.",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=DEFAULT_WINDOW_SIZE,
        help=f"Sliding window length (default: {DEFAULT_WINDOW_SIZE}).",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=DEFAULT_STRIDE,
        help=f"Sliding window step/stride (default: {DEFAULT_STRIDE}).",
    )
    parser.add_argument(
        "--handle-short",
        type=str,
        choices=["pad", "skip"],
        default="pad",
        help="Handling of sequences shorter than window-size: 'pad' with null vector or 'skip'.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Batch size for window forward passes (default: {DEFAULT_BATCH_SIZE}).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of sequences processed.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Inference device: 'cpu' or 'cuda' (default: 'cpu').",
    )
    return parser.parse_args(argv)


def main():
    try:
        args = parse_args()
    except SystemExit as e:
        sys.exit(2 if e.code != 0 else 0)

    # 1. Validation of inputs
    if not os.path.exists(args.input):
        sys.stderr.write(f"Error: Input file '{args.input}' does not exist.\n")
        sys.exit(2)

    supported_variants = list_model_variants()
    if args.model_variant not in supported_variants:
        sys.stderr.write(
            f"Error: Unsupported model variant '{args.model_variant}'. Supported: {supported_variants}\n"
        )
        sys.exit(2)

    # 2. Load model
    spec = get_model_spec(args.model_variant)
    model = build_model(
        model_type=spec.architecture,
        num_classes=spec.num_classes,
        max_seq_len=512,
        dropout_rate=spec.dropout_rate,
    )
    model.eval()

    # 3. Instantiate Voter
    try:
        voter = LongSequenceVoter(
            model=model,
            window_size=args.window_size,
            stride=args.stride,
            handle_short=args.handle_short,
            batch_size_inference=args.batch_size,
            device=args.device,
            expected_classes=spec.num_classes,
        )
    except ValueError as e:
        sys.stderr.write(f"ValueError initializing voter: {e}\n")
        sys.exit(2)

    # 4. Load dataset
    if args.input.endswith(".parquet"):
        df = pd.read_parquet(args.input)
    elif args.input.endswith(".csv"):
        df = pd.read_csv(args.input)
    else:
        sys.stderr.write(f"Error: Unsupported file format '{args.input}'. Must be CSV or Parquet.\n")
        sys.exit(2)

    if "sequence" not in df.columns:
        sys.stderr.write("Error: Input table must contain a 'sequence' column.\n")
        sys.exit(2)

    if args.limit is not None and args.limit > 0:
        df = df.iloc[: args.limit]

    # Detect ground truth column
    label_col = None
    for cand in ["species_id", "label", "target", "tax_id"]:
        if cand in df.columns:
            label_col = cand
            break

    # 5. Execute Voter inference
    records = []
    print(f"Running Long Voter on {len(df)} sequences using {args.model_variant}...")

    for idx, row in df.iterrows():
        seq = str(row["sequence"])
        res = voter.predict_sequence(seq)

        if res is None:
            continue

        truth_val = int(row[label_col]) if label_col is not None else None
        # Model predictions are 0-based index [0, num_classes-1].
        # 1-based tax ID offset is model_idx + 1 (AD-002).
        rec = {
            "query_id": row.get("fragment_id", row.get("id", idx)),
            "sequence_length": len(seq),
            "num_windows": res.num_windows,
            "pred_mean_logits": res.pred_mean_logits,
            "pred_majority_vote": res.pred_majority_vote,
            "pred_mean_tax_id": res.pred_mean_logits + 1,
            "pred_majority_tax_id": res.pred_majority_vote + 1,
            "truth_label": truth_val,
        }
        records.append(rec)

    df_out = pd.DataFrame(records)

    # 6. Compute summary metrics if ground truth is present
    if label_col is not None and len(df_out) > 0:
        # Check if truth is 1-based (tax_id) or 0-based
        if df_out["truth_label"].min() >= 1:
            mean_acc = (df_out["pred_mean_tax_id"] == df_out["truth_label"]).mean()
            maj_acc = (df_out["pred_majority_tax_id"] == df_out["truth_label"]).mean()
        else:
            mean_acc = (df_out["pred_mean_logits"] == df_out["truth_label"]).mean()
            maj_acc = (df_out["pred_majority_vote"] == df_out["truth_label"]).mean()

        print(f"Long Sequence Consensus Accuracy (Mean Logits):    {mean_acc * 100:.2f}%")
        print(f"Long Sequence Consensus Accuracy (Majority Vote):  {maj_acc * 100:.2f}%")

    # 7. Save output
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if str(out_path).endswith(".parquet"):
            try:
                df_out.to_parquet(out_path, index=False)
                print(f"Predictions saved to {args.output}")
            except (ImportError, ValueError):
                # Fallback to CSV if parquet engine is missing
                csv_path = out_path.with_suffix(".csv")
                df_out.to_csv(csv_path, index=False)
                print(f"Parquet engine missing; saved predictions to {csv_path}")
        else:
            df_out.to_csv(out_path, index=False)
            print(f"Predictions saved to {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
