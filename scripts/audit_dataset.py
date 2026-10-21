#!/usr/bin/env python3
"""
CLI entrypoint to audit dataset duplicates, sequence overlap, and partition leakage.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Add src to python path if not installed
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from viral_cnn.data.audit import (
    audit_cross_partition_overlap,
    audit_partition_duplicates,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit dataset sequence duplicates and cross-partition leakage."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Path to single dataset file (CSV or Parquet) to audit internal duplicates.",
    )
    parser.add_argument(
        "--train",
        type=str,
        default=None,
        help="Path to train partition file for cross-partition overlap audit.",
    )
    parser.add_argument(
        "--test",
        type=str,
        default=None,
        help="Path to test partition file for cross-partition overlap audit.",
    )
    parser.add_argument(
        "--seq-col",
        type=str,
        default="sequence",
        help="Name of sequence column in dataset (default: sequence).",
    )
    parser.add_argument(
        "--label-col",
        type=str,
        default="label",
        help="Name of label column in dataset (default: label).",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Max number of samples to process per partition (for fast audit).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional path to write JSON audit report.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress printing JSON report to stdout.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if not args.dataset and not (args.train and args.test):
        sys.stderr.write("Error: Must provide either --dataset OR both --train and --test.\n")
        return 2

    report: dict[str, object] = {}

    try:
        if args.dataset:
            if not os.path.isfile(args.dataset):
                sys.stderr.write(f"Error: Dataset file not found: {args.dataset}\n")
                return 2
            report["partition_audit"] = audit_partition_duplicates(
                file_path=args.dataset,
                seq_col=args.seq_col,
                label_col=args.label_col,
                sample_size=args.sample_size,
            )

        if args.train and args.test:
            if not os.path.isfile(args.train):
                sys.stderr.write(f"Error: Train file not found: {args.train}\n")
                return 2
            if not os.path.isfile(args.test):
                sys.stderr.write(f"Error: Test file not found: {args.test}\n")
                return 2

            report["cross_partition_overlap"] = audit_cross_partition_overlap(
                train_path=args.train,
                test_path=args.test,
                seq_col=args.seq_col,
                label_col=args.label_col,
                sample_size=args.sample_size,
            )

        json_output = json.dumps(report, indent=2)

        if not args.quiet:
            print(json_output)

        if args.output:
            out_path = Path(args.output)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, mode="w", encoding="utf-8") as f:
                f.write(json_output + "\n")

        return 0

    except Exception as e:
        sys.stderr.write(f"Error during audit: {e}\n")
        return 2


if __name__ == "__main__":
    sys.exit(main())
