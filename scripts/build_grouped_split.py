#!/usr/bin/env python3
"""
CLI entrypoint to generate genome-disjoint grouped splits and feasibility reports.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Add src to python path if not installed
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from viral_cnn.data.grouped_split import (
    build_genome_disjoint_split,
    load_records_from_file,
    save_partition_manifests,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build genome-disjoint train/val/test splits with zero accession leakage."
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to input dataset file (CSV or Parquet) containing sequence and genome IDs.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to save partition manifests and split summary.",
    )
    parser.add_argument(
        "--group-col",
        type=str,
        default="genome_id",
        help="Column identifying genome accession (default: genome_id).",
    )
    parser.add_argument(
        "--class-col",
        type=str,
        default="species_id",
        help="Column identifying taxonomic class (default: species_id).",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of genomes allocated to test partition (default: 0.2).",
    )
    parser.add_argument(
        "--val-size",
        type=float,
        default=0.0,
        help="Fraction of genomes allocated to validation partition (default: 0.0).",
    )
    parser.add_argument(
        "--min-genomes",
        type=int,
        default=2,
        help="Minimum genomes per class required for disjoint testing (default: 2).",
    )
    parser.add_argument(
        "--single-genome-policy",
        type=str,
        default="train_only",
        choices=["train_only", "exclude"],
        help="Policy for single-genome classes: train_only or exclude (default: train_only).",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for deterministic genome assignment (default: 42).",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress printing JSON summary to stdout.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if not os.path.isfile(args.input):
        sys.stderr.write(f"Error: Input file not found: {args.input}\n")
        return 2

    if not (0.0 < args.test_size < 1.0) or not (0.0 <= args.val_size < 1.0):
        sys.stderr.write("Error: test_size and val_size must be in (0, 1) and sum to < 1.0\n")
        return 2

    if args.test_size + args.val_size >= 1.0:
        sys.stderr.write("Error: test_size + val_size must be strictly < 1.0\n")
        return 2

    try:
        records = load_records_from_file(
            file_path=args.input,
            required_cols=[args.group_col, args.class_col],
        )

        split_result = build_genome_disjoint_split(
            records=records,
            group_col=args.group_col,
            class_col=args.class_col,
            test_size=args.test_size,
            val_size=args.val_size,
            min_genomes_per_class=args.min_genomes,
            single_genome_policy=args.single_genome_policy,
            random_state=args.random_state,
        )

        manifest_paths = save_partition_manifests(
            partitions=split_result["partitions"],
            output_dir=args.output_dir,
        )

        summary = {
            "metrics": split_result["metrics"],
            "manifest_paths": manifest_paths,
        }

        summary_json = json.dumps(summary, indent=2)

        summary_file = Path(args.output_dir) / "split_summary.json"
        with open(summary_file, mode="w", encoding="utf-8") as f:
            f.write(summary_json + "\n")

        if not args.quiet:
            print(summary_json)

        return 0

    except Exception as e:
        sys.stderr.write(f"Error during grouped split: {e}\n")
        return 2


if __name__ == "__main__":
    sys.exit(main())
