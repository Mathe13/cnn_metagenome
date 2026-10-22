#!/usr/bin/env python3
"""CLI runner for unified comparative evaluation of Kraken2, BLASTN, and CNN baselines."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import pandas as pd

# Ensure src/ is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from viral_cnn.baselines.taxonomy import parse_kraken_output, parse_blast_output
from viral_cnn.baselines.paired import (
    evaluate_predictions,
    paired_discordance,
    bootstrap_species_difference,
)
from viral_cnn.baselines.long_eval import (
    integrate_long_voter_predictions,
    evaluate_long_baselines,
)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Unified comparative benchmarking for metagenomic baselines."
    )
    parser.add_argument(
        "--regime",
        choices=["short", "long"],
        default="short",
        help="Evaluation regime: short (512 bp fragments) or long (variable-length reads).",
    )
    parser.add_argument(
        "--truth",
        required=True,
        type=Path,
        help="Path to CSV or Parquet file containing query ground truth (must have 'query_id' and 'label').",
    )
    parser.add_argument(
        "--predictions",
        nargs="+",
        required=True,
        metavar="NAME=PATH",
        help="One or more method predictions in NAME=PATH format (e.g. Kraken2=kraken.out BLASTN=blast.tsv).",
    )
    parser.add_argument(
        "--overlap-audit",
        type=Path,
        default=None,
        help="Optional path to exact overlap audit CSV (must have 'query_id' and 'has_overlap').",
    )
    parser.add_argument(
        "--genome-overlap-col",
        type=str,
        default="seen_genome",
        help="Column in truth file indicating presence of contig's reference genome in training (for long regime).",
    )
    parser.add_argument(
        "--length-col",
        type=str,
        default="length",
        help="Column in truth file indicating sequence length (for long regime).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmark_results"),
        help="Directory to save output CSVs and execution manifest.",
    )
    parser.add_argument(
        "--bootstrap-repeats",
        type=int,
        default=2000,
        help="Number of species-cluster bootstrap repetitions for paired difference CIs (default: 2000).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for bootstrap sampling (default: 42).",
    )
    return parser.parse_args(argv)


def load_truth_file(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Truth file not found: {path}")
    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)
    if "query_id" not in df.columns:
        if "id" in df.columns:
            df = df.rename(columns={"id": "query_id"})
        else:
            raise ValueError(f"Truth file must contain 'query_id' column: {path}")
    if "label" not in df.columns:
        raise ValueError(f"Truth file must contain 'label' column: {path}")
    df["query_id"] = df["query_id"].astype(str)
    df["label"] = df["label"].astype(int)
    return df


def load_method_predictions(
    name: str,
    path: Path,
    valid_query_ids: List[str],
) -> tuple[Dict[str, int], Optional[Dict[str, List[int]]]]:
    if not path.exists():
        raise FileNotFoundError(f"Prediction file not found for method {name}: {path}")

    # Check file type
    suffix = path.suffix.lower()
    if suffix in (".out", ".kraken"):
        preds = parse_kraken_output(path, valid_query_ids=valid_query_ids)
        return preds, None
    elif suffix in (".tsv", ".blast"):
        cands = parse_blast_output(path, valid_query_ids=valid_query_ids)
        top1 = {q: (cands[q][0] if cands[q] else -1) for q in valid_query_ids}
        return top1, cands
    elif suffix in (".parquet", ".csv"):
        df = pd.read_parquet(path) if suffix == ".parquet" else pd.read_csv(path)
        qid_col = "query_id" if "query_id" in df.columns else ("id" if "id" in df.columns else df.columns[0])
        df[qid_col] = df[qid_col].astype(str)
        if "prediction" in df.columns:
            pred_col = "prediction"
        elif "pred" in df.columns:
            pred_col = "pred"
        elif "label" in df.columns:
            pred_col = "label"
        elif "pred_mean_logits" in df.columns:
            pred_col = "pred_mean_logits"
        else:
            pred_col = df.columns[1]

        preds = dict(zip(df[qid_col], df[pred_col].astype(int)))
        # Check for top columns
        top_cols = [c for c in df.columns if c.startswith("top") and c[3:].isdigit()]
        top_cands = None
        if top_cols:
            sorted_top_cols = sorted(top_cols, key=lambda c: int(c[3:]))
            top_cands = {
                row[qid_col]: [row[c] for c in sorted_top_cols if row[c] >= 0]
                for _, row in df.iterrows()
            }
        return preds, top_cands
    else:
        raise ValueError(f"Unsupported prediction file format for method {name}: {path}")


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    start_time = time.time()

    truth_df = load_truth_file(args.truth)
    query_ids = truth_df["query_id"].tolist()
    y_true = truth_df["label"].to_numpy()

    # Merge overlap audit if provided
    if args.overlap_audit and args.overlap_audit.exists():
        overlap_df = pd.read_csv(args.overlap_audit)
        if "query_id" in overlap_df.columns and "has_overlap" in overlap_df.columns:
            overlap_df["query_id"] = overlap_df["query_id"].astype(str)
            truth_df = truth_df.merge(
                overlap_df[["query_id", "has_overlap"]],
                on="query_id",
                how="left",
            )
            truth_df["has_overlap"] = truth_df["has_overlap"].fillna(False)

    predictions: Dict[str, Dict[str, int]] = {}
    top_candidates: Dict[str, Dict[str, List[int]]] = {}

    for item in args.predictions:
        if "=" not in item:
            raise ValueError(f"Prediction argument must be NAME=PATH, got: {item}")
        name, path_str = item.split("=", 1)
        pred_dict, cand_dict = load_method_predictions(name, Path(path_str), query_ids)
        predictions[name] = pred_dict
        if cand_dict:
            top_candidates[name] = cand_dict

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.regime == "long":
        metrics_df, paired_df = evaluate_long_baselines(
            truth_df=truth_df,
            predictions=predictions,
            top_candidates=top_candidates if top_candidates else None,
            genome_overlap_col=args.genome_overlap_col,
            length_col=args.length_col,
        )
    else:
        # Short regime
        masks = {"all": np.ones(len(truth_df), dtype=bool)}
        if "has_overlap" in truth_df.columns:
            has_ov = truth_df["has_overlap"].to_numpy(dtype=bool)
            masks["overlap"] = has_ov
            masks["no_exact_overlap"] = ~has_ov

        # Align predictions to arrays
        aligned_preds = {
            m: np.array([preds.get(q, -1) for q in query_ids], dtype=np.int64)
            for m, preds in predictions.items()
        }
        aligned_cands = {
            m: [cands.get(q, []) for q in query_ids]
            for m, cands in top_candidates.items()
        }

        metrics_rows = []
        for stratum, mask in masks.items():
            if not mask.any():
                continue
            sub_y = y_true[mask]
            for m, arr in aligned_preds.items():
                sub_p = arr[mask]
                sub_c = [aligned_cands[m][i] for i in np.where(mask)[0]] if m in aligned_cands else None
                res = evaluate_predictions(sub_y, sub_p, top_candidates=sub_c)
                metrics_rows.append({"stratum": stratum, "method": m, **res})
        metrics_df = pd.DataFrame(metrics_rows)

        from itertools import combinations
        paired_rows = []
        m_names = list(aligned_preds.keys())
        for stratum, mask in masks.items():
            if not mask.any():
                continue
            sub_y = y_true[mask]
            for a_name, b_name in combinations(m_names, 2):
                a_arr = aligned_preds[a_name][mask]
                b_arr = aligned_preds[b_name][mask]
                disc = paired_discordance(sub_y, a_arr, b_arr)
                a_cor = (a_arr == sub_y)
                b_cor = (b_arr == sub_y)
                diff = a_cor.astype(float) - b_cor.astype(float)
                ci_low, ci_high = bootstrap_species_difference(
                    diff, sub_y, repeats=args.bootstrap_repeats, seed=args.seed
                )
                paired_rows.append({
                    "stratum": stratum,
                    "method_a": a_name,
                    "method_b": b_name,
                    "accuracy_a_pct": 100.0 * float(a_cor.mean()),
                    "accuracy_b_pct": 100.0 * float(b_cor.mean()),
                    "difference_pp": 100.0 * float(diff.mean()),
                    "ci95_low_pp": ci_low,
                    "ci95_high_pp": ci_high,
                    **disc,
                })
        paired_df = pd.DataFrame(paired_rows)

    metrics_csv = args.output_dir / "metrics_by_stratum.csv"
    paired_csv = args.output_dir / "paired_discordance_bootstrap.csv"
    manifest_json = args.output_dir / "comparison_manifest.json"

    metrics_df.to_csv(metrics_csv, index=False)
    paired_df.to_csv(paired_csv, index=False)

    manifest = {
        "regime": args.regime,
        "n_queries": len(truth_df),
        "methods": list(predictions.keys()),
        "bootstrap_repeats": args.bootstrap_repeats,
        "seed": args.seed,
        "execution_seconds": time.time() - start_time,
        "artifacts": {
            "metrics_csv": str(metrics_csv.resolve()),
            "paired_csv": str(paired_csv.resolve()),
        },
    }
    manifest_json.write_text(json.dumps(manifest, indent=2))

    print(f"Metrics written to: {metrics_csv}")
    print(f"Paired discordances written to: {paired_csv}")
    print(f"Manifest written to: {manifest_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
