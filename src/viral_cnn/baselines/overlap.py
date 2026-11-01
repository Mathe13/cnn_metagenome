"""Sequence overlap auditing between test queries and training references."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, Union
import numpy as np
import pandas as pd


@dataclass
class BenchmarkArtifacts:
    """Prepared baseline benchmark data including overlap audit and predictions."""

    audit_df: pd.DataFrame
    strata_masks: Dict[str, np.ndarray]
    predictions: Dict[str, Dict[float, np.ndarray]]
    top_candidates: Dict[str, Dict[float, Optional[np.ndarray]]]
    runtime: Optional[pd.DataFrame] = None


DEFAULT_METHOD_ALIASES: Dict[str, List[str]] = {
    "resnet50_species": ["resnet50_species", "ResNet50", "resnet_clean_species", "resnet50"],
    "resnet_noise_species": ["resnet_noise_species", "ResNet_F", "resnet_noise"],
    "convnext_zero_species": ["convnext_zero_species", "ConvNeXt_Z", "convnext_zero"],
    "convnext_transfer_species": ["convnext_transfer_species", "ConvNeXt_T", "convnext_transfer"],
    "convnext_noise_species": ["convnext_noise_species", "ConvNeXt_F", "convnext_noise"],
    "Kraken2-flat": ["Kraken2-flat", "kraken2_flat", "kraken2", "Kraken2"],
    "BLASTN": ["BLASTN", "blastn", "blast"],
}


def canonical_sequence_hash(seq: str) -> str:
    """Compute sha256 hash of uppercase DNA sequence."""
    clean = seq.strip().upper().encode("ascii")
    return hashlib.sha256(clean).hexdigest()


def _first_series(df: pd.DataFrame, col_name: str) -> pd.Series:
    """Return a 1D Series for col_name, selecting the first matching column if duplicated."""
    col_obj = df[col_name]
    return col_obj.iloc[:, 0] if isinstance(col_obj, pd.DataFrame) else col_obj


def audit_exact_overlap(
    test_queries: Union[pd.DataFrame, Iterable[Tuple[str, str, int]]],
    train_sequences: Union[Iterable[str], Iterable[Tuple[str, str, int]], Set[str]],
) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
    """Audit exact sequence overlap between test queries and training references.

    Args:
      test_queries: DataFrame with ['query_id', 'sequence' (or 'sequence_padded'/'padded_Representation'), 'label']
        or iterable of (qid, seq, label).
      train_sequences: iterable of training sequence strings, tuples, or set of precomputed hashes.

    Returns:
      audit_df: DataFrame with query_id, label, train_count, has_overlap.
      masks: dict with boolean masks for 'all', 'overlap', 'no_exact_overlap'.
    """
    if isinstance(test_queries, pd.DataFrame):
        seq_col = next(
            (c for c in ("sequence_padded", "padded_Representation", "sequence") if c in test_queries.columns),
            None,
        )
        if seq_col is None:
            raise KeyError(
                "test_queries DataFrame must contain one of ('sequence_padded', 'padded_Representation', 'sequence')"
            )
        qids = _first_series(test_queries, "query_id").tolist()
        seqs = _first_series(test_queries, seq_col).astype(str).tolist()
        labels = [int(lbl) for lbl in _first_series(test_queries, "label").tolist()]
        records = list(zip(qids, seqs, labels))
    else:
        records = list(test_queries)

    # Build reference hash counts
    train_counts: Dict[str, int] = {}
    for item in train_sequences:
        if isinstance(item, str):
            h = canonical_sequence_hash(item) if len(item) != 64 else item
        elif isinstance(item, (tuple, list)):
            h = canonical_sequence_hash(item[1])
        else:
            h = str(item)
        train_counts[h] = train_counts.get(h, 0) + 1

    rows = []
    for qid, seq, label in records:
        h = canonical_sequence_hash(seq)
        count = train_counts.get(h, 0)
        rows.append({
            "query_id": qid,
            "label": label,
            "train_count": count,
            "has_overlap": count > 0,
        })

    audit_df = pd.DataFrame(rows)
    has_overlap_arr = audit_df["has_overlap"].to_numpy()

    masks = {
        "all": np.ones(len(audit_df), dtype=bool),
        "overlap": has_overlap_arr,
        "no_exact_overlap": ~has_overlap_arr,
    }

    return audit_df, masks


def load_prepared_baseline_benchmark(
    benchmark_dir: Union[str, Path],
    methods: Optional[List[str]] = None,
    noise_levels: Optional[List[float]] = None,
) -> BenchmarkArtifacts:
    """Load and validate prepared baseline benchmark artifacts from cluster directory.

    Args:
      benchmark_dir: Root directory containing exact_overlap.csv (or truth.parquet)
        and comparison_results/ directory.
      methods: List of method names to load (default: 5 CNN variants, Kraken2, BLASTN).
      noise_levels: Nominal noise levels to load (default: [0.0, 0.02, 0.05, 0.10, 0.15]).

    Returns:
      BenchmarkArtifacts with audit_df, strata_masks, predictions, top_candidates, and runtime.

    Raises:
      FileNotFoundError: If benchmark_dir or essential artifacts are missing.
      ValueError: If query_ids are duplicate or misaligned with predictions.
    """
    bench_p = Path(benchmark_dir).resolve()
    if not bench_p.exists():
        raise FileNotFoundError(f"Baseline benchmark directory not found: {bench_p}")

    # 1. Locate truth / exact overlap audit table
    truth_candidates = [
        bench_p / "exact_overlap.csv",
        bench_p / "exact_overlap.parquet",
        bench_p / "truth.parquet",
        bench_p / "queries_truth.csv",
        bench_p / "comparison_results" / "exact_overlap.csv",
    ]
    truth_path = next((p for p in truth_candidates if p.exists()), None)
    if truth_path is None:
        raise FileNotFoundError(
            f"No exact overlap or truth table found in {bench_p}. Candidates: {[str(c.name) for c in truth_candidates]}"
        )

    if truth_path.suffix == ".parquet":
        audit_df = pd.read_parquet(truth_path)
    else:
        audit_df = pd.read_csv(truth_path)

    if "query_id" not in audit_df.columns:
        raise KeyError(f"Truth/overlap table {truth_path} must contain 'query_id' column")

    if not audit_df["query_id"].is_unique:
        raise ValueError(f"Duplicate query_id found in truth table {truth_path}")

    # Determine overlap mask
    if "has_overlap" in audit_df.columns:
        has_overlap_arr = audit_df["has_overlap"].astype(bool).to_numpy()
    elif "train_count" in audit_df.columns:
        has_overlap_arr = (audit_df["train_count"].fillna(0).astype(int) > 0).to_numpy()
        audit_df["has_overlap"] = has_overlap_arr
    else:
        raise KeyError(f"Truth/overlap table {truth_path} must contain 'has_overlap' or 'train_count'")

    strata_masks = {
        "all": np.ones(len(audit_df), dtype=bool),
        "overlap": has_overlap_arr,
        "no_exact_overlap": ~has_overlap_arr,
    }

    truth_qids = audit_df["query_id"].tolist()
    truth_qids_set = set(truth_qids)

    # 2. Results directory
    comparison_dir = bench_p / "comparison_results" if (bench_p / "comparison_results").is_dir() else bench_p

    target_methods = methods if methods is not None else list(DEFAULT_METHOD_ALIASES.keys())
    target_noise = noise_levels if noise_levels is not None else [0.0, 0.02, 0.05, 0.10, 0.15]

    predictions: Dict[str, Dict[float, np.ndarray]] = {m: {} for m in target_methods}
    top_candidates: Dict[str, Dict[float, Optional[np.ndarray]]] = {m: {} for m in target_methods}

    for method in target_methods:
        aliases = DEFAULT_METHOD_ALIASES.get(method, [method])
        for noise in target_noise:
            tag = f"noise_{round(noise * 100):02d}"
            # Locate prediction file trying aliases
            pred_file: Optional[Path] = None
            for alias in aliases:
                p_cand = comparison_dir / f"predictions_{alias}_{tag}.parquet"
                if p_cand.exists():
                    pred_file = p_cand
                    break
                p_csv = comparison_dir / f"predictions_{alias}_{tag}.csv"
                if p_csv.exists():
                    pred_file = p_csv
                    break

            if pred_file is None:
                raise FileNotFoundError(
                    f"Prediction file for method '{method}' (aliases={aliases}) at noise {noise} ({tag}) not found in {comparison_dir}"
                )

            if pred_file.suffix == ".parquet":
                pred_df = pd.read_parquet(pred_file)
            else:
                pred_df = pd.read_csv(pred_file)

            if "query_id" not in pred_df.columns:
                raise KeyError(f"Missing 'query_id' column in {pred_file}")

            if not pred_df["query_id"].is_unique:
                raise ValueError(f"Duplicate 'query_id' detected in prediction file: {pred_file}")

            if set(pred_df["query_id"]) != truth_qids_set:
                raise ValueError(
                    f"Query ID mismatch between truth ({len(truth_qids_set)}) and predictions in {pred_file} ({len(pred_df['query_id'])})"
                )

            pred_col = next(
                (c for c in ("prediction", "y_pred", "pred", "label_pred") if c in pred_df.columns),
                None,
            )
            if pred_col is None:
                raise KeyError(f"No prediction column ('prediction', 'y_pred') found in {pred_file}")

            aligned_series = pred_df.set_index("query_id").loc[truth_qids, pred_col]
            if aligned_series.isna().any():
                raise ValueError(f"Missing predictions (NaN) found in {pred_file}")

            predictions[method][noise] = aligned_series.to_numpy(dtype=np.int64)

            # Top-5 candidates if available
            top_cols = [f"top{k}" for k in range(1, 6)]
            if all(c in pred_df.columns for c in top_cols):
                top_arr = pred_df.set_index("query_id").loc[truth_qids, top_cols].to_numpy(dtype=np.int64)
                top_candidates[method][noise] = top_arr
            else:
                top_candidates[method][noise] = None

    # 3. Optional runtime records
    runtime_path = comparison_dir / "runtime.jsonl"
    runtime_df: Optional[pd.DataFrame] = None
    if runtime_path.exists():
        try:
            records = []
            with runtime_path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        records.append(json.loads(line))
            if records:
                runtime_df = pd.DataFrame(records)
        except Exception:
            runtime_df = None

    return BenchmarkArtifacts(
        audit_df=audit_df,
        strata_masks=strata_masks,
        predictions=predictions,
        top_candidates=top_candidates,
        runtime=runtime_df,
    )

