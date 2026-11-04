"""Long-read and long-sequence baseline evaluation and Voter integration."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

from viral_cnn.baselines.paired import (
    evaluate_predictions,
    paired_discordance,
    bootstrap_species_difference,
)

C02_LONG_VARIANTS = (
    "resnet_clean_species",
    "resnet_noise_species",
    "convnext_zero_species",
    "convnext_transfer_species",
    "convnext_noise_species",
)
C02_LONG_PROTOCOL = "c02_audit_v1"


def validate_c02_prediction_artifacts(artifacts: List[Dict[str, Any]], queries: pd.DataFrame, *, truth_col: str = "truth_label", query_id_col: str = "query_id") -> Dict[str, pd.DataFrame]:
    """Validate a five-variant C02 comparison by identity, never by row position."""
    issues: List[str] = []
    seen: Dict[str, pd.DataFrame] = {}
    expected_ids = queries[query_id_col].astype(str).tolist()
    if len(expected_ids) != len(set(expected_ids)):
        issues.append("query table contains duplicate query_id")
    expected_truth = dict(zip(expected_ids, queries[truth_col].astype(int)))
    expected_variants = set(C02_LONG_VARIANTS)
    conditions = set()
    for artifact in artifacts:
        variant = artifact.get("variant")
        if variant not in expected_variants:
            issues.append(f"unknown variant: {variant}")
            continue
        if variant in seen:
            issues.append(f"duplicate variant: {variant}")
            continue
        frame = artifact.get("predictions")
        if not isinstance(frame, pd.DataFrame):
            issues.append(f"{variant}: predictions must be a DataFrame")
            continue
        required = {"query_id", "truth_label", "pred_mean_logits", "pred_majority_vote", "variant", "noise_rate", "handle_short", "long_sequence_protocol"}
        missing = sorted(required - set(frame.columns))
        if missing:
            issues.append(f"{variant}: missing columns {missing}")
            continue
        if frame["query_id"].astype(str).duplicated().any():
            issues.append(f"{variant}: duplicate query_id")
        if frame["truth_label"].isna().any():
            issues.append(f"{variant}: missing truth_label")
        ids = set(frame["query_id"].astype(str))
        if ids != set(expected_ids):
            issues.append(f"{variant}: query IDs missing={sorted(set(expected_ids)-ids)} extra={sorted(ids-set(expected_ids))}")
        if set(frame["variant"].astype(str)) != {variant}:
            issues.append(f"{variant}: artifact variant identity mismatch")
        if set(frame["long_sequence_protocol"].astype(str)) != {C02_LONG_PROTOCOL}:
            issues.append(f"{variant}: protocol identity mismatch")
        if frame["noise_rate"].nunique(dropna=False) != 1 or frame["handle_short"].nunique(dropna=False) != 1:
            issues.append(f"{variant}: artifact must contain exactly one condition")
        for _, row in frame.iterrows():
            qid = str(row["query_id"])
            if qid in expected_truth and pd.notna(row["truth_label"]) and int(row["truth_label"]) != expected_truth[qid]:
                issues.append(f"{variant}: truth mismatch for {qid}")
        conditions.add((tuple(frame["noise_rate"].drop_duplicates().tolist()), tuple(frame["handle_short"].astype(str).drop_duplicates().tolist())))
        seen[variant] = frame.copy()
    missing_variants = sorted(expected_variants - set(seen))
    if missing_variants:
        issues.append(f"missing variants: {missing_variants}")
    if len(conditions) > 1:
        issues.append("condition mismatch across variants")
    if issues:
        raise ValueError("Invalid C02 long comparison: " + "; ".join(issues))
    return seen


def build_c02_table12(artifacts: List[Dict[str, Any]], queries: pd.DataFrame, *, truth_col: str = "truth_label", query_id_col: str = "query_id") -> pd.DataFrame:
    """Compute the paired Table 12 metrics from observed query rows."""
    aligned = validate_c02_prediction_artifacts(artifacts, queries, truth_col=truth_col, query_id_col=query_id_col)
    truth = queries.set_index(queries[query_id_col].astype(str))[truth_col].astype(int)
    rows = []
    for variant in C02_LONG_VARIANTS:
        frame = aligned[variant].copy()
        frame["query_id"] = frame["query_id"].astype(str)
        frame = frame.set_index("query_id").loc[truth.index]
        n = len(truth)
        for aggregation, pred_col in (("mean_logits", "pred_mean_logits"), ("majority_vote", "pred_majority_vote")):
            predictions = pd.to_numeric(frame[pred_col], errors="coerce")
            covered = predictions.notna()
            correct = predictions.eq(truth) & covered
            rows.append({"variant": variant, "aggregation": aggregation, "N": n, "covered": int(covered.sum()), "coverage": float(covered.mean()) if n else None, "correct": int(correct.sum()), "accuracy": float(correct.sum() / n) if n else None, "conditional_accuracy": float(correct.sum() / covered.sum()) if covered.any() else None, "excluded": int(n-covered.sum()), "long_sequence_protocol": C02_LONG_PROTOCOL})
    return pd.DataFrame(rows)


def integrate_long_voter_predictions(
    voter_source: Union[pd.DataFrame, Dict[str, Any]],
) -> Dict[str, Dict[str, int]]:
    """Extract standard prediction mappings from LongSequenceVoter output.

    Returns:
      dict with keys 'Voter_mean_logits' and 'Voter_majority_vote',
      each mapping query_id -> predicted_class_label.
    """
    if isinstance(voter_source, pd.DataFrame):
        df = voter_source
        qid_col = "query_id" if "query_id" in df.columns else ("id" if "id" in df.columns else df.columns[0])
        qids = df[qid_col].astype(str).tolist()

        out: Dict[str, Dict[str, int]] = {}
        if "pred_mean_logits" in df.columns:
            out["Voter_mean_logits"] = dict(zip(qids, df["pred_mean_logits"].astype(int)))
        if "pred_majority_vote" in df.columns:
            out["Voter_majority_vote"] = dict(zip(qids, df["pred_majority_vote"].astype(int)))
        return out

    # If dict of records or mappings
    records = voter_source.get("predictions", voter_source)
    out = {"Voter_mean_logits": {}, "Voter_majority_vote": {}}
    for rec in records:
        qid = str(rec.get("query_id", rec.get("id", "")))
        if "pred_mean_logits" in rec:
            out["Voter_mean_logits"][qid] = int(rec["pred_mean_logits"])
        if "pred_majority_vote" in rec:
            out["Voter_majority_vote"][qid] = int(rec["pred_majority_vote"])
    return out


def evaluate_long_baselines(
    truth_df: pd.DataFrame,
    predictions: Dict[str, Union[Dict[str, int], np.ndarray, List[int]]],
    top_candidates: Optional[Dict[str, Union[Dict[str, List[int]], List[List[int]]]]] = None,
    genome_overlap_col: Optional[str] = "seen_genome",
    length_col: Optional[str] = "length",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate CNN Voter, Kraken2, and BLASTN on long reads across genome & length strata.

    Returns:
      (metrics_df, paired_df)
    """
    df = truth_df.copy()
    qid_col = "query_id" if "query_id" in df.columns else "id"
    query_ids = df[qid_col].astype(str).tolist()
    y_true = df["label"].to_numpy(dtype=np.int64)

    # Align predictions to query_ids
    aligned_preds: Dict[str, np.ndarray] = {}
    for name, preds in predictions.items():
        if isinstance(preds, dict):
            aligned_preds[name] = np.array([preds.get(q, -1) for q in query_ids], dtype=np.int64)
        else:
            arr = np.asarray(preds, dtype=np.int64)
            if len(arr) != len(query_ids):
                raise ValueError(f"Method {name} prediction count {len(arr)} != {len(query_ids)}")
            aligned_preds[name] = arr

    # Align top candidates if present
    aligned_cands: Dict[str, List[List[int]]] = {}
    if top_candidates:
        for name, cands in top_candidates.items():
            if isinstance(cands, dict):
                aligned_cands[name] = [cands.get(q, []) for q in query_ids]
            else:
                aligned_cands[name] = list(cands)

    # Build strata
    strata_masks: Dict[str, np.ndarray] = {
        "all": np.ones(len(df), dtype=bool),
    }

    if genome_overlap_col and genome_overlap_col in df.columns:
        seen_mask = df[genome_overlap_col].astype(bool).to_numpy()
        strata_masks["seen_genome"] = seen_mask
        strata_masks["unseen_genome"] = ~seen_mask

    if length_col and length_col in df.columns:
        lengths = df[length_col].to_numpy(dtype=np.int64)
        strata_masks["len_512_to_2000"] = (lengths >= 512) & (lengths < 2000)
        strata_masks["len_2000_to_5000"] = (lengths >= 2000) & (lengths < 5000)
        strata_masks["len_ge_5000"] = (lengths >= 5000)

    # Compute metrics per stratum
    metrics_rows = []
    for stratum_name, mask in strata_masks.items():
        if not mask.any():
            continue
        sub_y_true = y_true[mask]
        for method_name, preds_arr in aligned_preds.items():
            sub_preds = preds_arr[mask]
            sub_cands = [aligned_cands[method_name][i] for i in np.where(mask)[0]] if method_name in aligned_cands else None
            m = evaluate_predictions(sub_y_true, sub_preds, top_candidates=sub_cands)
            metrics_rows.append({
                "stratum": stratum_name,
                "method": method_name,
                **m,
            })

    metrics_df = pd.DataFrame(metrics_rows)

    # Compute paired discordance and bootstrap differences across methods
    from itertools import combinations
    paired_rows = []
    method_names = list(aligned_preds.keys())

    for stratum_name, mask in strata_masks.items():
        if not mask.any():
            continue
        sub_y_true = y_true[mask]
        for a_name, b_name in combinations(method_names, 2):
            a_sub = aligned_preds[a_name][mask]
            b_sub = aligned_preds[b_name][mask]
            disc = paired_discordance(sub_y_true, a_sub, b_sub)

            a_correct = (a_sub == sub_y_true)
            b_correct = (b_sub == sub_y_true)
            diff = a_correct.astype(float) - b_correct.astype(float)

            ci_low, ci_high = bootstrap_species_difference(diff, sub_y_true, repeats=2000, seed=42)

            paired_rows.append({
                "stratum": stratum_name,
                "method_a": a_name,
                "method_b": b_name,
                "accuracy_a_pct": 100.0 * float(a_correct.mean()),
                "accuracy_b_pct": 100.0 * float(b_correct.mean()),
                "difference_pp": 100.0 * float(diff.mean()),
                "ci95_low_pp": ci_low,
                "ci95_high_pp": ci_high,
                **disc,
            })

    paired_df = pd.DataFrame(paired_rows)
    return metrics_df, paired_df

