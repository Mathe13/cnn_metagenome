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

