"""Unified evaluation metrics, Wilson confidence intervals, and paired species bootstrap."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np


def compute_macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Computes Macro-F1 across unique classes present in y_true without sklearn."""
    valid_t_mask = y_true >= 0
    if not valid_t_mask.any():
        return 0.0
    present_labels = np.unique(y_true[valid_t_mask])
    k = len(present_labels)
    if k == 0:
        return 0.0

    lbl_to_idx = {lbl: i for i, lbl in enumerate(present_labels)}
    y_t_mapped = np.array([lbl_to_idx[l] for l in y_true], dtype=np.int64)
    y_p_mapped = np.array([lbl_to_idx.get(l, -1) for l in y_pred], dtype=np.int64)

    support = np.bincount(y_t_mapped, minlength=k)[:k]
    valid_p = y_p_mapped[y_p_mapped >= 0]
    pred_count = np.bincount(valid_p, minlength=k)[:k]

    match_mask = (y_t_mapped == y_p_mapped) & (y_p_mapped >= 0)
    tp = np.bincount(y_t_mapped[match_mask], minlength=k)[:k]

    denom = pred_count + support
    f1s = np.zeros(k, dtype=np.float64)
    valid_d = denom > 0
    f1s[valid_d] = 2.0 * tp[valid_d] / denom[valid_d]
    return float(np.mean(f1s))


def wilson_score_interval(
    successes: int,
    n: int,
    z: float = 1.96,
) -> Tuple[float, float]:
    """Compute closed-form Wilson score confidence interval for a proportion."""
    if n <= 0:
        return (np.nan, np.nan)
    p = successes / n
    den = 1.0 + (z * z) / n
    center = (p + (z * z) / (2.0 * n)) / den
    half = (z * np.sqrt((p * (1.0 - p)) / n + (z * z) / (4.0 * n * n))) / den
    return (max(0.0, float(center - half)), min(1.0, float(center + half)))


def evaluate_predictions(
    y_true: Union[np.ndarray, List[int]],
    y_pred: Union[np.ndarray, List[int]],
    top_candidates: Optional[Union[np.ndarray, List[List[int]]]] = None,
    num_classes: Optional[int] = None,
) -> Dict[str, Any]:
    """Compute unified classification metrics treating abstentions (-1) as errors.

    Total queries N is the denominator for Top-1 accuracy.
    Coverage is fraction of queries where y_pred >= 0.
    Conditional accuracy is accuracy restricted to covered queries.
    """
    y_t = np.asarray(y_true, dtype=np.int64)
    y_p = np.asarray(y_pred, dtype=np.int64)

    if len(y_t) != len(y_p):
        raise ValueError(f"Length mismatch: y_true={len(y_t)} vs y_pred={len(y_p)}")

    n_total = len(y_t)
    if n_total == 0:
        return {
            "n": 0,
            "accuracy": np.nan,
            "accuracy_ci_low": np.nan,
            "accuracy_ci_high": np.nan,
            "coverage": np.nan,
            "conditional_accuracy": np.nan,
            "macro_f1": np.nan,
            "top5": np.nan,
        }

    # Hit is True only when y_p == y_t (since y_t >= 0, y_p == -1 is False)
    hit = (y_p == y_t)
    n_hits = int(hit.sum())
    accuracy = float(hit.mean())
    ci_low, ci_high = wilson_score_interval(n_hits, n_total)

    covered = (y_p >= 0)
    coverage = float(covered.mean())
    conditional_accuracy = float(hit[covered].mean()) if covered.any() else np.nan

    present_labels = np.unique(y_t)
    macro_f1 = compute_macro_f1(y_t, y_p)

    top5 = np.nan
    if top_candidates is not None:
        top_matches = []
        for true_label, cand_list in zip(y_t, top_candidates):
            cands = cand_list[:5] if cand_list is not None else []
            top_matches.append(int(true_label in cands))
        top5 = float(np.mean(top_matches))

    return {
        "n": n_total,
        "n_classes": len(present_labels),
        "accuracy": accuracy,
        "accuracy_ci_low": ci_low,
        "accuracy_ci_high": ci_high,
        "coverage": coverage,
        "conditional_accuracy": conditional_accuracy,
        "macro_f1": macro_f1,
        "top5": top5,
    }


def paired_discordance(
    y_true: Union[np.ndarray, List[int]],
    y_pred_a: Union[np.ndarray, List[int]],
    y_pred_b: Union[np.ndarray, List[int]],
) -> Dict[str, Any]:
    """Compute 2x2 contingency table and discordance statistics between two methods."""
    y_t = np.asarray(y_true, dtype=np.int64)
    a_pred = np.asarray(y_pred_a, dtype=np.int64)
    b_pred = np.asarray(y_pred_b, dtype=np.int64)

    if not (len(y_t) == len(a_pred) == len(b_pred)):
        raise ValueError(
            f"Mismatched query counts: truth={len(y_t)}, a={len(a_pred)}, b={len(b_pred)}"
        )

    a_correct = (a_pred == y_t)
    b_correct = (b_pred == y_t)

    both_correct = int((a_correct & b_correct).sum())
    only_a = int((a_correct & ~b_correct).sum())
    only_b = int((~a_correct & b_correct).sum())
    both_wrong = int((~a_correct & ~b_correct).sum())
    discordant = only_a + only_b

    return {
        "n": len(y_t),
        "both_correct": both_correct,
        "only_a_correct": only_a,
        "only_b_correct": only_b,
        "both_wrong": both_wrong,
        "discordant_queries": discordant,
        "discordant_pct": 100.0 * discordant / len(y_t) if len(y_t) > 0 else 0.0,
    }


def bootstrap_species_difference(
    differences: Union[np.ndarray, List[float]],
    labels: Union[np.ndarray, List[int]],
    repeats: int = 2000,
    seed: int = 42,
) -> Tuple[float, float]:
    """Estimate 95% CI for paired accuracy differences resampling true species with replacement."""
    diff_arr = np.asarray(differences, dtype=np.float64)
    lbl_arr = np.asarray(labels, dtype=np.int64)

    if len(diff_arr) != len(lbl_arr):
        raise ValueError("Length mismatch between differences and labels")

    _, inverse = np.unique(lbl_arr, return_inverse=True)
    n_species = int(inverse.max()) + 1 if len(inverse) > 0 else 0

    if n_species < 2:
        return (np.nan, np.nan)

    sums = np.bincount(inverse, weights=diff_arr)
    sizes = np.bincount(inverse)

    rng = np.random.default_rng(seed)
    estimates = np.empty(repeats, dtype=np.float64)

    chunk_size = 32
    for start in range(0, repeats, chunk_size):
        stop = min(start + chunk_size, repeats)
        sampled = rng.integers(0, n_species, size=(stop - start, n_species))
        estimates[start:stop] = sums[sampled].sum(axis=1) / sizes[sampled].sum(axis=1)

    low_high = np.quantile(estimates, [0.025, 0.975])
    return (float(100.0 * low_high[0]), float(100.0 * low_high[1]))
