"""Compact fragment bootstrap estimates for C04."""
from __future__ import annotations

from typing import Any, Dict, Optional, Sequence
import numpy as np
from .metrics import compute_f1_scores

PROTOCOL = "c04_stratified_fragment_v1"


def bootstrap_classification_intervals(
    y_true: Sequence[int], y_pred: Sequence[int], num_classes: int, *,
    top5_hit: Optional[Sequence[bool]] = None, repetitions: int = 2000,
    seed: int = 42, confidence: float = 0.95,
) -> Dict[str, Any]:
    """Resample fragments within true-class strata, preserving each support."""
    truth = np.asarray(y_true, dtype=np.int64)
    pred = np.asarray(y_pred, dtype=np.int64)
    if truth.ndim != 1 or pred.shape != truth.shape or truth.size == 0:
        raise ValueError("y_true/y_pred must be aligned, non-empty vectors")
    if num_classes <= 0 or np.any((truth < 0) | (truth >= num_classes)):
        raise ValueError("truth labels must be within the declared class range")
    if repetitions < 1 or not 0.0 < confidence < 1.0:
        raise ValueError("repetitions must be positive and confidence between 0 and 1")
    hits = None if top5_hit is None else np.asarray(top5_hit, dtype=bool)
    if hits is not None and hits.shape != truth.shape:
        raise ValueError("top5_hit must align with y_true")
    classes = np.unique(truth)
    strata = [np.flatnonzero(truth == cls) for cls in classes]
    point_f1, macro, weighted = compute_f1_scores(truth, pred, num_classes)
    del point_f1
    point = {"accuracy": float(np.mean(truth == pred)), "macro_f1": macro, "weighted_f1": weighted}
    if hits is not None:
        point["top5_accuracy"] = float(hits.mean())
    draws = {key: np.empty(repetitions, dtype=np.float64) for key in point}
    rng = np.random.default_rng(seed)
    for rep in range(repetitions):
        support = np.bincount(truth, minlength=num_classes)
        predicted_count = np.zeros(num_classes, dtype=np.int64)
        true_positive = np.zeros(num_classes, dtype=np.int64)
        correct = 0
        top5_count = 0
        for cls, ix in zip(classes, strata):
            values, counts = np.unique(pred[ix], return_counts=True)
            sampled = rng.multinomial(len(ix), counts / counts.sum())
            valid = (values >= 0) & (values < num_classes)
            np.add.at(predicted_count, values[valid], sampled[valid])
            pos = np.flatnonzero(values == cls)
            if pos.size:
                true_positive[cls] = sampled[pos[0]]
                correct += int(sampled[pos[0]])
            if hits is not None:
                hv, hc = np.unique(hits[ix], return_counts=True)
                top5_count += int(rng.multinomial(len(ix), hc / hc.sum())[hv].sum())
        denom = predicted_count + support
        f1 = np.divide(2 * true_positive, denom, out=np.zeros(num_classes, dtype=np.float64), where=denom > 0)
        total = int(support.sum())
        sample = {"accuracy": correct / total, "macro_f1": float(f1.mean()), "weighted_f1": float(np.dot(f1, support) / total)}
        if hits is not None:
            sample["top5_accuracy"] = top5_count / total
        for key, value in sample.items():
            draws[key][rep] = value
    alpha = (1.0 - confidence) / 2.0
    estimates: Dict[str, Any] = {}
    for key, value in point.items():
        estimates[key] = {
            "estimate": value, "unit": "proportion", "ci_level": confidence,
            "ci_lower": float(np.quantile(draws[key], alpha)),
            "ci_upper": float(np.quantile(draws[key], 1.0 - alpha)),
            "n": int(truth.size), "declared_classes": int(num_classes),
            "present_classes": int(len(classes)), "resampling_unit": "fragment",
            "strata": "true_taxon_with_fixed_support", "repetitions": repetitions,
            "seed": seed, "protocol": PROTOCOL,
        }
    if hits is None:
        estimates["top5_accuracy"] = {"estimate": None, "ci_lower": None, "ci_upper": None,
                                      "unavailable_reason": "top5_hit was not retained per query"}
    return {"protocol": PROTOCOL, "estimates": estimates}
