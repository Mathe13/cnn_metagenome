"""
Classification metrics, performance bands, and paired comparisons for taxonomic evaluation.
"""

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class ClassificationMetrics:
    top1_accuracy: float
    top5_accuracy: Optional[float]
    macro_f1: float
    weighted_f1: float
    cross_entropy_loss: Optional[float]
    n_samples: int
    n_classes: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def compute_f1_scores(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int,
) -> Tuple[np.ndarray, float, float]:
    """
    Computes per-class F1, Macro-F1, and Weighted-F1 matching scikit-learn (zero_division=0).

    Uses the exact identity: 2*TP + FP + FN = pred_count + support
    where F1 = (2 * TP) / (pred_count + support).
    """
    y_true_valid = y_true[(y_true >= 0) & (y_true < num_classes)]
    y_pred_valid = y_pred[(y_pred >= 0) & (y_pred < num_classes)]

    support = np.bincount(y_true_valid, minlength=num_classes)[:num_classes]
    pred_count = np.bincount(y_pred_valid, minlength=num_classes)[:num_classes]

    # True positives: where both match and are within class range
    match_mask = (y_true == y_pred) & (y_true >= 0) & (y_true < num_classes)
    tp = np.bincount(y_true[match_mask], minlength=num_classes)[:num_classes]

    denom = pred_count + support
    per_class_f1 = np.zeros(num_classes, dtype=np.float64)
    valid_denom = denom > 0
    per_class_f1[valid_denom] = (2.0 * tp[valid_denom]) / denom[valid_denom]

    macro_f1 = float(np.mean(per_class_f1))
    total_support = int(np.sum(support))
    weighted_f1 = (
        float(np.sum(per_class_f1 * support) / total_support)
        if total_support > 0
        else 0.0
    )

    return per_class_f1, macro_f1, weighted_f1


def compute_classification_metrics(
    y_true: Union[np.ndarray, torch.Tensor, List[int]],
    y_pred: Union[np.ndarray, torch.Tensor, List[int]],
    y_top5: Optional[Union[np.ndarray, torch.Tensor, List[List[int]]]] = None,
    logits: Optional[Union[np.ndarray, torch.Tensor]] = None,
    num_classes: Optional[int] = None,
) -> ClassificationMetrics:
    """
    Computes Top-1 accuracy, Top-5 accuracy, cross-entropy loss, Macro-F1, and Weighted-F1.
    """
    if isinstance(y_true, torch.Tensor):
        y_true_np = y_true.detach().cpu().numpy()
    else:
        y_true_np = np.asarray(y_true, dtype=np.int64)

    if isinstance(y_pred, torch.Tensor):
        y_pred_np = y_pred.detach().cpu().numpy()
    else:
        y_pred_np = np.asarray(y_pred, dtype=np.int64)

    n_samples = len(y_true_np)
    if n_samples == 0:
        raise ValueError("y_true cannot be empty.")

    if len(y_pred_np) != n_samples:
        raise ValueError(
            f"Shape mismatch: y_true has {n_samples} samples but y_pred has {len(y_pred_np)}"
        )

    # Inferred class count
    classes_in_data = max(
        int(np.max(y_true_np)) if len(y_true_np) > 0 else 0,
        int(np.max(y_pred_np)) if len(y_pred_np) > 0 else 0,
    ) + 1
    total_classes = num_classes if num_classes is not None else classes_in_data

    # Top-1 Accuracy (%)
    top1_acc = float(np.mean(y_true_np == y_pred_np) * 100.0)

    # Top-5 Accuracy (%)
    top5_acc: Optional[float] = None
    if y_top5 is not None:
        if isinstance(y_top5, torch.Tensor):
            y_top5_np = y_top5.detach().cpu().numpy()
        else:
            y_top5_np = np.asarray(y_top5, dtype=np.int64)

        if len(y_top5_np) != n_samples:
            raise ValueError("y_top5 sample count must match y_true")

        # Check if true label is in any of the top 5 predictions for each sample
        matches = np.any(y_top5_np == y_true_np[:, np.newaxis], axis=1)
        top5_acc = float(np.mean(matches) * 100.0)

    # Cross-Entropy Loss
    loss_val: Optional[float] = None
    if logits is not None:
        if isinstance(logits, np.ndarray):
            logits_tensor = torch.tensor(logits, dtype=torch.float32)
        else:
            logits_tensor = logits.detach().cpu().float()

        y_true_tensor = torch.tensor(y_true_np, dtype=torch.long)
        # Filter valid targets in range [0, total_classes - 1]
        valid_mask = (y_true_tensor >= 0) & (y_true_tensor < logits_tensor.shape[1])
        if valid_mask.any():
            loss_val = float(
                F.cross_entropy(logits_tensor[valid_mask], y_true_tensor[valid_mask]).item()
            )

    # Macro-F1 and Weighted-F1 across all declared classes
    _, macro_f1, weighted_f1 = compute_f1_scores(
        y_true=y_true_np,
        y_pred=y_pred_np,
        num_classes=total_classes,
    )

    return ClassificationMetrics(
        top1_accuracy=round(top1_acc, 4),
        top5_accuracy=round(top5_acc, 4) if top5_acc is not None else None,
        macro_f1=round(macro_f1, 6),
        weighted_f1=round(weighted_f1, 6),
        cross_entropy_loss=round(loss_val, 6) if loss_val is not None else None,
        n_samples=n_samples,
        n_classes=total_classes,
    )


def compute_species_f1_bands(
    y_true: Union[np.ndarray, List[int]],
    y_pred: Union[np.ndarray, List[int]],
    num_classes: int,
) -> Dict[str, Any]:
    """
    Partitions per-species F1-scores into four continuous, gapless performance bands:
      - band_0_to_50: [0.00, 0.50)
      - band_50_to_80: [0.50, 0.80)
      - band_80_to_95: [0.80, 0.95)
      - band_95_to_100: [0.95, 1.00] (closed on right)

    Guarantees that the sum of counts across all bands strictly equals num_classes.
    """
    y_true_np = np.asarray(y_true, dtype=np.int64)
    y_pred_np = np.asarray(y_pred, dtype=np.int64)

    per_class_f1, _, _ = compute_f1_scores(
        y_true=y_true_np,
        y_pred=y_pred_np,
        num_classes=num_classes,
    )

    bands = {
        "band_0_to_50": {"range": "[0.00, 0.50)", "count": 0, "percent": 0.0},
        "band_50_to_80": {"range": "[0.50, 0.80)", "count": 0, "percent": 0.0},
        "band_80_to_95": {"range": "[0.80, 0.95)", "count": 0, "percent": 0.0},
        "band_95_to_100": {"range": "[0.95, 1.00]", "count": 0, "percent": 0.0},
    }

    for f1 in per_class_f1:
        if f1 < 0.50:
            bands["band_0_to_50"]["count"] += 1
        elif f1 < 0.80:
            bands["band_50_to_80"]["count"] += 1
        elif f1 < 0.95:
            bands["band_80_to_95"]["count"] += 1
        else:
            bands["band_95_to_100"]["count"] += 1

    total_allocated = sum(b["count"] for b in bands.values())
    if total_allocated != num_classes:
        raise ValueError(
            f"F1 band allocation conservation violation: allocated {total_allocated} classes "
            f"but num_classes is {num_classes}"
        )

    for b in bands.values():
        b["percent"] = round(100.0 * b["count"] / num_classes, 2)

    return {
        "total_classes": num_classes,
        "bands": bands,
    }


def compute_paired_comparison(
    preds_a: Union[np.ndarray, List[int]],
    preds_b: Union[np.ndarray, List[int]],
    y_true: Union[np.ndarray, List[int]],
) -> Dict[str, Any]:
    """
    Calculates pairwise difference in percentage points and paired contingency table
    on identical query sets.
    """
    p_a = np.asarray(preds_a, dtype=np.int64)
    p_b = np.asarray(preds_b, dtype=np.int64)
    truth = np.asarray(y_true, dtype=np.int64)

    n = len(truth)
    if len(p_a) != n or len(p_b) != n:
        raise ValueError(
            f"Length mismatch: y_true has {n}, preds_a has {len(p_a)}, preds_b has {len(p_b)}"
        )

    correct_a = (p_a == truth)
    correct_b = (p_b == truth)

    both_correct = int((correct_a & correct_b).sum())
    only_a_correct = int((correct_a & ~correct_b).sum())
    only_b_correct = int((~correct_a & correct_b).sum())
    both_wrong = int((~correct_a & ~correct_b).sum())

    acc_a = float(np.mean(correct_a) * 100.0)
    acc_b = float(np.mean(correct_b) * 100.0)
    acc_diff_pp = round(acc_a - acc_b, 4)

    return {
        "n_samples": n,
        "accuracy_a": round(acc_a, 4),
        "accuracy_b": round(acc_b, 4),
        "accuracy_difference_pp": acc_diff_pp,
        "contingency": {
            "both_correct": both_correct,
            "only_a_correct": only_a_correct,
            "only_b_correct": only_b_correct,
            "both_wrong": both_wrong,
        },
    }
