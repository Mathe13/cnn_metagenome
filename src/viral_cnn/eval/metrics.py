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
    precomputed_loss: Optional[float] = None,
    precomputed_top5_accuracy: Optional[float] = None,
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
    if precomputed_top5_accuracy is not None:
        top5_acc = float(precomputed_top5_accuracy)
    elif y_top5 is not None:
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
    if precomputed_loss is not None:
        loss_val = float(precomputed_loss)
    elif logits is not None:
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
    thresholds: Tuple[float, ...] = (0.0, 0.50, 0.80, 0.95, 1.0),
    confidences: Optional[Union[np.ndarray, List[float]]] = None,
) -> Dict[str, Any]:
    """
    Partitions per-species F1-scores into continuous, gapless performance bands.

    Parameters:
      y_true: Ground truth class indices [0 .. num_classes - 1].
      y_pred: Predicted class indices [0 .. num_classes - 1].
      num_classes: Total declared class count.
      thresholds: Strictly increasing tuple of cutoffs spanning [0.0, 1.0].
                  Default: (0.0, 0.50, 0.80, 0.95, 1.0).
                  Table 4: (0.0, 0.50, 0.80, 0.90, 1.0).
      confidences: Optional per-sample prediction confidence values aligned with y_true.

    Guarantees that the sum of counts across all bands strictly equals num_classes.
    """
    if len(thresholds) < 2:
        raise ValueError("thresholds must contain at least 2 cutoff values spanning [0.0, 1.0]")
    if thresholds[0] != 0.0 or thresholds[-1] != 1.0:
        raise ValueError(
            f"thresholds must start at 0.0 and end at 1.0, got start={thresholds[0]}, end={thresholds[-1]}"
        )
    for i in range(len(thresholds) - 1):
        if thresholds[i] >= thresholds[i + 1]:
            raise ValueError(
                f"thresholds must be strictly increasing, got {thresholds[i]} >= {thresholds[i + 1]}"
            )

    y_true_np = np.asarray(y_true, dtype=np.int64)
    y_pred_np = np.asarray(y_pred, dtype=np.int64)

    per_class_f1, _, _ = compute_f1_scores(
        y_true=y_true_np,
        y_pred=y_pred_np,
        num_classes=num_classes,
    )

    # Optional per-class average confidence (vectorized via bincount)
    class_mean_conf: Optional[np.ndarray] = None
    if confidences is not None:
        confs_np = np.asarray(confidences, dtype=np.float64)
        if len(confs_np) != len(y_true_np):
            raise ValueError(
                f"Length mismatch: confidences has {len(confs_np)} samples but y_true has {len(y_true_np)}"
            )
        class_mean_conf = np.zeros(num_classes, dtype=np.float64)
        valid_mask = (y_true_np >= 0) & (y_true_np < num_classes)
        valid_y = y_true_np[valid_mask]
        valid_c = confs_np[valid_mask]
        sample_counts = np.bincount(valid_y, minlength=num_classes)[:num_classes]
        conf_sums = np.bincount(valid_y, weights=valid_c, minlength=num_classes)[:num_classes]
        nonzero = sample_counts > 0
        class_mean_conf[nonzero] = conf_sums[nonzero] / sample_counts[nonzero]

    bands: Dict[str, Dict[str, Any]] = {}
    band_keys: List[str] = []
    band_class_indices: Dict[str, List[int]] = {}

    for i in range(len(thresholds) - 1):
        t_low = thresholds[i]
        t_high = thresholds[i + 1]
        is_last = (i == len(thresholds) - 2)
        range_str = f"[{t_low:.2f}, {t_high:.2f}]" if is_last else f"[{t_low:.2f}, {t_high:.2f})"
        key = f"band_{int(round(t_low * 100))}_to_{int(round(t_high * 100))}"
        band_keys.append(key)
        band_class_indices[key] = []
        bands[key] = {
            "range": range_str,
            "count": 0,
            "percent": 0.0,
        }
        if confidences is not None:
            bands[key]["mean_confidence"] = 0.0

    for c_idx, f1 in enumerate(per_class_f1):
        allocated = False
        for i in range(len(thresholds) - 1):
            t_low = thresholds[i]
            t_high = thresholds[i + 1]
            is_last = (i == len(thresholds) - 2)
            in_band = (t_low <= f1 <= t_high) if is_last else (t_low <= f1 < t_high)
            if in_band:
                k = band_keys[i]
                bands[k]["count"] += 1
                band_class_indices[k].append(c_idx)
                allocated = True
                break
        if not allocated:
            # Handle numerical edge cases at boundary 1.0
            last_key = band_keys[-1]
            bands[last_key]["count"] += 1
            band_class_indices[last_key].append(c_idx)

    total_allocated = sum(b["count"] for b in bands.values())
    if total_allocated != num_classes:
        raise ValueError(
            f"F1 band allocation conservation violation: allocated {total_allocated} classes "
            f"but num_classes is {num_classes}"
        )

    for k, b in bands.items():
        b["percent"] = round(100.0 * b["count"] / num_classes, 2)
        if confidences is not None and class_mean_conf is not None:
            c_indices = band_class_indices[k]
            if c_indices:
                b["mean_confidence"] = round(float(np.mean(class_mean_conf[c_indices])), 4)
            else:
                b["mean_confidence"] = 0.0

    return {
        "total_classes": num_classes,
        "bands": bands,
    }


def compute_species_specialist_comparison(
    y_true: Union[np.ndarray, List[int]],
    preds_resnet: Union[np.ndarray, List[int]],
    preds_convnext: Union[np.ndarray, List[int]],
    confs_resnet: Optional[Union[np.ndarray, List[float]]] = None,
    confs_convnext: Optional[Union[np.ndarray, List[float]]] = None,
    species_names: Optional[Dict[int, str]] = None,
    num_classes: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Computes per-species specialist comparison between ResNet50 and ConvNeXt models.
    Matches the historical logic from long/cnn-long-voter-compare.ipynb (cell 16)
    and resolves Table 4 / Note 45 tie accounting:
      - convnext_superior: ConvNeXt has strictly more correct predictions than ResNet50
      - resnet_superior: ResNet50 has strictly more correct predictions than ConvNeXt
      - ties: both models achieved equal correct predictions (> 0)
      - both_zero: both models failed completely (0 correct predictions)

    Conservation invariant:
      convnext_superior + resnet_superior + ties + both_zero == total_evaluated_species
    """
    y_true_np = np.asarray(y_true, dtype=np.int64)
    p_res = np.asarray(preds_resnet, dtype=np.int64)
    p_conv = np.asarray(preds_convnext, dtype=np.int64)

    n_samples = len(y_true_np)
    if len(p_res) != n_samples or len(p_conv) != n_samples:
        raise ValueError(
            f"Length mismatch: y_true ({n_samples}), preds_resnet ({len(p_res)}), preds_convnext ({len(p_conv)})"
        )

    total_classes = (
        num_classes
        if num_classes is not None
        else max(int(np.max(y_true_np)), int(np.max(p_res)), int(np.max(p_conv))) + 1
    )

    c_res = (
        np.asarray(confs_resnet, dtype=np.float64)
        if confs_resnet is not None
        else None
    )
    c_conv = (
        np.asarray(confs_convnext, dtype=np.float64)
        if confs_convnext is not None
        else None
    )

    # Per-class F1
    f1_res, _, _ = compute_f1_scores(y_true_np, p_res, total_classes)
    f1_conv, _, _ = compute_f1_scores(y_true_np, p_conv, total_classes)

    # Vectorized per-species aggregations in O(N)
    valid_mask = (y_true_np >= 0) & (y_true_np < total_classes)
    valid_y = y_true_np[valid_mask]
    sp_counts = np.bincount(valid_y, minlength=total_classes)[:total_classes]

    res_match = valid_mask & (p_res == y_true_np)
    conv_match = valid_mask & (p_conv == y_true_np)
    res_correct_counts = np.bincount(y_true_np[res_match], minlength=total_classes)[:total_classes]
    conv_correct_counts = np.bincount(y_true_np[conv_match], minlength=total_classes)[:total_classes]

    res_conf_sums = (
        np.bincount(valid_y, weights=c_res[valid_mask], minlength=total_classes)[:total_classes]
        if c_res is not None
        else None
    )
    conv_conf_sums = (
        np.bincount(valid_y, weights=c_conv[valid_mask], minlength=total_classes)[:total_classes]
        if c_conv is not None
        else None
    )

    unique_species = np.flatnonzero(sp_counts > 0)

    convnext_wins = 0
    resnet_wins = 0
    ties = 0
    both_zero = 0

    per_species_list: List[Dict[str, Any]] = []

    for sp_id_raw in unique_species:
        sp_id = int(sp_id_raw)
        n_sp = int(sp_counts[sp_id])
        if n_sp == 0:
            continue

        acertos_res = int(res_correct_counts[sp_id])
        acertos_conv = int(conv_correct_counts[sp_id])

        acc_res = float(acertos_res / n_sp)
        acc_conv = float(acertos_conv / n_sp)

        conf_res = float(res_conf_sums[sp_id] / n_sp) if res_conf_sums is not None else None
        conf_conv = float(conv_conf_sums[sp_id] / n_sp) if conv_conf_sums is not None else None

        if acertos_conv > acertos_res:
            winner = "convnext_superior"
            convnext_wins += 1
        elif acertos_res > acertos_conv:
            winner = "resnet_superior"
            resnet_wins += 1
        elif acertos_res > 0 and acertos_res == acertos_conv:
            winner = "ties"
            ties += 1
        else:
            winner = "both_zero"
            both_zero += 1

        sp_name = (
            species_names.get(sp_id, f"Species_{sp_id}")
            if species_names
            else f"Species_{sp_id}"
        )

        per_species_list.append({
            "species_id": int(sp_id),
            "species_name": sp_name,
            "samples": n_sp,
            "correct_resnet": acertos_res,
            "correct_convnext": acertos_conv,
            "accuracy_resnet": round(acc_res, 4),
            "accuracy_convnext": round(acc_conv, 4),
            "f1_resnet": round(float(f1_res[sp_id]), 6),
            "f1_convnext": round(float(f1_conv[sp_id]), 6),
            "confidence_resnet": round(conf_res, 4) if conf_res is not None else None,
            "confidence_convnext": round(conf_conv, 4) if conf_conv is not None else None,
            "winner": winner,
        })

    total_evaluated = len(unique_species)
    assert (
        convnext_wins + resnet_wins + ties + both_zero == total_evaluated
    ), "Specialist comparison category conservation failed"

    return {
        "convnext_superior": convnext_wins,
        "resnet_superior": resnet_wins,
        "ties": ties,
        "both_zero": both_zero,
        "total_evaluated_species": total_evaluated,
        "per_species": per_species_list,
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
