"""
Confusion matrix analysis, top-K off-diagonal pair extraction, and sample collection.

Scientific invariants (AGENTS.md, AD-002):
- Model labels are 0-based integers: [0 .. NUM_CLASSES - 1].
- CSV taxonomic IDs are 1-based integers: [1 .. NUM_CLASSES].
- model_label = taxonomic_id - 1
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader

from viral_cnn.taxonomy.mapping import load_taxonomy_mapping, model_label_to_taxonomic_id

logger = logging.getLogger(__name__)

ConfusedPairInfo = Tuple[float, int, int]


def compute_and_plot_confusion_insights(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    num_classes: int,
    top_k_confusion: int = 10,
    mapping_csv_path: Optional[str] = None,
    col_name: str = "species",
    show: bool = False,
) -> List[ConfusedPairInfo]:
    """
    Computes row-normalized confusion matrix and identifies Top-K most confused class pairs.

    Returns:
        List of tuples: [(confusion_score, true_label_0based, pred_label_0based), ...]
    """
    model.eval()
    all_labels: List[int] = []
    all_preds: List[int] = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1)

            all_labels.extend(labels.cpu().tolist())
            all_preds.extend(preds.cpu().tolist())

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)

    cm = confusion_matrix(y_true, y_pred, labels=range(num_classes))
    row_sums = cm.sum(axis=1)[:, np.newaxis]
    row_sums[row_sums == 0] = 1.0  # avoid division by zero
    cm_normalized = cm.astype(float) / row_sums

    confused_pairs: List[ConfusedPairInfo] = []
    for i in range(num_classes):
        for j in range(num_classes):
            if i != j:
                score = float(cm_normalized[i, j])
                if score > 0.0:
                    confused_pairs.append((score, i, j))

    confused_pairs.sort(key=lambda x: x[0], reverse=True)
    top_k_confused = confused_pairs[:top_k_confusion]

    if show:
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        im = axes[0].imshow(cm_normalized, interpolation="nearest", cmap=plt.cm.Blues)
        axes[0].set_title("Normalized Confusion Matrix")
        fig.colorbar(im, ax=axes[0])
        axes[0].set_xlabel("Predicted Label")
        axes[0].set_ylabel("True Label")

        labels_bar = []
        scores_bar = []
        mapping = {}
        if mapping_csv_path and os.path.exists(mapping_csv_path):
            mapping = load_taxonomy_mapping(mapping_csv_path, rank_column=col_name)

        for score, i, j in top_k_confused:
            true_label = f"{i} - {mapping.get(i, f'ID {i}')}"
            pred_label = f"{j} - {mapping.get(j, f'ID {j}')}"
            labels_bar.append(f"True: {true_label} -> Pred: {pred_label}")
            scores_bar.append(score)

        y_pos = np.arange(len(labels_bar))
        axes[1].barh(y_pos, scores_bar, align="center", color="darkred")
        axes[1].set_yticks(y_pos)
        axes[1].set_yticklabels(labels_bar, fontsize=8)
        axes[1].invert_yaxis()
        axes[1].set_xlabel("Confusion Rate (Normalized)")
        axes[1].set_title(f"Top {top_k_confusion} Confused Class Pairs")
        plt.tight_layout()
        plt.show()

    return top_k_confused


# Backward-compatible alias
plot_confusion_matrix_insights = compute_and_plot_confusion_insights


def get_class_label(mapping_csv_path: str, target_label_0based: int, col_name: str = "species") -> str:
    """Retrieves taxon name from CSV for a 0-based model label."""
    if not os.path.exists(mapping_csv_path):
        return "Not Found"
    mapping = load_taxonomy_mapping(mapping_csv_path, rank_column=col_name)
    return mapping.get(target_label_0based, "Not Found")


def inspect_csv(mapping_csv_path: str, target_label_0based: int, col_name: str = "species") -> None:
    """Prints diagnostic information for a confused class."""
    if not os.path.exists(mapping_csv_path):
        print(f"File not found: {mapping_csv_path}")
        return
    df = pd.read_csv(mapping_csv_path)
    target_id_1based = model_label_to_taxonomic_id(target_label_0based)
    row = df[df["id"] == target_id_1based]
    if not row.empty:
        print(f"Inspection for Model Label {target_label_0based} (CSV ID {target_id_1based}):")
        print(row.to_string())
    else:
        print(f"No entry found in CSV for ID: {target_id_1based}")


def get_confused_class_samples(
    confused_pairs_info: List[ConfusedPairInfo],
    test_loader: DataLoader,
    device: torch.device,
    n_examples_per_class: int = 5,
) -> List[Dict[str, Any]]:
    """
    Collects up to n_examples_per_class input tensors for each unique true_id in confused pairs.
    """
    target_classes_map: Dict[int, Dict[str, Any]] = {}
    for score, true_id, pred_id in confused_pairs_info:
        true_id_int = int(true_id)
        pred_id_int = int(pred_id)
        if true_id_int not in target_classes_map:
            target_classes_map[true_id_int] = {
                "id_true": true_id_int,
                "total_confused_score": score,
                "examples": [],
                "target_pred_id": pred_id_int,
                "score": score,
            }

    counts = {cid: 0 for cid in target_classes_map}
    all_targets_met = False

    with torch.no_grad():
        for inputs, labels in test_loader:
            if all_targets_met:
                break
            inputs_cpu = inputs.cpu()
            labels_cpu = labels.cpu()

            for idx in range(inputs_cpu.size(0)):
                true_lbl = int(labels_cpu[idx].item())
                if true_lbl in target_classes_map and counts[true_lbl] < n_examples_per_class:
                    target_classes_map[true_lbl]["examples"].append(inputs_cpu[idx])
                    counts[true_lbl] += 1
                    if all(c >= n_examples_per_class for c in counts.values()):
                        all_targets_met = True
                        break

    final_output: List[Dict[str, Any]] = []
    for cid, data in target_classes_map.items():
        final_output.append({
            "score_principal_confusao": data["score"],
            "class_true": {
                "id": cid,
                "target_pred_id": data["target_pred_id"],
                "examples": data["examples"],
            },
        })
    return final_output

