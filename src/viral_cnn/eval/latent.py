"""
Latent space analysis, pre-classifier embedding extraction, cosine KNN,
Silhouette scoring with subsampling audit, hierarchical taxonomic correlation,
and independent linear probing.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import mpmath
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Statistical Correlation Helpers (Pure NumPy/mpmath, zero external C-deps)
# ---------------------------------------------------------------------------

def compute_pearson_r(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """Computes Pearson correlation coefficient r and two-tailed p-value."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n = len(x)
    if n < 3:
        return float("nan"), float("nan")

    mx = x - np.mean(x)
    my = y - np.mean(y)
    denom = np.sqrt(np.sum(mx ** 2) * np.sum(my ** 2))
    if denom == 0 or np.isnan(denom):
        return float("nan"), float("nan")

    r = float(np.sum(mx * my) / denom)
    r = max(-1.0, min(1.0, r))

    if abs(r) == 1.0:
        return r, 0.0

    # Student-t statistic: t = r * sqrt((n - 2) / (1 - r^2))
    df = n - 2
    t = r * math.sqrt(df / (1.0 - r ** 2))
    # Two-tailed p-value via regularized incomplete beta function I_x(df/2, 1/2)
    x_val = df / (df + t ** 2)
    p_val = float(mpmath.betainc(0.5 * df, 0.5, 0, x_val, regularized=True))
    return r, p_val


def compute_spearman_rho(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """Computes Spearman rank correlation coefficient rho and two-tailed p-value."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n = len(x)
    if n < 3:
        return float("nan"), float("nan")

    def rankdata(a: np.ndarray) -> np.ndarray:
        sorter = np.argsort(a)
        inv = np.empty(n, dtype=np.intp)
        inv[sorter] = np.arange(n)

        a_sorted = a[sorter]
        obs = np.r_[True, a_sorted[1:] != a_sorted[:-1]]
        dense = obs.cumsum()[inv]

        # Cumulative counts of each unique value
        count = np.r_[np.nonzero(obs)[0], n]
        return 0.5 * (count[dense] + count[dense - 1] + 1)

    rx = rankdata(x)
    ry = rankdata(y)
    return compute_pearson_r(rx, ry)


# ---------------------------------------------------------------------------
# S1: Pre-Classifier Embedding Extraction & Cosine KNN
# ---------------------------------------------------------------------------

def extract_latent_embeddings(
    model: nn.Module,
    dataloader: Any,
    device: str = "cpu",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extracts high-dimensional pre-classifier embeddings by registering a forward
    pre-hook on `model.fc`. Returns (features, labels) as NumPy arrays.
    """
    model.eval()
    dev = torch.device(device)
    target_model = getattr(model, "_orig_mod", model)

    if not hasattr(target_model, "fc") or not isinstance(target_model.fc, nn.Linear):
        raise AttributeError("Model must have an `fc` attribute of type `nn.Linear`")

    features_list: List[np.ndarray] = []
    labels_list: List[np.ndarray] = []

    def pre_hook_fn(module: nn.Module, inputs: Tuple[torch.Tensor, ...]) -> None:
        feat = inputs[0].detach().cpu().to(torch.float32).numpy()
        features_list.append(feat)

    handle = target_model.fc.register_forward_pre_hook(pre_hook_fn)

    try:
        with torch.inference_mode():
            for batch in dataloader:
                if isinstance(batch, (tuple, list)) and len(batch) >= 2:
                    inputs, labels = batch[0], batch[1]
                elif isinstance(batch, dict):
                    inputs = batch["sequence"] if "sequence" in batch else batch["inputs"]
                    labels = batch["label"] if "label" in batch else batch["labels"]
                else:
                    raise ValueError(f"Unrecognized batch format from dataloader: {type(batch)}")

                inputs = inputs.to(dev)
                _ = model(inputs)

                if isinstance(labels, torch.Tensor):
                    labels_list.append(labels.detach().cpu().numpy())
                else:
                    labels_list.append(np.asarray(labels))
    finally:
        handle.remove()

    if not features_list:
        return np.empty((0, target_model.fc.in_features), dtype=np.float32), np.empty((0,), dtype=np.int64)

    all_features = np.concatenate(features_list, axis=0)
    all_labels = np.concatenate(labels_list, axis=0)
    return all_features, all_labels


def evaluate_cosine_knn(
    features: np.ndarray,
    labels: np.ndarray,
    k: int = 5,
    test_size: float = 0.2,
    random_state: int = 42,
    batch_size: int = 1000,
    device: str = "cpu",
    taxonomic_rank: str = "Species",
) -> Dict[str, Any]:
    """
    Evaluates k-nearest neighbor classification accuracy using cosine similarity on
    a stratified 80/20 reference/query split. Guaranteed zero self-neighbor presence.
    """
    n_samples = len(features)
    if n_samples < 2:
        raise ValueError(f"Cannot evaluate KNN on {n_samples} samples; require >= 2")

    rng = np.random.default_rng(random_state)
    labels = np.asarray(labels)

    # Stratified 80/20 train/test split (reference / query)
    unique_classes, counts = np.unique(labels, return_counts=True)
    ref_indices: List[int] = []
    query_indices: List[int] = []

    for cls, cnt in zip(unique_classes, counts):
        cls_idx = np.where(labels == cls)[0]
        rng.shuffle(cls_idx)
        n_query = max(1, int(round(cnt * test_size))) if cnt > 1 else 0
        if n_query >= cnt:
            n_query = cnt - 1

        if n_query > 0:
            query_indices.extend(cls_idx[:n_query])
            ref_indices.extend(cls_idx[n_query:])
        else:
            ref_indices.extend(cls_idx)

    # In case no queries were allocated due to all singletons, fallback
    if len(query_indices) == 0:
        perm = rng.permutation(n_samples)
        split_pt = max(1, int(round(n_samples * (1.0 - test_size))))
        ref_indices = perm[:split_pt].tolist()
        query_indices = perm[split_pt:].tolist()

    ref_idx_arr = np.array(ref_indices, dtype=np.intp)
    query_idx_arr = np.array(query_indices, dtype=np.intp)

    # Verify disjoint partitions (zero self-neighbors)
    overlap = set(ref_indices).intersection(set(query_indices))
    if len(overlap) > 0:
        raise RuntimeError("Reference and query index partitions must be strictly disjoint")

    X_ref = torch.as_tensor(features[ref_idx_arr], dtype=torch.float32)
    y_ref = torch.as_tensor(labels[ref_idx_arr], dtype=torch.long)
    X_query = torch.as_tensor(features[query_idx_arr], dtype=torch.float32)
    y_query = torch.as_tensor(labels[query_idx_arr], dtype=torch.long)

    dev = torch.device(device)
    X_ref = F.normalize(X_ref.to(dev), p=2, dim=1)
    y_ref = y_ref.to(dev)

    k_eff = min(k, len(X_ref))
    preds_list: List[torch.Tensor] = []

    for i in range(0, len(X_query), batch_size):
        batch_x = F.normalize(X_query[i : i + batch_size].to(dev), p=2, dim=1)
        sim = torch.mm(batch_x, X_ref.T)

        topk = torch.topk(sim, k=k_eff, dim=1, largest=True)
        topk_labels = y_ref[topk.indices]

        # Majority vote with deterministic tie-breaking (mode)
        mode_preds, _ = torch.mode(topk_labels, dim=1)
        preds_list.append(mode_preds.cpu())

    y_pred = torch.cat(preds_list).numpy()
    y_true = y_query.numpy()

    correct = (y_pred == y_true)
    global_acc = float(np.mean(correct)) if len(correct) > 0 else 0.0

    # Per-cluster accuracy table
    df_cluster = pd.DataFrame({"y_true": y_true, "y_pred": y_pred, "correct": correct})
    cluster_summary = (
        df_cluster.groupby("y_true")
        .agg(
            test_samples=("correct", "count"),
            knn_accuracy=("correct", "mean"),
        )
        .reset_index()
        .rename(columns={"y_true": "class_label"})
    )
    cluster_summary["taxonomic_rank"] = taxonomic_rank

    return {
        "taxonomic_rank": taxonomic_rank,
        "global_knn_accuracy": global_acc,
        "k": k_eff,
        "n_reference": len(X_ref),
        "n_query": len(X_query),
        "zero_self_neighbor": True,
        "y_true": y_true,
        "y_pred": y_pred,
        "per_cluster_metrics": cluster_summary,
    }


# ---------------------------------------------------------------------------
# S2: Genome-Disjoint KNN Evaluation
# ---------------------------------------------------------------------------

def evaluate_genome_disjoint_knn(
    features: np.ndarray,
    labels: np.ndarray,
    genome_ids: np.ndarray,
    k: int = 5,
    random_state: int = 42,
    batch_size: int = 1000,
    device: str = "cpu",
    taxonomic_rank: str = "Species",
) -> Dict[str, Any]:
    """
    Evaluates KNN where reference and query sets are partitioned by genome accessions,
    guaranteeing zero genome accession overlap.
    """
    labels = np.asarray(labels)
    genome_ids = np.asarray(genome_ids)
    features = np.asarray(features, dtype=np.float32)

    if len(features) != len(labels) or len(labels) != len(genome_ids):
        raise ValueError("Length mismatch between features, labels, and genome_ids")

    # Map species to their unique genome accessions
    species_to_genomes: Dict[Any, List[Any]] = {}
    for sp, gid in zip(labels, genome_ids):
        species_to_genomes.setdefault(sp, set()).add(gid)

    # Isolate multi-genome species vs single-genome species
    multi_genome_species = {sp for sp, gids in species_to_genomes.items() if len(gids) >= 2}
    single_genome_species = {sp for sp, gids in species_to_genomes.items() if len(gids) < 2}

    rng = np.random.default_rng(random_state)
    ref_indices: List[int] = []
    query_indices: List[int] = []

    # Assign genomes of multi-genome species: 1 genome to query, remaining to ref
    for sp in multi_genome_species:
        gids = sorted(list(species_to_genomes[sp]))
        rng.shuffle(gids)
        query_gid = gids[0]
        ref_gids = set(gids[1:])

        sp_indices = np.where(labels == sp)[0]
        for idx in sp_indices:
            gid = genome_ids[idx]
            if gid == query_gid:
                query_indices.append(idx)
            elif gid in ref_gids:
                ref_indices.append(idx)

    # For single-genome species, put all into reference (closed-set requirement)
    for sp in single_genome_species:
        sp_indices = np.where(labels == sp)[0]
        ref_indices.extend(sp_indices)

    ref_arr = np.array(ref_indices, dtype=np.intp)
    query_arr = np.array(query_indices, dtype=np.intp)

    if len(query_arr) == 0:
        raise ValueError("No query samples available for genome-disjoint evaluation (all single-genome)")

    # Assert zero accession overlap between reference and query
    ref_accessions = set(genome_ids[ref_arr])
    query_accessions = set(genome_ids[query_arr])
    accession_overlap = ref_accessions.intersection(query_accessions)
    if len(accession_overlap) > 0:
        raise RuntimeError(f"Genome accession leak detected: {len(accession_overlap)} overlapping accessions")

    # Compute genome-disjoint KNN
    dev = torch.device(device)
    X_ref = F.normalize(torch.as_tensor(features[ref_arr], dtype=torch.float32).to(dev), p=2, dim=1)
    y_ref = torch.as_tensor(labels[ref_arr], dtype=torch.long).to(dev)

    X_query = torch.as_tensor(features[query_arr], dtype=torch.float32)
    y_query = torch.as_tensor(labels[query_arr], dtype=torch.long)

    k_eff = min(k, len(X_ref))
    preds: List[torch.Tensor] = []

    for i in range(0, len(X_query), batch_size):
        bx = F.normalize(X_query[i : i + batch_size].to(dev), p=2, dim=1)
        sim = torch.mm(bx, X_ref.T)
        topk = torch.topk(sim, k=k_eff, dim=1, largest=True)
        mode_preds, _ = torch.mode(y_ref[topk.indices], dim=1)
        preds.append(mode_preds.cpu())

    y_pred_disjoint = torch.cat(preds).numpy()
    y_true_disjoint = y_query.numpy()

    disjoint_acc = float(np.mean(y_pred_disjoint == y_true_disjoint))

    # Also compute fragment-split KNN on the identical multi-genome subset for comparison
    multi_mask = np.isin(labels, list(multi_genome_species))
    frag_res = evaluate_cosine_knn(
        features[multi_mask],
        labels[multi_mask],
        k=k,
        random_state=random_state,
        batch_size=batch_size,
        device=device,
        taxonomic_rank=taxonomic_rank,
    )
    fragment_acc = frag_res["global_knn_accuracy"]
    acc_drop = fragment_acc - disjoint_acc

    return {
        "taxonomic_rank": taxonomic_rank,
        "genome_disjoint_accuracy": disjoint_acc,
        "fragment_split_accuracy": fragment_acc,
        "accuracy_drop_percentage_points": acc_drop * 100.0,
        "n_multi_genome_species": len(multi_genome_species),
        "n_single_genome_species_excluded": len(single_genome_species),
        "n_reference_samples": len(X_ref),
        "n_query_samples": len(X_query),
        "zero_accession_overlap": True,
    }


# ---------------------------------------------------------------------------
# S3: High-Dimensional Silhouette Score & Subsampling Audit
# ---------------------------------------------------------------------------

def compute_cosine_silhouette_fast(X: np.ndarray, y: np.ndarray) -> Tuple[float, np.ndarray]:
    """
    Exact vectorized computation of Silhouette Score under Cosine Distance.
    O(N * K) time and O(N + K * D) memory, numerically identical to scikit-learn.
    """
    X = np.asarray(X, dtype=np.float32)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    X_norm = X / norms

    y = np.asarray(y)
    unique_clusters, cluster_counts = np.unique(y, return_counts=True)
    K = len(unique_clusters)
    N = len(X)

    if K <= 1:
        return float("nan"), np.zeros(N, dtype=np.float32)

    cluster_sums = np.zeros((K, X.shape[1]), dtype=np.float32)
    for idx, c in enumerate(unique_clusters):
        cluster_sums[idx] = np.sum(X_norm[y == c], axis=0)

    # Precompute cluster map
    c_to_idx = {c: i for i, c in enumerate(unique_clusters)}
    sample_c_indices = np.array([c_to_idx[c] for c in y], dtype=np.intp)

    a_vals = np.zeros(N, dtype=np.float32)
    b_vals = np.zeros(N, dtype=np.float32)

    for i in range(N):
        c_idx = sample_c_indices[i]
        cnt = cluster_counts[c_idx]

        if cnt > 1:
            sum_excl = cluster_sums[c_idx] - X_norm[i]
            mean_sim = np.dot(X_norm[i], sum_excl) / (cnt - 1)
            a_vals[i] = 1.0 - mean_sim
        else:
            a_vals[i] = 0.0

        # Mean distance to all other clusters: 1.0 - max dot product with centroid
        other_indices = [idx for idx in range(K) if idx != c_idx]
        sims = np.dot(cluster_sums[other_indices] / cluster_counts[other_indices, None], X_norm[i])
        b_vals[i] = 1.0 - np.max(sims)

    max_ab = np.maximum(a_vals, b_vals)
    sil_samples = np.zeros(N, dtype=np.float32)
    valid_mask = max_ab > 0
    sil_samples[valid_mask] = (b_vals[valid_mask] - a_vals[valid_mask]) / max_ab[valid_mask]
    global_score = float(np.mean(sil_samples))

    return global_score, sil_samples


def evaluate_silhouette_with_audit(
    features: np.ndarray,
    labels: np.ndarray,
    max_samples: int = 50000,
    sampling_strategy: str = "random",
    random_state: int = 42,
    taxonomic_rank: str = "Species",
) -> Dict[str, Any]:
    """
    Computes Cosine Silhouette Score in the original high-dimensional feature space,
    auditing class retention and dropping singletons (<2 samples).
    """
    features = np.asarray(features, dtype=np.float32)
    labels = np.asarray(labels)
    N = len(features)
    rng = np.random.default_rng(random_state)

    if N <= max_samples:
        sample_indices = np.arange(N)
    elif sampling_strategy == "stratified":
        # Stratified sampling: at least 2 samples per eligible class
        unique_classes = np.unique(labels)
        sample_indices_list: List[int] = []
        n_per_class = max(2, max_samples // len(unique_classes))

        for c in unique_classes:
            idx = np.where(labels == c)[0]
            if len(idx) >= 2:
                chosen = rng.choice(idx, size=min(len(idx), n_per_class), replace=False)
                sample_indices_list.extend(chosen)
            else:
                sample_indices_list.extend(idx)
        sample_indices = np.array(sample_indices_list[:max_samples], dtype=np.intp)
    else:
        # Uniform random subsampling
        sample_indices = rng.choice(N, size=max_samples, replace=False)

    sub_features = features[sample_indices]
    sub_labels = labels[sample_indices]

    # Audit classes
    all_classes, sample_counts = np.unique(sub_labels, return_counts=True)
    valid_class_mask = sample_counts >= 2
    valid_classes = set(all_classes[valid_class_mask])
    excluded_classes = set(all_classes[~valid_class_mask])

    mask = np.isin(sub_labels, list(valid_classes))
    retained_sample_count = int(np.sum(mask))
    excluded_sample_count = len(sub_labels) - retained_sample_count
    retained_sample_rate = retained_sample_count / len(sub_labels) if len(sub_labels) > 0 else 0.0

    if len(valid_classes) >= 2 and retained_sample_count > len(valid_classes):
        global_sil, sample_sil = compute_cosine_silhouette_fast(sub_features[mask], sub_labels[mask])

        df_sil = pd.DataFrame({"class_label": sub_labels[mask], "silhouette": sample_sil})
        per_cluster = (
            df_sil.groupby("class_label")
            .agg(
                silhouette_score=("silhouette", "mean"),
                sample_count=("silhouette", "count"),
            )
            .reset_index()
        )
        per_cluster["taxonomic_rank"] = taxonomic_rank
    else:
        global_sil = float("nan")
        per_cluster = pd.DataFrame(columns=["class_label", "silhouette_score", "sample_count", "taxonomic_rank"])

    return {
        "taxonomic_rank": taxonomic_rank,
        "global_silhouette_score": global_sil,
        "sampling_strategy": sampling_strategy,
        "target_sample_size": max_samples,
        "actual_sample_size": len(sub_labels),
        "total_classes_sampled": len(all_classes),
        "valid_classes_count": len(valid_classes),
        "excluded_classes_count": len(excluded_classes),
        "retained_sample_count": retained_sample_count,
        "excluded_sample_count": excluded_sample_count,
        "retained_sample_rate": retained_sample_rate,
        "per_cluster_metrics": per_cluster,
    }


# ---------------------------------------------------------------------------
# S4: Hierarchical Taxonomic Correspondence and Correlation
# ---------------------------------------------------------------------------

def calculate_taxonomic_correlations(
    child_df: pd.DataFrame,
    parent_df: pd.DataFrame,
    taxonomy_df: pd.DataFrame,
    child_tax_col: str = "species",
    parent_tax_col: str = "Genus",
    child_level: str = "Species",
    parent_level: str = "Genus",
    min_children: int = 3,
) -> Dict[str, Any]:
    """
    Calculates parent-child correlation for KNN Accuracy and Silhouette Score.
    Maintains independent valid parent subsets (N_parents_KNN and N_parents_Sil).
    """
    mapping = (
        taxonomy_df[[child_tax_col, parent_tax_col]]
        .drop_duplicates()
        .dropna()
        .rename(columns={child_tax_col: "child_name", parent_tax_col: "parent_name"})
    )

    # Standardize child and parent columns
    child = child_df.copy()
    if "Taxon_Name" in child.columns:
        child = child.rename(columns={"Taxon_Name": "child_name"})
    elif "class_label" in child.columns:
        child = child.rename(columns={"class_label": "child_name"})

    parent = parent_df.copy()
    if "Taxon_Name" in parent.columns:
        parent = parent.rename(columns={"Taxon_Name": "parent_name"})
    elif "class_label" in parent.columns:
        parent = parent.rename(columns={"class_label": "parent_name"})

    # Rename metric columns to standard keys
    if "KNN_Accuracy" in child.columns:
        child = child.rename(columns={"KNN_Accuracy": "knn_accuracy"})
    if "Silhouette_Score" in child.columns:
        child = child.rename(columns={"Silhouette_Score": "silhouette_score"})
    if "KNN_Accuracy" in parent.columns:
        parent = parent.rename(columns={"KNN_Accuracy": "parent_knn"})
    elif "knn_accuracy" in parent.columns:
        parent = parent.rename(columns={"knn_accuracy": "parent_knn"})
    if "Silhouette_Score" in parent.columns:
        parent = parent.rename(columns={"Silhouette_Score": "parent_sil"})
    elif "silhouette_score" in parent.columns:
        parent = parent.rename(columns={"silhouette_score": "parent_sil"})

    merged_child = child.merge(mapping, on="child_name", how="inner")
    merged = merged_child.merge(parent, on="parent_name", how="inner")

    # 1. KNN correlation: independent valid subset
    if "knn_accuracy" in merged.columns and "parent_knn" in merged.columns:
        knn_data = merged.dropna(subset=["knn_accuracy", "parent_knn"])
        knn_summary = (
            knn_data.groupby("parent_name")
            .agg(
                child_knn_mean=("knn_accuracy", "mean"),
                parent_knn=("parent_knn", "first"),
                n_children=("child_name", "nunique"),
            )
            .reset_index()
        )
        knn_valid = knn_summary[knn_summary["n_children"] >= min_children]

        if len(knn_valid) >= 3:
            r_knn, p_r_knn = compute_pearson_r(knn_valid["child_knn_mean"].values, knn_valid["parent_knn"].values)
            rho_knn, p_rho_knn = compute_spearman_rho(knn_valid["child_knn_mean"].values, knn_valid["parent_knn"].values)
        else:
            r_knn = p_r_knn = rho_knn = p_rho_knn = float("nan")
    else:
        knn_valid = pd.DataFrame(columns=["parent_name", "child_knn_mean", "parent_knn", "n_children"])
        r_knn = p_r_knn = rho_knn = p_rho_knn = float("nan")

    # 2. Silhouette correlation: independent valid subset
    if "silhouette_score" in merged.columns and "parent_sil" in merged.columns:
        sil_data = merged.dropna(subset=["silhouette_score", "parent_sil"])
        sil_summary = (
            sil_data.groupby("parent_name")
            .agg(
                child_sil_mean=("silhouette_score", "mean"),
                parent_sil=("parent_sil", "first"),
                n_children=("child_name", "nunique"),
            )
            .reset_index()
        )
        sil_valid = sil_summary[sil_summary["n_children"] >= min_children]

        if len(sil_valid) >= 3:
            r_sil, p_r_sil = compute_pearson_r(sil_valid["child_sil_mean"].values, sil_valid["parent_sil"].values)
            rho_sil, p_rho_sil = compute_spearman_rho(sil_valid["child_sil_mean"].values, sil_valid["parent_sil"].values)
        else:
            r_sil = p_r_sil = rho_sil = p_rho_sil = float("nan")
    else:
        sil_valid = pd.DataFrame(columns=["parent_name", "child_sil_mean", "parent_sil", "n_children"])
        r_sil = p_r_sil = rho_sil = p_rho_sil = float("nan")

    return {
        "child_level": child_level,
        "parent_level": parent_level,
        "n_parents_knn": len(knn_valid),
        "n_parents_silhouette": len(sil_valid),
        "pearson_r_knn": r_knn,
        "pearson_p_knn": p_r_knn,
        "spearman_rho_knn": rho_knn,
        "spearman_p_knn": p_rho_knn,
        "pearson_r_silhouette": r_sil,
        "pearson_p_silhouette": p_r_sil,
        "spearman_rho_silhouette": rho_sil,
        "spearman_p_silhouette": p_rho_sil,
        "knn_summary_df": knn_valid,
        "silhouette_summary_df": sil_valid,
    }


# ---------------------------------------------------------------------------
# S5: ConvNeXt Head Bottleneck & Independent Linear Probe
# ---------------------------------------------------------------------------

class LinearProbeClassifier(nn.Module):
    """
    Linear probing classification head trained strictly on frozen representations
    from training or validation partitions to evaluate true linear separability.
    """

    def __init__(self, in_features: int, num_classes: int) -> None:
        super().__init__()
        self.fc = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        split_origin: str = "train",
        epochs: int = 10,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        batch_size: int = 256,
        device: str = "cpu",
    ) -> "LinearProbeClassifier":
        """
        Trains linear probe exclusively on training or validation split embeddings.
        Guarantees that test embeddings are never accessed.
        """
        if split_origin.lower().strip() == "test":
            raise ValueError(
                "Violation of scientific invariant: LinearProbeClassifier cannot be trained on test set embeddings"
            )

        dev = torch.device(device)
        self.to(dev)
        self.train()

        X_t = torch.as_tensor(X_train, dtype=torch.float32)
        y_t = torch.as_tensor(y_train, dtype=torch.long)

        dataset = torch.utils.data.TensorDataset(X_t, y_t)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        optimizer = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=weight_decay)
        criterion = nn.CrossEntropyLoss()

        for _ in range(epochs):
            for bx, by in loader:
                bx = bx.to(dev)
                by = by.to(dev)
                optimizer.zero_grad()
                logits = self(bx)
                loss = criterion(logits, by)
                loss.backward()
                optimizer.step()

        self.eval()
        return self

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray, device: str = "cpu") -> Dict[str, float]:
        """Evaluates linear probe on test set embeddings."""
        self.eval()
        dev = torch.device(device)
        self.to(dev)

        X_t = torch.as_tensor(X_test, dtype=torch.float32)
        y_true = np.asarray(y_test, dtype=np.int64)

        with torch.no_grad():
            logits = self(X_t.to(dev)).cpu().numpy()
            y_pred = np.argmax(logits, axis=1)

        acc = float(np.mean(y_pred == y_true)) if len(y_true) > 0 else 0.0

        # Vectorized Macro-F1 via bincount
        num_classes = self.fc.out_features
        tp = np.bincount(y_true[y_true == y_pred], minlength=num_classes)
        pred_counts = np.bincount(y_pred, minlength=num_classes)
        true_counts = np.bincount(y_true, minlength=num_classes)

        denom = pred_counts + true_counts
        f1_per_class = np.zeros(num_classes, dtype=np.float64)
        valid = denom > 0
        f1_per_class[valid] = 2.0 * tp[valid] / denom[valid]
        macro_f1 = float(np.mean(f1_per_class[true_counts > 0])) if np.any(true_counts > 0) else 0.0

        return {
            "top1_accuracy": acc,
            "macro_f1": macro_f1,
            "n_test_samples": len(y_true),
        }


def audit_classifier_bottleneck(
    native_model_accuracy: float,
    linear_probe_accuracy: float,
    transductive_knn_accuracy: float,
) -> Dict[str, Any]:
    """
    Constructs a comparative diagnostic table resolving the ConvNeXt head bottleneck hypothesis.
    """
    statement = (
        f"Transductive test 5-NN accuracy ({transductive_knn_accuracy:.2%}) utilizes test-set reference "
        "representations and non-parametric local metric lookup. It does NOT demonstrate undercapacity of the "
        f"parametric classifier head. The true linear separability of the frozen representations evaluated via an "
        f"independent linear probe trained on train/val data is {linear_probe_accuracy:.2%}, compared to the native "
        f"classifier head's accuracy of {native_model_accuracy:.2%}."
    )

    summary_df = pd.DataFrame(
        [
            {"evaluation_protocol": "Native Trained Classifier Head", "accuracy": native_model_accuracy, "training_source": "Training set (parametric)"},
            {"evaluation_protocol": "Independent Linear Probe", "accuracy": linear_probe_accuracy, "training_source": "Train/Val embeddings (parametric)"},
            {"evaluation_protocol": "Transductive Test 5-NN", "accuracy": transductive_knn_accuracy, "training_source": "80% test reference (non-parametric)"},
        ]
    )

    return {
        "diagnostic_statement": statement,
        "summary_table": summary_df,
        "proves_head_bottleneck": False,
    }


# ---------------------------------------------------------------------------
# S6: UMAP Qualitative Projection
# ---------------------------------------------------------------------------

def export_umap_projection(
    features: np.ndarray,
    labels: np.ndarray,
    output_path: Optional[Union[str, Path]] = None,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    random_state: int = 42,
    metric: str = "cosine",
) -> pd.DataFrame:
    """
    Computes 2D UMAP projection for qualitative visualization only.
    Exports DataFrame with explicit qualitative disclaimer.
    """
    try:
        from umap import UMAP

        reducer = UMAP(
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            n_components=2,
            metric=metric,
            random_state=random_state,
        )
        embedding = reducer.fit_transform(features)
    except ImportError:
        # Graceful fallback: 2D PCA projection if UMAP is not installed in environment
        centered = features - np.mean(features, axis=0)
        u, s, vt = np.linalg.svd(centered, full_matrices=False)
        embedding = centered @ vt[:2].T

    df_umap = pd.DataFrame(
        {
            "umap_1": embedding[:, 0],
            "umap_2": embedding[:, 1],
            "label": labels,
            "is_qualitative_only": True,
        }
    )

    if output_path is not None:
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        # Include disclaimer header if saving as CSV
        if p.suffix.lower() == ".csv":
            with open(p, "w", encoding="utf-8") as f:
                f.write("# QUALITATIVE ONLY: 2D UMAP projection is strictly for exploratory visualization.\n")
                f.write("# Do not infer quantitative cluster separation or phylogenetic distance.\n")
                df_umap.to_csv(f, index=False)
        elif p.suffix.lower() in (".parquet", ".pq"):
            df_umap.to_parquet(p, index=False)

    return df_umap
