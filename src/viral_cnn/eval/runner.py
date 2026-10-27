"""
Evaluation runner orchestrating batched forward passes, noise injection, and artifact emission.
"""

from collections.abc import Sequence as AbcSequence
from dataclasses import dataclass, asdict
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any, Dict, Iterator, List, Optional, Sequence, Union
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from ..data.encoding import encode_sequence_tokens
from .noise import inject_noise, verify_effective_noise_rate
from .metrics import (
    ClassificationMetrics,
    compute_classification_metrics,
    compute_species_f1_bands,
)


@dataclass
class QueryPrediction:
    query_id: str
    truth_label: int
    predicted_label: int
    top5_labels: List[int]
    confidence: float


class LazyPredictionList(AbcSequence):
    """
    Memory-efficient sequence of QueryPrediction objects backed by compact NumPy arrays.
    Reduces memory footprint for N=1,706,270 queries from ~2.0 GB of Python objects
    to ~20 MB of contiguous NumPy buffers while preserving full Sequence[QueryPrediction] compatibility.
    """

    def __init__(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        confs: np.ndarray,
        top5: Optional[np.ndarray] = None,
        query_ids: Optional[Sequence[str]] = None,
    ):
        self.y_true = np.asarray(y_true, dtype=np.int32)
        self.y_pred = np.asarray(y_pred, dtype=np.int32)
        self.confs = np.asarray(confs, dtype=np.float32)
        self.top5 = np.asarray(top5, dtype=np.int32) if top5 is not None else None
        self.query_ids = query_ids

    def __len__(self) -> int:
        return int(self.y_true.shape[0])

    def _make_item(self, idx: int) -> QueryPrediction:
        if idx < 0:
            idx += len(self)
        if idx < 0 or idx >= len(self):
            raise IndexError(idx)
        qid = self.query_ids[idx] if self.query_ids is not None else f"q_{idx:07d}"
        t5 = (
            self.top5[idx].tolist()
            if self.top5 is not None
            else [int(self.y_pred[idx])]
        )
        return QueryPrediction(
            query_id=qid,
            truth_label=int(self.y_true[idx]),
            predicted_label=int(self.y_pred[idx]),
            top5_labels=t5,
            confidence=round(float(self.confs[idx]), 6),
        )

    def __getitem__(self, index: Union[int, slice]) -> Union[QueryPrediction, List[QueryPrediction]]:
        if isinstance(index, slice):
            return [self._make_item(i) for i in range(*index.indices(len(self)))]
        return self._make_item(int(index))

    def __iter__(self) -> Iterator[QueryPrediction]:
        for i in range(len(self)):
            yield self._make_item(i)


@dataclass
class EvaluationResult:
    model_name: str
    noise_rate: float
    effective_noise_rate: float
    seed: int
    metrics: ClassificationMetrics
    f1_bands: Dict[str, Any]
    predictions: Sequence[QueryPrediction]
    y_true_arr: Optional[np.ndarray] = None
    y_pred_arr: Optional[np.ndarray] = None
    confs_arr: Optional[np.ndarray] = None

    def summary_dict(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "nominal_noise_rate": self.noise_rate,
            "effective_noise_rate": self.effective_noise_rate,
            "seed": self.seed,
            "metrics": self.metrics.to_dict(),
            "f1_bands": self.f1_bands,
        }


def evaluate_model(
    model: torch.nn.Module,
    sequences: List[str],
    labels: List[int],
    query_ids: Optional[List[str]] = None,
    noise_rate: float = 0.0,
    model_name: str = "custom_model",
    device: str = "cpu",
    batch_size: int = 256,
    num_classes: Optional[int] = None,
    max_len: int = 512,
    seed: int = 42,
    store_predictions: bool = True,
) -> EvaluationResult:
    """
    Executes batched evaluation of a 1D-CNN model on input sequences with controlled noise.
    Streams cross-entropy loss and Top-5 accuracy on device without accumulating (N, C) logits in RAM.
    """
    if len(sequences) != len(labels):
        raise ValueError(
            f"Length mismatch: sequences ({len(sequences)}) vs labels ({len(labels)})"
        )

    n_samples = len(sequences)
    if query_ids is None:
        q_ids = [f"q_{i:07d}" for i in range(n_samples)]
    else:
        q_ids = query_ids

    total_classes = (
        num_classes
        if num_classes is not None
        else (getattr(model, "num_classes", None) or max(labels) + 1)
    )

    dev = torch.device(device) if isinstance(device, str) else device
    model.eval()
    model.to(dev)

    # Encode all sequences to token tensors
    encoded_tokens = [encode_sequence_tokens(s, max_len=max_len) for s in sequences]
    tokens_tensor = torch.tensor(encoded_tokens, dtype=torch.long)
    labels_np = np.asarray(labels, dtype=np.int32)

    # Deterministic generator for noise injection
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)

    pred_chunks: List[np.ndarray] = []
    conf_chunks: List[np.ndarray] = []
    top5_chunks: List[np.ndarray] = []
    loss_sum = 0.0
    loss_valid_count = 0
    top5_correct_count = 0

    with torch.no_grad():
        for start_idx in range(0, n_samples, batch_size):
            end_idx = min(start_idx + batch_size, n_samples)
            batch_tokens = tokens_tensor[start_idx:end_idx]
            batch_labels_np = labels_np[start_idx:end_idx]

            if noise_rate > 0.0:
                batch_tokens = inject_noise(
                    sequences=batch_tokens,
                    noise_rate=noise_rate,
                    generator=generator,
                )

            batch_tokens = batch_tokens.to(dev)
            batch_labels_dev = torch.as_tensor(batch_labels_np, dtype=torch.long, device=dev)

            logits = model(batch_tokens)
            logits_fp32 = logits.float()

            valid_mask = (batch_labels_dev >= 0) & (batch_labels_dev < logits_fp32.shape[-1])
            if valid_mask.any():
                loss_sum += float(
                    F.cross_entropy(
                        logits_fp32[valid_mask],
                        batch_labels_dev[valid_mask],
                        reduction="sum",
                    ).item()
                )
                loss_valid_count += int(valid_mask.sum().item())

            probs = F.softmax(logits_fp32, dim=-1)

            # Top-1 and Top-5
            k = min(5, logits_fp32.shape[-1])
            topk_probs, topk_indices = torch.topk(probs, k=k, dim=-1)

            top5_matches = (topk_indices == batch_labels_dev.unsqueeze(1)).any(dim=1)
            top5_correct_count += int(top5_matches.sum().item())

            pred_chunks.append(topk_indices[:, 0].cpu().numpy().astype(np.int32))
            conf_chunks.append(topk_probs[:, 0].cpu().numpy().astype(np.float32))
            if store_predictions or n_samples <= 50_000:
                top5_chunks.append(topk_indices.cpu().numpy().astype(np.int32))

    y_pred_arr = np.concatenate(pred_chunks, axis=0) if pred_chunks else np.empty(0, dtype=np.int32)
    confs_arr = np.concatenate(conf_chunks, axis=0) if conf_chunks else np.empty(0, dtype=np.float32)
    top5_arr = np.concatenate(top5_chunks, axis=0) if top5_chunks else None

    precomputed_loss = (loss_sum / loss_valid_count) if loss_valid_count > 0 else None
    precomputed_top5 = (100.0 * top5_correct_count / n_samples) if n_samples > 0 else None

    # Compute metrics
    metrics = compute_classification_metrics(
        y_true=labels_np,
        y_pred=y_pred_arr,
        num_classes=total_classes,
        precomputed_loss=precomputed_loss,
        precomputed_top5_accuracy=precomputed_top5,
    )

    # Compute continuous F1 bands
    f1_bands = compute_species_f1_bands(
        y_true=labels_np,
        y_pred=y_pred_arr,
        num_classes=total_classes,
    )

    # Build query prediction objects (materialized list for small datasets, lazy sequence for huge datasets)
    lazy_preds = LazyPredictionList(
        y_true=labels_np,
        y_pred=y_pred_arr,
        confs=confs_arr,
        top5=top5_arr,
        query_ids=q_ids,
    )
    predictions: Sequence[QueryPrediction] = (
        list(lazy_preds) if (store_predictions and n_samples <= 50_000) else lazy_preds
    )

    return EvaluationResult(
        model_name=model_name,
        noise_rate=noise_rate,
        effective_noise_rate=round(0.75 * noise_rate, 6),
        seed=seed,
        metrics=metrics,
        f1_bands=f1_bands,
        predictions=predictions,
        y_true_arr=labels_np,
        y_pred_arr=y_pred_arr,
        confs_arr=confs_arr,
    )


def save_evaluation_artifacts(
    result: EvaluationResult,
    output_dir: Union[str, Path],
) -> Dict[str, str]:
    """
    Saves metrics.json, predictions.parquet (or csv), and execution manifest.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # 1. Save metrics summary JSON
    metrics_file = out_path / "metrics_summary.json"
    metrics_file.write_text(
        json.dumps(result.summary_dict(), indent=2),
        encoding="utf-8",
    )

    # 2. Save predictions table
    pred_data = {
        "query_id": [p.query_id for p in result.predictions],
        "truth_label": [p.truth_label for p in result.predictions],
        "predicted_label": [p.predicted_label for p in result.predictions],
        "confidence": [p.confidence for p in result.predictions],
    }
    # Store top 5 as separate columns or JSON strings
    for rank_idx in range(5):
        pred_data[f"top5_rank_{rank_idx + 1}"] = [
            p.top5_labels[rank_idx] if rank_idx < len(p.top5_labels) else -1
            for p in result.predictions
        ]

    df = pd.DataFrame(pred_data)
    predictions_file = out_path / "predictions.csv"
    df.to_csv(predictions_file, index=False)

    parquet_file = out_path / "predictions.parquet"
    try:
        df.to_parquet(parquet_file, index=False)
        primary_preds = str(parquet_file)
    except Exception:
        primary_preds = str(predictions_file)

    # 3. Save run manifest
    manifest_file = out_path / "run_manifest.json"
    manifest = {
        "model_name": result.model_name,
        "nominal_noise_rate": result.noise_rate,
        "effective_noise_rate": result.effective_noise_rate,
        "seed": result.seed,
        "n_samples": len(result.predictions),
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "output_files": {
            "metrics": str(metrics_file),
            "predictions": primary_preds,
        },
    }
    manifest_file.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    return {
        "metrics": str(metrics_file),
        "predictions": primary_preds,
        "manifest": str(manifest_file),
    }


def evaluate_dataloader(
    model: torch.nn.Module,
    dataloader: Any,
    device: Union[str, torch.device] = "cpu",
    noise_rate: float = 0.0,
    noise_rates: Optional[List[float]] = None,
    seed: int = 42,
    model_name: str = "custom_model",
    num_classes: Optional[int] = None,
    max_batches: Optional[int] = None,
    verify_noise: bool = True,
    store_predictions: bool = True,
    use_amp: Optional[bool] = None,
) -> Union[EvaluationResult, Dict[float, EvaluationResult]]:
    """
    Evaluates a PyTorch model over a DataLoader (or IterableDataset) of batches (inputs, labels),
    supporting clean evaluation and controlled noise sweeps in a single I/O pass over the dataloader.
    Streams cross-entropy loss and Top-5 accuracy on device without accumulating (N, C) logits in RAM.
    """
    dev = torch.device(device) if isinstance(device, str) else device
    model = model.to(dev)
    model.eval()

    amp_enabled = (dev.type == "cuda") if use_amp is None else bool(use_amp and dev.type == "cuda")
    rates_to_run: List[float] = list(noise_rates) if noise_rates is not None else [float(noise_rate)]

    # Independent deterministic CPU PRNG per noise rate for exact reproducibility
    generators: Dict[float, torch.Generator] = {}
    for r in rates_to_run:
        gen = torch.Generator(device="cpu")
        gen.manual_seed(seed)
        generators[r] = gen

    label_chunks: List[np.ndarray] = []
    pred_chunks: Dict[float, List[np.ndarray]] = {r: [] for r in rates_to_run}
    conf_chunks: Dict[float, List[np.ndarray]] = {r: [] for r in rates_to_run}
    top5_chunks: Dict[float, List[np.ndarray]] = {r: [] for r in rates_to_run}
    loss_sums: Dict[float, float] = {r: 0.0 for r in rates_to_run}
    loss_counts: Dict[float, int] = {r: 0 for r in rates_to_run}
    top5_correct_counts: Dict[float, int] = {r: 0 for r in rates_to_run}

    orig_tokens_sample: Dict[float, List[torch.Tensor]] = {r: [] for r in rates_to_run}
    noisy_tokens_sample: Dict[float, List[torch.Tensor]] = {r: [] for r in rates_to_run}
    sampled_rows: Dict[float, int] = {r: 0 for r in rates_to_run}

    batch_idx = 0
    total_seen = 0
    with torch.no_grad():
        for batch in dataloader:
            if max_batches is not None and batch_idx >= max_batches:
                break

            if isinstance(batch, (list, tuple)):
                inputs, labels = batch[0], batch[1]
            elif isinstance(batch, dict):
                inputs = batch.get("inputs") or batch.get("sequences")
                labels = batch.get("labels") or batch.get("label")
            else:
                raise ValueError(f"Unsupported batch type: {type(batch)}")

            # Squeeze outer worker dimensions if present
            if inputs.dim() == 3 and inputs.shape[0] == 1:
                inputs = inputs.squeeze(0)
            if labels.dim() == 2 and labels.shape[0] == 1:
                labels = labels.squeeze(0)

            batch_labels_np = labels.detach().cpu().numpy().astype(np.int32)
            label_chunks.append(batch_labels_np)
            batch_rows = int(batch_labels_np.shape[0])
            total_seen += batch_rows

            labels_dev = labels.to(dev, non_blocking=(dev.type == "cuda")).long()
            inputs_cpu = inputs.cpu() if inputs.device.type != "cpu" else inputs

            for r in rates_to_run:
                if r > 0.0:
                    noisy_in = inject_noise(
                        sequences=inputs_cpu,
                        noise_rate=r,
                        generator=generators[r],
                    )
                    if verify_noise and sampled_rows[r] < 16384:
                        orig_tokens_sample[r].append(inputs_cpu.clone())
                        noisy_tokens_sample[r].append(noisy_in.clone())
                        sampled_rows[r] += batch_rows
                else:
                    noisy_in = inputs_cpu

                inputs_dev = noisy_in.to(dev, non_blocking=(dev.type == "cuda"))
                with torch.amp.autocast("cuda", enabled=amp_enabled):
                    logits = model(inputs_dev)
                logits_fp32 = logits.float()

                valid_mask = (labels_dev >= 0) & (labels_dev < logits_fp32.shape[-1])
                if valid_mask.any():
                    loss_sums[r] += float(
                        F.cross_entropy(
                            logits_fp32[valid_mask],
                            labels_dev[valid_mask],
                            reduction="sum",
                        ).item()
                    )
                    loss_counts[r] += int(valid_mask.sum().item())

                probs = F.softmax(logits_fp32, dim=-1)
                k = min(5, logits_fp32.shape[-1])
                topk_probs, topk_indices = torch.topk(probs, k=k, dim=-1)

                top5_matches = (topk_indices == labels_dev.unsqueeze(1)).any(dim=1)
                top5_correct_counts[r] += int(top5_matches.sum().item())

                pred_chunks[r].append(topk_indices[:, 0].cpu().numpy().astype(np.int32))
                conf_chunks[r].append(topk_probs[:, 0].cpu().numpy().astype(np.float32))
                if store_predictions and total_seen <= 50_000:
                    top5_chunks[r].append(topk_indices.cpu().numpy().astype(np.int32))
                elif not store_predictions and top5_chunks[r]:
                    top5_chunks[r].clear()

            batch_idx += 1

    if not label_chunks or total_seen == 0:
        raise ValueError("Dataloader produced zero samples.")

    y_true_arr = np.concatenate(label_chunks, axis=0)
    total_classes = (
        num_classes
        if num_classes is not None
        else (getattr(model, "num_classes", None) or int(np.max(y_true_arr)) + 1)
    )

    sweep_results: Dict[float, EvaluationResult] = {}
    for r in rates_to_run:
        empirical_rate = 0.0
        if verify_noise and r > 0.0 and orig_tokens_sample[r]:
            orig_cat = torch.cat(orig_tokens_sample[r], dim=0)
            noisy_cat = torch.cat(noisy_tokens_sample[r], dim=0)
            empirical_rate = verify_effective_noise_rate(
                orig_cat,
                noisy_cat,
                epsilon=r,
                tolerance=0.03,
            )
            orig_tokens_sample[r].clear()
            noisy_tokens_sample[r].clear()

        y_pred_arr = np.concatenate(pred_chunks[r], axis=0)
        confs_arr = np.concatenate(conf_chunks[r], axis=0)
        top5_arr = (
            np.concatenate(top5_chunks[r], axis=0)
            if (store_predictions and total_seen <= 50_000 and top5_chunks[r])
            else None
        )
        pred_chunks[r].clear()
        conf_chunks[r].clear()
        top5_chunks[r].clear()

        precomputed_loss = (loss_sums[r] / loss_counts[r]) if loss_counts[r] > 0 else None
        precomputed_top5 = (100.0 * top5_correct_counts[r] / total_seen) if total_seen > 0 else None

        metrics = compute_classification_metrics(
            y_true=y_true_arr,
            y_pred=y_pred_arr,
            num_classes=total_classes,
            precomputed_loss=precomputed_loss,
            precomputed_top5_accuracy=precomputed_top5,
        )

        f1_bands = compute_species_f1_bands(
            y_true=y_true_arr,
            y_pred=y_pred_arr,
            num_classes=total_classes,
            thresholds=(0.0, 0.50, 0.80, 0.90, 1.0),
            confidences=confs_arr,
        )

        lazy_preds = LazyPredictionList(
            y_true=y_true_arr,
            y_pred=y_pred_arr,
            confs=confs_arr,
            top5=top5_arr,
        )
        predictions: Sequence[QueryPrediction] = (
            list(lazy_preds) if (store_predictions and total_seen <= 50_000) else lazy_preds
        )

        sweep_results[r] = EvaluationResult(
            model_name=model_name,
            noise_rate=r,
            effective_noise_rate=round(empirical_rate if r > 0.0 else 0.0, 6),
            seed=seed,
            metrics=metrics,
            f1_bands=f1_bands,
            predictions=predictions,
            y_true_arr=y_true_arr,
            y_pred_arr=y_pred_arr,
            confs_arr=confs_arr,
        )

    if noise_rates is not None:
        return sweep_results
    return sweep_results[rates_to_run[0]]

