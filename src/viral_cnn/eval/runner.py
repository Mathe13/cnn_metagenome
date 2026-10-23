"""
Evaluation runner orchestrating batched forward passes, noise injection, and artifact emission.
"""

from dataclasses import dataclass, asdict
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Union
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


@dataclass
class EvaluationResult:
    model_name: str
    noise_rate: float
    effective_noise_rate: float
    seed: int
    metrics: ClassificationMetrics
    f1_bands: Dict[str, Any]
    predictions: List[QueryPrediction]

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
) -> EvaluationResult:
    """
    Executes batched evaluation of a 1D-CNN model on input sequences with controlled noise.
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

    model.eval()
    model.to(device)

    # Encode all sequences to token tensors
    encoded_tokens = [encode_sequence_tokens(s, max_len=max_len) for s in sequences]
    tokens_tensor = torch.tensor(encoded_tokens, dtype=torch.long)
    labels_tensor = torch.tensor(labels, dtype=torch.long)

    # Deterministic generator for noise injection
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)

    all_preds: List[int] = []
    all_top5: List[List[int]] = []
    all_confs: List[float] = []
    all_logits_list: List[np.ndarray] = []

    with torch.no_grad():
        for start_idx in range(0, n_samples, batch_size):
            end_idx = min(start_idx + batch_size, n_samples)
            batch_tokens = tokens_tensor[start_idx:end_idx]

            if noise_rate > 0.0:
                batch_tokens = inject_noise(
                    sequences=batch_tokens,
                    noise_rate=noise_rate,
                    generator=generator,
                )

            batch_tokens = batch_tokens.to(device)
            logits = model(batch_tokens)
            probs = F.softmax(logits, dim=-1)

            # Top-1 and Top-5
            k = min(5, logits.shape[-1])
            topk_probs, topk_indices = torch.topk(probs, k=k, dim=-1)

            top1_preds = topk_indices[:, 0].cpu().tolist()
            top1_confs = topk_probs[:, 0].cpu().tolist()
            top5_indices = topk_indices.cpu().tolist()

            all_preds.extend(top1_preds)
            all_top5.extend(top5_indices)
            all_confs.extend(top1_confs)
            all_logits_list.append(logits.cpu().numpy())

    all_logits = np.concatenate(all_logits_list, axis=0) if all_logits_list else None

    # Compute metrics
    metrics = compute_classification_metrics(
        y_true=labels,
        y_pred=all_preds,
        y_top5=all_top5,
        logits=all_logits,
        num_classes=total_classes,
    )

    # Compute continuous F1 bands
    f1_bands = compute_species_f1_bands(
        y_true=labels,
        y_pred=all_preds,
        num_classes=total_classes,
    )

    # Build query prediction objects
    predictions: List[QueryPrediction] = []
    for i in range(n_samples):
        predictions.append(
            QueryPrediction(
                query_id=q_ids[i],
                truth_label=labels[i],
                predicted_label=all_preds[i],
                top5_labels=all_top5[i],
                confidence=round(all_confs[i], 6),
            )
        )

    return EvaluationResult(
        model_name=model_name,
        noise_rate=noise_rate,
        effective_noise_rate=round(0.75 * noise_rate, 6),
        seed=seed,
        metrics=metrics,
        f1_bands=f1_bands,
        predictions=predictions,
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
) -> Union[EvaluationResult, Dict[float, EvaluationResult]]:
    """
    Evaluates a PyTorch model over a DataLoader (or IterableDataset) of batches (inputs, labels),
    supporting clean evaluation and controlled noise sweeps.
    """
    dev = torch.device(device) if isinstance(device, str) else device
    model = model.to(dev)
    model.eval()

    def _eval_for_rate(rate: float) -> EvaluationResult:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)

        all_labels: List[int] = []
        all_preds: List[int] = []
        all_top5: List[List[int]] = []
        all_confs: List[float] = []
        all_logits_list: List[np.ndarray] = []

        orig_tokens_sample: List[torch.Tensor] = []
        noisy_tokens_sample: List[torch.Tensor] = []

        batch_idx = 0
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

                orig_inputs = inputs.clone()

                if rate > 0.0:
                    inputs = inject_noise(
                        sequences=inputs,
                        noise_rate=rate,
                        generator=generator,
                    )

                if verify_noise and rate > 0.0 and len(orig_tokens_sample) < 50:
                    orig_tokens_sample.append(orig_inputs.cpu())
                    noisy_tokens_sample.append(inputs.cpu())

                inputs = inputs.to(dev)
                logits = model(inputs)
                probs = F.softmax(logits, dim=-1)

                k = min(5, logits.shape[-1])
                topk_probs, topk_indices = torch.topk(probs, k=k, dim=-1)

                top1_preds = topk_indices[:, 0].cpu().tolist()
                top1_confs = topk_probs[:, 0].cpu().tolist()
                top5_indices = topk_indices.cpu().tolist()

                all_labels.extend(labels.cpu().tolist())
                all_preds.extend(top1_preds)
                all_top5.extend(top5_indices)
                all_confs.extend(top1_confs)
                all_logits_list.append(logits.cpu().numpy())

                batch_idx += 1

        if not all_labels:
            raise ValueError("Dataloader produced zero samples.")

        empirical_rate = 0.0
        if verify_noise and rate > 0.0 and orig_tokens_sample:
            orig_cat = torch.cat(orig_tokens_sample, dim=0)
            noisy_cat = torch.cat(noisy_tokens_sample, dim=0)
            empirical_rate = verify_effective_noise_rate(
                orig_cat,
                noisy_cat,
                epsilon=rate,
                tolerance=0.03,
            )

        total_classes = (
            num_classes
            if num_classes is not None
            else (getattr(model, "num_classes", None) or max(all_labels) + 1)
        )

        all_logits = np.concatenate(all_logits_list, axis=0) if all_logits_list else None

        metrics = compute_classification_metrics(
            y_true=all_labels,
            y_pred=all_preds,
            y_top5=all_top5,
            logits=all_logits,
            num_classes=total_classes,
        )

        f1_bands = compute_species_f1_bands(
            y_true=all_labels,
            y_pred=all_preds,
            num_classes=total_classes,
            thresholds=(0.0, 0.50, 0.80, 0.90, 1.0),
            confidences=all_confs,
        )

        predictions: List[QueryPrediction] = []
        for i in range(len(all_labels)):
            predictions.append(
                QueryPrediction(
                    query_id=f"q_{i:07d}",
                    truth_label=all_labels[i],
                    predicted_label=all_preds[i],
                    top5_labels=all_top5[i],
                    confidence=round(all_confs[i], 6),
                )
            )

        return EvaluationResult(
            model_name=model_name,
            noise_rate=rate,
            effective_noise_rate=round(empirical_rate if rate > 0.0 else 0.0, 6),
            seed=seed,
            metrics=metrics,
            f1_bands=f1_bands,
            predictions=predictions,
        )

    if noise_rates is not None:
        sweep_results: Dict[float, EvaluationResult] = {}
        for r in noise_rates:
            sweep_results[r] = _eval_for_rate(r)
        return sweep_results
    else:
        return _eval_for_rate(noise_rate)
