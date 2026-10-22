"""
Calibration and probabilistic confidence evaluation engine for 1D-CNN viral models.
Implements ECE, MCE, multiclass NLL, memory-efficient Brier score, reliability diagrams,
and validation-only temperature scaling.
"""

from dataclasses import dataclass, asdict
import json
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Union
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from ..data.encoding import encode_sequence_tokens
from .noise import inject_noise


@dataclass
class CalibrationMetrics:
    top1_accuracy: float
    top1_confidence: float
    ece: float
    mce: float
    nll: float
    brier_score: float
    n_samples: int
    n_bins: int
    binning_strategy: str
    sum_bin_weights: float
    conf_on_correct: float
    conf_on_incorrect: float
    overconfidence_gap: float
    is_overconfident_regime: bool

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ReliabilityDiagramBin:
    bin_idx: int
    bin_lower: float
    bin_upper: float
    count: int
    confidence: float
    accuracy: float
    calibration_gap: float
    weight: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _to_numpy_probs_and_labels(
    probs: Union[np.ndarray, torch.Tensor],
    y_true: Union[np.ndarray, torch.Tensor, List[int]],
) -> Tuple[np.ndarray, np.ndarray]:
    """Converts input probabilities and ground-truth labels to aligned numpy arrays."""
    if isinstance(probs, torch.Tensor):
        p_np = probs.detach().cpu().numpy().astype(np.float64)
    else:
        p_np = np.asarray(probs, dtype=np.float64)

    if isinstance(y_true, torch.Tensor):
        y_np = y_true.detach().cpu().numpy().astype(np.int64)
    else:
        y_np = np.asarray(y_true, dtype=np.int64)

    if len(p_np) != len(y_np):
        raise ValueError(
            f"Sample count mismatch: probs has {len(p_np)} but y_true has {len(y_np)}"
        )
    if len(p_np) == 0:
        raise ValueError("Input probability array cannot be empty.")

    return p_np, y_np


def compute_reliability_diagram(
    probs: Union[np.ndarray, torch.Tensor],
    y_true: Union[np.ndarray, torch.Tensor, List[int]],
    n_bins: int = 15,
    binning_strategy: Literal["equal_width", "equal_mass"] = "equal_width",
) -> List[ReliabilityDiagramBin]:
    """
    Partitions Top-1 predictions into declared confidence bins and computes
    bin-level empirical accuracy, average confidence, sample count, and calibration gap.
    Guarantees sample conservation: sum(count) strictly equals N.
    """
    p_np, y_np = _to_numpy_probs_and_labels(probs, y_true)
    n_samples = len(y_np)

    top1_preds = np.argmax(p_np, axis=1)
    top1_confs = np.max(p_np, axis=1)
    is_correct = (top1_preds == y_np)

    # Determine bin edges
    if binning_strategy == "equal_width":
        bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
        # Assign samples to bins: [b_i, b_{i+1}), last bin closed [b_{M-1}, 1.0]
        bin_indices = np.digitize(top1_confs, bin_edges) - 1
        # Handle edge cases (e.g. conf == 1.0 or conf < 0.0)
        bin_indices = np.clip(bin_indices, 0, n_bins - 1)
    elif binning_strategy == "equal_mass":
        quantiles = np.linspace(0.0, 100.0, n_bins + 1)
        bin_edges = np.percentile(top1_confs, quantiles)
        # Ensure strictly non-decreasing edges
        bin_edges[0] = 0.0
        bin_edges[-1] = 1.0
        bin_indices = np.digitize(top1_confs, bin_edges) - 1
        bin_indices = np.clip(bin_indices, 0, n_bins - 1)
    else:
        raise ValueError(
            f"Unsupported binning_strategy '{binning_strategy}'. Choose 'equal_width' or 'equal_mass'."
        )

    diagram_bins: List[ReliabilityDiagramBin] = []
    for b_idx in range(n_bins):
        mask = (bin_indices == b_idx)
        bin_count = int(np.sum(mask))
        bin_lower = float(bin_edges[b_idx])
        bin_upper = float(bin_edges[b_idx + 1])
        weight = float(bin_count / n_samples)

        if bin_count > 0:
            bin_conf = float(np.mean(top1_confs[mask]))
            bin_acc = float(np.mean(is_correct[mask]))
            cal_gap = float(abs(bin_acc - bin_conf))
        else:
            bin_conf = 0.0
            bin_acc = 0.0
            cal_gap = 0.0

        diagram_bins.append(
            ReliabilityDiagramBin(
                bin_idx=b_idx,
                bin_lower=round(bin_lower, 4),
                bin_upper=round(bin_upper, 4),
                count=bin_count,
                confidence=round(bin_conf, 6),
                accuracy=round(bin_acc, 6),
                calibration_gap=round(cal_gap, 6),
                weight=round(weight, 6),
            )
        )

    # Sanity check: sample conservation
    total_binned = sum(b.count for b in diagram_bins)
    if total_binned != n_samples:
        raise AssertionError(
            f"Sample conservation violation: {total_binned} binned vs {n_samples} total."
        )

    return diagram_bins


def compute_calibration_metrics(
    probs: Union[np.ndarray, torch.Tensor],
    y_true: Union[np.ndarray, torch.Tensor, List[int]],
    n_bins: int = 15,
    binning_strategy: Literal["equal_width", "equal_mass"] = "equal_width",
    eps: float = 1e-15,
) -> CalibrationMetrics:
    """
    Computes formal probabilistic calibration metrics:
      - Expected Calibration Error (ECE)
      - Maximum Calibration Error (MCE)
      - Multiclass Negative Log-Likelihood (NLL)
      - Exact multiclass Brier score via memory-efficient identity:
          Brier = (1/N) * sum_i (sum_k p_{ik}^2 - 2*p_{i, y_i} + 1)
      - Partitioned confidence on correct vs incorrect predictions
      - Overconfidence gap and regime indicator
    """
    p_np, y_np = _to_numpy_probs_and_labels(probs, y_true)
    n_samples, n_classes = p_np.shape

    top1_preds = np.argmax(p_np, axis=1)
    top1_confs = np.max(p_np, axis=1)
    is_correct = (top1_preds == y_np)

    top1_acc = float(np.mean(is_correct) * 100.0)
    top1_conf = float(np.mean(top1_confs))

    # Correct vs incorrect confidence partitioning
    correct_mask = is_correct
    incorrect_mask = ~is_correct

    conf_on_correct = float(np.mean(top1_confs[correct_mask])) if np.any(correct_mask) else 0.0
    conf_on_incorrect = float(np.mean(top1_confs[incorrect_mask])) if np.any(incorrect_mask) else 0.0
    overconfidence_gap = float(top1_conf - (top1_acc / 100.0))
    is_overconfident = overconfidence_gap > 0.10

    # Reliability diagram bins for ECE and MCE
    bins = compute_reliability_diagram(
        probs=p_np,
        y_true=y_np,
        n_bins=n_bins,
        binning_strategy=binning_strategy,
    )

    ece = 0.0
    mce = 0.0
    sum_weights = 0.0
    for b in bins:
        if b.count > 0:
            ece += (b.count / n_samples) * b.calibration_gap
            if b.calibration_gap > mce:
                mce = b.calibration_gap
        sum_weights += (b.count / n_samples)

    # Multiclass Negative Log-Likelihood (NLL) with epsilon clamping
    valid_target_mask = (y_np >= 0) & (y_np < n_classes)
    p_true = np.zeros(n_samples, dtype=np.float64)
    if np.any(valid_target_mask):
        p_true[valid_target_mask] = p_np[np.arange(n_samples)[valid_target_mask], y_np[valid_target_mask]]
    p_true_clamped = np.clip(p_true, eps, 1.0)
    nll = float(-np.mean(np.log(p_true_clamped)))

    # Multiclass Brier score: memory-efficient expansion
    # sum_k (p_ik - y_ik)^2 = sum_k p_ik^2 - 2 * p_{i, y_i} + 1
    sum_p_sq = np.sum(p_np ** 2, axis=1)
    brier_per_sample = sum_p_sq - 2.0 * p_true + 1.0
    brier_score = float(np.mean(brier_per_sample))

    return CalibrationMetrics(
        top1_accuracy=round(top1_acc, 4),
        top1_confidence=round(top1_conf, 6),
        ece=round(ece, 6),
        mce=round(mce, 6),
        nll=round(nll, 6),
        brier_score=round(brier_score, 6),
        n_samples=n_samples,
        n_bins=n_bins,
        binning_strategy=binning_strategy,
        sum_bin_weights=round(sum_weights, 6),
        conf_on_correct=round(conf_on_correct, 6),
        conf_on_incorrect=round(conf_on_incorrect, 6),
        overconfidence_gap=round(overconfidence_gap, 6),
        is_overconfident_regime=is_overconfident,
    )


def save_reliability_diagram(
    bins: List[ReliabilityDiagramBin],
    output_dir: Union[str, Path],
    title: str = "Reliability Diagram",
    filename_prefix: str = "reliability_diagram",
) -> Dict[str, str]:
    """
    Exports structured reliability diagram data (CSV, JSON) and renders
    a publication-ready Matplotlib plot comparing accuracy vs confidence.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # 1. Export CSV
    records = [b.to_dict() for b in bins]
    df = pd.DataFrame(records)
    csv_file = out_path / f"{filename_prefix}.csv"
    df.to_csv(csv_file, index=False)

    # 2. Export JSON
    json_file = out_path / f"{filename_prefix}.json"
    json_file.write_text(json.dumps(records, indent=2), encoding="utf-8")

    # 3. Render Plot
    plot_file = out_path / f"{filename_prefix}.png"
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=(7, 7), gridspec_kw={"height_ratios": [3, 1]}, sharex=True
        )

        bin_centers = [(b.bin_lower + b.bin_upper) / 2.0 for b in bins]
        bin_widths = [b.bin_upper - b.bin_lower for b in bins]
        accuracies = [b.accuracy for b in bins]
        confidences = [b.confidence for b in bins]
        counts = [b.count for b in bins]

        # Top panel: Reliability curve and bar gaps
        ax1.plot([0, 1], [0, 1], "--", color="gray", label="Perfect Calibration")
        ax1.bar(
            bin_centers,
            accuracies,
            width=bin_widths,
            alpha=0.6,
            color="#2b5c8f",
            edgecolor="black",
            label="Empirical Accuracy",
        )
        ax1.plot(
            bin_centers,
            confidences,
            marker="o",
            color="#d95f02",
            linewidth=2,
            label="Mean Confidence",
        )
        ax1.set_ylabel("Accuracy")
        ax1.set_ylim(-0.02, 1.02)
        ax1.set_title(title)
        ax1.legend(loc="upper left")
        ax1.grid(True, linestyle=":", alpha=0.6)

        # Bottom panel: Sample count histogram
        ax2.bar(
            bin_centers,
            counts,
            width=bin_widths,
            color="#7570b3",
            edgecolor="black",
            alpha=0.7,
        )
        ax2.set_xlabel("Confidence")
        ax2.set_ylabel("Count")
        ax2.set_xlim(0.0, 1.0)
        ax2.grid(True, linestyle=":", alpha=0.6)

        plt.tight_layout()
        plt.savefig(plot_file, dpi=300)
        plt.close(fig)
    except Exception as e:
        # If matplotlib plotting fails, create a placeholder text
        plot_file.write_text(f"Plot generation skipped: {e}", encoding="utf-8")

    return {
        "csv": str(csv_file),
        "json": str(json_file),
        "plot": str(plot_file),
    }


class TemperatureScaler:
    """
    Post-hoc probability calibration via Temperature Scaling (Guo et al. 2017).
    Enforces that temperature parameters are fitted strictly on validation logits.
    """

    def __init__(self, initial_temperature: float = 1.0):
        self.temperature = float(initial_temperature)

    def fit(
        self,
        val_logits: Union[np.ndarray, torch.Tensor],
        val_labels: Union[np.ndarray, torch.Tensor, List[int]],
        split_name: str = "validation",
        max_iter: int = 100,
        lr: float = 0.05,
    ) -> "TemperatureScaler":
        """
        Learns scalar temperature T > 0 strictly by minimizing Negative Log-Likelihood
        on validation logits. Rejects fitting on test or undefined splits to prevent data snooping.
        """
        if split_name != "validation":
            raise ValueError(
                f"TemperatureScaler must be fitted strictly on validation split to prevent "
                f"data leakage. Received split_name='{split_name}'."
            )

        if isinstance(val_logits, np.ndarray):
            logits_tensor = torch.tensor(val_logits, dtype=torch.float32)
        else:
            logits_tensor = val_logits.detach().cpu().float()

        if isinstance(val_labels, (np.ndarray, list)):
            labels_tensor = torch.tensor(val_labels, dtype=torch.long)
        else:
            labels_tensor = val_labels.detach().cpu().long()

        # Parameterize via log_temperature to strictly enforce T > 0
        log_temp = torch.nn.Parameter(torch.zeros(1, dtype=torch.float32))
        optimizer = torch.optim.LBFGS([log_temp], lr=lr, max_iter=max_iter)

        def eval_loss():
            optimizer.zero_grad()
            t = torch.exp(log_temp)
            scaled_logits = logits_tensor / t
            loss = F.cross_entropy(scaled_logits, labels_tensor)
            loss.backward()
            return loss

        optimizer.step(eval_loss)
        self.temperature = float(torch.exp(log_temp).item())
        return self

    def transform(
        self,
        test_logits: Union[np.ndarray, torch.Tensor],
        test_labels: Optional[Union[np.ndarray, torch.Tensor, List[int]]] = None,
        n_bins: int = 15,
    ) -> Dict[str, Any]:
        """
        Scales test logits by 1/T and evaluates post-calibration metrics if labels are provided.
        """
        if isinstance(test_logits, np.ndarray):
            logits_tensor = torch.tensor(test_logits, dtype=torch.float32)
        else:
            logits_tensor = test_logits.detach().cpu().float()

        scaled_logits = logits_tensor / self.temperature
        probs_uncal = F.softmax(logits_tensor, dim=-1).numpy()
        probs_cal = F.softmax(scaled_logits, dim=-1).numpy()

        result: Dict[str, Any] = {
            "temperature": round(self.temperature, 6),
            "calibrated_logits": scaled_logits.numpy(),
            "calibrated_probs": probs_cal,
        }

        if test_labels is not None:
            uncal_metrics = compute_calibration_metrics(
                probs=probs_uncal,
                y_true=test_labels,
                n_bins=n_bins,
            )
            cal_metrics = compute_calibration_metrics(
                probs=probs_cal,
                y_true=test_labels,
                n_bins=n_bins,
            )
            result["uncalibrated_metrics"] = uncal_metrics.to_dict()
            result["calibrated_metrics"] = cal_metrics.to_dict()

        return result


def audit_model_calibration_by_noise(
    models_dict: Dict[str, torch.nn.Module],
    sequences: List[str],
    labels: List[int],
    noise_rates: List[float] = [0.0, 0.02, 0.05, 0.10, 0.15],
    num_classes: Optional[int] = None,
    device: str = "cpu",
    batch_size: int = 256,
    seed: int = 42,
    n_bins: int = 15,
) -> Dict[str, Any]:
    """
    Evaluates probabilistic calibration (ECE, MCE, NLL, Brier score, accuracy, confidence)
    across multiple 1D-CNN model variants and controlled nucleotide noise levels.
    """
    total_classes = num_classes if num_classes is not None else (max(labels) + 1)
    tokens_tensor = torch.tensor(
        [encode_sequence_tokens(s, max_len=512) for s in sequences],
        dtype=torch.long,
    )
    labels_tensor = torch.tensor(labels, dtype=torch.long)
    n_samples = len(sequences)

    audit_report: Dict[str, Any] = {
        "n_samples": n_samples,
        "num_classes": total_classes,
        "nominal_noise_rates": noise_rates,
        "seed": seed,
        "models": {},
    }

    for model_name, model in models_dict.items():
        model.eval()
        model.to(device)
        audit_report["models"][model_name] = {}

        for rate in noise_rates:
            rate_key = f"noise_{rate}"
            gen = torch.Generator(device="cpu").manual_seed(seed)

            all_probs_list = []
            with torch.no_grad():
                for start_idx in range(0, n_samples, batch_size):
                    end_idx = min(start_idx + batch_size, n_samples)
                    batch_tokens = tokens_tensor[start_idx:end_idx]

                    if rate > 0.0:
                        batch_tokens = inject_noise(
                            sequences=batch_tokens,
                            noise_rate=rate,
                            generator=gen,
                        )

                    batch_tokens = batch_tokens.to(device)
                    logits = model(batch_tokens)
                    probs = F.softmax(logits, dim=-1)
                    all_probs_list.append(probs.cpu().numpy())

            model_probs = np.concatenate(all_probs_list, axis=0)
            metrics = compute_calibration_metrics(
                probs=model_probs,
                y_true=labels,
                n_bins=n_bins,
            )
            audit_report["models"][model_name][rate_key] = metrics.to_dict()

    return audit_report

