"""Status report for the optional controlled architecture experiment."""

from pathlib import Path
from typing import Any, Optional


CAUSAL_DEMARCATION_TEXT = (
    "In the absence of multi-seed full-scale retraining across the entire closed-set taxonomy, "
    "causal conclusions are strictly confined to the evaluated training configurations and cannot "
    "be attributed exclusively to architecture."
)


def generate_controlled_experiment_report(
    runs_dir: Path | str,
    output_path: Path | str,
    full_dataset_size: int = 1706270,
) -> Path:
    """Generate a report that does not turn missing execution evidence into results.

    The optional C05 scratch comparison has not been activated. Until it has a
    protocol-specific implementation and complete manifests, historical generic
    run logs and modeled resource figures are not admissible as controlled evidence.
    """
    del runs_dir, full_dataset_size
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    content = f"""# Controlled Architecture Experiment and Causal Audit

## 1. Executive Summary & Confounding Factors in Historical Evidence

**Controlled comparison status: `NOT_RUN`.** No complete C05 `c05_controlled_scratch_v1` manifest set is available. Historical fine-tuning starts from different checkpoints, and the ConvNeXt notebook and model registry disagree on dropout (0.5 and 0.7). The reproduction code selects ResNet by `last` and ConvNeXt by validation Macro-F1; an older best-validation-loss claim is not established by the checkpoint filename. These differences prevent an architecture-only interpretation.

## 2. Hyperparameter Parity Matrix

The controlled protocol is a planned configuration, not observed training parity. Its required configuration is AdamW, CosineAnnealingLR with warmup, label smoothing 0.1, dropout 0.5, validation Macro-F1 selection and seeds 42, 43 and 44. The configured 200-step budget is a pilot and does not establish the full experiment.

| Parameter | ResNet50-1D | ConvNeXt-Tiny-1D | Evidence status |
| :--- | :--- | :--- | :--- |
| Optimizer | AdamW | AdamW | planned; not measured |
| Scheduler | CosineAnnealingLR + warmup | CosineAnnealingLR + warmup | planned; not measured |
| Checkpoint metric | best validation Macro-F1 | best validation Macro-F1 | planned; not measured |
| Seeds | 42, 43, 44 | 42, 43, 44 | planned; NOT_RUN |

## 3. GPU Cost Estimation & Compute Resource Profiling

**Resource status: `NOT_MEASURED`.** No C05 hardware, processed-sequence count, elapsed time, GPU-hours, peak VRAM or throughput measurements are available. FLOPs and elapsed-time extrapolations are not reported as measurements.

| Metric | ResNet50-1D | ConvNeXt-Tiny-1D | Status |
| :--- | :--- | :--- | :--- |
| Trainable parameters | NOT_MEASURED | NOT_MEASURED | no C05 run manifest |
| Step latency | NOT_MEASURED | NOT_MEASURED | no C05 run manifest |
| Throughput | NOT_MEASURED | NOT_MEASURED | no C05 run manifest |
| Peak VRAM | NOT_MEASURED | NOT_MEASURED | no C05 run manifest |
| GPU-hours | NOT_MEASURED | NOT_MEASURED | no C05 run manifest |

## 4. Controlled Cross-Architecture Benchmark Results

**Results status: `NOT_RUN`.** A complete comparison requires all six architecture/seed runs, the declared full budget, and evaluation on the same identified test split.

| Architecture | Seed | Noise Rate | Top-1 Accuracy | Top-5 Accuracy | Macro-F1 | Step Latency | Throughput | Peak VRAM |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `resnet50_1d` | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN |
| `convnext_tiny_1d` | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN |

## 5. Scientific Causal Demarcation & Manuscript Invariants

> **Scientific Invariant (Causal Demarcation):**
> {CAUSAL_DEMARCATION_TEXT}

Until the controlled protocol has complete, manifest-backed runs, reported differences must remain specific to the identified checkpoints and training configurations. They do not establish universal architecture superiority. The primary evaluation split remains fragment-level and closed-set, not generalization to unseen genomes or species.
"""
    out_file.write_text(content, encoding="utf-8")
    return out_file


def verify_controlled_report_consistency(
    report_path: Path | str,
    run_logs: Optional[list[dict[str, Any]]] = None,
) -> bool:
    """Check the status report's required sections and no-run evidence boundary."""
    path = Path(report_path)
    if not path.exists():
        return False
    content = path.read_text(encoding="utf-8")
    required_sections = (
        "# Controlled Architecture Experiment",
        "## 1. Executive Summary & Confounding Factors in Historical Evidence",
        "## 2. Hyperparameter Parity Matrix",
        "## 3. GPU Cost Estimation & Compute Resource Profiling",
        "## 4. Controlled Cross-Architecture Benchmark Results",
        "## 5. Scientific Causal Demarcation & Manuscript Invariants",
    )
    if any(section not in content for section in required_sections):
        return False
    if CAUSAL_DEMARCATION_TEXT not in content:
        return False
    if not run_logs and ("NOT_RUN" not in content or "NOT_MEASURED" not in content):
        return False
    if not run_logs and any(value in content for value in ("84.88%", "17,650.0", "4,210.0")):
        return False
    return True
