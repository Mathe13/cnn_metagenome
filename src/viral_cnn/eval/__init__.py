"""
Evaluation, noise injection, and metric computation modules.
"""

from .noise import inject_noise, verify_effective_noise_rate
from .metrics import (
    ClassificationMetrics,
    compute_classification_metrics,
    compute_species_f1_bands,
    compute_paired_comparison,
)
from .runner import (
    QueryPrediction,
    EvaluationResult,
    evaluate_model,
    save_evaluation_artifacts,
)

__all__ = [
    "inject_noise",
    "verify_effective_noise_rate",
    "ClassificationMetrics",
    "compute_classification_metrics",
    "compute_species_f1_bands",
    "compute_paired_comparison",
    "QueryPrediction",
    "EvaluationResult",
    "evaluate_model",
    "save_evaluation_artifacts",
]
