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
from .latent import (
    extract_latent_embeddings,
    evaluate_cosine_knn,
    evaluate_genome_disjoint_knn,
    evaluate_silhouette_with_audit,
    calculate_taxonomic_correlations,
    LinearProbeClassifier,
    audit_classifier_bottleneck,
    export_umap_projection,
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
    "extract_latent_embeddings",
    "evaluate_cosine_knn",
    "evaluate_genome_disjoint_knn",
    "evaluate_silhouette_with_audit",
    "calculate_taxonomic_correlations",
    "LinearProbeClassifier",
    "audit_classifier_bottleneck",
    "export_umap_projection",
]
