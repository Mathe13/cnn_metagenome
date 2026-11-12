"""Interpretability and motif analysis module for viral 1D-CNN architectures."""

from viral_cnn.interpretability.motifs import (
    extract_first_layer_motifs,
    scan_motif_occurrences,
    MatchedControl,
    sample_matched_control_region,
    sample_substitution_offsets,
    perturb_region_with_offsets,
    hash_mutation_offsets,
    perturb_sequence_mask,
    evaluate_motif_perturbation,
    summarize_motif_control,
    summarize_matched_control_coverage,
    audit_motif_composition_robustness,
    compute_jaccard_cooccurrence,
    analyze_cooccurrence_pairs,
    compute_exploratory_triplet_rscu,
    audit_translation_boundary_signals,
    generate_motif_audit_figures,
)

__all__ = [
    "extract_first_layer_motifs",
    "scan_motif_occurrences",
    "MatchedControl",
    "sample_matched_control_region",
    "sample_substitution_offsets",
    "perturb_region_with_offsets",
    "hash_mutation_offsets",
    "perturb_sequence_mask",
    "evaluate_motif_perturbation",
    "summarize_motif_control",
    "summarize_matched_control_coverage",
    "audit_motif_composition_robustness",
    "compute_jaccard_cooccurrence",
    "analyze_cooccurrence_pairs",
    "compute_exploratory_triplet_rscu",
    "audit_translation_boundary_signals",
    "generate_motif_audit_figures",
]

