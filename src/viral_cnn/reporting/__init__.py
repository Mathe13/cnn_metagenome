"""Reporting, tables, figures, and manuscript generation modules."""

from viral_cnn.reporting.tables import (
    generate_all_tables,
    generate_publication_tables,
    generate_historical_snapshot_tables,
    generate_table_1_metrics,
    generate_table_2_profiling,
    generate_table_3_noise_degradation,
    generate_table_4_pairwise_comparison,
    generate_table_8_motifs_summary,
    generate_table_10_boundary_triplets,
    generate_table_11_architecture_correlations,
    generate_table_12_voter_consensus,
)
from viral_cnn.reporting.figures import (
    generate_all_figures,
    verify_latex_figure_paths,
    verify_figure_publication_readiness,
)
from viral_cnn.reporting.manuscript import (
    build_manuscript_variants,
    compile_manuscript,
    generate_manuscript_content,
)
from viral_cnn.reporting.publication import (
    validate_artifact_provenance,
    validate_selection,
    assemble_publication,
    build_candidate_publication_config,
)

__all__ = [
    "generate_all_tables",
    "generate_publication_tables",
    "generate_historical_snapshot_tables",
    "generate_table_1_metrics",
    "generate_table_2_profiling",
    "generate_table_3_noise_degradation",
    "generate_table_4_pairwise_comparison",
    "generate_table_8_motifs_summary",
    "generate_table_10_boundary_triplets",
    "generate_table_11_architecture_correlations",
    "generate_table_12_voter_consensus",
    "generate_all_figures",
    "verify_latex_figure_paths",
    "verify_figure_publication_readiness",
    "build_manuscript_variants",
    "compile_manuscript",
    "generate_manuscript_content",
    "validate_artifact_provenance",
    "validate_selection",
    "assemble_publication",
    "build_candidate_publication_config",
]

