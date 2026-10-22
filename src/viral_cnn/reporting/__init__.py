"""Reporting, tables, figures, and manuscript generation modules."""

from viral_cnn.reporting.tables import (
    generate_all_tables,
    generate_table_1_metrics,
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
)

__all__ = [
    "generate_all_tables",
    "generate_table_1_metrics",
    "generate_table_3_noise_degradation",
    "generate_table_4_pairwise_comparison",
    "generate_table_8_motifs_summary",
    "generate_table_10_boundary_triplets",
    "generate_table_11_architecture_correlations",
    "generate_table_12_voter_consensus",
    "generate_all_figures",
    "verify_latex_figure_paths",
]

