"""Baselines and comparative benchmarking modules for Kraken2, BLASTN, and k-mer models."""

from viral_cnn.baselines.taxonomy import parse_kraken_output, parse_blast_output
from viral_cnn.baselines.overlap import audit_exact_overlap
from viral_cnn.baselines.paired import (
    wilson_score_interval,
    evaluate_predictions,
    paired_discordance,
    bootstrap_species_difference,
)
from viral_cnn.baselines.long_eval import (
    integrate_long_voter_predictions,
    evaluate_long_baselines,
)
from viral_cnn.baselines.kmer_baseline import KmerCentroidClassifier

__all__ = [
    "parse_kraken_output",
    "parse_blast_output",
    "audit_exact_overlap",
    "wilson_score_interval",
    "evaluate_predictions",
    "paired_discordance",
    "bootstrap_species_difference",
    "integrate_long_voter_predictions",
    "evaluate_long_baselines",
    "KmerCentroidClassifier",
]
