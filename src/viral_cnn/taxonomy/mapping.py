"""
Taxonomic mapping helpers and offset conventions.

Scientific invariant (AGENTS.md):
- CSV files in `dataset_kraken/*_name_ids.csv` use 1-based IDs: `species,id` [1 .. 13683].
- Model classifier heads output 0-based logits: [0 .. 13682].
- Strict conversion offset:
    model_label = taxonomic_id - 1
    taxonomic_id = model_label + 1
"""

import csv
import os
from pathlib import Path

from viral_cnn.utils.logging_utils import log_artifact_path


def taxonomic_id_to_model_label(taxonomic_id: int) -> int:
    """Converts a 1-based CSV taxonomic ID into a 0-based model class index."""
    if taxonomic_id < 1:
        raise ValueError(f"Taxonomic ID must be >= 1, got {taxonomic_id}")
    return taxonomic_id - 1


def model_label_to_taxonomic_id(model_label: int) -> int:
    """Converts a 0-based model class index into a 1-based CSV taxonomic ID."""
    if model_label < 0:
        raise ValueError(f"Model label must be >= 0, got {model_label}")
    return model_label + 1


def load_taxonomy_mapping(
    csv_path: str | Path, rank_column: str = "species"
) -> dict[int, str]:
    """
    Loads a taxonomy mapping CSV and returns a dictionary:
      {model_label (0-based int): taxon_name (str)}
    """
    csv_file = Path(csv_path)
    if not csv_file.exists():
        raise FileNotFoundError(f"Taxonomy CSV not found: {csv_file}")

    log_artifact_path("taxonomy mapping", csv_file, extra=f"rank: {rank_column}")

    mapping: dict[int, str] = {}
    with open(csv_file, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            tax_id = int(row["id"])
            taxon_name = row[rank_column]
            model_label = taxonomic_id_to_model_label(tax_id)
            mapping[model_label] = taxon_name
    return mapping

