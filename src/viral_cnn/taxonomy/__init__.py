"""Taxonomic mapping and label alignment."""
from .mapping import (
    taxonomic_id_to_model_label,
    model_label_to_taxonomic_id,
    load_taxonomy_mapping,
)

__all__ = [
    "taxonomic_id_to_model_label",
    "model_label_to_taxonomic_id",
    "load_taxonomy_mapping",
]

