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
from typing import Optional, Union

import numpy as np
import pandas as pd
import torch

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


def build_species_to_genus_tensor(
    taxonomy_parquet_path: Union[str, Path],
    num_species: int = 13683,
    device: Union[str, torch.device] = "cpu",
) -> torch.Tensor:
    """
    Builds a PyTorch LongTensor mapping species model labels (0 .. num_species - 1)
    to contiguous genus indices (0 .. N_genera - 1) from a curated taxonomy Parquet.

    Scientific invariant & contract:
    - Species model labels are strictly 0-indexed [0 .. num_species - 1].
    - Genus indices are strictly 0-indexed [0 .. N_genera - 1] and contiguous.
    - Rejects degenerate mappings where all species map to a single genus (raises ValueError).
    - Requires tensor length to match num_species exactly (raises ValueError).
    """
    parquet_file = Path(taxonomy_parquet_path)
    if not parquet_file.exists():
        raise FileNotFoundError(f"Taxonomy Parquet not found: {parquet_file}")

    log_artifact_path("taxonomy parquet for species-to-genus", parquet_file)
    df = pd.read_parquet(parquet_file)

    # Resolve genus column
    genus_col: Optional[str] = None
    for cand in ["genus_idx", "genus_id", "Genus", "genus"]:
        if cand in df.columns:
            genus_col = cand
            break

    if genus_col is None:
        raise ValueError(
            f"Taxonomy Parquet at {parquet_file} must contain one of: "
            f"['genus_idx', 'genus_id', 'Genus', 'genus']. Found columns: {list(df.columns)}"
        )

    # Encode genus as dense categories
    df_clean = df.copy()
    if genus_col == "genus_idx":
        genus_series = df_clean["genus_idx"].fillna(-1).astype(int)
    else:
        # Categorical codes
        genus_series = df_clean[genus_col].fillna("Unknown_Genus").astype("category").cat.codes

    df_clean["_genus_code"] = genus_series.values

    # Map model labels [0 .. num_species - 1]
    mapping_arr = np.full(num_species, fill_value=-1, dtype=np.int64)

    if "model_label" in df_clean.columns:
        for _, row in df_clean.iterrows():
            lbl = int(row["model_label"])
            if 0 <= lbl < num_species:
                mapping_arr[lbl] = int(row["_genus_code"])
    elif "id" in df_clean.columns:
        for _, row in df_clean.iterrows():
            lbl = taxonomic_id_to_model_label(int(row["id"]))
            if 0 <= lbl < num_species:
                mapping_arr[lbl] = int(row["_genus_code"])
    else:
        if len(df_clean) == num_species:
            mapping_arr = df_clean["_genus_code"].values.astype(np.int64)
        else:
            raise ValueError(
                f"Cannot map taxonomy to model labels: missing 'model_label' or 'id' column, "
                f"and dataframe length ({len(df_clean)}) != num_species ({num_species})."
            )

    unassigned_count = int(np.sum(mapping_arr < 0))
    if unassigned_count > 0:
        raise ValueError(
            f"Incomplete taxonomy mapping: {unassigned_count} species out of {num_species} "
            f"have no genus assignment in {parquet_file}."
        )

    # Re-index genus codes to contiguous 0 .. N_genera - 1
    unique_genera, inverse_indices = np.unique(mapping_arr, return_inverse=True)
    contiguous_mapping = inverse_indices.astype(np.int64)

    if len(contiguous_mapping) != num_species:
        raise ValueError(
            f"Species-to-genus mapping length {len(contiguous_mapping)} does not match "
            f"num_species {num_species}."
        )

    if len(unique_genera) <= 1 and num_species > 1:
        raise ValueError(
            f"Degenerate taxonomy mapping detected in {parquet_file}: "
            f"all {num_species} species mapped to a single genus (unique_genera={len(unique_genera)})."
        )

    tensor = torch.tensor(contiguous_mapping, dtype=torch.long, device=device)
    return tensor


