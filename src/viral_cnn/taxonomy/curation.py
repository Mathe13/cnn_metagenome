"""
Modular taxonomy curation, ICTV harmonization, and hybrid taxonomy cleansing.

Scientific invariants (AD-001, AD-002, AD-004):
- Preserves 1-based taxonomic IDs in metadata and 0-based model labels: model_label = id - 1.
- Standardizes 8-level ICTV hierarchical ranks: Realm, Kingdom, Phylum, Class, Order, Family, Genus, Species.
- Supports deterministic offline synthetic taxonomy generation for smoke-testing and CI.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from viral_cnn.taxonomy.mapping import (
    model_label_to_taxonomic_id,
    taxonomic_id_to_model_label,
)
from viral_cnn.utils.logging_utils import log_artifact_path

logger = logging.getLogger(__name__)

ICTV_RANKS: Tuple[str, ...] = (
    "Realm",
    "Kingdom",
    "Phylum",
    "Class",
    "Order",
    "Family",
    "Genus",
    "Species",
)


def build_ictv_rename_map(df_renames: pd.DataFrame) -> Dict[str, str]:
    """
    Parses the ICTV 'Taxa Renamed or Abolished' sheet and extracts a dictionary
    mapping lowercase old species names to lowercase official updated names.
    """
    if "Change" not in df_renames.columns or "Old Name" not in df_renames.columns or "New Name" not in df_renames.columns:
        raise ValueError("df_renames must contain columns 'Change', 'Old Name', and 'New Name'.")

    rename_map: Dict[str, str] = {}
    is_renamed = df_renames["Change"].astype(str).str.contains("Renamed", na=False, case=False)
    filtered = df_renames[is_renamed]

    for _, row in filtered.iterrows():
        old_full = str(row["Old Name"])
        new_full = str(row["New Name"])

        # ICTV stores full lineage (Realm;...;Genus;Species); extract the last element
        old_species = old_full.split(";")[-1].strip().lower()
        new_species = new_full.split(";")[-1].strip().lower()

        if old_species and new_species and old_species != new_species:
            rename_map[old_species] = new_species

    logger.info("Built ICTV rename map with %d rules.", len(rename_map))
    return rename_map


def match_ictv_taxonomy(
    species_df: pd.DataFrame,
    ictv_master: pd.DataFrame,
    rename_map: Optional[Dict[str, str]] = None,
    fuzzy_cutoff: float = 90.0,
    species_col: str = "species",
) -> pd.DataFrame:
    """
    Matches species names against the ICTV Master Species List using:
      1. Direct exact match (case-insensitive).
      2. Renames lookup match via rename_map.
      3. RapidFuzz fuzzy similarity match (if rapidfuzz is available and score >= fuzzy_cutoff).
      4. Fallback attribution for remaining species.
    """
    df = species_df.copy()
    if species_col not in df.columns:
        raise ValueError(f"Column '{species_col}' not found in species_df.")
    if "Species" not in ictv_master.columns:
        raise ValueError("ictv_master must contain column 'Species'.")

    # Clean species names
    df["species_clean"] = df[species_col].astype(str).str.strip().str.lower()
    if rename_map:
        df["species_ictv"] = df["species_clean"].map(rename_map).fillna(df["species_clean"])
    else:
        df["species_ictv"] = df["species_clean"]

    ictv_species_clean = ictv_master["Species"].astype(str).str.strip().dropna().unique()
    ictv_master_set = set(s.lower() for s in ictv_species_clean)
    ictv_case_map = {s.lower(): s for s in ictv_species_clean}

    # Optional rapidfuzz support
    try:
        from rapidfuzz import fuzz, process
        has_rapidfuzz = True
    except ImportError:
        has_rapidfuzz = False

    statuses: List[str] = []
    validated_names: List[str] = []

    for _, row in df.iterrows():
        target = row["species_ictv"]
        orig = row["species_clean"]

        if target in ictv_master_set:
            statuses.append("Homologado (ICTV Direct)")
            validated_names.append(ictv_case_map.get(target, target))
        elif orig != target and target in ictv_master_set:
            statuses.append("Atualizado via Renames (ICTV)")
            validated_names.append(ictv_case_map.get(target, target))
        elif has_rapidfuzz and len(ictv_master_set) > 0:
            match = process.extractOne(target, ictv_master_set, scorer=fuzz.WRatio, score_cutoff=fuzzy_cutoff)
            if match:
                matched_name, score, _ = match
                statuses.append(f"Homologado via Similaridade (Score: {int(score)}%)")
                validated_names.append(ictv_case_map.get(matched_name, matched_name))
            else:
                statuses.append("Não Encontrado no ICTV (NCBI Fallback)")
                validated_names.append(target)
        else:
            statuses.append("Não Encontrado no ICTV (NCBI Fallback)")
            validated_names.append(target)

    df["ictv_status"] = statuses
    df["species_validated_name"] = validated_names
    return df


def resolve_local_ictv_taxonomy(
    df: pd.DataFrame,
    ictv_master: pd.DataFrame,
    species_col: str = "species_ictv",
) -> pd.DataFrame:
    """
    Enriches a species DataFrame with lineage columns from ICTV Master Species List:
    Realm, Kingdom, Phylum, Class, Order, Family, Genus.
    """
    df_out = df.copy()
    if species_col not in df_out.columns:
        df_out[species_col] = df_out["species"].astype(str).str.strip().str.lower()

    ranks = ["Realm", "Kingdom", "Phylum", "Class", "Order", "Family", "Genus"]
    missing_ranks = [r for r in ranks if r not in ictv_master.columns]
    if missing_ranks:
        raise ValueError(f"ictv_master is missing expected taxonomic ranks: {missing_ranks}")

    master_sub = ictv_master[["Species"] + ranks].copy()
    master_sub["_species_key"] = master_sub["Species"].astype(str).str.strip().str.lower()
    master_sub = master_sub.drop_duplicates(subset=["_species_key"])

    df_out["_species_key"] = df_out[species_col].astype(str).str.strip().str.lower()

    merged = df_out.merge(master_sub, on="_species_key", how="left", suffixes=("", "_ictv"))
    merged.drop(columns=["_species_key"], inplace=True)

    # Determine taxonomy source
    matched = merged["Order"].notna() | merged["Genus"].notna()
    merged["taxonomy_source"] = np.where(matched, "ICTV Master", "NCBI Fallback")

    return merged


def sanitize_hybrid_taxonomy(
    df: pd.DataFrame, placeholder: str = "Unassigned"
) -> pd.DataFrame:
    """
    Ensures 0-based model_label and 1-based id consistency, backfills missing ranks,
    and constructs full_lineage.
    """
    out = df.copy()

    # Model label and ID alignment (AD-002)
    if "id" in out.columns and "model_label" not in out.columns:
        out["model_label"] = out["id"].apply(taxonomic_id_to_model_label)
    elif "model_label" in out.columns and "id" not in out.columns:
        out["id"] = out["model_label"].apply(model_label_to_taxonomic_id)
    elif "id" in out.columns and "model_label" in out.columns:
        # Verify alignment
        mismatch = (out["model_label"] != out["id"] - 1).sum()
        if mismatch > 0:
            out["model_label"] = out["id"].apply(taxonomic_id_to_model_label)

    ranks = ["Realm", "Kingdom", "Phylum", "Class", "Order", "Family", "Genus"]
    for r in ranks:
        if r not in out.columns:
            out[r] = f"{placeholder}_{r}"
        else:
            out[r] = out[r].fillna(f"{placeholder}_{r}").replace("", f"{placeholder}_{r}")

    if "species" not in out.columns and "Species" in out.columns:
        out["species"] = out["Species"]

    # Build full lineage string if missing
    if "full_lineage" not in out.columns:
        lineage_cols = [r for r in ["Realm", "Kingdom", "Phylum", "Class", "Order", "Family", "Genus", "species"] if r in out.columns]
        out["full_lineage"] = out[lineage_cols].apply(
            lambda row: " / ".join([str(val) for val in row if pd.notna(val)]), axis=1
        )

    return out


def generate_synthetic_taxonomy(num_classes: int = 100) -> pd.DataFrame:
    """
    Generates a deterministic synthetic taxonomy DataFrame representing num_classes
    with complete hierarchical ranks for testing and offline execution.
    """
    records = []
    for i in range(num_classes):
        tax_id = i + 1
        model_label = i
        order_idx = i // 20
        family_idx = i // 10
        genus_idx = i // 5

        records.append({
            "id": tax_id,
            "model_label": model_label,
            "species": f"Synthetic virus {tax_id}",
            "Genus": f"Syntheticgenus_{genus_idx}",
            "Family": f"Syntheticfamily_{family_idx}",
            "Order": f"Syntheticorder_{order_idx}",
            "Class": "Syntheticclass_0",
            "Phylum": "Syntheticphylum_0",
            "Kingdom": "Orthornavirae",
            "Realm": "Riboviria",
            "taxonomy_source": "Synthetic",
            "full_lineage": f"Riboviria / Orthornavirae / Syntheticphylum_0 / Syntheticclass_0 / Syntheticorder_{order_idx} / Syntheticfamily_{family_idx} / Syntheticgenus_{genus_idx} / Synthetic virus {tax_id}",
        })

    df = pd.DataFrame(records)
    log_artifact_path("synthetic taxonomy", f"<synthetic_{num_classes}_classes>")
    return df


def build_hybrid_taxonomy(
    species_labels: pd.DataFrame,
    ictv_master: pd.DataFrame,
    ictv_renames: Optional[pd.DataFrame] = None,
    ncbi_df: Optional[pd.DataFrame] = None,
    fuzzy_cutoff: float = 90.0,
) -> pd.DataFrame:
    """
    Full taxonomy curation pipeline:
      1. Builds rename map if ictv_renames provided.
      2. Matches species against ICTV master with rename and fuzzy matching.
      3. Resolves local ICTV lineage ranks.
      4. Fills missing ranks from ncbi_df if provided.
      5. Sanitizes ranks and aligns label offsets.
    """
    rename_map = build_ictv_rename_map(ictv_renames) if ictv_renames is not None else None
    matched_df = match_ictv_taxonomy(
        species_labels,
        ictv_master,
        rename_map=rename_map,
        fuzzy_cutoff=fuzzy_cutoff,
        species_col="species" if "species" in species_labels.columns else "Species",
    )
    resolved_df = resolve_local_ictv_taxonomy(matched_df, ictv_master, species_col="species_ictv")

    if ncbi_df is not None and not ncbi_df.empty:
        # Merge NCBI entries for non-ICTV matches
        fallback_mask = resolved_df["taxonomy_source"] == "NCBI Fallback"
        if fallback_mask.any():
            ranks = ["Realm", "Kingdom", "Phylum", "Class", "Order", "Family", "Genus"]
            ncbi_sub = ncbi_df[["species"] + [r for r in ranks if r in ncbi_df.columns]].copy()
            ncbi_sub["_sp_key"] = ncbi_sub["species"].astype(str).str.strip().str.lower()
            ncbi_sub = ncbi_sub.drop_duplicates(subset=["_sp_key"])

            resolved_df["_sp_key"] = resolved_df["species"].astype(str).str.strip().str.lower()
            merged_ncbi = resolved_df[fallback_mask].merge(ncbi_sub, on="_sp_key", how="left", suffixes=("", "_ncbi"))
            for r in ranks:
                if f"{r}_ncbi" in merged_ncbi.columns:
                    resolved_df.loc[fallback_mask, r] = merged_ncbi[f"{r}_ncbi"].values
            resolved_df.drop(columns=["_sp_key"], inplace=True)

    return sanitize_hybrid_taxonomy(resolved_df)

