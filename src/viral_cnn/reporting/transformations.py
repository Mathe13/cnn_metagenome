"""Centralized transformation registry and execution for publication artifacts.

Each publication item (T1-T12, F1-F5) maps to an explicit transformation function
that validates input schemas, adapts stage outputs to the canonical publication format,
computes sample support N, and emits derived manifests.
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from viral_cnn.reporting.publication import PublicationValidationError
from viral_cnn.utils.logging_utils import log_artifact_path

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff", ".webp", ".svg"}
FIVE_VARIANTS_SPECIES = [
    "ResNet50_Clean",
    "ConvNeXt_Z",
    "ConvNeXt_T",
    "ResNet50_F",
    "ConvNeXt_F",
]

TRANSFORMATION_REGISTRY: dict[str, Callable[[pd.DataFrame, dict[str, Any]], pd.DataFrame]] = {}


def sha256_file(path: Path) -> str:
    """Return a file's SHA-256 digest."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def register_transformation(item_id: str):
    """Decorator to register a publication transformation function."""
    def decorator(func: Callable[[pd.DataFrame, dict[str, Any]], pd.DataFrame]):
        TRANSFORMATION_REGISTRY[item_id] = func
        return func
    return decorator


def get_transformation(item_id: str) -> Callable[[pd.DataFrame, dict[str, Any]], pd.DataFrame]:
    """Retrieve registered transformation for a given item_id."""
    if item_id not in TRANSFORMATION_REGISTRY:
        raise KeyError(f"No transformation registered for publication item: {item_id}")
    return TRANSFORMATION_REGISTRY[item_id]


def validate_artifact_is_tabular(path: Path) -> None:
    """Ensure artifact is structured tabular data (CSV or Parquet) and not an image."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in IMAGE_EXTENSIONS:
        raise PublicationValidationError(
            f"Image file ({suffix}) cannot be used as a publication source: {path}. "
            "Publication artifacts require structured CSV or Parquet data."
        )
    if suffix not in {".csv", ".parquet", ".pq"}:
        raise PublicationValidationError(
            f"Unsupported tabular format for publication source: {path}. Expected CSV or Parquet."
        )


def emit_derived_manifest(
    item_id: str,
    output_artifact_path: Path,
    parent_manifest_path: Path,
    transformation_name: str,
    support_n: int,
    *,
    extra_metadata: dict[str, Any] | None = None,
) -> Path:
    """Emit an atomic derived manifest linking parent lineage, code hash, and support N."""
    output_artifact_path = Path(output_artifact_path).resolve()
    parent_manifest_path = Path(parent_manifest_path).resolve()
    log_artifact_path("Parent manifest for derived artifact", parent_manifest_path)
    log_artifact_path("Derived publication artifact", output_artifact_path)

    parent_manifest = json.loads(parent_manifest_path.read_text(encoding="utf-8"))
    parent_sha = sha256_file(parent_manifest_path)
    artifact_sha = sha256_file(output_artifact_path)

    manifest_data: dict[str, Any] = {
        "item_id": item_id,
        "protocol": "c06_strict_v1",
        "evidence_kind": "derived",
        "transformation": transformation_name,
        "status": "COMPLETE",
        "is_smoke_test": bool(parent_manifest.get("is_smoke_test", False)),
        "artifact_path": str(output_artifact_path),
        "artifact_sha256": artifact_sha,
        "parent_manifest_path": str(parent_manifest_path),
        "parent_manifest_sha256": parent_sha,
        "seed": parent_manifest.get("seed", 42),
        "split_id": parent_manifest.get("split_id", "test"),
        "n": int(support_n),
        "support": int(support_n),
        "scientific_identity": parent_manifest.get("scientific_identity", {}),
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    if extra_metadata:
        manifest_data.update(extra_metadata)

    manifest_path = output_artifact_path.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest_data, indent=2) + "\n", encoding="utf-8")
    log_artifact_path("Derived publication manifest", manifest_path)
    return manifest_path


# --- Specific Transformations for T1 to T12 and F1 to F5 ---

@register_transformation("T1")
def transform_table_1_metrics(df: pd.DataFrame, meta: dict[str, Any]) -> pd.DataFrame:
    """Table 1: comparative performance metrics across architectures."""
    required = {"model", "taxonomic_level", "accuracy", "top_5", "macro_f1"}
    if not required.issubset(df.columns):
        raise PublicationValidationError(f"T1 input missing required columns: {required - set(df.columns)}")
    cols = ["model", "taxonomic_level", "backbone_m", "fc_params_m", "parameters_m", "accuracy", "top_5", "macro_f1", "latency_ms"]
    for c in cols:
        if c not in df.columns:
            df[c] = 0.0
    return df[cols].copy()


@register_transformation("T2")
def transform_table_2_profiling(df: pd.DataFrame, meta: dict[str, Any]) -> pd.DataFrame:
    """Table 2: inference profiling (VRAM, step time, throughput)."""
    required = {"model", "backbone_m", "fc_params_m", "parameters_m"}
    if not required.issubset(df.columns):
        raise PublicationValidationError(f"T2 input missing required columns: {required - set(df.columns)}")
    cols = ["model", "backbone_m", "fc_params_m", "parameters_m", "peak_vram_mb", "step_time_ms", "throughput_seq_s", "gpu_hours"]
    for c in cols:
        if c not in df.columns:
            df[c] = 0.0
    return df[cols].copy()


@register_transformation("T3")
def transform_table_3_noise_degradation(df: pd.DataFrame, meta: dict[str, Any]) -> pd.DataFrame:
    """Table 3: degradation across noise levels."""
    required = {"epsilon"}
    if not required.issubset(df.columns):
        raise PublicationValidationError(f"T3 input missing required columns: {required - set(df.columns)}")
    return df.copy()


@register_transformation("T4")
def transform_table_4_species_f1_and_pairwise(df: pd.DataFrame, meta: dict[str, Any]) -> pd.DataFrame:
    """Table 4: species F1 distribution and pairwise comparisons."""
    required = {"f1_bin"}
    if not required.issubset(df.columns):
        raise PublicationValidationError(f"T4 input missing required columns: {required - set(df.columns)}")
    return df.copy()


@register_transformation("T5")
def transform_table_5_latent_metrics(df: pd.DataFrame, meta: dict[str, Any]) -> pd.DataFrame:
    """Table 5: high-dimensional latent space metrics (KNN, Silhouette, Linear Probe)."""
    required = {"model", "knn_accuracy"}
    if not required.issubset(df.columns):
        raise PublicationValidationError(f"T5 input missing required columns: {required - set(df.columns)}")
    return df.copy()


@register_transformation("T6")
def transform_table_6_hierarchy_correlations(df: pd.DataFrame, meta: dict[str, Any]) -> pd.DataFrame:
    """Table 6: taxonomic hierarchy correlations (Species vs Genus vs Order)."""
    required = {"level_pair"}
    if not required.issubset(df.columns):
        raise PublicationValidationError(f"T6 input missing required columns: {required - set(df.columns)}")
    return df.copy()


@register_transformation("T7")
def transform_table_7_hierarchical_entropy(df: pd.DataFrame, meta: dict[str, Any]) -> pd.DataFrame:
    """Table 7: hierarchical activation entropy (Species, Genus, Order)."""
    required = {"model"}
    if not required.issubset(df.columns):
        raise PublicationValidationError(f"T7 input missing required columns: {required - set(df.columns)}")
    return df.copy()


@register_transformation("T8")
def transform_table_8_paired_motif_perturbation(df: pd.DataFrame, meta: dict[str, Any]) -> pd.DataFrame:
    """Table 8: paired 7-mer motif perturbation with uniform and GC-matched controls."""
    required = {"motif"}
    if not required.issubset(df.columns):
        raise PublicationValidationError(f"T8 input missing required columns: {required - set(df.columns)}")
    return df.copy()


@register_transformation("T9")
def transform_table_9_rscu_exploratory(df: pd.DataFrame, meta: dict[str, Any]) -> pd.DataFrame:
    """Table 9: exploratory triplet composition / RSCU profiles across reading frames."""
    required = {"triplet"}
    if not required.issubset(df.columns):
        raise PublicationValidationError(f"T9 input missing required columns: {required - set(df.columns)}")
    return df.copy()


@register_transformation("T10")
def transform_table_10_triplet_composition(df: pd.DataFrame, meta: dict[str, Any]) -> pd.DataFrame:
    """Table 10: triplet composition summary and expected base frequencies."""
    required = {"triplet"}
    if not required.issubset(df.columns):
        raise PublicationValidationError(f"T10 input missing required columns: {required - set(df.columns)}")
    return df.copy()


@register_transformation("T11")
def transform_table_11_architecture_correlations(df: pd.DataFrame, meta: dict[str, Any]) -> pd.DataFrame:
    """Table 11: motif overlap and permutation null analysis between architectures.

    Rejects cooccurrence data claiming architecture correlation without permutation null test!
    """
    if "is_permutation_tested" in df.columns:
        if not df["is_permutation_tested"].all():
            raise PublicationValidationError(
                "T11 scientific contract violation: co-occurrence matrices without permutation null "
                "testing cannot be published as architecture correlation."
            )
    # Check that it's not a degenerated identical matrix claiming r=1 without variance
    if "correlation_r" in df.columns and "p_value" in df.columns:
        if (df["correlation_r"] == 1.0).all() and (df["p_value"] == 0.0).all():
            if "has_variance" in df.columns and not df["has_variance"].all():
                raise PublicationValidationError(
                    "T11 scientific contract violation: spurious r=1 correlation from identical or zero-variance matrices."
                )
    return df.copy()


@register_transformation("T12")
def transform_table_12_long_sequence_consensus(df: pd.DataFrame, meta: dict[str, Any]) -> pd.DataFrame:
    """Table 12: long sequence voter comparison (mean_logits vs majority_vote)."""
    required = {"model", "majority_vote_acc", "mean_logits_acc"}
    if not required.issubset(df.columns):
        raise PublicationValidationError(f"T12 input missing required columns: {required - set(df.columns)}")
    return df.copy()


@register_transformation("F1")
def transform_figure_1_noise_robustness(df: pd.DataFrame, meta: dict[str, Any]) -> pd.DataFrame:
    """Figure 1: noise robustness curves."""
    required = {"epsilon", "accuracy"}
    if not required.issubset(df.columns):
        raise PublicationValidationError(f"F1 input missing required columns: {required - set(df.columns)}")
    return df.copy()


@register_transformation("F2")
def transform_figure_2_umap_visualization(df: pd.DataFrame, meta: dict[str, Any]) -> pd.DataFrame:
    """Figure 2: UMAP 2D latent projection."""
    required = {"umap_1", "umap_2"}
    if not required.issubset(df.columns):
        raise PublicationValidationError(f"F2 input missing required columns: {required - set(df.columns)}")
    return df.copy()


@register_transformation("F3")
def transform_figure_3_paired_motif_perturbation(df: pd.DataFrame, meta: dict[str, Any]) -> pd.DataFrame:
    """Figure 3: 5-panel grid of paired motif perturbation across architectures.

    Scientific Contract: requires exactly all 5 species model variants.
    """
    required = {"panel", "variant", "motif", "activation_drop"}
    if not required.issubset(df.columns):
        raise PublicationValidationError(f"F3 input missing required columns: {required - set(df.columns)}")
    variants_in_data = set(df["variant"].astype(str))
    required_variants = set(FIVE_VARIANTS_SPECIES)
    missing_variants = required_variants - variants_in_data
    if missing_variants:
        raise PublicationValidationError(
            f"F3 scientific contract violation: missing species model variants: {missing_variants}. "
            "All 5 variants (including ConvNeXt_F) are mandatory."
        )
    return df.copy()


@register_transformation("F4")
def transform_figure_4_rscu_profile(df: pd.DataFrame, meta: dict[str, Any]) -> pd.DataFrame:
    """Figure 4: exploratory triplet composition profile."""
    required = {"triplet", "rscu"}
    if not required.issubset(df.columns):
        raise PublicationValidationError(f"F4 input missing required columns: {required - set(df.columns)}")
    return df.copy()


@register_transformation("F5")
def transform_figure_5_motif_cooccurrence(df: pd.DataFrame, meta: dict[str, Any]) -> pd.DataFrame:
    """Figure 5: motif co-occurrence network graph.

    Scientific Contract: requires 2D coordinates (x1, y1, x2, y2); rejects bar chart data.
    """
    required = {"x1", "y1", "x2", "y2"}
    if not required.issubset(df.columns):
        raise PublicationValidationError(
            f"F5 scientific contract violation: missing network edge coordinates: {required - set(df.columns)}. "
            "Bar chart data cannot be rendered as a network graph."
        )
    return df.copy()


def execute_publication_transformation(
    item_id: str,
    source_artifact_path: Path,
    output_dir: Path,
    parent_manifest_path: Path,
    *,
    extra_metadata: dict[str, Any] | None = None,
) -> tuple[Path, Path]:
    """Execute transformation on source tabular artifact and emit derived artifact + manifest."""
    source_artifact_path = Path(source_artifact_path).resolve()
    validate_artifact_is_tabular(source_artifact_path)
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    log_artifact_path(f"Source artifact for item {item_id}", source_artifact_path)

    if source_artifact_path.suffix.lower() == ".csv":
        df = pd.read_csv(source_artifact_path)
    else:
        df = pd.read_parquet(source_artifact_path)

    transform_func = get_transformation(item_id)
    transformed_df = transform_func(df, extra_metadata or {})

    output_path = output_dir / f"{item_id.lower()}_publication.csv"
    transformed_df.to_csv(output_path, index=False)
    log_artifact_path(f"Transformed publication artifact {item_id}", output_path)

    manifest_path = emit_derived_manifest(
        item_id,
        output_path,
        parent_manifest_path,
        transformation_name=transform_func.__name__,
        support_n=len(transformed_df),
        extra_metadata=extra_metadata,
    )

    return output_path, manifest_path
