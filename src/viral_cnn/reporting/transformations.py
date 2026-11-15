"""Centralized transformation registry and execution for publication artifacts.

Each publication item (T1-T12, F1-F5) maps to an explicit transformation function
that validates input schemas, adapts stage outputs to the canonical publication format,
computes sample support N, and emits derived manifests.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from viral_cnn.reporting.publication import PublicationValidationError, _git_state
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


def _compute_transformation_code_hash(name: str, item_id: str | None = None) -> str:
    """Return SHA-256 digest of transformation function source code."""
    func = None
    if item_id and item_id in TRANSFORMATION_REGISTRY:
        func = TRANSFORMATION_REGISTRY[item_id]
    elif name:
        for f in TRANSFORMATION_REGISTRY.values():
            if f.__name__ == name:
                func = f
                break
    if func is not None:
        try:
            import inspect
            src = inspect.getsource(func)
            return hashlib.sha256(src.encode("utf-8")).hexdigest()
        except Exception:
            pass
    return hashlib.sha256((name or "identity").encode("utf-8")).hexdigest()


def emit_derived_manifest(
    item_id: str,
    output_artifact_path: Path,
    parent_manifest_path: Path,
    source_artifact_path: Path | str | None = None,
    transformation_name: str | None = None,
    support_n: int | None = None,
    *,
    output_rows: int | None = None,
    extra_metadata: dict[str, Any] | None = None,
) -> Path:
    """Emit an atomic derived manifest linking parent lineage, code hash, and support N."""
    # Backward compatibility with positional (item_id, output_path, parent_path, transformation_name, support_n)
    if isinstance(transformation_name, (int, float)) and support_n is None:
        support_n = int(transformation_name)
        transformation_name = str(source_artifact_path)
        source_artifact_path = None
    elif source_artifact_path is not None and transformation_name is None:
        p = Path(str(source_artifact_path))
        if not p.is_file() and not p.suffix:
            transformation_name = str(source_artifact_path)
            source_artifact_path = None

    output_artifact_path = Path(output_artifact_path).resolve()
    parent_manifest_path = Path(parent_manifest_path).resolve()
    log_artifact_path("Parent manifest for derived artifact", parent_manifest_path)
    log_artifact_path("Derived publication artifact", output_artifact_path)

    parent_manifest = json.loads(parent_manifest_path.read_text(encoding="utf-8"))
    parent_sha = sha256_file(parent_manifest_path)
    artifact_sha = sha256_file(output_artifact_path)

    if source_artifact_path is not None:
        source_path = Path(source_artifact_path).resolve()
    elif parent_manifest.get("artifact_path"):
        source_path = Path(parent_manifest["artifact_path"]).resolve()
    else:
        source_path = None

    if source_path and source_path.is_file():
        source_sha = sha256_file(source_path)
        log_artifact_path("Source artifact for derived artifact", source_path)
    else:
        source_sha = ""

    git_commit = parent_manifest.get("git_commit")
    if not git_commit:
        project_root = next(
            (p for p in output_artifact_path.parents if (p / ".git").exists() or (p / "src" / "viral_cnn").is_dir()),
            output_artifact_path.parent,
        )
        git_commit, _ = _git_state(project_root)

    versions = parent_manifest.get("library_versions") or parent_manifest.get("versions")
    if not versions or not isinstance(versions, dict):
        versions = {"python": platform.python_version()}
        for pkg in ("torch", "numpy", "pandas", "pyarrow", "matplotlib", "PyYAML"):
            try:
                versions[pkg] = importlib.metadata.version(pkg)
            except importlib.metadata.PackageNotFoundError:
                versions[pkg] = "unavailable"

    # Compute output_rows if not provided
    if output_rows is None and output_artifact_path.is_file():
        try:
            if output_artifact_path.suffix.lower() == ".csv":
                output_rows = len(pd.read_csv(output_artifact_path, comment="#"))
            elif output_artifact_path.suffix.lower() in {".parquet", ".pq"}:
                output_rows = len(pd.read_parquet(output_artifact_path))
        except Exception:
            output_rows = None

    # Experimental support: parent audited support takes precedence over output_rows
    resolved_n = parent_manifest.get("n")
    if resolved_n is None and support_n is not None:
        resolved_n = support_n

    resolved_support = parent_manifest.get("support")
    if resolved_support is None:
        if parent_manifest.get("n") is not None:
            resolved_support = parent_manifest["n"]
        elif support_n is not None:
            resolved_support = support_n

    resolved_denom = parent_manifest.get("denominator_n")
    if resolved_denom is None and resolved_n is not None:
        resolved_denom = resolved_n

    inputs = [{"path": str(parent_manifest_path), "sha256": parent_sha}]
    if source_path and source_path.is_file():
        inputs.append({"path": str(source_path), "sha256": source_sha})

    trans_code_sha = _compute_transformation_code_hash(transformation_name or "", item_id)

    manifest_data: dict[str, Any] = {
        "item_id": item_id,
        "protocol": parent_manifest.get("protocol", "c06_strict_v1"),
        "evidence_kind": "derived",
        "transformation": transformation_name,
        "status": "COMPLETE",
        "is_smoke_test": bool(parent_manifest.get("is_smoke_test", False)),
        "synthetic": bool(parent_manifest.get("synthetic", False)),
        "artifact_path": str(output_artifact_path),
        "artifact_sha256": artifact_sha,
        "parent_manifest_path": str(parent_manifest_path),
        "parent_manifest_sha256": parent_sha,
        "git_commit": git_commit or "unknown",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "library_versions": versions,
        "configuration_sha256": parent_manifest.get("configuration_sha256", parent_sha),
        "inputs": inputs,
        "parent_artifacts": [{"path": str(parent_manifest_path), "sha256": parent_sha}],
        "derivation": {
            "source_artifact": {
                "path": str(source_path) if source_path else "",
                "sha256": source_sha,
            },
            "parent_manifest": {
                "path": str(parent_manifest_path),
                "sha256": parent_sha,
            },
            "transformation": {
                "name": transformation_name or "",
                "code_sha256": trans_code_sha,
            },
        },
        "output_rows": int(output_rows) if output_rows is not None else None,
    }

    if resolved_n is not None:
        manifest_data["n"] = int(resolved_n)
    elif parent_manifest.get("n_not_applicable_reason"):
        manifest_data["n_not_applicable_reason"] = parent_manifest["n_not_applicable_reason"]

    if resolved_support is not None:
        manifest_data["support"] = int(resolved_support)
    elif parent_manifest.get("support_not_applicable_reason"):
        manifest_data["support_not_applicable_reason"] = parent_manifest["support_not_applicable_reason"]
    elif extra_metadata and extra_metadata.get("support_not_applicable_reason"):
        manifest_data["support_not_applicable_reason"] = extra_metadata["support_not_applicable_reason"]
    else:
        raise PublicationValidationError(
            f"{item_id}: derived manifest requires explicit support or support_not_applicable_reason"
        )

    if resolved_denom is not None:
        manifest_data["denominator_n"] = int(resolved_denom)

    # Scientific identity inheritance
    parent_sci_id = dict(parent_manifest.get("scientific_identity", {}))
    for field in ("dataset_sha256", "population_id", "split_id", "seed", "taxonomy_version", "class_order_sha256"):
        if field not in parent_sci_id and field in parent_manifest:
            parent_sci_id[field] = parent_manifest[field]
    if "checkpoints" not in parent_sci_id and "checkpoints" in parent_manifest:
        parent_sci_id["checkpoints"] = parent_manifest["checkpoints"]

    manifest_data["scientific_identity"] = parent_sci_id
    manifest_data["seed"] = parent_manifest.get("seed", parent_sci_id.get("seed", 42))
    manifest_data["split_id"] = parent_manifest.get("split_id", parent_sci_id.get("split_id", "test"))

    # Figures selection inheritance / requirements
    if item_id.startswith("F"):
        selection = dict(parent_manifest.get("selection", {}))
        if extra_metadata and "selection" in extra_metadata:
            selection.update(extra_metadata["selection"])
        if "seed" not in selection:
            selection["seed"] = manifest_data["seed"]
        if "ids" not in selection:
            selection["ids"] = parent_manifest.get("selection_ids", ["all"])
        if "levels" not in selection:
            selection["levels"] = ["species"]
        if "sampling" not in selection:
            selection["sampling"] = "stratified"
        if "filters" not in selection:
            selection["filters"] = ["none"]
        manifest_data["selection"] = selection

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
    df = df.copy()
    if "nominal_noise" in df.columns:
        df = df[df["nominal_noise"] == 0.0].copy()
    if "model_name" in df.columns and "model" not in df.columns:
        df["model"] = df["model_name"]
    if "rank" in df.columns and "taxonomic_level" not in df.columns:
        df["taxonomic_level"] = df["rank"]
    if "top1_accuracy" in df.columns and "accuracy" not in df.columns:
        df["accuracy"] = df["top1_accuracy"]
    if "top5_accuracy" in df.columns and "top_5" not in df.columns:
        df["top_5"] = df["top5_accuracy"]

    required = {"model", "taxonomic_level", "accuracy", "top_5", "macro_f1"}
    missing = required - set(df.columns)
    if missing:
        raise PublicationValidationError(f"T1 input missing required columns: {missing}")

    cols = ["model", "taxonomic_level", "backbone_m", "fc_params_m", "parameters_m", "accuracy", "top_5", "macro_f1", "latency_ms"]
    missing_measurements = set(cols) - set(df.columns)
    if missing_measurements:
        raise PublicationValidationError(
            f"T1 input missing required measurement columns: {missing_measurements}. Zero-filling is forbidden."
        )

    # Rejection of null values
    if df[cols].isna().any().any():
        null_cols = [c for c in cols if df[c].isna().any()]
        raise PublicationValidationError(
            f"T1 input contains null measurements in columns {null_cols}. Null values are forbidden."
        )

    # Validation of metric bounds: must be in [0.0, 1.0] or [0.0, 100.0]
    for metric_col in ("accuracy", "top_5", "macro_f1"):
        vals = pd.to_numeric(df[metric_col], errors="coerce")
        if vals.isna().any():
            raise PublicationValidationError(f"T1 metric {metric_col} contains non-numeric values.")
        is_unit_interval = (vals >= 0.0).all() and (vals <= 1.0).all()
        is_percentage = (vals >= 0.0).all() and (vals <= 100.0).all()
        if not (is_unit_interval or is_percentage):
            raise PublicationValidationError(
                f"T1 metric {metric_col} values outside valid bounds [0.0, 1.0] or [0.0, 100.0]: "
                f"min={vals.min()}, max={vals.max()}"
            )
    return df[cols].copy()


@register_transformation("T2")
def transform_table_2_profiling(df: pd.DataFrame, meta: dict[str, Any]) -> pd.DataFrame:
    """Table 2: inference profiling (VRAM, step time, throughput)."""
    df = df.copy()
    if "model_name" in df.columns and "model" not in df.columns:
        df["model"] = df["model_name"]
    if "backbone_params" in df.columns and "backbone_m" not in df.columns:
        df["backbone_m"] = (df["backbone_params"] / 1e6).round(2)
    if "head_params" in df.columns and "fc_params_m" not in df.columns:
        df["fc_params_m"] = (df["head_params"] / 1e6).round(2)
    if "total_params" in df.columns and "parameters_m" not in df.columns:
        df["parameters_m"] = (df["total_params"] / 1e6).round(2)
    if "throughput_seq_sec" in df.columns and "throughput_seq_s" not in df.columns:
        df["throughput_seq_s"] = df["throughput_seq_sec"]
    if "avg_latency_ms" in df.columns and "step_time_ms" not in df.columns:
        df["step_time_ms"] = df["avg_latency_ms"]
    if "size_mb" in df.columns and "peak_vram_mb" not in df.columns:
        df["peak_vram_mb"] = df["size_mb"]

    cols = ["model", "backbone_m", "fc_params_m", "parameters_m", "peak_vram_mb", "step_time_ms", "throughput_seq_s"]
    if "gpu_hours" in df.columns:
        cols.append("gpu_hours")

    missing = set(cols) - set(df.columns)
    if missing:
        raise PublicationValidationError(
            f"T2 input missing required measurement columns: {missing}. Zero-filling is forbidden."
        )

    # Rejection of null values
    if df[cols].isna().any().any():
        null_cols = [c for c in cols if df[c].isna().any()]
        raise PublicationValidationError(
            f"T2 input contains null measurements in columns {null_cols}. Null values are forbidden."
        )
    return df[cols].copy()


@register_transformation("T3")
def transform_table_3_noise_degradation(df: pd.DataFrame, meta: dict[str, Any]) -> pd.DataFrame:
    """Table 3: degradation across noise levels."""
    df = df.copy()
    if "noise_rate" in df.columns and "epsilon" not in df.columns:
        df["epsilon"] = df["noise_rate"]
    if "epsilon" not in df.columns:
        raise PublicationValidationError("T3 input missing required columns: {'epsilon'}")
    return df


@register_transformation("T4")
def transform_table_4_species_f1_and_pairwise(df: pd.DataFrame, meta: dict[str, Any]) -> pd.DataFrame:
    """Table 4: species F1 distribution and pairwise comparisons."""
    required = {"f1_bin"}
    if not required.issubset(df.columns):
        if "f1_band" in df.columns:
            df = df.rename(columns={"f1_band": "f1_bin"})
        else:
            raise PublicationValidationError(f"T4 input missing required columns: {required - set(df.columns)}")
    return df.copy()


@register_transformation("T5")
def transform_table_5_latent_metrics(df: pd.DataFrame, meta: dict[str, Any]) -> pd.DataFrame:
    """Table 5: high-dimensional latent space metrics (KNN, Silhouette, Linear Probe)."""
    df = df.copy()
    if "Model" in df.columns and "model" not in df.columns:
        df["model"] = df["Model"]
    if "KNN_Accuracy" in df.columns and "knn_accuracy" not in df.columns:
        df["knn_accuracy"] = df["KNN_Accuracy"]
    required = {"model", "knn_accuracy"}
    if not required.issubset(df.columns):
        raise PublicationValidationError(f"T5 input missing required columns: {required - set(df.columns)}")
    return df


@register_transformation("T6")
def transform_table_6_hierarchy_correlations(df: pd.DataFrame, meta: dict[str, Any]) -> pd.DataFrame:
    """Table 6: taxonomic hierarchy correlations (Species vs Genus vs Order)."""
    df = df.copy()
    if "Transition" in df.columns and "level_pair" not in df.columns:
        df["level_pair"] = df["Transition"]
    if "Model" in df.columns and "model" not in df.columns:
        df["model"] = df["Model"]
    if "Pearson_r" in df.columns and "pearson_r" not in df.columns:
        df["pearson_r"] = df["Pearson_r"]
    if "Spearman_rho" in df.columns and "spearman_rho" not in df.columns:
        df["spearman_rho"] = df["Spearman_rho"]
    if "N_taxa" in df.columns and "n_taxa" not in df.columns:
        df["n_taxa"] = df["N_taxa"]
    if "level_pair" not in df.columns:
        raise PublicationValidationError("T6 input missing required columns: {'level_pair'}")
    return df


@register_transformation("T7")
def transform_table_7_hierarchical_entropy(df: pd.DataFrame, meta: dict[str, Any]) -> pd.DataFrame:
    """Table 7: hierarchical activation entropy (Species, Genus, Order)."""
    df = df.copy()
    if "model_name" in df.columns and "model" not in df.columns:
        df["model"] = df["model_name"]
    required = {"model"}
    if not required.issubset(df.columns):
        raise PublicationValidationError(f"T7 input missing required columns: {required - set(df.columns)}")
    return df


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
    df = df.copy()
    if "codon" in df.columns and "triplet" not in df.columns:
        df["triplet"] = df["codon"]
    if "triplet" not in df.columns:
        raise PublicationValidationError("T9 input missing required columns: {'triplet'}")
    return df


@register_transformation("T10")
def transform_table_10_triplet_composition(df: pd.DataFrame, meta: dict[str, Any]) -> pd.DataFrame:
    """Table 10: triplet composition summary and expected base frequencies."""
    df = df.copy()
    if "codon" in df.columns and "triplet" not in df.columns:
        df["triplet"] = df["codon"]
    if "triplet" not in df.columns and "signal" not in df.columns:
        raise PublicationValidationError("T10 input missing required columns: {'triplet'}")
    return df


@register_transformation("T11")
def transform_table_11_architecture_correlations(df: pd.DataFrame, meta: dict[str, Any]) -> pd.DataFrame:
    """Table 11: motif overlap and permutation null analysis between architectures / co-occurrence synergy.

    Rejects cooccurrence data claiming architecture correlation without permutation null test!
    """
    df = df.copy()

    # Check that it's not a degenerated identical matrix claiming r=1 without variance
    if "correlation_r" in df.columns and "p_value" in df.columns:
        if (df["correlation_r"] == 1.0).all() and (df["p_value"] == 0.0).all():
            if "has_variance" in df.columns and not df["has_variance"].all():
                raise PublicationValidationError(
                    "T11 scientific contract violation: spurious r=1 correlation from identical or zero-variance matrices."
                )

    if "is_permutation_tested" in df.columns:
        if not df["is_permutation_tested"].all():
            raise PublicationValidationError(
                "T11 scientific contract violation: co-occurrence matrices without permutation null "
                "testing cannot be published as architecture correlation."
            )
    elif "p_permutation_null" in df.columns:
        # Verified permutation null test from short/08
        pass
    else:
        raise PublicationValidationError(
            "T11 scientific contract violation: permutation null testing evidence missing."
        )

    return df


@register_transformation("T12")
def transform_table_12_long_sequence_consensus(df: pd.DataFrame, meta: dict[str, Any]) -> pd.DataFrame:
    """Table 12: long sequence voter comparison (mean_logits vs majority_vote)."""
    df = df.copy()
    # Support pivoting from build_c02_table12 (variant x aggregation rows)
    if "aggregation" in df.columns and ("variant" in df.columns or "model" in df.columns):
        var_col = "variant" if "variant" in df.columns else "model"
        acc_col = "accuracy" if "accuracy" in df.columns else "acc"
        if acc_col not in df.columns:
            raise PublicationValidationError(f"T12 unpivoted input missing accuracy column: {df.columns.tolist()}")

        # Check for duplicate aggregations
        dups = df.duplicated(subset=[var_col, "aggregation"])
        if dups.any():
            dup_pairs = df[dups][[var_col, "aggregation"]].to_dict(orient="records")
            raise PublicationValidationError(
                f"T12 input contains duplicated variant-aggregation pairs: {dup_pairs}"
            )

        # Check that both majority_vote and mean_logits are present for each variant
        grouped_aggs = df.groupby(var_col)["aggregation"].apply(set)
        required_aggs = {"majority_vote", "mean_logits"}
        for variant, aggs in grouped_aggs.items():
            missing_aggs = required_aggs - aggs
            if missing_aggs:
                raise PublicationValidationError(
                    f"T12 variant '{variant}' is missing required aggregation mode(s): {missing_aggs}"
                )

        pivoted = df.pivot(index=var_col, columns="aggregation", values=acc_col).reset_index()
        pivoted = pivoted.rename(columns={
            var_col: "model",
            "majority_vote": "majority_vote_acc",
            "mean_logits": "mean_logits_acc",
        })
        if "coverage" in df.columns:
            cov = df.groupby(var_col)["coverage"].first().reset_index().rename(columns={var_col: "model"})
            pivoted = pivoted.merge(cov, on="model", how="left")
        if "N" in df.columns:
            n_vals = df.groupby(var_col)["N"].first().reset_index().rename(columns={var_col: "model"})
            pivoted = pivoted.merge(n_vals, on="model", how="left")
        df = pivoted

    required = {"model", "majority_vote_acc", "mean_logits_acc"}
    if not required.issubset(df.columns):
        raise PublicationValidationError(f"T12 input missing required columns: {required - set(df.columns)}")
    if df[["majority_vote_acc", "mean_logits_acc"]].isna().any().any():
        raise PublicationValidationError("T12 input contains null accuracy values for required aggregations")
    return df


@register_transformation("F1")
def transform_figure_1_noise_robustness(df: pd.DataFrame, meta: dict[str, Any]) -> pd.DataFrame:
    """Figure 1: noise robustness curves."""
    df = df.copy()
    if "noise_rate" in df.columns and "epsilon" not in df.columns:
        df["epsilon"] = df["noise_rate"]
    required = {"epsilon", "accuracy"}
    if not required.issubset(df.columns):
        raise PublicationValidationError(f"F1 input missing required columns: {required - set(df.columns)}")
    return df


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
    df = df.copy()
    if "codon" in df.columns and "triplet" not in df.columns:
        df["triplet"] = df["codon"]
    if "rscu" not in df.columns and "rscu_frame0" in df.columns:
        df["rscu"] = df["rscu_frame0"]
    required = {"triplet", "rscu"}
    if not required.issubset(df.columns):
        raise PublicationValidationError(f"F4 input missing required columns: {required - set(df.columns)}")
    return df


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
        df = pd.read_csv(source_artifact_path, comment="#")
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
        source_artifact_path=source_artifact_path,
        transformation_name=transform_func.__name__,
        support_n=len(transformed_df),
        output_rows=len(transformed_df),
        extra_metadata=extra_metadata,
    )

    return output_path, manifest_path
