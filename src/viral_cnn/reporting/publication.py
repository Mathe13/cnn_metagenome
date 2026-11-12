"""Strict, provenance-first assembly of publication artifacts.

This module treats a publication config as an explicit selection. It never searches
``results_reproduction`` for substitutes and it never promotes legacy table/figure
constants to empirical evidence.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import re
import shutil
import subprocess
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from viral_cnn.utils.logging_utils import log_artifact_path

PUBLICATION_PROTOCOL = "c06_strict_v1"
ITEM_IDS = [*(f"T{i}" for i in range(1, 13)), *(f"F{i}" for i in range(1, 6))]
EVIDENCE_STATUSES = {
    "EMPIRICAL_VERIFIED", "DERIVED_VERIFIED", "HISTORICAL_UNREPRODUCED",
    "SYNTHETIC", "MISSING", "INVALID",
}
SCIENTIFIC_IDENTITY_FIELDS = (
    "dataset_sha256", "population_id", "split_id", "seed", "taxonomy_version", "class_order_sha256",
    "checkpoint_sha256s", "model_variants", "noise_conditions", "metric_definition",
)


class PublicationValidationError(RuntimeError):
    """Raised when a selected publication input cannot satisfy its contract."""


def sha256_file(path: Path) -> str:
    """Return a file's SHA-256 digest."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_path(path: Path) -> str:
    """Hash a file or deterministic directory inventory including relative names."""
    path = Path(path)
    if path.is_file():
        return sha256_file(path)
    if not path.is_dir():
        raise FileNotFoundError(path)
    digest = hashlib.sha256()
    files = sorted(item for item in path.rglob("*") if item.is_file())
    if not files:
        raise PublicationValidationError(f"cannot hash empty artifact directory: {path}")
    for item in files:
        relative = item.relative_to(path).as_posix().encode("utf-8")
        digest.update(relative)
        digest.update(b"\0")
        digest.update(bytes.fromhex(sha256_file(item)))
    return digest.hexdigest()


def load_publication_config(path: Path) -> dict[str, Any]:
    """Read and structurally validate one explicit C06 YAML selection."""
    path = Path(path).resolve()
    if not path.is_file():
        raise PublicationValidationError(f"configuration missing: {path}")
    log_artifact_path("C06 publication configuration", path)
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("publication_protocol") != PUBLICATION_PROTOCOL:
        raise PublicationValidationError(f"expected publication_protocol={PUBLICATION_PROTOCOL}: {path}")
    if data.get("schema_version") != 1:
        raise PublicationValidationError(f"unsupported schema_version in {path}")
    items = data.get("items")
    if not isinstance(items, list):
        raise PublicationValidationError("items must be an explicit list")
    ids = [item.get("item_id") for item in items if isinstance(item, dict)]
    if len(ids) != 17 or set(ids) != set(ITEM_IDS) or len(set(ids)) != 17:
        raise PublicationValidationError("items must enumerate T1-T12 and F1-F5 exactly once")
    for item in items:
        for key in ("artifact", "manifest", "sha256", "protocol", "transformation", "expected"):
            if key not in item:
                raise PublicationValidationError(f"{item['item_id']}: missing config field {key}")
    data["_config_path"] = str(path)
    project_root = next(
        (parent for parent in path.parents if (parent / ".git").exists() or (parent / "src" / "viral_cnn").is_dir()),
        path.parent.parent,
    )
    data["_project_root"] = str(project_root)
    return data


def _read_manifest(path: Path) -> dict[str, Any]:
    """Load JSON or YAML evidence manifest with a path-aware error."""
    log_artifact_path("C06 evidence manifest", path)
    try:
        if path.suffix.lower() == ".json":
            value = json.loads(path.read_text(encoding="utf-8"))
        else:
            value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, yaml.YAMLError) as exc:
        raise PublicationValidationError(f"manifest unreadable at {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise PublicationValidationError(f"manifest must contain an object: {path}")
    return value


def _resolve(project_root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project_root / path


def _inspect_transitive_manifest(path: Path, project_root: Path, visited: set[Path] | None = None) -> bool:
    """Validate manifest mode and hashes recursively; return whether any input is synthetic."""
    visited = visited or set()
    path = path.resolve()
    if path in visited:
        return False
    visited.add(path)
    value = _read_manifest(path)
    if "is_smoke_test" not in value:
        raise PublicationValidationError(f"transitive manifest lacks explicit is_smoke_test mode: {path}")
    synthetic = value["is_smoke_test"] is True or value.get("synthetic") is True
    records = value.get("inputs", [])
    dependencies = value.get("dependencies", [])
    if not isinstance(records, list) or not isinstance(dependencies, list):
        raise PublicationValidationError(f"transitive manifest input/dependency lists are invalid: {path}")
    for record in [*records, *dependencies]:
        if not isinstance(record, dict) or not record.get("path") or not record.get("sha256"):
            raise PublicationValidationError(f"transitive input must declare path and sha256: {path}")
        child = _resolve(project_root, str(record["path"]))
        if not child.exists():
            raise PublicationValidationError(f"transitive input missing: {child}")
        if sha256_path(child) != record["sha256"]:
            raise PublicationValidationError(f"transitive input hash mismatch: {child}")
        if child.is_file() and _is_manifest_input(child):
            synthetic = _inspect_transitive_manifest(child, project_root, visited) or synthetic
    return synthetic


def _is_manifest_input(path: Path) -> bool:
    """Distinguish JSON/YAML manifests from arbitrary structured data inputs."""
    if "manifest" in path.name.lower():
        return True
    if path.suffix.lower() not in {".json", ".yaml", ".yml"}:
        return False
    try:
        if path.suffix.lower() == ".json":
            value = json.loads(path.read_text(encoding="utf-8"))
        else:
            value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, yaml.YAMLError):
        return False
    return isinstance(value, dict) and bool({"is_smoke_test", "protocol", "inputs", "dependencies"} & set(value))


def validate_selection(config: dict[str, Any], *, strict_empirical: bool = True) -> list[dict[str, Any]]:
    """Validate only selected evidence and its declared transitive dependencies.

    Invalid entries become explicit diagnostic rows. Strict mode raises after all
    selected entries have been inspected, preserving the complete gap report.
    """
    root = Path(config["_project_root"])
    rows: list[dict[str, Any]] = []
    for item in config["items"]:
        iid = item["item_id"]
        artifact = _resolve(root, str(item.get("artifact", ""))) if item.get("artifact") else None
        manifest = _resolve(root, str(item.get("manifest", ""))) if item.get("manifest") else None
        row: dict[str, Any] = {
            "item_id": iid, "artifact": str(artifact) if artifact else "",
            "manifest": str(manifest) if manifest else "", "protocol": item.get("protocol"),
            "transformation": item.get("transformation"), "n": None,
            "support": item.get("expected", {}).get("support"), "source_inputs": [],
            "checkpoints": [], "status": "MISSING",
            "diagnostic": "", "sha256": "", "n_not_applicable_reason": "",
        }
        try:
            if not artifact or not artifact.exists():
                raise PublicationValidationError(f"{iid}: artifact missing: {artifact or '<unset>'}")
            if not manifest or not manifest.is_file():
                raise PublicationValidationError(f"{iid}: manifest missing: {manifest or '<unset>'}")
            if not item.get("protocol") or item.get("protocol") == "TO_BE_SELECTED":
                raise PublicationValidationError(f"{iid}: source protocol not identified in the selected artifact")
            log_artifact_path(f"C06 selected artifact {iid}", artifact)
            actual_hash = sha256_path(artifact)
            row["sha256"] = actual_hash
            if item.get("sha256") != actual_hash:
                raise PublicationValidationError(f"{iid}: sha256 mismatch for {artifact}")
            declared = _read_manifest(manifest)
            row["manifest_sha256"] = sha256_file(manifest)
            if declared.get("is_smoke_test") is True or declared.get("synthetic") is True:
                row["status"] = "SYNTHETIC"
                raise PublicationValidationError(f"{iid}: selected manifest declares synthetic evidence")
            if "is_smoke_test" not in declared or declared["is_smoke_test"] is not False:
                raise PublicationValidationError(
                    f"{iid}: manifest must explicitly declare is_smoke_test=false at {manifest}"
                )
            if declared.get("artifact_sha256") != actual_hash:
                raise PublicationValidationError(f"{iid}: artifact hash in manifest does not match {artifact}")
            declared_inputs = declared.get("inputs")
            if not isinstance(declared_inputs, list) or not declared_inputs:
                raise PublicationValidationError(f"{iid}: manifest must identify source inputs")
            dependency_paths = declared.get("dependencies", [])
            if not isinstance(dependency_paths, list):
                raise PublicationValidationError(f"{iid}: manifest dependencies must be a list")
            synthetic = bool(declared.get("is_smoke_test", declared.get("synthetic", False)))
            verified_input_hashes: set[str] = set()
            all_inputs = [*declared_inputs, *({"path": p} for p in dependency_paths)]
            row["source_inputs"] = all_inputs
            row["checkpoints"] = declared.get("checkpoints", [])
            for input_record in all_inputs:
                if not isinstance(input_record, dict) or not input_record.get("path"):
                    raise PublicationValidationError(f"{iid}: each source input must declare path and sha256")
                dep_path = _resolve(root, str(input_record["path"]))
                if not dep_path.exists():
                    raise PublicationValidationError(f"{iid}: dependency missing: {dep_path}")
                log_artifact_path(f"C06 source input {iid}", dep_path)
                input_sha = sha256_path(dep_path)
                if input_record.get("sha256") != input_sha:
                    raise PublicationValidationError(f"{iid}: source input hash mismatch: {dep_path}")
                verified_input_hashes.add(input_sha)
                if dep_path.is_file() and _is_manifest_input(dep_path):
                    synthetic = _inspect_transitive_manifest(dep_path, root) or synthetic
            if declared.get("protocol") != item.get("protocol"):
                raise PublicationValidationError(f"{iid}: protocol mismatch in {manifest}")
            if declared.get("status") != "COMPLETE":
                raise PublicationValidationError(f"{iid}: manifest status must be COMPLETE")
            if declared.get("evidence_kind") not in {"empirical", "structural"}:
                raise PublicationValidationError(f"{iid}: manifest {manifest}: evidence_kind must be empirical or structural")
            if declared.get("evidence_kind") == "structural":
                derivation = declared.get("derivation")
                if not isinstance(derivation, dict) or not {"source", "config"}.issubset(derivation):
                    raise PublicationValidationError(f"{iid}: structural evidence must identify derivation source and config")
                for kind in ("source", "config"):
                    record = derivation[kind]
                    if not isinstance(record, dict) or not record.get("path") or not record.get("sha256"):
                        raise PublicationValidationError(f"{iid}: structural derivation {kind} must record path and sha256")
                    derivation_path = _resolve(root, str(record["path"]))
                    if not derivation_path.is_file() or sha256_file(derivation_path) != record["sha256"]:
                        raise PublicationValidationError(f"{iid}: structural derivation {kind} missing or hash mismatch: {derivation_path}")
                    if record["sha256"] not in verified_input_hashes:
                        raise PublicationValidationError(f"{iid}: structural derivation {kind} is not a hash-verified input")
            if declared.get("transformation") != item.get("transformation"):
                raise PublicationValidationError(f"{iid}: transformation mismatch in {manifest}")
            if "seed" not in declared or "split_id" not in declared:
                raise PublicationValidationError(f"{iid}: manifest must identify seed and split_id")
            if "n" not in declared and not declared.get("n_not_applicable_reason"):
                raise PublicationValidationError(f"{iid}: manifest must record N or why N is not applicable")
            if "support" not in declared and not declared.get("support_not_applicable_reason"):
                raise PublicationValidationError(f"{iid}: manifest must record support or why it is not applicable")
            if item["item_id"].startswith("F"):
                selection = declared.get("selection")
                if not isinstance(selection, dict) or not all(selection.get(k) is not None for k in ("ids", "levels", "sampling", "seed", "filters")):
                    raise PublicationValidationError(f"{iid}: figure manifest must record IDs, levels, sampling, seed and filters")
                if selection["seed"] != declared["seed"]:
                    raise PublicationValidationError(f"{iid}: figure selection seed differs from run seed")
            for key, expected in item.get("expected", {}).items():
                if key == "columns":
                    continue
                if key == "scientific_identity":
                    if not isinstance(expected, dict):
                        raise PublicationValidationError(f"{iid}: expected scientific_identity must be an object")
                    declared_identity = declared.get("scientific_identity")
                    if not isinstance(declared_identity, dict):
                        raise PublicationValidationError(f"{iid}: manifest scientific_identity must be an object at {manifest}")
                    for identity_field in SCIENTIFIC_IDENTITY_FIELDS:
                        value = expected.get(identity_field)
                        if value in (None, "", "TO_BE_SELECTED"):
                            raise PublicationValidationError(f"{iid}: scientific identity {identity_field} not selected")
                        if declared_identity.get(identity_field) != value:
                            raise PublicationValidationError(
                                f"{iid}: scientific identity {identity_field} mismatch in {manifest}"
                            )
                    if expected.get("seed") != declared.get("seed"):
                        raise PublicationValidationError(f"{iid}: selected scientific identity seed differs from manifest")
                    if expected.get("split_id") != declared.get("split_id"):
                        raise PublicationValidationError(f"{iid}: selected scientific identity split_id differs from manifest")
                    for source_field in ("dataset_sha256", "class_order_sha256"):
                        source_hash = expected.get(source_field)
                        if isinstance(source_hash, str) and source_hash.lower().startswith("n/a -"):
                            continue
                        if source_hash not in verified_input_hashes:
                            raise PublicationValidationError(f"{iid}: {source_field} is not backed by a hash-verified input")
                    checkpoints = declared.get("checkpoints", [])
                    expected_checkpoints = expected.get("checkpoint_sha256s")
                    if isinstance(expected_checkpoints, dict):
                        actual_checkpoints = {}
                        for checkpoint in checkpoints:
                            if not isinstance(checkpoint, dict) or not checkpoint.get("path") or not checkpoint.get("variant"):
                                raise PublicationValidationError(f"{iid}: checkpoint identity requires variant, path and sha256")
                            checkpoint_path = _resolve(root, str(checkpoint["path"]))
                            if not checkpoint_path.is_file():
                                raise PublicationValidationError(f"{iid}: checkpoint missing: {checkpoint_path}")
                            log_artifact_path(f"C06 checkpoint {iid}", checkpoint_path)
                            actual_sha = sha256_file(checkpoint_path)
                            if actual_sha != checkpoint.get("sha256"):
                                raise PublicationValidationError(f"{iid}: checkpoint hash mismatch: {checkpoint_path}")
                            actual_checkpoints[checkpoint["variant"]] = actual_sha
                        if actual_checkpoints != expected_checkpoints:
                            raise PublicationValidationError(f"{iid}: selected checkpoint variants/hashes differ from manifest")
                    elif not (isinstance(expected_checkpoints, str) and expected_checkpoints.lower().startswith("n/a -")):
                        raise PublicationValidationError(f"{iid}: checkpoint hashes must be a map or explicit n/a reason")
                    elif checkpoints:
                        raise PublicationValidationError(f"{iid}: manifest has checkpoints but selection declares none")
                    panel_variants = item.get("expected", {}).get("panel_variants", {})
                    if panel_variants:
                        required_variants = set(panel_variants.values())
                        selected_variants = set(expected.get("model_variants", []))
                        if not required_variants.issubset(selected_variants) or not isinstance(expected_checkpoints, dict) or not required_variants.issubset(expected_checkpoints):
                            raise PublicationValidationError(f"{iid}: panel variants lack matching selected checkpoints")
                if key not in declared:
                    raise PublicationValidationError(f"{iid}: manifest lacks expected field {key}")
                if declared[key] != expected:
                    raise PublicationValidationError(f"{iid}: expected {key}={expected!r}, got {declared[key]!r}")
            if iid.startswith("T"):
                expected_columns = item.get("expected", {}).get("columns", [])
                if not expected_columns:
                    raise PublicationValidationError(f"{iid}: expected table columns are not selected")
                try:
                    table_frame = _load_table_artifact(artifact)
                except Exception as exc:
                    raise PublicationValidationError(f"{iid}: artifact unreadable at {artifact}: {exc}") from exc
                if not set(expected_columns).issubset(table_frame.columns):
                    raise PublicationValidationError(f"{iid}: artifact columns do not satisfy {expected_columns}")
            if synthetic:
                row["status"] = "SYNTHETIC"
                raise PublicationValidationError(f"{iid}: selected evidence has synthetic dependency")
            row["n"] = declared.get("n")
            row["n_not_applicable_reason"] = declared.get("n_not_applicable_reason", "")
            row["support"] = declared.get("support", item.get("expected", {}).get("support"))
            row["support_not_applicable_reason"] = declared.get("support_not_applicable_reason", "")
            row["denominator"] = declared.get("denominator_n", declared.get("n"))
            row["class_order_sha256"] = declared.get("scientific_identity", {}).get("class_order_sha256")
            row["checkpoint_sha256s"] = declared.get("scientific_identity", {}).get("checkpoint_sha256s")
            row["selection"] = declared.get("selection", {})
            row["panel_variants"] = declared.get("panel_variants", {})
            row["status"] = "DERIVED_VERIFIED" if declared.get("evidence_kind") == "structural" else "EMPIRICAL_VERIFIED"
        except PublicationValidationError as exc:
            row["diagnostic"] = str(exc)
            if row["status"] != "SYNTHETIC":
                row["status"] = "MISSING" if "missing" in str(exc).lower() or "unset" in str(exc).lower() else "INVALID"
        rows.append(row)
    failures = [r for r in rows if r["status"] not in {"EMPIRICAL_VERIFIED", "DERIVED_VERIFIED"}]
    if strict_empirical and failures:
        detail = "; ".join(r["diagnostic"] or f"{r['item_id']}: inadmissible status {r['status']}" for r in failures)
        raise PublicationValidationError(f"strict empirical gate blocked ({len(failures)} items): {detail}")
    return rows


def evidence_matrix(config: dict[str, Any], rows: list[dict[str, Any]]) -> pd.DataFrame:
    """Build one traceability row per publication table/figure."""
    by_id = {row["item_id"]: row for row in rows}
    output = []
    for item in config["items"]:
        row = by_id[item["item_id"]]
        output.append({
            **row, "claim_role": "computed", "historical_status": "HISTORICAL_UNREPRODUCED",
            "historical_source": config.get("historical_source", {}).get("manuscript_submission", ""),
            "expected_output": item.get("output", ""), "n": row.get("n"), "support": row.get("support"),
        })
    return pd.DataFrame(output)


def reconcile_value(
    observed: Any, historical: Any, *, observed_unit: str, historical_unit: str,
    observed_protocol: str, historical_protocol: str,
) -> dict[str, Any]:
    """Compare values without intermediate rounding when estimands are compatible."""
    if observed_unit != historical_unit or observed_protocol != historical_protocol:
        return {"observed": observed, "historical": historical, "difference": None,
                "unit": None, "status": "INCOMPATIBLE", "reason": "unit or protocol/population differs"}
    try:
        difference = Decimal(str(observed)) - Decimal(str(historical))
    except (InvalidOperation, ValueError):
        return {"observed": observed, "historical": historical, "difference": None,
                "unit": None, "status": "INVALID", "reason": "value is not numeric"}
    unit = "pp" if observed_unit.lower() in {"%", "percent", "percentage"} else observed_unit
    return {"observed": observed, "historical": historical, "difference": str(difference),
            "unit": unit, "status": "COMPATIBLE", "reason": ""}


def _load_table_artifact(path: Path) -> pd.DataFrame:
    """Load one explicitly selected CSV or Parquet table artifact."""
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if path.suffix.lower() in {".parquet", ".pq"} or path.is_dir():
        return pd.read_parquet(path)
    raise PublicationValidationError(f"unsupported table artifact format: {path}")


def render_tables(config: dict[str, Any], output_dir: Path, rows: list[dict[str, Any]]) -> dict[str, Path]:
    """Serialize validated table inputs and a separate historical comparison matrix."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    by_id = {row["item_id"]: row for row in rows}
    outputs: dict[str, Path] = {}
    reconciliation: list[dict[str, Any]] = []
    for item in config["items"]:
        if not item["item_id"].startswith("T"):
            continue
        row = by_id[item["item_id"]]
        if row["status"] not in {"EMPIRICAL_VERIFIED", "DERIVED_VERIFIED"}:
            continue
        frame = _load_table_artifact(Path(row["artifact"]))
        expected_columns = item.get("expected", {}).get("columns", [])
        if not set(expected_columns).issubset(frame.columns):
            raise PublicationValidationError(f"{item['item_id']}: artifact columns do not satisfy {expected_columns}")
        base = output_dir / f"table_{item['item_id'][1:]}"
        frame.to_csv(base.with_suffix(".csv"), index=False)
        from viral_cnn.reporting.tables import _df_to_markdown
        base.with_suffix(".md").write_text(_df_to_markdown(frame), encoding="utf-8")
        base.with_suffix(".tex").write_text(frame.to_latex(index=False), encoding="utf-8")
        outputs[f"{item['item_id']}:csv"] = base.with_suffix(".csv")
        outputs[f"{item['item_id']}:markdown"] = base.with_suffix(".md")
        outputs[f"{item['item_id']}:latex"] = base.with_suffix(".tex")
        historical = item.get("historical_values", {})
        for metric, old in historical.items():
            if metric not in frame.columns:
                reconciliation.append({"item_id": item["item_id"], "metric": metric,
                                       "status": "INVALID", "reason": "observed metric absent"})
                continue
            for row_index, observed in enumerate(frame[metric].tolist()):
                comparison = reconcile_value(
                    observed, old.get("value"), observed_unit=item.get("metric_units", {}).get(metric, old.get("unit", "")),
                    historical_unit=old.get("unit", ""), observed_protocol=item.get("protocol", ""),
                    historical_protocol=old.get("protocol", ""),
                )
                reconciliation.append({"item_id": item["item_id"], "metric": metric,
                                       "row_index": row_index, **comparison})
        if not historical:
            reconciliation.append({"item_id": item["item_id"], "metric": "*", "status": "HISTORICAL_UNREPRODUCED",
                                   "reason": "no versioned historical numeric value selected"})
    report = pd.DataFrame(reconciliation)
    report_csv = output_dir / "reconciliation_report.csv"
    report_md = output_dir / "reconciliation_report.md"
    report.to_csv(report_csv, index=False)
    from viral_cnn.reporting.tables import _df_to_markdown
    report_md.write_text(_df_to_markdown(report), encoding="utf-8")
    outputs["reconciliation:csv"] = report_csv
    outputs["reconciliation:markdown"] = report_md
    return outputs


def render_figures(config: dict[str, Any], output_dir: Path, rows: list[dict[str, Any]]) -> dict[str, Path]:
    """Render figures from explicitly selected CSV data; no extracted/procedural fallback."""
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    by_id = {row["item_id"]: row for row in rows}
    outputs: dict[str, Path] = {}
    rendered_sources: dict[str, Any] = {}
    for item in config["items"]:
        if not item["item_id"].startswith("F"):
            continue
        row = by_id[item["item_id"]]
        if row["status"] not in {"EMPIRICAL_VERIFIED", "DERIVED_VERIFIED"}:
            continue
        source = Path(row["artifact"])
        frame = _load_table_artifact(source)
        x_col, y_col = item.get("plot", {}).get("x"), item.get("plot", {}).get("y")
        kind = item.get("plot", {}).get("kind", "line")
        if kind != "network_edges" and (x_col not in frame.columns or y_col not in frame.columns):
            raise PublicationValidationError(f"{item['item_id']}: figure x/y columns absent: {x_col}, {y_col}")
        plot = item.get("plot", {})
        rendered_sources[item["item_id"]] = {
            "render_source": "selected_artifact", "source_path": str(source),
            "source_sha256": sha256_path(source), "plot_parameters": plot,
        }
        if kind == "panel_grid":
            if not {"panel", "variant", x_col, y_col}.issubset(frame.columns):
                raise PublicationValidationError(f"{item['item_id']}: panel grid requires panel, variant, x and y columns")
            panel_map = item.get("expected", {}).get("panel_variants", {})
            if set(frame["panel"].astype(str)) != set(item.get("expected", {}).get("panels", [])):
                raise PublicationValidationError(f"{item['item_id']}: artifact panel set differs from configured panel set")
            fig, axes = plt.subplots(1, len(panel_map), figsize=tuple(plot.get("figsize", [15, 4])), squeeze=False)
            panel_dir = output_dir / "fig_3_panels"
            panel_dir.mkdir(parents=True, exist_ok=True)
            for axis, panel in zip(axes[0], item["expected"]["panels"]):
                subset = frame[frame["panel"].astype(str) == str(panel)]
                variants = set(subset["variant"].astype(str))
                if variants != {panel_map[panel]}:
                    raise PublicationValidationError(f"{item['item_id']} panel {panel}: variant identity mismatch {variants}")
                axis.scatter(subset[x_col], subset[y_col], marker=plot.get("marker", "o"))
                axis.set_title(f"({panel}) {panel_map[panel]}")
                axis.set_xlabel(x_col)
                axis.set_ylabel(y_col)
                panel_path = panel_dir / f"fig_3_panel_{panel}_{panel_map[panel]}.png"
                fig_one, axis_one = plt.subplots(figsize=(5, 4))
                axis_one.scatter(subset[x_col], subset[y_col], marker=plot.get("marker", "o"))
                axis_one.set_title(f"({panel}) {panel_map[panel]}")
                axis_one.set_xlabel(x_col)
                axis_one.set_ylabel(y_col)
                fig_one.tight_layout()
                fig_one.savefig(panel_path, dpi=180)
                plt.close(fig_one)
                outputs[f"{item['item_id']}:panel_{panel}"] = panel_path
        else:
            fig, axis = plt.subplots(figsize=tuple(plot.get("figsize", [7, 4])))
            if kind == "network_edges":
                edge_cols = (plot.get("x1"), plot.get("y1"), plot.get("x2"), plot.get("y2"))
                if not all(col in frame.columns for col in edge_cols):
                    raise PublicationValidationError(f"{item['item_id']}: edge artifact requires x1/y1/x2/y2 coordinates")
                for _, edge in frame.iterrows():
                    axis.plot([edge[edge_cols[0]], edge[edge_cols[2]]], [edge[edge_cols[1]], edge[edge_cols[3]]],
                              linewidth=plot.get("linewidth", 0.5), alpha=plot.get("alpha", 0.35), color=plot.get("color", "#777777"))
                axis.scatter(frame[edge_cols[0]], frame[edge_cols[1]], s=plot.get("node_size", 8), color=plot.get("node_color", "#1f77b4"))
                axis.scatter(frame[edge_cols[2]], frame[edge_cols[3]], s=plot.get("node_size", 8), color=plot.get("node_color", "#1f77b4"))
            elif kind == "scatter":
                axis.scatter(frame[x_col], frame[y_col], marker=plot.get("marker", "o"),
                             c=frame[plot["color_by"]] if plot.get("color_by") in frame.columns else None,
                             s=plot.get("point_size", 12), alpha=plot.get("alpha", 0.8))
            else:
                axis.plot(frame[x_col], frame[y_col], marker=plot.get("marker", "o"))
            axis.set_xlabel(x_col)
            axis.set_ylabel(y_col)
            axis.set_title(plot.get("title", item["item_id"]))
        fig.tight_layout()
        base = output_dir / f"fig_{item['item_id'][1:]}"
        fig.savefig(base.with_suffix(".png"), dpi=180)
        fig.savefig(base.with_suffix(".pdf"))
        plt.close(fig)
        if not base.with_suffix(".png").read_bytes().startswith(b"\x89PNG\r\n\x1a\n"):
            raise PublicationValidationError(f"{item['item_id']}: rendered PNG is invalid")
        if not base.with_suffix(".pdf").read_bytes().startswith(b"%PDF-"):
            raise PublicationValidationError(f"{item['item_id']}: rendered PDF is invalid")
        outputs[item["item_id"]] = base.with_suffix(".png")
    manifest_path = output_dir / "figure_render_manifest.json"
    manifest_path.write_text(json.dumps({
        "publication_protocol": PUBLICATION_PROTOCOL,
        "rendered_figures": rendered_sources,
    }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    outputs["figure_render_manifest"] = manifest_path
    return outputs


def _extract_review_notes(content: str) -> tuple[str, list[str]]:
    """Remove and return balanced ``\\notavdois{...}`` calls without editing their text."""
    spans: list[tuple[int, int]] = []
    notes: list[str] = []
    cursor = 0
    marker = r"\notavdois{"
    while True:
        start = content.find(marker, cursor)
        if start < 0:
            break
        open_brace = start + len(marker) - 1
        depth = 0
        index = open_brace
        while index < len(content):
            char = content[index]
            escaped = index > 0 and content[index - 1] == "\\"
            if not escaped and char == "{":
                depth += 1
            elif not escaped and char == "}":
                depth -= 1
                if depth == 0:
                    end = index + 1
                    spans.append((start, end))
                    notes.append(content[start:end])
                    cursor = end
                    break
            index += 1
        else:
            raise PublicationValidationError("unbalanced editorial review note macro")
    for start, end in reversed(spans):
        content = content[:start] + content[end:]
    return content, notes


def render_manuscript_variants(config: dict[str, Any], output_dir: Path) -> dict[str, Path]:
    """Build review/submission sources from the selected manuscript skeleton.

    All numerical tables are replaced by inputs to the package-generated tables.
    Historical numerical prose in Results, Abstract Results, Discussion and
    Conclusions is replaced with evidence-status language before serialization.
    """
    root = Path(config["_project_root"])
    history = config.get("historical_source", {})
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    generated: dict[str, Path] = {}
    tabular = re.compile(r"\\begin\{tabular\}.*?\\end\{tabular\}", re.DOTALL)
    selected_variant_hashes: dict[str, str] = {}
    selected_figure_selections: list[str] = []
    interval_calibration_populations: list[str] = []
    for item in config["items"]:
        manifest_path = _resolve(root, str(item["manifest"])) if item.get("manifest") else None
        if not manifest_path or not manifest_path.is_file():
            continue
        selected_manifest = _read_manifest(manifest_path)
        identity = selected_manifest.get("scientific_identity", {})
        checkpoint_hashes = identity.get("checkpoint_sha256s", {})
        if isinstance(checkpoint_hashes, dict):
            selected_variant_hashes.update({str(k): str(v) for k, v in checkpoint_hashes.items()})
        for variant in selected_manifest.get("panel_variants", {}).values():
            selected_variant_hashes.setdefault(str(variant), str(checkpoint_hashes.get(variant, "not selected")))
        selection = selected_manifest.get("selection")
        if selection:
            selected_figure_selections.append(
                f"{item['item_id']}: IDs={selection.get('ids')}; levels={selection.get('levels')}; "
                f"sampling={selection.get('sampling')}; seed={selection.get('seed')}; filters={selection.get('filters')}"
            )
        if selected_manifest.get("ci_population") or selected_manifest.get("calibration_population"):
            interval_calibration_populations.append(
                f"{item['item_id']}: CI={selected_manifest.get('ci_population', 'not reported')}; "
                f"calibration={selected_manifest.get('calibration_population', 'not reported')}"
            )
    for mode, key in (("review", "manuscript_review"), ("submission", "manuscript_submission")):
        source = _resolve(root, str(history.get(key, ""))) if history.get(key) else None
        if not source or not source.is_file():
            raise PublicationValidationError(f"historical {mode} manuscript source missing: {source or '<unset>'}")
        log_artifact_path(f"historical {mode} manuscript", source)
        content = source.read_text(encoding="utf-8")
        content, review_notes = _extract_review_notes(content)
        result_start = content.find(r"\section{Results}")
        discussion_start = content.find(r"\section{Discussion}")
        if result_start < 0 or discussion_start < 0 or discussion_start <= result_start:
            raise PublicationValidationError(f"{mode} manuscript lacks ordered Results and Discussion sections")
        old_results = content[result_start:discussion_start]
        result_envs = re.findall(r"\\begin\{(?:table|figure)\}(?:\[[^]]*\])?.*?\\end\{(?:table|figure)\}", old_results, re.DOTALL)
        tab_index = 0
        new_envs: list[str] = []
        for block in result_envs:
            if block.startswith(r"\begin{table}"):
                tab_index += 1
                if tab_index > 12:
                    raise PublicationValidationError(f"{mode} manuscript has more than 12 result tables")
                replacement = rf"\input{{tables/table_{tab_index}.tex}}"
                block, count = tabular.subn(lambda _match, value=replacement: value, block, count=1)
                if count != 1:
                    raise PublicationValidationError(f"{mode} manuscript Table {tab_index} has no tabular body")
            new_envs.append(block)
        if tab_index != 12:
            raise PublicationValidationError(f"{mode} manuscript contains {tab_index} tables; expected 12")
        results = (
            r"\section{Results}" + "\n" +
            "The following tables and figures are assembled from the explicitly selected C06 evidence. "
            "Each matrix row records provenance, support, transformation and evidence status. Historical "
            "values are retained separately and are not substituted for missing empirical inputs.\n\n" +
            "\n\n".join(new_envs) + "\n\n"
        )
        methods_start = content.find(r"\section{Methods}")
        if methods_start < 0 or methods_start >= result_start:
            raise PublicationValidationError(f"{mode} manuscript lacks Methods before Results")
        methods = (
            r"\section{Methods}" + "\n"
            "The analyses use explicitly selected, versioned inputs and preserve the class order, checkpoint, "
            "split identity, seed, transformation and support declared by their manifests. The primary task is "
            "closed-set classification of known taxa on fragments; the split is not a genome-disjoint test. "
            "Input windows use 512 positions and channels A/C/G/T, with ambiguous or padded positions encoded "
            "as null vectors. The substitution benchmark resamples selected positions uniformly over A/C/G/T, "
            "including the original nucleotide, so the expected effective change rate is $3\\epsilon/4$; it does "
            "not model insertions, deletions or context-dependent sequencing errors. Embedding KNN uses the "
            "declared high-dimensional representation, $k=5$, $L_2$ normalization and cosine distance; UMAP is "
            "used only for visualization. Triplet composition summaries are exploratory and do not establish "
            "codon use in annotated coding frames. Observed model/checkpoint selection: "
            + ("; ".join(f"{k}={v}" for k, v in sorted(selected_variant_hashes.items())) or "not identified") + ".\n"
            "Figure selections: " + ("; ".join(selected_figure_selections) or "not reported") + ".\n"
            "Confidence-interval/calibration populations: "
            + ("; ".join(interval_calibration_populations) or "not reported by selected manifests") + ".\n"
            "Long-sequence windowing, aggregation and metric denominators are taken "
            "from the selected manifests and are reported separately from fragment evaluation.\n\n"
            f"The controlled C05 scratch comparison is {config.get('c05_status', 'NOT_RUN')}; where it is not complete, "
            "claims remain limited to the configurations with complete evidence.\n\n"
        )
        content = content[:methods_start] + methods + content[result_start:]
        result_start = content.find(r"\section{Results}")
        discussion_start = content.find(r"\section{Discussion}", result_start)
        content = content[:result_start] + results + content[discussion_start:]
        discussion_start = content.find(r"\section{Discussion}")
        conclusion_start = content.find(r"\section{Conclusions}", discussion_start)
        if conclusion_start < 0:
            raise PublicationValidationError(f"{mode} manuscript lacks Conclusions section")
        discussion = (
            r"\section{Discussion}" + "\n"
            "Interpretation is limited to the populations, variants and protocols identified by the selected "
            "evidence. Fragment-level closed-set evaluation does not establish performance on unseen genomes. "
            "Uniform resampling has expected effective change rate $3\\epsilon/4$ and does not represent "
            "context-dependent errors, insertions or deletions. Latent-space neighborhood results use the "
            "declared high-dimensional embedding and cosine distance; UMAP is descriptive only. Unknown "
            "genome provenance and inconclusive mechanisms remain unresolved rather than being inferred.\n\n"
            r"\subsection{Limitations}" + "\n"
            "The evidence matrix distinguishes implementation checks from empirical regeneration and editorial "
            "readiness. Missing or incompatible inputs remain explicit gaps. Compilation does not certify "
            "scientific claims or editorial approval.\n\n"
        )
        content = content[:discussion_start] + discussion + content[conclusion_start:]
        conclusion_start = content.find(r"\section{Conclusions}")
        availability_start = content.find(r"\subsection*{Availability of data and materials}", conclusion_start)
        if availability_start < 0:
            raise PublicationValidationError(f"{mode} manuscript lacks availability section")
        conclusion = (
            r"\section{Conclusions}" + "\n"
            "The C06 package reports only results supported by selected, traceable evidence. Historical values "
            "and unresolved scientific or editorial questions remain separately identified. This package does "
            "not establish performance on unseen genomes or a causal explanation for long-sequence changes. "
            "It does not claim full empirical reproduction unless all required evidence gates pass.\n\n"
        )
        content = content[:conclusion_start] + conclusion + content[availability_start:]
        # Keep the review/submission note distinction while making stale abstract results non-numeric.
        content = re.sub(
            r"(\\textbf\{Results\}:).*?(?=\\par\\vspace\{0\.5em\}\\noindent\\textbf\{Conclusions\}:)",
            r"\1 Results and figures are derived from the selected evidence package; gaps and historical discrepancies are listed in its evidence matrix.\n\n",
            content, count=1, flags=re.DOTALL,
        )
        content = re.sub(
            r"(\\textbf\{Conclusions\}:).*?(?=\\end\{abstract\})",
            r"\1 Fragment-level evaluation does not establish performance on unseen genomes; unresolved mechanisms are not demonstrated causes.\n\n",
            content, count=1, flags=re.DOTALL,
        )
        if mode == "review" and review_notes:
            notes_block = "\\section*{Editorial review notes}\n" + "\n\n".join(review_notes) + "\n"
            content = content.replace(r"\end{document}", notes_block + r"\end{document}", 1)
        target = output_dir / f"manuscript_{mode}.tex"
        target.write_text(content, encoding="utf-8")
        generated[mode] = target
    bibliography = _resolve(root, "paper/references.bib")
    if bibliography.is_file():
        log_artifact_path("historical manuscript bibliography", bibliography)
        shutil.copy2(bibliography, output_dir / bibliography.name)
    return generated


def write_editorial_review(output_dir: Path) -> Path:
    """Record editorial checks as pending until a person supplies review evidence."""
    report = {
        "references": {"status": "PENDING", "evidence": None},
        "affiliations": {"status": "PENDING", "evidence": None},
        "wording": {"status": "PENDING", "evidence": None},
        "compilation_is_approval": False,
    }
    target = Path(output_dir) / "editorial_review.json"
    target.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return target


def compile_manuscript_package(tex_paths: dict[str, Path], output_dir: Path) -> dict[str, dict[str, Any]]:
    """Compile each manuscript in an isolated package directory and retain logs."""
    output_dir = Path(output_dir)
    results: dict[str, dict[str, Any]] = {}
    for mode, tex_path in tex_paths.items():
        run_dir = output_dir / mode
        run_dir.mkdir(parents=True, exist_ok=True)
        command = ["pdflatex", "-interaction=nonstopmode", f"-output-directory={run_dir}", str(Path(tex_path).resolve())]
        try:
            first = subprocess.run(command, cwd=Path(tex_path).parent, capture_output=True, text=True, errors="replace")
        except OSError as exc:
            first = subprocess.CompletedProcess(command, 127, "", str(exc))
        second = first
        if first.returncode == 0:
            try:
                second = subprocess.run(command, cwd=Path(tex_path).parent, capture_output=True, text=True, errors="replace")
            except OSError as exc:
                second = subprocess.CompletedProcess(command, 127, "", str(exc))
        log = run_dir / "compile.log"
        log.write_text(first.stdout + first.stderr + "\n--- second pass ---\n" + second.stdout + second.stderr, encoding="utf-8")
        pdf = run_dir / f"{Path(tex_path).stem}.pdf"
        if second.returncode == 0 and (not pdf.is_file() or not pdf.read_bytes().startswith(b"%PDF-")):
            second_return = 1
        else:
            second_return = second.returncode
        results[mode] = {"returncode": second_return, "pdf": str(pdf), "log": str(log), "command": command}
    return results


def validate_manuscript_paths(tex_paths: dict[str, Path], package_root: Path) -> dict[str, list[str]]:
    """Validate case-sensitive table and figure references before compilation."""
    from viral_cnn.reporting.figures import verify_latex_figure_paths

    package_root = Path(package_root)
    errors: dict[str, list[str]] = {}
    input_pattern = re.compile(r"\\input\{([^}]+)\}")
    for mode, tex_path in tex_paths.items():
        content = Path(tex_path).read_text(encoding="utf-8")
        mode_errors = []
        for ref in input_pattern.findall(content):
            target = package_root / ref
            if not target.is_file():
                mode_errors.append(f"missing or case-mismatched table input: {ref}")
        _matches, mismatches = verify_latex_figure_paths(Path(tex_path), package_root / "figures")
        mode_errors.extend(f"{row['status']}: {row['ref']}" for row in mismatches)
        if mode_errors:
            errors[mode] = mode_errors
    return errors


def _git_state(project_root: Path) -> tuple[str, bool]:
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=project_root, text=True, stderr=subprocess.DEVNULL).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=project_root, text=True, stderr=subprocess.DEVNULL).strip())
        return commit, dirty
    except (OSError, subprocess.CalledProcessError):
        return "unavailable", True


def write_publication_status_report(
    output_dir: Path, rows: list[dict[str, Any]], outputs: dict[str, Path], *,
    command: str, interface: str, gate_states: dict[str, str], manifest_path: Path,
) -> Path:
    """Write a concise report separating diagnostics from reproduction claims."""
    missing = [row for row in rows if row["status"] not in {"EMPIRICAL_VERIFIED", "DERIVED_VERIFIED"}]
    reconciliation = outputs.get("reconciliation:csv")
    if reconciliation and Path(reconciliation).is_file():
        discrepancy_summary = f"Reconciliation rows: {len(pd.read_csv(reconciliation))}."
    else:
        discrepancy_summary = "Numeric discrepancies were not computed because required evidence was unavailable."
    lines = [
        "# C06 publication assembly status", "",
        f"- Interface: {interface}", f"- Command: `{command or '<not supplied>'}`",
        f"- Manifest: `{manifest_path}`", f"- Gates: `{json.dumps(gate_states, sort_keys=True)}`", "",
        "## Outputs", "",
        *[f"- `{key}`: `{path}`" for key, path in sorted(outputs.items())], "",
        "## Unavailable or inadmissible selected dependencies", "",
        *([f"- {row['item_id']} [{row['status']}]: {row['diagnostic']}" for row in missing]
          or ["- None; all selected evidence rows passed local admissibility."]), "",
        "## Discrepancies", "", discrepancy_summary, "",
        "This report distinguishes local assembly from empirical reproduction and editorial readiness. "
        "A diagnostic or synthetic run does not reproduce the article.", "",
    ]
    target = Path(output_dir) / "publication_status.md"
    target.write_text("\n".join(lines), encoding="utf-8")
    return target


def write_publication_manifest(
    config: dict[str, Any], output_dir: Path, rows: list[dict[str, Any]], outputs: dict[str, Path],
    *, command: str, interface: str, gate_states: dict[str, str],
) -> Path:
    """Persist run identity, provenance, output hashes, versions, and readiness gates."""
    root = Path(config["_project_root"])
    log_artifact_path("C06 publication configuration", Path(config["_config_path"]))
    commit, dirty = _git_state(root)
    versions = {"python": platform.python_version()}
    for package in ("torch", "numpy", "pandas", "pyarrow", "matplotlib", "PyYAML"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "unavailable"
    historical_sources = {}
    for name, value in config.get("historical_source", {}).items():
        if not value:
            continue
        source = _resolve(root, value)
        if source.is_file():
            log_artifact_path(f"historical source {name}", source)
            historical_sources[name] = {"path": str(source), "sha256": sha256_file(source)}
        elif source.is_dir():
            files = sorted(path for path in source.rglob("*") if path.is_file())
            for path in files:
                log_artifact_path(f"historical source {name}", path)
            historical_sources[name] = {
                "path": str(source), "files": {str(path.relative_to(source)): sha256_file(path) for path in files}
            }
    manifest = {
        "publication_protocol": PUBLICATION_PROTOCOL, "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(), "command": command,
        "run_id": (
            Path(output_dir).resolve().parent.name
            if Path(output_dir).resolve().name == "publication"
            else Path(output_dir).resolve().name
        ),
        "interface": interface, "git_commit": commit, "git_dirty": dirty,
        "configuration": config["_config_path"], "configuration_sha256": sha256_file(Path(config["_config_path"])),
        "historical_sources": historical_sources,
        "versions": versions, "inputs": rows,
        "outputs": {key: {"path": str(path), "sha256": sha256_file(path)} for key, path in outputs.items() if path.is_file()},
        "gate_states": gate_states,
    }
    target = Path(output_dir) / "publication_manifest.json"
    target.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return target


PROPOSED_SOURCES: dict[str, str] = {
    "T1": "results_reproduction/runs/<run_id>/short/01/ (short evaluation metrics)",
    "T2": "results_reproduction/runs/<run_id>/short/02/ (profiling and latency)",
    "T3": "results_reproduction/runs/<run_id>/short/09/ (noise degradation benchmark)",
    "T4": "results_reproduction/runs/<run_id>/short/01/ (species F1 distribution)",
    "T5": "results_reproduction/runs/<run_id>/short/06/ (latent space KNN/Silhouette)",
    "T6": "results_reproduction/runs/<run_id>/short/06/ (hierarchy correlations)",
    "T7": "results_reproduction/runs/<run_id>/short/06b/ (hierarchical entropy)",
    "T8": "results_reproduction/runs/<run_id>/short/07/motif_perturbation_by_variant.csv",
    "T9": "results_reproduction/runs/<run_id>/short/08/ (exploratory RSCU)",
    "T10": "results_reproduction/runs/<run_id>/short/08/ (triplet composition)",
    "T11": "results_reproduction/runs/<run_id>/short/08/ (permutation null architecture comparison)",
    "T12": "results_reproduction/runs/<run_id>/long/03/table_12_long_voter_comparison.csv",
    "F1": "results_reproduction/runs/<run_id>/short/09/ (noise robustness curves)",
    "F2": "results_reproduction/runs/<run_id>/short/06b/selected_genera.csv (UMAP projections)",
    "F3": "results_reproduction/runs/<run_id>/short/07/motif_perturbation_by_variant.csv (5 panels)",
    "F4": "results_reproduction/runs/<run_id>/short/08/ (RSCU profile data)",
    "F5": "results_reproduction/runs/<run_id>/short/08/edge_table.csv (network graph coordinates)",
}

FIVE_SPECIES_VARIANTS = {
    "ResNet50_Clean",
    "ConvNeXt_Z",
    "ConvNeXt_T",
    "ResNet50_F",
    "ConvNeXt_F",
}


def diagnose_publication_readiness(
    config_or_path: dict[str, Any] | Path | str,
    *,
    project_root: Path | None = None,
    strict_empirical: bool = True,
) -> pd.DataFrame:
    """Audit all 17 publication items and classify them into 5 granular readiness states:

    - OK: all artifact, manifest, schema, and scientific invariants satisfied.
    - FALTA: artifact or manifest path is unset or missing on disk.
    - SCHEMA_MISMATCH: file exists, but table columns or plot coordinates are missing.
    - IDENTITY_MISMATCH: seed, split_id, checkpoints, or class order mismatch.
    - SCIENTIFIC_MISMATCH: image file passed as source, synthetic evidence under empirical mode,
      F3 missing 5 models, T11 claiming architecture correlation without permutation test,
      or F5 network graph missing edge coordinates.
    """
    if isinstance(config_or_path, (str, Path)):
        config = load_publication_config(Path(config_or_path))
    else:
        config = config_or_path

    root = Path(project_root or config.get("_project_root", Path.cwd()))
    results: list[dict[str, Any]] = []

    for item in config.get("items", []):
        iid = item.get("item_id", "")
        proposed = PROPOSED_SOURCES.get(iid, "results_reproduction/runs/<run_id>/...")
        artifact_raw = item.get("artifact", "")
        manifest_raw = item.get("manifest", "")

        artifact = _resolve(root, str(artifact_raw)) if artifact_raw else None
        manifest = _resolve(root, str(manifest_raw)) if manifest_raw else None

        # Check FALTA
        if not artifact or not artifact.exists() or not manifest or not manifest.is_file():
            missing_parts = []
            if not artifact or not artifact.exists():
                missing_parts.append(f"artifact missing: {artifact_raw or '<unset>'}")
            if not manifest or not manifest.is_file():
                missing_parts.append(f"manifest missing: {manifest_raw or '<unset>'}")
            results.append({
                "item_id": iid,
                "status": "FALTA",
                "diagnostic": "; ".join(missing_parts),
                "proposed_source": proposed,
                "recommendation": f"Generate or select {iid} output from {proposed}",
            })
            continue

        log_artifact_path(f"Diagnosing publication item {iid}", artifact)

        # Check if an image file was supplied instead of tabular data (SCIENTIFIC_MISMATCH)
        if artifact.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tiff", ".svg"}:
            results.append({
                "item_id": iid,
                "status": "SCIENTIFIC_MISMATCH",
                "diagnostic": f"Image file ({artifact.suffix}) passed as publication source; tabular data (CSV/Parquet) required",
                "proposed_source": proposed,
                "recommendation": "Provide structured CSV or Parquet data representing the underlying numbers",
            })
            continue

        # Load manifest
        try:
            declared_manifest = _read_manifest(manifest)
        except Exception as exc:
            results.append({
                "item_id": iid,
                "status": "SCHEMA_MISMATCH",
                "diagnostic": f"Manifest unreadable: {exc}",
                "proposed_source": proposed,
                "recommendation": "Re-emit manifest with valid JSON format",
            })
            continue

        # Check synthetic evidence in strict empirical mode
        if strict_empirical and (declared_manifest.get("is_smoke_test") is True or declared_manifest.get("synthetic") is True):
            results.append({
                "item_id": iid,
                "status": "SCIENTIFIC_MISMATCH",
                "diagnostic": "Selected evidence declares synthetic/smoke mode, forbidden under strict empirical publication",
                "proposed_source": proposed,
                "recommendation": "Run stage in empirical mode (SMOKE_TEST=0) on cluster to produce real evidence",
            })
            continue

        # Check hash
        actual_hash = sha256_path(artifact)
        if item.get("sha256") and item.get("sha256") != actual_hash:
            results.append({
                "item_id": iid,
                "status": "IDENTITY_MISMATCH",
                "diagnostic": f"Artifact hash mismatch: config={item.get('sha256')[:10]} vs disk={actual_hash[:10]}",
                "proposed_source": proposed,
                "recommendation": "Update configuration sha256 or regenerate artifact",
            })
            continue

        # Check scientific identity
        expected = item.get("expected", {})
        expected_identity = expected.get("scientific_identity", {})
        declared_identity = declared_manifest.get("scientific_identity", {})
        identity_errors = []

        for field in ("seed", "split_id"):
            if field in expected_identity and expected_identity[field] not in (None, "", "TO_BE_SELECTED"):
                if declared_identity.get(field) != expected_identity.get(field):
                    identity_errors.append(f"{field} mismatch: expected {expected_identity.get(field)}, got {declared_identity.get(field)}")

        if identity_errors:
            results.append({
                "item_id": iid,
                "status": "IDENTITY_MISMATCH",
                "diagnostic": "; ".join(identity_errors),
                "proposed_source": proposed,
                "recommendation": "Re-run stage with matching split and seed parameters",
            })
            continue

        # Load artifact table
        try:
            table_frame = _load_table_artifact(artifact)
        except Exception as exc:
            results.append({
                "item_id": iid,
                "status": "SCHEMA_MISMATCH",
                "diagnostic": f"Table artifact unreadable: {exc}",
                "proposed_source": proposed,
                "recommendation": "Verify file integrity and CSV/Parquet format",
            })
            continue

        # Schema check for tables
        if iid.startswith("T"):
            expected_columns = expected.get("columns", [])
            if expected_columns and not set(expected_columns).issubset(table_frame.columns):
                missing_cols = list(set(expected_columns) - set(table_frame.columns))
                results.append({
                    "item_id": iid,
                    "status": "SCHEMA_MISMATCH",
                    "diagnostic": f"Missing expected table columns: {missing_cols}",
                    "proposed_source": proposed,
                    "recommendation": f"Add missing columns {missing_cols} via registered transformation",
                })
                continue

            # Scientific contract for T11
            if iid == "T11":
                if "is_permutation_tested" in table_frame.columns and not table_frame["is_permutation_tested"].all():
                    results.append({
                        "item_id": iid,
                        "status": "SCIENTIFIC_MISMATCH",
                        "diagnostic": "T11 co-occurrence matrix without permutation null test cannot claim architecture correlation",
                        "proposed_source": proposed,
                        "recommendation": "Apply permutation null test across nodes and report empirical p-values",
                    })
                    continue
                if "correlation_r" in table_frame.columns and (table_frame["correlation_r"] == 1.0).all():
                    if "has_variance" in table_frame.columns and not table_frame["has_variance"].all():
                        results.append({
                            "item_id": iid,
                            "status": "SCIENTIFIC_MISMATCH",
                            "diagnostic": "T11 exhibits spurious r=1 correlation from identical or zero-variance matrices",
                            "proposed_source": proposed,
                            "recommendation": "Evaluate independent model representations with verified variance",
                        })
                        continue

        # Schema & scientific checks for figures
        if iid.startswith("F"):
            plot = item.get("plot", {})
            kind = plot.get("kind", "line")

            if kind == "network_edges":
                edge_cols = [plot.get(c) for c in ("x1", "y1", "x2", "y2") if plot.get(c)]
                if len(edge_cols) < 4 or not all(c in table_frame.columns for c in edge_cols):
                    results.append({
                        "item_id": iid,
                        "status": "SCHEMA_MISMATCH",
                        "diagnostic": f"F5 network graph missing required 2D coordinates ({edge_cols})",
                        "proposed_source": proposed,
                        "recommendation": "Export edge_table.csv with explicit x1, y1, x2, y2 node coordinates",
                    })
                    continue
            elif kind == "panel_grid":
                x_col, y_col = plot.get("x"), plot.get("y")
                if not {"panel", "variant", x_col, y_col}.issubset(table_frame.columns):
                    results.append({
                        "item_id": iid,
                        "status": "SCHEMA_MISMATCH",
                        "diagnostic": f"F3 panel grid missing required columns (panel, variant, {x_col}, {y_col})",
                        "proposed_source": proposed,
                        "recommendation": "Format motif perturbation data with panel, variant, motif, activation_drop",
                    })
                    continue

                # Scientific contract for F3: requires 5 models
                variants_present = set(table_frame["variant"].astype(str))
                missing_variants = FIVE_SPECIES_VARIANTS - variants_present
                if missing_variants:
                    results.append({
                        "item_id": iid,
                        "status": "SCIENTIFIC_MISMATCH",
                        "diagnostic": f"F3 missing required species model variants: {missing_variants}",
                        "proposed_source": proposed,
                        "recommendation": "Include all 5 species models (including ConvNeXt_F) in motif perturbation",
                    })
                    continue
            else:
                x_col, y_col = plot.get("x"), plot.get("y")
                if x_col not in table_frame.columns or y_col not in table_frame.columns:
                    results.append({
                        "item_id": iid,
                        "status": "SCHEMA_MISMATCH",
                        "diagnostic": f"Figure missing x/y columns: {x_col}, {y_col}",
                        "proposed_source": proposed,
                        "recommendation": f"Ensure columns {x_col} and {y_col} exist in data",
                    })
                    continue

        # If it reached here without failing, it is OK
        results.append({
            "item_id": iid,
            "status": "OK",
            "diagnostic": "All artifact, manifest, schema, and scientific contracts verified.",
            "proposed_source": proposed,
            "recommendation": "Ready for publication assembly.",
        })

    return pd.DataFrame(results)


def assemble_publication(
    config_path: Path, output_dir: Path, *, strict_empirical: bool = True,
    command: str = "", interface: str = "cli", diagnostic_only: bool = False,
) -> dict[str, Any]:
    """Assemble a C06 package into a caller-reserved, isolated run directory.

    Incomplete selections always produce a diagnostics matrix. Strict mode returns a
    blocked result and does not emit tables, figures, or manuscripts from fallbacks.
    """
    config = load_publication_config(config_path)
    output_dir = Path(output_dir).resolve()
    if output_dir.exists():
        if not output_dir.is_dir() or any(output_dir.iterdir()):
            raise FileExistsError(f"Refusing to reuse non-empty publication output: {output_dir}")
    else:
        output_dir.mkdir(parents=True, exist_ok=False)

    # Run batch diagnosis
    diag_df = diagnose_publication_readiness(config, project_root=Path(config["_project_root"]), strict_empirical=strict_empirical)
    diag_csv = output_dir / "publication_diagnostic_report.csv"
    diag_md = output_dir / "publication_diagnostic_report.md"
    diag_df.to_csv(diag_csv, index=False)
    from viral_cnn.reporting.tables import _df_to_markdown
    diag_md.write_text(_df_to_markdown(diag_df), encoding="utf-8")

    rows = validate_selection(config, strict_empirical=False)
    matrix = evidence_matrix(config, rows)
    matrix_path = output_dir / "evidence_matrix.csv"
    matrix.to_csv(matrix_path, index=False)
    outputs: dict[str, Path] = {
        "evidence_matrix": matrix_path,
        "diagnostic_report_csv": diag_csv,
        "diagnostic_report_md": diag_md,
    }
    all_ready = len(rows) == 17 and all(r["status"] in {"EMPIRICAL_VERIFIED", "DERIVED_VERIFIED"} for r in rows)
    gates = {"local": "PASS", "empirical": "PASS" if all_ready else "BLOCKED", "scientific": "PENDING", "editorial": "PENDING"}
    compile_results: dict[str, dict[str, Any]] = {}
    if all_ready and not diagnostic_only:
        try:
            outputs.update(render_tables(config, output_dir / "tables", rows))
            outputs.update(render_figures(config, output_dir / "figures", rows))
            manuscripts = render_manuscript_variants(config, output_dir)
            outputs.update({f"manuscript_{mode}": path for mode, path in manuscripts.items()})
            outputs["editorial_review"] = write_editorial_review(output_dir)
            path_errors = validate_manuscript_paths(manuscripts, output_dir)
            if path_errors:
                raise PublicationValidationError(f"case-sensitive manuscript path validation failed: {path_errors}")
            compile_results = compile_manuscript_package(manuscripts, output_dir)
            for mode, report in compile_results.items():
                pdf_path, log_path = Path(report["pdf"]), Path(report["log"])
                if pdf_path.is_file():
                    outputs[f"compile_{mode}:pdf"] = pdf_path
                outputs[f"compile_{mode}:log"] = log_path
            if any(report["returncode"] != 0 for report in compile_results.values()):
                raise PublicationValidationError(f"manuscript compilation failed: {compile_results}")
        except Exception as exc:
            gates["local"] = "BLOCKED"
            error_path = output_dir / "assembly_errors.json"
            error_path.write_text(json.dumps({"error": str(exc), "type": type(exc).__name__}, indent=2) + "\n", encoding="utf-8")
            outputs["assembly_errors"] = error_path
        gates["scientific"] = "PENDING_REVIEW"
        gates["editorial"] = "PENDING_REVIEW"
    elif strict_empirical:
        gates["empirical"] = "BLOCKED"
    report_path = write_publication_status_report(
        output_dir, rows, outputs, command=command, interface=interface, gate_states=gates,
        manifest_path=output_dir / "publication_manifest.json",
    )
    outputs["publication_status_report"] = report_path
    manifest_path = write_publication_manifest(
        config, output_dir, rows, outputs, command=command, interface=interface, gate_states=gates,
    )
    if strict_empirical and not all_ready:
        return {"status": "BLOCKED", "rows": rows, "matrix": matrix_path, "manifest": manifest_path, "gates": gates, "diagnostics": diag_df}
    status = "DIAGNOSTIC" if not all_ready or diagnostic_only else (
        "FAILED" if gates["local"] != "PASS" else "ASSEMBLED_REVIEW_REQUIRED"
    )
    return {"status": status, "rows": rows, "matrix": matrix_path, "manifest": manifest_path,
            "outputs": outputs, "gates": gates, "compilation": compile_results, "diagnostics": diag_df}
