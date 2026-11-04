"""Audit helpers for identifiable fine-tuning runs."""

from __future__ import annotations

import hashlib
import csv
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from viral_cnn.utils.logging_utils import log_artifact_path


def sha256_file(path: str | Path) -> str:
    """Return the SHA-256 digest of a file while logging its source path."""
    source = Path(path).resolve()
    log_artifact_path("fine-tuning input", source)
    digest = hashlib.sha256()
    with source.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_empirical_inputs(
    train_files: Sequence[str | Path],
    validation_files: Sequence[str | Path],
    test_files: Sequence[str | Path],
    taxonomy_mapping: str | Path,
    base_checkpoint: str | Path,
) -> dict[str, Any]:
    """Fail closed unless empirical train, validation, taxonomy and weights exist."""
    required = {
        "train": [Path(path) for path in train_files],
        "validation": [Path(path) for path in validation_files],
        "test": [Path(path) for path in test_files],
        "taxonomy_mapping": [Path(taxonomy_mapping)],
        "base_checkpoint": [Path(base_checkpoint)],
    }
    missing = [f"{group}: {path}" for group, paths in required.items() for path in paths if not path.is_file()]
    empty = [group for group, paths in required.items() if not paths]
    if missing or empty:
        details = "; ".join([*(f"missing {item}" for item in missing), *(f"no {group} files" for group in empty)])
        raise FileNotFoundError(f"Empirical C05 run blocked before model construction: {details}")
    return {
        group: [{"path": str(path.resolve()), "sha256": sha256_file(path)} for path in paths]
        for group, paths in required.items()
    }


def taxonomy_order_sha256(path: str | Path) -> tuple[str, int]:
    """Hash the ordered 1-based taxonomy IDs and labels used by the logits head."""
    source = Path(path)
    log_artifact_path("taxonomy class order", source)
    rows: list[tuple[int, str]] = []
    with source.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or "id" not in reader.fieldnames:
            raise ValueError(f"Taxonomy mapping must contain an 'id' column: {source}")
        label_column = next((name for name in reader.fieldnames if name.lower() in {"species", "genus", "order"}), None)
        if label_column is None:
            raise ValueError(f"Taxonomy mapping has no recognized rank column: {source}")
        for row in reader:
            rows.append((int(row["id"]), row[label_column]))
    ids = [item[0] for item in rows]
    if ids != list(range(1, len(ids) + 1)):
        raise ValueError("Taxonomy IDs must be in canonical contiguous 1-based logits order")
    canonical = "\n".join(f"{tax_id}\t{label}" for tax_id, label in rows).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest(), len(rows)
def protocol_audit(
    *,
    base_variant: str,
    base_checkpoint_sha256: str,
    rank: str,
    num_classes: int,
    class_order_sha256: str,
    effective_dropout: float | None,
    dropout_source: str,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Build a factual protocol record without inferring settings from filenames."""
    return {
        "protocol": "c05_finetune_audit_v1",
        "base_variant": base_variant,
        "base_checkpoint_sha256": base_checkpoint_sha256,
        "rank": rank,
        "num_classes": int(num_classes),
        "class_order_sha256": class_order_sha256,
        "effective_dropout": effective_dropout,
        "dropout_source": dropout_source,
        "configuration": dict(config),
        "historical_comparison": {
            "scheduler_step_unit": "epoch",
            "resnet_selection": "last",
            "convnext_selection": "best_val_macro_f1; historical best-val-loss claim is unverified",
            "convnext_dropout_registry": 0.7,
            "convnext_dropout_historical_notebook": 0.5,
            "published_checkpoint_equivalence": "unknown without original run configuration/logs",
        },
    }


def select_checkpoint_epoch(history: Sequence[Mapping[str, Any]], rule: str) -> int:
    """Select a 1-based epoch using the declared rule; ties keep the first epoch."""
    if not history:
        raise ValueError("Cannot select a checkpoint from empty training history")
    if rule == "last":
        return int(history[-1]["epoch"])
    if rule != "best_val_macro_f1":
        raise ValueError(f"Unsupported C05 checkpoint selection rule: {rule}")
    return int(max(history, key=lambda row: float(row.get("val_f1_macro", 0.0)))["epoch"])


def write_c05_manifest(
    path: str | Path,
    *,
    protocol: Mapping[str, Any],
    status: str,
    is_smoke_test: bool,
    inputs: Mapping[str, Any],
    outputs: Sequence[str | Path],
    completed_epochs: int,
    planned_epochs: int,
    selected_checkpoint: Mapping[str, Any] | None,
    metrics: Mapping[str, Any],
    execution: Mapping[str, Any] | None = None,
    checkpoint_roles: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Write a C05 manifest whose status cannot promote smoke or incomplete runs."""
    if status not in {"RUNNING", "PARTIAL", "FAILED", "COMPLETE", "NOT_RUN"}:
        raise ValueError(f"Unsupported C05 run status: {status}")
    effective_status = "SYNTHETIC" if is_smoke_test else status
    if status == "COMPLETE" and (is_smoke_test or completed_epochs < planned_epochs):
        effective_status = "PARTIAL" if not is_smoke_test else "SYNTHETIC"
    if status == "COMPLETE" and not selected_checkpoint and not is_smoke_test:
        effective_status = "PARTIAL"
    output_records = []
    for output in outputs:
        output_path = Path(output)
        output_records.append({
            "path": str(output_path.resolve()),
            "sha256": sha256_file(output_path) if output_path.is_file() else None,
        })
    manifest = {
        **dict(protocol),
        "status": effective_status,
        "requested_status": status,
        "is_smoke_test": bool(is_smoke_test),
        "inputs": dict(inputs),
        "outputs": output_records,
        "completed_epochs": int(completed_epochs),
        "planned_epochs": int(planned_epochs),
        "selected_checkpoint": dict(selected_checkpoint) if selected_checkpoint else None,
        "checkpoint_roles": dict(checkpoint_roles or {}),
        "metrics": dict(metrics),
        "execution": dict(execution or {}),
    }
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    os.replace(temporary, destination)
    return manifest
