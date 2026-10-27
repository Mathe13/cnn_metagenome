"""Audit label-to-taxonomy mapping against observed train/test overlap.

The ICTV/NCBI mapping is one row per taxon, not one row per source genome.
Taxonomic validation must never be represented as genome provenance.
"""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def _rows(path: str | Path) -> list[dict[str, Any]]:
    file = Path(path).expanduser().resolve()
    if not file.is_file():
        raise FileNotFoundError(file)
    if file.suffix.lower() == ".csv":
        with file.open(newline="", encoding="utf-8") as handle:
            return list(csv.DictReader(handle))
    if file.suffix.lower() in (".parquet", ".pq"):
        import pyarrow.parquet as pq

        return pq.read_table(file).to_pylist()  # one row per class, ~13,683 rows
    raise ValueError("Taxonomy mapping must be CSV or Parquet")


def _label_from_id(value: Any) -> int:
    try:
        number = int(value)
    except (ValueError, TypeError) as error:
        raise ValueError(f"Invalid 1-based taxonomy id: {value!r}") from error
    if number < 1 or str(value).strip() not in (str(number), f"{number}.0"):
        raise ValueError(f"Taxonomy id must be a positive integer: {value!r}")
    return number - 1


def audit_taxonomy_against_overlap(
    taxonomy_path: str | Path,
    per_species_path: str | Path,
    decision_path: str | Path,
    output_dir: str | Path,
    original_mapping_path: str | Path | None = None,
) -> dict[str, Any]:
    """Check label integrity and stratify *observed* exact overlap by taxonomy.

    Does not infer accession, coordinate, independent genome count, or retraining
    necessity. Accepts an older unpadded overlap report but labels it as a lower
    bound, never as the final strand-invariant overlap percentage.
    """
    taxonomy = _rows(taxonomy_path)
    species_rows = _rows(per_species_path)
    with Path(decision_path).expanduser().open(encoding="utf-8") as handle:
        decision = json.load(handle)
    if not taxonomy or not species_rows:
        raise ValueError("Taxonomy and per-species overlap reports must not be empty")
    for col in ("id", "species"):
        if col not in taxonomy[0]:
            raise ValueError(f"Taxonomy mapping missing '{col}' (expected ICTV notebook output)")
    for col in ("species_id", "test_rows", "exact_matches"):
        if col not in species_rows[0]:
            raise ValueError(f"Overlap report missing '{col}'")
    if decision.get("scope") != "full_test":
        raise ValueError("Taxonomic audit requires a full-test overlap report, not a pilot")

    original = _rows(original_mapping_path) if original_mapping_path else []
    original_by_id = {_label_from_id(row["id"]): str(row["species"]).strip() for row in original}
    original_ids = [_label_from_id(row["id"]) for row in original]
    if len(set(original_ids)) != len(original_ids):
        raise ValueError("Original species_name_ids.csv has duplicate IDs")

    by_label: dict[int, dict[str, Any]] = {}
    duplicated: set[int] = set()
    for row in taxonomy:
        label = _label_from_id(row["id"])
        if label in by_label:
            duplicated.add(label)
        else:
            by_label[label] = row

    observed: dict[int, dict[str, int]] = {}
    for row in species_rows:
        label = int(row["species_id"])
        if label in observed:
            raise ValueError(f"Duplicate species_id in overlap report: {label}")
        total, matches = int(row["test_rows"]), int(row["exact_matches"])
        if not (0 <= matches <= total):
            raise ValueError(f"Invalid overlap counts for label {label}")
        observed[label] = {"test_rows": total, "exact_matches": matches}
    if sum(row["test_rows"] for row in observed.values()) != decision.get("test_rows"):
        raise ValueError("per_species.csv test total does not match decision.json")
    if sum(row["exact_matches"] for row in observed.values()) != decision.get("exact_test_matches_train"):
        raise ValueError("per_species.csv match total does not match decision.json")

    missing = sorted(set(observed) - set(by_label))
    extra = sorted(set(by_label) - set(observed))
    original_mismatch = sorted(
        label for label, row in by_label.items()
        if label in original_by_id and str(row["species"]).strip() != original_by_id[label]
    )
    original_id_missing = sorted(set(original_by_id) - set(by_label))
    original_id_extra = sorted(set(by_label) - set(original_by_id)) if original else []
    source_counts: Counter[str] = Counter()
    status_counts: Counter[str] = Counter()
    by_rank: dict[str, dict[str, Counter[str]]] = {
        rank: defaultdict(Counter) for rank in ("Genus", "Family", "Order")
    }
    label_rows = []
    for label in sorted(observed):
        mapping = by_label.get(label, {})
        counts = observed[label]
        source = str(mapping.get("taxonomy_source") or "[not recorded]")
        status = str(mapping.get("ictv_status") or "[not recorded]")
        source_counts[source] += 1
        status_counts[status.split(" (Score:")[0]] += 1
        output_row = {
            "model_label": label,
            "mapping_id": label + 1 if mapping else "",
            "original_species": mapping.get("species", ""),
            "validated_species": mapping.get("species_validated_name", ""),
            "taxonomy_source": source,
            "ictv_status": status,
            **{rank.lower(): mapping.get(rank, "") for rank in by_rank},
            **counts,
        }
        label_rows.append(output_row)
        for rank, groups in by_rank.items():
            name = str(mapping.get(rank) or "[missing]")
            groups[name]["test_rows"] += counts["test_rows"]
            groups[name]["exact_matches"] += counts["exact_matches"]
            groups[name]["species_labels"] += 1

    normalized = bool(decision.get("sequence_normalization"))
    integrity_ok = not (missing or duplicated or original_mismatch or original_id_missing or original_id_extra)
    report: dict[str, Any] = {
        "status": "TAXONOMY_ONLY_PROVENANCE_REQUIRED",
        "taxonomy_rows": len(taxonomy),
        "unique_taxonomy_labels": len(by_label),
        "observed_test_species_labels": len(observed),
        "mapping_integrity_ok": integrity_ok,
        "missing_test_labels": missing,
        "duplicate_taxonomy_labels": sorted(duplicated),
        "taxonomy_labels_without_test_rows": extra,
        "original_label_name_mismatch": original_mismatch,
        "original_mapping_ids_missing_in_taxonomy": original_id_missing,
        "taxonomy_ids_absent_from_original_mapping": original_id_extra,
        "taxonomy_source_counts": dict(source_counts),
        "ictv_status_counts": dict(status_counts),
        "overlap_report_normalized_for_padding": normalized,
        "overlap_rate_interpretation": "corrected_full_test" if normalized else "legacy_raw_hash_lower_bound_rerun_recommended",
        "exact_test_matches_train": decision["exact_test_matches_train"],
        "test_rows": decision["test_rows"],
        "exact_overlap_pct": decision["exact_overlap_pct"],
        "genome_provenance_resolved": False,
        "interpretation": "The mapping validates taxonomic identity per label; it does not associate a fragment with its source accession or prove a genome-disjoint split.",
    }
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "taxonomy_audit.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    with (output / "per_taxon.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(label_rows[0]))
        writer.writeheader()
        writer.writerows(label_rows)
    for rank, groups in by_rank.items():
        with (output / f"per_{rank.lower()}.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=[rank, "species_labels", "test_rows", "exact_matches", "exact_overlap_pct"])
            writer.writeheader()
            for name, counts in sorted(groups.items()):
                writer.writerow({rank: name, **counts,
                                 "exact_overlap_pct": round(100 * counts["exact_matches"] / counts["test_rows"], 4)})
    return report
