"""
Genome-disjoint grouped splitting engine and taxonomic coverage analysis.
Enforces strict zero genome accession overlap between train, val, and test partitions.
"""

from __future__ import annotations

import csv
import random
from collections import defaultdict
from pathlib import Path
from typing import Any


def load_records_from_file(
    file_path: str | Path,
    required_cols: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Loads records from a CSV or Parquet file."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    records: list[dict[str, Any]] = []

    if path.suffix.lower() == ".csv":
        with open(path, mode="r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                raise ValueError(f"Empty or headerless CSV: {path}")
            if required_cols:
                missing = [col for col in required_cols if col not in reader.fieldnames]
                if missing:
                    raise ValueError(f"Missing required columns in CSV: {missing}")
            for row in reader:
                records.append(dict(row))
    elif path.suffix.lower() in (".parquet", ".pq"):
        import pyarrow.parquet as pq

        table = pq.read_table(path, columns=required_cols)
        pydict = table.to_pydict()
        num_rows = len(table)
        cols = list(pydict.keys())
        for i in range(num_rows):
            records.append({col: pydict[col][i] for col in cols})
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}. Expected .csv or .parquet")

    return records


def build_genome_disjoint_split(
    records: list[dict[str, Any]],
    group_col: str = "genome_id",
    class_col: str = "species_id",
    test_size: float = 0.2,
    val_size: float = 0.0,
    min_genomes_per_class: int = 2,
    single_genome_policy: str = "train_only",
    random_state: int = 42,
) -> dict[str, Any]:
    """
    Constructs a grouped split where all sequence fragments from a given genome accession
    belong exclusively to a single partition (train, val, or test).

    Guarantee:
      set(train_genomes) ∩ set(test_genomes) == ∅
      set(train_genomes) ∩ set(val_genomes) == ∅
      set(val_genomes) ∩ set(test_genomes) == ∅
    """
    if not records:
        raise ValueError("Cannot split empty record list")
    if not (0.0 < test_size < 1.0):
        raise ValueError(f"test_size must be between 0.0 and 1.0, got {test_size}")
    if not (0.0 <= val_size < 1.0):
        raise ValueError(f"val_size must be between 0.0 and 1.0, got {val_size}")
    if test_size + val_size >= 1.0:
        raise ValueError(f"test_size + val_size must be < 1.0, got {test_size + val_size}")
    if single_genome_policy not in ("train_only", "exclude"):
        raise ValueError(
            f"Unsupported single_genome_policy: {single_genome_policy}. Expected 'train_only' or 'exclude'"
        )

    # 1. Group records by class (species) and by genome
    # class_id -> {genome_id -> list of records}
    class_genome_records: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )

    for r in records:
        if group_col not in r:
            raise ValueError(f"Record missing group_col '{group_col}': {r}")
        if class_col not in r:
            raise ValueError(f"Record missing class_col '{class_col}': {r}")
        cls = str(r[class_col])
        gid = str(r[group_col])
        class_genome_records[cls][gid].append(r)

    train_records: list[dict[str, Any]] = []
    val_records: list[dict[str, Any]] = []
    test_records: list[dict[str, Any]] = []

    train_genomes: set[str] = set()
    val_genomes: set[str] = set()
    test_genomes: set[str] = set()

    eligible_species: set[str] = set()
    infeasible_species: set[str] = set()

    rng = random.Random(random_state)

    # Sort classes for determinism
    sorted_classes = sorted(class_genome_records.keys())

    for cls in sorted_classes:
        genome_dict = class_genome_records[cls]
        distinct_genomes = sorted(genome_dict.keys())
        k_genomes = len(distinct_genomes)

        if k_genomes < min_genomes_per_class:
            # Single-genome or under-supported class
            infeasible_species.add(cls)
            if single_genome_policy == "train_only":
                for gid in distinct_genomes:
                    train_records.extend(genome_dict[gid])
                    train_genomes.add(gid)
            # If 'exclude', records are omitted
        else:
            # Multi-genome eligible class
            eligible_species.add(cls)
            # Shuffle genomes deterministically per class
            class_seed = rng.randint(0, 2**31 - 1)
            class_rng = random.Random(class_seed)
            shuffled_genomes = distinct_genomes.copy()
            class_rng.shuffle(shuffled_genomes)

            # Ensure at least 1 genome in train and at least 1 in test
            num_test = max(1, int(round(k_genomes * test_size)))
            num_val = int(round(k_genomes * val_size)) if val_size > 0.0 else 0

            # Guard so train gets at least 1 genome
            if num_test + num_val >= k_genomes:
                num_test = 1
                num_val = 1 if (val_size > 0.0 and k_genomes > 2) else 0

            g_test = shuffled_genomes[:num_test]
            g_val = shuffled_genomes[num_test : num_test + num_val]
            g_train = shuffled_genomes[num_test + num_val :]

            if not g_train:
                # Fallback: guarantee at least 1 in train
                g_train = [g_test.pop()]

            for gid in g_train:
                train_records.extend(genome_dict[gid])
                train_genomes.add(gid)
            for gid in g_val:
                val_records.extend(genome_dict[gid])
                val_genomes.add(gid)
            for gid in g_test:
                test_records.extend(genome_dict[gid])
                test_genomes.add(gid)

    # 2. Strict Invariant Asserts: zero genome overlap
    assert train_genomes.isdisjoint(test_genomes), "Fatal: train and test share genome accessions"
    assert train_genomes.isdisjoint(val_genomes), "Fatal: train and val share genome accessions"
    assert val_genomes.isdisjoint(test_genomes), "Fatal: val and test share genome accessions"

    total_species = len(class_genome_records)
    taxonomic_coverage_pct = (
        (len(eligible_species) / total_species * 100.0) if total_species > 0 else 0.0
    )

    metrics = {
        "total_species": total_species,
        "eligible_species_count": len(eligible_species),
        "infeasible_species_count": len(infeasible_species),
        "taxonomic_coverage_pct": round(taxonomic_coverage_pct, 4),
        "single_genome_policy": single_genome_policy,
        "total_fragments": len(train_records) + len(val_records) + len(test_records),
        "train_fragments_count": len(train_records),
        "val_fragments_count": len(val_records),
        "test_fragments_count": len(test_records),
        "train_genomes_count": len(train_genomes),
        "val_genomes_count": len(val_genomes),
        "test_genomes_count": len(test_genomes),
        "genome_overlap_count": len(train_genomes.intersection(test_genomes)),
    }

    return {
        "partitions": {
            "train": train_records,
            "val": val_records,
            "test": test_records,
        },
        "metrics": metrics,
    }


def save_partition_manifests(
    partitions: dict[str, list[dict[str, Any]]],
    output_dir: str | Path,
) -> dict[str, str]:
    """Saves train, val, and test partition manifests to CSV files."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}

    for split_name, recs in partitions.items():
        split_path = out_dir / f"{split_name}_manifest.csv"
        if recs:
            fieldnames = list(recs[0].keys())
            with open(split_path, mode="w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(recs)
        else:
            with open(split_path, mode="w", encoding="utf-8", newline="") as f:
                f.write("fragment_id,sequence,species_id,genome_id\n")
        paths[split_name] = str(split_path)

    return paths
