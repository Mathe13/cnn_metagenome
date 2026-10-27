"""Read-only, disk-backed audit of whether a genome-disjoint evaluation needs retraining.

An exact sequence match is evidence of contamination, but its absence does not
establish genome independence. Genome identity must be present on the actual
fragment rows; taxonomic names or a separate list of NCBI genomes are not a
substitute for that provenance.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
import sqlite3
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterator

SEQUENCE_COLUMNS = ("sequence_padded", "padded_Representation", "sequence")
SPECIES_COLUMNS = ("species_id", "label")
GENOME_COLUMNS = ("genome_id", "accession", "source_accession")
_BENCHMARK_ALPHABET = re.compile(r"[ACGTN]+\Z")
_IUPAC_ALPHABET = re.compile(r"[ACGTRYSWKMBDHVN]+\Z")
_IUPAC_COMPLEMENT = str.maketrans("ACGTRYSWKMBDHVN", "TGCAYRSWMKVHDBN")


def split_files(path: str | Path) -> list[Path]:
    """Accept an explicit split directory/file, never traverse sibling splits."""
    p = Path(path).expanduser().resolve()
    if p.is_file() and p.suffix.lower() in (".parquet", ".csv"):
        return [p]
    if p.is_dir():
        files = sorted(f for f in p.glob("*") if f.suffix.lower() in (".parquet", ".csv"))
        if files:
            return files
    raise FileNotFoundError(f"Expected split CSV/Parquet file(s) at {p}; no fallback to synthetic data")


def _columns(file: Path) -> list[str]:
    if file.suffix.lower() == ".csv":
        with file.open(newline="", encoding="utf-8") as handle:
            return csv.DictReader(handle).fieldnames or []
    import pyarrow.parquet as pq

    return pq.ParquetFile(file).schema_arrow.names


def _first_available(columns: list[str], options: tuple[str, ...], required: bool) -> str | None:
    chosen = next((option for option in options if option in columns), None)
    if required and chosen is None:
        raise ValueError(f"Missing required column: one of {options}; found {columns}")
    return chosen


def _iter_rows(files: list[Path], columns: tuple[str, str, str | None]) -> Iterator[tuple[str, str, str | None]]:
    seq_col, species_col, genome_col = columns
    names = [seq_col, species_col] + ([genome_col] if genome_col else [])
    for file in files:
        if file.suffix.lower() == ".csv":
            with file.open(newline="", encoding="utf-8") as handle:
                yield from ((row[seq_col], row[species_col], row.get(genome_col) if genome_col else None)
                            for row in csv.DictReader(handle))
        else:
            import pyarrow.parquet as pq

            parquet = pq.ParquetFile(file)
            for batch in parquet.iter_batches(batch_size=8192, columns=names):
                data = batch.to_pydict()
                for i in range(batch.num_rows):
                    yield (data[seq_col][i], data[species_col][i],
                           data[genome_col][i] if genome_col else None)


def _schema(files: list[Path]) -> tuple[str, str, str | None]:
    schemas = [_columns(file) for file in files]
    common = set(schemas[0]).intersection(*(set(s) for s in schemas[1:]))
    return (
        _first_available(list(common), SEQUENCE_COLUMNS, True),
        _first_available(list(common), SPECIES_COLUMNS, True),
        _first_available(list(common), GENOME_COLUMNS, False),
    )


def _normalized(seq: Any, species: Any, genome: Any) -> tuple[str, str, str | None]:
    if not isinstance(seq, str):
        raise ValueError(f"DNA sequence must be a string, got {type(seq).__name__}; inspect the Parquet encoding")
    if species is None or str(species).strip() == "":
        raise ValueError("Missing species label: cannot compute taxonomic coverage")
    # Match short/benchmark_cnns_kraken2_blast_overlap.ipynb:clean_seq:
    # right padding is not biological sequence, internal '_' denotes unknown N.
    clean = seq.strip().upper().rstrip("_").replace("_", "N")
    return clean, str(species).strip(), str(genome).strip() if genome is not None and str(genome).strip() else None


def _sequence_quality(seq: str, counters: Counter[str]) -> bool:
    """Count non-benchmark rows; return false for rows with no informative bases."""
    if not seq:
        counters["empty_after_padding"] += 1
        return False
    has_acgt = any(base in seq for base in "ACGT")
    if not has_acgt:
        counters["no_acgt"] += 1
    if len(seq) > 512:
        counters["over_512_bp"] += 1
    benchmark_alphabet = _BENCHMARK_ALPHABET.fullmatch(seq) is not None
    if not benchmark_alphabet:
        counters["outside_ACGTN"] += 1
        if _IUPAC_ALPHABET.fullmatch(seq) is None:
            counters["outside_IUPAC"] += 1
    if has_acgt and len(seq) <= 512 and benchmark_alphabet:
        counters["benchmark_compatible"] += 1
    return has_acgt


def _canonical_hash(seq: str) -> str:
    """Full IUPAC reverse complement; unknown symbols allow only forward matching."""
    if _IUPAC_ALPHABET.fullmatch(seq):
        seq = min(seq, seq.translate(_IUPAC_COMPLEMENT)[::-1])
    return hashlib.sha256(seq.encode("utf-8")).hexdigest()


def preflight_split(path: str | Path, limit: int = 1000) -> dict[str, Any]:
    """Inspect the first rows without indexing millions of fragments."""
    if limit < 1:
        raise ValueError("Preflight limit must be positive")
    files = split_files(path)
    columns = _schema(files)
    quality: Counter[str] = Counter()
    rows = 0
    for raw in _iter_rows(files, columns):
        if rows >= limit:
            break
        seq, _, _ = _normalized(*raw)
        rows += 1
        _sequence_quality(seq, quality)
    return {"rows_checked": rows, "columns": columns, "sequence_quality": dict(quality),
            "warning": "First rows only; full audit reports all rows and may differ."}


def audit_retraining_need(
    train_path: str | Path,
    test_path: str | Path,
    output_dir: str | Path,
    *,
    max_test_rows: int | None = None,
    work_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Index every train fragment on disk and audit test fragments and source genomes.

    ``max_test_rows`` is a first-rows pilot only: it cannot yield a final decision.
    This function never trains, downloads references, or modifies its inputs.
    """
    if max_test_rows is not None and max_test_rows < 1:
        raise ValueError("max_test_rows must be positive or None for the full test")
    train_files, test_files = split_files(train_path), split_files(test_path)
    if set(train_files).intersection(test_files):
        raise ValueError("Train and test resolve to the same file")
    train_cols, test_cols = _schema(train_files), _schema(test_files)
    if bool(train_cols[2]) != bool(test_cols[2]):
        raise ValueError("Genome provenance is present in only one split; supply matching fragment-level columns")
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    if work_dir is not None:
        work = Path(work_dir).expanduser().resolve()
        work.mkdir(parents=True, exist_ok=True)
    else:
        work = None

    train_species_genomes: dict[str, set[str]] = defaultdict(set)
    test_species_genomes: dict[str, set[str]] = defaultdict(set)
    train_species: set[str] = set()
    test_species: set[str] = set()
    group_labels: dict[str, str] = {}
    inconsistent_groups: set[str] = set()
    per_species: dict[str, Counter[str]] = defaultdict(Counter)
    n_train = n_test = n_match = n_match_compatible = missing_train_groups = missing_test_groups = 0
    unseen_test_rows = 0
    train_quality: Counter[str] = Counter()
    test_quality: Counter[str] = Counter()

    with tempfile.TemporaryDirectory(prefix="viral_cnn_audit_", dir=work) as temp:
        db = sqlite3.connect(str(Path(temp) / "train_hashes.sqlite"))
        try:
            db.execute("PRAGMA journal_mode=OFF")
            db.execute("PRAGMA synchronous=OFF")
            db.execute("CREATE TABLE hashes (hash TEXT PRIMARY KEY) WITHOUT ROWID")
            batch: list[tuple[str]] = []
            for raw in _iter_rows(train_files, train_cols):
                seq, species, genome = _normalized(*raw)
                n_train += 1
                train_species.add(species)
                if _sequence_quality(seq, train_quality):
                    batch.append((_canonical_hash(seq),))
                if train_cols[2]:
                    if genome is None:
                        missing_train_groups += 1
                    else:
                        train_species_genomes[species].add(genome)
                        prior = group_labels.setdefault(genome, species)
                        if prior != species:
                            inconsistent_groups.add(genome)
                if len(batch) >= 5000:
                    db.executemany("INSERT OR IGNORE INTO hashes VALUES (?)", batch)
                    batch.clear()
            if batch:
                db.executemany("INSERT OR IGNORE INTO hashes VALUES (?)", batch)
            db.commit()
            unique_train = db.execute("SELECT count(*) FROM hashes").fetchone()[0]

            test_chunk: list[tuple[str | None, str, bool]] = []

            def _flush_test_chunk(chunk: list[tuple[str | None, str, bool]]) -> None:
                nonlocal n_match, n_match_compatible
                candidate_digests = list({d for d, _, _ in chunk if d is not None})
                matched_digests: set[str] = set()
                for i in range(0, len(candidate_digests), 900):
                    sub = candidate_digests[i : i + 900]
                    placeholders = ",".join("?" for _ in sub)
                    rows = db.execute(
                        f"SELECT hash FROM hashes WHERE hash IN ({placeholders})",
                        sub,
                    ).fetchall()
                    matched_digests.update(r[0] for r in rows)
                for digest, sp, comp in chunk:
                    if digest is not None and digest in matched_digests:
                        n_match += 1
                        per_species[sp]["exact_matches"] += 1
                        if comp:
                            n_match_compatible += 1
                            per_species[sp]["benchmark_compatible_exact_matches"] += 1
                chunk.clear()

            for raw in _iter_rows(test_files, test_cols):
                if max_test_rows is not None and n_test >= max_test_rows:
                    break
                seq, species, genome = _normalized(*raw)
                n_test += 1
                test_species.add(species)
                per_species[species]["test_rows"] += 1
                hashable = _sequence_quality(seq, test_quality)
                compatible = bool(hashable and len(seq) <= 512 and _BENCHMARK_ALPHABET.fullmatch(seq))
                if compatible:
                    per_species[species]["benchmark_compatible_test_rows"] += 1
                digest = _canonical_hash(seq) if hashable else None
                test_chunk.append((digest, species, compatible))
                if len(test_chunk) >= 5000:
                    _flush_test_chunk(test_chunk)
                if test_cols[2]:
                    if genome is None:
                        missing_test_groups += 1
                    else:
                        test_species_genomes[species].add(genome)
                        prior = group_labels.setdefault(genome, species)
                        if prior != species:
                            inconsistent_groups.add(genome)
                        if genome not in train_species_genomes.get(species, set()) and species in train_species:
                            unseen_test_rows += 1
                            per_species[species]["unseen_genome_rows"] += 1
            if test_chunk:
                _flush_test_chunk(test_chunk)
        finally:
            db.close()

    if n_train == 0 or n_test == 0:
        raise ValueError("Train and test must both contain sequence rows")
    known_groups = bool(train_cols[2]) and not missing_train_groups and not missing_test_groups and not inconsistent_groups
    all_species = train_species | test_species
    eligible_species = sum(
        len(train_species_genomes.get(species, set()) | test_species_genomes.get(species, set())) >= 2
        for species in all_species
    ) if known_groups else None
    shared_genomes = sum(
        len(train_species_genomes.get(species, set()) & test_species_genomes.get(species, set()))
        for species in all_species
    ) if known_groups else None
    full_test = max_test_rows is None
    if not full_test:
        status = "INCONCLUSIVE_PARTIAL_TEST"
        recommendation = "Complete the full test audit; a first-rows pilot cannot establish prevalence or retraining necessity."
    elif not known_groups:
        status = "PROVENANCE_REQUIRED"
        recommendation = "Recover trustworthy fragment-level genome IDs before deciding whether an existing checkpoint can be evaluated on unseen genomes."
    elif unseen_test_rows:
        status = "EVALUATE_UNSEEN_GENOMES_FIRST"
        recommendation = "Evaluate the existing checkpoint on these unseen-genome test rows first; a new split and retraining are needed for a different target population."
    elif eligible_species:
        status = "RETRAIN_FOR_GENOME_GENERALIZATION"
        recommendation = "No currently evaluated test genome is unseen in training. Create a genome-disjoint split and retrain to assess unseen-genome generalization."
    else:
        status = "INSUFFICIENT_INDEPENDENT_GENOMES"
        recommendation = "A closed-set unseen-genome benchmark cannot be constructed for these species from the observed source genomes."

    report: dict[str, Any] = {
        "status": status,
        "recommendation": recommendation,
        "target": "unseen genomes of species represented in training",
        "scope": "full_test" if full_test else f"first_{max_test_rows}_test_rows_pilot",
        "train_rows": n_train,
        "unique_train_sequences_strand_invariant": unique_train,
        "test_rows": n_test,
        "exact_test_matches_train": n_match,
        "exact_overlap_pct": round(100 * n_match / n_test, 4),
        "benchmark_compatible_test_rows": test_quality["benchmark_compatible"],
        "benchmark_compatible_exact_matches": n_match_compatible,
        "benchmark_compatible_exact_overlap_pct": (
            round(100 * n_match_compatible / test_quality["benchmark_compatible"], 4)
            if test_quality["benchmark_compatible"] else None
        ),
        "sequence_normalization": "uppercase; remove trailing '_' padding; internal '_' -> N; IUPAC reverse complement when defined; unknown symbols forward-only",
        "train_sequence_quality": dict(train_quality),
        "test_sequence_quality": dict(test_quality),
        "benchmark_incompatible_train_rows": n_train - train_quality["benchmark_compatible"],
        "benchmark_incompatible_test_rows": n_test - test_quality["benchmark_compatible"],
        "test_species": len(test_species),
        "test_species_absent_from_train": len(test_species - train_species),
        "genome_column_train": train_cols[2],
        "genome_column_test": test_cols[2],
        "genome_provenance_complete": known_groups,
        "missing_train_genome_ids": missing_train_groups,
        "missing_test_genome_ids": missing_test_groups,
        "genome_ids_assigned_multiple_species": len(inconsistent_groups),
        "eligible_species_at_least_two_genomes": eligible_species,
        "shared_train_test_genomes": shared_genomes,
        "test_rows_from_unseen_genomes_with_species_in_train": unseen_test_rows if known_groups else None,
        "source_files": {"train": [str(p) for p in train_files], "test": [str(p) for p in test_files]},
        "interpretation": [
            "Exact matches prove exposure to identical or reverse-complement sequences; a zero rate does not exclude shifted or highly similar fragments.",
            "Rows with symbols outside IUPAC are compared in the forward orientation only; their reverse-complement overlap may be underestimated.",
            "Rows without A/C/G/T are counted in the denominator but excluded from the sequence hash index.",
            "The benchmark-compatible subset is reported separately when lengths or alphabet differ from the 512-bp protocol.",
            "Fragment-level performance on reference genomes is distinct from unseen-genome generalization.",
            "Genome-disjoint evaluation also requires checking biological independence and taxonomic coverage of accessions.",
            "This audit does not measure predictive performance and never proves that a new training run will improve accuracy.",
        ],
    }
    with (output / "decision.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
    with (output / "per_species.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["species_id", "test_rows", "exact_matches",
                                                    "benchmark_compatible_test_rows",
                                                    "benchmark_compatible_exact_matches", "unseen_genome_rows"])
        writer.writeheader()
        for species in sorted(per_species):
            counts = per_species[species]
            writer.writerow({"species_id": species, **{key: counts[key] for key in writer.fieldnames[1:]}})
    return report
