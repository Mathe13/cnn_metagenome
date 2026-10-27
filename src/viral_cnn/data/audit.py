"""
Dataset audit module: strand-invariant canonical hashing, duplicate quantification,
and cross-partition split leakage analysis.
"""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path
from typing import Any, Iterator

# Translation table for nucleotide reverse complement
_RC_TRANS = str.maketrans("ACGTNacgtn", "TGCANtgcan")


def reverse_complement(seq: str) -> str:
    """Computes the reverse complement of a DNA sequence."""
    return seq.translate(_RC_TRANS)[::-1]


def canonical_sequence_hash(seq: str) -> str:
    """
    Computes a strand-invariant SHA-256 digest of a nucleotide sequence:
      SHA-256(min(seq.upper(), reverse_complement(seq.upper())))
    This guarantees that a sequence and its reverse complement yield identical hashes.
    """
    seq_upper = seq.upper()
    rc_upper = reverse_complement(seq_upper)
    canonical = min(seq_upper, rc_upper)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def iter_sequence_records(
    file_path: str | Path | list[str] | list[Path],
    seq_col: str = "sequence",
    label_col: str = "label",
    sample_size: int | None = None,
) -> Iterator[tuple[str, int]]:
    """
    Iterates over (sequence, label) tuples from a CSV or Parquet file (or directory of Parquet partitions).
    Streams rows efficiently to avoid loading entire multi-gigabyte files into memory.
    """
    if isinstance(file_path, (list, tuple)):
        files = [Path(p) for p in file_path]
    else:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        if path.is_dir():
            if (path / "test").is_dir():
                files = sorted((path / "test").glob("*.parquet"))
            elif (path / "test.parquet").exists():
                files = [path / "test.parquet"]
            else:
                files = sorted(path.glob("*.parquet")) or sorted(path.rglob("*.parquet"))
            if not files:
                raise FileNotFoundError(f"No Parquet files found in directory: {path}")
        else:
            files = [path]

    yielded = 0
    for path in files:
        if path.suffix.lower() == ".csv":
            with open(path, mode="r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                if reader.fieldnames is None or seq_col not in reader.fieldnames:
                    raise ValueError(
                        f"Column '{seq_col}' not found in CSV header: {reader.fieldnames}"
                    )
                has_label = label_col in reader.fieldnames

                for row in reader:
                    seq = row[seq_col].strip()
                    label = int(row[label_col]) if has_label and row[label_col] else -1
                    yield seq, label
                    yielded += 1
                    if sample_size is not None and yielded >= sample_size:
                        return
        elif path.suffix.lower() in (".parquet", ".pq"):
            import pyarrow.parquet as pq

            pf = pq.ParquetFile(path)
            schema_names = pf.schema_arrow.names
            actual_seq_col = seq_col
            if actual_seq_col not in schema_names and seq_col in ("sequence", "sequence_padded", "padded_Representation"):
                for cand in ("sequence_padded", "padded_Representation", "sequence"):
                    if cand in schema_names:
                        actual_seq_col = cand
                        break
            if actual_seq_col not in schema_names:
                raise ValueError(f"Column '{seq_col}' not found in Parquet schema: {schema_names}")
            cols = [actual_seq_col]
            if label_col in schema_names:
                cols.append(label_col)

            for batch in pf.iter_batches(batch_size=8192, columns=cols):
                batch_dict = batch.to_pydict()
                seqs = batch_dict[actual_seq_col]
                labels = batch_dict.get(label_col, [-1] * len(seqs))
                for s, l in zip(seqs, labels):
                    yield s, int(l) if l is not None else -1
                    yielded += 1
                    if sample_size is not None and yielded >= sample_size:
                        return
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}. Expected .csv or .parquet")


def audit_partition_duplicates(
    file_path: str | Path,
    seq_col: str = "sequence",
    label_col: str = "label",
    sample_size: int | None = None,
) -> dict[str, Any]:
    """
    Audits duplicate sequences within a single dataset partition using canonical hashing.
    Returns metrics on total count, unique canonical sequences, and exact duplication percentage.
    """
    total_sequences = 0
    seen_hashes: set[str] = set()
    class_counts: dict[int, int] = {}

    for seq, label in iter_sequence_records(
        file_path, seq_col=seq_col, label_col=label_col, sample_size=sample_size
    ):
        total_sequences += 1
        chash = canonical_sequence_hash(seq)
        seen_hashes.add(chash)
        if label >= 0:
            class_counts[label] = class_counts.get(label, 0) + 1

    unique_canonical = len(seen_hashes)
    duplicate_count = total_sequences - unique_canonical
    duplication_rate_pct = (
        (duplicate_count / total_sequences * 100.0) if total_sequences > 0 else 0.0
    )
    single_sequence_classes = sum(1 for count in class_counts.values() if count == 1)

    return {
        "file_path": str(file_path),
        "total_sequences": total_sequences,
        "unique_canonical_sequences": unique_canonical,
        "duplicate_sequences": duplicate_count,
        "duplication_rate_pct": round(duplication_rate_pct, 4),
        "classes_count": len(class_counts),
        "single_sequence_classes": single_sequence_classes,
    }


def audit_cross_partition_overlap(
    train_path: str | Path,
    test_path: str | Path,
    seq_col: str = "sequence",
    label_col: str = "label",
    sample_size: int | None = None,
) -> dict[str, Any]:
    """
    Audits sequence overlap and data leakage between train and test partitions.
    Uses strand-invariant canonical hashing to detect identical and reverse-complement matches.
    """
    # 1. Collect reference train hashes
    train_hashes: set[str] = set()
    train_classes: set[int] = set()
    total_train = 0

    for seq, label in iter_sequence_records(
        train_path, seq_col=seq_col, label_col=label_col, sample_size=sample_size
    ):
        total_train += 1
        train_hashes.add(canonical_sequence_hash(seq))
        if label >= 0:
            train_classes.add(label)

    # 2. Audit test queries against train hashes
    total_test = 0
    test_matches = 0
    test_hashes: set[str] = set()
    test_classes: set[int] = set()
    classes_with_overlap: set[int] = set()
    classes_without_overlap: set[int] = set()

    for seq, label in iter_sequence_records(
        test_path, seq_col=seq_col, label_col=label_col, sample_size=sample_size
    ):
        total_test += 1
        chash = canonical_sequence_hash(seq)
        test_hashes.add(chash)
        if label >= 0:
            test_classes.add(label)

        if chash in train_hashes:
            test_matches += 1
            if label >= 0:
                classes_with_overlap.add(label)
        else:
            if label >= 0:
                classes_without_overlap.add(label)

    overlap_pct = (test_matches / total_test * 100.0) if total_test > 0 else 0.0
    disjoint_queries = total_test - test_matches
    single_source_classes = len(test_classes - train_classes)

    return {
        "train_file": str(train_path),
        "test_file": str(test_path),
        "total_train_sequences": total_train,
        "unique_train_canonical": len(train_hashes),
        "total_test_sequences": total_test,
        "unique_test_canonical": len(test_hashes),
        "train_matches": test_matches,
        "disjoint_queries": disjoint_queries,
        "overlap_pct": round(overlap_pct, 4),
        "test_classes_count": len(test_classes),
        "classes_with_overlap_count": len(classes_with_overlap),
        "classes_without_overlap_count": len(classes_without_overlap),
        "single_source_classes": single_source_classes,
    }
