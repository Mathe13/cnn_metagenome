"""Sequence overlap auditing between test queries and training references."""

from __future__ import annotations

import hashlib
from typing import Dict, Iterable, List, Tuple, Union
import numpy as np
import pandas as pd


def canonical_sequence_hash(seq: str) -> str:
    """Compute sha256 hash of uppercase DNA sequence."""
    clean = seq.strip().upper().encode("ascii")
    return hashlib.sha256(clean).hexdigest()


def audit_exact_overlap(
    test_queries: Union[pd.DataFrame, Iterable[Tuple[str, str, int]]],
    train_sequences: Union[Iterable[str], Iterable[Tuple[str, str, int]], Set[str]],
) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
    """Audit exact sequence overlap between test queries and training references.

    Args:
      test_queries: DataFrame with ['query_id', 'sequence', 'label'] or iterable of (qid, seq, label).
      train_sequences: iterable of training sequence strings, tuples, or set of precomputed hashes.

    Returns:
      audit_df: DataFrame with query_id, label, train_count, has_overlap.
      masks: dict with boolean masks for 'all', 'overlap', 'no_exact_overlap'.
    """
    if isinstance(test_queries, pd.DataFrame):
        records = [
            (row["query_id"], row["sequence"], int(row["label"]))
            for _, row in test_queries.iterrows()
        ]
    else:
        records = list(test_queries)

    # Build reference hash counts
    train_counts: Dict[str, int] = {}
    for item in train_sequences:
        if isinstance(item, str):
            h = canonical_sequence_hash(item) if len(item) != 64 else item
        elif isinstance(item, (tuple, list)):
            h = canonical_sequence_hash(item[1])
        else:
            h = str(item)
        train_counts[h] = train_counts.get(h, 0) + 1

    rows = []
    for qid, seq, label in records:
        h = canonical_sequence_hash(seq)
        count = train_counts.get(h, 0)
        rows.append({
            "query_id": qid,
            "label": label,
            "train_count": count,
            "has_overlap": count > 0,
        })

    audit_df = pd.DataFrame(rows)
    has_overlap_arr = audit_df["has_overlap"].to_numpy()

    masks = {
        "all": np.ones(len(audit_df), dtype=bool),
        "overlap": has_overlap_arr,
        "no_exact_overlap": ~has_overlap_arr,
    }

    return audit_df, masks
