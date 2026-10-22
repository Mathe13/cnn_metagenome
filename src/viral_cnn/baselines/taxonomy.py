"""Taxonomy conversion and raw output parsers for Kraken2 and BLASTN baselines."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Union


def parse_kraken_output(
    source: Union[str, Path, Iterable[str]],
    valid_query_ids: Optional[Iterable[str]] = None,
    num_classes: int = 13683,
) -> Dict[str, int]:
    """Parse Kraken2 standard report output into integer species predictions.

    Maps:
      - 'C' (classified) with 2 <= taxid < num_classes + 2 -> label = taxid - 2
      - 'U' (unclassified) or taxid == 1 (root) -> label = -1 (abstention)
      - out-of-range taxid -> label = -1

    Raises:
      ValueError: on duplicate query IDs or if queries do not match valid_query_ids.
    """
    valid_set: Optional[Set[str]] = set(valid_query_ids) if valid_query_ids is not None else None
    predictions: Dict[str, int] = {}

    lines: Iterable[str]
    if isinstance(source, (str, Path)):
        p = Path(source)
        if not p.exists():
            raise FileNotFoundError(f"Kraken output file not found: {p}")
        with p.open("r", encoding="utf-8") as f:
            lines = f.readlines()
    else:
        lines = source

    for line in lines:
        line_str = line.strip()
        if not line_str:
            continue
        parts = line_str.split("\t")
        if len(parts) < 3:
            continue
        status, qid, taxid_str = parts[0], parts[1], parts[2]

        if qid in predictions:
            raise ValueError(f"Duplicate query ID in Kraken output: {qid}")
        if valid_set is not None and qid not in valid_set:
            raise ValueError(f"Unexpected query ID in Kraken output: {qid}")

        try:
            tid = int(taxid_str)
        except ValueError:
            tid = 1

        if status == "C" and 2 <= tid < num_classes + 2:
            label = tid - 2
        else:
            label = -1

        predictions[qid] = label

    if valid_set is not None:
        missing = valid_set - set(predictions.keys())
        if missing:
            raise ValueError(f"Kraken output is missing {len(missing)} expected queries")

    return predictions


def parse_blast_output(
    source: Union[str, Path, Iterable[str]],
    valid_query_ids: Optional[Iterable[str]] = None,
    max_targets: int = 5,
) -> Dict[str, List[int]]:
    """Parse BLASTN tabular output into species candidate rankings.

    Expected format: -outfmt "6 qseqid sseqid bitscore evalue pident length"
    where subject IDs end with `_label_{int}`.

    Candidate selection per query:
      - Keeps maximum bitscore per species label.
      - Resolves ties by lowest E-value, then lowest label index.
      - Truncates to top `max_targets` candidate labels.
      - For unaligned queries in valid_query_ids, returns an empty list [].

    Raises:
      ValueError: on subject ID lacking `_label_{int}` or unrecognized query ID.
    """
    valid_set: Optional[Set[str]] = set(valid_query_ids) if valid_query_ids is not None else None
    top_candidates: Dict[str, Dict[int, tuple[float, float]]] = {}

    lines: Iterable[str]
    if isinstance(source, (str, Path)):
        p = Path(source)
        if not p.exists():
            raise FileNotFoundError(f"BLAST output file not found: {p}")
        with p.open("r", encoding="utf-8") as f:
            lines = f.readlines()
    else:
        lines = source

    label_regex = re.compile(r"_label_(\d+)$")

    for line in lines:
        line_str = line.strip()
        if not line_str:
            continue
        parts = line_str.split("\t")
        if len(parts) < 4:
            continue
        qid, sid, score_str, ev_str = parts[0], parts[1], parts[2], parts[3]

        if valid_set is not None and qid not in valid_set:
            raise ValueError(f"Unknown query ID in BLAST output: {qid}")

        match = label_regex.search(sid)
        if not match:
            raise ValueError(f"Subject ID lacks '_label_{{int}}' suffix: {sid}")

        label = int(match.group(1))
        bitscore = float(score_str)
        evalue = float(ev_str)

        candidates = top_candidates.setdefault(qid, {})
        current = candidates.get(label)
        val = (bitscore, -evalue)
        if current is None or val > current:
            candidates[label] = val

    results: Dict[str, List[int]] = {}
    
    # Process queries with hits
    for qid, candidates in top_candidates.items():
        # Sort key: (-bitscore, evalue, label)
        sorted_labels = sorted(
            candidates.keys(),
            key=lambda lbl: (-candidates[lbl][0], -candidates[lbl][1], lbl),
        )
        results[qid] = sorted_labels[:max_targets]

    # For queries in valid_set with no hits, assign empty list []
    if valid_set is not None:
        for qid in valid_set:
            if qid not in results:
                results[qid] = []

    return results
