"""Dataset overlap scientific contract, audit artifact emission, and terminology auditor (S13)."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterable

from viral_cnn.baselines.overlap import canonical_sequence_hash
from viral_cnn.utils.logging_utils import log_artifact_path

# Negative lookbehind to allow negations / contrasts: "rather than unseen-genome generalization" or "not generalization to unseen"
FORBIDDEN_PATTERNS = [
    # False claims of sequence-level disjointness
    (
        re.compile(r"fragment sequences are disjunct", re.IGNORECASE),
        "False claim: fragment sequences are not disjunct (~95.91% exact or RC overlap)",
    ),
    (
        re.compile(r"no identical (?:512-bp )?sequence fragment appears across partitions", re.IGNORECASE),
        "False claim: identical or RC sequence fragments appear across partitions",
    ),
    (
        re.compile(r"(?<!lack of )(?<!no )sequence-level disjointness is guaranteed", re.IGNORECASE),
        "False claim: sequence-level disjointness is not guaranteed",
    ),
    # Unqualified assertions of unseen-genome generalization as an established property of the fragment split
    (
        re.compile(r"(?<!rather than )(?<!not )(?<!distinct from )(?<!without establishing )unseen-genome generalization", re.IGNORECASE),
        "Unqualified claim of unseen-genome generalization",
    ),
    (
        re.compile(r"(?<!rather than )(?<!not )(?<!does not reflect )generalization to (?:unseen|novel) (?:viral )?genomes", re.IGNORECASE),
        "Unqualified claim of generalization to unseen/novel genomes",
    ),
]

REQUIRED_TERMS = [
    (
        "fragment-level closed-set classification",
        re.compile(r"fragment-level closed-set classification", re.IGNORECASE),
    ),
    (
        "known-reference distribution",
        re.compile(r"known-reference distribution", re.IGNORECASE),
    ),
    (
        "no-exact-overlap subset",
        re.compile(r"no-exact-overlap subset", re.IGNORECASE),
    ),
]


def build_dataset_overlap_audit(
    train_sequences: Iterable[str],
    test_queries: Iterable[str],
    *,
    canonicalize_rc: bool = True,
    row_disjoint: bool = True,
    genome_disjoint: bool = False,
    split_unit: str = "fragment",
) -> dict[str, Any]:
    """Audit exact and reverse-complement sequence overlap between train and test splits.

    Returns the canonical S13 machine-readable schema:
    {
      "split_unit": "fragment",
      "row_disjoint": true,
      "genome_disjoint": false,
      "sequence_identity_audit": {
        "canonicalizes_reverse_complement": true,
        "n_queries": int,
        "n_exact_or_rc_overlap": int,
        "overlap_fraction": float
      }
    }
    """
    train_hashes: set[str] = set()
    for seq in train_sequences:
        clean = seq.strip().upper()
        if canonicalize_rc:
            h = canonical_sequence_hash(clean)
        else:
            h = hashlib.sha256(clean.encode("ascii")).hexdigest()
        train_hashes.add(h)

    n_queries = 0
    n_overlap = 0

    for qseq in test_queries:
        clean = qseq.strip().upper()
        if canonicalize_rc:
            qh = canonical_sequence_hash(clean)
        else:
            qh = hashlib.sha256(clean.encode("ascii")).hexdigest()
        n_queries += 1
        if qh in train_hashes:
            n_overlap += 1

    overlap_fraction = float(n_overlap / n_queries) if n_queries > 0 else 0.0

    return {
        "split_unit": str(split_unit),
        "row_disjoint": bool(row_disjoint),
        "genome_disjoint": bool(genome_disjoint),
        "sequence_identity_audit": {
            "canonicalizes_reverse_complement": bool(canonicalize_rc),
            "n_queries": int(n_queries),
            "n_exact_or_rc_overlap": int(n_overlap),
            "overlap_fraction": round(overlap_fraction, 4),
        },
    }


def write_dataset_overlap_audit_artifact(audit_data: dict[str, Any], output_path: Path) -> Path:
    """Serialize the dataset overlap audit dictionary to formatted JSON."""
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(audit_data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    log_artifact_path("Dataset overlap audit artifact", output_path)
    return output_path


def audit_manuscript_terminology(text: str) -> dict[str, Any]:
    """Audit manuscript text for compliance with S13 scientific contracts.

    Checks:
    1. Forbidden claims (sequence disjointness, unqualified unseen-genome generalization).
    2. Required terms (fragment-level closed-set classification, known-reference distribution, no-exact-overlap subset).

    Returns a report dictionary:
    {
      "passed": bool,
      "violations": list[dict[str, str]],
      "missing_required": list[str],
      "found_required": list[str],
    }
    """
    violations = []
    for pattern, description in FORBIDDEN_PATTERNS:
        matches = pattern.findall(text)
        if matches:
            for match in matches:
                violations.append({
                    "pattern": pattern.pattern,
                    "matched": match,
                    "description": description,
                })

    missing_required = []
    found_required = []
    for term_name, pattern in REQUIRED_TERMS:
        if pattern.search(text):
            found_required.append(term_name)
        else:
            missing_required.append(term_name)

    passed = (len(violations) == 0) and (len(missing_required) == 0)

    return {
        "passed": passed,
        "violations": violations,
        "missing_required": missing_required,
        "found_required": found_required,
    }
