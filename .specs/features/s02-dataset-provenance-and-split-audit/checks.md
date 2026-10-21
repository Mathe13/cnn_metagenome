# S02: Dataset Provenance and Fragment Split Audit - Checks

Profile: light
Plan: `.specs/features/s02-dataset-provenance-and-split-audit/plan.md`

## Checks

### S1: Dataset Provenance and Reference Manifest · 1 file · 12 KB · ~3k

**C1** - `docs/dataset_manifest.md` documents dataset provenance distinguishing observed fields from inferred/reconstructed fields across Kraken2, NCBI, and ICTV (AC 1)
Proof: `.venv/bin/python3 -m unittest discover -s tests -k test_manifest_observed_vs_inferred_fields`

**C2** - `docs/dataset_manifest.md` records 80/10/10 stratified fragment split parameters (`seed=42`) and the 95.91% (262,470 / 273,660) baseline exact overlap finding (AC 2)
Proof: `.venv/bin/python3 -m unittest discover -s tests -k test_manifest_split_parameters_and_baseline_overlap`

**C3** - `docs/dataset_manifest.md` documents non-zero sequence overlap risks from tiled strides and high-similarity sub-sequences when identical matches are absent (AC 3)
Proof: `.venv/bin/python3 -m unittest discover -s tests -k test_manifest_subsequence_overlap_caveat`

### S2: Canonical Sequence Hashing and Duplicate Audit Engine · 2 files · 20 KB · ~5k

**C4** - `viral_cnn.data.audit.canonical_sequence_hash` computes strand-invariant SHA-256 digest on `min(seq, reverse_complement(seq))` across A, C, G, T, N (AC 4, AC 6)
Proof: `.venv/bin/python3 -m unittest discover -s tests -k test_canonical_sequence_hash_strand_invariance`

**C5** - `viral_cnn.data.audit.audit_partition_duplicates` measures total sequences, unique canonical sequences, and exact duplication percentage (AC 5)
Proof: `.venv/bin/python3 -m unittest discover -s tests -k test_audit_partition_duplicates_metrics`

**C6** - `viral_cnn.data.audit.audit_cross_partition_overlap` identifies forward and reverse complement duplicates across split partitions (AC 7, AC 10)
Proof: `.venv/bin/python3 -m unittest discover -s tests -k test_cross_partition_overlap_detection`

### S3: Split Overlap and Leakage CLI · 3 files · 18 KB · ~5k

**C7** - CLI `scripts/audit_dataset.py` executes partition audit and emits structured JSON report with total, unique, and overlap counts with exit code 0 (AC 8, AC 10)
Proof: `.venv/bin/python3 -m unittest discover -s tests -k test_audit_cli_exit_zero_and_json_payload`

**C8** - CLI `scripts/audit_dataset.py` validates arguments and exits with code 2 on missing files or unsupported file formats (AC 9)
Proof: `.venv/bin/python3 -m unittest discover -s tests -k test_audit_cli_invalid_arguments_exit_code_2`

**C9** - Audit routines and CLI verify deterministic calculations against synthetic fixtures without scanning multi-gigabyte files (AC 11)
Proof: `.venv/bin/python3 -m unittest discover -s tests -k test_audit_synthetic_fixtures_deterministic_execution`

## Coverage

| Set (size) | Member -> proof | Unproven |
| --- | --- | --- |
| Canonical sequence hash invariants (2) | `forward_identical` C4 · `reverse_complement_identical` C4 | - |
| Partition duplicate metrics reported (3) | `total_sequences` C5 · `unique_canonical_sequences` C5 · `duplication_rate_pct` C5 | - |
| Cross-partition overlap modes detected (2) | `identical_forward_overlap` C6 · `reverse_complement_overlap` C6 | - |
| CLI exit status codes (2) | `0` C7 · `2` C8 | - |
| Dataset reference sources audited (3) | `Kraken2` C1 · `NCBI` C1 · `ICTV` C1 | - |

- Claims naming a status code or response shape: C7, C8 - each has a proof crossing the process boundary
- No other check claims more than the single case its proof exercises

## Swept

- validation: C8
- failure modes: C8
- idempotency: C4
- authorization: n/a - local scientific CLI tool with no network or authorization boundary
- concurrency: n/a - deterministic batch processing on single machine with no concurrent state mutation
- data lifecycle: C1
- dependency failure: C9
- state transitions: n/a - stateless data streaming without persisted database transitions
- observability: C7

## Handoff

- S1 = 3k in docs; S2 = 5k in src/viral_cnn/data/audit.py and tests; S3 = 5k in scripts and fixtures. Total = ~13k tokens, well under the 150k budget - one builder

- **Mechanism:** one builder
