# S13 Dataset Overlap Contract checks

Profile: light
Plan: `.specs/features/s13-dataset-overlap-contract/plan.md`

## Intent

Historical drafts of the manuscript asserted or implied sequence-level disjointness between partitions despite empirical audits showing ~95.91% exact or reverse-complement overlap. This feature formally enforces the scientific contract separating row-disjoint partitioning from sequence redundancy, eliminates inaccurate sequence-disjointness assertions from the manuscript, implements an automated manuscript terminology auditor, and generates canonical machine-readable overlap audit artifacts.

8 checks in 1 slice · 4 one-way doors · 0 open, of which 0 block

## Checks

### S1 - Dataset overlap scientific contract and manuscript reconciliation · 3 files · ~35 KB · ~9k

**C1** - `build_dataset_overlap_audit` returns dictionary containing `split_unit="fragment"`, `row_disjoint=True`, `genome_disjoint=False`, and `sequence_identity_audit` (AC 1)
Proof: `REMOTE_SERVER=False PYTHONPATH=src .venv/bin/python3 -m unittest tests.test_s13_dataset_overlap_contract.TestS13DatasetOverlapContract.test_build_dataset_overlap_audit_schema -k test_build_dataset_overlap_audit_schema`

**C2** - `sequence_identity_audit` canonicalizes reverse complement matching forward and RC equally (AC 2)
Proof: `REMOTE_SERVER=False PYTHONPATH=src .venv/bin/python3 -m unittest tests.test_s13_dataset_overlap_contract.TestS13DatasetOverlapContract.test_canonical_reverse_complement_matching -k test_canonical_reverse_complement_matching`

**C3** - `write_dataset_overlap_audit_artifact` persists audit dictionary into valid JSON matching canonical schema (AC 3)
Proof: `REMOTE_SERVER=False PYTHONPATH=src .venv/bin/python3 -m unittest tests.test_s13_dataset_overlap_contract.TestS13DatasetOverlapContract.test_write_dataset_overlap_audit_artifact -k test_write_dataset_overlap_audit_artifact`

**C4** - `audit_manuscript_terminology` reports violations when text claims fragment sequences are disjunct or no identical fragments appear (AC 4)
Proof: `REMOTE_SERVER=False PYTHONPATH=src .venv/bin/python3 -m unittest tests.test_s13_dataset_overlap_contract.TestS13DatasetOverlapContract.test_audit_manuscript_detects_sequence_disjointness_claims -k test_audit_manuscript_detects_sequence_disjointness_claims`

**C5** - `audit_manuscript_terminology` reports violations when text asserts unqualified unseen-genome generalization (AC 5)
Proof: `REMOTE_SERVER=False PYTHONPATH=src .venv/bin/python3 -m unittest tests.test_s13_dataset_overlap_contract.TestS13DatasetOverlapContract.test_audit_manuscript_detects_unqualified_unseen_genome_claims -k test_audit_manuscript_detects_unqualified_unseen_genome_claims`

**C6** - `generate_manuscript_content()` produces LaTeX content passing `audit_manuscript_terminology` with zero violations in review and submission modes (AC 6)
Proof: `REMOTE_SERVER=False PYTHONPATH=src .venv/bin/python3 -m unittest tests.test_s13_dataset_overlap_contract.TestS13DatasetOverlapContract.test_reconciled_manuscript_passes_terminology_audit -k test_reconciled_manuscript_passes_terminology_audit`

**C7** - `generate_manuscript_content()` contains required terms: `fragment-level closed-set classification`, `known-reference distribution`, `no-exact-overlap subset` (AC 7)
Proof: `REMOTE_SERVER=False PYTHONPATH=src .venv/bin/python3 -m unittest tests.test_s13_dataset_overlap_contract.TestS13DatasetOverlapContract.test_reconciled_manuscript_contains_required_terminology -k test_reconciled_manuscript_contains_required_terminology`

**C8** - `compile_manuscript` compiles the reconciled `manuscript_submission.tex` cleanly with exit code 0 (AC 8)
Proof: `REMOTE_SERVER=False PYTHONPATH=src .venv/bin/python3 -m unittest tests.test_s13_dataset_overlap_contract.TestS13DatasetOverlapContract.test_reconciled_manuscript_compiles_cleanly -k test_reconciled_manuscript_compiles_cleanly`

## Coverage

| Set (size) | Member -> proof | Unproven |
| --- | --- | --- |
| audit schema invariants (2) | `schema_fields` C1 · `json_serialization` C3 | - |
| sequence canonicalization (1) | `forward_and_rc` C2 | - |
| forbidden claim detection (2) | `sequence_disjointness` C4 · `unseen_generalization` C5 | - |
| manuscript compliance (3) | `zero_violations` C6 · `required_terms` C7 · `pdf_compilation` C8 | - |

- No check claims more than the single case its proof exercises

## Swept

- validation : C1, C2, C4, C5
- failure modes : C4, C5
- idempotency : C3, C6
- authorization : n/a - internal module logic without user authentication
- concurrency : n/a - stateless data transformations
- data lifecycle : n/a - immutable JSON artifact emission
- dependency failure : n/a - pure local filesystem checks
- state transitions : n/a - deterministic terminology checking
- observability : C1, C3, C6

## Handoff

Files touched:
- `src/viral_cnn/data/overlap_contract.py` (8 KB)
- `src/viral_cnn/reporting/manuscript.py` (22 KB)
- `tests/test_s13_dataset_overlap_contract.py` (10 KB)

Total estimate: ~40 KB · ~10k tokens.
Budget: 150k tokens. Estimate is well within budget: one builder.
