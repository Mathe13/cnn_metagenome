# S15 Empirical Artifact Provenance Hardening checks

Profile: light
Plan: `.specs/features/s15-artifact-provenance-hardening/plan.md`

## Intent

In `src/viral_cnn/reporting/publication.py` and `src/viral_cnn/reporting/transformations.py`, empirical artifacts and derived publication manifests track subsets of metadata, but do not uniformly validate or enforce the full 12-attribute provenance contract required by the roadmap. This feature introduces `validate_artifact_provenance` to validate the full 12-attribute provenance contract fail-closed, integrates it into `validate_selection` under `strict_empirical=True`, and hardens `emit_derived_manifest` to propagate full upstream lineage.

9 checks in 1 slice · 4 one-way doors · 0 open, of which 0 block

## Checks

### S1 - Complete artifact provenance validation and lineage propagation · 4 files · ~45 KB · ~11k

**C1** - `validate_artifact_provenance` returns verified provenance dictionary when all 12 attributes are present and parent hashes match (AC 1)
Proof: `REMOTE_SERVER=False PYTHONPATH=src .venv/bin/python3 -m unittest tests.test_s15_artifact_provenance_hardening.TestS15ArtifactProvenanceHardening.test_valid_full_provenance_passes -k test_valid_full_provenance_passes`

**C2** - `validate_artifact_provenance` raises `PublicationValidationError` when `is_smoke_test` is True or missing (AC 2)
Proof: `REMOTE_SERVER=False PYTHONPATH=src .venv/bin/python3 -m unittest tests.test_s15_artifact_provenance_hardening.TestS15ArtifactProvenanceHardening.test_smoke_test_manifest_fails_provenance -k test_smoke_test_manifest_fails_provenance`

**C3** - `validate_artifact_provenance` raises `PublicationValidationError` when `git_commit` is missing or empty (AC 3)
Proof: `REMOTE_SERVER=False PYTHONPATH=src .venv/bin/python3 -m unittest tests.test_s15_artifact_provenance_hardening.TestS15ArtifactProvenanceHardening.test_missing_git_commit_fails_provenance -k test_missing_git_commit_fails_provenance`

**C4** - `validate_artifact_provenance` raises `PublicationValidationError` when timestamp (`timestamp_utc` or `created_utc`) is missing (AC 4)
Proof: `REMOTE_SERVER=False PYTHONPATH=src .venv/bin/python3 -m unittest tests.test_s15_artifact_provenance_hardening.TestS15ArtifactProvenanceHardening.test_missing_timestamp_fails_provenance -k test_missing_timestamp_fails_provenance`

**C5** - `validate_artifact_provenance` raises `PublicationValidationError` when software versions (`versions` or `library_versions`) is missing (AC 5)
Proof: `REMOTE_SERVER=False PYTHONPATH=src .venv/bin/python3 -m unittest tests.test_s15_artifact_provenance_hardening.TestS15ArtifactProvenanceHardening.test_missing_software_versions_fails_provenance -k test_missing_software_versions_fails_provenance`

**C6** - `validate_artifact_provenance` raises `PublicationValidationError` when a parent artifact is missing on disk or has a hash mismatch (AC 6)
Proof: `REMOTE_SERVER=False PYTHONPATH=src .venv/bin/python3 -m unittest tests.test_s15_artifact_provenance_hardening.TestS15ArtifactProvenanceHardening.test_broken_parent_artifact_fails_provenance -k test_broken_parent_artifact_fails_provenance`

**C7** - `validate_artifact_provenance` raises `PublicationValidationError` when any upstream parent artifact in the transitive lineage declares `is_smoke_test=True` (AC 7)
Proof: `REMOTE_SERVER=False PYTHONPATH=src .venv/bin/python3 -m unittest tests.test_s15_artifact_provenance_hardening.TestS15ArtifactProvenanceHardening.test_transitive_smoke_parent_fails_provenance -k test_transitive_smoke_parent_fails_provenance`

**C8** - `emit_derived_manifest` emits a derived manifest with git commit, timestamp, library versions, non-smoke flag, and parent manifest link (AC 8)
Proof: `REMOTE_SERVER=False PYTHONPATH=src .venv/bin/python3 -m unittest tests.test_s15_artifact_provenance_hardening.TestS15ArtifactProvenanceHardening.test_emit_derived_manifest_propagates_full_lineage -k test_emit_derived_manifest_propagates_full_lineage`

**C9** - `validate_selection` under `strict_empirical=True` enforces `validate_artifact_provenance` on all empirical entries and rejects invalid/synthetic provenance (AC 9)
Proof: `REMOTE_SERVER=False PYTHONPATH=src .venv/bin/python3 -m unittest tests.test_s15_artifact_provenance_hardening.TestS15ArtifactProvenanceHardening.test_validate_selection_enforces_artifact_provenance -k test_validate_selection_enforces_artifact_provenance`

## Coverage

| Set (size) | Member -> proof | Unproven |
| --- | --- | --- |
| provenance completeness (2) | `complete_provenance` C1 · `missing_attribute` C3, C4, C5 | - |
| smoke test rejection (2) | `direct_smoke` C2 · `transitive_smoke` C7 | - |
| parent integrity (2) | `missing_parent` C6 · `corrupted_parent_hash` C6 | - |
| lineage propagation (1) | `derived_manifest` C8 | - |
| publication selection integration (2) | `valid_selection` C9 · `invalid_selection_rejected` C9 | - |

- No check claims more than the single case its proof exercises

## Swept

- validation : C1, C2, C3, C4, C5, C6, C7
- failure modes : C2, C3, C4, C5, C6, C7, C9
- idempotency : C1, C8
- authorization : n/a - internal module logic without user authentication
- concurrency : n/a - stateless script invocations without shared mutable state
- data lifecycle : n/a - immutable output file generation without background retention
- dependency failure : C6, C7
- state transitions : n/a - stateless data verification
- observability : C8, C9

## Handoff

Files touched:
- `src/viral_cnn/reporting/publication.py` (35 KB)
- `src/viral_cnn/reporting/transformations.py` (12 KB)
- `tests/test_s15_artifact_provenance_hardening.py` (10 KB)

Total estimate: ~57 KB · ~14k tokens.
Budget: 150k tokens. Estimate is well within budget: one builder.
