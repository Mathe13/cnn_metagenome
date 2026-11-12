# Verification Report: s15-artifact-provenance-hardening

**Verdict**: PASS
**Status**: VERIFIED
**Profile**: light
**Round**: 1 - full
**Verifier**: Independent Subagent (author != verifier)
**Date**: 2026-11-11
**Commit**: e53259e

## Criteria Checked against Plan and Checks

All 9 acceptance criteria from `.specs/features/s15-artifact-provenance-hardening/plan.md` and `.specs/features/s15-artifact-provenance-hardening/checks.md` were independently verified against commit `e53259e`:

- **AC 1 / C1 (Complete Provenance Passes):** `validate_artifact_provenance` returns a verified provenance dictionary when all 12 canonical provenance fields (git commit, is_smoke_test=False, execution timestamp, library versions, seed, split, model variants, noise conditions, dataset SHA256, checkpoints, configuration/derivation, and parent artifacts) are valid and present.
- **AC 2 / C2 (Smoke Test Rejection):** `validate_artifact_provenance` raises `PublicationValidationError` fail-closed if `is_smoke_test` is missing or set to `True`.
- **AC 3 / C3 (Git Commit Enforcement):** `validate_artifact_provenance` raises `PublicationValidationError` when `git_commit` is omitted or empty.
- **AC 4 / C4 (Execution Timestamp Enforcement):** `validate_artifact_provenance` raises `PublicationValidationError` when execution timestamp (`created_utc` or `timestamp_utc`) is missing or whitespace.
- **AC 5 / C5 (Software Environment Enforcement):** `validate_artifact_provenance` raises `PublicationValidationError` when software versions (`versions` or `library_versions`) are omitted or non-dictionary.
- **AC 6 / C6 (Parent Integrity Enforcement):** `validate_artifact_provenance` raises `PublicationValidationError` when any declared parent artifact is missing on disk or exhibits a SHA-256 checksum mismatch.
- **AC 7 / C7 (Transitive Smoke Rejection):** `validate_artifact_provenance` raises `PublicationValidationError` when any upstream parent manifest in the transitive lineage declares `is_smoke_test=True`.
- **AC 8 / C8 (Lineage Propagation in Derived Manifests):** `emit_derived_manifest` emits a derived manifest that automatically populates `git_commit`, `created_utc`, `library_versions`, `is_smoke_test=False`, `parent_manifest_path`, and `parent_manifest_sha256`.
- **AC 9 / C9 (Publication Selection Strict Provenance):** `validate_selection` under `strict_empirical=True` and `strict_provenance=True` validates artifact provenance on empirical manifests and fails closed if any required attribute is missing or invalid.

## Checks Table

| Check | Claim | Proof run | Evidence | Result |
| --- | --- | --- | --- | --- |
| C1 | `validate_artifact_provenance` returns verified provenance dictionary when all 12 attributes are present and parent hashes match (AC 1) | `REMOTE_SERVER=False PYTHONPATH=src .venv/bin/python3 -m unittest tests.test_s15_artifact_provenance_hardening.TestS15ArtifactProvenanceHardening.test_valid_full_provenance_passes -k test_valid_full_provenance_passes` | `tests/test_s15_artifact_provenance_hardening.py:77` - `result = validate_artifact_provenance(self.valid_manifest_data, self.root)` | PASS |
| C2 | `validate_artifact_provenance` raises `PublicationValidationError` when `is_smoke_test` is True or missing (AC 2) | `REMOTE_SERVER=False PYTHONPATH=src .venv/bin/python3 -m unittest tests.test_s15_artifact_provenance_hardening.TestS15ArtifactProvenanceHardening.test_smoke_test_manifest_fails_provenance -k test_smoke_test_manifest_fails_provenance` | `tests/test_s15_artifact_provenance_hardening.py:90` - `with self.assertRaises(PublicationValidationError) as ctx:` | PASS |
| C3 | `validate_artifact_provenance` raises `PublicationValidationError` when `git_commit` is missing or empty (AC 3) | `REMOTE_SERVER=False PYTHONPATH=src .venv/bin/python3 -m unittest tests.test_s15_artifact_provenance_hardening.TestS15ArtifactProvenanceHardening.test_missing_git_commit_fails_provenance -k test_missing_git_commit_fails_provenance` | `tests/test_s15_artifact_provenance_hardening.py:106` - `with self.assertRaises(PublicationValidationError) as ctx:` | PASS |
| C4 | `validate_artifact_provenance` raises `PublicationValidationError` when timestamp (`timestamp_utc` or `created_utc`) is missing (AC 4) | `REMOTE_SERVER=False PYTHONPATH=src .venv/bin/python3 -m unittest tests.test_s15_artifact_provenance_hardening.TestS15ArtifactProvenanceHardening.test_missing_timestamp_fails_provenance -k test_missing_timestamp_fails_provenance` | `tests/test_s15_artifact_provenance_hardening.py:120` - `with self.assertRaises(PublicationValidationError) as ctx:` | PASS |
| C5 | `validate_artifact_provenance` raises `PublicationValidationError` when software versions (`versions` or `library_versions`) is missing (AC 5) | `REMOTE_SERVER=False PYTHONPATH=src .venv/bin/python3 -m unittest tests.test_s15_artifact_provenance_hardening.TestS15ArtifactProvenanceHardening.test_missing_software_versions_fails_provenance -k test_missing_software_versions_fails_provenance` | `tests/test_s15_artifact_provenance_hardening.py:134` - `with self.assertRaises(PublicationValidationError) as ctx:` | PASS |
| C6 | `validate_artifact_provenance` raises `PublicationValidationError` when a parent artifact is missing on disk or has a hash mismatch (AC 6) | `REMOTE_SERVER=False PYTHONPATH=src .venv/bin/python3 -m unittest tests.test_s15_artifact_provenance_hardening.TestS15ArtifactProvenanceHardening.test_broken_parent_artifact_fails_provenance -k test_broken_parent_artifact_fails_provenance` | `tests/test_s15_artifact_provenance_hardening.py:149` - `with self.assertRaises(PublicationValidationError) as ctx:` | PASS |
| C7 | `validate_artifact_provenance` raises `PublicationValidationError` when any upstream parent artifact in the transitive lineage declares `is_smoke_test=True` (AC 7) | `REMOTE_SERVER=False PYTHONPATH=src .venv/bin/python3 -m unittest tests.test_s15_artifact_provenance_hardening.TestS15ArtifactProvenanceHardening.test_transitive_smoke_parent_fails_provenance -k test_transitive_smoke_parent_fails_provenance` | `tests/test_s15_artifact_provenance_hardening.py:178` - `with self.assertRaises(PublicationValidationError) as ctx:` | PASS |
| C8 | `emit_derived_manifest` emits a derived manifest with git commit, timestamp, library versions, non-smoke flag, and parent manifest link (AC 8) | `REMOTE_SERVER=False PYTHONPATH=src .venv/bin/python3 -m unittest tests.test_s15_artifact_provenance_hardening.TestS15ArtifactProvenanceHardening.test_emit_derived_manifest_propagates_full_lineage -k test_emit_derived_manifest_propagates_full_lineage` | `tests/test_s15_artifact_provenance_hardening.py:193` - `manifest_path = emit_derived_manifest(...)` | PASS |
| C9 | `validate_selection` under `strict_empirical=True` enforces `validate_artifact_provenance` on all empirical entries and rejects invalid/synthetic provenance (AC 9) | `REMOTE_SERVER=False PYTHONPATH=src .venv/bin/python3 -m unittest tests.test_s15_artifact_provenance_hardening.TestS15ArtifactProvenanceHardening.test_validate_selection_enforces_artifact_provenance -k test_validate_selection_enforces_artifact_provenance` | `tests/test_s15_artifact_provenance_hardening.py:251` - `with self.assertRaises(PublicationValidationError) as ctx:` | PASS |

## Full Test Suite Gate

- `REMOTE_SERVER=False PYTHONPATH=src .venv/bin/python3 -m unittest -v tests/test_s15_artifact_provenance_hardening.py`: 9 passed, 0 failed.
- `REMOTE_SERVER=False PYTHONPATH=src .venv/bin/python3 -m unittest tests/test_m07_publication_contracts.py`: 16 passed, 0 failed.

## Summary and Signature

The implementation of `s15-artifact-provenance-hardening` in commit `e53259e` provides comprehensive fail-closed validation of empirical artifact provenance across all 12 required fields in `validate_artifact_provenance`, verifies on-disk presence and SHA-256 integrity of all parent dependencies, rejects synthetic or smoke evidence transitively across upstream lineages, extends `emit_derived_manifest` to propagate full git and environment metadata, and integrates strict provenance validation into `validate_selection`. All acceptance criteria C1 through C9 have been independently tested and verified with reproducible proofs under profile light.

**Signed by:** Independent Subagent Verifier
