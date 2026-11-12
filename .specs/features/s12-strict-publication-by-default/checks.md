# S12 Strict Publication by Default checks

Profile: light
Plan: `.specs/features/s12-strict-publication-by-default/plan.md`

## Intent

In `src/viral_cnn/reporting/tables.py:299`, `generate_all_tables` defaulted to `strict_empirical=False`, allowing silent emission of historical snapshot constants without empirical artifact verification. This feature enforces `strict_empirical=True` by default, raises a fail-closed `RuntimeError` when `publication_config` is omitted, introduces explicit dedicated APIs `generate_publication_tables` and `generate_historical_snapshot_tables` with non-admissible metadata, and deletes unreachable dead code.

9 checks in 1 slice · 4 one-way doors · 0 open, of which 0 block

## Checks

### S1 - Fail-closed publication reporting and historical isolation · 4 files · 41 KB · ~10k

**C1** - `generate_all_tables` called without `strict_empirical` executes with `strict_empirical=True` (AC 1)
Proof: `REMOTE_SERVER=False PYTHONPATH=src .venv/bin/python3 -m unittest tests.test_s12_strict_publication_by_default.TestS12StrictPublicationByDefault.test_default_strict_empirical_is_true -k test_default_strict_empirical_is_true`

**C2** - `generate_all_tables` with `strict_empirical=True` and `publication_config=None` raises `RuntimeError("Publication generation requires an explicit admissible empirical artifact selection.")` (AC 2)
Proof: `REMOTE_SERVER=False PYTHONPATH=src .venv/bin/python3 -m unittest tests.test_s12_strict_publication_by_default.TestS12StrictPublicationByDefault.test_missing_publication_config_raises_runtime_error -k test_missing_publication_config_raises_runtime_error`

**C3** - `generate_publication_tables` raises `FileNotFoundError` or `RuntimeError` when `publication_config` is missing or invalid (AC 3)
Proof: `REMOTE_SERVER=False PYTHONPATH=src .venv/bin/python3 -m unittest tests.test_s12_strict_publication_by_default.TestS12StrictPublicationByDefault.test_generate_publication_tables_validates_config -k test_generate_publication_tables_validates_config`

**C4** - `generate_publication_tables` with valid config renders all publication tables and returns dictionary of DataFrames (AC 4)
Proof: `REMOTE_SERVER=False PYTHONPATH=src .venv/bin/python3 -m unittest tests.test_s12_strict_publication_by_default.TestS12StrictPublicationByDefault.test_generate_publication_tables_renders_dataframe_dict -k test_generate_publication_tables_renders_dataframe_dict`

**C5** - `generate_historical_snapshot_tables` returns 12 DataFrames and metadata containing `evidence_class="historical_snapshot"` and `publication_admissible=False` (AC 5)
Proof: `REMOTE_SERVER=False PYTHONPATH=src .venv/bin/python3 -m unittest tests.test_s12_strict_publication_by_default.TestS12StrictPublicationByDefault.test_historical_snapshot_tables_emits_metadata -k test_historical_snapshot_tables_emits_metadata`

**C6** - `generate_all_tables` with explicit `strict_empirical=False` outputs legacy snapshot tables and returns 12 DataFrames (AC 6)
Proof: `REMOTE_SERVER=False PYTHONPATH=src .venv/bin/python3 -m unittest tests.test_s12_strict_publication_by_default.TestS12StrictPublicationByDefault.test_generate_all_tables_legacy_opt_in -k test_generate_all_tables_legacy_opt_in`

**C7** - Unreachable secondary `if strict_empirical:` directory globbing branch is completely absent from `tables.py` (AC 7)
Proof: `REMOTE_SERVER=False PYTHONPATH=src .venv/bin/python3 -m unittest tests.test_s12_strict_publication_by_default.TestS12StrictPublicationByDefault.test_dead_strict_branch_is_removed -k test_dead_strict_branch_is_removed`

**C8** - `scripts/generate_paper_tables.py` under default `--mode strict` returns exit code 2 when artifact selection is blocked (AC 8)
Proof: `REMOTE_SERVER=False PYTHONPATH=src .venv/bin/python3 -m unittest tests.test_s12_strict_publication_by_default.TestS12StrictPublicationByDefault.test_cli_strict_mode_fails_closed_when_blocked -k test_cli_strict_mode_fails_closed_when_blocked`

**C9** - `scripts/generate_paper_tables.py` under `--mode legacy` executes the historical generator without raising missing config error (AC 9)
Proof: `REMOTE_SERVER=False PYTHONPATH=src .venv/bin/python3 -m unittest tests.test_s12_strict_publication_by_default.TestS12StrictPublicationByDefault.test_cli_legacy_mode_generates_snapshot -k test_cli_legacy_mode_generates_snapshot`

## Coverage

| Set (size) | Member -> proof | Unproven |
| --- | --- | --- |
| generation modes (2) | `strict_empirical` C1 · `legacy_snapshot` C6 | - |
| publication config validation (2) | `missing_config` C2 · `invalid_config` C3 | - |
| historical snapshot metadata fields (2) | `evidence_class` C5 · `publication_admissible` C5 | - |
| dead code branches (1) | `rglob_dead_branch` C7 | - |
| cli invocation modes (2) | `mode_strict` C8 · `mode_legacy` C9 | - |

- No check claims more than the single case its proof exercises

## Swept

- validation : C2, C3
- failure modes : C2, C8
- idempotency : C4
- authorization : n/a - internal module logic without user authentication
- concurrency : n/a - stateless script invocations without shared mutable state
- data lifecycle : n/a - immutable output file generation without background retention
- dependency failure : n/a - pure local filesystem checks with standard python modules
- state transitions : n/a - stateless CLI execution modes
- observability : C5, C8

## Handoff

Files touched:
- `src/viral_cnn/reporting/tables.py` (25 KB)
- `scripts/generate_paper_tables.py` (3 KB)
- `tests/test_reporting_tables.py` (5 KB)
- `tests/test_s12_strict_publication_by_default.py` (8 KB)

Total estimate: ~41 KB · ~10k tokens.
Budget: 150k tokens. Estimate is well within budget: one builder.
