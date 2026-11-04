# C05 — Fine-tuning auditável e comparação controlada verification

**Verdict**: PASS
**Profile**: light
**Diff range**: e43acda80faff14c8cc4527a310f7d204b8f2df9..048f9ce1f2ab5c78c1e8055d72f93d393fa1de76
**Round**: 3 - scoped
**Verifier**: independent sub-agent (author != verifier)

## Provenance

- Binding-source review, plan/source reconciliation, and `Swept existing` findings are carried from Round 1 at `29d305cc99501ce2c4dd8732ea79b670e1ee62f5`; Round 2 at `2791d696c5f15bfb7da5c4379ae496d31338f1b8` likewise recorded that the implementation fix did not change source interfaces or documents. No binding-source contradiction was reported. N03/R02/S12 artifacts remain unavailable in this checkout.
- Round 1's test-policy/profile finding carries forward: `checks.md` remains `Profile: light` with no `Test policy` rows.
- Round 2 isolated C2 as the only failure. The Round 3 fix commit `048f9ce1f2ab5c78c1e8055d72f93d393fa1de76` changes only `tests/test_c05_finetune_audit.py`; the C2 proof now asserts every declared input hash. All C1-C15 named proofs were rerun at Round 3 HEAD below. Test-file citations are refreshed to current line numbers.

## Binding sources

Carried from Round 1 at `29d305cc99501ce2c4dd8732ea79b670e1ee62f5`; not reopened in this scoped round.

| Source | Provenance | Contradiction | Uncovered |
| --- | --- | --- | --- |
| `ROADMAP_CORRECOES_REPRODUCAO.md` C05/C06 | carried; opened Round 1 | none; conditional causal scope retained | - |
| `AGENTS.md`, `src/viral_cnn/AGENTS.md`, `.specs/STATE.md` AD-016/018/023 | carried; opened Round 1 | none found for C05 claims | - |
| historical `short/finetune_resnet50_species.ipynb` and `short/finetune_convnext_species.ipynb` | carried; opened Round 1 | none; historical differences remain documented | - |
| specs N03/R02/S12 | carried; searched Round 1, unavailable in checkout | not assessable; unavailable specs were not treated as binding evidence | - |
| reproduction notebooks 03/04, training/reporting code, docs and test file | carried; opened Round 1 | none found in scoped decisions | - |

## Checks

Proof run at HEAD `048f9ce1f2ab5c78c1e8055d72f93d393fa1de76`: `.venv/bin/python3 -m unittest tests.test_c05_finetune_audit -v` — exit 0, 16 tests passed in 68.546s. This invocation executes every named proof in C1-C15 (C8 has two proofs). C13 ran both notebooks end-to-end in forced local synthetic smoke mode; this is contract evidence, not empirical fine-tuning or controlled-comparison evidence.

| Check | Proof run at HEAD | Evidence (assertion expression) | Result |
| --- | --- | --- | --- |
| C1 | `test_empirical_preflight_requires_all_inputs` PASS | `tests/test_c05_finetune_audit.py:43-55` removes each required input in turn and asserts `FileNotFoundError` before model construction, then checks all five identities; `:56-59` checks preflight precedes model construction in both notebooks. | PASS |
| C2 | `test_manifest_records_lineage_and_checkpoint_roles` PASS | `tests/test_c05_finetune_audit.py:103-108` supplies train, validation, test, taxonomy mapping and base checkpoint hashes; `:126-135` compares the complete hash mapping against expected values computed from all five inputs. `:118-125` checks protocol/base lineage/class order/rank/dropout/config; `:136-150` checks selected/last/best roles and unknown dropout. | PASS |
| C3 | `test_protocol_audit_reports_historical_divergences` PASS | `tests/test_c05_finetune_audit.py:152-167` asserts epoch scheduler unit, notebook/registry dropout values, and published-checkpoint equivalence remains unknown. | PASS |
| C4 | `test_train_and_validation_noise_are_audited` PASS | `tests/test_c05_finetune_audit.py:230-248` asserts train and validation both receive epsilon 0.05; resampling covers A/C/G/T and can retain or change the original; notebook manifest/history includes seed, support, attempted steps and history append. | PASS |
| C5 | `test_pretraining_evaluation_labels_support_and_noise` PASS | `tests/test_c05_finetune_audit.py:250-266` checks clean and epsilon 0.05 evaluation labels, separate support, preview limit/completeness and test support/completeness. | PASS |
| C6 | `test_checkpoint_selection_last_best_and_zero_tie` PASS | `tests/test_c05_finetune_audit.py:268-277` asserts last selects epoch 4, all-zero best selects first epoch 1, and tied maximum selects first epoch 3. | PASS |
| C7 | `test_checkpoint_contains_full_versioned_state_and_rng` PASS | `tests/test_c05_finetune_audit.py:285-306` checks schema, model/optimizer/scheduler/scaler/history, Python/NumPy/Torch CPU/CUDA/noise RNG, run identity/role/config, best metric and step counts. Atomic publication is separately proven by C9. | PASS |
| C8 | `test_resume_validates_identity_and_restores_required_state`; `test_resume_rejects_incompatible_or_missing_state` PASS | `tests/test_c05_finetune_audit.py:308-344` checks next epoch/history/best metric, noise RNG, model, optimizer, scheduler and scaler restoration. `:346-387` rejects identity mismatch and each required state/RNG omission, asserting model tensors remain unchanged after rejection. | PASS |
| C9 | `test_atomic_checkpoint_preserves_previous_file_on_replace_failure` PASS | `tests/test_c05_finetune_audit.py:389-397` injects replace failure and asserts prior checkpoint bytes remain intact. | PASS |
| C10 | `test_notebooks_isolate_runs_and_require_explicit_resume` PASS | `tests/test_c05_finetune_audit.py:401-418` checks exclusive run reservation, explicit resume, checkpoint/input hashes, prevalidated base path, selected payload loaded into target model, and no automatic resume by `last` existence. | PASS |
| C11 | `test_manifest_never_promotes_smoke_or_incomplete_run` PASS | `tests/test_c05_finetune_audit.py:169-205` asserts smoke becomes `SYNTHETIC`, incomplete or unselected runs become `PARTIAL`, and a complete run records selected hash, preview support and actual stages. | PASS |
| C12 | `test_report_marks_optional_comparison_not_run_and_limits_claims` PASS | `tests/test_c05_finetune_audit.py:440-444` asserts `not_run`, causal unavailability and one run per configuration. | PASS |
| C13 | `test_report_separates_test_evaluation_from_selection_and_architecture_claims` PASS | `tests/test_c05_finetune_audit.py:446-461` asserts identified test split, no test-based selection, distinct origins and no architecture-only attribution; `:462-493` executes both notebooks with `SMOKE_TEST=1` and `REMOTE_SERVER=False`, then asserts synthetic status, test-independent selection, selected hash/role, five noise evaluations, support 16, incomplete split and test-file identity. | PASS |
| C14 | `test_parquet_loader_yields_without_identity_tracking` PASS | `tests/test_c05_finetune_audit.py:61-70` creates Parquet with sequence/label only and asserts encoded sequence and label batches. | PASS |
| C15 | `test_generated_report_never_invents_controlled_results` PASS | `tests/test_c05_finetune_audit.py:421-438` asserts `NOT_RUN`/`NOT_MEASURED`, rejects known historical metrics and requires every benchmark metric cell to be `NOT_RUN`. | PASS |

## Coverage

Not run under Profile `light`; no coverage table is required. Round 1 provenance for coverage carries unchanged; the Round 3 fix adds no source-set members.

## Test policy rows

No `Test policy` rows are present in `checks.md` (carried from Round 1).

## Level and sampling judgment

The 16 proofs are CPU/synthetic contracts and static audit checks, except C13, which executes both notebook smoke paths on fixtures. The run emitted `SYNTHETIC` manifests, not empirical results. No canonical empirical checkpoints or cluster data were available for this verification. CUDA initialization warned that the local NVIDIA driver is too old; CUDA determinism and worker-resume behavior remain unproven. Round 3 closes the only Round 2 gap: assertions now cover all five C2 input-hash categories. The untracked `results_reproduction/` directories were preserved.

## Swept existing

Carried from Round 1 at `29d305cc99501ce2c4dd8732ea79b670e1ee62f5`; unchanged sources. That pass re-read local-only/no-HTTP, C00 run reservation, LOG_PATH artifact logging and preservation of historical artifacts; it found no cited constraint absent in the inspected surface. N/A HTTP/visual rows remain policy declarations.

## Gate

`python3 .agent/skills/tlc-spec-lean/scripts/validate_verification.py c05-finetune-e-comparacao-controlada` — exit 0: 0 errors, 0 warnings.

## Ranked gaps

None identified in the scoped Round 3 audit. The C2 test fixture uses a fixed synthetic `trainhash`, while validation/test/taxonomy/base checkpoint hashes are computed from fixture files and all five manifest values are asserted. Empirical data/checkpoint validation remains outside this test proof.

## Lessons

Round 2 grounded gap is closed by the Round 3 fix; no new lesson signal identified.
