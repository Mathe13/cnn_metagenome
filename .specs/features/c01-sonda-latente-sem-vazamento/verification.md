# C01 verification

**Verdict**: PASS
**Profile**: light
**Diff range**: `4cbb0c27041be3c019b85b73bb4a4f88b201d419..33f2085`
**Round**: 2 - scoped (C5/C10 fix commit; complete proof set rerun at HEAD)
**Verifier**: independent sub-agent (author != verifier)

## Checks

| Check | Claim | Proof run | Evidence | Result |
| --- | --- | --- | --- | --- |
| C1 | Pre-`fc` values, labels, stable IDs; reject partition aliases | `MPLCONFIGDIR=/tmp/mpl .venv/bin/python -m unittest -v tests.test_c01_latent_audit` exit 0 | `tests/test_c01_latent_audit.py:54-56` - `assert_allclose(feats, x.numpy())`, labels and row IDs; `:66` - hashed partition row IDs; `:72-76` - alias raises `ValueError`. | PASS |
| C2 | Cache identity gates reuse; stale, legacy and corrupt caches rejected | same invocation, `test_cache_identity_rejects_legacy_and_stale_artifacts` passed | `tests/test_c01_latent_audit.py:84-96` - valid load, mismatched split, missing manifest, and checksum mismatch assertions. | PASS |
| C3 | Sampling population, extracted count, rule and seed recorded | same invocation, `test_sampling_provenance_is_recorded` passed | `tests/test_c01_latent_audit.py:102-104` - asserts population, count, and `{"rule": "none", "seed": ctx.seed}` metadata in notebook source. | PASS |
| C4 | Test-origin artifacts cannot fit, even with a forged train label; no update | same invocation, `test_test_artifact_cannot_be_fitted_even_with_train_label` passed | `tests/test_c01_latent_audit.py:113-118` - fit rejects test artifact and raw fit rejects `split_origin="test"`; `:115-116` confirms weights unchanged. | PASS |
| C5 | Earliest validation-accuracy maximum selected, hash sealed before test access; no refit or repeated evaluation | same invocation, `test_validation_selection_and_seal_are_irreversible` passed | `tests/test_c01_latent_audit.py:125-140` - hash/rule/epoch assertions, test IDs and hash match, second evaluation/refit raise, and `assertLess(cell.index("fit_select_and_seal"), cell.index('split="test"'))`. Notebook ordering is also visible at `notebooks/reproduction/short/06_ictv_latent_space_and_hierarchy.ipynb:228-245`. | PASS |
| C6 | Five variants, same identified test rows, probe/native comparison metrics | same invocation, `test_all_five_variants_and_shared_test_identities` passed; smoke proof below also passed | `tests/test_c01_latent_audit.py:204-216` - all five variant names and row IDs/metrics fields; notebook `:449-465` loops variants and writes Accuracy, Macro-F1, N, supported classes, test IDs, and native accuracy from the same test artifact. | PASS |
| C7 | Stratified cosine KNN has disjoint IDs/no self-neighbor and explicit Top-5 meaning | same invocation, `test_knn_records_disjoint_identity_and_denominators` passed | `tests/test_c01_latent_audit.py:149-154` - requested `k=5`, zero overlap/self-neighbors, query denominator, and Top-5 definition. | PASS |
| C8 | Degenerate KNN unavailable with reason; no fallback or silent k reduction | same invocation, `test_degenerate_support_is_unavailable` passed | `tests/test_c01_latent_audit.py:159-167` - no stratified queries and fewer-than-five references produce unavailable results with explicit reasons. | PASS |
| C9 | KNN and Silhouette supports remain separate; class and sample rates distinct | same invocation, `test_taxonomic_supports_remain_independent` passed | `tests/test_c01_latent_audit.py:172-181` - KNN class counts plus independent Silhouette valid/excluded classes, retained sample count/rate, and valid-class rate. | PASS |
| C10 | Missing genome IDs remain unknown; supplied IDs yield disjoint result and explicit species exclusions | same invocation, `test_genome_independence_requires_explicit_ids` passed | `tests/test_c01_latent_audit.py:183-193` - no inferred `genome_ids`, excluded species label `[2]`, exclusion count, and zero accession overlap. Notebook unknown status is asserted at `:228`. | PASS |
| C11 | Preflight, five variants, reserved run namespace and manifest | same invocation, `test_notebook_preflight_and_run_manifest_contract` passed; `.venv/bin/python -m unittest -v tests.test_n04_taxonomy_and_latent_space.TestN04TaxonomyAndLatentSpace.test_c11_notebook_06_execution_smoke_test` exit 0 (6.289 s) | `tests/test_c01_latent_audit.py:219-224` checks preflight/reservation/run ID/train/validation; `:72-76` exercises alias rejection; `tests/test_n04_taxonomy_and_latent_space.py:247-256` executes every code cell and asserts under 60 s. The run completed five variants; generated manifest reports `is_smoke_test=true`, five models and 36 outputs. Runtime validation logic is at `src/viral_cnn/notebook_runtime.py:254-284`. | PASS |
| C12 | Smoke uses isolated synthetic inputs and is marked; empirical branch does not select those fixtures | same C01 invocation, `test_smoke_and_empirical_inputs_are_separated` passed; C11 smoke execution passed | `tests/test_c01_latent_audit.py:225-228` asserts explicit smoke/empirical branch and unknown genome status. Notebook `:128-154` creates separate train/validation/test fixtures only under smoke and resolves data partitions in the other branch; generated run manifest has `is_smoke_test=true`. | PASS |
| C13 | Report separates local from empirical evidence and preserves discrepancy without claiming reproduction | same C01 invocation, `test_evidence_boundary_is_documented` passed | `tests/test_c01_latent_audit.py:230-232` asserts audit export and `scientific_reproduction_certified` marker. This report records the unavailable empirical inputs and the historical 768/2048 wording discrepancy below. | PASS |

## Coverage

| Set (size) | Recomputed from | Member -> proof | Unproven |
| --- | --- | --- | --- |
| Landing doors (2) | `plan.md` Landing | identity-persisted embeddings -> C1,C2,C3; sealed train/validation probe -> C4,C5 | - |
| Relations (6) | `plan.md` Relations | run/variant/split -> C1,C2,C11; embedding row/label/identity -> C1,C7; probe/selection -> C4,C5; final evaluation -> C6; KNN reference/query -> C7,C8; genome IDs -> C10 | - |
| Notebook preflight statuses (3) | `plan.md` Surface | missing inputs/taxonomy -> C11 runtime checks and smoke execution; split alias -> C1 | - |
| Cache statuses (4) | `plan.md` Surface | valid hit/stale/legacy/corrupt -> C2 | - |
| Extraction statuses (3) | `plan.md` Surface | dimension/alignment -> C1; count/sampling -> C3 | - |
| Probe statuses (4) | `plan.md` Surface | invalid origin -> C4; validation selection/seal/test evaluation -> C5,C6 | - |
| KNN statuses (3) | `plan.md` Surface | available -> C7; empty/insufficient -> C8 | - |
| Genome KNN statuses (2) | `plan.md` Surface | IDs absent/present -> C10 | - |
| Table export (1) | `plan.md` Surface | probe metrics and independent supports -> C6,C9 | - |
| Run mode statuses (2) | `plan.md` Surface | reserved outputs/manifest -> C11; synthetic mode -> C12 | - |
| Evidence status (1) | `plan.md` Surface | local vs empirical boundary -> C13 and this report | - |

## Test policy rows

| Row | Files it classifies | Required proof | Expectation met |
| --- | --- | --- | --- |
| Scientific eval decisions | `tests/test_c01_latent_audit.py` | Provenance, irreversible probe states, and support boundaries | yes; 13 tests ran and passed individually. |
| Notebook assembly | `tests/test_c01_latent_audit.py`, `tests/test_n04_taxonomy_and_latent_space.py` | Source contracts plus controlled smoke fixtures | yes; contract tests passed and notebook 06 executed end-to-end with per-run synthetic fixtures. |

## Faults injected

Not run: the approved profile is `light`, which does not require mutation testing.

## Swept existing

- Validation: C1-C5 and C11 have test/runtime evidence for required identities, ordering, and preflight handling.
- Failure modes: C2 rejects stale/legacy/corrupt cache inputs; C8 rejects degenerate KNN; C10 requires explicit genome IDs.
- Idempotency: C5 prohibits repeat evaluation/refit; run reservation behavior is exercised by the existing C00 test policy.
- Authorization: n/a, local pipeline.
- Concurrency: C2 writes immutable fingerprinted cache names; C11 uses C00 run reservation.
- Data lifecycle: C1-C3 and C11-C12 preserve provenance and synthetic/empirical separation.
- Dependency failure: C11 fails closed for absent required paths/checkpoints in runtime; empirical artifacts unavailable here, noted below.
- State transitions: C4-C5 cover fit, selection, seal, and evaluation.
- Observability: manifests, path logging, selection history, KNN identities/denominators, and evidence boundary are covered by C1-C13.

## Evidence limits and divergences

All 13 focused tests passed (`Ran 13 tests in 1.364s`), and the additional notebook 06 smoke test passed in 6.289s. The smoke used three distinct 16-row synthetic fixtures, smoke model initialization, and CPU. It is an integration check only; its metrics are not scientific evidence.

No empirical execution was possible: `/home/mlima/workspace` and the five Species checkpoints (ResNet50, ResNet50_F, ConvNeXt_Z, ConvNeXt_T, ConvNeXt_F) were not available/verified. Test-fragment genome IDs are also unavailable, so genome independence remains unknown. The notebook introduction still says ConvNeXt is 768-dimensional while the active model dimension is read from `model.fc.in_features` and the plan/manuscript evidence says 2048; the existing manuscript audit also records the article's Euclidean Silhouette wording versus cosine in method/code. These editorial differences are not silently reconciled by C01.

Cache API identity and rejection are tested, but notebook 06 currently writes cache artifacts and does not call `load_embedding_artifact` to reuse them on a later run. The plan makes reuse conditional on a cache being consulted; this round verifies the loader contract and the notebook's traceable per-run writes, not an end-to-end cache-hit path.

## Gate

`python3 .agent/skills/tlc-spec-lean/scripts/validate_verification.py c01-sonda-latente-sem-vazamento` - exit 0; 0 errors, 0 warnings.
