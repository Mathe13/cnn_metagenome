# C01 checks

Profile: light
Plan: `.specs/features/c01-sonda-latente-sem-vazamento/plan.md`

## Checks

### S1 — embeddings and cache identity

**C1** — Extraction returns pre-`fc` vectors, labels, and stable row identities from the same observed batches; partition aliases are rejected.
Proof: `.venv/bin/python -m unittest -v tests.test_c01_latent_audit -k test_extraction_preserves_rows_and_rejects_partition_alias`

**C2** — A cache is reused only when its versioned manifest matches split, partitions, checkpoint, class order, encoding, mode, and extractor; incomplete or stale caches remain untouched and trigger a fresh cache path.
Proof: `.venv/bin/python -m unittest -v tests.test_c01_latent_audit -k test_cache_identity_rejects_legacy_and_stale_artifacts`

**C3** — Sampling metadata reports population, extracted count, limit rule, and seed for each split.
Proof: `.venv/bin/python -m unittest -v tests.test_c01_latent_audit -k test_sampling_provenance_is_recorded`

### S2 — independent frozen probe

**C4** — Audited fit accepts only verified train embeddings and fails before any parameter update when test provenance is supplied, including a forged `split_origin`.
Proof: `.venv/bin/python -m unittest -v tests.test_c01_latent_audit -k test_test_artifact_cannot_be_fitted_even_with_train_label`

**C5** — The probe selects the earliest maximum validation accuracy, stores the per-epoch history and seals weights with a hash before test loading; fit/selection after sealing and repeated test evaluation fail.
Proof: `.venv/bin/python -m unittest -v tests.test_c01_latent_audit -k test_validation_selection_and_seal_are_irreversible`

**C6** — The final probe and native-head comparison use the same identified test rows and report Accuracy, Macro-F1, N, and supported-class count for all five Species variants.
Proof: `.venv/bin/python -m unittest -v tests.test_c01_latent_audit -k test_all_five_variants_and_shared_test_identities`

### S3 — fragment KNN audit

**C7** — Stratified cosine KNN records reference/query identities, effective proportions, zero overlap, and zero self-neighbors; Top-5 is explicitly neighbor-label inclusion.
Proof: `.venv/bin/python -m unittest -v tests.test_c01_latent_audit -k test_knn_records_disjoint_identity_and_denominators`

**C8** — KNN is unavailable with a reason when there are no eligible queries or fewer than five references; it never falls back to an unstratified split or silently lowers k.
Proof: `.venv/bin/python -m unittest -v tests.test_c01_latent_audit -k test_degenerate_support_is_unavailable`

**C9** — Species, Genus, and Order metrics retain independent valid-class supports from Silhouette; class fraction and retained-sample fraction have distinct fields.
Proof: `.venv/bin/python -m unittest -v tests.test_c01_latent_audit -k test_taxonomic_supports_remain_independent`

**C10** — Missing genome IDs yield `genome_independence: unknown` and no genome-disjoint result; supplied valid IDs produce disjoint genome sets and explicit exclusions.
Proof: `.venv/bin/python -m unittest -v tests.test_c01_latent_audit -k test_genome_independence_requires_explicit_ids`

### S4 — execution evidence

**C11** — Notebook 06 preflights explicit nonempty train/validation/test inputs, aligned taxonomy, and all five checkpoints before metrics, and writes artifacts under a reserved run namespace with a C00 manifest.
Proof: `.venv/bin/python -m unittest -v tests.test_c01_latent_audit -k test_notebook_preflight_and_run_manifest_contract`

**C12** — Smoke outputs are explicitly synthetic and marked `is_smoke_test: true`; empirical mode never substitutes synthetic inputs.
Proof: `.venv/bin/python -m unittest -v tests.test_c01_latent_audit -k test_smoke_and_empirical_inputs_are_separated`

**C13** — Verification records local proofs separately from empirical execution, names unavailable data/checkpoints, and preserves the historical notebook discrepancy without claiming reproduction.
Proof: `.venv/bin/python -m unittest -v tests.test_c01_latent_audit -k test_evidence_boundary_is_documented`

## Coverage

| Set (size) | Member -> proof | Unproven |
| --- | --- | --- |
| Landing doors (2) | identity-persisted embeddings -> C1,C2,C3; train/validation-fitted sealed probe -> C4,C5 | - |
| Relations (6) | run/variant/split -> C1,C2,C11; embedding row/label/identity -> C1,C7; probe/selection -> C4,C5; final evaluation -> C6; KNN reference/query -> C7,C8; optional genome IDs -> C10 | - |
| Surface: notebook preflight (3 statuses) | missing input -> C11; invalid taxonomy -> C11; split alias -> C1,C11 | - |
| Surface: cache (4 statuses) | valid hit -> C2; stale -> C2; legacy -> C2; interrupted -> C2 | - |
| Surface: extraction (3 statuses) | observed dimension -> C1; row support/order -> C1,C3; bounded sampling -> C3 | - |
| Surface: probe (4 statuses) | invalid origin -> C4; validation selection -> C5; sealed -> C5; test evaluation -> C5,C6 | - |
| Surface: KNN (3 statuses) | valid -> C7; empty -> C8; insufficient reference -> C8 | - |
| Surface: genome KNN (2 statuses) | IDs missing -> C10; IDs provided -> C10 | - |
| Surface: tables (1 status) | exports -> C6,C9 | - |
| Surface: run mode (2 statuses) | repeated/reserved output -> C11; smoke mode -> C12 | - |
| Surface: evidence (1 status) | local vs empirical -> C13 | - |

## Test policy

| Layer | Level | Required assertion | Precedent |
| --- | --- | --- | --- |
| Scientific eval decisions | unit | Exercise every named provenance/state/support boundary | `tests/test_latent_space.py` |
| Notebook assembly | contract | Inspect source contract plus controlled smoke fixtures; no remote data required | `tests/test_n04_taxonomy_and_latent_space.py` |

## Swept

- validation: C1–C5,C11
- failure modes: C2,C8,C10
- idempotency: C5,C11
- authorization: n/a — local pipeline, no external service
- concurrency: C2,C11 — atomic cache publication and exclusive run reservation
- data lifecycle: C1–C3,C11–C12
- dependency failure: C11,C13 — local checks versus unavailable remote artifacts
- state transitions: C4–C5
- observability: C1–C3,C7–C13

## Handoff

13 checks, 1 slice · 5 files · 119,019 bytes / 4 = ~29,755 tokens · 2 one-way doors · 0 open, 0 blocked. One builder. Changes are local and bounded to the latent evaluator, Parquet row identity, notebook runtime integration, notebook 06, and focused proofs; estimate fits the skill's 150k budget.
