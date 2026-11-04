# C02 checks

Profile: light
Plan: `.specs/features/c02-auditoria-empirica-sequencias-longas/plan.md`

## Checks

### S1 — observed inputs and sampling

**C1** — Long-input audit reports available, eligible and excluded counts, input hash, detected data kind, sequence lengths, ambiguity and observed taxon/genome support; absent genome identity remains `unknown`.
Proof: `.venv/bin/python -m unittest -v tests.test_c02_long_audit -k test_audit_reports_observed_support_and_unknown_genomes`

**C2** — Explicit sampling records planned/realized sizes, deterministic selection rule, seed and selected query IDs; overlap declares separate train/query denominators, k, canonicalization and queries without valid k-mers.
Proof: `.venv/bin/python -m unittest -v tests.test_c02_long_audit -k test_sampling_and_overlap_provenance`

### S2 — identified paired inference and comparison

**C3** — Canonical label mapping requires an identified mapping and preserves source labels; missing, ambiguous or out-of-domain labels fail with the responsible IDs.
Proof: `.venv/bin/python -m unittest -v tests.test_c02_long_audit -k test_mapping_requires_unique_identified_canonical_order`

**C4** — Paired comparison accepts exactly the five distinct registered variants on identical unique query IDs, truth labels and condition; missing, duplicate, extra, mislabeled or incompatible artifacts fail with enumerated issues.
Proof: `.venv/bin/python -m unittest -v tests.test_c02_long_audit -k test_comparison_rejects_incomplete_or_misaligned_variants`

**C5** — Table 12 values are computed from observed correct/N for both aggregations and export N, coverage, exclusions; a skipped prediction remains an error in the global denominator and conditional accuracy is separate.
Proof: `.venv/bin/python -m unittest -v tests.test_c02_long_audit -k test_table_metrics_use_observed_denominator_and_skip_is_error`

**C6** — Prediction artifacts use `long_sequence_protocol=c02_audit_v1`, unique `(query_id, variant, noise_rate, handle_short)` keys, canonical Species truth, and manifests identify query/mapping/checkpoint hashes and window parameters.
Proof: `.venv/bin/python -m unittest -v tests.test_c02_long_audit -k test_prediction_artifact_contract_and_checkpoint_hashes`

### S3 — window protocol and diagnostics

**C7** — Voter records 512-bp windows, 256-bp stride, dropped incomplete tail, pad/skip mode, min-index tie break and observed window count; singleton FP32 logits agree within `1e-5`.
Proof: `.venv/bin/python -m unittest -v tests.test_c02_long_audit -k test_window_metadata_and_singleton_equivalence`

**C8** — Stratification uses complete disjoint length and ambiguity bins, includes unknown metadata, and emits `N=0` with unavailable metrics for empty strata.
Proof: `.venv/bin/python -m unittest -v tests.test_c02_long_audit -k test_strata_are_complete_and_empty_is_unavailable`

**C9** — Composition audit compares query-weighted and taxon-macro accuracy on the same identified rows and reports weights and percentage-point difference without genome attribution.
Proof: `.venv/bin/python -m unittest -v tests.test_c02_long_audit -k test_composition_reports_query_and_taxon_weighting`

**C10** — Window diagnostics report mean/vote disagreement, logit magnitudes and margins, plus identified leave-one-window-out flips; insufficient logits produce `INCONCLUSIVE`.
Proof: `.venv/bin/python -m unittest -v tests.test_c02_long_audit -k test_logit_influence_requires_and_measures_window_logits`

**C11** — Consensus calibration calculates 15-bin ECE and NLL from softmax of mean logits and records N/confidence; missing logits are `INCONCLUSIVE`.
Proof: `.venv/bin/python -m unittest -v tests.test_c02_long_audit -k test_calibration_uses_mean_logits_and_fifteen_bins`

**C12** — Findings use only `SUPPORTED`, `HYPOTHESIS` or `INCONCLUSIVE`, cite test/effect/support/inputs/limitations, and fixtures cannot be marked empirically supported.
Proof: `.venv/bin/python -m unittest -v tests.test_c02_long_audit -k test_evidence_status_rejects_synthetic_support`

### S4 — notebooks, provenance and evidence boundary

**C13** — Long/01–03 use exclusive reserved-run run directories, explicit predecessor manifests, input/output hashes and runtime versions; artifact paths are logged.
Proof: `.venv/bin/python -m unittest -v tests.test_c02_long_audit -k test_notebooks_use_c00_run_manifests_and_artifact_logging`

**C14** — Smoke execution uses deterministic CPU fixtures and marks `is_smoke_test=true`; empirical mode blocks absent inputs/checkpoints and cannot promote synthetic outputs.
Proof: `.venv/bin/python -m unittest -v tests.test_c02_long_audit -k test_smoke_and_empirical_evidence_are_separated`

**C15** — Evidence report separates static inspection, synthetic contracts and remote empirical execution, and documents historical/article divergences and unavailable artifacts without claiming reproduction.
Proof: `.venv/bin/python -m unittest -v tests.test_c02_long_audit -k test_evidence_boundary_and_ad010_revision_are_documented`

## Coverage

| Set (size) | Member -> proof | Unproven |
| --- | --- | --- |
| Landing doors (3) | paired prediction artifact -> C4,C5,C6; diagnostic evidence record -> C10,C11,C12; AD-010 superseding decision -> C15 | - |
| Relations (5) | universe/sample -> C1,C2; query/sequence/truth/origin -> C1,C3,C4; prediction/checkpoint/condition -> C4,C6; five-variant comparison -> C4,C5; diagnostic finding -> C10–C12 | - |
| Surface: long/01 and origin (2) | universe/sample observed -> C1,C2; genome identity unknown -> C1 | - |
| Surface: long/02 and voter (3) | identified inference -> C3,C6; mapping failure -> C3; window protocol -> C7 | - |
| Surface: long/03 and Table 12 (4) | complete comparison -> C4; invalid/misaligned artifacts -> C4; observed metrics -> C5; skipped query -> C5 | - |
| Surface: strata and diagnostics (5) | complete/empty strata -> C8; weighted composition -> C9; window influence/inconclusive -> C10; calibration -> C11; bounded evidence states -> C12 | - |
| Surface: execution/evidence (3) | reserved manifests -> C13; smoke -> C14; empirical blocked/local evidence -> C14,C15 | - |
| Protocol conditions (5) | clean epsilon=0 -> C4,C6; shared optional corruption protocol -> C6; pad -> C7; skip -> C5,C7; 512/256 tail drop -> C7 | - |

## Swept

- validation: C3–C8
- failure modes: C3,C4,C10,C14
- idempotency: C13
- authorization: n/a — execução exclusivamente local, sem serviço externo
- concurrency: C13 — reserva exclusiva do runtime e publicação em namespace novo
- data lifecycle: C1,C2,C6,C13,C14
- dependency failure: C13–C15
- state transitions: C10–C12
- observability: C1,C2,C6–C15

## Handoff

15 checks, 1 slice · ~12 touched files · estimated 240 KB / 4 = ~60k tokens · 3 one-way doors · 0 open, 0 blocked. One builder; under the 150k budget. Proofs are synthetic and contract-level; no empirical run is possible without remote inputs/checkpoints.

- **Boundary:** pending
- **Settled mid-build:** none
- **Abandoned:** none
