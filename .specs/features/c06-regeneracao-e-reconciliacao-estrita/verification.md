# C06 verification

**Verdict: FAIL (carried assertion gaps)**  
Profile: light  
**Diff range:** `7237031a0dd219622297c757d5fefecf7343f606..7ce74cdca220287f515006a914edb842046c51bf`  
**Round:** 4 - scoped  
**Verifier:** independent sub-agent (author != verifier)

Round 4 reviews the latest diff plus every non-PASS finding carried from Round 3. HEAD is `7ce74cdca220287f515006a914edb842046c51bf`. The fix adds population identity, better source/manifest hashes, unreadable-artifact diagnostics, strict transitive-synthetic rejection, C05 package proof, report content assertions, and fixture assembly. It resolves the earlier findings for C2-C4, C6, and C19, and strengthens C12, C17, C26, C27. C17, C23, C26, and C27 still do not prove their entire named claims.

## Binding sources

| Source | Opened | Contradiction | Uncovered |
| --- | --- | --- | --- |
| `AGENTS.md`, `README.md` | yes | none; fixture evidence is not empirical reproduction and the fragment-level scientific limits are retained | none in this narrow comparison |
| `ROADMAP_CORRECOES_REPRODUCAO.md` C00-C06 and gates | yes | none; C05 remains optional and C06 strict evidence remains required | none |
| `.specs/STATE.md` AD-024 / handoff | yes | none; explicit 17-item selection and missing empirical bundle are consistent | none |
| C00-C05 plans/checks/reports, `src/viral_cnn/AGENTS.md` | yes | none found in the C06 claims reviewed | none |
| C06 plan/checks, config, changed reporting/CLI/notebook sources and tests | yes | none in selectors/IDs or intended scientific boundary | none in listed 17 publication items and three Landing doors |

Plan sources (`ROADMAP_CORRECOES_REPRODUCAO.md`, repository and source `AGENTS.md`, `.specs/STATE.md`, C00-C05 specs, reporting sources/manuscript/CLIs) were opened. They do not specify exact UI composition. C06 has no HTTP/API routes. Config maps all T1–T12/F1–F5; checks enumerate each item, all six evidence statuses, four gates, four CLI/notebook surfaces, and both editorial variants. No binding-source contradiction or unenumerated plan member was found. The test suite is fixture-only and does not establish empirical reproduction; canonical data/checkpoints and `/home/mlima/workspace` remain unavailable per `.specs/STATE.md`.

## Checks

One proof-target invocation at HEAD:

```bash
env -u MPLCONFIGDIR PYTHONPATH=src python3 -m unittest -v tests.test_c06_publication_package
```

Result: **28 tests passed**, exit 0. Every named selector in `checks.md` appeared and ran: C1 `test_index_covers_all_publication_items`; C2 `test_selected_input_validation_reports_missing_corrupt_and_unproven_inputs`; C3 `test_scientific_identity_and_composition_must_match`; C4 `test_strict_gate_fails_closed_per_required_item`; C5 `test_table_11_requires_architecture_correlation_semantics`; C6 `test_only_selected_and_transitive_inputs_affect_admissibility`; C7 `test_partial_selection_is_diagnostic_and_never_complete`; C8 `test_tables_are_derived_and_historical_values_are_separate`; C9 `test_structural_counts_are_separated_from_unmeasured_costs`; C10 `test_metrics_preserve_estimands_support_and_protocol_strata`; C11 `test_discrepancies_use_compatible_units_and_keep_full_precision`; C12 `test_figures_render_from_validated_inputs_without_fallbacks`; C13 `test_figure_3_panel_identity_uses_variant_provenance`; C14 `test_qualitative_and_statistical_selections_are_recorded`; C15 `test_render_and_case_sensitive_path_failures_block_completion`; C16 `test_both_variants_consume_the_same_package_results`; C17 `test_methods_and_results_follow_observed_protocols`; C18 `test_inconclusive_evidence_limits_genome_and_causal_claims`; C19 `test_optional_c05_state_limits_claims_without_blocking_package`; C20 `test_historical_values_remain_unreproduced_and_noncomplete`; C21 `test_review_notes_are_preserved_and_submission_is_clean`; C22 `test_editorial_review_status_is_explicit_and_independent_of_compile`; C23 `test_cli_and_notebook_use_isolated_run_namespace`; C24 `test_master_notebook_bootstraps_src_without_heavy_side_effects`; C25 both `test_compile_artifacts_are_isolated_and_historical_sources_unchanged` and `test_complete_fixture_assembly_renders_and_compiles_in_reserved_package`; C26 `test_manifest_records_inputs_outputs_versions_and_gates`; C27 `test_synthetic_run_and_final_report_do_not_claim_reproduction`.

| Check | Claim | Proof run | Evidence | Result |
| --- | --- | --- | --- | --- |
| C1 | 17-item index | suite above | `tests/test_c06_publication_package.py:155` - `self.assertEqual(set(item["item_id"] for item in self.config["items"]), set(ITEM_IDS))` | PASS |
| C2 | missing/corrupt/unproven inputs with field/path diagnostics | named selector ran | `:181-194` - status INVALID, item/path/expected diagnostic, unreadable artifact path | PASS |
| C3 | identity/composition mismatch including population | named selector ran | `:203-217` - class order and `dataset_sha256`, `population_id`, split, seed, noise, variants, checkpoint mismatch all INVALID | PASS |
| C4 | strict fail-closed incl. transitive synthetic | named selector ran | `:224-247` - strict exceptions for missing/hash/smoke/PARTIAL/empty inputs and `synthetic dependency` | PASS |
| C5 | T11 semantics | named selector ran | `:256` - `self.assertIn("transformation mismatch", t11["diagnostic"])` | PASS |
| C6 | selected/transitive synthetic logic | named selector ran | `:262-275` - unselected smoke ignored; selected child yields BLOCKED and matrix | PASS |
| C7 | partial package diagnostic | named selector ran | `:282-286` - BLOCKED, 17 rows, T3 MISSING, no tables dir | PASS |
| C8 | tables derived and history separate | named selector ran | `:292-302` - 12 exports per format, generated T1 equals selected input, historical label separated | PASS |
| C9 | structural costs vs unmeasured | named selector ran | `:326-338` - DERIVED_VERIFIED, no latency/FLOPs, N reason, derivation keys | PASS |
| C10 | metric units/support/strata | named selector ran | `:347-358` - N, support, denominator, estimand, unit, unsupported rule and CI/calibration populations | PASS |
| C11 | full precision/compatibility | named selector ran | `:363-368` - `0.001` pp and incompatible comparison has no difference | PASS |
| C12 | figures source data and no fallback | named selector ran | `:380-388` - PNG/PDF signatures and render manifest ties each output to selected artifact hash and plot config | PASS |
| C13 | F3 panel provenance | named selector ran | `:393-400` - d ResNet50_F/e ConvNeXt_F, checkpoint IDs; wrong panel identity diagnostic | PASS |
| C14 | selection/interpretation | named selector ran | `:408-414` - IDs/levels/sampling/seed/filters exact; UMAP qualitative, triplets exploratory | PASS |
| C15 | render/case errors block | named selector ran | `:422-432` - case mismatch, missing columns exception, FAILED/local BLOCKED/error artifact | PASS |
| C16 | shared manuscript result source | named selector ran | `:463-471` - 12 table inputs, old numeric text absent, bodies match, embedded tabular absent | PASS |
| C17 | methods describe observed variants and populations | named selector ran | `:483-485` - protocol phrases plus `fixture_model=` and named CI/calibration populations. **Does not assert selected variant set/selection rule against evidence**; `fixture_model=` is one generic fixture label. | FAIL |
| C18 | inconclusive genome/causal claims qualified | named selector ran | `:492-495` - unseen-genome limitation in abstract/conclusions; unresolved discussion; causal phrase absent | PASS |
| C19 | C05 optional | named selector ran | `:517-519` - NOT_RUN manuscript plus strict complete fixture assembly has empirical PASS | PASS |
| C20 | historical values not promoted | named selector ran | `:524-529` - historical status, computed role, diagnostic not empirical complete | PASS |
| C21 | review/submission notes | named selector ran | `:535-536` - exact 56 note macros, none in submission | PASS |
| C22 | editorial pending independent of compile | named selector ran | `:541-542` - references/affiliations/wording PENDING, compile not approval | PASS |
| C23 | CLIs/notebook share isolated gate; protect sources | named selector ran | `:548-564` - reservation collision and three CLI modules share assembler/block on missing evidence. It does **not** execute notebook gate parity or assert no writes to selected inputs/historical PDFs; those are not covered by C24's bootstrap-only static checks. | FAIL |
| C24 | notebook bootstrap/no heavy side effects | named selector ran | `:570-577` - src insertion precedes package import; heavy calls absent; assembler/reservation present | PASS |
| C25 | compile outputs/references/source hashes | both named selectors ran | `:587-607` - references validated, PDF/log created, historical TeX hash stable; full fixture assembly creates both PDFs/logs (`:656-663`) | PASS |
| C26 | manifest provenance completeness | named selector ran | `:678-692` - command, commit/dirty, config hash, inputs, history, run ID, pandas, output hash and gates. `:713-716` checks input manifest hashes all exist and only **some** source input has a hash, not every source/checkpoint hash; complete version inventory and source/output correspondence are not proven. | FAIL |
| C27 | synthetic status/final report completeness | named selector ran | `:702-723` - synthetic, blocked gate, command/interface, manifest path, discrepancy wording and no tables. No assertions that report identifies config/outputs changed or unavailable dependency inventory exactly. | FAIL |

The C06 `Coverage` join was recomputed against config and tests: all 17 configured publication IDs have indexing and relevant transformation/status proof; all three Landing doors have checks; six statuses and four gates have evidence; CLI/notebook surfaces are enumerated. No uncovered member was found in the listed sets. `Relations (0)` and `HTTP/API routes (0)` are justified by the plan. No `Test policy` rows are present under the approved `light` profile. No faults were injected because the approved profile is light.

## Swept existing

| Dimension | Round 4 review |
| --- | --- |
| validation | C2-C5 and C8-C15 have direct valid/invalid assertions; C17/C26 retain scope gaps above. |
| failure modes | Missing, malformed, unreadable, hash, synthetic/transitive, partial, rendering and compile paths are asserted. |
| idempotency | C23 collision refusal is asserted for reservation helper; not a concurrency/race proof. |
| authorization | n/a remains justified for local pipeline. |
| concurrency | Exclusive reservation collision is covered; race safety is not asserted, same limitation as prior round. |
| data lifecycle | Selected source/manifest hash fields and historical TeX preservation are now asserted; completeness gaps remain C23/C26. |
| dependency failure | Missing inputs, transform/render errors and compile output paths are exercised; missing external dependency reporting remains C27 gap. |
| state transitions | BLOCKED/FAILED/DIAGNOSTIC, C05 NOT_RUN with empirical PASS and fixture assembly are exercised. |
| observability | Diagnostics and matrix/status output are asserted; missing selected-variant reporting and report dependency inventory remain. |

## Gate

`python3 .agent/skills/tlc-spec-lean/scripts/validate_verification.py c06-regeneracao-e-reconciliacao-estrita` — exit 1, expected because the report verdict is FAIL; after correcting the required `Profile:` format, the gate reported no format warnings.

### Ranked gaps

1. C23 - prove notebook/CLI gate parity and non-overwrite of selected inputs and historical PDFs (`tests/test_c06_publication_package.py:548-564`).
2. C26 - assert hashes for every source/checkpoint, complete relevant version inventory, and source/output correspondence (`:713-716`).
3. C27 - assert report records configuration, changed outputs, discrepancies and unavailable dependencies (`:717-723`).
4. C17 - assert actual selected variant set and selection rule against evidence, not only a generic fixture label (`:483-485`).

No results_reproduction output directory was accessed or changed by this verification.
