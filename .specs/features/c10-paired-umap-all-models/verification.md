# C10 paired UMAP verification

**Verdict**: PASS
**Profile**: light
**Diff range**: df795a263c9032b44e5bb3ec541ac87a3e346206..38518c77695343e8c44655f81012dbfd9c165a43
**Round**: 2 - scoped
**Verifier**: independent sub-agent (author != verifier)

17/17 checks proven. Round 1 found a C2 assertion gap; the test-only fix snapshots and compares labels and IDs for all five variants, resolving that gap without changing implementation or frozen checks.

## Proof runs

Rerun independently at `38518c77695343e8c44655f81012dbfd9c165a43`. Round 1 proofs also passed at `206f7f738c19074fcb2025a0727d736dcdd3a76d`; their initial C2 evidence gap is resolved below:

- R1: `REMOTE_SERVER=False OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLCONFIGDIR=/tmp PYTHONPATH=src .venv/bin/python3 -m unittest -v tests.test_c01_latent_audit tests.test_c07_latent_memory tests.test_c08_knn_workspace tests.test_latent_space tests.test_c09_latent_resume tests.test_c10_paired_umap` — exit 0; 93 tests, including all 14 C10 methods; verbose output `/tmp/c10-verifier-round2-regression.log`.
- R2: `REMOTE_SERVER=False OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLCONFIGDIR=/tmp PYTHONPATH=src .venv/bin/python3 -m unittest -v tests.test_n04_taxonomy_and_latent_space -k notebook_06` — exit 0; 9 tests; verbose output `/tmp/c10-verifier-round2-notebook.log`.

All 13 distinct named proof methods exist and ran in R1; one additional fitting-error method also ran. `rg -n -A 22 '^    def test_' tests/test_c10_paired_umap.py` located test bodies and assertions. Full feature diff, approved plan/checks, repository instructions, README, dataset manifest/configuration, historical notebook cells 6/7/10/11, manuscript latent method/Figure 2 caption, and C09 cache code were read. Assertions below belong to the newly added C10 tests.

## Checks

All proof entries refer to R1, exit 0, at the header commit; method names are exact unittest method names.

| Check | Claim | Proof run | Evidence | Result |
| --- | --- | --- | --- | --- |
| C1 | Limits 100/10; invalid values rejected before runtime/projection | `test_configuration_boundary` | `tests/test_c10_paired_umap.py:94` — `self.assertEqual(scope[name], expected)` for adjacent literal defaults 100/10; :98 `self.assertRaises(ValueError)`; :100 `setup.assert_not_called()`; :109 override equals `(17, 4)` | PASS |
| C2 | Missing lineage counted; all input labels/IDs unchanged | `test_coverage_and_balanced_quotas` | `tests/test_c10_paired_umap.py:118` — `self.assertEqual(sample.audit['excluded_missing_genus'], 2)`; :128 `np.testing.assert_array_equal(self.labels[name], all_labels_before[name])`; :129 `np.testing.assert_array_equal(self.ids[name], all_ids_before[name])`, over all `MODEL_NAMES` after both selections | PASS |
| C3 | Deterministic bounded balanced selection without replacement | `test_coverage_and_balanced_quotas` | `tests/test_c10_paired_umap.py:119` — `self.assertEqual(sample.frame['Genus'].value_counts().to_dict(), {'Beta': 5, 'Gamma': 5, 'Alpha': 2})`; :121 unique IDs; :122 `assert_frame_equal(sample.frame, self.sample().frame)`; :125–126 limits 7/2 exercise configurable bound; C1 establishes defaults | PASS |
| C4 | Storage permutation preserves sample and selected feature alignment | `test_permuted_models_project_identical_rows` | `tests/test_c10_paired_umap.py:138` — `pd.testing.assert_frame_equal(sample.frame, original)`; :146 `np.testing.assert_array_equal(features, self.features[name][expected_positions])` | PASS |
| C5 | Missing/duplicate/changed-label ID fails before fitting | `test_identity_errors_before_projection` | `tests/test_c10_paired_umap.py:161` — `self.assertRaises(ValueError)` over adjacent cases `('missing', 'duplicate', 'label')`; :163 `self.assertEqual(FakeUMAP.calls, [])` | PASS |
| C6 | Five projections share ID/label sequence | `test_permuted_models_project_identical_rows` | `tests/test_c10_paired_umap.py:140` — `self.assertEqual(set(result['projections']), set(MODEL_NAMES))`; :141 call count 5; :143–144 frame IDs and labels equal the shared original selection | PASS |
| C7 | Shared UMAP parameters, small-sample neighbor adjustment | `test_projection_parameters_and_small_sample` | `tests/test_c10_paired_umap.py:170` — `self.assertEqual(params, {'n_neighbors': min(15, n - 1), 'min_dist': 0.1, 'n_components': 2, 'metric': 'cosine', 'random_state': 42})` for 5/20 samples | PASS |
| C8 | Fewer than three eligible samples refuses complete publication | `test_insufficient_sample` | `tests/test_c10_paired_umap.py:177` — `self.assertRaisesRegex(ValueError, 'at least 3')`; :180 `self.assertFalse(list(self.root.rglob('umap_comparison.json')))` | PASS |
| C9 | Selected-row feature access; quantitative contracts preserved | `test_selected_feature_access_and_quantitative_contracts` | `tests/test_c10_paired_umap.py:185` — `self.assertTrue(all(len(values.accesses) == 1 for values in self.features.values()))`; bounded feature wrapper rejects array materialization; :187–189 assert `k=5`, 200/5000 Silhouette and `features=raw_feats`. Full diff leaves quantitative implementation unchanged; R1/R2 regressions passed | PASS |
| C10 | Five identified CSVs, selection, hashes and provenance | `test_outputs_provenance_and_panel` | `tests/test_c10_paired_umap.py:203` — selected counts equal `{'Alpha': 2, 'Beta': 5, 'Gamma': 5}`; :204 count 5; :206 output SHA equals recorded SHA; :208 required ID/label/Genus/method columns; :209 IDs equal selection; :211 selection/panel SHA match. Parameters directly asserted by C7 | PASS |
| C11 | Five panels with shared legend/colors and qualitative warning | `test_outputs_provenance_and_panel` | `tests/test_c10_paired_umap.py:213` — `self.assertEqual(len(fig.axes), 5)`; :214 legend equals `['Alpha', 'Beta', 'Gamma']`; :215 qualitative/independent warning; :221 `np.testing.assert_array_equal(color, colors[0])` across every panel | PASS |
| C12 | PCA fallback explicitly labeled | `test_pca_fallback_is_explicit` | `tests/test_c10_paired_umap.py:232` — `self.assertEqual(result['metadata']['projection_method'], 'PCA')`; :234 frame method set `{'PCA'}`; :236–237 PCA in titles and axes | PASS |
| C13 | Compatible reuse restores five stages without fits | `test_resume_and_variant_corruption` | `tests/test_c10_paired_umap.py:244` — `self.assertEqual(FakeUMAP.calls, [])`; :258 selection change refits 5; :265 embedding hash change refits 2 including corrupt stage; :270 backend change calls exporter 5 times. Dependency equality/hash enforcement inspected in `src/viral_cnn/latent_resume.py:245` | PASS |
| C14 | Corruption refits only affected variant and preserves sources | `test_resume_and_variant_corruption` | `tests/test_c10_paired_umap.py:251` — `self.assertEqual(len(FakeUMAP.calls), 1)`; :252 source byte equality; :254 recalculated component equals `[f'umap_{chosen}']` | PASS |
| C15 | Interrupted writes never publish complete comparison; partial stages reusable | `test_interrupted_publication` | `tests/test_c10_paired_umap.py:282` — no comparison JSON; :286 no fitting after final-publication failure; :295 no JSON after second CSV failure; :296 exactly one completed stage; :299 four fits on resume | PASS |
| C16 | Full notebook smoke exports five projections/panel, records protocol, reuses | `test_notebook_smoke_resume` | `tests/test_c10_paired_umap.py:332` — five projections; :333 PNG exists; :335 `self.assertEqual(manifest['configuration']['umap_protocol'], 'c10_paired_umap_v1')`; :336 five variants. :326 fitting exporter blocked on second complete notebook execution | PASS |
| C17 | Documented historic difference, global invalidation and CPU limitations | `test_documented_limits` | `tests/test_c10_paired_umap.py:343` — `self.assertIn(token, doc)` over adjacent explicit tokens including `primeiras 100`, `hash global`, `sintética`, `CPU`, `remota`; prose read directly in `docs/paired_umap.md:17`, :35, :37 | PASS |

## Resolved gap and scoped review

Round 1 at `206f7f738c19074fcb2025a0727d736dcdd3a76d`: C2 failed because only reference labels were compared; all IDs and the other label arrays lacked preservation assertions. The initial gate returned exit 1, one error and zero warnings.

Round 2 at `38518c77695343e8c44655f81012dbfd9c165a43`: inspected the complete fix diff (only five added lines in the named test), refreshed every touched test citation and reran both complete proof targets. Snapshots at `tests/test_c10_paired_umap.py:113`–114 now cover both dictionaries; assertions at :128–129 compare every model's inputs after ordinary and limited selections. Existing assertions remain. No unresolved gaps.

## Level, sampling and scientific review

Implementation/scientific review carried from `206f7f738c19074fcb2025a0727d736dcdd3a76d`; only a test changed afterward. Proof level and C2 assertion scope reverified at `38518c77695343e8c44655f81012dbfd9c165a43`. Module-level fixtures cover imbalanced genera, missing lineage, permutations, identity errors, small samples, methods and failure/reuse states; notebook smoke reaches the actual edited notebook boundary. Genus mapping uses model-label keys; selection canonicalizes operational IDs and validates every selected ID and Species label across all variants before fitting. Projection copies only selected rows. UMAP parameters are verified through an estimator stub; PCA executes the real historical exporter. An extra passing test confirms fitting-time ImportError propagates without a complete JSON. No claim of numeric UMAP verification is justified locally.

Cache identity binds variant, source embedding hash, exact selection, configuration, method, module/exporter hashes and environment. C09 checks hashes before restoring and preserves source files. Atomic CSV/panel publication precedes the complete comparison JSON. Existing unique run namespaces remain enforced by the notebook/C09 infrastructure (`reserve_run_directory` in notebook initialization); Swept's concurrency-existing claim is present. Other Swept entries reference proven checks or explicit approved n/a decisions.

Historical `eval/latent.py`, manuscript and historical notebooks are unchanged. The approved C08 static expectation replacement tracks the new sampling protocol while preserving the 100 default and Silhouette constraints. New coordinates and sampled population are exploratory `c10_paired_umap_v1`, not exact reproduction of published Figure 2. The global C09 hash can cause earlier non-embedding stages to rerun, as approved and documented.

Profile light: coverage join recomputation and fault injection were not performed. No Test policy section exists in checks.md. Canonical remote ICTV data/checkpoints and `/home/mlima/workspace` are unavailable; local UMAP is absent. This verification establishes synthetic CPU contracts, not empirical regeneration or unseen-genome generalization.

## Gate

`python3 .agent/skills/tlc-spec-lean/scripts/validate_verification.py c10-paired-umap-all-models` — exit 0, zero errors and zero warnings. R1/R2 independently passed at final HEAD: 93 + 9 = 102 tests, zero failed.
