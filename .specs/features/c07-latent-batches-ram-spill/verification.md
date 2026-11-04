# C07 verification

**Verdict**: PASS
**Profile**: light
**Diff range**: 6c07806ee27ac596bcef1ce3662907f5c9c13f3a..94df6f3871eeb8024359d8ac2759765e15975f86
**Round**: 2 - scoped
**Verifier**: independent sub-agent (author != verifier)

## Binding sources — carried from 6c07806ee27ac596bcef1ce3662907f5c9c13f3a

The prior verifier opened `AGENTS.md`, `src/viral_cnn/AGENTS.md`, `.specs/STATE.md` (AD-012, AD-017, AD-018, AD-019), C01 `plan.md`, `src/viral_cnn/eval/latent.py`, and Notebook 06. It found no contradiction or uncovered binding decision. The fix diff changes only C07 test code, so it does not touch an interface or alter the basis for that comparison. The source-level statements below were re-read at HEAD; no changed implementation source or notebook was introduced by this fix.

| Source | Opened / scope | Contradiction | Uncovered |
| --- | --- | --- | --- |
| `AGENTS.md`, `src/viral_cnn/AGENTS.md`, `.specs/STATE.md` AD-012/017/018/019 | carried from 6c07806ee27ac596bcef1ce3662907f5c9c13f3a | none | none |
| C01 `plan.md` | carried from 6c07806ee27ac596bcef1ce3662907f5c9c13f3a | none | none |
| `src/viral_cnn/eval/latent.py`, Notebook 06 | carried from 6c07806ee27ac596bcef1ce3662907f5c9c13f3a; unchanged in fix diff and re-read at HEAD | none | none |

## Checks — verified at 94df6f3871eeb8024359d8ac2759765e15975f86

One invocation ran the complete target module at HEAD: `.venv/bin/python3 -m unittest tests.test_c07_latent_memory -v` — 9 tests passed, 0 failed. Each named test below was individually present in that output as `... ok`; each body was located with `rg -n -C 5` in `tests/test_c07_latent_memory.py`. The fix diff touches this test file, so all citations are refreshed. These are synthetic CPU tests, not empirical reproduction.

| Check | Claim | Proof run | Evidence | Result |
| --- | --- | --- | --- | --- |
| C1 | 30% RAM default; MiB override, zero and invalid values | `test_budget_defaults_to_available_ram_fraction_and_validates_overrides` — PASS | `tests/test_c07_latent_memory.py:81-89` — `self.assertEqual(LatentMemoryBudget.from_environment().budget_bytes, 3_000_000)`, override equals `2 * 1024 * 1024`, zero equals `0`, and invalid strings raise `ValueError` | PASS |
| C2 | Spill roundtrip preserves values, dtype, shape, unique paths and owned cleanup | `test_spill_roundtrip_exclusive_paths_and_owned_cleanup` — PASS | `tests/test_c07_latent_memory.py:103-117` — equality checks for all three arrays; explicit `dtype` and `shape` assertions for `a`, `b`, and `c`; `self.assertNotEqual(left.spill_paths, right.spill_paths)`; after `left.close()`, `other.exists()` and `right.spill_paths[0].exists()` | PASS |
| C3 | Extraction of 1,031 rows uses batches ≤256 and aligns ordered outputs including tail 7 | `test_extraction_batches_preserve_order_alignment_and_spill` — PASS | `tests/test_c07_latent_memory.py:120-146` — observed sizes equal `[256, 256, 256, 256, 7]`, all sizes ≤256, and feature/label/row-ID output assertions preserve expected order | PASS |
| C4 | Artifact save/load streams in blocks and preserves output and validated manifest without a second full managed copy | `test_embedding_artifact_streaming_roundtrip` — PASS | `tests/test_c07_latent_memory.py:150-178` — `guarded_features.batch_sizes == [128] * 8 + [7]`; output equals saved reference; feature/label/ID dtype and shape assertions; all loaded arrays are `np.memmap`; `manager.ram_bytes == 0`; spill-byte total equals the three array sizes; manifest artifact hash matches and spill count is positive | PASS |
| C5 | Probe validation batches ≤256, global accuracy drives selection, earliest tie wins and selected weights are sealed by hash | `test_probe_validation_batches_and_preserves_global_selection` — PASS | `tests/test_c07_latent_memory.py:180-205` — max observed batch ≤256; declared selection rule; selected epoch equals earliest max-accuracy epoch; direct whole-validation accuracy equals history accuracy; `result["state"]` and `_audit_state` are `SEALED`; `weights_sha256 == _sealed_sha256` and hash length is 64 | PASS |
| C6 | Batched test metrics match the whole-set reference | `test_probe_test_evaluation_batches_preserve_metrics` — PASS | `tests/test_c07_latent_memory.py:207-234` — batch bound, `n_test_samples == 1031`, support 2, Top-1 accuracy and Macro-F1 equal independently computed same-set references | PASS |
| C7 | Explicit logits remain spill-managed; notebook uses Top-1 and compact row-ID hash | `test_logits_are_spill_managed_and_notebook_keeps_top1` — PASS | `tests/test_c07_latent_memory.py:236-260` — notebook source asserts `return_predictions=True`, native predictions, absence of full-logit request, and `"Test_Row_IDs_SHA256": eval_res["row_ids_sha256"]`; explicit logits have expected shape and are `np.memmap`; evaluation hash equals SHA-256 of IDs and `row_ids` is absent | PASS |
| C8 | Invalid setup/write failure reports involved path and partial artifact is rejected | `test_invalid_budget_and_spill_failure_do_not_publish_partial_artifact` — PASS | `tests/test_c07_latent_memory.py:262-277` — negative budget raises; propagated `OSError` includes the spill path; no Parquet is published by failed allocation; loading `partial.parquet` raises `ValueError` matching `incomplete.*manifest.*partial.parquet` | PASS |
| C9 | Notebook has batch default/override, budget default/override, and records effective settings | `test_notebook_configures_and_records_memory_budget` — PASS | `tests/test_c07_latent_memory.py:279-321` — AST-executed assignment asserts batch 256 and override 64; budget asserts 30% and 2 MiB override; manifest configuration expressions are checked for batch, RAM budget, resolved spill path and storage report; AST confirms `LatentMemoryBudget.from_environment`; log prefix `[LATENT_MEMORY] batch_size=` is asserted | PASS |

## Swept existing — re-read at 94df6f3871eeb8024359d8ac2759765e15975f86

| Constraint | Evidence | Result |
| --- | --- | --- |
| AD-012 pre-classifier features and C01 probe provenance | `latent.py` keeps the `fc` input capture and split/partition validation in `fit_select_and_seal`; source unchanged by fix | present |
| AD-017 local CPU/offline behavior | all nine named proofs use synthetic CPU data and completed locally | present for local proofs; remote path remains untested |
| AD-018 metric counters avoid per-batch host sync | evaluator accumulates device counters and transfers after its loop; source unchanged by fix | present |
| `LOG_PATH` for spill and artifact paths | spill allocation uses `log_artifact_path`; current unittest output includes `[LOG_PATH]` paths | present |
| Notebook effective memory settings | Notebook 06 configures `LatentMemoryBudget` and writes batch, budget, spill path and storage report into manifest configuration; C9 now asserts those AST expressions | present and asserted |
| C01 seal/hash and evaluate-once semantics | C5 now asserts sealed states and matching 64-character weight hash; C7 asserts compact row-ID hash and omitted IDs | present and asserted |

Profile `light` does not require fault injection, Coverage recomputation, Test policy verdicts, or a user-facing flow. No UI applies. The plan's enumerated sets were swept against C1–C9; the check table and claims leave no set member without an assigned proof. No code, checks, or lessons were changed by this verifier.

## Environment and scientific status

All nine checks are synthetic CPU proofs over local fixtures, including 1,031-row artifacts. The remote cluster, real ICTV partitions and canonical Species checkpoints were unavailable. No scientific result was reproduced; GPU numerical equivalence and empirical peak-memory behavior remain pending remote validation. The test process emitted a Matplotlib cache fallback warning and a CUDA driver compatibility warning; neither prevented the CPU suite from passing.

## Gate

`python3 .agent/skills/tlc-spec-lean/scripts/validate_verification.py c07-latent-batches-ram-spill` — 0 errors, 0 warnings.
