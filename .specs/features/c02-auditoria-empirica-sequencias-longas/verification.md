## Verification: c02-auditoria-empirica-sequencias-longas - PASS

Verifier: independent fresh sub-agent
Round: 1 - full
Profile: light
Diff: `b85fbcfe907333e26f607ee7e19a80e3d5d9e480..70cc3070eec547733665582ecb5758369c765c65`
Verified at: `70cc3070eec547733665582ecb5758369c765c65`

**Checks**: 15/15 proven with located assertions. All named proof tests were found using `rg -n` and individually appeared as passing in the single batched module run.

**Binding sources**: Not required by the light profile. Read plan-listed project `AGENTS.md`, `src/viral_cnn/AGENTS.md`, `.specs/STATE.md`, `ROADMAP_CORRECOES_REPRODUCAO.md`, AD-020, the C02 long-sequence audit documents, and the historical/reproduction source inventory called out in the plan. No contradiction found between the check claims and concrete decisions relevant to these checks. Historical inference claims are expressly bounded in the updated documents; no empirical reproduction is claimed.

**Proof command**: `.venv/bin/python -m unittest -v tests.test_c02_long_audit` — exit 0; 15 tests passed, 0 failed. The output listed each C1–C15 named test separately. This is the complete test module containing every proof, batched once at HEAD. It is synthetic/contract evidence only, not a remote empirical execution.

## Checks

| Check | Claim | Proof run | Evidence | Result |
| --- | --- | --- | --- | --- |
| C1 | Long-input observed support/provenance; unknown genome identity remains unknown | Batched module above — PASS | `tests/test_c02_long_audit.py:57-63` — asserts observed eligible count, length max, taxon counts, and `{"status": "unknown", "genome_count": None}`; `:66-67` asserts source SHA-256 | PASS |
| C2 | Sampling and overlap record deterministic provenance and distinct denominators | Batched module above — PASS | `tests/test_c02_long_audit.py:72-75` — asserts realized count, seed and selected IDs; `:77-80` asserts query denominator, queries without valid k-mers and canonicalization | PASS |
| C3 | Mapping requires an identified canonical ordering and rejects ambiguous/unmapped labels | Batched module above — PASS | `tests/test_c02_long_audit.py:99-101` — `assertEqual(labels, [1, 0])` and preserves source IDs; `:102-105` expects errors containing `ambiguous` and `unmapped` | PASS |
| C4 | Paired comparison requires five registered variants and rejects identity/alignment defects | Batched module above — PASS | `tests/test_c02_long_audit.py:112-114` asserts exact registered variant set and missing-variant rejection; `:116-127` asserts duplicate, missing, truth/condition/variant mismatch failures | PASS |
| C5 | Table metrics use observed N; skipped prediction remains an error globally | Batched module above — PASS | `tests/test_c02_long_audit.py:134-138` — asserts `N == 2`, `covered == 1`, `correct == 1`, `accuracy == 0.5`, and conditional accuracy `1.0` | PASS |
| C6 | Prediction artifact contract and checkpoint hash identity | Batched module above — PASS | `tests/test_c02_long_audit.py:145-150` — asserts SHA-256 length 64, exact variant set, and required prediction columns including protocol | PASS |
| C7 | Window metadata and singleton FP32 equivalence | Batched module above — PASS | `tests/test_c02_long_audit.py:155-159` — asserts one window/logit and maximum equivalence difference `<= 1e-5` | PASS |
| C8 | Strata are complete, include unknown metadata, and empty strata have unavailable metrics | Batched module above — PASS | `tests/test_c02_long_audit.py:164-171` — asserts total counts, empty `N=0`/None accuracy, boundary bins, unknown ambiguity and taxon | PASS |
| C9 | Query weighted and taxon macro composition metrics use the same identified rows | Batched module above — PASS | `tests/test_c02_long_audit.py:176-180` — asserts N, both weighting results, percentage-point difference and no genome attribution | PASS |
| C10 | Window diagnostics measure disagreement/influence and mark missing logits inconclusive | Batched module above — PASS | `tests/test_c02_long_audit.py:183-188` — asserts aggregation disagreement, leave-one-window-out flips, and `INCONCLUSIVE` for missing/smoke logits | PASS |
| C11 | Calibration uses 15-bin ECE/NLL from softmax of mean pre-softmax logits | Batched module above — PASS | `tests/test_c02_long_audit.py:192-195` — asserts 15 bins, N, 15 bin metrics, and exact probability-source description | PASS |
| C12 | Synthetic fixtures cannot claim empirically supported findings | Batched module above — PASS | `tests/test_c02_long_audit.py:198-202` — expects ValueError for synthetic `SUPPORTED` and confirms `INCONCLUSIVE` output | PASS |
| C13 | Reproduction notebooks use exclusive run manifests and artifact logging | Batched module above — PASS | `tests/test_c02_long_audit.py:210-218` — asserts reserved run ID/path, collision failure, manifest writer and logging hooks in the three notebooks | PASS |
| C14 | Smoke evidence is separated; empirical mode blocks missing inputs | Batched module above — PASS | `tests/test_c02_long_audit.py:223-226` — asserts smoke manifest is rejected and empirical preflight is `BLOCKED` with a blocked requirement | PASS |
| C15 | Evidence report records AD-010 boundary, historical context, and lack of empirical execution | Batched module above — PASS | `tests/test_c02_long_audit.py:229-236` — asserts AD-020/report existence, AD-010 mention, literal `No C02 remote empirical execution`, historical and unknown references | PASS |

## Level and sampling judgment

The checks are contract-level and their located assertions exercise the changed C02 code or notebooks. The fixtures are synthetic, which is consistent with the claims as worded and the plan's stated remote-data limitation; C15 and the audit report explicitly avoid representing these tests as empirical reproduction. No claim here proves the paper's Table 12 values on real checkpoints/data.

## Swept existing

| Existing constraint | Evidence | Result |
| --- | --- | --- |
| Artifact path logging (`LOG_PATH`) | `src/viral_cnn/AGENTS.md` §5 requires `log_artifact_path`; C13 checks it in long/02 and long/03 notebooks at `tests/test_c02_long_audit.py:218` | Present |
| Exclusive C00 reserved-run capability | C13 exercises `reserve_c02_long_run`, confirms run path and duplicate reservation raises at `tests/test_c02_long_audit.py:210-213` | Present |
| N/A authorization/HTTP and visual surfaces | Plan explicitly says local-only/no HTTP service and no new visual UI | N/A as declared |

## Gate

`.venv/bin/python -m unittest -v tests.test_c02_long_audit` — 15 passed, 0 failed.
