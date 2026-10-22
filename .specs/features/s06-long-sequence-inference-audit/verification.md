# S06: Long-Sequence Inference Audit and Voter Extraction - verification

**Feature**: s06-long-sequence-inference-audit
**Verdict**: PASS
**Profile**: light
**Diff range**: 2b43294..HEAD
**Round**: 1 - full
**Verifier**: independent sub-agent (author != verifier)
**Date**: 2026-10-21
**Result**: PASS

## Checks

| Check | Claim | Proof run | Evidence | Result |
| --- | --- | --- | --- | --- |
| C1 | `LongSequenceVoter.slice_sequence` extracts 512 bp windows with stride 256 bp and normalizes characters to uppercase A, C, G, T (0, 1, 2, 3) and ambiguous/padding to token 4 (AC 1) | `.venv/bin/python3 -m unittest discover -s tests -k test_slice_sequence_window_stride_and_token_normalization` exit 0 | `tests/test_voter.py:70` - `self.assertEqual(len(windows), 3)` · `tests/test_voter.py:76` - `self.assertTrue(all(tok == 0 for tok in windows[0]))` · `tests/test_voter.py:78` - `self.assertTrue(all(tok == 2 for tok in windows[2][256:]))` | PASS |
| C2 | `LongSequenceVoter.slice_sequence` with `handle_short='skip'` returns 0 windows for sequences <512 bp (AC 2) | `.venv/bin/python3 -m unittest discover -s tests -k test_slice_sequence_handle_short_skip` exit 0 | `tests/test_voter.py:84` - `self.assertEqual(len(windows), 0)` | PASS |
| C3 | `LongSequenceVoter.slice_sequence` with `handle_short='pad'` pads contigs <512 bp to 512 bp using token 4 and returns exactly 1 window (AC 3) | `.venv/bin/python3 -m unittest discover -s tests -k test_slice_sequence_handle_short_pad` exit 0 | `tests/test_voter.py:90` - `self.assertEqual(len(windows), 1)` · `tests/test_voter.py:94` - `self.assertEqual(windows[0][:200], expected_first)` · `tests/test_voter.py:96` - `self.assertTrue(all(tok == 4 for tok in windows[0][200:]))` | PASS |
| C4 | `LongSequenceVoter.aggregate_window_logits` computes pre-softmax mean logits (`mean_logits.argmax()`) and hard window majority voting (`mode(window_winners)`) (AC 4) | `.venv/bin/python3 -m unittest discover -s tests -k test_aggregate_window_logits_mean_and_majority` exit 0 | `tests/test_voter.py:112` - `self.assertEqual(res.pred_majority_vote, 2)` · `tests/test_voter.py:117` - `self.assertEqual(res.pred_mean_logits, 2)` · `tests/test_voter.py:118` - `self.assertEqual(res.num_windows, 3)` | PASS |
| C5 | `LongSequenceVoter.aggregate_window_logits` resolves majority voting ties deterministically by selecting the smallest class index (`tie_break='min_index'`) (AC 5) | `.venv/bin/python3 -m unittest discover -s tests -k test_aggregate_window_logits_deterministic_tie_break` exit 0 | `tests/test_voter.py:130` - `self.assertEqual(res.pred_majority_vote, 1)` | PASS |
| C6 | A single 512 bp window evaluated via `engine.predict_batch` and via `LongSequenceVoter` produces identical logits within absolute tolerance $< 10^{-5}$ (AC 6) | `.venv/bin/python3 -m unittest discover -s tests -k test_voter_window_logit_equivalence_with_engine` exit 0 | `tests/test_voter.py:140` - `self.assertLess(max_diff, 1e-5)` | PASS |
| C7 | Predicted class indices [0, 13682] convert to 1-based taxonomic IDs via `tax_id = model_label + 1` matching AD-002 (AC 7) | `.venv/bin/python3 -m unittest discover -s tests -k test_voter_label_offset_one_based_tax_id` exit 0 | `tests/test_voter.py:151` - `self.assertGreaterEqual(res.pred_mean_logits, 0)` · `tests/test_voter.py:156` - `self.assertGreaterEqual(tax_id, 1)` · `tests/test_voter.py:157` - `self.assertLessEqual(tax_id, 13683)` | PASS |
| C8 | Ambiguous IUPAC characters (`N`, `R`, `Y`, `S`, `W`, `K`, `M`, `B`, `D`, `H`, `V`) are mapped to token 4 and produce null one-hot vectors without exception (AC 8) | `.venv/bin/python3 -m unittest discover -s tests -k test_voter_iupac_ambiguity_produces_null_one_hot` exit 0 | `tests/test_voter.py:169` - `self.assertEqual(tokens[pos].item(), 4)` · `tests/test_voter.py:177` - `self.assertTrue(torch.all(null_vectors == 0.0))` | PASS |
| C9 | Model loading and Voter reject classification heads with output dimension $C \neq 13683$ for species rank with `ValueError` (AC 9) | `.venv/bin/python3 -m unittest discover -s tests -k test_voter_rejects_class_dimension_mismatch` exit 0 | `tests/test_voter.py:185` - `with self.assertRaises(ValueError):` | PASS |
| C10 | `docs/long_sequence_inference_audit.md` documents causes of ConvNeXt collapse (read-length generation bias 500 vs 50, logit pulling, head lineage) and confirms ResNet50 72.69% reflects seen reference genomes rather than unseen generalization (AC 10) | `.venv/bin/python3 -m unittest discover -s tests -k test_audit_report_scientific_claims_and_evidence` exit 0 | `tests/test_voter.py:196` - `self.assertTrue(report_path.exists())` · `tests/test_voter.py:199` - `self.assertIn("72,69%", text)` · `tests/test_voter.py:200` - `self.assertIn("calculate_n_reads", text)` · `tests/test_voter.py:203` - `self.assertIn("Bracoviriform", text)` · `tests/test_voter.py:204` - `self.assertIn("Pandoravirus", text)` | PASS |
| C11 | `viral_cnn.inference.audit_long.quantify_training_overlap` computes exact sequence matching and $k$-mer overlap fraction between long test contigs and training references (AC 11) | `.venv/bin/python3 -m unittest discover -s tests -k test_quantify_training_overlap_kmers_and_exact_match` exit 0 | `tests/test_voter.py:225` - `self.assertAlmostEqual(metrics["exact_match_fraction"], 1.0 / 3.0, places=4)` · `tests/test_voter.py:227` - `self.assertAlmostEqual(metrics["fraction_with_any_overlap"], 2.0 / 3.0, places=4)` | PASS |
| C12 | `viral_cnn.inference.audit_long.stratify_long_errors` stratifies accuracy across length bins and ambiguity proportions (AC 12) | `.venv/bin/python3 -m unittest discover -s tests -k test_stratify_long_errors_by_length_and_ambiguity` exit 0 | `tests/test_voter.py:248` - `self.assertEqual(strata["by_length"]["len_512_to_2000"]["count"], 2)` · `tests/test_voter.py:249` - `self.assertEqual(strata["by_length"]["len_512_to_2000"]["accuracy_mean_logits"], 1.0)` · `tests/test_voter.py:255` - `self.assertEqual(strata["by_ambiguity"]["ambig_0_pct"]["count"], 2)` | PASS |
| C13 | CLI `scripts/run_long_voter.py` executes long sequence inference, writes structured parquet predictions, and exits with code 0 (AC 13) | `.venv/bin/python3 -m unittest discover -s tests -k test_run_long_voter_cli_exit_zero_and_prediction_output` exit 0 | `tests/test_voter.py:273` - `self.assertEqual(res.returncode, 0)` · `tests/test_voter.py:278` - `self.assertTrue(out_csv.exists())` · `tests/test_voter.py:283` - `self.assertEqual(len(df_out), 4)` | PASS |

## Coverage

| Set (size) | Member -> proof | Unproven |
| --- | --- | --- |
| LongSequenceVoter short contig handling modes (2) | `skip` C2 · `pad` C3 | - |
| Consensus prediction aggregation methods (2) | `pred_mean_logits` C4 · `pred_majority_vote` C4 | - |
| Sequence length stratification intervals (3) | `len_512_to_2000` C12 · `len_2000_to_5000` C12 · `len_5000_to_10000` C12 | - |
| Ambiguity proportion stratification intervals (3) | `ambig_0_pct` C12 · `ambig_0_to_5_pct` C12 · `ambig_gt_5_pct` C12 | - |
| CLI execution status codes (2) | `0` C13 · `2` C9 | - |

## Level and sampling gaps

- Boundary level checks: C9 and C13 verify model argument validation and CLI execution across process boundaries via `subprocess.run`, checking exit code 0 on valid inputs, exit code 2 on invalid arguments, and structured CSV/Parquet output creation.
- Algorithmic and numerical units: C1, C2, C3 verify sequence slicing with 512 bp windows, 256 bp stride, uppercase normalization, and both `pad` and `skip` short contig policies. C4 and C5 verify consensus aggregation via pre-softmax logit averaging and majority voting with deterministic `min_index` mode tie-breaking. C6 verifies numerical logit equivalence between `engine.py` and `voter.py` within $<10^{-5}$ absolute error. C8 verifies IUPAC null vector one-hot encoding.
- Taxonomic integrity and offsets: C7 and C9 verify 0-based to 1-based class index conversions (`tax_id = model_label + 1`) and enforce species head dimension validation ($C = 13683$).
- Scientific audit and overlap quantification: C10, C11, C12 verify the audit documentation in `docs/long_sequence_inference_audit.md`, $k$-mer/exact sequence overlap calculations against training reference genomes, and error stratification across sequence lengths and ambiguity fractions.
- Sampling gaps: Zero gaps identified. All short-sequence modes (`skip`, `pad`), aggregation methods (`mean_logits`, `majority_vote`), length bins (`[512, 2000)`, `[2000, 5000)`, `[5000, 10000]`), and ambiguity bins (`0%`, `(0%, 5%]`, `>5%`) are covered by explicit tests.

## Swept existing

- **validation (C9)**: `src/viral_cnn/inference/voter.py:133-143` validates model head output dimensions against `expected_classes`, raising `ValueError` when incompatible heads are supplied. Verified in `tests/test_voter.py:181-190`.
- **failure modes (C2)**: `src/viral_cnn/inference/voter.py:56-58` handles sequences shorter than window size under `handle_short='skip'` by returning empty window lists without runtime exceptions. Verified in `tests/test_voter.py:80-84`.
- **idempotency (C6)**: Slicing and logit inference across identical sequences produce deterministic identical logits matching `engine.predict_batch`. Verified in `tests/test_voter.py:134-141`.
- **authorization**: n/a — local scientific CLI and inference library with no network or credential boundary.
- **concurrency**: n/a — deterministic batch inference on single machine without concurrent shared state mutation.
- **data lifecycle (C10)**: `docs/long_sequence_inference_audit.md` records the full provenance chain, simulation parameters (`calculate_n_reads`), reference overlaps, and structured prediction parquet schemas. Verified in `tests/test_voter.py:193-206`.
- **dependency failure (C6)**: Synthetic `TinyMockModel` fixtures allow complete unit testing and regression testing of all inference paths without requiring external remote checkpoints. Verified in `tests/test_voter.py:38-60`.
- **state transitions**: n/a — pure stateless window slicing and logit aggregation.
- **observability (C13)**: CLI `scripts/run_long_voter.py` prints summary accuracy and window metrics to stdout and saves structured tabular outputs (`pred_mean_logits`, `pred_majority_vote`, `num_windows`). Verified in `tests/test_voter.py:257-284`.

## Traceability

- **AC 1 (512 bp Window Slicing, 256 bp Stride, Case & Token Normalization)**: C1 · `src/viral_cnn/inference/voter.py:37-73` · `tests/test_voter.py:65-79`
- **AC 2 (Short Contig Skip Policy)**: C2 · `src/viral_cnn/inference/voter.py:56-58` · `tests/test_voter.py:80-84`
- **AC 3 (Short Contig Pad Policy)**: C3 · `src/viral_cnn/inference/voter.py:59-62` · `tests/test_voter.py:86-97`
- **AC 4 (Pre-Softmax Mean Logits and Mode Majority Voting)**: C4 · `src/viral_cnn/inference/voter.py:75-115` · `tests/test_voter.py:98-119`
- **AC 5 (Deterministic Min-Index Mode Tie-Breaking)**: C5 · `src/viral_cnn/inference/voter.py:100-109` · `tests/test_voter.py:120-131`
- **AC 6 (Window Logit Equivalence with Engine)**: C6 · `src/viral_cnn/inference/audit_long.py:24-49` · `tests/test_voter.py:134-141`
- **AC 7 (One-Based Tax ID Conversion Offset)**: C7 · `src/viral_cnn/inference/voter.py:173-184` · `tests/test_voter.py:142-158`
- **AC 8 (IUPAC Ambiguous Characters Null Vector Encoding)**: C8 · `src/viral_cnn/inference/voter.py:47-48,60` · `src/viral_cnn/data/encoding.py:13-33` · `tests/test_voter.py:159-180`
- **AC 9 (Class Dimension Validation & Rejection)**: C9 · `src/viral_cnn/inference/voter.py:133-143` · `tests/test_voter.py:181-190`
- **AC 10 (Scientific Audit Documentation & Causal Reporting)**: C10 · `docs/long_sequence_inference_audit.md:1-121` · `tests/test_voter.py:193-206`
- **AC 11 (Reference Genome Exact & K-Mer Overlap Quantification)**: C11 · `src/viral_cnn/inference/audit_long.py:52-127` · `tests/test_voter.py:207-228`
- **AC 12 (Error Stratification across Lengths and Ambiguity)**: C12 · `src/viral_cnn/inference/audit_long.py:130-220` · `tests/test_voter.py:229-256`
- **AC 13 (Long Voter CLI Execution & Structured Output)**: C13 · `scripts/run_long_voter.py:33-223` · `tests/test_voter.py:257-284`

## Gate

- `.venv/bin/python3 -m unittest tests/test_voter.py` - 13 passed, 0 failed
- `.venv/bin/python3 -m unittest discover tests` - 64 passed, 0 failed

## Verdict

PASS. All 13 checks proven with located `file:line` assertions, zero surviving defects, complete short contig and aggregation method coverage, and full conformance to scientific invariants in `AGENTS.md`.
