# s07-baselines-kraken2-blastn verification

**Verdict**: PASS
**Profile**: light
**Diff range**: b5b9ff3..HEAD
**Round**: 1 - full
**Verifier**: independent sub-agent (author != verifier)

## Checks

| Check | Claim | Proof run | Evidence | Result |
| --- | --- | --- | --- | --- |
| C1 | `parse_kraken_output` maps classified queries (`status == 'C'`) with valid taxon IDs ($2 \le \text{taxid} < 13685$) to zero-based class label `taxid - 2` | `.venv/bin/python3 -m unittest discover -s tests -k test_parse_kraken_output_classified_species_mapping` exit 0 | `tests/test_baselines.py:44` - `self.assertEqual(res["q1"], 0)` | PASS |
| C2 | `parse_kraken_output` maps unclassified queries (`status == 'U'`) and root taxid 1 to label `-1` denoting abstention | `.venv/bin/python3 -m unittest discover -s tests -k test_parse_kraken_output_unclassified_and_root_abstention` exit 0 | `tests/test_baselines.py:56` - `self.assertEqual(res["q1"], -1)` | PASS |
| C3 | `parse_kraken_output` rejects duplicate query IDs or queries absent from expected manifest with `ValueError` | `.venv/bin/python3 -m unittest discover -s tests -k test_parse_kraken_output_rejects_duplicate_or_unknown_queries` exit 0 | `tests/test_baselines.py:66` - `with self.assertRaises(ValueError):` | PASS |
| C4 | `parse_blast_output` aggregates hits by species label, retains maximum bitscore per species, resolves ties by lowest E-value and lowest label index, and returns top-5 candidates | `.venv/bin/python3 -m unittest discover -s tests -k test_parse_blast_output_bitscore_aggregation_and_top_candidates` exit 0 | `tests/test_baselines.py:88` - `self.assertEqual(res["q1"], [20, 10, 5])` | PASS |
| C5 | `parse_blast_output` assigns label `-1` and empty top-5 candidates for queries with no hits | `.venv/bin/python3 -m unittest discover -s tests -k test_parse_blast_output_unaligned_abstention` exit 0 | `tests/test_baselines.py:97` - `self.assertEqual(res["q_no_hit"], [])` | PASS |
| C6 | `parse_blast_output` rejects subject IDs lacking `_label_{int}` suffix with `ValueError` | `.venv/bin/python3 -m unittest discover -s tests -k test_parse_blast_output_rejects_invalid_subject_format` exit 0 | `tests/test_baselines.py:104` - `with self.assertRaises(ValueError):` | PASS |
| C7 | `audit_exact_overlap` hashes short queries and partitions into `all`, `overlap` ($\text{train\_count} > 0$), and `no_exact_overlap` ($\text{train\_count} == 0$) | `.venv/bin/python3 -m unittest discover -s tests -k test_audit_exact_overlap_partitioning_strata` exit 0 | `tests/test_baselines.py:128` - `self.assertTrue(masks["overlap"][0])` | PASS |
| C8 | `evaluate_predictions` counts abstentions (`pred == -1`) as errors in total denominator $N$ for Top-1 accuracy | `.venv/bin/python3 -m unittest discover -s tests -k test_evaluate_predictions_abstention_counted_in_denominator` exit 0 | `tests/test_baselines.py:140` - `self.assertAlmostEqual(res["accuracy"], 0.50, places=4)` | PASS |
| C9 | `evaluate_predictions` computes closed-form Wilson score confidence intervals for Top-1 accuracy | `.venv/bin/python3 -m unittest discover -s tests -k test_evaluate_predictions_wilson_confidence_intervals` exit 0 | `tests/test_baselines.py:146` - `self.assertGreater(low, 0.70)` | PASS |
| C10 | `evaluate_predictions` computes coverage fraction ($\text{pred} \ge 0$) and conditional accuracy on covered queries | `.venv/bin/python3 -m unittest discover -s tests -k test_evaluate_predictions_coverage_and_conditional_accuracy` exit 0 | `tests/test_baselines.py:156` - `self.assertAlmostEqual(res["coverage"], 0.50, places=4)` | PASS |
| C11 | `evaluate_predictions` computes Top-5 accuracy when candidate columns are present | `.venv/bin/python3 -m unittest discover -s tests -k test_evaluate_predictions_top5_accuracy` exit 0 | `tests/test_baselines.py:170` - `self.assertAlmostEqual(res["top5"], 2.0 / 3.0, places=4)` | PASS |
| C12 | `evaluate_predictions` computes Macro-F1 across species present in each stratum with zero division mapped to 0.0 | `.venv/bin/python3 -m unittest discover -s tests -k test_evaluate_predictions_macro_f1_with_zero_division` exit 0 | `tests/test_baselines.py:177` - `self.assertGreaterEqual(res["macro_f1"], 0.0)` | PASS |
| C13 | `paired_discordance` rejects comparisons between mismatched query sets with `ValueError` | `.venv/bin/python3 -m unittest discover -s tests -k test_paired_discordance_rejects_mismatched_query_sets` exit 0 | `tests/test_baselines.py:186` - `with self.assertRaises(ValueError):` | PASS |
| C14 | `paired_discordance` computes $2 \times 2$ contingency table (`both_correct`, `only_a_correct`, `only_b_correct`, `both_wrong`) and discordant query count and percentage | `.venv/bin/python3 -m unittest discover -s tests -k test_paired_discordance_contingency_counts_and_percentage` exit 0 | `tests/test_baselines.py:199` - `self.assertEqual(res["discordant_queries"], 2)` | PASS |
| C15 | `bootstrap_species_difference` computes 95% confidence intervals for paired accuracy differences resampling species clusters over 2,000 repeats with seed 42 | `.venv/bin/python3 -m unittest discover -s tests -k test_bootstrap_species_difference_quantile_intervals` exit 0 | `tests/test_baselines.py:208` - `self.assertLessEqual(ci_low, 5.0)` | PASS |
| C16 | `integrate_long_voter_predictions` ingests `LongSequenceVoter` outputs (`pred_mean_logits`, `pred_majority_vote`) into unified baseline schema alongside Kraken2 and BLASTN | `.venv/bin/python3 -m unittest discover -s tests -k test_integrate_long_voter_predictions_schema_unification` exit 0 | `tests/test_baselines.py:225` - `self.assertEqual(integrated["Voter_mean_logits"]["read1"], 10)` | PASS |
| C17 | `evaluate_long_baselines` stratifies Top-1 accuracy and coverage across reference genome presence (`seen_genome` vs `unseen_genome`) | `.venv/bin/python3 -m unittest discover -s tests -k test_evaluate_long_baselines_reference_genome_stratification` exit 0 | `tests/test_baselines.py:243` - `self.assertIn("seen_genome", strata)` | PASS |
| C18 | `evaluate_long_baselines` stratifies Top-1 accuracy across contig length bins ($[512, 2000)$, $[2000, 5000)$, $\ge 5000$ bp) | `.venv/bin/python3 -m unittest discover -s tests -k test_evaluate_long_baselines_length_bin_stratification` exit 0 | `tests/test_baselines.py:260` - `self.assertIn("len_512_to_2000", strata)` | PASS |
| C19 | Paired comparison on long reads computes paired contingency table and species-cluster bootstrap 95% CI between CNN Voter and baseline tools | `.venv/bin/python3 -m unittest discover -s tests -k test_long_reads_paired_discordance_and_bootstrap_intervals` exit 0 | `tests/test_baselines.py:277` - `self.assertIn("ci95_low_pp", paired_df.columns)` | PASS |
| C20 | CLI `scripts/run_baseline_comparison.py` supports `--regime {short,long}`, writes structured CSV outputs, and exits with code 0 | `.venv/bin/python3 -m unittest discover -s tests -k test_run_baseline_comparison_cli_short_and_long_regimes` exit 0 | `tests/test_baselines.py:313` - `self.assertEqual(res.returncode, 0, f"CLI error: {res.stderr}")` | PASS |
| C21 | `docs/baseline_benchmark_protocol.md` documents tool versions, commands, parameters, denominators, and overlap/genome stratification | `.venv/bin/python3 -m unittest discover -s tests -k test_baseline_protocol_documentation_scientific_sections` exit 0 | `tests/test_baselines.py:320` - `self.assertTrue(doc_path.exists())` | PASS |
| C22 | `KmerCentroidClassifier` fits $k$-mer frequency profiles and predicts labels under controlled noise | `.venv/bin/python3 -m unittest discover -s tests -k test_kmer_centroid_classifier_fit_predict_under_noise` exit 0 | `tests/test_baselines.py:340` - `self.assertEqual(preds[0], 0)` | PASS |

## Coverage

| Set (size) | Recomputed from | Member -> proof | Unproven |
| --- | --- | --- | --- |
| Evaluation regimes (2) | `scripts/run_baseline_comparison.py` CLI parser choices | `short` C20 · `long` C20 | - |
| Baseline output parsers (2) | `src/viral_cnn/baselines/taxonomy.py` module exports | `Kraken2` C1 · `BLASTN` C4 | - |
| Kraken2 status mappings (2) | `src/viral_cnn/baselines/taxonomy.py:parse_kraken_output` status parsing | `classified` C1 · `unclassified` C2 | - |
| Exact overlap strata for short fragments (3) | `src/viral_cnn/baselines/overlap.py:audit_exact_overlap` strata output | `all` C7 · `overlap` C7 · `no_exact_overlap` C7 | - |
| Long-read reference genome strata (2) | `src/viral_cnn/baselines/long_eval.py:evaluate_long_baselines` strata | `seen_genome` C17 · `unseen_genome` C17 | - |
| Long-read length bin intervals (3) | `src/viral_cnn/baselines/long_eval.py:evaluate_long_baselines` length partitions | `len_512_to_2000` C18 · `len_2000_to_5000` C18 · `len_ge_5000` C18 | - |
| CLI execution status codes (2) | `scripts/run_baseline_comparison.py` execution outcomes | `0` C20 · `2` C3 | - |

## Level and sampling gaps

- **Level judgment**:
  - The CLI surface `scripts/run_baseline_comparison.py` is invoked across the process boundary using `subprocess.run` (C20), proving arguments parsing, output artifact creation (`metrics_by_stratum.csv`, `comparison_manifest.json`), and exit code 0.
  - Parsing, exact overlap audit, and evaluation functions are verified at unit and integration levels with synthetic fixtures that reproduce exact real-world boundary cases: root taxid 1, minimum species taxid 2, maximum species taxid 13684, abstention label `-1`, tie-breaking on BLAST E-value and index, exact Wilson score bounds, and species cluster bootstrap quantile estimation.
  - No level gaps found.
- **Sampling judgment**:
  - Exact overlap partition is verified on matching, non-matching, and multi-match sequences.
  - Long read strata cover both reference genome presence (`seen_genome` vs `unseen_genome`) and all three length intervals ($[512, 2000)$, $[2000, 5000)$, $\ge 5000$ bp).
  - No sampling gaps found.

## Swept existing

- **validation**: C3 - duplicate queries or unmanifested queries raise `ValueError` in `parse_kraken_output`.
- **failure modes**: C2 - unclassified or root queries map to `-1` abstention without raising unhandled errors.
- **idempotency**: C8 - `evaluate_predictions` computes deterministic closed-set metrics with abstentions counted in denominator.
- **authorization**: n/a - local scientific CLI tool with no network or authorization boundary.
- **concurrency**: n/a - deterministic batch processing on single machine with no concurrent state mutation.
- **data lifecycle**: C21 - `docs/baseline_benchmark_protocol.md` documents provenance, database versions, parameters, and manifest persistence.
- **dependency failure**: C5 - unaligned queries in BLAST output map to empty candidate list rather than crashing.
- **state transitions**: n/a - stateless output parsing and metric evaluation.
- **observability**: C20 - CLI logs execution parameters and writes machine-readable manifest and CSV summary tables.

## Gate

`.venv/bin/python3 -m unittest -v tests/test_baselines.py` - 22 passed, 0 failed
