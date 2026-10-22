# S07: Baselines Kraken2 and BLASTN under Unified Protocol - Checks

Profile: light
Plan: `.specs/features/s07-baselines-kraken2-blastn/plan.md`

## Checks

### S1: Baseline Output Parsing and Taxonomy Conversion · 2 files · 20 KB · ~5k

**C1** - `viral_cnn.baselines.taxonomy.parse_kraken_output` maps classified queries (`status == 'C'`) with valid taxon IDs ($2 \le \text{taxid} < 13685$) to zero-based class label `taxid - 2` (AC 1)
Proof: `.venv/bin/python3 -m unittest discover -s tests -k test_parse_kraken_output_classified_species_mapping`

**C2** - `viral_cnn.baselines.taxonomy.parse_kraken_output` maps unclassified queries (`status == 'U'`) and root taxid 1 to label `-1` denoting abstention (AC 2)
Proof: `.venv/bin/python3 -m unittest discover -s tests -k test_parse_kraken_output_unclassified_and_root_abstention`

**C3** - `viral_cnn.baselines.taxonomy.parse_kraken_output` rejects duplicate query IDs or queries absent from expected manifest with `ValueError` (AC 3)
Proof: `.venv/bin/python3 -m unittest discover -s tests -k test_parse_kraken_output_rejects_duplicate_or_unknown_queries`

**C4** - `viral_cnn.baselines.taxonomy.parse_blast_output` aggregates hits by species label, retains maximum bitscore per species, resolves ties by lowest E-value and lowest label index, and returns top-5 candidates (AC 4)
Proof: `.venv/bin/python3 -m unittest discover -s tests -k test_parse_blast_output_bitscore_aggregation_and_top_candidates`

**C5** - `viral_cnn.baselines.taxonomy.parse_blast_output` assigns label `-1` and empty top-5 candidates for queries with no hits (AC 5)
Proof: `.venv/bin/python3 -m unittest discover -s tests -k test_parse_blast_output_unaligned_abstention`

**C6** - `viral_cnn.baselines.taxonomy.parse_blast_output` rejects subject IDs lacking `_label_{int}` suffix with `ValueError` (AC 6)
Proof: `.venv/bin/python3 -m unittest discover -s tests -k test_parse_blast_output_rejects_invalid_subject_format`

### S2: Exact Overlap Stratification and Unified Evaluator · 2 files · 22 KB · ~6k

**C7** - `viral_cnn.baselines.overlap.audit_exact_overlap` hashes short queries and partitions into `all`, `overlap` ($\text{train\_count} > 0$), and `no_exact_overlap` ($\text{train\_count} == 0$) (AC 7)
Proof: `.venv/bin/python3 -m unittest discover -s tests -k test_audit_exact_overlap_partitioning_strata`

**C8** - `viral_cnn.baselines.paired.evaluate_predictions` counts abstentions (`pred == -1`) as errors in total denominator $N$ for Top-1 accuracy (AC 8)
Proof: `.venv/bin/python3 -m unittest discover -s tests -k test_evaluate_predictions_abstention_counted_in_denominator`

**C9** - `viral_cnn.baselines.paired.evaluate_predictions` computes closed-form Wilson score confidence intervals for Top-1 accuracy (AC 9)
Proof: `.venv/bin/python3 -m unittest discover -s tests -k test_evaluate_predictions_wilson_confidence_intervals`

**C10** - `viral_cnn.baselines.paired.evaluate_predictions` computes coverage fraction ($\text{pred} \ge 0$) and conditional accuracy on covered queries (AC 10)
Proof: `.venv/bin/python3 -m unittest discover -s tests -k test_evaluate_predictions_coverage_and_conditional_accuracy`

**C11** - `viral_cnn.baselines.paired.evaluate_predictions` computes Top-5 accuracy when candidate columns are present (AC 11)
Proof: `.venv/bin/python3 -m unittest discover -s tests -k test_evaluate_predictions_top5_accuracy`

**C12** - `viral_cnn.baselines.paired.evaluate_predictions` computes Macro-F1 across species present in each stratum with zero division mapped to 0.0 (AC 12)
Proof: `.venv/bin/python3 -m unittest discover -s tests -k test_evaluate_predictions_macro_f1_with_zero_division`

### S3: Paired Discordance and Cluster Bootstrap Analysis · 2 files · 20 KB · ~5k

**C13** - `viral_cnn.baselines.paired.paired_discordance` rejects comparisons between mismatched query sets with `ValueError` (AC 13)
Proof: `.venv/bin/python3 -m unittest discover -s tests -k test_paired_discordance_rejects_mismatched_query_sets`

**C14** - `viral_cnn.baselines.paired.paired_discordance` computes $2 \times 2$ contingency table (`both_correct`, `only_a_correct`, `only_b_correct`, `both_wrong`) and discordant query count and percentage (AC 14)
Proof: `.venv/bin/python3 -m unittest discover -s tests -k test_paired_discordance_contingency_counts_and_percentage`

**C15** - `viral_cnn.baselines.paired.bootstrap_species_difference` computes 95% confidence intervals for paired accuracy differences resampling species clusters over 2,000 repeats with seed 42 (AC 15)
Proof: `.venv/bin/python3 -m unittest discover -s tests -k test_bootstrap_species_difference_quantile_intervals`

### S4: Long-Read Protocol and Baseline Comparison · 2 files · 22 KB · ~6k

**C16** - `viral_cnn.baselines.long_eval.integrate_long_voter_predictions` ingests `LongSequenceVoter` outputs (`pred_mean_logits`, `pred_majority_vote`) into unified baseline schema alongside Kraken2 and BLASTN (AC 16)
Proof: `.venv/bin/python3 -m unittest discover -s tests -k test_integrate_long_voter_predictions_schema_unification`

**C17** - `viral_cnn.baselines.long_eval.evaluate_long_baselines` stratifies Top-1 accuracy and coverage across reference genome presence (`seen_genome` vs `unseen_genome`) (AC 17)
Proof: `.venv/bin/python3 -m unittest discover -s tests -k test_evaluate_long_baselines_reference_genome_stratification`

**C18** - `viral_cnn.baselines.long_eval.evaluate_long_baselines` stratifies Top-1 accuracy across contig length bins ($[512, 2000)$, $[2000, 5000)$, $\ge 5000$ bp) (AC 18)
Proof: `.venv/bin/python3 -m unittest discover -s tests -k test_evaluate_long_baselines_length_bin_stratification`

**C19** - Paired comparison on long reads computes paired contingency table and species-cluster bootstrap 95% CI between CNN Voter and baseline tools (AC 19)
Proof: `.venv/bin/python3 -m unittest discover -s tests -k test_long_reads_paired_discordance_and_bootstrap_intervals`

### S5: CLI Runner, Protocol Documentation, and $k$-mer Extension · 3 files · 28 KB · ~8k

**C20** - CLI `scripts/run_baseline_comparison.py` supports `--regime {short,long}`, writes structured CSV outputs, and exits with code 0 (AC 20)
Proof: `.venv/bin/python3 -m unittest discover -s tests -k test_run_baseline_comparison_cli_short_and_long_regimes`

**C21** - `docs/baseline_benchmark_protocol.md` documents tool versions, commands, parameters, denominators, and overlap/genome stratification (AC 21)
Proof: `.venv/bin/python3 -m unittest discover -s tests -k test_baseline_protocol_documentation_scientific_sections`

**C22** - `viral_cnn.baselines.kmer_baseline.KmerCentroidClassifier` fits $k$-mer frequency profiles and predicts labels under controlled noise (AC 22)
Proof: `.venv/bin/python3 -m unittest discover -s tests -k test_kmer_centroid_classifier_fit_predict_under_noise`

## Coverage

| Set (size) | Member -> proof | Unproven |
| --- | --- | --- |
| Evaluation regimes (2) | `short` C20 · `long` C20 | - |
| Baseline output parsers (2) | `Kraken2` C1 · `BLASTN` C4 | - |
| Kraken2 status mappings (2) | `classified` C1 · `unclassified` C2 | - |
| Exact overlap strata for short fragments (3) | `all` C7 · `overlap` C7 · `no_exact_overlap` C7 | - |
| Long-read reference genome strata (2) | `seen_genome` C17 · `unseen_genome` C17 | - |
| Long-read length bin intervals (3) | `len_512_to_2000` C18 · `len_2000_to_5000` C18 · `len_ge_5000` C18 | - |
| CLI execution status codes (2) | `0` C20 · `2` C3 | - |

- Claims naming a status code or response shape: C3, C20 - each has a proof crossing the process boundary
- No other check claims more than the single case its proof exercises

## Swept

- validation: C3
- failure modes: C2
- idempotency: C8
- authorization: n/a - local scientific CLI tool with no network or authorization boundary
- concurrency: n/a - deterministic batch processing on single machine with no concurrent state mutation
- data lifecycle: C21
- dependency failure: C5
- state transitions: n/a - stateless output parsing and metric evaluation
- observability: C20

## Handoff

- S1 = 5k in baseline parsers and taxonomy conversion; S2 = 6k in overlap audit and metric evaluation; S3 = 5k in paired discordance and cluster bootstrap; S4 = 6k in long-read integration and stratification; S5 = 8k in CLI runner, protocol documentation, and k-mer baseline. Total = ~30k tokens, well under the 150k budget - one builder

- **Mechanism:** one builder
