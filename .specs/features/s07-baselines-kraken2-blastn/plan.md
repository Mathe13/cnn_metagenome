# S07: Baselines Kraken2 and BLASTN under Unified Protocol

## Problem

The manuscript compares viral CNNs against classic metagenomic baselines (Kraken2 and BLASTN), but historical comparisons risk methodological invalidity if evaluated under mismatched protocols:
1. Historical benchmark workflows are isolated in Jupyter notebooks (`short/benchmark.ipynb` and `short/benchmark_cnns_kraken2_blast_overlap.ipynb`) without reusable CLI integration or automated regression contracts.
2. In the random fragment split, an audit reveals that approximately 95.89% of test queries share exact 512 bp sequence identity with training fragments (`exact_overlap.csv`). When a sequence search tool like BLASTN or Kraken2 accesses the exact identical sequence in its database, the resulting performance reflects exact database lookup rather than taxonomic generalization. Comparing tools without stratifying by reference overlap conflates database memorization with generalizable classification.
3. For long sequences and reads (manuscript Section 3.5, Table 12), the comparison is further compounded: CNNs evaluate variable-length contigs via sliding-window consensus voting (window 512 bp, stride 256 bp), whereas Kraken2 and BLASTN evaluate the full contig directly. When long reads originate from reference genomes present in the database, full-length local alignments (BLASTN) or accumulated $k$-mer chains (Kraken2) achieve near-perfect lookup, which cannot be directly equated to generalizable classification on unseen viral genomes without explicit reference genome stratification.
4. Baselines can exhibit abstention (unclassified queries for Kraken2; queries with no alignment meeting E-value or bitscore thresholds for BLASTN). If unclassified queries are filtered out from the denominator, accuracy is artificially inflated (conditional accuracy) rather than reflecting standard closed-set classification accuracy where abstention counts as incorrect.
5. Directly comparing accuracy percentages across differing query subsets, varying sequence length regimes, or distinct noise models without paired discordance statistics and cluster-bootstrap confidence intervals violates scientific reporting standards declared in `AGENTS.md`.

When this ships, `src/viral_cnn/baselines/` provides reusable, auditable modules for taxonomy conversion, output parsing, exact overlap and reference genome auditing, and paired statistical evaluation for both short fragments (512 bp) and long reads; `scripts/run_baseline_comparison.py` provides an auditable CLI; `docs/baseline_benchmark_protocol.md` provides complete documentation of tool versions, commands, parameters, denominators, and overlap stratification across both regimes; and contract tests verify all invariants deterministically.

## Flow

Reuses `src/viral_cnn/eval/metrics.py` and `src/viral_cnn/inference/voter.py` while preserving notebook artifacts and external databases.

1. raw Kraken2 report files -> `viral_cnn.baselines.taxonomy:parse_kraken_output` (new, door 1) - parses classified taxids to 0-based species labels and unclassified/root to -1 for short fragments and long reads
2. raw BLASTN tabular files -> `viral_cnn.baselines.taxonomy:parse_blast_output` (new, door 2) - aggregates top-5 candidates per query by bitscore and maps subjects to species labels for short fragments and long reads
3. evaluation queries, training references, and genome metadata -> `viral_cnn.baselines.overlap:audit_exact_overlap` (new, door 3) - partitions short queries into all/overlap/no_exact_overlap and long reads into seen_genome/unseen_genome and length bins
4. long-sequence voter predictions -> `viral_cnn.baselines.long_eval:integrate_long_voter_predictions` (new, door 4) - extracts mean-logits and majority-vote predictions from S06 Voter into the unified baseline schema
5. parsed predictions, strata, and ground truth -> `viral_cnn.baselines.paired:evaluate_paired_baselines` (new, door 5) - computes stratified Top-1/Top-5 accuracy with Wilson CI, coverage, conditional accuracy, Macro-F1, and paired species-cluster bootstrap differences
6. `scripts/run_baseline_comparison.py` (new, no door - placement per conventions) - CLI executing baseline evaluation on short or long reads and writing structured comparison CSVs and json execution manifest
7. out: auditable stratified metrics tables, paired discordance tables, and protocol documentation in `docs/baseline_benchmark_protocol.md`

## Impact

| Front | What changes |
| --- | --- |
| domain | new term: `Kraken2FlatTaxonomy` - flat NCBI-compatible taxonomy mapping species labels $0 \le \text{label} < 13683$ directly to taxids $\text{taxid} = \text{label} + 2$ with root at 1 |
| domain | new term: `AbstentionHandling` - standard accounting rule assigning prediction label -1 to unclassified queries and including all queries in the total accuracy denominator |
| domain | new term: `ExactOverlapStratum` - tripartite stratification of short evaluation queries into `all` (100%), `overlap` (exact 512 bp match in training references), and `no_exact_overlap` (no exact 512 bp match in training references) |
| domain | new term: `LongReadStratum` - stratification of long reads into reference genome presence (`seen_genome` vs `unseen_genome`) and sequence length bins ($[512, 2000)$, $[2000, 5000)$, $\ge 5000$ bp) |
| domain | new term: `SpeciesClusterBootstrap` - paired difference confidence interval estimation resampling species labels with replacement to account for intra-species query correlation |
| stored data | nothing to migrate: historical notebooks and parquet files remain unmodified in place |

## Relations

None - no stored-data shape change

## Surface

None - nothing consumed outside

## Landing

| One-way door | Literal shape | Alternative rejected |
| --- | --- | --- |
| Kraken2 flat taxonomy mapping | `custom_taxid = label + 2` with root at 1, parsing status `'C'` as `label = taxid - 2` and `'U'` or `taxid < 2` as `-1` | hierarchical NCBI taxonomy traversal - rejected because custom viral training classes are closed-set and flat mapping prevents taxonomic node mismatch |
| BLASTN Top-K candidate parsing | maximum bitscore per species label, tie-broken by lowest E-value and lowest label index, truncated to top 5 candidates | taking only the single first hit in TSV - rejected because Table 11/12 and notebook benchmarks evaluate Top-5 retrieval capabilities for BLASTN |
| Denominator accounting for abstentions | `accuracy = (y_pred == y_true).sum() / N_total` with `y_pred == -1` counted as error, alongside explicit `coverage = (y_pred >= 0).sum() / N_total` | filtering unclassified queries from denominator - rejected because conditional accuracy distorts comparative benchmarking against models that do not abstain |
| Paired difference bootstrap grouping | cluster bootstrap resampling unique species labels with replacement across 2,000 repeats with seed 42 | query-level independent bootstrap - rejected because fragments from the same species are non-independent, leading to artificially narrow confidence intervals |
| Long-read evaluation unification | unified evaluation schema consuming both sliding-window Voter outputs (`pred_mean_logits`, `pred_majority_vote`) and direct long-read baseline outputs (`Kraken2`, `BLASTN`) | evaluating long sequences in an isolated disconnected voter script - rejected because comparative baselines must use identical query IDs, ground truth, and statistical metrics |

- Nothing else in this change is hard to reverse

## Criteria

### S1: Baseline Output Parsing and Taxonomy Conversion (P1)

Parsers converting raw Kraken2 and BLASTN outputs into canonical integer class predictions matching CNN label indices for both short and long reads.

**Acceptance Criteria**

1. WHEN parsing a Kraken2 report output THEN the system SHALL map classified queries (`status == 'C'`) with valid taxon IDs ($2 \le \text{taxid} < 13685$) to zero-based class label `taxid - 2`.
2. WHEN parsing a Kraken2 report output THEN the system SHALL map unclassified queries (`status == 'U'`) and root taxid 1 to label `-1` denoting abstention.
3. IF a Kraken2 output contains duplicate query IDs or query IDs absent from the evaluated query manifest THEN the system SHALL reject parsing with `ValueError`.
4. WHEN parsing a BLASTN tabular hit output (`-outfmt "6 qseqid sseqid bitscore evalue pident length"`) with subjects formatted as `{rid}_label_{label}` THEN the system SHALL aggregate hits by species label, retain the maximum bitscore per species, resolve ties by lowest E-value then lowest label index, and return top-$K$ candidates ordered by rank.
5. WHEN a query in the manifest has no alignments in the BLASTN output THEN the system SHALL assign label `-1` for Top-1 and empty top-$K$ candidates.
6. IF a BLASTN output row contains an unparseable subject ID lacking the `_label_{int}` suffix THEN the system SHALL raise `ValueError`.

**Independent test:** unit tests in `tests/test_baselines.py` asserting that sample Kraken2 and BLASTN outputs parse correctly to exact label arrays and handle unclassified/missing queries deterministically.

### S2: Exact Overlap Stratification and Unified Evaluator (P1)

Auditing exact sequence overlap between queries and training references and computing stratified metrics.

**Acceptance Criteria**

7. WHEN provided with short evaluation queries and training reference sequences THEN the system SHALL compute canonical uppercase SHA256 hashes and partition queries into `all`, `overlap` ($\text{train\_count} > 0$), and `no_exact_overlap` ($\text{train\_count} == 0$).
8. WHEN computing classification metrics for any method THEN the system SHALL use total queries $N$ (including abstentions where $\text{pred} == -1$) as the denominator for Top-1 accuracy.
9. The system SHALL compute Wilson score confidence intervals for Top-1 accuracy across all strata and methods.
10. The system SHALL compute coverage fraction ($\text{pred} \ge 0$) and conditional accuracy restricted to covered queries ($\text{pred} \ge 0$).
11. WHERE top-$K$ candidate columns are present (BLASTN and CNNs) THEN the system SHALL compute Top-5 accuracy across all queries.
12. The system SHALL compute Macro-F1 across species present in each stratum with zero division mapped to 0.0.

**Independent test:** unit tests in `tests/test_baselines.py` verifying metrics computation on known predictions, confirming that abstentions are counted in the denominator and Wilson intervals match closed-form formulas.

### S3: Paired Discordance and Cluster Bootstrap Analysis (P1)

Paired comparison across methods evaluated on the exact same query sets.

**Acceptance Criteria**

13. IF two methods are compared on mismatched query IDs or differing query counts THEN the system SHALL reject comparison with `ValueError`.
14. WHEN comparing two methods on identical query sets THEN the system SHALL compute the $2 \times 2$ contingency table (`both_correct`, `only_a_correct`, `only_b_correct`, `both_wrong`) and discordant query count and percentage.
15. WHEN computing paired accuracy difference confidence intervals THEN the system SHALL perform cluster bootstrap resampling true species labels with replacement over 2,000 repeats with seed 42 to estimate the 95% confidence interval in percentage points.

**Independent test:** unit tests in `tests/test_baselines.py` verifying paired contingency counts and verifying that cluster bootstrap reproduces exact quantile intervals on deterministic inputs.

### S4: Long-Read Protocol and Baseline Comparison (P1)

Unified evaluation of CNN sliding-window Voter against Kraken2 and BLASTN on variable-length long reads.

**Acceptance Criteria**

16. WHEN evaluating long sequences THEN the system SHALL support comparative benchmarking of `LongSequenceVoter` (`pred_mean_logits` and `pred_majority_vote`), Kraken2, and BLASTN across identical sequence IDs and noise levels $\epsilon \in \{0.0, 0.02, 0.05, 0.10, 0.15\}$.
17. WHEN evaluating long reads THEN the system SHALL stratify Top-1 accuracy, coverage, and conditional accuracy across reference genome presence (`seen_genome` vs `unseen_genome`).
18. WHEN evaluating long reads with sequence length annotations THEN the system SHALL stratify Top-1 accuracy across length bins ($[512, 2000)$, $[2000, 5000)$, $\ge 5000$ bp).
19. WHEN comparing CNN Voter predictions against Kraken2 or BLASTN on long reads THEN the system SHALL compute paired contingency tables and species-cluster bootstrap 95% confidence intervals on the identical long query set.

**Independent test:** unit tests in `tests/test_baselines.py` running paired evaluation on synthetic long-read queries comparing Voter outputs with Kraken2 and BLASTN outputs, stratified by reference genome and length bins.

### S5: CLI Runner, Protocol Documentation, and $k$-mer Extension (P2)

CLI workflow, comprehensive protocol documentation, and optional $k$-mer baseline.

**Acceptance Criteria**

20. The system SHALL provide an auditable CLI in `scripts/run_baseline_comparison.py` that accepts `--regime {short,long}`, ground truth, method prediction files, and optional overlap/genome metadata, and outputs structured CSV summary tables and a JSON execution manifest.
21. The system SHALL publish an exhaustive scientific protocol report in `docs/baseline_benchmark_protocol.md` documenting tool versions, database build parameters, exact query IDs, noise rates, denominator conventions, exact overlap statistics (~95.89% training overlap on short fragments, reference genome presence on long reads), and paired comparison findings across both short and long regimes.
22. WHERE evaluated against sequence queries THEN the system SHALL provide an optional $k$-mer frequency baseline classifier (`KmerCentroidClassifier`) evaluated under the exact same queries and noise levels.

**Independent test:** run `python3 scripts/run_baseline_comparison.py --help` exiting with code 0, execute synthetic baseline comparison generating output CSVs for short and long regimes, and verify that `docs/baseline_benchmark_protocol.md` exists and contains required sections.

## Out of scope

| Excluded | Why |
| --- | --- |
| In-line re-execution of full Kraken2 database build on 13,683 species during unit test runs | Kraken2 DB build requires >50 GB disk and external compiled binaries; unit tests use deterministic parsers and synthetic fixtures |
| Modification of original benchmark notebooks | Notebooks are historical evidence protected under AGENTS.md invariant AD-004 |
| Cross-database contamination without identical queries | Any evaluation claiming direct comparison must use the identical query manifest |

## Assumptions

| Assumption | Chosen default | Rationale | Confirmed? |
| --- | --- | --- | --- |
| BLASTN max targets parameter | `max_target_seqs = 5` | Matches the default parameter used in `benchmark_cnns_kraken2_blast_overlap.ipynb` for species Top-5 retrieval | y |
| Default bootstrap repeats | 2,000 repeats with seed 42 | Matches Cell 30 of `benchmark_cnns_kraken2_blast_overlap.ipynb` for stable 95% quantile estimation | y |
| Abstention label representation | `-1` | Standard integer representation for invalid or abstained class in scikit-learn and project codebase | y |
| Long-read length bin thresholds | 512, 2000, 5000 bp | Matches length stratification boundaries established in S06 long-sequence audit | y |

**Open questions:** none - all resolved or logged above.

## Observable

| Surface | Decision | Landing |
| --- | --- | --- |
| CLI `scripts/run_baseline_comparison.py` | exit codes and help output | AC 20 |
| CLI `scripts/run_baseline_comparison.py` | structured CSV outputs and schema for short and long regimes | AC 20 |
| CLI `scripts/run_baseline_comparison.py` | input validation and mismatched queries error | AC 13 |
| Documentation `docs/baseline_benchmark_protocol.md` | structure, denominator explanation, short overlap, and long-read genome audit | AC 21 |
| all new modules in `src/viral_cnn/baselines/*` | error codes and invalid format exception | AC 3, AC 6 |

## Sources

- `short/benchmark_cnns_kraken2_blast_overlap.ipynb` - reference implementation of Kraken2/BLASTN parsing, overlap stratification, and paired bootstrap
- `long/cnn-long-voter-compare.ipynb` - long sequence voter inference under noise
- `AGENTS.md` - comparative benchmarking invariants, denominator requirements, and controlled noise specifications
