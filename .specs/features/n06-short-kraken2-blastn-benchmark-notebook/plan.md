# n06-short-kraken2-blastn-benchmark-notebook

## Problem

In the original manuscript and historical benchmarking notebook (`short/benchmark_cnns_kraken2_blast_overlap.ipynb`), viral 1D CNNs were compared against classic metagenomic baselines (Kraken2 and BLASTN) across short 512 bp fragments. However, several critical methodological and reproducibility limitations affected this historical setup:

1. **Coupling to monolithic legacy scripts**: the historical notebook contains inline pip installations, local bash commands with hardcoded cluster paths, and unmodularized custom parsing functions.
2. **Conflating database lookup with generalizable classification**: approximately 95.89% of short test fragments share exact 512 bp sequence identity with training references (`exact_overlap.csv`). Without stratifying evaluation queries into `all`, `overlap`, and `no_exact_overlap`, exact database matching by BLASTN and Kraken2 is conflated with generalizable taxonomic classification (AD-011).
3. **Denominator distortions on baseline abstentions**: Kraken2 yields unclassified queries (`status == 'U'`) and BLASTN yields queries without significant alignments (E-value / bitscore threshold). Filtering out abstentions inflates accuracy (conditional accuracy) rather than reflecting standard closed-set classification accuracy where abstention (`y_pred == -1`) counts as an error (AD-011).
4. **Mismatched query sets and lack of paired statistical rigor**: historical comparisons did not consistently enforce evaluation on the exact same query IDs across noise levels $\epsilon \in \{0.0, 0.02, 0.05, 0.10, 0.15\}$, nor did they compute paired discordance or cluster-bootstrap confidence intervals (2,000 repeats) resampling species with replacement to account for intra-species query correlation.
5. **Absence of a clean reproduction notebook**: there is currently no clean reproduction notebook in `notebooks/reproduction/short/` reproducing `short/benchmark_cnns_kraken2_blast_overlap.ipynb` using `viral_cnn.baselines.*`.

When this feature ships, researchers will have a clean, modular reproduction notebook (`09_benchmark_cnns_kraken2_blast_overlap.ipynb`) that consumes `viral_cnn.baselines.*`, runs end-to-end on remote JupyterHub or locally under `SMOKE_TEST=1`, enforces exact overlap stratification (`all`, `overlap`, `no_exact_overlap`), adheres to the strict single-denominator rule ($N$), computes paired discordance and cluster bootstrap confidence intervals, and exports verified benchmark artifacts to `results_reproduction/baselines/`.

## Flow

This feature reuses taxonomy and output parsing from `viral_cnn.baselines.taxonomy`, exact sequence overlap auditing from `viral_cnn.baselines.overlap`, paired evaluation and cluster bootstrap from `viral_cnn.baselines.paired`, optional $k$-mer frequency classification from `viral_cnn.baselines.kmer_baseline`, and environment management from `viral_cnn.notebook_runtime`.

1. Raw Kraken2 reports -> `viral_cnn.baselines.taxonomy:parse_kraken_output` (exists) - maps classified queries to zero-based species labels and unclassified/root to -1.
2. Raw BLASTN hits -> `viral_cnn.baselines.taxonomy:parse_blast_output` (exists) - parses top-5 candidate hits per query by bitscore and maps unaligned queries to -1.
3. Query and training reference sequences -> `viral_cnn.baselines.overlap:audit_exact_overlap` (exists) - generates canonical SHA256 hashes and stratifies queries into `all`, `overlap`, and `no_exact_overlap`.
4. CNN, Kraken2, and BLASTN predictions -> `viral_cnn.baselines.paired:evaluate_predictions` (exists) - computes Top-1, Top-5, Coverage, and Conditional Top-1 accuracy with Wilson 95% confidence intervals under strict single denominator $N$.
5. Paired comparison -> `viral_cnn.baselines.paired:paired_discordance` and `bootstrap_species_difference` (exists) - quantifies discordant predictions on identical query IDs and computes 95% cluster-bootstrap confidence intervals (2,000 repeats, seed 42).
6. Optional k-mer baseline -> `viral_cnn.baselines.kmer_baseline:KmerCentroidClassifier` (exists) - evaluates $k$-mer frequency centroid classification across overlap strata.
7. Benchmark notebook orchestration -> `notebooks/reproduction/short/09_benchmark_cnns_kraken2_blast_overlap.ipynb` (new, door 1) - coordinates loading, stratification, evaluation across noise levels, generation of comparison tables and figures, and manifest emission.

## Impact

| Front | What changes |
| --- | --- |
| domain | existing modules: `viral_cnn.baselines.overlap`, `viral_cnn.baselines.taxonomy`, `viral_cnn.baselines.paired`, `viral_cnn.baselines.kmer_baseline` - directly consumed by reproduction notebook 09 |
| stored data | nothing to migrate - remote benchmark outputs in `cnn_kraken/benchmark_kraken_blast_20_per_species` remain untouched; reproduction benchmark tables and figures are written to `results_reproduction/baselines/` |

## Relations

None - no stored-data shape change

## Surface

None - nothing consumed outside

## Landing

| One-way door | Literal shape | Alternative rejected |
| --- | --- | --- |
| Comparative benchmark reproduction notebook | `notebooks/reproduction/short/09_benchmark_cnns_kraken2_blast_overlap.ipynb` | Modifying historical notebook `short/benchmark_cnns_kraken2_blast_overlap.ipynb`; rejected because historical notebooks must remain immutable evidence per AD-004. |
| Strict single-denominator abstention policy | `accuracy = (y_pred == y_true).sum() / N_total` with `y_pred == -1` counted as error alongside explicit `coverage = (y_pred >= 0).sum() / N_total` | Filtering out unclassified queries from denominator; rejected because conditional accuracy artificially distorts comparative benchmarking against models that do not abstain (AD-011). |
| Overlap tripartite stratification | Stratification of test queries into `all` (100%), `overlap` (train_count > 0), and `no_exact_overlap` (train_count == 0) | Aggregated comparison without overlap partition; rejected because ~95.89% exact sequence identity in fragment split conflates database lookup with generalizable classification (AD-011). |

- Nothing else in this change is hard to reverse

## Criteria

Grouped by slice - one observable outcome each, never a layer.

### S1: Overlap Stratification, Parsers & Unified Evaluator Integration in Notebook 09 (P1)

Creates `notebooks/reproduction/short/09_benchmark_cnns_kraken2_blast_overlap.ipynb` reproducing exact overlap auditing, Kraken2/BLASTN output integration, and stratified evaluation across noise levels $\epsilon \in \{0.0, 0.02, 0.05, 0.10, 0.15\}$.

**Acceptance Criteria**

1. WHEN `09_benchmark_cnns_kraken2_blast_overlap.ipynb` executes under `SMOKE_TEST=1` THEN the system SHALL complete execution without exceptions, compute stratified metrics across `all`, `overlap`, and `no_exact_overlap`, and save artifacts to `results_reproduction/baselines/`.
2. The system SHALL ensure `09_benchmark_cnns_kraken2_blast_overlap.ipynb` imports all baseline and evaluation logic exclusively from `viral_cnn.baselines.*` and contains zero imports of `short/utils/*`.
3. WHEN baseline predictions are evaluated in `09_benchmark_cnns_kraken2_blast_overlap.ipynb` THEN the system SHALL count abstentions (`label == -1`) as errors in the total accuracy denominator while reporting coverage and conditional accuracy separately.
4. WHEN exact sequence overlap is audited in `09_benchmark_cnns_kraken2_blast_overlap.ipynb` THEN the system SHALL partition queries into `all`, `overlap`, and `no_exact_overlap` using `viral_cnn.baselines.overlap:audit_exact_overlap`.
5. WHEN evaluation metrics are computed across methods and noise levels in `09_benchmark_cnns_kraken2_blast_overlap.ipynb` THEN the system SHALL export the stratified metrics table (`benchmark_metrics_stratified.csv`) containing Top-1 accuracy, Wilson 95% CI, Coverage, and Conditional Top-1 accuracy.
6. The system SHALL preserve historical notebook `short/benchmark_cnns_kraken2_blast_overlap.ipynb` strictly unmodified.

**Independent test:** Run execution of `09_benchmark_cnns_kraken2_blast_overlap.ipynb` under `SMOKE_TEST=1` and verify generation of `benchmark_metrics_stratified.csv` and valid JSON structure.

### S2: Paired Discordance, Species Cluster Bootstrap & K-mer Baseline (P2)

Integrates paired discordance analysis on identical query IDs, cluster bootstrap confidence intervals resampling species with replacement, and optional $k$-mer centroid baseline.

**Acceptance Criteria**

7. WHEN paired comparisons between CNNs, Kraken2, and BLASTN are executed in `09_benchmark_cnns_kraken2_blast_overlap.ipynb` THEN the system SHALL compute paired discordance counts on the exact same query IDs using `viral_cnn.baselines.paired:paired_discordance` and export `paired_discordance.csv`.
8. WHEN paired difference confidence intervals are estimated in `09_benchmark_cnns_kraken2_blast_overlap.ipynb` THEN the system SHALL perform cluster bootstrap resampling species with replacement across 2,000 repeats with seed 42 via `viral_cnn.baselines.paired:bootstrap_species_difference` and export `species_cluster_bootstrap.csv`.
9. WHEN the optional $k$-mer baseline is evaluated in `09_benchmark_cnns_kraken2_blast_overlap.ipynb` THEN the system SHALL execute `KmerCentroidClassifier` on evaluation queries and record its stratified performance.
10. WHEN execution completes in `09_benchmark_cnns_kraken2_blast_overlap.ipynb` THEN the system SHALL generate comparison figure `figure_benchmark_overlap.png` and export execution manifest `manifest_09_baselines.json` documenting methods, noise levels, and evaluation parameters.

**Independent test:** Verify existence and schema of `paired_discordance.csv`, `species_cluster_bootstrap.csv`, `figure_benchmark_overlap.png`, and `manifest_09_baselines.json` generated by `09_benchmark_cnns_kraken2_blast_overlap.ipynb`.

## Out of scope

| Excluded | Why |
| --- | --- |
| Rebuilding full Kraken2 database indexes on disk locally | Local environment lacks RAM/disk for 13,683 species Kraken2 indexes; precomputed outputs from remote storage or synthetic fixtures are used per AD-005/AD-006 |
| Running local BLASTN binaries on millions of queries | Precomputed BLASTN tabular results from remote storage or synthetic fixtures are used per AD-005/AD-006 |
| In-place modification of historical notebook `short/benchmark_cnns_kraken2_blast_overlap.ipynb` | Historical notebooks are preserved as immutable evidence per AD-004 |

## Assumptions

| Assumption | Chosen default | Rationale | Confirmed? |
| --- | --- | --- | --- |
| Noise levels evaluated | 0.0, 0.02, 0.05, 0.10, 0.15 | Matches published evaluation grid across CNNs, Kraken2, and BLASTN | y |
| Bootstrap repeats | 2000 repeats with seed 42 | Matches parameterization in `short/benchmark_cnns_kraken2_blast_overlap.ipynb` cell 30 | y |
| Abstention label | -1 | Standard sentinel for unclassified/no-hit queries in `viral_cnn.baselines.taxonomy` | y |
| Overlap strata | all, overlap, no_exact_overlap | Tripartite partition established in `short/benchmark_cnns_kraken2_blast_overlap.ipynb` cell 9 | y |

**Open questions:** none - all resolved or logged above.

## Observable

| Surface | Decision | Landing |
| --- | --- | --- |
| Reproduction notebook 09 | execution exit 0, outputs stratified metrics, paired discordance | AC 1 |
| Stratified metrics CSV | schema with `method`, `noise_rate`, `stratum`, `accuracy`, `wilson_low`, `wilson_high`, `coverage` | AC 5 |
| Paired discordance CSV | schema with `method_a`, `method_b`, `noise_rate`, `stratum`, `only_a`, `only_b`, `both_correct`, `both_wrong` | AC 7 |
| Cluster bootstrap CSV | schema with `method_a`, `method_b`, `noise_rate`, `diff_mean`, `ci_lower`, `ci_upper` | AC 8 |

## Sources

- Manuscript Section 3.2, Section 3.3, and comparative baselines discussion: Published comparison between 1D CNNs, Kraken2, and BLASTN.
- Historical notebook `short/benchmark_cnns_kraken2_blast_overlap.ipynb`: Reference implementation for exact overlap auditing, baseline output parsing, and paired comparison.
- Decision `AD-011`: Protocolo unificado de baselines Kraken2 e BLASTN com denominador estrito e estratificação de overlap/genoma.
- Repository scientific invariant rule `AGENTS.md` and Roadmap `ROADMAP_REPRODUCAO_NOTEBOOKS.md` Section N06.
