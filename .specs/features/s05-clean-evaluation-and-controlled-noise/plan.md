# S05: Clean Evaluation and Controlled Noise Perturbation

## Problem

The manuscript's core empirical claims rest on comparing 1D CNNs against each other and against baselines across clean data and controlled synthetic nucleotide corruptions ($\epsilon \in \{0.0, 0.02, 0.05, 0.10, 0.15\}$).
However, evaluation and noise generation logic currently exist across disconnected notebook cells (`short/finetune_resnet50_species.ipynb`, `short/finetune_convnext_species.ipynb`, and `short/benchmark_cnns_kraken2_blast_overlap.ipynb`) without a standardized, modular execution pipeline.
Under the historical perturbation protocol, positions are selected with Bernoulli probability $\epsilon$ and resampled uniformly from $\{A, C, G, T\}$.
Because resampling includes the original nucleotide with probability $1/4$, the expected effective substitution rate is $3\epsilon/4$ (e.g., $\epsilon=0.05 \implies 3.75\%$ effective substitution rate).
Without a verified implementation, there is a risk of confusing nominal noise rate $\epsilon$ with effective error rate, inadvertently corrupting padding or ambiguous tokens, losing query identifiers across evaluations, or computing inconsistent metrics (e.g. confounding Macro-F1 with Weighted-F1, using discontinuous F1 performance bands, or conflating fragment-level bootstrap confidence intervals with genome-level variance).

When this ships, `docs/noise_evaluation_protocol.md` provides an auditable protocol specification for clean and noise evaluation; `src/viral_cnn/eval/noise.py` implements the historical Bernoulli corruption with verified effective substitution rate $3\epsilon/4$; `src/viral_cnn/eval/metrics.py` calculates Top-1, Top-5, Macro-F1, Weighted-F1, continuous F1 bands, and paired percentage-point differences; `src/viral_cnn/eval/runner.py` orchestrates evaluations and saves query-level predictions; `scripts/run_evaluation.py` provides a standalone CLI producing reproducible evaluation manifests; and deterministic synthetic contract tests verify all calculations.

## Flow

Reuses `src/viral_cnn/models/registry.py` and `src/viral_cnn/data/encoding.py` while preserving existing architecture definitions in place.

1. input sequences / test queries and nominal noise rate $\epsilon \in \{0.0, 0.02, 0.05, 0.10, 0.15\}$ -> `viral_cnn.eval.noise:inject_noise` (new, door 1) - applies Bernoulli selection and uniform A/C/G/T resampling with expected effective rate $3\epsilon/4$ while preserving padding
2. noisy or clean inputs -> `viral_cnn.eval.runner:evaluate_model` (new, door 2) - performs batched forward passes, collects Top-1, Top-5, probabilities, and writes structured query predictions
3. predictions and ground truth labels -> `viral_cnn.eval.metrics:compute_classification_metrics` (new, door 3) - calculates Top-1 accuracy, Top-5 accuracy, cross-entropy loss, Macro-F1, Weighted-F1, continuous per-species F1 bands, and paired comparisons
4. `scripts/run_evaluation.py` (new, no door - placement per conventions) - CLI executing evaluation runs and emitting JSON metrics summaries and prediction tables
5. out: auditable prediction records, metrics summary JSON, and execution manifest with reproducible seeds and hashes

## Impact

| Front | What changes |
| --- | --- |
| domain | new term: `EffectiveNoiseRate` - the actual substitution rate under uniform 4-base resampling, mathematically equal to $(3/4)\epsilon$, distinguished from the nominal Bernoulli rate $\epsilon$ |
| domain | new term: `ContinuousF1Bands` - partitioned performance bins for per-species F1 scores ($[0.0, 0.5), [0.5, 0.8), [0.8, 0.95), [0.95, 1.0]$) that are continuous and exhaustive without interval gaps |
| domain | new term: `PairedComparisonDifference` - pairwise percentage-point delta ($\Delta_{pp} = 100 \times (\text{acc}_A - \text{acc}_B)$) evaluated on identical query subsets with contingency counts (`both_correct`, `only_a`, `only_b`, `both_wrong`) |
| stored data | nothing to migrate: historical notebooks and parquet splits remain unmodified; evaluation runs write structured prediction files to user-configured output directories |

## Relations

None - no stored-data shape change

## Surface

None - nothing consumed outside

## Landing

| One-way door | Literal shape | Alternative rejected |
| --- | --- | --- |
| Bernoulli nucleotide corruption contract | `mask = torch.rand(seq.shape) < epsilon` followed by `random_bases = torch.randint(0, 4, seq.shape)` and `torch.where(mask & (seq >= 0), random_bases, seq)` | strict non-identical base substitution (which would yield effective rate $\epsilon$ rather than the manuscript's $3\epsilon/4$) or perturbing padding tokens - rejected because it violates historical protocol fidelity |
| continuous F1 performance bands definition | four continuous intervals: `[0.0, 0.50)`, `[0.50, 0.80)`, `[0.80, 0.95)`, `[0.95, 1.00]` | overlapping or disjoint intervals (e.g. `0.5-0.79` and `0.81-0.94`) - rejected because discrete gaps drop borderline species and invalidate total class counts |
| query-level prediction artifact schema | parquet/jsonl rows containing `query_id`, `truth_label`, `predicted_label`, `top5_labels`, `confidence`, `noise_rate`, and `seed` | saving only aggregated scalar metrics without individual query predictions - rejected because downstream calibration, error analysis, and paired comparisons require query-level outputs |

- Nothing else in this change is hard to reverse

## Criteria

### S1: Controlled Noise Injection Engine (P1)

Implementation and mathematical verification of the historical nucleotide perturbation protocol.

**Acceptance Criteria**

1. The system SHALL document the controlled noise perturbation protocol in `docs/noise_evaluation_protocol.md`, recording nominal rates $\epsilon \in \{0.0, 0.02, 0.05, 0.10, 0.15\}$, the uniform 4-base resampling mechanism, and the mathematical proof that expected effective substitution rate is $3\epsilon/4$.
2. WHEN given an input sequence tensor and a nominal noise rate $\epsilon$ THEN the system SHALL select positions with Bernoulli probability $\epsilon$ and replace them with bases drawn uniformly from $\{0, 1, 2, 3\}$.
3. The system SHALL preserve padding and ambiguous positions (token index $< 0$) without applying noise perturbations.
4. The system SHALL verify that empirical substitution rates across large synthetic sequences converge to $(3/4)\epsilon$ within a tolerance of $\pm 0.005$.

**Independent test:** run unit tests in `tests/test_noise_and_evaluation.py` asserting that empirical mutation rates match $(3/4)\epsilon$, padding is preserved, and random seeds produce deterministic outputs.

### S2: Comprehensive Metrics and Paired Comparisons (P1)

Calculation of multi-class accuracy, Top-5, Macro/Weighted F1, continuous performance bands, and paired deltas.

**Acceptance Criteria**

5. WHEN evaluated on predicted labels and ground-truth classes THEN the system SHALL calculate Top-1 accuracy, Top-5 accuracy, Macro-F1, and Weighted-F1 using the declared number of taxonomic classes.
6. The system SHALL partition per-species F1 scores into four continuous, gapless performance bands: $[0.0, 0.50)$, $[0.50, 0.80)$, $[0.80, 0.95)$, and $[0.95, 1.00]$, ensuring the sum of species across all bands equals the total evaluated species count.
7. WHEN comparing predictions from two models on identical queries THEN the system SHALL calculate the pairwise difference in percentage points ($\Delta_{pp}$) alongside paired contingency counts (`both_correct`, `only_a_correct`, `only_b_correct`, `both_wrong`).
8. The system SHALL document in `docs/noise_evaluation_protocol.md` that fragment-level bootstrap intervals measure sampling variation of the test queries and SHALL NOT be presented as uncertainty across independent genomes or training seeds.

**Independent test:** run unit tests in `tests/test_noise_and_evaluation.py` verifying metric calculations, gapless F1 band partitions, and paired contingency calculations against known synthetic ground truth.

### S3: Evaluation Runner and CLI Interface (P1)

Modular evaluation runner and executable CLI producing auditable prediction files and metrics manifests.

**Acceptance Criteria**

9. WHEN executing `scripts/run_evaluation.py` with an input dataset, model variant, and noise level THEN the system SHALL return exit code 0, generate structured query-level prediction records, and output a JSON metrics summary.
10. IF an unsupported noise level, invalid input path, or incompatible model variant is passed to `scripts/run_evaluation.py` THEN the system SHALL exit with code 2 and emit an informative error message to standard error.
11. The system SHALL verify all evaluation, metric computation, and noise injection routines with deterministic synthetic fixtures without requiring GPU hardware or multi-gigabyte remote checkpoint downloads.

**Independent test:** run `python3 scripts/run_evaluation.py --model-variant resnet_clean_species --input tests/fixtures/sample_genomes.csv --noise 0.05 --output-dir /tmp/test_eval_out --seed 42` and assert exit code 0 and valid output files.

## Out of scope

| Excluded | Why |
| --- | --- |
| Execution of multi-gigabyte evaluation over all 256 parquet shards | Full dataset evaluation runs on remote JupyterHub environment (AD-006); tests verify pipeline mechanics and numerical accuracy via deterministic synthetic fixtures (AD-005) |
| Simulating Oxford Nanopore insertions and deletions | As recorded in AGENTS.md, the historical noise model is point substitution only; indels and context-dependent ONT models are separate extensions |
| Post-hoc temperature scaling or calibration optimization | S05 computes raw predictions and standard metrics; formal calibration (ECE, reliability diagrams) is addressed in S09 |

## Assumptions

| Assumption | Chosen default | Rationale | Confirmed? |
| --- | --- | --- | --- |
| Random seed for evaluation noise realization | 42 | Standard project random seed specified across historical notebooks | y |
| Default nominal noise grid | 0.0, 0.02, 0.05, 0.10, 0.15 | The exact five noise levels evaluated in manuscript Section 3 | y |
| Treatment of unpredicted / abstaining queries | assigned label -1, treated as incorrect for accuracy and Top-5 | Consistent with Kraken2/BLASTN evaluation protocol in S07 | y |

**Open questions:** none - all resolved or logged above.

## Observable

| Surface | Decision | Landing |
| --- | --- | --- |
| Python API `viral_cnn.eval.noise` | function `inject_noise(sequences, noise_rate, num_features=4)` | AC 2 |
| Python API `viral_cnn.eval.metrics` | functions `compute_classification_metrics`, `compute_species_f1_bands`, `compute_paired_comparison` | AC 5, 6, 7 |
| Python API `viral_cnn.eval.runner` | function `evaluate_model` returning structured `EvaluationResult` | AC 9 |
| CLI `scripts/run_evaluation.py` | CLI flags (`--input`, `--model-variant`, `--noise`, `--output-dir`, `--seed`, `--batch-size`) | AC 9 |
| CLI `scripts/run_evaluation.py` | error handling and exit codes (exit 0 on success, exit 2 on bad arguments) | AC 10 |
| screen | n/a - this feature exposes no user-facing GUI; all interaction occurs via CLI and Python API |

## Sources

- Manuscript: "Spatial one-hot nucleotide representations of DNA enable robust and interpretable taxonomic classification with 1D convolutional networks" (Biodata Mining / submitted)
- Project rule: `AGENTS.md` (Scientific invariants: Ruído, Métricas, análise e figuras, Treinamento com ruído)
- Historical notebooks: `short/finetune_resnet50_species.ipynb`, `short/finetune_convnext_species.ipynb`, `short/benchmark_cnns_kraken2_blast_overlap.ipynb`
- Roadmap: `ROADMAP_SPECS_ARTIGO.md` (Spec S05: Avaliação limpa e corrupção controlada reproduzíveis)
