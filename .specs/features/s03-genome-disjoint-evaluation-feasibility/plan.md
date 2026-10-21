# S03: Genome-Disjoint Evaluation Feasibility

## Problem

The published manuscript evaluates 1D CNNs (ResNet50-1D and ConvNeXt-1D) on an 80/10/10 stratified split by fragment across 13,683 viral species, where the S02 audit demonstrated that 95.91% of test queries share identical sequence matches with training fragments.
While the published closed-set results reflect fragment identification capability, measuring true taxonomic generalization to unobserved genomes within known species requires evaluating fragments from independent genomes that were never exposed during training.
However, the codebase currently lacks an automated tool to perform genome-disjoint grouping, and public databases contain only a single reference genome for thousands of viral species.
For single-genome species, closed-set genome-disjoint testing is mathematically impossible: placing the single genome in train leaves test with zero samples, while placing it in test violates closed-set supervision.
Attempting to construct a grouped split without a formal feasibility audit risks silently dropping classes, misrepresenting the evaluated population, or creating invalid cross-partition leaks.

When this ships, the feasibility and taxonomic coverage of genome-disjoint evaluation are formally codified in `docs/genome_split_feasibility.md`, a grouped splitting engine (`src/viral_cnn/data/grouped_split.py`) guarantees zero genome accession overlap between train and test partitions while accounting for single-genome classes, and a CLI tool (`scripts/build_grouped_split.py`) generates auditable split manifests and JSON feasibility reports with proof-backed deterministic contract tests.

## Flow

Reuses `src/viral_cnn/taxonomy/mapping.py:load_taxonomy_mapping` while preserving existing files in place.

1. input genome-annotated sequences or metadata -> `viral_cnn.data.grouped_split` (new, door 1) - groups fragments by `genome_id` / `accession`, counts genomes per species, and isolates single-genome classes
2. `viral_cnn.data.grouped_split` -> `viral_cnn.data.coverage_analyzer` (new, door 2) - computes taxonomic coverage, clade distribution, and assigns multi-genome classes to disjoint train/val/test splits
3. `scripts/build_grouped_split.py` (new, no door - placement per conventions) - CLI interface parsing inputs, executing grouped split, and generating partition manifests
4. out: structured JSON feasibility report and partition manifests with zero genome overlap (door 3)

## Impact

| Front | What changes |
| --- | --- |
| domain | new term: `GenomeDisjointSplit` - partition protocol where all sequence fragments derived from a specific genome accession belong exclusively to a single split (train, val, or test), lives in `src/viral_cnn/data/grouped_split.py` |
| domain | new term: `PopulationCoverageShift` - the reduction in evaluated species count and change in baseline population when restricting closed-set testing to species with $\ge 2$ independent reference genomes |
| stored data | nothing to migrate: historical datasets in `dataset_kraken/` remain read-only; new grouped split manifests generated in `configs/splits/` or user-specified output directories |

## Relations

None - no stored-data shape change

## Surface

None - nothing consumed outside

## Landing

| One-way door | Literal shape | Alternative rejected |
| --- | --- | --- |
| grouped partition constraint | `set(train_genomes) ∩ set(test_genomes) == ∅` with `min_genomes_per_class = 2` | relaxed splitting allowing partial genome overlap or fragment-level fallback - rejected because any shared genome between train and test reintroduces the 95.91% fragment leakage identified in S02 |
| infeasible single-genome class policy | explicit separation into `train_only` or excluded category with documented population delta | silently placing single-genome species in test with zero training samples (violates closed-set classification) or in train and test simultaneously (violates genome-disjoint constraint) |
| CLI grouped split output schema | stdout/JSON emits `summary` (`total_species`, `eligible_multi_genome_species`, `infeasible_single_genome_species`, `coverage_pct`), `partitions` (`train_genomes`, `test_genomes`, `genome_overlap_count`), and manifest file paths | plain unstructured console text - rejected because automated verification scripts and future training pipelines require structured machine-readable manifests |

- Nothing else in this change is hard to reverse

## Criteria

### S1: Genome-Disjoint Feasibility Analysis and Manifest (P1)

Formal mathematical and biological feasibility audit for genome-disjoint viral classification.

**Acceptance Criteria**

1. The system SHALL document the mathematical and biological feasibility criteria for genome-disjoint viral classification in `docs/genome_split_feasibility.md`, proving that closed-set evaluation without genome leakage requires $\ge 2$ independent reference genomes per species.
2. The system SHALL document in `docs/genome_split_feasibility.md` that single-genome species cannot undergo disjoint train/test evaluation under closed-set constraints, recording the resulting taxonomic coverage and population shift relative to the published 13,683-species benchmark.
3. The system SHALL record in `docs/genome_split_feasibility.md` the protocol requirement stating that grouped genome accuracies cannot be directly compared against fragment-split numbers without explicitly reporting the population difference.

**Independent test:** run verification script asserting that `docs/genome_split_feasibility.md` exists, records the $\ge 2$ genome requirement, details the population shift, and documents protocol comparability constraints.

### S2: Grouped Split Partitioning Engine (P1)

Core module for genome-disjoint grouping, zero-leakage assignment, and coverage analysis.

**Acceptance Criteria**

4. WHEN given a dataset containing sequence records, class identifiers, and genome accession group identifiers THEN the system SHALL construct a grouped partition such that all fragments belonging to any single genome accession reside exclusively within either train, validation, or test.
5. The system SHALL guarantee zero genome accession overlap between train and test partitions (`set(train_genomes) ∩ set(test_genomes) == ∅`).
6. WHEN partitioning a dataset containing species with only 1 genome THEN the system SHALL declare those species infeasible for disjoint test evaluation and assign them to an explicit `train_only` or excluded category.
7. The system SHALL calculate and report taxonomic coverage metrics including total species, eligible multi-genome species, infeasible single-genome species, and retained fragment counts.

**Independent test:** run unit tests in `tests/test_grouped_split.py` asserting zero genome overlap between train and test, proper isolation of single-genome classes, and accurate coverage calculations on synthetic test fixtures.

### S3: Grouped Split CLI and Plan (P1)

Executable CLI tool generating disjoint split manifests and feasibility reports.

**Acceptance Criteria**

8. WHEN executing `scripts/build_grouped_split.py --input <path> --output-dir <path>` THEN the system SHALL return exit code 0 and emit disjoint train, test, and validation partition manifests alongside a structured JSON feasibility report.
9. IF an invalid input path, missing genome group column, or unsupported parameter is passed to `scripts/build_grouped_split.py` THEN the system SHALL exit with code 2 and emit an informative error message to standard error.
10. The system SHALL verify all grouped split partitioning routines against deterministic synthetic test fixtures without requiring full model retraining or massive cluster downloads.

**Independent test:** run `python3 scripts/build_grouped_split.py --input tests/fixtures/sample_genomes.csv --output-dir /tmp/test_grouped_out` and assert exit code 0 with zero genome overlap in the generated manifest.

## Out of scope

| Excluded | Why |
| --- | --- |
| Execution of full model retraining on new grouped split | S03 is dedicated to feasibility analysis, coverage audit, and split construction; model retraining requires dedicated GPU compute and separate execution budgets under S04 |
| Replacing the published 80/10/10 fragment split baseline | Scientific invariant in AGENTS.md mandates preserving the published baseline split; grouped split serves as a secondary generalization benchmark with explicit population reporting |
| Real-time NCBI Entrez downloading | Direct network calls in unit tests violate repeatability; pipeline uses existing metadata and synthetic fixtures |

## Assumptions

| Assumption | Chosen default | Rationale | Confirmed? |
| --- | --- | --- | --- |
| Minimum reference genomes required for disjoint evaluation | 2 independent genomes per class | At least 1 genome must be in train and at least 1 in test to satisfy closed-set supervision | y |
| Grouped split proportions | 80% train, 20% test (or 80/10/10) grouped by accession | Standard machine learning benchmark proportions | y |
| Execution mode for unit tests | Deterministic synthetic sequence and accession fixtures | Fast CI execution without requiring external multi-gigabyte downloads | y |

**Open questions:** none - all resolved or logged above.

## Observable

| Surface | Decision | Landing |
| --- | --- | --- |
| CLI `scripts/build_grouped_split.py` | CLI flags and options (`--input`, `--output-dir`, `--group-col`, `--class-col`, `--test-size`, `--min-genomes`) | AC 8 |
| CLI `scripts/build_grouped_split.py` | output format and structured JSON report schema | AC 8 |
| CLI `scripts/build_grouped_split.py` | error handling and exit codes (exit 0 on success, exit 2 on bad arguments or missing file) | AC 9 |
| screen | n/a - this feature exposes no user-facing GUI; all interaction occurs via CLI and Python API |

## Sources

- Manuscript: "Spatial one-hot nucleotide representations of DNA enable robust and interpretable taxonomic classification with 1D convolutional networks" (Biodata Mining / submitted)
- Project rule: `AGENTS.md` (Scientific invariants: Closed-set, generalização por genoma, preservação do split histórico)
- Roadmap: `ROADMAP_SPECS_ARTIGO.md` (Spec S03: Viabilidade de avaliação por genoma de origem)

