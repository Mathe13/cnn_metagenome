# n06-short-kraken2-blastn-benchmark-notebook checks

Profile: light
Plan: `.specs/features/n06-short-kraken2-blastn-benchmark-notebook/plan.md`

10 checks in 2 slices · 3 one-way doors · 0 open, of which 0 block

## Checks

Grouped by the spec's slices; numbering runs across the whole feature.

### S1 - Overlap Stratification, Parsers & Unified Evaluator Integration in Notebook 09 · 3 files · 45 KB · ~18k

**C1** - `09_benchmark_cnns_kraken2_blast_overlap.ipynb` contains valid JSON, zero imports of `short/utils/*`, and imports exclusively from `viral_cnn.*`
Proof: `MPLCONFIGDIR=/tmp PYTHONPATH=src SMOKE_TEST=1 .venv/bin/python3 -m unittest -k test_c1_notebook_09_imports_and_decoupling tests/test_n06_kraken2_blastn_benchmark.py`

**C2** - `09_benchmark_cnns_kraken2_blast_overlap.ipynb` executes end-to-end under `SMOKE_TEST=1` in under 60 seconds without exceptions
Proof: `MPLCONFIGDIR=/tmp PYTHONPATH=src SMOKE_TEST=1 .venv/bin/python3 -m unittest -k test_c2_notebook_09_execution_smoke_test tests/test_n06_kraken2_blastn_benchmark.py`

**C3** - Baseline predictions count abstentions (`label == -1`) as errors in total accuracy denominator while reporting coverage and conditional accuracy separately
Proof: `MPLCONFIGDIR=/tmp PYTHONPATH=src SMOKE_TEST=1 .venv/bin/python3 -m unittest -k test_c3_abstention_accounting_and_coverage tests/test_n06_kraken2_blastn_benchmark.py`

**C4** - Exact sequence overlap audit partitions queries into `all`, `overlap`, and `no_exact_overlap`
Proof: `MPLCONFIGDIR=/tmp PYTHONPATH=src SMOKE_TEST=1 .venv/bin/python3 -m unittest -k test_c4_overlap_tripartite_stratification tests/test_n06_kraken2_blastn_benchmark.py`

**C5** - Stratified metrics exported to `results_reproduction/baselines/benchmark_metrics_stratified.csv` with Top-1, Wilson 95% CI, Coverage, and Conditional Top-1
Proof: `MPLCONFIGDIR=/tmp PYTHONPATH=src SMOKE_TEST=1 .venv/bin/python3 -m unittest -k test_c5_stratified_metrics_export tests/test_n06_kraken2_blastn_benchmark.py`

**C6** - Historical notebook `short/benchmark_cnns_kraken2_blast_overlap.ipynb` remains strictly unmodified (`git status --porcelain short/benchmark_cnns_kraken2_blast_overlap.ipynb` empty)
Proof: `MPLCONFIGDIR=/tmp PYTHONPATH=src SMOKE_TEST=1 .venv/bin/python3 -m unittest -k test_c6_historical_benchmark_notebook_preserved tests/test_n06_kraken2_blastn_benchmark.py`

### S2 - Paired Discordance, Species Cluster Bootstrap & K-mer Baseline · 3 files · 50 KB · ~20k

**C7** - Paired comparisons between methods compute paired discordance counts on identical query IDs and export `results_reproduction/baselines/paired_discordance.csv`
Proof: `MPLCONFIGDIR=/tmp PYTHONPATH=src SMOKE_TEST=1 .venv/bin/python3 -m unittest -k test_c7_paired_discordance_on_same_queries tests/test_n06_kraken2_blastn_benchmark.py`

**C8** - Paired difference confidence intervals estimated via cluster bootstrap resampling species with replacement across 2,000 repeats (seed 42) and exported to `results_reproduction/baselines/species_cluster_bootstrap.csv`
Proof: `MPLCONFIGDIR=/tmp PYTHONPATH=src SMOKE_TEST=1 .venv/bin/python3 -m unittest -k test_c8_species_cluster_bootstrap tests/test_n06_kraken2_blastn_benchmark.py`

**C9** - Optional $k$-mer baseline evaluated via `KmerCentroidClassifier` and recorded across overlap strata
Proof: `MPLCONFIGDIR=/tmp PYTHONPATH=src SMOKE_TEST=1 .venv/bin/python3 -m unittest -k test_c9_kmer_centroid_baseline tests/test_n06_kraken2_blastn_benchmark.py`

**C10** - Comparison figure `figure_benchmark_overlap.png` and execution manifest `manifest_09_baselines.json` generated under `results_reproduction/baselines/`
Proof: `MPLCONFIGDIR=/tmp PYTHONPATH=src SMOKE_TEST=1 .venv/bin/python3 -m unittest -k test_c10_figure_and_manifest_generated tests/test_n06_kraken2_blastn_benchmark.py`

## Coverage

| Set (size) | Member -> proof | Unproven |
| --- | --- | --- |
| Benchmark evaluation methods (3) | `CNNs` C1, C2, C5 · `Kraken2-flat` C3, C5, C7 · `BLASTN` C3, C5, C7 | - |
| Overlap strata evaluated (3) | `all` C4, C5 · `overlap` C4, C5 · `no_exact_overlap` C4, C5 | - |
| Baseline statistical metrics (4) | `Top-1 with Wilson CI` C5 · `Coverage and Conditional Top-1` C3, C5 · `Paired Discordance` C7 · `Species Cluster Bootstrap (2,000 repeats)` C8 | - |
| Additional baseline classifiers (1) | `KmerCentroidClassifier` C9 | - |
| Historical notebooks preserved (1) | `short/benchmark_cnns_kraken2_blast_overlap.ipynb` C6 | - |

- No check claims more than the single case its proof exercises

## Swept

Where each unwritten requirement landed. All nine, one line each, every time.

- validation: C1 (valid notebook JSON schema and no legacy imports)
- failure modes: C2 (deterministic synthetic data fallback under `SMOKE_TEST=1` when cluster outputs are absent)
- idempotency: C6 (`git status --porcelain short/` empty)
- authorization: n/a - local scientific computation with no remote auth endpoints
- concurrency: n/a - sequential cell execution and deterministic PRNG
- data lifecycle: C5, C7, C8, C10 (`results_reproduction/baselines/` CSVs, PNGs, and JSON manifest)
- dependency failure: C2, C9 (smoke test runs entirely offline with synthetic data)
- state transitions: C4, C5 (raw queries to hash audit to stratified metric calculation)
- observability: C10 (`manifest_09_baselines.json` under `results_reproduction/baselines/`)

## Handoff

S1 = 18k; S2 enters Paired Bootstrap at 38k total, under the 150k budget - one builder

- **Boundary:** C1-C10 closed at HEAD
- **Settled mid-build:** Reused `viral_cnn.baselines.*` modules directly, supporting both remote cluster files and synthetic fixtures under `SMOKE_TEST=1`.
- **Abandoned:** None

