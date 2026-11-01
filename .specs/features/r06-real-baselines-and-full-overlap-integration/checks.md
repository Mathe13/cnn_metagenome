# r06-real-baselines-and-full-overlap-integration checks

Profile: light
Plan: `.specs/features/r06-real-baselines-and-full-overlap-integration/plan.md`

## Checks

### S1 - Robust baseline artifacts loading and validation via load_prepared_baseline_benchmark · 2 files · 20 KB · ~5k

**C1** - `load_prepared_baseline_benchmark` loads `exact_overlap.csv` and all method prediction parquets across noise levels, returning a `BenchmarkArtifacts` object with aligned `audit_df`, boolean `has_overlap` mask, strata masks for `all`, `overlap`, and `no_exact_overlap`, and aligned prediction dictionaries
Proof: `PYTHONPATH=src .venv/bin/python3 -m unittest tests/test_n06_kraken2_blastn_benchmark.py -k test_r06_load_prepared_baseline_benchmark_success`

**C2** - `load_prepared_baseline_benchmark` detects duplicate `query_id` in prediction parquets or mismatched query IDs with the truth table and raises `ValueError`
Proof: `PYTHONPATH=src .venv/bin/python3 -m unittest tests/test_n06_kraken2_blastn_benchmark.py -k test_r06_load_prepared_baseline_benchmark_validation_errors`

**C3** - `load_prepared_baseline_benchmark` raises `FileNotFoundError` when `benchmark_dir` does not exist on disk or when `exact_overlap.csv` is missing
Proof: `PYTHONPATH=src .venv/bin/python3 -m unittest tests/test_n06_kraken2_blastn_benchmark.py -k test_r06_load_prepared_baseline_benchmark_missing_dir`

### S2 - Elimination of fabricated predictions and notebook 09 fail-fast integration · 2 files · 30 KB · ~7k

**C4** - `notebooks/reproduction/short/09_benchmark_cnns_kraken2_blast_overlap.ipynb` raises `FileNotFoundError` in Cell 3 and Cell 5 when `is_smoke_test == False` and `kraken_blast_dir` does not exist, never executing synthetic prediction generation outside smoke mode
Proof: `PYTHONPATH=src .venv/bin/python3 -m unittest tests/test_n06_kraken2_blastn_benchmark.py -k test_r06_notebook_09_fail_fast_outside_smoke`

**C5** - `notebooks/reproduction/short/09_benchmark_cnns_kraken2_blast_overlap.ipynb` does not contain any fabricated predictions or random degradation masks (`acc_target`, `rng.uniform(0, 1, size=num_queries) > acc_target`) when `is_smoke_test == False`
Proof: `PYTHONPATH=src .venv/bin/python3 -m unittest tests/test_n06_kraken2_blastn_benchmark.py -k test_r06_notebook_09_no_synthetic_fabrication_outside_smoke`

**C6** - Notebook 09 executes end-to-end within 60 seconds in smoke test mode (`is_smoke_test == True`), producing exactly 40 balanced queries (20 overlap, 20 non-overlap) and exporting all required CSVs, figures, and manifest with 2,000 bootstrap repeats
Proof: `PYTHONPATH=src .venv/bin/python3 -m unittest tests/test_n06_kraken2_blastn_benchmark.py -k test_r06_notebook_09_smoke_execution_and_exports`

## Coverage

| Set (size) | Member -> proof | Unproven |
| --- | --- | --- |
| Benchmark loader inputs and formats (2) | `synthetic_fixture_directory` C1 · `missing_directory_error` C3 | - |
| Prediction integrity validation checks (2) | `duplicate_query_ids` C2 · `mismatched_query_ids` C2 | - |
| Notebook 09 execution regimes (2) | `smoke_test_mode` C6 · `production_cluster_mode` C4 | - |
| Prediction generation policies (2) | `production_real_artifacts` C5 · `smoke_controlled_synthesis` C6 | - |

- No other check claims more than the single case its proof exercises

## Swept

Where each unwritten requirement landed. All nine, one line each, every time.

- validation: C1, C2 (validates unique query IDs, identical query alignment and boolean overlap masks)
- failure modes: C3, C4 (fails fast with FileNotFoundError when real baseline directories or truth artifacts are missing outside smoke mode)
- idempotency: C1, C6 (deterministic cluster bootstrap with seed 42 and fixed Wilson score confidence intervals)
- authorization: n/a - local and cluster offline scripts with no network credentials
- concurrency: n/a - single-process baseline loading and vectorized evaluation
- data lifecycle: C1, C6 (persists clean CSV and Parquet outputs in results_reproduction/baselines without lingering partial files)
- dependency failure: C4, C5 (prevents silent fallback to synthetic prediction masks when real baseline results are unavailable)
- state transitions: C1, C4 (seamless ingestion of exact overlap audit and model predictions into stratified metric evaluation)
- observability: C1, C6 (prints detailed overlap percentages and export paths, and writes complete manifest_09_baselines.json)

## Handoff

- S1 = ~5k tokens, S2 = ~7k tokens, total ~12k tokens across 4 files (`src/viral_cnn/baselines/overlap.py`, `src/viral_cnn/baselines/__init__.py`, `notebooks/reproduction/short/09_benchmark_cnns_kraken2_blast_overlap.ipynb`, `tests/test_n06_kraken2_blastn_benchmark.py`), well under the 150k budget - one builder
- **Boundary:** C1-C6 closed at HEAD
- **Settled mid-build:** none
- **Abandoned:** none
