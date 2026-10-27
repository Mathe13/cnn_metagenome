# n06-short-kraken2-blastn-benchmark-notebook verification

**Verdict**: PASS
**Profile**: light
**Diff range**: cd20e1f..HEAD
**Round**: 1 - full
**Verifier**: independent sub-agent (author != verifier)

## Checks

| Check | Claim | Proof run | Evidence | Result |
| --- | --- | --- | --- | --- |
| C1 | `09_benchmark_cnns_kraken2_blast_overlap.ipynb` contains valid JSON, zero imports of `short/utils/*`, and imports exclusively from `viral_cnn.*` | `MPLCONFIGDIR=/tmp PYTHONPATH=src SMOKE_TEST=1 .venv/bin/python3 -m unittest -k test_c1_notebook_09_imports_and_decoupling tests/test_n06_kraken2_blastn_benchmark.py` exit 0 | `tests/test_n06_kraken2_blastn_benchmark.py:28` - `self.assertNotIn("short.utils", code)` | PASS |
| C2 | `09_benchmark_cnns_kraken2_blast_overlap.ipynb` executes end-to-end under `SMOKE_TEST=1` in under 60 seconds without exceptions | `MPLCONFIGDIR=/tmp PYTHONPATH=src SMOKE_TEST=1 .venv/bin/python3 -m unittest -k test_c2_notebook_09_execution_smoke_test tests/test_n06_kraken2_blastn_benchmark.py` exit 0 | `tests/test_n06_kraken2_blastn_benchmark.py:50` - `self.assertLess(duration, 60.0)` | PASS |
| C3 | Baseline predictions count abstentions (`label == -1`) as errors in total accuracy denominator while reporting coverage and conditional accuracy separately | `MPLCONFIGDIR=/tmp PYTHONPATH=src SMOKE_TEST=1 .venv/bin/python3 -m unittest -k test_c3_abstention_accounting_and_coverage tests/test_n06_kraken2_blastn_benchmark.py` exit 0 | `tests/test_n06_kraken2_blastn_benchmark.py:62` - `self.assertEqual(metrics["accuracy"], 0.25)` | PASS |
| C4 | Exact sequence overlap audit partitions queries into `all`, `overlap`, and `no_exact_overlap` | `MPLCONFIGDIR=/tmp PYTHONPATH=src SMOKE_TEST=1 .venv/bin/python3 -m unittest -k test_c4_overlap_tripartite_stratification tests/test_n06_kraken2_blastn_benchmark.py` exit 0 | `tests/test_n06_kraken2_blastn_benchmark.py:79` - `self.assertTrue(has_overlap.any(), "Should have some overlap queries")` | PASS |
| C5 | Stratified metrics exported to `results_reproduction/baselines/benchmark_metrics_stratified.csv` with Top-1, Wilson 95% CI, Coverage, and Conditional Top-1 | `MPLCONFIGDIR=/tmp PYTHONPATH=src SMOKE_TEST=1 .venv/bin/python3 -m unittest -k test_c5_stratified_metrics_export tests/test_n06_kraken2_blastn_benchmark.py` exit 0 | `tests/test_n06_kraken2_blastn_benchmark.py:93` - `self.assertTrue(expected_cols.issubset(df.columns))` | PASS |
| C6 | Historical notebook `short/benchmark_cnns_kraken2_blast_overlap.ipynb` remains strictly unmodified (`git status --porcelain short/benchmark_cnns_kraken2_blast_overlap.ipynb` empty) | `MPLCONFIGDIR=/tmp PYTHONPATH=src SMOKE_TEST=1 .venv/bin/python3 -m unittest -k test_c6_historical_benchmark_notebook_preserved tests/test_n06_kraken2_blastn_benchmark.py` exit 0 | `tests/test_n06_kraken2_blastn_benchmark.py:106` - `self.assertEqual(res.stdout.strip(), "")` | PASS |
| C7 | Paired comparisons between methods compute paired discordance counts on identical query IDs and export `results_reproduction/baselines/paired_discordance.csv` | `MPLCONFIGDIR=/tmp PYTHONPATH=src SMOKE_TEST=1 .venv/bin/python3 -m unittest -k test_c7_paired_discordance_on_same_queries tests/test_n06_kraken2_blastn_benchmark.py` exit 0 | `tests/test_n06_kraken2_blastn_benchmark.py:114` - `self.assertIn("method_a", df.columns)` | PASS |
| C8 | Paired difference confidence intervals estimated via cluster bootstrap resampling species with replacement across 2,000 repeats (seed 42) and exported to `results_reproduction/baselines/species_cluster_bootstrap.csv` | `MPLCONFIGDIR=/tmp PYTHONPATH=src SMOKE_TEST=1 .venv/bin/python3 -m unittest -k test_c8_species_cluster_bootstrap tests/test_n06_kraken2_blastn_benchmark.py` exit 0 | `tests/test_n06_kraken2_blastn_benchmark.py:128` - `self.assertIn("method_a", df.columns)` | PASS |
| C9 | Optional $k$-mer baseline evaluated via `KmerCentroidClassifier` and recorded across overlap strata | `MPLCONFIGDIR=/tmp PYTHONPATH=src SMOKE_TEST=1 .venv/bin/python3 -m unittest -k test_c9_kmer_centroid_baseline tests/test_n06_kraken2_blastn_benchmark.py` exit 0 | `tests/test_n06_kraken2_blastn_benchmark.py:147` - `self.assertGreaterEqual(acc_overlap, acc_no_overlap)` | PASS |
| C10 | Comparison figure `figure_benchmark_overlap.png` and execution manifest `manifest_09_baselines.json` generated under `results_reproduction/baselines/` | `MPLCONFIGDIR=/tmp PYTHONPATH=src SMOKE_TEST=1 .venv/bin/python3 -m unittest -k test_c10_figure_and_manifest_generated tests/test_n06_kraken2_blastn_benchmark.py` exit 0 | `tests/test_n06_kraken2_blastn_benchmark.py:163` - `self.assertEqual(manifest.get("bootstrap_repeats"), 2000)` | PASS |

## Coverage

| Set (size) | Member -> proof | Unproven |
| --- | --- | --- |
| Benchmark evaluation methods (3) | `CNNs` C1, C2, C5 · `Kraken2-flat` C3, C5, C7 · `BLASTN` C3, C5, C7 | - |
| Overlap strata evaluated (3) | `all` C4, C5 · `overlap` C4, C5 · `no_exact_overlap` C4, C5 | - |
| Baseline statistical metrics (4) | `Top-1 with Wilson CI` C5 · `Coverage and Conditional Top-1` C3, C5 · `Paired Discordance` C7 · `Species Cluster Bootstrap (2,000 repeats)` C8 | - |
| Additional baseline classifiers (1) | `KmerCentroidClassifier` C9 | - |
| Historical notebooks preserved (1) | `short/benchmark_cnns_kraken2_blast_overlap.ipynb` C6 | - |

## Gate

`MPLCONFIGDIR=/tmp PYTHONPATH=src SMOKE_TEST=1 .venv/bin/python3 -m unittest tests/test_n06_kraken2_blastn_benchmark.py` - 10 passed, 0 failed
