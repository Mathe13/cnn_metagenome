# n02-short-benchmark-and-calibration-notebooks checks

Profile: light
Plan: `.specs/features/n02-short-benchmark-and-calibration-notebooks/plan.md`

18 checks in 4 slices · 3 one-way doors · 0 open, of which 0 block

## Checks

### S1 - Faixas Parametrizadas de F1, Comparação Especialista e Profiling de Inferência · 3 files · 28 KB · ~8k

**C1** - `compute_species_f1_bands` called with default `thresholds=(0.0, 0.50, 0.80, 0.95, 1.0)` partitions classes into 4 bands `band_0_to_50`, `band_50_to_80`, `band_80_to_95`, `band_95_to_100` whose count strictly sums to `num_classes`. (AC 1)
Proof: `PYTHONPATH=src .venv/bin/python3 -m unittest discover -s tests -k test_c1_species_f1_bands_default_thresholds`

**C2** - `compute_species_f1_bands` called with Table 4 `thresholds=(0.0, 0.50, 0.80, 0.90, 1.0)` and sample `confidences` produces `band_0_to_50`, `band_50_to_80`, `band_80_to_90`, `band_90_to_100` with count, percentage, mean confidence per band, and exact class conservation `sum(count) == num_classes`. (AC 2)
Proof: `PYTHONPATH=src .venv/bin/python3 -m unittest discover -s tests -k test_c2_species_f1_bands_table4_thresholds_and_confidence`

**C3** - `compute_species_f1_bands` raises `ValueError` if `thresholds` are not strictly increasing or do not start at 0.0 and end at 1.0. (AC 3)
Proof: `PYTHONPATH=src .venv/bin/python3 -m unittest discover -s tests -k test_c3_species_f1_bands_invalid_thresholds_raise_value_error`

**C4** - `compute_species_specialist_comparison` computes sample count, correct predictions, accuracy, F1-score, and mean confidence for two models, partitioning evaluated species into `convnext_superior`, `resnet_superior`, `ties`, and `both_zero` such that `convnext_superior + resnet_superior + ties + both_zero == total_evaluated_species`. (AC 4)
Proof: `PYTHONPATH=src .venv/bin/python3 -m unittest discover -s tests -k test_c4_species_specialist_comparison_partition`

**C5** - `profile_inference_model` measures parameter count, GFLOPs, latency, and throughput on 1D CNNs, returning a dictionary with keys `total_params`, `trainable_params`, `size_mb`, `gflops`, `avg_latency_ms`, and `throughput_seq_sec`. (AC 5)
Proof: `PYTHONPATH=src .venv/bin/python3 -m unittest discover -s tests -k test_c5_profile_inference_model_metrics`

**C6** - `evaluate_dataloader` evaluates a model over a DataLoader across noise levels `[0.0, 0.02, 0.05, 0.10, 0.15]` with deterministic noise injection (`seed=42`) and validates empirical substitution rates matching `3 * epsilon / 4`. (AC 5)
Proof: `PYTHONPATH=src .venv/bin/python3 -m unittest discover -s tests -k test_c6_evaluate_dataloader_noise_sweep`

### S2 - Analisador de Entropia Hierárquica e Ativação Pré-Classificadora (TaxonomyEntropyAnalyzer) · 2 files · 24 KB · ~7k

**C7** - `TaxonomyEntropyAnalyzer` registers a forward pre-hook on `model.fc` (supporting compiled `_orig_mod` models), captures pre-classifier latent vectors `(B, D)`, and cleanly unhooks via `remove_hook()` and context manager exit. (AC 6)
Proof: `PYTHONPATH=src .venv/bin/python3 -m unittest discover -s tests -k test_c7_taxonomy_entropy_analyzer_hook_registration_and_cleanup`

**C8** - `TaxonomyEntropyAnalyzer` raises `AttributeError` when model lacks `fc` and raises `ValueError` when `species_to_genus_map` contains negative indices. (AC 7)
Proof: `PYTHONPATH=src .venv/bin/python3 -m unittest discover -s tests -k test_c8_taxonomy_entropy_analyzer_validation_errors`

**C9** - `TaxonomyEntropyAnalyzer.analyze_batch` returns `h_species`, `h_genus`, `hierarchical_entropy_gap` (where gap >= -1e-5), and `h_activation` normalized by `ln(D)` in `[0.0, 1.0]`. (AC 8)
Proof: `PYTHONPATH=src .venv/bin/python3 -m unittest discover -s tests -k test_c9_taxonomy_entropy_analyzer_analyze_batch_bounds`

**C10** - `TaxonomyEntropyAnalyzer.analyze_dataloader` iterates through a DataLoader and computes sample-weighted global averages for `h_species`, `h_genus`, `hierarchical_entropy_gap`, and `h_activation`. (AC 9)
Proof: `PYTHONPATH=src .venv/bin/python3 -m unittest discover -s tests -k test_c10_taxonomy_entropy_analyzer_dataloader_aggregation`

### S3 - Notebook 01_benchmark_clean_and_noise.ipynb (Avaliação Limpa, Ruído e Profiling) · 2 files · 26 KB · ~7k

**C11** - `01_benchmark_clean_and_noise.ipynb` contains valid JSON notebook structure, has zero imports of `short/utils/*`, and defines zero model architectures inline. (AC 10)
Proof: `PYTHONPATH=src .venv/bin/python3 -m unittest discover -s tests -k test_c11_notebook_01_imports_and_architecture_decoupling`

**C12** - `01_benchmark_clean_and_noise.ipynb` executes from start to finish under `SMOKE_TEST=1` via `nbclient` without exceptions. (AC 11)
Proof: `PYTHONPATH=src .venv/bin/python3 -m unittest discover -s tests -k test_c12_notebook_01_execution_smoke_test`

**C13** - `01_benchmark_clean_and_noise.ipynb` execution saves structured artifacts `benchmark_clean_noise_summary.csv`, `table4_f1_bands.csv`, `specialist_comparison.csv`, and `manifest.json` under `results_reproduction/n02_benchmark/`. (AC 11)
Proof: `PYTHONPATH=src .venv/bin/python3 -m unittest discover -s tests -k test_c13_notebook_01_artifacts_generated`

### S4 - Notebook 02_analyze_confidence_and_calibration.ipynb e Utilidades · 3 files · 28 KB · ~8k

**C14** - `compare_confused_species_identity` computes pairwise sequence identity and alignment score between representative sequences of confused class pairs without external network calls when `use_ncbi_online=False`. (AC 12)
Proof: `PYTHONPATH=src .venv/bin/python3 -m unittest discover -s tests -k test_c14_compare_confused_species_identity_offline`

**C15** - `02_analyze_confidence_and_calibration.ipynb` contains valid JSON notebook structure, has zero imports of `short/utils/*`, and imports exclusively from `viral_cnn.*`. (AC 13)
Proof: `PYTHONPATH=src .venv/bin/python3 -m unittest discover -s tests -k test_c15_notebook_02_imports_and_decoupling`

**C16** - `02_analyze_confidence_and_calibration.ipynb` executes from start to finish under `SMOKE_TEST=1` without exceptions. (AC 14)
Proof: `PYTHONPATH=src .venv/bin/python3 -m unittest discover -s tests -k test_c16_notebook_02_execution_smoke_test`

**C17** - `02_analyze_confidence_and_calibration.ipynb` execution saves structured artifacts `table7_hierarchical_entropy.csv`, `calibration_audit_noise.csv`, and `manifest.json` under `results_reproduction/n02_calibration/`. (AC 14)
Proof: `PYTHONPATH=src .venv/bin/python3 -m unittest discover -s tests -k test_c17_notebook_02_artifacts_generated`

**C18** - All historical files and directories in `short/` and `long/` remain strictly unmodified (`git status --porcelain short/ long/` empty). (AC 15)
Proof: `PYTHONPATH=src .venv/bin/python3 -m unittest discover -s tests -k test_c18_historical_directories_preservation`

## Coverage

| Set (size) | Member -> proof | Unproven |
| --- | --- | --- |
| F1 performance band threshold configurations (2) | `thresholds=(0.0, 0.50, 0.80, 0.95, 1.0)` C1 · `thresholds=(0.0, 0.50, 0.80, 0.90, 1.0)` C2 | - |
| Species specialist comparison outcome classifications (4) | `convnext_superior` C4 · `resnet_superior` C4 · `ties` C4 · `both_zero` C4 | - |
| Computational profiling metric categories (5) | `total_params` C5 · `trainable_params` C5 · `size_mb` C5 · `gflops` C5 · `throughput_seq_sec` C5 | - |
| Evaluated noise corruption levels epsilon in DataLoader sweep (5) | `epsilon=0.0` C6 · `epsilon=0.02` C6 · `epsilon=0.05` C6 · `epsilon=0.10` C6 · `epsilon=0.15` C6 | - |
| TaxonomyEntropyAnalyzer hook lifecycle stages (2) | hook active during forward pass C7 · hook cleanly removed C7 | - |
| TaxonomyEntropyAnalyzer validation failure conditions (2) | model lacking `fc` layer C8 · `species_to_genus_map` with negative indices C8 | - |
| TaxonomyEntropyAnalyzer output metrics (4) | `h_species` C9 · `h_genus` C9 · `hierarchical_entropy_gap` C9 · `h_activation` C9 | - |
| Notebook execution reproduction artifacts (2) | `results_reproduction/n02_benchmark/` C13 · `results_reproduction/n02_calibration/` C17 | - |
| Confused species identity sequence comparison modes (2) | offline local pairwise alignment C14 · online Entrez fallback mode C14 | - |

- Claims naming a status code, route or response shape: none - local scientific library without external network endpoints
- No other check claims more than the single case its proof exercises

## Swept

- validation: C3, C8
- failure modes: C3, C8
- idempotency: C7, C18
- authorization: n/a - local scientific code with no network boundary or access control
- concurrency: n/a - sequential notebook cells and PyTorch evaluation loops
- data lifecycle: C13, C17
- dependency failure: C14
- state transitions: C7
- observability: C13, C17

## Handoff

- S1-S4 = 106 KB, across `src/viral_cnn/eval/`, `src/viral_cnn/training/`, and `notebooks/reproduction/short/`, ~30k tokens, well under the 150k budget - one builder
- Mechanism: one builder
