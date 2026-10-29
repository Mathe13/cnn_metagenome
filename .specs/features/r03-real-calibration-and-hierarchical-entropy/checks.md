# r03-real-calibration-and-hierarchical-entropy checks

Profile: light
Plan: `.specs/features/r03-real-calibration-and-hierarchical-entropy/plan.md`

## Checks

### S1 - Hierarchical taxonomy mapping, non-degeneracy validation, and entropy bounds · 2 files · 25 KB · ~6k

**C1** - `build_species_to_genus_tensor` loads taxonomy mapping and returns a `torch.LongTensor` of length equal to `num_species` with distinct genus indices in $[0, N_{\text{genera}}-1]$ where unique genus count is $> 1$
Proof: `PYTHONPATH=src .venv/bin/python3 -m unittest tests/test_n02_benchmark_and_calibration.py -k test_r03_build_species_to_genus_tensor_valid`

**C2** - `build_species_to_genus_tensor` raises `ValueError` if given a degenerate mapping where all species collapse into a single genus
Proof: `PYTHONPATH=src .venv/bin/python3 -m unittest tests/test_n02_benchmark_and_calibration.py -k test_r03_build_species_to_genus_tensor_degenerate_error`

**C3** - `TaxonomyEntropyAnalyzer` produces $H_{\text{genus}} > 0$, $H_{\text{species}} \ge H_{\text{genus}}$, and finite $H_{\text{act}}$ with zero NaN entries under FP16 and FP32 with non-degenerate `species_to_genus` mapping
Proof: `PYTHONPATH=src .venv/bin/python3 -m unittest tests/test_n02_benchmark_and_calibration.py -k test_r03_taxonomy_entropy_analyzer_valid_bounds`

### S2 - Streaming DataLoader calibration, temperature scaling, and notebook 02 sanitation · 3 files · 40 KB · ~8k

**C4** - `audit_dataloader_calibration_by_noise` computes Top-1 accuracy, confidence, ECE, MCE, NLL, and Brier score streaming batches across noise rates without allocating full $(N, C)$ matrices
Proof: `PYTHONPATH=src .venv/bin/python3 -m unittest tests/test_n02_benchmark_and_calibration.py -k test_r03_audit_dataloader_calibration_streaming`

**C5** - `audit_dataloader_calibration_by_noise` fits optimal temperature $T^*$ exclusively on `val_loader` when provided and returns calibrated metrics
Proof: `PYTHONPATH=src .venv/bin/python3 -m unittest tests/test_n02_benchmark_and_calibration.py -k test_r03_audit_dataloader_calibration_temperature_scaling`

**C6** - `notebooks/reproduction/short/02_analyze_confidence_and_calibration.ipynb` extracts real sequences for confused pairs in Cell 4, uses non-degenerate `species_to_genus` and evaluates 5 species variants in Cell 6, runs `audit_dataloader_calibration_by_noise` in Cell 7, and prints full dataframes `df_identity`, `df_entropy`, and `df_cal` without `"ACGT" * 128` or `torch.zeros(13683)` outside smoke mode
Proof: `PYTHONPATH=src .venv/bin/python3 -m unittest tests/test_n02_benchmark_and_calibration.py -k test_r03_notebook_02_real_data_and_table_prints`

## Coverage

| Set (size) | Member -> proof | Unproven |
| --- | --- | --- |
| Taxonomy mapping validity regimes (2) | `valid_multiclass` C1 · `degenerate_single_class` C2 | - |
| Precision modes verified in TaxonomyEntropyAnalyzer (2) | `FP16` C3 · `FP32` C3 | - |
| Calibration evaluation loader inputs (2) | `test_loader_only` C4 · `test_with_val_temperature` C5 | - |
| Notebook 02 audited cells sanitized from synthetic strings (3) | `cell_4_confused_pairs` C6 · `cell_6_hierarchical_entropy` C6 · `cell_7_calibration` C6 | - |

- No other check claims more than the single case its proof exercises

## Swept

Where each unwritten requirement landed. All nine, one line each, every time.

- validation: C1, C2 (`build_species_to_genus_tensor` verifies tensor length and rejects degenerate single-genus mappings)
- failure modes: C2 (prevents silent collapse of genus predictions to constant 0)
- idempotency: C1 (repeated invocations on the same parquet produce identical deterministic genus index assignments)
- authorization: n/a - local and cluster offline scripts with no network credentials
- concurrency: n/a - single-process dataloader streaming and evaluation
- data lifecycle: C6 (saves CSV tables in results_reproduction without lingering temporary files)
- dependency failure: C1, C6 (fails fast when taxonomy parquet or test checkpoints are missing outside smoke mode)
- state transitions: C5 (fits optimal temperature scaling on validation set without leaking test predictions)
- observability: C6 (prints full dataframes df_identity, df_entropy, and df_cal directly to notebook cell outputs)

## Handoff

- S1 = ~6k tokens, S2 = ~8k tokens, total ~14k tokens across 4 files (`src/viral_cnn/taxonomy/mapping.py`, `src/viral_cnn/eval/calibration.py`, `notebooks/reproduction/short/02_analyze_confidence_and_calibration.ipynb`, `tests/test_n02_benchmark_and_calibration.py`), well under the 150k budget - one builder
- **Boundary:** C1-C6 closed at HEAD
- **Settled mid-build:** none
- **Abandoned:** none
