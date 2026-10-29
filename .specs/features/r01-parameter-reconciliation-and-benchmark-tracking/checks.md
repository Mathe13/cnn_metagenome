# r01-parameter-reconciliation-and-benchmark-tracking checks

Profile: light
Plan: `.specs/features/r01-parameter-reconciliation-and-benchmark-tracking/plan.md`

## Checks

### S1 - Parameter decomposition in profiler and selective git tracking · 2 files · 15 KB · ~4k

**C1** - `profile_inference_model` decomposes parameters for `ResNet50_1D` returning `backbone_params == 15955584`, `head_params == 28036467` and `total_params == 43992051` for Species ($13.683$ classes), and `backbone_params == 15955584`, `head_params == 5536398` and `total_params == 21491982` for Genus ($2.702$ classes)
Proof: `python3 -m unittest tests/test_n02_benchmark_and_calibration.py -k test_r01_resnet_parameter_decomposition`

**C2** - `profile_inference_model` decomposes parameters for `ConvNeXt_Tiny_1D` returning `backbone_params == 135977344`, `head_params == 28036467` and `total_params == 164013811` for Species ($13.683$ classes), and `backbone_params == 135977344`, `head_params == 5536398` and `total_params == 141513742` for Genus ($2.702$ classes)
Proof: `python3 -m unittest tests/test_n02_benchmark_and_calibration.py -k test_r01_convnext_parameter_decomposition`

**C3** - `git check-ignore` permits tracking of `.csv`, `.json`, `.yaml`, and `.png` files under `results_reproduction/` while ignoring `.pt`, `.parquet`, and `.sqlite` files
Proof: `python3 -m unittest tests/test_n02_benchmark_and_calibration.py -k test_r01_gitignore_selective_tracking`

### S2 - Fail-fast guards, table outputs, and manifest provenance in Notebook 01 · 2 files · 25 KB · ~6k

**C4** - `notebooks/reproduction/short/01_benchmark_clean_and_noise.ipynb` raises `FileNotFoundError` when `ctx.is_smoke_test == False` and any required checkpoint file does not exist on disk
Proof: `python3 -m unittest tests/test_n02_benchmark_and_calibration.py -k test_r01_notebook_01_fail_fast_missing_checkpoint`

**C5** - `notebooks/reproduction/short/01_benchmark_clean_and_noise.ipynb` raises `RuntimeError` when `ctx.is_smoke_test == False` and sample counts diverge from $1.706.270$ (Species) or $816.756$ (Genus)
Proof: `python3 -m unittest tests/test_n02_benchmark_and_calibration.py -k test_r01_notebook_01_fail_fast_sample_count`

**C6** - `notebooks/reproduction/short/01_benchmark_clean_and_noise.ipynb` prints `df_prof`, `df_summary`, `df_f1_bands`, and specialist comparisons for $\epsilon \in \{0.05, 0.10\}$ into cell stdout, saves `specialist_comparison.csv` and `specialist_comparison_eps010.csv`, and records all 7 evaluated model checkpoint paths in `manifest.json`
Proof: `python3 -m unittest tests/test_n02_benchmark_and_calibration.py -k test_r01_notebook_01_prints_tables_and_manifest`

## Coverage

| Set (size) | Member -> proof | Unproven |
| --- | --- | --- |
| Model architectures profiled for parameter decomposition (4) | `ResNet50 Species` C1 · `ResNet50 Genus` C1 · `ConvNeXt Species` C2 · `ConvNeXt Genus` C2 | - |
| Selective .gitignore filetype handling under results_reproduction/ (6) | `csv` C3 · `json` C3 · `yaml` C3 · `png` C3 · `pt` C3 · `parquet` C3 | - |
| Notebook 01 fail-fast and execution invariants (3) | `missing checkpoint FileNotFoundError` C4 · `sample count RuntimeError` C5 · `output printing and manifest logging` C6 | - |

- No other check claims more than the single case its proof exercises

## Swept

Where each unwritten requirement landed. All nine, one line each, every time.

- validation: C1, C2 (`profile_inference_model` validates and returns exact integer counts for backbone, head, and total parameters)
- failure modes: C4, C5 (guards missing checkpoint files with `FileNotFoundError` and divergent sample counts with `RuntimeError` when `ctx.is_smoke_test == False`)
- idempotency: C3, C6 (repeated execution of `git check-ignore` and notebook 01 generates deterministic outputs and file matching)
- authorization: n/a - offline scientific metagenomic classification codebase with no authentication credentials or network calls
- concurrency: n/a - sequential notebook execution and single-process in-memory parameter profiling
- data lifecycle: C3 (preserves small reproducibility records like `.csv` and `.json` in Git while excluding multi-gigabyte checkpoints and Parquet partitions)
- dependency failure: C4 (explicit fail-fast guards against missing cluster files)
- state transitions: n/a - pure profiling calculations and immutable evaluation steps
- observability: C6 (`profile_inference_model` and notebook cells print tabular summaries and serialize execution manifests with checkpoint metadata)

## Handoff

- S1 = ~4k tokens, S2 = ~6k tokens, total ~10k tokens across 4 files (`src/viral_cnn/training/profiler.py`, `.gitignore`, `notebooks/reproduction/short/01_benchmark_clean_and_noise.ipynb`, `tests/test_n02_benchmark_and_calibration.py`), well under the 150k budget - one builder
- **Boundary:** C1-C6 closed at `c1c15da`
- **Settled mid-build:** none
- **Abandoned:** none
