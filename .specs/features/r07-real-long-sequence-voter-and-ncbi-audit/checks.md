# r07-real-long-sequence-voter-and-ncbi-audit checks

Profile: light
Plan: `.specs/features/r07-real-long-sequence-voter-and-ncbi-audit/plan.md`

## Checks

### S1 - Multi-candidate topology resolution and strict fail-fast across long notebooks · 3 files · 25 KB · ~6k

**C1** - `JupyterHubDataTopology.resolve_long_artifact_path` searches candidate directories (`root`, `root/long`, `root/cnn_long`, `root/cnn_kraken`, `root.parent/long`, and cwd) and returns the resolved existing `Path`
Proof: `PYTHONPATH=src .venv/bin/python3 -m unittest tests/test_n07_long_sequence_voter.py -k test_r07_resolve_long_artifact_path`

**C2** - `notebooks/reproduction/long/01_ncbi_download_and_prepare_database.ipynb` raises `FileNotFoundError` in Cell 3 when `is_smoke_test == False` and `ncbi_parquet_dir` does not exist on disk, without generating synthetic sequences
Proof: `PYTHONPATH=src .venv/bin/python3 -m unittest tests/test_n07_long_sequence_voter.py -k test_r07_notebook_01_fail_fast_outside_smoke`

**C3** - `notebooks/reproduction/long/02_cnn_long_voter_inference.ipynb` raises `FileNotFoundError` in Cell 3 when `is_smoke_test == False` and real checkpoints or parquets are missing, never using `MockVoterModel` outside smoke mode
Proof: `PYTHONPATH=src .venv/bin/python3 -m unittest tests/test_n07_long_sequence_voter.py -k test_r07_notebook_02_fail_fast_outside_smoke`

### S2 - Dynamic Table 12 derivation, error stratification, and motif analysis in long notebooks · 3 files · 35 KB · ~8k

**C4** - `notebooks/reproduction/long/03_cnn_long_voter_compare.ipynb` derives Table 12 dynamically from evaluated prediction DataFrames and raises `FileNotFoundError` when `is_smoke_test == False` and prediction files are missing, without using `table_12_data` literal or `rng.randint`
Proof: `PYTHONPATH=src .venv/bin/python3 -m unittest tests/test_n07_long_sequence_voter.py -k test_r07_notebook_03_dynamic_table_12_and_fail_fast`

**C5** - `notebooks/reproduction/long/04_long_motifs_and_gc_analysis.ipynb` raises `FileNotFoundError` in Cell 3 when `is_smoke_test == False` and real contigs are missing, without generating random nucleotide strings
Proof: `PYTHONPATH=src .venv/bin/python3 -m unittest tests/test_n07_long_sequence_voter.py -k test_r07_notebook_04_fail_fast_outside_smoke`

**C6** - Notebooks `long/01` to `long/04` execute end-to-end within 60 seconds each under `is_smoke_test == True`, producing Table 12, error stratification, and all required manifests and figures
Proof: `PYTHONPATH=src .venv/bin/python3 -m unittest tests/test_n07_long_sequence_voter.py -k test_r07_smoke_execution_all_long_notebooks`

## Coverage

| Set (size) | Member -> proof | Unproven |
| --- | --- | --- |
| Topology resolution search scopes (2) | `root_and_subdirectories` C1 · `missing_artifact_none` C1 | - |
| Notebook execution regimes (2) | `smoke_test_mode` C6 · `production_cluster_fail_fast` C2 | - |
| Model instantiation policies (2) | `production_real_checkpoints` C3 · `smoke_mock_voter` C6 | - |
| Table 12 calculation methodologies (2) | `dynamic_dataframe_derivation` C4 · `error_stratification_audit` C6 | - |

- No other check claims more than the single case its proof exercises

## Swept

Where each unwritten requirement landed. All nine, one line each, every time.

- validation: C1, C4 (validates resolved artifact paths and dynamically computes Table 12 accuracy metrics)
- failure modes: C2, C3, C4, C5 (fails fast with FileNotFoundError when real long sequence datasets or checkpoints are missing outside smoke mode)
- idempotency: C1, C6 (deterministic candidate scanning and reproducible sliding-window voter inference)
- authorization: n/a - local and cluster offline scripts with no network credentials
- concurrency: n/a - single-process sliding-window voting and notebook execution
- data lifecycle: C4, C6 (persists clean CSV, JSON and PNG outputs in results_reproduction/long_voter without lingering partial files)
- dependency failure: C2, C3, C4, C5 (prevents silent fallback to synthetic sequences or mock models when production data is missing)
- state transitions: C3, C4 (chains sliding-window consensus outputs from Notebook 02 into Table 12 comparison and error stratification in Notebook 03)
- observability: C4, C6 (prints formatted Table 12 dataframes to cell stdout and generates complete execution manifests)

## Handoff

- S1 = ~6k tokens, S2 = ~8k tokens, total ~14k tokens across 5 files (`src/viral_cnn/notebook_runtime.py`, `notebooks/reproduction/long/01_ncbi_download_and_prepare_database.ipynb`, `notebooks/reproduction/long/02_cnn_long_voter_inference.ipynb`, `notebooks/reproduction/long/03_cnn_long_voter_compare.ipynb`, `notebooks/reproduction/long/04_long_motifs_and_gc_analysis.ipynb`, `tests/test_n07_long_sequence_voter.py`), well under the 150k budget - one builder
- **Boundary:** C1-C6 closed at HEAD
- **Settled mid-build:** none
- **Abandoned:** none
