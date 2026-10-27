# n07-long-sequence-voter-notebooks checks

Profile: light
Plan: `.specs/features/n07-long-sequence-voter-notebooks/plan.md`

15 checks in 4 slices · 4 one-way doors · 0 open, of which 0 block

## Checks

Grouped by the spec's slices; numbering runs across the whole feature.

### S1 - NCBI Download, Database Preparation & Overlap Audit Notebook · 3 files · 40 KB · ~15k

**C1** - `01_ncbi_download_and_prepare_database.ipynb` contains valid JSON, zero legacy imports, and imports exclusively from `viral_cnn.*`
Proof: `MPLCONFIGDIR=/tmp PYTHONPATH=src SMOKE_TEST=1 .venv/bin/python3 -m unittest -k test_c1_notebook_01_imports_and_decoupling tests/test_n07_long_sequence_voter.py`

**C2** - `01_ncbi_download_and_prepare_database.ipynb` executes end-to-end under `SMOKE_TEST=1` in under 60 seconds without exceptions, exporting `long_database_preparation_audit.csv`
Proof: `MPLCONFIGDIR=/tmp PYTHONPATH=src SMOKE_TEST=1 .venv/bin/python3 -m unittest -k test_c2_notebook_01_execution_smoke_test tests/test_n07_long_sequence_voter.py`

**C3** - Sequence preparation audit quantifies training reference overlap via `quantify_training_overlap` and records giant vs small virus read disparity (500 vs 50 reads)
Proof: `MPLCONFIGDIR=/tmp PYTHONPATH=src SMOKE_TEST=1 .venv/bin/python3 -m unittest -k test_c3_overlap_quantification_and_sampling_bias tests/test_n07_long_sequence_voter.py`

**C4** - Historical notebooks `long/download_from_ncbi.ipynb` and `long/prepare database.ipynb` remain strictly unmodified
Proof: `MPLCONFIGDIR=/tmp PYTHONPATH=src SMOKE_TEST=1 .venv/bin/python3 -m unittest -k test_c4_historical_download_and_prep_notebooks_preserved tests/test_n07_long_sequence_voter.py`

### S2 - CNN Long Sequence Voter Inference & Noise Robustness Notebook · 3 files · 50 KB · ~20k

**C5** - `02_cnn_long_voter_inference.ipynb` contains valid JSON, zero legacy imports, and imports exclusively from `viral_cnn.*`
Proof: `MPLCONFIGDIR=/tmp PYTHONPATH=src SMOKE_TEST=1 .venv/bin/python3 -m unittest -k test_c5_notebook_02_imports_and_decoupling tests/test_n07_long_sequence_voter.py`

**C6** - `02_cnn_long_voter_inference.ipynb` executes end-to-end under `SMOKE_TEST=1` in under 60 seconds without exceptions, verifying window logit equivalence via `verify_window_logit_equivalence`
Proof: `MPLCONFIGDIR=/tmp PYTHONPATH=src SMOKE_TEST=1 .venv/bin/python3 -m unittest -k test_c6_notebook_02_execution_smoke_test tests/test_n07_long_sequence_voter.py`

**C7** - Sliding-window consensus evaluates pre-softmax mean logits and majority voting across clean and noise levels $\epsilon \in \{0.0, 0.02, 0.05, 0.10, 0.15\}$, comparing `handle_short='skip'` vs `handle_short='pad'`
Proof: `MPLCONFIGDIR=/tmp PYTHONPATH=src SMOKE_TEST=1 .venv/bin/python3 -m unittest -k test_c7_sliding_window_consensus_and_short_handling tests/test_n07_long_sequence_voter.py`

**C8** - Historical notebook `long/cnn-long-voter.ipynb` remains strictly unmodified (`git status --porcelain long/cnn-long-voter.ipynb` empty)
Proof: `MPLCONFIGDIR=/tmp PYTHONPATH=src SMOKE_TEST=1 .venv/bin/python3 -m unittest -k test_c8_historical_voter_notebook_preserved tests/test_n07_long_sequence_voter.py`

### S3 - Long Sequence Comparison, Error Stratification & ConvNeXt Collapse Diagnosis Notebook · 3 files · 55 KB · ~22k

**C9** - `03_cnn_long_voter_compare.ipynb` contains valid JSON, zero legacy imports, and imports exclusively from `viral_cnn.*`
Proof: `MPLCONFIGDIR=/tmp PYTHONPATH=src SMOKE_TEST=1 .venv/bin/python3 -m unittest -k test_c9_notebook_03_imports_and_decoupling tests/test_n07_long_sequence_voter.py`

**C10** - `03_cnn_long_voter_compare.ipynb` executes end-to-end under `SMOKE_TEST=1` in under 60 seconds without exceptions, reproducing Table 12 and exporting `table_12_long_voter_comparison.csv`
Proof: `MPLCONFIGDIR=/tmp PYTHONPATH=src SMOKE_TEST=1 .venv/bin/python3 -m unittest -k test_c10_notebook_03_execution_smoke_test tests/test_n07_long_sequence_voter.py`

**C11** - Error stratification partitions performance across contig length bins, ambiguity rates, and reference presence (`seen_genome` vs `unseen_genome`), diagnosing ConvNeXt collapse
Proof: `MPLCONFIGDIR=/tmp PYTHONPATH=src SMOKE_TEST=1 .venv/bin/python3 -m unittest -k test_c11_error_stratification_and_collapse_diagnosis tests/test_n07_long_sequence_voter.py`

**C12** - Historical notebook `long/cnn-long-voter-compare.ipynb` remains strictly unmodified (`git status --porcelain long/cnn-long-voter-compare.ipynb` empty)
Proof: `MPLCONFIGDIR=/tmp PYTHONPATH=src SMOKE_TEST=1 .venv/bin/python3 -m unittest -k test_c12_historical_compare_notebook_preserved tests/test_n07_long_sequence_voter.py`

### S4 - Long Contig Motifs and GC Profile Analysis Notebook · 3 files · 40 KB · ~15k

**C13** - `04_long_motifs_and_gc_analysis.ipynb` contains valid JSON, zero legacy imports, and imports exclusively from `viral_cnn.*`
Proof: `MPLCONFIGDIR=/tmp PYTHONPATH=src SMOKE_TEST=1 .venv/bin/python3 -m unittest -k test_c13_notebook_04_imports_and_decoupling tests/test_n07_long_sequence_voter.py`

**C14** - `04_long_motifs_and_gc_analysis.ipynb` executes end-to-end under `SMOKE_TEST=1` in under 60 seconds without exceptions, generating `long_contig_motif_summary.csv` and `figure_long_motifs_gc.png`
Proof: `MPLCONFIGDIR=/tmp PYTHONPATH=src SMOKE_TEST=1 .venv/bin/python3 -m unittest -k test_c14_notebook_04_execution_smoke_test tests/test_n07_long_sequence_voter.py`

**C15** - Historical notebook `long/long_loss_function.ipynb` remains strictly unmodified (`git status --porcelain long/long_loss_function.ipynb` empty)
Proof: `MPLCONFIGDIR=/tmp PYTHONPATH=src SMOKE_TEST=1 .venv/bin/python3 -m unittest -k test_c15_historical_long_loss_notebook_preserved tests/test_n07_long_sequence_voter.py`

## Coverage

| Set (size) | Member -> proof | Unproven |
| --- | --- | --- |
| Reproduction notebooks in long (4) | `01_ncbi_download_and_prepare_database.ipynb` C1, C2 · `02_cnn_long_voter_inference.ipynb` C5, C6 · `03_cnn_long_voter_compare.ipynb` C9, C10 · `04_long_motifs_and_gc_analysis.ipynb` C13, C14 | - |
| Voter consensus aggregation modes (2) | `mean_logits` C7, C10 · `majority_vote` C7, C10 | - |
| Short contig handling policies (2) | `handle_short='skip'` C7 · `handle_short='pad'` C7 | - |
| Error stratification dimensions (3) | `length_bins` C11 · `ambiguity_bins` C11 · `reference_genome_presence (seen vs unseen)` C11 | - |
| Historical notebooks in long preserved (5) | `long/download_from_ncbi.ipynb` C4 · `long/prepare database.ipynb` C4 · `long/cnn-long-voter.ipynb` C8 · `long/cnn-long-voter-compare.ipynb` C12 · `long/long_loss_function.ipynb` C15 | - |

- No check claims more than the single case its proof exercises

## Swept

Where each unwritten requirement landed. All nine, one line each, every time.

- validation: C1, C5, C9, C13 (valid notebook JSON and zero legacy imports)
- failure modes: C2, C6, C10, C14 (deterministic synthetic contig data fallback under `SMOKE_TEST=1` when remote cluster NCBI files are absent)
- idempotency: C4, C8, C12, C15 (`git status --porcelain long/` empty)
- authorization: n/a - local scientific computation with no remote auth endpoints
- concurrency: n/a - sequential cell execution and deterministic PRNG
- data lifecycle: C2, C6, C10, C14 (`results_reproduction/long_voter/` CSVs, Parquets, PNGs, and JSON manifests)
- dependency failure: C2, C6, C10, C14 (smoke test runs entirely offline with synthetic data)
- state transitions: C6, C10 (sequence slicing to window inference to consensus vote to error stratification)
- observability: C2, C10 (`manifest_long_voter.json` under `results_reproduction/long_voter/`)

## Handoff

S1 = 15k; S2 = 20k; S3 = 22k; S4 enters Motifs at 57k total, under the 150k budget - one builder

- **Boundary:** C1-C15 closed at HEAD
- **Settled mid-build:** Reused `viral_cnn.inference.*`, `viral_cnn.baselines.long_eval`, and `viral_cnn.interpretability.motifs` modules directly, generating standalone figure and CSV artifacts.
- **Abandoned:** None

