# r05-real-motif-jaccard-and-triplet-analysis checks

Profile: light
Plan: `.specs/features/r05-real-motif-jaccard-and-triplet-analysis/plan.md`

## Checks

### S1 - Empirical motif presence extraction, Jaccard co-occurrence, and 5-variant perturbation in Notebook 07 · 2 files · 25 KB · ~6k

**C1** - `compute_motif_presence_matrix_from_loader` scans DataLoader batches, decodes nucleotide tokens, and returns a binary matrix of shape $(N, M)$ matching exact 7-mer occurrences without fabricated constant lines
Proof: `PYTHONPATH=src .venv/bin/python3 -m unittest tests/test_n05_motifs_and_triplets.py -k test_r05_compute_motif_presence_matrix_loader`

**C2** - `compute_jaccard_cooccurrence` computes symmetric pairwise Jaccard index from empirical presence matrices with unit diagonal and non-degenerate off-diagonal values
Proof: `PYTHONPATH=src .venv/bin/python3 -m unittest tests/test_n05_motifs_and_triplets.py -k test_r05_jaccard_cooccurrence_symmetry_and_values`

**C3** - `notebooks/reproduction/short/07_first_layer_motifs_and_perturbation.ipynb` evaluates motif perturbation across all 5 species model variants, saves `table_08_motif_perturbation.csv` and `motif_summary.parquet`, and exports empirical Jaccard matrices from Cell 9
Proof: `PYTHONPATH=src .venv/bin/python3 -m unittest tests/test_n05_motifs_and_triplets.py -k test_r05_notebook_07_five_variants_and_jaccard_matrices`

### S2 - Translation boundary audit, inter-architecture correlations, and fail-fast in Notebook 08 · 2 files · 35 KB · ~8k

**C4** - `notebooks/reproduction/short/08_triplet_composition_and_cooccurrence.ipynb` loads learned motifs from `results_reproduction/motifs/motif_summary.parquet` in Cell 5 and raises `FileNotFoundError` immediately when `is_smoke_test == False` and the artifact is missing on disk
Proof: `PYTHONPATH=src .venv/bin/python3 -m unittest tests/test_n05_motifs_and_triplets.py -k test_r05_notebook_08_fail_fast_missing_motif_summary`

**C5** - `notebooks/reproduction/short/08_triplet_composition_and_cooccurrence.ipynb` generates Table 10 auditing translation boundary signals (`start_atg`, `stop_taa`, `stop_tag`, `stop_tga`) on learned motifs and saves `table_10_translation_boundary_signals.csv`
Proof: `PYTHONPATH=src .venv/bin/python3 -m unittest tests/test_n05_motifs_and_triplets.py -k test_r05_notebook_08_table_10_boundary_signals`

**C6** - `notebooks/reproduction/short/08_triplet_composition_and_cooccurrence.ipynb` evaluates Table 11 symmetry and computes inter-architecture co-occurrence correlations without degenerate $r = 1.0$ across distinct networks, and both notebooks complete in under 60 seconds in smoke mode
Proof: `PYTHONPATH=src .venv/bin/python3 -m unittest tests/test_n05_motifs_and_triplets.py -k test_r05_notebook_08_table_11_and_inter_architecture_correlations`

## Coverage

| Set (size) | Member -> proof | Unproven |
| --- | --- | --- |
| Motif presence extraction data formats (2) | `tensor_dataset_tokens` C1 · `parquet_stream_tokens` C1 | - |
| Model variants evaluated for motif perturbation (2) | `resnet_variants` C3 · `convnext_variants` C3 | - |
| Translation boundary signal categories (2) | `start_codon_atg` C5 · `stop_codons_taa_tag_tga` C5 | - |
| Execution regimes tested in Notebook 08 (2) | `smoke_test_execution` C6 · `cluster_fail_fast_missing_summary` C4 | - |

- No other check claims more than the single case its proof exercises

## Swept

Where each unwritten requirement landed. All nine, one line each, every time.

- validation: C1, C2 (validates binary presence shape $(N, M)$ and symmetric Jaccard values in $[0.0, 1.0]$)
- failure modes: C4 (fails fast with FileNotFoundError when motif_summary.parquet is missing outside smoke mode)
- idempotency: C1, C5, C6 (deterministic token decoding and fixed permutation seed produce reproducible statistics)
- authorization: n/a - local and cluster offline scripts with no network credentials
- concurrency: n/a - single-process dataloader scanning and matrix multiplication
- data lifecycle: C3, C5, C6 (persists clean CSV and Parquet tables in results_reproduction/motifs without lingering temp files)
- dependency failure: C4 (prevents silent fallback to static hardcoded lists when dependencies are unavailable)
- state transitions: C3, C4 (chains learned motif filters from Notebook 07 into translation boundary audit in Notebook 08)
- observability: C3, C5, C6 (prints complete dataframes to cell stdout and records execution metadata in manifest.json)

## Handoff

- S1 = ~6k tokens, S2 = ~8k tokens, total ~14k tokens across 4 files (`src/viral_cnn/interpretability/motifs.py`, `notebooks/reproduction/short/07_first_layer_motifs_and_perturbation.ipynb`, `notebooks/reproduction/short/08_triplet_composition_and_cooccurrence.ipynb`, `tests/test_n05_motifs_and_triplets.py`), well under the 150k budget - one builder
- **Boundary:** C1-C6 closed at HEAD
- **Settled mid-build:** none
- **Abandoned:** none

