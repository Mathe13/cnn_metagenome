# n05-short-motifs-and-triplets-notebooks checks

Profile: light
Plan: `.specs/features/n05-short-motifs-and-triplets-notebooks/plan.md`

14 checks in 2 slices · 3 one-way doors · 0 open, of which 0 block

## Checks

Grouped by the spec's slices; numbering runs across the whole feature.

### S1 - First-Layer Motifs, Paired Perturbation & Jaccard Co-occurrence Notebook · 3 files · 45 KB · ~18k

**C1** - `07_first_layer_motifs_and_perturbation.ipynb` contains valid JSON, zero imports of `short/utils/*`, and imports exclusively from `viral_cnn.*`
Proof: `MPLCONFIGDIR=/tmp PYTHONPATH=src SMOKE_TEST=1 .venv/bin/python3 -m unittest -k test_c1_notebook_07_imports_and_decoupling tests/test_n05_motifs_and_triplets.py`

**C2** - `07_first_layer_motifs_and_perturbation.ipynb` executes end-to-end under `SMOKE_TEST=1` in under 60 seconds without exceptions
Proof: `MPLCONFIGDIR=/tmp PYTHONPATH=src SMOKE_TEST=1 .venv/bin/python3 -m unittest -k test_c2_notebook_07_execution_smoke_test tests/test_n05_motifs_and_triplets.py`

**C3** - Figure 3 panel (e) links to checkpoint `convnext_noise_species` distinct from panel (b) `resnet_noise_species`
Proof: `MPLCONFIGDIR=/tmp PYTHONPATH=src SMOKE_TEST=1 .venv/bin/python3 -m unittest -k test_c3_notebook_07_panel_e_checkpoint_attribution tests/test_n05_motifs_and_triplets.py`

**C4** - Paired motif perturbation evaluates both uniform random control and GC-matched control (`|ΔGC| = 0`) on same sequences without overlap
Proof: `MPLCONFIGDIR=/tmp PYTHONPATH=src SMOKE_TEST=1 .venv/bin/python3 -m unittest -k test_c4_notebook_07_paired_gc_controls tests/test_n05_motifs_and_triplets.py`

**C5** - Motif perturbation summary reports paired Wilcoxon signed-rank test statistics and Benjamini-Hochberg FDR $q$-values for both control policies
Proof: `MPLCONFIGDIR=/tmp PYTHONPATH=src SMOKE_TEST=1 .venv/bin/python3 -m unittest -k test_c5_notebook_07_wilcoxon_and_fdr tests/test_n05_motifs_and_triplets.py`

**C6** - Symmetric Jaccard co-occurrence matrices exported to Parquet under `results_reproduction/motifs/` and Table 8 / Fig 3 artifacts generated
Proof: `MPLCONFIGDIR=/tmp PYTHONPATH=src SMOKE_TEST=1 .venv/bin/python3 -m unittest -k test_c6_notebook_07_artifacts_and_jaccard_export tests/test_n05_motifs_and_triplets.py`

**C7** - Historical notebook `short/full_loss_function.ipynb` remains strictly unmodified (`git status --porcelain short/full_loss_function.ipynb` empty)
Proof: `MPLCONFIGDIR=/tmp PYTHONPATH=src SMOKE_TEST=1 .venv/bin/python3 -m unittest -k test_c7_historical_full_loss_notebook_preserved tests/test_n05_motifs_and_triplets.py`

### S2 - Exploratory Triplet Composition, Boundary Signals & Top-200 Co-occurrence Notebook · 3 files · 50 KB · ~20k

**C8** - `08_triplet_composition_and_cooccurrence.ipynb` contains valid JSON, zero imports of `short/utils/*`, and imports exclusively from `viral_cnn.*`
Proof: `MPLCONFIGDIR=/tmp PYTHONPATH=src SMOKE_TEST=1 .venv/bin/python3 -m unittest -k test_c8_notebook_08_imports_and_decoupling tests/test_n05_motifs_and_triplets.py`

**C9** - `08_triplet_composition_and_cooccurrence.ipynb` executes end-to-end under `SMOKE_TEST=1` in under 60 seconds without exceptions
Proof: `MPLCONFIGDIR=/tmp PYTHONPATH=src SMOKE_TEST=1 .venv/bin/python3 -m unittest -k test_c9_notebook_08_execution_smoke_test tests/test_n05_motifs_and_triplets.py`

**C10** - Triplet composition and exploratory RSCU evaluated across frame 0, 3-frames, and 6-frames with Benjamini-Hochberg FDR correction and unannotated metagenomic warning
Proof: `MPLCONFIGDIR=/tmp PYTHONPATH=src SMOKE_TEST=1 .venv/bin/python3 -m unittest -k test_c10_notebook_08_rscu_and_fdr tests/test_n05_motifs_and_triplets.py`

**C11** - Translation boundary signals (`ATG` and `TAA`/`TAG`/`TGA`) audited across learned motifs against genomic background frequencies
Proof: `MPLCONFIGDIR=/tmp PYTHONPATH=src SMOKE_TEST=1 .venv/bin/python3 -m unittest -k test_c11_notebook_08_translation_boundary_signals tests/test_n05_motifs_and_triplets.py`

**C12** - Top-200 co-occurrence pairs consumed from notebook 07 Jaccard artifact, tested for orientation symmetry $(A, B)$ vs $(B, A)$ and evaluated against permutation null model
Proof: `MPLCONFIGDIR=/tmp PYTHONPATH=src SMOKE_TEST=1 .venv/bin/python3 -m unittest -k test_c12_notebook_08_cooccurrence_symmetry_and_null_model tests/test_n05_motifs_and_triplets.py`

**C13** - Inter-architecture co-occurrence correlations report Pearson $r$ and Spearman $\rho$ with two-tailed $p$-values and FDR correction
Proof: `MPLCONFIGDIR=/tmp PYTHONPATH=src SMOKE_TEST=1 .venv/bin/python3 -m unittest -k test_c13_notebook_08_inter_architecture_correlations tests/test_n05_motifs_and_triplets.py`

**C14** - Historical notebook `short/codon_usage.ipynb` remains strictly unmodified (`git status --porcelain short/codon_usage.ipynb` empty)
Proof: `MPLCONFIGDIR=/tmp PYTHONPATH=src SMOKE_TEST=1 .venv/bin/python3 -m unittest -k test_c14_historical_codon_usage_notebook_preserved tests/test_n05_motifs_and_triplets.py`

## Coverage

| Set (size) | Member -> proof | Unproven |
| --- | --- | --- |
| Reproduction notebooks (2) | `07_first_layer_motifs_and_perturbation.ipynb` C1, C2 · `08_triplet_composition_and_cooccurrence.ipynb` C8, C9 | - |
| Control policies audited (2) | `uniform_random` C4 · `gc_matched` C4, C5 | - |
| Figure 3 panel variants (5) | `resnet50_species` (a) C3, C6 · `resnet_noise_species` (b) C3, C6 · `convnext_zero_species` (c) C6 · `convnext_transfer_species` (d) C6 · `convnext_noise_species` (e) C3, C6 | - |
| Reading frame modes evaluated (3) | `frame0` C10 · `forward_3frames` C10 · `six_frames` C10 | - |
| Translation boundary codons audited (2) | `start_codons (ATG)` C11 · `stop_codons (TAA, TAG, TGA)` C11 | - |
| Co-occurrence pair evaluations (3) | `top_200_edges` C12 · `orientation_symmetry` C12 · `permutation_null_model` C12, C13 | - |
| Historical notebooks preserved (2) | `short/full_loss_function.ipynb` C7 · `short/codon_usage.ipynb` C14 | - |

- No check claims more than the single case its proof exercises

## Swept

Where each unwritten requirement landed. All nine, one line each, every time.

- validation: C1, C8 (valid JSON schemas and no legacy imports)
- failure modes: C2, C9 (deterministic synthetic data fallback under `SMOKE_TEST=1` when remote cluster data is absent)
- idempotency: C7, C14 (`git status --porcelain short/` empty)
- authorization: n/a - local scientific computation with no remote auth endpoints
- concurrency: n/a - sequential cell execution and deterministic PRNG
- data lifecycle: C6, C12 (`results_reproduction/motifs/` parquet artifacts and images)
- dependency failure: C2, C9 (smoke test runs entirely offline with synthetic data)
- state transitions: C6, C12 (Jaccard matrix chaining from notebook 07 to notebook 08)
- observability: C6, C12 (`manifest.json` under `results_reproduction/motifs/`)

## Handoff

S1 = 18k; S2 enters Triplet Notebook at 38k total, under the 150k budget - one builder

- **Boundary:** C1-C14 closed at HEAD
- **Settled mid-build:** Reused `viral_cnn.interpretability.motifs` functions directly, generating standalone figure and parquet artifacts for Tables 8–11 and Figures 3–5.
- **Abandoned:** None

