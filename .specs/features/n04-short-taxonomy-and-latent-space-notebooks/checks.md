# n04-short-taxonomy-and-latent-space-notebooks checks

Profile: light
Plan: `.specs/features/n04-short-taxonomy-and-latent-space-notebooks/plan.md`

16 checks in 3 slices · 3 one-way doors · 0 open, of which 0 block

## Checks

Grouped by the spec's slices; numbering runs across the whole feature.

### S1 - Modular Taxonomy Curation and Provenance in viral_cnn.taxonomy.curation · 3 files · 40 KB · ~15k

**C1** - `build_ictv_rename_map` extracts lowercase old-to-new species mapping from ICTV rename DataFrame for all rows containing `Renamed`
Proof: `MPLCONFIGDIR=/tmp PYTHONPATH=src SMOKE_TEST=1 .venv/bin/python3 -m unittest -k test_c1_build_ictv_rename_map tests/test_n04_taxonomy_and_latent_space.py`

**C2** - `resolve_local_ictv_taxonomy` matches species names against ICTV master list and populates all 8 lineage ranks with `ICTV Master` attribution
Proof: `MPLCONFIGDIR=/tmp PYTHONPATH=src SMOKE_TEST=1 .venv/bin/python3 -m unittest -k test_c2_resolve_local_ictv_taxonomy tests/test_n04_taxonomy_and_latent_space.py`

**C3** - `sanitize_hybrid_taxonomy` ensures `model_label = id - 1` offset alignment and backfills missing lineage ranks with placeholder strings
Proof: `MPLCONFIGDIR=/tmp PYTHONPATH=src SMOKE_TEST=1 .venv/bin/python3 -m unittest -k test_c3_sanitize_hybrid_taxonomy tests/test_n04_taxonomy_and_latent_space.py`

**C4** - `generate_synthetic_taxonomy` produces a deterministic mock taxonomy DataFrame containing all 8 ranks and `model_label` without network or external files
Proof: `MPLCONFIGDIR=/tmp PYTHONPATH=src SMOKE_TEST=1 .venv/bin/python3 -m unittest -k test_c4_generate_synthetic_taxonomy tests/test_n04_taxonomy_and_latent_space.py`

**C5** - `JupyterHubDataTopology` includes canonical path entries for `ictv_master` and `hybrid_taxonomy_parquet`
Proof: `MPLCONFIGDIR=/tmp PYTHONPATH=src SMOKE_TEST=1 .venv/bin/python3 -m unittest -k test_c5_topology_ictv_paths tests/test_n04_taxonomy_and_latent_space.py`

### S2 - ICTV Taxonomy Curation and Provenance Reproduction Notebook · 3 files · 55 KB · ~20k

**C6** - `05_ictv_taxonomy_curation.ipynb` executes end-to-end under `SMOKE_TEST=1` in under 60 seconds without exceptions
Proof: `MPLCONFIGDIR=/tmp PYTHONPATH=src SMOKE_TEST=1 .venv/bin/python3 -m unittest -k test_c6_notebook_05_execution_smoke_test tests/test_n04_taxonomy_and_latent_space.py`

**C7** - `05_ictv_taxonomy_curation.ipynb` contains valid JSON, zero imports of `short/utils/*`, and imports exclusively from `viral_cnn.*`
Proof: `MPLCONFIGDIR=/tmp PYTHONPATH=src SMOKE_TEST=1 .venv/bin/python3 -m unittest -k test_c7_notebook_05_imports_and_decoupling tests/test_n04_taxonomy_and_latent_space.py`

**C8** - `05_ictv_taxonomy_curation.ipynb` integrates dataset duplicate audit (`viral_cnn.data.audit`) and genome-disjoint split feasibility (`viral_cnn.data.grouped_split`), saving parquet and manifest to `results_reproduction/taxonomy/`
Proof: `MPLCONFIGDIR=/tmp PYTHONPATH=src SMOKE_TEST=1 .venv/bin/python3 -m unittest -k test_c8_notebook_05_artifacts_generated tests/test_n04_taxonomy_and_latent_space.py`

**C9** - Historical notebooks `short/ICTV_improve_kraken_database.ipynb` and `short/improve_kraken_database-Copy1.ipynb` remain strictly unmodified
Proof: `MPLCONFIGDIR=/tmp PYTHONPATH=src SMOKE_TEST=1 .venv/bin/python3 -m unittest -k test_c9_historical_curation_notebooks_preserved tests/test_n04_taxonomy_and_latent_space.py`

### S3 - ICTV Latent Space and Taxonomic Hierarchy Reproduction Notebook · 4 files · 70 KB · ~25k

**C10** - `06_ictv_latent_space_and_hierarchy.ipynb` contains valid JSON, zero imports of `short/utils/*`, and imports exclusively from `viral_cnn.*`
Proof: `MPLCONFIGDIR=/tmp PYTHONPATH=src SMOKE_TEST=1 .venv/bin/python3 -m unittest -k test_c10_notebook_06_imports_and_decoupling tests/test_n04_taxonomy_and_latent_space.py`

**C11** - `06_ictv_latent_space_and_hierarchy.ipynb` executes end-to-end under `SMOKE_TEST=1` in under 60 seconds without exceptions
Proof: `MPLCONFIGDIR=/tmp PYTHONPATH=src SMOKE_TEST=1 .venv/bin/python3 -m unittest -k test_c11_notebook_06_execution_smoke_test tests/test_n04_taxonomy_and_latent_space.py`

**C12** - `06_ictv_latent_space_and_hierarchy.ipynb` evaluates 5-NN cosine accuracy and high-dimensional Silhouette with valid class inclusion reporting across Species, Genus, and Order
Proof: `MPLCONFIGDIR=/tmp PYTHONPATH=src SMOKE_TEST=1 .venv/bin/python3 -m unittest -k test_c12_notebook_06_table5_metrics tests/test_n04_taxonomy_and_latent_space.py`

**C13** - `06_ictv_latent_space_and_hierarchy.ipynb` computes hierarchical correlations between Species, Genus, and Order, reporting Pearson $r$, Spearman $\rho$, and sample size $N$ without truncating valid genera
Proof: `MPLCONFIGDIR=/tmp PYTHONPATH=src SMOKE_TEST=1 .venv/bin/python3 -m unittest -k test_c13_notebook_06_table6_correlations tests/test_n04_taxonomy_and_latent_space.py`

**C14** - `06_ictv_latent_space_and_hierarchy.ipynb` evaluates independent linear probe trained on train/val representations to test head bottleneck hypothesis without test leakage
Proof: `MPLCONFIGDIR=/tmp PYTHONPATH=src SMOKE_TEST=1 .venv/bin/python3 -m unittest -k test_c14_notebook_06_linear_probe_audit tests/test_n04_taxonomy_and_latent_space.py`

**C15** - `06_ictv_latent_space_and_hierarchy.ipynb` exports qualitative UMAP 2D projections with explicit qualitative disclaimers and saves metrics/manifest to `results_reproduction/latent_space/`
Proof: `MPLCONFIGDIR=/tmp PYTHONPATH=src SMOKE_TEST=1 .venv/bin/python3 -m unittest -k test_c15_notebook_06_umap_and_artifacts tests/test_n04_taxonomy_and_latent_space.py`

**C16** - Historical notebook `short/ICTV_latent_space-bkp.ipynb` remains strictly unmodified (`git status --porcelain short/` empty)
Proof: `MPLCONFIGDIR=/tmp PYTHONPATH=src SMOKE_TEST=1 .venv/bin/python3 -m unittest -k test_c16_historical_latent_notebook_preserved tests/test_n04_taxonomy_and_latent_space.py`

## Coverage

| Set (size) | Member -> proof | Unproven |
| --- | --- | --- |
| ICTV Taxonomy Curation functions (5) | `build_ictv_rename_map` C1 · `resolve_local_ictv_taxonomy` C2 · `sanitize_hybrid_taxonomy` C3 · `generate_synthetic_taxonomy` C4 · `JupyterHubDataTopology` C5 | - |
| Taxonomy hierarchy ranks evaluated (8) | `Realm` C2, C4 · `Kingdom` C2, C4 · `Phylum` C2, C4 · `Class` C2, C4 · `Order` C2, C4, C12, C13 · `Family` C2, C4 · `Genus` C2, C4, C12, C13 · `Species` C2, C4, C12, C13 | - |
| Reproduction notebooks (2) | `05_ictv_taxonomy_curation.ipynb` C6, C7, C8 · `06_ictv_latent_space_and_hierarchy.ipynb` C10, C11, C12, C13, C14, C15 | - |
| Latent space evaluation metrics (5) | `Cosine KNN (k=5)` C12 · `Silhouette Score (high-dim)` C12 · `Hierarchical Correlation (Pearson/Spearman/N)` C13 · `Linear Probe Bottleneck Audit` C14 · `UMAP 2D Qualitative Projection` C15 | - |
| Historical notebooks preserved (3) | `short/ICTV_improve_kraken_database.ipynb` C9 · `short/improve_kraken_database-Copy1.ipynb` C9 · `short/ICTV_latent_space-bkp.ipynb` C16 | - |

- No check claims more than the single case its proof exercises

## Swept

Where each unwritten requirement landed. All nine, one line each, every time.

- validation: C3, C4 (taxonomy label alignment `model_label = id - 1` and column schemas)
- failure modes: C4 (offline deterministic synthetic fallback when remote Excel or NCBI API is absent)
- idempotency: C9, C16 (`git status --porcelain short/` empty)
- authorization: n/a - local scientific computation with no remote auth endpoints
- concurrency: n/a - sequential cell execution and deterministic PRNG
- data lifecycle: C8 (`results_reproduction/taxonomy/`), C15 (`results_reproduction/latent_space/`)
- dependency failure: C4, C6, C11 (smoke test runs entirely offline with synthetic data)
- state transitions: C3, C14 (taxonomy cleansing, embedding extraction, probe training and evaluation)
- observability: C8 (`manifest.json` under taxonomy), C15 (`manifest.json` under latent_space)

## Handoff

S1 = 15k; S2 = 20k; S3 enters Latent Notebook at 45k total, under the 150k budget - one builder

- **Boundary:** C1-C16 closed at HEAD
- **Settled mid-build:** Dynamically resolved `sequence_padded` column for test parquet duplicate audit and accommodated UMAP `#` disclaimer headers in pandas CSV readers.
- **Abandoned:** None
