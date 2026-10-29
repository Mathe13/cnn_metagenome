# r04-real-latent-embeddings-and-taxonomy-audit checks

Profile: light
Plan: `.specs/features/r04-real-latent-embeddings-and-taxonomy-audit/plan.md`

## Checks

### S1 - Vectorized Cosine KNN Top-1/Top-5 accuracy, zero self-neighbor presence, and dimension alignment · 2 files · 25 KB · ~6k

**C1** - `evaluate_cosine_knn` computes and returns both `global_knn_accuracy` and `global_knn_top5_accuracy` with `global_knn_top5_accuracy >= global_knn_accuracy` on stratified disjoint reference/query partitions with zero self-neighbors
Proof: `PYTHONPATH=src .venv/bin/python3 -m unittest tests/test_n04_taxonomy_and_latent_space.py -k test_r04_evaluate_cosine_knn_top1_and_top5`

**C2** - Latent space model registry and notebook 06 configuration define pre-classifier embedding dimension of 2048 for all 5 evaluated species models (ResNet50, ResNet50_F, ConvNeXt_Z, ConvNeXt_T, ConvNeXt_F)
Proof: `PYTHONPATH=src .venv/bin/python3 -m unittest tests/test_n04_taxonomy_and_latent_space.py -k test_r04_models_embedding_dimension_2048`

### S2 - Latent space extraction, Table 5 and Table 6 metrics, and fail-fast compliance in Notebook 06 · 2 files · 35 KB · ~8k

**C3** - `notebooks/reproduction/short/06_ictv_latent_space_and_hierarchy.ipynb` raises `FileNotFoundError` immediately when `is_smoke_test == False` and model checkpoints or test parquet files are missing on disk
Proof: `PYTHONPATH=src .venv/bin/python3 -m unittest tests/test_n04_taxonomy_and_latent_space.py -k test_r04_notebook_06_fail_fast_missing_artifacts`

**C4** - `notebooks/reproduction/short/06_ictv_latent_space_and_hierarchy.ipynb` generates Table 5 with distinct `KNN_Top1` and `KNN_Top5` metrics, saves `table5_latent_clustering_metrics.csv`, and prints the dataframe in cell output
Proof: `PYTHONPATH=src .venv/bin/python3 -m unittest tests/test_n04_taxonomy_and_latent_space.py -k test_r04_notebook_06_table5_distinct_top1_top5`

**C5** - `notebooks/reproduction/short/06_ictv_latent_space_and_hierarchy.ipynb` generates Table 6 reporting hierarchical correlations with valid parent taxon counts, saves `table6_taxonomic_rank_correlations.csv`, and prints the dataframe in cell output
Proof: `PYTHONPATH=src .venv/bin/python3 -m unittest tests/test_n04_taxonomy_and_latent_space.py -k test_r04_notebook_06_table6_correlations`

### S3 - Provenance sanitation and smoke test execution in Notebook 05 and Notebook 06 · 2 files · 30 KB · ~7k

**C6** - `notebooks/reproduction/short/05_ictv_taxonomy_curation.ipynb` isolates mock records to smoke mode, sets `"genome_provenance_complete": False` and `"genome_disjoint_test_classes": None` in `manifest.json` outside smoke mode, and both notebooks execute end-to-end in smoke mode in under 60 seconds
Proof: `PYTHONPATH=src .venv/bin/python3 -m unittest tests/test_n04_taxonomy_and_latent_space.py -k test_r04_notebook_05_provenance_and_smoke_execution`

## Coverage

| Set (size) | Member -> proof | Unproven |
| --- | --- | --- |
| KNN evaluation metrics computed (2) | `top1_majority_vote` C1 · `top5_nearest_hit` C1 | - |
| Model architecture families aligned to dim 2048 (2) | `resnet50_species_family` C2 · `convnext_species_family` C2 | - |
| Execution regimes tested in Notebook 06 (2) | `smoke_test_synthetic` C4, C5 · `cluster_fail_fast_missing_artifacts` C3 | - |
| Notebook 05 curation provenance output modes (2) | `smoke_mock_split` C6 · `production_unsplit_genome_provenance` C6 | - |

- No other check claims more than the single case its proof exercises

## Swept

Where each unwritten requirement landed. All nine, one line each, every time.

- validation: C1, C2 (validates distinct KNN Top-1/Top-5 and 2048-d embedding shapes across all models)
- failure modes: C3 (fails fast with FileNotFoundError when real checkpoints/parquets are missing outside smoke mode)
- idempotency: C1, C4, C5 (deterministic seeded evaluation guarantees identical clustering and correlation metrics)
- authorization: n/a - local and cluster offline scripts with no network credentials
- concurrency: n/a - single-process GPU/CPU evaluation of latent embeddings
- data lifecycle: C4, C5 (persists clean CSV tables in results_reproduction/latent_space without lingering temp files)
- dependency failure: C3 (prevents silent fallback to random arrays when dependencies are unavailable)
- state transitions: C1 (splits 80/20 reference/query partitions without data leakage or self-neighbor presence)
- observability: C4, C5, C6 (prints complete dataframes to cell stdout and records execution metadata in manifest.json)

## Handoff

- S1 = ~6k tokens, S2 = ~8k tokens, S3 = ~7k tokens, total ~21k tokens across 4 files (`src/viral_cnn/eval/latent.py`, `notebooks/reproduction/short/05_ictv_taxonomy_curation.ipynb`, `notebooks/reproduction/short/06_ictv_latent_space_and_hierarchy.ipynb`, `tests/test_n04_taxonomy_and_latent_space.py`), well under the 150k budget - one builder
- **Boundary:** C1-C6 closed at HEAD
- **Settled mid-build:** none
- **Abandoned:** none
