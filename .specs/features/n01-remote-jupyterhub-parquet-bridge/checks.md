# n01-remote-jupyterhub-parquet-bridge checks

Profile: light
Plan: `.specs/features/n01-remote-jupyterhub-parquet-bridge/plan.md`

19 checks in 4 slices · 4 one-way doors · 0 open, of which 0 block

## Checks

### S1 - Dataset Iterável de Partições Parquet com Equivalência Histórica e Divisão Disjunta · 2 files · 25 KB · ~6k

**C1** - `ParquetBatchIterableDataset` yields `(inputs_tensor, labels_tensor)` tuples with shapes `(batch_size, 512)` and `(batch_size,)`, tokenized with `A=0, C=1, G=2, T=3`, padding/ambiguous=4, and dtype `torch.long`. (AC 1)
Proof: `PYTHONPATH=src .venv/bin/python3 -m unittest discover -s tests -k test_c1_tokenized_batch_shapes_and_types`

**C2** - `ParquetBatchIterableDataset.__len__` returns exact `math.ceil(total_samples / batch_size)` by reading row count from PyArrow parquet footer metadata without loading partitions into RAM. (AC 2)
Proof: `PYTHONPATH=src .venv/bin/python3 -m unittest discover -s tests -k test_c2_fast_metadata_row_count_and_len`

**C3** - When iterated with `shuffle=True` and `legacy_global_rng=True` under the same global seed, `ParquetBatchIterableDataset` yields batches numerically identical to `short/utils/dna_dataset_2.py:ParquetCPUDataset`. (AC 3)
Proof: `PYTHONPATH=src .venv/bin/python3 -m unittest discover -s tests -k test_c3_legacy_global_rng_numerical_equivalence`

**C4** - When `shuffle=False`, `ParquetBatchIterableDataset` preserves the exact input partition order and row sequence deterministically across multiple iteration passes. (AC 4)
Proof: `PYTHONPATH=src .venv/bin/python3 -m unittest discover -s tests -k test_c4_deterministic_order_when_shuffle_false`

**C5** - When wrapped in a `DataLoader` with `batch_size=None` and `num_workers > 1`, `ParquetBatchIterableDataset` disjointly splits file partitions across workers with zero duplicated or omitted rows. (AC 5)
Proof: `PYTHONPATH=src .venv/bin/python3 -m unittest discover -s tests -k test_c5_multiprocessing_disjoint_worker_partitioning`

**C6** - If `backend='cudf'` is requested without `cudf` installed and `fallback_to_cpu=False`, `ParquetBatchIterableDataset` raises `ImportError` with a diagnostic message; if `fallback_to_cpu=True`, it logs a warning and falls back to CPU. (AC 6)
Proof: `PYTHONPATH=src .venv/bin/python3 -m unittest discover -s tests -k test_c6_cudf_import_error_or_fallback`

### S2 - Explicabilidade 1D (GradCAM1D e ScoreCAM1D) Desacoplada e Agnóstica de Arquitetura · 2 files · 22 KB · ~6k

**C7** - `GradCAM1D` and `ScoreCAM1D` compute 1D attribution maps of shape `(seq_len,)` with values bounded in `[0.0, 1.0]`. (AC 7)
Proof: `PYTHONPATH=src .venv/bin/python3 -m unittest discover -s tests -k test_c7_heatmap_shape_and_bounds`

**C8** - `GradCAM1D` produces heatmaps numerically matching (`atol=1e-5`) the historical `short/utils/gradcam.py:GradCAM1D` on `ResNet50_1D`. (AC 8)
Proof: `PYTHONPATH=src .venv/bin/python3 -m unittest discover -s tests -k test_c8_resnet_numerical_equivalence_with_historical`

**C9** - `GradCAM1D` and `ScoreCAM1D` execute successfully on `ConvNeXt_Tiny_1D` models without attribute or dimension errors. (AC 9)
Proof: `PYTHONPATH=src .venv/bin/python3 -m unittest discover -s tests -k test_c9_convnext_compatibility`

**C10** - `visualize_heat_map_examples_by_class` masks out attribution values at padded positions (`is_padding == 0`) and returns a `matplotlib.figure.Figure` in headless mode when `show=False`. (AC 10)
Proof: `PYTHONPATH=src .venv/bin/python3 -m unittest discover -s tests -k test_c10_padding_masking_and_headless_figure`

**C11** - `visualize_heat_map_examples_by_class` raises `ValueError` if an invalid `cam_type` is supplied. (AC 11)
Proof: `PYTHONPATH=src .venv/bin/python3 -m unittest discover -s tests -k test_c11_invalid_cam_type_raises_value_error`

### S3 - Utilitários de Matriz de Confusão e Coleta de Pares Confundidos Integrados à Taxonomia · 2 files · 20 KB · ~5k

**C12** - `compute_and_plot_confusion_insights` computes the row-normalized confusion matrix and returns Top-K off-diagonal confused pairs sorted by confusion rate descending. (AC 12)
Proof: `PYTHONPATH=src .venv/bin/python3 -m unittest discover -s tests -k test_c12_top_k_confusion_pairs_ranking`

**C13** - `compute_and_plot_confusion_insights`, `get_class_label`, and `inspect_csv` resolve 0-based `model_label` to 1-based taxonomic taxon names using `viral_cnn.taxonomy.mapping`. (AC 13)
Proof: `PYTHONPATH=src .venv/bin/python3 -m unittest discover -s tests -k test_c13_taxonomic_name_resolution`

**C14** - `get_confused_class_samples` collects up to `n_examples_per_class` input tensors per target true class and stops iteration early when all target quotas are satisfied. (AC 14)
Proof: `PYTHONPATH=src .venv/bin/python3 -m unittest discover -s tests -k test_c14_get_confused_class_samples_quota`

### S4 - Runtime de Notebooks para JupyterHub Remoto, Manifesto de Execução e Smoke-Test · 2 files · 24 KB · ~6k

**C15** - `setup_notebook_runtime` deterministically seeds Python, NumPy, and PyTorch, detects available device and libraries, and resolves `DATA_ROOT` without personal or absolute user paths. (AC 15)
Proof: `PYTHONPATH=src .venv/bin/python3 -m unittest discover -s tests -k test_c15_setup_runtime_and_reproducible_seeding`

**C16** - `JupyterHubDataTopology` maps all canonical `short/` and `long/` dataset splits, taxonomy CSVs, and model checkpoint relative paths from `configs/models.yaml`. (AC 16)
Proof: `PYTHONPATH=src .venv/bin/python3 -m unittest discover -s tests -k test_c16_jupyterhub_data_topology_resolution`

**C17** - `ensure_smoke_test_artifacts` creates deterministic synthetic Parquet partitions for `short/` (`sequence_padded`, `label`) and `long/` (`species_id`, `genus_id`, `origin_len`, `read_len`, `sequence`) with matching taxonomy mappings. (AC 17)
Proof: `PYTHONPATH=src .venv/bin/python3 -m unittest discover -s tests -k test_c17_synthetic_smoke_test_artifacts`

**C18** - `write_notebook_manifest` serializes a comprehensive execution manifest containing git commit, seed, data root, device, library versions, and input files in JSON and YAML formats. (AC 18)
Proof: `PYTHONPATH=src .venv/bin/python3 -m unittest discover -s tests -k test_c18_write_notebook_manifest_json_yaml`

**C19** - All historical files and directories in `short/` and `long/` remain strictly unmodified (`git status --porcelain short/ long/` empty). (AC 19)
Proof: `PYTHONPATH=src .venv/bin/python3 -m unittest discover -s tests -k test_c19_historical_directories_preservation`

## Coverage

| Set (size) | Member -> proof | Unproven |
| --- | --- | --- |
| Parquet dataset column fallbacks (3) | `sequence_padded` C1 · `padded_Representation` C1 · `sequence` C1 | - |
| Parquet dataset shuffle modes (3) | `shuffle=False` C4 · `shuffle=True` C5 · `legacy_global_rng=True` C3 | - |
| Parquet dataset execution backends (2) | `backend="cpu"` C1 · `backend="cudf"` C6 | - |
| CAM explainer algorithms (2) | `GradCAM1D` C7 · `ScoreCAM1D` C7 | - |
| Supported 1D CNN architectures in CAM (2) | `ResNet50_1D` C8 · `ConvNeXt_Tiny_1D` C9 | - |
| CAM display modes (2) | `show=False` (headless Figure return) C10 · `show=True` C10 | - |
| Confusion analysis helper outputs (3) | `compute_and_plot_confusion_insights` C12 · `get_confused_class_samples` C14 · `inspect_csv` C13 | - |
| Runtime execution modes (2) | Remote JupyterHub cluster mode C15 · Local deterministic smoke-test mode C17 | - |
| Dataset tracks managed by topology (2) | `short/` (fragment-level classification) C16 · `long/` (long sequence contigs/reads) C16 | - |
| Manifest serialization formats (2) | JSON C18 · YAML C18 | - |

- Claims naming a status code, route or response shape: none - local scientific library without external network endpoints
- No other check claims more than the single case its proof exercises

## Swept

- validation: C1, C11
- failure modes: C6
- idempotency: C4, C18
- authorization: n/a - local scientific code with no network boundary or access control
- concurrency: C5
- data lifecycle: C17
- dependency failure: C6
- state transitions: n/a - stateless batch dataset iterators and analytical functions
- observability: C18

## Handoff

- S1-S4 = 91 KB, across `src/viral_cnn/data/`, `src/viral_cnn/interpretability/`, `src/viral_cnn/eval/`, and `src/viral_cnn/notebook_runtime.py`, ~23k tokens, well under the 150k budget - one builder
- Mechanism: one builder

