# n01-remote-jupyterhub-parquet-bridge verification

**Verdict**: PASS
**Profile**: light
**Diff range**: a66dd4075e5f12ad71f3080eb619d01c0a217730..36e27441fd9f71b0daa6ba803c4a2e636c3cbd9a
**Round**: 1 - full
**Verifier**: independent sub-agent (author != verifier)

## Checks

| Check | Claim | Proof run | Evidence | Result |
| --- | --- | --- | --- | --- |
| C1 | `ParquetBatchIterableDataset` yields `(inputs, labels)` with shapes `(B, 512)` and `(B,)`, tokens `0..4`, dtype `torch.long` | `MPLCONFIGDIR=/tmp/mpl PYTHONPATH=src .venv/bin/python3 -m unittest -v tests.test_n01_parquet_bridge.TestParquetBatchIterableDataset.test_c1_tokenized_batch_shapes_and_types` exit 0 | `tests/test_n01_parquet_bridge.py:103` - `self.assertEqual(inputs.shape[1], 512)` | PASS |
| C2 | `ParquetBatchIterableDataset.__len__` returns exact `math.ceil(total_samples / batch_size)` via fast footer metadata | `MPLCONFIGDIR=/tmp/mpl PYTHONPATH=src .venv/bin/python3 -m unittest -v tests.test_n01_parquet_bridge.TestParquetBatchIterableDataset.test_c2_fast_metadata_row_count_and_len` exit 0 | `tests/test_n01_parquet_bridge.py:126` - `self.assertEqual(len(ds), math.ceil(expected_total / 7))` | PASS |
| C3 | When `shuffle=True` and `legacy_global_rng=True`, yields batches numerically identical to `ParquetCPUDataset` | `MPLCONFIGDIR=/tmp/mpl PYTHONPATH=src .venv/bin/python3 -m unittest -v tests.test_n01_parquet_bridge.TestParquetBatchIterableDataset.test_c3_legacy_global_rng_numerical_equivalence` exit 0 | `tests/test_n01_parquet_bridge.py:157` - `self.assertTrue(torch.equal(h_in, n_in))` | PASS |
| C4 | When `shuffle=False`, preserves exact input partition order and sequence deterministically across repeated passes | `MPLCONFIGDIR=/tmp/mpl PYTHONPATH=src .venv/bin/python3 -m unittest -v tests.test_n01_parquet_bridge.TestParquetBatchIterableDataset.test_c4_deterministic_order_when_shuffle_false` exit 0 | `tests/test_n01_parquet_bridge.py:167` - `self.assertTrue(torch.equal(in1, in2))` | PASS |
| C5 | DataLoader with `num_workers > 1` disjointly splits file partitions across workers with zero loss or duplication | `MPLCONFIGDIR=/tmp/mpl PYTHONPATH=src .venv/bin/python3 -m unittest -v tests.test_n01_parquet_bridge.TestParquetBatchIterableDataset.test_c5_multiprocessing_disjoint_worker_partitioning` exit 0 | `tests/test_n01_parquet_bridge.py:179` - `self.assertEqual(len(all_labels), expected_count)` | PASS |
| C6 | `backend='cudf'` raises `ImportError` if missing without fallback, or falls back to CPU when requested | `MPLCONFIGDIR=/tmp/mpl PYTHONPATH=src .venv/bin/python3 -m unittest -v tests.test_n01_parquet_bridge.TestParquetBatchIterableDataset.test_c6_cudf_import_error_or_fallback` exit 0 | `tests/test_n01_parquet_bridge.py:189` - `with self.assertRaises(ImportError): list(ds_fail)` | PASS |
| C7 | `GradCAM1D` and `ScoreCAM1D` compute 1D attribution maps of shape `(seq_len,)` bounded in `[0.0, 1.0]` | `MPLCONFIGDIR=/tmp/mpl PYTHONPATH=src .venv/bin/python3 -m unittest -v tests.test_n01_parquet_bridge.TestInterpretabilityGradCAM.test_c7_heatmap_shape_and_bounds` exit 0 | `tests/test_n01_parquet_bridge.py:212` - `self.assertEqual(cam.shape, (512,))` | PASS |
| C8 | `GradCAM1D` produces output numerically matching (`atol=1e-5`) historical `short/utils/gradcam.py` on `ResNet50_1D` | `MPLCONFIGDIR=/tmp/mpl PYTHONPATH=src .venv/bin/python3 -m unittest -v tests.test_n01_parquet_bridge.TestInterpretabilityGradCAM.test_c8_resnet_numerical_equivalence_with_historical` exit 0 | `tests/test_n01_parquet_bridge.py:226` - `self.assertTrue(np.allclose(new_cam, hist_cam, atol=1e-5))` | PASS |
| C9 | `GradCAM1D` and `ScoreCAM1D` execute successfully on `ConvNeXt_Tiny_1D` without attribute or dimension errors | `MPLCONFIGDIR=/tmp/mpl PYTHONPATH=src .venv/bin/python3 -m unittest -v tests.test_n01_parquet_bridge.TestInterpretabilityGradCAM.test_c9_convnext_compatibility` exit 0 | `tests/test_n01_parquet_bridge.py:232` - `self.assertEqual(cam.shape, (512,))` | PASS |
| C10 | `visualize_heat_map_examples_by_class` masks padding and returns `matplotlib.figure.Figure` in headless mode | `MPLCONFIGDIR=/tmp/mpl PYTHONPATH=src .venv/bin/python3 -m unittest -v tests.test_n01_parquet_bridge.TestInterpretabilityGradCAM.test_c10_padding_masking_and_headless_figure` exit 0 | `tests/test_n01_parquet_bridge.py:255` - `self.assertIsInstance(fig, plt.Figure)` | PASS |
| C11 | `visualize_heat_map_examples_by_class` raises `ValueError` if an invalid `cam_type` is supplied | `MPLCONFIGDIR=/tmp/mpl PYTHONPATH=src .venv/bin/python3 -m unittest -v tests.test_n01_parquet_bridge.TestInterpretabilityGradCAM.test_c11_invalid_cam_type_raises_value_error` exit 0 | `tests/test_n01_parquet_bridge.py:260` - `with self.assertRaises(ValueError):` | PASS |
| C12 | `compute_and_plot_confusion_insights` returns sorted Top-K off-diagonal confused pairs | `MPLCONFIGDIR=/tmp/mpl PYTHONPATH=src .venv/bin/python3 -m unittest -v tests.test_n01_parquet_bridge.TestConfusionAnalysis.test_c12_top_k_confusion_pairs_ranking` exit 0 | `tests/test_n01_parquet_bridge.py:319` - `self.assertGreaterEqual(pairs[0][0], pairs[1][0])` | PASS |
| C13 | Resolves 0-based `model_label` to 1-based taxonomic taxon names using `viral_cnn.taxonomy.mapping` | `MPLCONFIGDIR=/tmp/mpl PYTHONPATH=src .venv/bin/python3 -m unittest -v tests.test_n01_parquet_bridge.TestConfusionAnalysis.test_c13_taxonomic_name_resolution` exit 0 | `tests/test_n01_parquet_bridge.py:326` - `self.assertEqual(name, "Alpha virus")` | PASS |
| C14 | `get_confused_class_samples` collects up to `n_examples_per_class` input tensors per true class and stops early | `MPLCONFIGDIR=/tmp/mpl PYTHONPATH=src .venv/bin/python3 -m unittest -v tests.test_n01_parquet_bridge.TestConfusionAnalysis.test_c14_get_confused_class_samples_quota` exit 0 | `tests/test_n01_parquet_bridge.py:345` - `self.assertLessEqual(len(item["class_true"]["examples"]), 2)` | PASS |
| C15 | `setup_notebook_runtime` sets seeds deterministically and resolves `DATA_ROOT` without personal paths | `MPLCONFIGDIR=/tmp/mpl PYTHONPATH=src .venv/bin/python3 -m unittest -v tests.test_n01_parquet_bridge.TestNotebookRuntime.test_c15_setup_runtime_and_reproducible_seeding` exit 0 | `tests/test_n01_parquet_bridge.py:358` - `self.assertEqual(ctx.seed, 42)` | PASS |
| C16 | `JupyterHubDataTopology` maps all canonical `short/` and `long/` paths from `configs/models.yaml` | `MPLCONFIGDIR=/tmp/mpl PYTHONPATH=src .venv/bin/python3 -m unittest -v tests.test_n01_parquet_bridge.TestNotebookRuntime.test_c16_jupyterhub_data_topology_resolution` exit 0 | `tests/test_n01_parquet_bridge.py:365` - `self.assertIn("short", topology.paths)` | PASS |
| C17 | `ensure_smoke_test_artifacts` synthesizes deterministic temporary Parquet partitions and taxonomy CSVs | `MPLCONFIGDIR=/tmp/mpl PYTHONPATH=src .venv/bin/python3 -m unittest -v tests.test_n01_parquet_bridge.TestNotebookRuntime.test_c17_synthetic_smoke_test_artifacts` exit 0 | `tests/test_n01_parquet_bridge.py:372` - `self.assertTrue(Path(artifacts["short"]["test_parquet"]).exists())` | PASS |
| C18 | `write_notebook_manifest` serializes structured execution manifest in JSON and YAML formats | `MPLCONFIGDIR=/tmp/mpl PYTHONPATH=src .venv/bin/python3 -m unittest -v tests.test_n01_parquet_bridge.TestNotebookRuntime.test_c18_write_notebook_manifest_json_yaml` exit 0 | `tests/test_n01_parquet_bridge.py:397` - `self.assertTrue(os.path.exists(out_json))` | PASS |
| C19 | Historical files and directories in `short/` and `long/` remain strictly unmodified | `MPLCONFIGDIR=/tmp/mpl PYTHONPATH=src .venv/bin/python3 -m unittest -v tests.test_n01_parquet_bridge.TestNotebookRuntime.test_c19_historical_directories_preservation` exit 0 | `tests/test_n01_parquet_bridge.py:405` - `self.assertEqual(res, "")` | PASS |

## Coverage

| Set (size) | Recomputed from | Member -> proof | Unproven |
| --- | --- | --- | --- |
| Parquet dataset column fallbacks (3) | code (`sequence_padded`, `padded_Representation`, `sequence`) | C1, tested across all 3 column configurations | - |
| Parquet dataset shuffle modes (3) | code (`shuffle=False`, `shuffle=True`, `legacy_global_rng=True`) | C4 (`shuffle=False`), C5 (`shuffle=True`), C3 (`legacy_global_rng=True`) | - |
| Parquet dataset execution backends (2) | code (`backend="cpu"`, `backend="cudf"`) | C1 (`cpu`), C6 (`cudf`) | - |
| CAM explainer algorithms (2) | code (`GradCAM1D`, `ScoreCAM1D`) | C7 (both bounded in [0, 1]), C8 (`GradCAM1D`), C9 (`ScoreCAM1D`) | - |
| Supported 1D CNN architectures in CAM (2) | code (`ResNet50_1D`, `ConvNeXt_Tiny_1D`) | C8 (`ResNet50_1D`), C9 (`ConvNeXt_Tiny_1D`) | - |
| CAM display modes (2) | code (`show=False`, `show=True`) | C10 (`show=False` returning Figure), C10 | - |
| Confusion analysis helper outputs (3) | code (`compute_and_plot_confusion_insights`, `get_confused_class_samples`, `inspect_csv`) | C12 (`compute_and_plot_confusion_insights`), C14 (`get_confused_class_samples`), C13 (`inspect_csv`) | - |
| Runtime execution modes (2) | code (Remote JupyterHub cluster mode, Local smoke-test mode) | C15 (cluster mode with data_root), C17 (smoke-test mode) | - |
| Dataset tracks managed by topology (2) | code (`short/`, `long/`) | C16 (both short and long paths validated) | - |
| Manifest serialization formats (2) | code (JSON, YAML) | C18 (both JSON and YAML validated) | - |

## Swept existing

- validation: C1 (inputs shapes/dtypes/tokens 0..4), C11 (invalid `cam_type` raises `ValueError`)
- failure modes: C6 (`backend='cudf'` without cudf raises `ImportError` or warns on fallback)
- idempotency: C4 (`shuffle=False` deterministic iteration), C18 (`write_notebook_manifest` output consistency)
- authorization: n/a - local scientific code with no network boundary or access control
- concurrency: C5 (`DataLoader(num_workers=2)` disjoint file partition assignment)
- data lifecycle: C17 (`ensure_smoke_test_artifacts` temporary partition creation and schema compliance)
- dependency failure: C6 (graceful handling and explicit diagnostic message when optional GPU backend is absent)
- state transitions: n/a - stateless batch dataset iterators and analytical functions
- observability: C18 (`NotebookRunManifest` serialization with run metadata)

## Gate

`MPLCONFIGDIR=/tmp/mpl PYTHONPATH=src .venv/bin/python3 -m unittest -v tests/test_n01_parquet_bridge.py` - 19 passed, 0 failed
