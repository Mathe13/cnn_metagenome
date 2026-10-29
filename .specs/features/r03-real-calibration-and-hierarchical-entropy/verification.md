# r03-real-calibration-and-hierarchical-entropy verification

**Verdict**: PASS
**Profile**: light
**Diff range**: 392bedc..ce987c2
**Round**: 1 - full
**Verifier**: independent sub-agent (author != verifier)

## Checks

| Check | Claim | Proof run | Evidence | Result |
| --- | --- | --- | --- | --- |
| C1 | `build_species_to_genus_tensor` loads taxonomy mapping and returns a `torch.LongTensor` of length equal to `num_species` with distinct genus indices in $[0, N_{\text{genera}}-1]$ where unique genus count is $> 1$ | `MPLCONFIGDIR=/tmp/matplotlib PYTHONPATH=src .venv/bin/python3 -m unittest tests/test_n02_benchmark_and_calibration.py -k test_r03_build_species_to_genus_tensor_valid` exit 0 | `tests/test_n02_benchmark_and_calibration.py:636` - `self.assertIsInstance(tensor, torch.Tensor)`, `self.assertEqual(tensor.shape, (10,))`, `self.assertEqual(tensor.unique().numel(), 5)` | PASS |
| C2 | `build_species_to_genus_tensor` raises `ValueError` if given a degenerate mapping where all species collapse into a single genus | `MPLCONFIGDIR=/tmp/matplotlib PYTHONPATH=src .venv/bin/python3 -m unittest tests/test_n02_benchmark_and_calibration.py -k test_r03_build_species_to_genus_tensor_degenerate_error` exit 0 | `tests/test_n02_benchmark_and_calibration.py:666` - `with self.assertRaises(ValueError) as ctx_err: build_species_to_genus_tensor(...)`, `self.assertIn("Degenerate taxonomy mapping", str(ctx_err.exception))` | PASS |
| C3 | `TaxonomyEntropyAnalyzer` produces $H_{\text{genus}} > 0$, $H_{\text{species}} \ge H_{\text{genus}}$, and finite $H_{\text{act}}$ with zero NaN entries under FP16 and FP32 with non-degenerate `species_to_genus` mapping | `MPLCONFIGDIR=/tmp/matplotlib PYTHONPATH=src .venv/bin/python3 -m unittest tests/test_n02_benchmark_and_calibration.py -k test_r03_taxonomy_entropy_analyzer_valid_bounds` exit 0 | `tests/test_n02_benchmark_and_calibration.py:706` - `self.assertTrue(np.all(h_species > 0.0))`, `self.assertTrue(np.all(h_genus > 0.0))`, `self.assertTrue(np.all(gap >= -1e-6))`, `self.assertFalse(np.isnan(h_act).any())` | PASS |
| C4 | `audit_dataloader_calibration_by_noise` computes Top-1 accuracy, confidence, ECE, MCE, NLL, and Brier score streaming batches across noise rates without allocating full $(N, C)$ matrices | `MPLCONFIGDIR=/tmp/matplotlib PYTHONPATH=src .venv/bin/python3 -m unittest tests/test_n02_benchmark_and_calibration.py -k test_r03_audit_dataloader_calibration_streaming` exit 0 | `tests/test_n02_benchmark_and_calibration.py:746` - `self.assertEqual(metrics["n_samples"], 64)`, `self.assertGreaterEqual(metrics["top1_accuracy"], 0.0)`, `self.assertGreaterEqual(metrics["ece"], 0.0)`, `self.assertGreaterEqual(metrics["brier_score"], 0.0)` | PASS |
| C5 | `audit_dataloader_calibration_by_noise` fits optimal temperature $T^*$ exclusively on `val_loader` when provided and returns calibrated metrics | `MPLCONFIGDIR=/tmp/matplotlib PYTHONPATH=src .venv/bin/python3 -m unittest tests/test_n02_benchmark_and_calibration.py -k test_r03_audit_dataloader_calibration_temperature_scaling` exit 0 | `tests/test_n02_benchmark_and_calibration.py:790` - `self.assertIn("optimal_temperature", metrics)`, `self.assertGreater(metrics["optimal_temperature"], 0.0)`, `self.assertIn("calibrated_ece", metrics)`, `self.assertIn("calibrated_brier", metrics)` | PASS |
| C6 | `notebooks/reproduction/short/02_analyze_confidence_and_calibration.ipynb` extracts real sequences for confused pairs in Cell 4, uses non-degenerate `species_to_genus` and evaluates 5 species variants in Cell 6, runs `audit_dataloader_calibration_by_noise` in Cell 7, and prints full dataframes `df_identity`, `df_entropy`, and `df_cal` without `"ACGT" * 128` or `torch.zeros(13683)` outside smoke mode | `MPLCONFIGDIR=/tmp/matplotlib PYTHONPATH=src .venv/bin/python3 -m unittest tests/test_n02_benchmark_and_calibration.py -k test_r03_notebook_02_real_data_and_table_prints` exit 0 | `tests/test_n02_benchmark_and_calibration.py:810` - `self.assertIn("samples_by_id", cell_4_src)`, `self.assertNotIn('"ACGT" * 128', cell_4_src)`, `self.assertIn("build_species_to_genus_tensor", cell_6_src)`, `self.assertIn("audit_dataloader_calibration_by_noise", cell_7_src)` | PASS |

## Gate

`MPLCONFIGDIR=/tmp/matplotlib PYTHONPATH=src .venv/bin/python3 -m unittest tests/test_n02_benchmark_and_calibration.py -k "test_r03_"` - 6 passed, 0 failed
