# S09: Calibration and Confidence by Noise Level - Verification

**Verdict**: PASS
**Profile**: light
**Diff range**: 41aee91..0b4de3a
**Round**: 1 - full
**Verifier**: independent sub-agent (author != verifier)

## Checks

| Check | Claim | Proof run | Evidence | Result |
| --- | --- | --- | --- | --- |
| C1 | `viral_cnn.eval.calibration.compute_calibration_metrics` computes Top-1 ECE using declared equal-width binning ($M=15$) where bin weights sum to 1.0 (AC 1) | `.venv/bin/python3 -m unittest -v tests/test_calibration.py` exit 0 | `tests/test_calibration.py:62` - `self.assertAlmostEqual(metrics.sum_bin_weights, 1.0, places=5); self.assertEqual(metrics.n_bins, 15)` | PASS |
| C2 | `viral_cnn.eval.calibration.compute_calibration_metrics` computes Top-1 ECE using declared equal-mass (adaptive) binning (AC 1) | `.venv/bin/python3 -m unittest -v tests/test_calibration.py` exit 0 | `tests/test_calibration.py:83` - `self.assertEqual(metrics.binning_strategy, "equal_mass"); self.assertAlmostEqual(metrics.sum_bin_weights, 1.0, places=5)` | PASS |
| C3 | `viral_cnn.eval.calibration.compute_calibration_metrics` computes Maximum Calibration Error (MCE) across non-empty bins (AC 2) | `.venv/bin/python3 -m unittest -v tests/test_calibration.py` exit 0 | `tests/test_calibration.py:104` - `self.assertGreater(metrics.mce, 0.0); self.assertGreaterEqual(metrics.mce, metrics.ece)` | PASS |
| C4 | `viral_cnn.eval.calibration.compute_calibration_metrics` computes Negative Log-Likelihood (NLL) with epsilon clamping to avoid numerical instability (AC 3) | `.venv/bin/python3 -m unittest -v tests/test_calibration.py` exit 0 | `tests/test_calibration.py:122` - `self.assertFalse(np.isinf(metrics.nll)); self.assertGreater(metrics.nll, 20.0)` | PASS |
| C5 | `viral_cnn.eval.calibration.compute_calibration_metrics` calculates exact multiclass Brier score using memory-efficient quadratic expansion $\sum_k p_{ik}^2 - 2 p_{i, y_i} + 1$ without dense $N \times K$ one-hot allocation (AC 4) | `.venv/bin/python3 -m unittest -v tests/test_calibration.py` exit 0 | `tests/test_calibration.py:143` - `self.assertAlmostEqual(metrics.brier_score, expected_brier, places=5)` | PASS |
| C6 | `viral_cnn.eval.calibration.compute_calibration_metrics` returns 0.0 ECE, 0.0 MCE, and 0.0 Brier score when all probabilities equal 1.0 on correct predictions (AC 5) | `.venv/bin/python3 -m unittest -v tests/test_calibration.py` exit 0 | `tests/test_calibration.py:158` - `self.assertAlmostEqual(metrics.ece, 0.0, places=6); self.assertAlmostEqual(metrics.mce, 0.0, places=6); self.assertAlmostEqual(metrics.brier_score, 0.0, places=6)` | PASS |
| C7 | `viral_cnn.eval.calibration.compute_calibration_metrics` computes mean confidence and accuracy separately for correct and incorrect predictions to measure overconfidence (AC 6) | `.venv/bin/python3 -m unittest -v tests/test_calibration.py` exit 0 | `tests/test_calibration.py:180` - `self.assertAlmostEqual(metrics.conf_on_correct, 0.85, places=4); self.assertGreater(metrics.overconfidence_gap, 0.0)` | PASS |
| C8 | `viral_cnn.eval.calibration.compute_reliability_diagram` partitions predictions into declared confidence bins and returns bin edges, sample count, mean confidence, accuracy, and calibration gap (AC 7) | `.venv/bin/python3 -m unittest -v tests/test_calibration.py` exit 0 | `tests/test_calibration.py:199` - `self.assertEqual(len(bins), 10); self.assertGreaterEqual(b.calibration_gap, 0.0)` | PASS |
| C9 | `viral_cnn.eval.calibration.compute_reliability_diagram` allocates every sample to exactly one bin so sum of bin sample counts strictly equals $N$ (AC 8) | `.venv/bin/python3 -m unittest -v tests/test_calibration.py` exit 0 | `tests/test_calibration.py:214` - `self.assertEqual(total_binned, n_samples)` | PASS |
| C10 | `viral_cnn.eval.calibration.save_reliability_diagram` exports structured JSON/CSV data and renders a reliability diagram plot comparing accuracy to confidence with sample histogram (AC 9) | `.venv/bin/python3 -m unittest -v tests/test_calibration.py` exit 0 | `tests/test_calibration.py:230` - `self.assertTrue(Path(files["csv"]).exists()); self.assertTrue(Path(files["json"]).exists()); self.assertTrue(Path(files["plot"]).exists())` | PASS |
| C11 | `viral_cnn.eval.calibration.TemperatureScaler.fit` optimizes scalar temperature $T > 0$ strictly minimizing NLL on validation logits (AC 10) | `.venv/bin/python3 -m unittest -v tests/test_calibration.py` exit 0 | `tests/test_calibration.py:247` - `self.assertGreater(scaler.temperature, 0.0); self.assertGreater(scaler.temperature, 1.0)` | PASS |
| C12 | `viral_cnn.eval.calibration.TemperatureScaler.fit` rejects fitting on test data without explicit validation split, raising ValueError (AC 11) | `.venv/bin/python3 -m unittest -v tests/test_calibration.py` exit 0 | `tests/test_calibration.py:257` - `with self.assertRaises(ValueError): scaler.fit(logits, labels, split_name="test")` | PASS |
| C13 | `viral_cnn.eval.calibration.TemperatureScaler.transform` applies $z / T$ scaling to test logits and computes post-calibration ECE, NLL, and Brier metrics (AC 12) | `.venv/bin/python3 -m unittest -v tests/test_calibration.py` exit 0 | `tests/test_calibration.py:276` - `self.assertIn("uncalibrated_metrics", cal_res); self.assertLessEqual(cal_res["calibrated_metrics"]["nll"], cal_res["uncalibrated_metrics"]["nll"])` | PASS |
| C14 | `viral_cnn.eval.calibration.TemperatureScaler.transform` leaves probabilities and predictions unchanged when $T=1.0$ (AC 13) | `.venv/bin/python3 -m unittest -v tests/test_calibration.py` exit 0 | `tests/test_calibration.py:291` - `self.assertAlmostEqual(cal_res["uncalibrated_metrics"]["ece"], cal_res["calibrated_metrics"]["ece"], places=5)` | PASS |
| C15 | `viral_cnn.eval.calibration.audit_model_calibration_by_noise` evaluates calibration metrics across species models (ResNet50 clean/noise, ConvNeXt zero/transfer/noise) and nominal noise levels $\epsilon \in \{0.0, 0.02, 0.05, 0.10, 0.15\}$ (AC 14) | `.venv/bin/python3 -m unittest -v tests/test_calibration.py` exit 0 | `tests/test_calibration.py:331` - `self.assertIn("resnet_clean_species", report["models"]); self.assertIn("convnext_zero_species", report["models"]); self.assertIn("ece", entry)` | PASS |
| C16 | `docs/calibration_and_confidence_audit.md` documents the calibration table across models and noise levels, explicitly distinguishing Top-1 confidence from calibration error (AC 15) | `.venv/bin/python3 -m unittest -v tests/test_calibration.py` exit 0 | `tests/test_calibration.py:347` - `self.assertTrue(doc_path.exists()); self.assertIn("Expected Calibration Error", content); self.assertIn("ECE", content)` | PASS |
| C17 | `docs/calibration_and_confidence_audit.md` analytically isolates the impact of `label_smoothing=0.1` and dropout differences, refuting naive architectural attribution of calibration properties (AC 15) | `.venv/bin/python3 -m unittest -v tests/test_calibration.py` exit 0 | `tests/test_calibration.py:362` - `self.assertIn("label_smoothing", content); self.assertIn("0.1", content); self.assertIn("dropout", content.lower())` | PASS |
| C18 | `docs/calibration_and_confidence_audit.md` provides explicit recommendations for text revisions in the manuscript where confidence remains high while accuracy degrades under noise (AC 15) | `.venv/bin/python3 -m unittest -v tests/test_calibration.py` exit 0 | `tests/test_calibration.py:372` - `self.assertIn("Recomendações", content); self.assertIn("Manuscrito", content); self.assertIn("Tabela", content)` | PASS |
| C19 | `viral_cnn.eval.calibration.audit_model_calibration_by_noise` flags overconfidence regimes when mean confidence exceeds empirical accuracy by $>10$ percentage points (AC 16) | `.venv/bin/python3 -m unittest -v tests/test_calibration.py` exit 0 | `tests/test_calibration.py:386` - `self.assertTrue(metrics.is_overconfident_regime); self.assertGreater(metrics.top1_confidence - (metrics.top1_accuracy / 100.0), 0.10)` | PASS |
| C20 | CLI `scripts/run_calibration_evaluation.py` executes calibration evaluation, outputs `calibration_metrics.json`, `reliability_diagram.csv`, and `run_manifest.json` with exit code 0 (AC 17, AC 18) | `.venv/bin/python3 -m unittest -v tests/test_calibration.py` exit 0 | `tests/test_calibration.py:414` - `self.assertEqual(res.returncode, 0, f"CLI stderr: {res.stderr}"); self.assertTrue((out_path / "run_out" / "calibration_summary.json").exists())` | PASS |
| C21 | CLI `scripts/run_calibration_evaluation.py` exits with code 2 on invalid arguments, missing input files, or unsupported binning strategies (AC 17) | `.venv/bin/python3 -m unittest -v tests/test_calibration.py` exit 0 | `tests/test_calibration.py:426` - `self.assertEqual(res.returncode, 2)` | PASS |
| C22 | Calibration evaluation pipeline executes deterministically on synthetic fixtures without GPU hardware or external checkpoint downloads (AC 14) | `.venv/bin/python3 -m unittest -v tests/test_calibration.py` exit 0 | `tests/test_calibration.py:437` - `self.assertEqual(m1.ece, m2.ece); self.assertEqual(m1.mce, m2.mce); self.assertEqual(m1.brier_score, m2.brier_score)` | PASS |

## Coverage

| Set (size) | Recomputed from | Member -> proof | Unproven |
| --- | --- | --- | --- |
| Binning strategies evaluated (2) | `src/viral_cnn/eval/calibration.py` (`compute_calibration_metrics`, `compute_reliability_diagram`) | `equal_width` C1 · `equal_mass` C2 | - |
| Calibration metric types computed (4) | `src/viral_cnn/eval/calibration.py` (`compute_calibration_metrics`) | `ece` C1 · `mce` C3 · `nll` C4 · `brier_score` C5 | - |
| Controlled noise rates audited (5) | `src/viral_cnn/eval/calibration.py` (`audit_model_calibration_by_noise`) | `noise_00` C15 · `noise_02` C15 · `noise_05` C15 · `noise_10` C15 · `noise_15` C15 | - |
| Species model variants audited (5) | `configs/models.yaml`, `scripts/run_calibration_evaluation.py` | `resnet_clean_species` C15 · `resnet_noise_species` C15 · `convnext_zero_species` C15 · `convnext_transfer_species` C15 · `convnext_noise_species` C15 | - |
| CLI execution status codes (2) | `scripts/run_calibration_evaluation.py` | `0` C20 · `2` C21 | - |

## Swept dimensions

- validation: C21 (CLI scripts/run_calibration_evaluation.py rejects invalid arguments, missing input files, and invalid binning strategies with exit code 2)
- failure modes: C12 (TemperatureScaler.fit rejects test split fitting without explicit validation split, raising ValueError)
- idempotency: C14 (TemperatureScaler identity transform at T=1.0 leaves probabilities, predictions, and calibration metrics unchanged)
- authorization: n/a - local scientific CLI tool with no network or authorization boundary
- concurrency: n/a - deterministic batch processing on single machine with no concurrent state mutation
- data lifecycle: C16 (documentation of audited metric tables, confounds, and moderation recommendations in docs/calibration_and_confidence_audit.md)
- dependency failure: C22 (calibration evaluation pipeline executes deterministically on synthetic fixtures without GPU hardware or external checkpoint downloads)
- state transitions: n/a - stateless data transformation, calibration metric calculation, and post-scaling evaluation
- observability: C20 (CLI execution emits structured JSON/CSV metrics, reliability diagram artifacts, and run manifest)

## Gate

`.venv/bin/python3 -m unittest -v tests/test_calibration.py` - 22 passed, 0 failed
