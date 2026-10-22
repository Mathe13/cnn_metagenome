# S09: Calibration and Confidence by Noise Level - Checks

Profile: light
Plan: `.specs/features/s09-calibration-and-confidence-by-noise/plan.md`

## Checks

### S1: Formal Probabilistic Calibration Metrics · 2 files · 22 KB · ~6k

**C1** - `viral_cnn.eval.calibration.compute_calibration_metrics` computes Top-1 ECE using declared equal-width binning ($M=15$) where bin weights sum to 1.0 (AC 1)
Proof: `.venv/bin/python3 -m unittest discover -s tests -k test_compute_calibration_metrics_ece_equal_width`

**C2** - `viral_cnn.eval.calibration.compute_calibration_metrics` computes Top-1 ECE using declared equal-mass (adaptive) binning (AC 1)
Proof: `.venv/bin/python3 -m unittest discover -s tests -k test_compute_calibration_metrics_ece_equal_mass`

**C3** - `viral_cnn.eval.calibration.compute_calibration_metrics` computes Maximum Calibration Error (MCE) across non-empty bins (AC 2)
Proof: `.venv/bin/python3 -m unittest discover -s tests -k test_compute_calibration_metrics_mce`

**C4** - `viral_cnn.eval.calibration.compute_calibration_metrics` computes Negative Log-Likelihood (NLL) with epsilon clamping to avoid numerical instability (AC 3)
Proof: `.venv/bin/python3 -m unittest discover -s tests -k test_compute_calibration_metrics_nll_clamped`

**C5** - `viral_cnn.eval.calibration.compute_calibration_metrics` calculates exact multiclass Brier score using memory-efficient quadratic expansion $\sum_k p_{ik}^2 - 2 p_{i, y_i} + 1$ without dense $N \times K$ one-hot allocation (AC 4)
Proof: `.venv/bin/python3 -m unittest discover -s tests -k test_compute_calibration_metrics_brier_memory_efficient`

**C6** - `viral_cnn.eval.calibration.compute_calibration_metrics` returns 0.0 ECE, 0.0 MCE, and 0.0 Brier score when all probabilities equal 1.0 on correct predictions (AC 5)
Proof: `.venv/bin/python3 -m unittest discover -s tests -k test_compute_calibration_metrics_perfect_predictions_zero_error`

**C7** - `viral_cnn.eval.calibration.compute_calibration_metrics` computes mean confidence and accuracy separately for correct and incorrect predictions to measure overconfidence (AC 6)
Proof: `.venv/bin/python3 -m unittest discover -s tests -k test_compute_calibration_metrics_overconfidence_partition`

### S2: Reliability Diagrams and Bin Analysis · 2 files · 20 KB · ~5k

**C8** - `viral_cnn.eval.calibration.compute_reliability_diagram` partitions predictions into declared confidence bins and returns bin edges, sample count, mean confidence, accuracy, and calibration gap (AC 7)
Proof: `.venv/bin/python3 -m unittest discover -s tests -k test_compute_reliability_diagram_structured_table`

**C9** - `viral_cnn.eval.calibration.compute_reliability_diagram` allocates every sample to exactly one bin so sum of bin sample counts strictly equals $N$ (AC 8)
Proof: `.venv/bin/python3 -m unittest discover -s tests -k test_compute_reliability_diagram_sample_conservation`

**C10** - `viral_cnn.eval.calibration.save_reliability_diagram` exports structured JSON/CSV data and renders a reliability diagram plot comparing accuracy to confidence with sample histogram (AC 9)
Proof: `.venv/bin/python3 -m unittest discover -s tests -k test_save_reliability_diagram_plot_and_table`

### S3: Validation-Only Temperature Scaling · 2 files · 20 KB · ~5k

**C11** - `viral_cnn.eval.calibration.TemperatureScaler.fit` optimizes scalar temperature $T > 0$ strictly minimizing NLL on validation logits (AC 10)
Proof: `.venv/bin/python3 -m unittest discover -s tests -k test_temperature_scaler_fit_minimizes_validation_nll`

**C12** - `viral_cnn.eval.calibration.TemperatureScaler.fit` rejects fitting on test data without explicit validation split, raising ValueError (AC 11)
Proof: `.venv/bin/python3 -m unittest discover -s tests -k test_temperature_scaler_rejects_test_fitting`

**C13** - `viral_cnn.eval.calibration.TemperatureScaler.transform` applies $z / T$ scaling to test logits and computes post-calibration ECE, NLL, and Brier metrics (AC 12)
Proof: `.venv/bin/python3 -m unittest discover -s tests -k test_temperature_scaler_transform_reduces_test_ece`

**C14** - `viral_cnn.eval.calibration.TemperatureScaler.transform` leaves probabilities and predictions unchanged when $T=1.0$ (AC 13)
Proof: `.venv/bin/python3 -m unittest discover -s tests -k test_temperature_scaler_identity_at_t1`

### S4: Controlled Noise Calibration Audit and Scientific Moderation · 3 files · 30 KB · ~8k

**C15** - `viral_cnn.eval.calibration.audit_model_calibration_by_noise` evaluates calibration metrics across species models (ResNet50 clean/noise, ConvNeXt zero/transfer/noise) and nominal noise levels $\epsilon \in \{0.0, 0.02, 0.05, 0.10, 0.15\}$ (AC 14)
Proof: `.venv/bin/python3 -m unittest discover -s tests -k test_audit_model_calibration_by_noise_multi_rate`

**C16** - `docs/calibration_and_confidence_audit.md` documents the calibration table across models and noise levels, explicitly distinguishing Top-1 confidence from calibration error (AC 15)
Proof: `.venv/bin/python3 -m unittest discover -s tests -k test_calibration_audit_doc_distinguishes_confidence_and_calibration`

**C17** - `docs/calibration_and_confidence_audit.md` analytically isolates the impact of `label_smoothing=0.1` and dropout differences, refuting naive architectural attribution of calibration properties (AC 15)
Proof: `.venv/bin/python3 -m unittest discover -s tests -k test_calibration_audit_doc_isolates_label_smoothing_confounds`

**C18** - `docs/calibration_and_confidence_audit.md` provides explicit recommendations for text revisions in the manuscript where confidence remains high while accuracy degrades under noise (AC 15)
Proof: `.venv/bin/python3 -m unittest discover -s tests -k test_calibration_audit_doc_provides_text_moderation_recommendations`

**C19** - `viral_cnn.eval.calibration.audit_model_calibration_by_noise` flags overconfidence regimes when mean confidence exceeds empirical accuracy by $>10$ percentage points (AC 16)
Proof: `.venv/bin/python3 -m unittest discover -s tests -k test_audit_model_calibration_flags_overconfidence_regime`

### S5: CLI Entrypoint and Reproducible Manifests · 2 files · 24 KB · ~6k

**C20** - CLI `scripts/run_calibration_evaluation.py` executes calibration evaluation, outputs `calibration_metrics.json`, `reliability_diagram.csv`, and `run_manifest.json` with exit code 0 (AC 17, AC 18)
Proof: `.venv/bin/python3 -m unittest discover -s tests -k test_run_calibration_cli_exit_zero_and_artifacts`

**C21** - CLI `scripts/run_calibration_evaluation.py` exits with code 2 on invalid arguments, missing input files, or unsupported binning strategies (AC 17)
Proof: `.venv/bin/python3 -m unittest discover -s tests -k test_run_calibration_cli_invalid_args_exit_2`

**C22** - Calibration evaluation pipeline executes deterministically on synthetic fixtures without GPU hardware or external checkpoint downloads (AC 14)
Proof: `.venv/bin/python3 -m unittest discover -s tests -k test_calibration_pipeline_synthetic_deterministic_execution`

## Coverage

| Set (size) | Member -> proof | Unproven |
| --- | --- | --- |
| Binning strategies evaluated (2) | `equal_width` C1 · `equal_mass` C2 | - |
| Calibration metric types computed (4) | `ece` C1 · `mce` C3 · `nll` C4 · `brier_score` C5 | - |
| Controlled noise rates audited (5) | `noise_00` C15 · `noise_02` C15 · `noise_05` C15 · `noise_10` C15 · `noise_15` C15 | - |
| Species model variants audited (5) | `resnet_clean_species` C15 · `resnet_noise_species` C15 · `convnext_zero_species` C15 · `convnext_transfer_species` C15 · `convnext_noise_species` C15 | - |
| CLI execution status codes (2) | `0` C20 · `2` C21 | - |

- Claims naming a status code or response shape: C20, C21 - each has a proof crossing the process boundary
- No other check claims more than the single case its proof exercises

## Swept

- validation: C21
- failure modes: C12
- idempotency: C14
- authorization: n/a - local scientific CLI tool with no network or authorization boundary
- concurrency: n/a - deterministic batch processing on single machine with no concurrent state mutation
- data lifecycle: C16
- dependency failure: C22
- state transitions: n/a - stateless data transformation and metric computation
- observability: C20

## Handoff

- S1 = 6k in calibration metrics engine; S2 = 5k in reliability diagrams; S3 = 5k in temperature scaling; S4 = 8k in noise audit and documentation; S5 = 6k in CLI and tests. Total = ~30k tokens, well under the 150k budget - one builder

- **Mechanism:** one builder

