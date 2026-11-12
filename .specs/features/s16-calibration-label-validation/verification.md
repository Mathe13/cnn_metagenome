# Verification Report: s16-calibration-label-validation

**Verdict**: PASS
**Status**: VERIFIED
**Profile**: light
**Round**: 1 - full
**Verifier**: Independent Subagent (author != verifier)
**Date**: 2026-11-11
**Commit**: 3751937

## Criteria Checked against Plan and Checks

All 7 acceptance criteria from `.specs/features/s16-calibration-label-validation/plan.md` and `.specs/features/s16-calibration-label-validation/checks.md` were independently evaluated and verified against commit `3751937`:

- **AC 1 / C1 (Valid Labels Default Policy):** `compute_calibration_metrics` called with ground-truth labels in `[0, n_classes)` under default `invalid_label_policy="raise"` returns a `CalibrationMetrics` dataclass with `n_samples` matching the input length.
- **AC 2 / C2 (Invalid Labels Fail-Fast):** `compute_calibration_metrics` called with negative labels (`-1`) or out-of-bounds labels (`>= n_classes`) under default `invalid_label_policy="raise"` raises `ValueError` detailing the count of invalid labels, `n_classes`, and sample invalid values.
- **AC 3 / C3 (Explicit Exclude Policy):** `compute_calibration_metrics` called with `invalid_label_policy="exclude"` filters out invalid samples, updates `n_samples` to match the filtered count, and computes ECE, NLL, and Brier score over valid samples only without silent denominator inflation.
- **AC 4 / C4 (Exclude Policy All Invalid):** `compute_calibration_metrics` called with `invalid_label_policy="exclude"` when 0 valid samples remain raises `ValueError("All samples were excluded due to invalid labels.")`.
- **AC 5 / C5 (Policy String Validation):** `compute_calibration_metrics` called with an unsupported policy string (e.g. `"clamp"`) raises `ValueError` stating the policy must be 'raise' or 'exclude'.
- **AC 6 / C6 (Reliability Diagram Parity - Raise):** `compute_reliability_diagram` called with invalid labels under `invalid_label_policy="raise"` raises `ValueError`.
- **AC 7 / C7 (Reliability Diagram Parity - Exclude):** `compute_reliability_diagram` called with `invalid_label_policy="exclude"` conserves the filtered sample count across bins (`sum(bin.count) == n_valid`).

## Checks Table

| Check | Claim | Proof run | Evidence | Result |
| --- | --- | --- | --- | --- |
| C1 | `compute_calibration_metrics` with valid labels under default `invalid_label_policy="raise"` returns `CalibrationMetrics` with `n_samples` equal to input length (AC 1) | `REMOTE_SERVER=False PYTHONPATH=src .venv/bin/python3 -m unittest tests.test_s16_calibration_label_validation.TestS16CalibrationLabelValidation.test_valid_labels_compute_metrics_default_policy -k test_valid_labels_compute_metrics_default_policy` | `tests/test_s16_calibration_label_validation.py:32` - `metrics = compute_calibration_metrics(self.probs, self.valid_labels)` | PASS |
| C2 | `compute_calibration_metrics` with labels outside `[0, n_classes)` under default `invalid_label_policy="raise"` raises `ValueError` with descriptive message (AC 2) | `REMOTE_SERVER=False PYTHONPATH=src .venv/bin/python3 -m unittest tests.test_s16_calibration_label_validation.TestS16CalibrationLabelValidation.test_invalid_labels_fail_fast_default_policy -k test_invalid_labels_fail_fast_default_policy` | `tests/test_s16_calibration_label_validation.py:46` - `with self.assertRaises(ValueError) as ctx:` | PASS |
| C3 | `compute_calibration_metrics` with `invalid_label_policy="exclude"` calculates metrics over valid samples only, with `n_samples` reflecting filtered count (AC 3) | `REMOTE_SERVER=False PYTHONPATH=src .venv/bin/python3 -m unittest tests.test_s16_calibration_label_validation.TestS16CalibrationLabelValidation.test_exclude_policy_filters_invalid_samples -k test_exclude_policy_filters_invalid_samples` | `tests/test_s16_calibration_label_validation.py:67` - `metrics = compute_calibration_metrics(self.probs, corrupted_labels, invalid_label_policy="exclude")` | PASS |
| C4 | `compute_calibration_metrics` with `invalid_label_policy="exclude"` and zero valid samples raises `ValueError` (AC 4) | `REMOTE_SERVER=False PYTHONPATH=src .venv/bin/python3 -m unittest tests.test_s16_calibration_label_validation.TestS16CalibrationLabelValidation.test_exclude_policy_all_invalid_raises_value_error -k test_exclude_policy_all_invalid_raises_value_error` | `tests/test_s16_calibration_label_validation.py:92` - `with self.assertRaises(ValueError) as ctx:` | PASS |
| C5 | `compute_calibration_metrics` with unsupported policy string raises `ValueError` (AC 5) | `REMOTE_SERVER=False PYTHONPATH=src .venv/bin/python3 -m unittest tests.test_s16_calibration_label_validation.TestS16CalibrationLabelValidation.test_unrecognized_policy_raises_value_error -k test_unrecognized_policy_raises_value_error` | `tests/test_s16_calibration_label_validation.py:102` - `with self.assertRaises(ValueError) as ctx:` | PASS |
| C6 | `compute_reliability_diagram` with invalid labels under `invalid_label_policy="raise"` raises `ValueError` (AC 6) | `REMOTE_SERVER=False PYTHONPATH=src .venv/bin/python3 -m unittest tests.test_s16_calibration_label_validation.TestS16CalibrationLabelValidation.test_reliability_diagram_raises_on_invalid_labels -k test_reliability_diagram_raises_on_invalid_labels` | `tests/test_s16_calibration_label_validation.py:115` - `with self.assertRaises(ValueError) as ctx:` | PASS |
| C7 | `compute_reliability_diagram` with `invalid_label_policy="exclude"` returns bins whose total count strictly matches filtered sample count (AC 7) | `REMOTE_SERVER=False PYTHONPATH=src .venv/bin/python3 -m unittest tests.test_s16_calibration_label_validation.TestS16CalibrationLabelValidation.test_reliability_diagram_exclude_policy_conserves_samples -k test_reliability_diagram_exclude_policy_conserves_samples` | `tests/test_s16_calibration_label_validation.py:124` - `bins = compute_reliability_diagram(self.probs, corrupted_labels, invalid_label_policy="exclude")` | PASS |

## Coverage

| Set (size) | Member -> proof | Unproven |
| --- | --- | --- |
| policy options (2) | `raise_policy` C1, C2 · `exclude_policy` C3, C4 | - |
| input boundary violations (2) | `negative_label` C2 · `out_of_bounds_label` C2 | - |
| policy validation (1) | `unrecognized_policy` C5 | - |
| reliability diagram parity (2) | `diagram_raise` C6 · `diagram_exclude` C7 | - |

## Landing Doors Verification

- **Door 1 (fail-fast por padrão para rótulos inválidos):** Implemented in `src/viral_cnn/eval/calibration.py:86` via `_validate_and_filter_labels` and configured as `invalid_label_policy: Literal["raise", "exclude"] = "raise"` in both `compute_calibration_metrics` (`src/viral_cnn/eval/calibration.py:220`) and `compute_reliability_diagram` (`src/viral_cnn/eval/calibration.py:145`). Proven by C1, C2, and C6.
- **Door 2 (mensagem de erro acionável com contagem e exemplos):** Implemented in `src/viral_cnn/eval/calibration.py:117-122` where `sample_invalids = y_np[invalid_mask][:5].tolist()` and `raise ValueError(f"Found {n_invalid} invalid labels outside [0, {n_classes}). Examples: {sample_invalids}.")`. Proven by C2 and C6.
- **Door 3 (exclusão explícita com conservação do denominador):** Implemented in `src/viral_cnn/eval/calibration.py:123-127` where invalid samples are filtered out before metric calculation and `p_true = p_np[np.arange(n_samples), y_np]` (`src/viral_cnn/eval/calibration.py:274`) indexes directly into valid targets without silent zero-clamping. Proven by C3, C4, and C7.

## Full Test Suite Gate

- `REMOTE_SERVER=False PYTHONPATH=src .venv/bin/python3 -m unittest -v tests/test_s16_calibration_label_validation.py`: 7 passed, 0 failed.
- `MPLCONFIGDIR=/tmp REMOTE_SERVER=False PYTHONPATH=src .venv/bin/python3 -m unittest -v tests/test_calibration.py`: 22 passed, 0 failed.

## Summary and Signature

The implementation of `s16-calibration-label-validation` in commit `3751937` eliminates silent zero-probability masking on out-of-vocabulary and negative labels, establishes strict closed-set fail-fast validation by default (`invalid_label_policy="raise"`), provides an explicit opt-in policy (`invalid_label_policy="exclude"`) that transparently updates sample sizes and denominators, and maintains behavioral parity between `compute_calibration_metrics` and `compute_reliability_diagram`. All checks C1 through C7 have been independently reproduced and verified under profile light.

**Signed by:** Independent Subagent Verifier
