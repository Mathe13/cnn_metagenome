# S16 Calibration Label Validation checks

Profile: light
Plan: `.specs/features/s16-calibration-label-validation/plan.md`

## Intent

Calibration metrics historically tolerated labels outside `[0, n_classes)` by clamping and assigning `p_true=0`, inflating Negative Log-Likelihood and distorting Brier scores on malformed inputs. This feature enforces closed-set label validity (`0 <= y_true < n_classes`) fail-fast by default (`invalid_label_policy="raise"`), reporting error counts and examples, while providing an explicit opt-in policy (`invalid_label_policy="exclude"`) that updates sample counts and denominators transparently.

7 checks in 1 slice · 3 one-way doors · 0 open, of which 0 block

## Checks

### S1 - Calibration label fail-fast validation and policy enforcement · 2 files · ~25 KB · ~7k

**C1** - `compute_calibration_metrics` with valid labels under default `invalid_label_policy="raise"` returns `CalibrationMetrics` with `n_samples` equal to input length (AC 1)
Proof: `REMOTE_SERVER=False PYTHONPATH=src .venv/bin/python3 -m unittest tests.test_s16_calibration_label_validation.TestS16CalibrationLabelValidation.test_valid_labels_compute_metrics_default_policy -k test_valid_labels_compute_metrics_default_policy`

**C2** - `compute_calibration_metrics` with labels outside `[0, n_classes)` under default `invalid_label_policy="raise"` raises `ValueError` with descriptive message (AC 2)
Proof: `REMOTE_SERVER=False PYTHONPATH=src .venv/bin/python3 -m unittest tests.test_s16_calibration_label_validation.TestS16CalibrationLabelValidation.test_invalid_labels_fail_fast_default_policy -k test_invalid_labels_fail_fast_default_policy`

**C3** - `compute_calibration_metrics` with `invalid_label_policy="exclude"` calculates metrics over valid samples only, with `n_samples` reflecting filtered count (AC 3)
Proof: `REMOTE_SERVER=False PYTHONPATH=src .venv/bin/python3 -m unittest tests.test_s16_calibration_label_validation.TestS16CalibrationLabelValidation.test_exclude_policy_filters_invalid_samples -k test_exclude_policy_filters_invalid_samples`

**C4** - `compute_calibration_metrics` with `invalid_label_policy="exclude"` and zero valid samples raises `ValueError` (AC 4)
Proof: `REMOTE_SERVER=False PYTHONPATH=src .venv/bin/python3 -m unittest tests.test_s16_calibration_label_validation.TestS16CalibrationLabelValidation.test_exclude_policy_all_invalid_raises_value_error -k test_exclude_policy_all_invalid_raises_value_error`

**C5** - `compute_calibration_metrics` with unsupported policy string raises `ValueError` (AC 5)
Proof: `REMOTE_SERVER=False PYTHONPATH=src .venv/bin/python3 -m unittest tests.test_s16_calibration_label_validation.TestS16CalibrationLabelValidation.test_unrecognized_policy_raises_value_error -k test_unrecognized_policy_raises_value_error`

**C6** - `compute_reliability_diagram` with invalid labels under `invalid_label_policy="raise"` raises `ValueError` (AC 6)
Proof: `REMOTE_SERVER=False PYTHONPATH=src .venv/bin/python3 -m unittest tests.test_s16_calibration_label_validation.TestS16CalibrationLabelValidation.test_reliability_diagram_raises_on_invalid_labels -k test_reliability_diagram_raises_on_invalid_labels`

**C7** - `compute_reliability_diagram` with `invalid_label_policy="exclude"` returns bins whose total count strictly matches filtered sample count (AC 7)
Proof: `REMOTE_SERVER=False PYTHONPATH=src .venv/bin/python3 -m unittest tests.test_s16_calibration_label_validation.TestS16CalibrationLabelValidation.test_reliability_diagram_exclude_policy_conserves_samples -k test_reliability_diagram_exclude_policy_conserves_samples`

## Coverage

| Set (size) | Member -> proof | Unproven |
| --- | --- | --- |
| policy options (2) | `raise_policy` C1, C2 · `exclude_policy` C3, C4 | - |
| input boundary violations (2) | `negative_label` C2 · `out_of_bounds_label` C2 | - |
| policy validation (1) | `unrecognized_policy` C5 | - |
| reliability diagram parity (2) | `diagram_raise` C6 · `diagram_exclude` C7 | - |

- No check claims more than the single case its proof exercises

## Swept

- validation : C1, C2, C5, C6
- failure modes : C2, C4, C5, C6
- idempotency : C1, C3
- authorization : n/a - internal module logic without user authentication
- concurrency : n/a - stateless mathematical transformations
- data lifecycle : n/a - ephemeral numpy calculations without storage
- dependency failure : n/a - pure local math and tensor checks
- state transitions : n/a - stateless metric evaluations
- observability : C2, C3

## Handoff

Files touched:
- `src/viral_cnn/eval/calibration.py` (22 KB)
- `tests/test_s16_calibration_label_validation.py` (8 KB)

Total estimate: ~30 KB · ~8k tokens.
Budget: 150k tokens. Estimate is well within budget: one builder.
