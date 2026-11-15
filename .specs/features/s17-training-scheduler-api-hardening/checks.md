# S17 Training Scheduler API Hardening checks

Profile: light
Plan: `.specs/features/s17-training-scheduler-api-hardening/plan.md`

## Intent

Optimization interfaces historically permitted ambiguous scheduling: `train_step` applied an internal warmup/cosine schedule per step while `train_epoch` accepted an external scheduler, creating risks of dual conflicting schedules or active warmup under constant learning rate mode without schedule progression. This feature introduces `scheduler_mode: Literal["internal_step", "external_batch", "external_epoch", "constant"]`, enforces exclusive scheduler ownership fail-fast, and preserves historical reproduction invariants.

7 checks in 1 slice · 2 one-way doors · 0 open, of which 0 block

## Checks

### S1 - Training scheduler ownership enforcement · 3 files · ~25 KB · ~7k

**C1** - `ControlledExperimentConfig` stores valid `scheduler_mode` values (`"internal_step"`, `"external_batch"`, `"external_epoch"`, `"constant"`) without error (AC 1)
Proof: `REMOTE_SERVER=False PYTHONPATH=src .venv/bin/python3 -m unittest tests.test_s17_training_scheduler_api_hardening.TestS17TrainingSchedulerApiHardening.test_valid_scheduler_modes_accepted -k test_valid_scheduler_modes_accepted`

**C2** - `ControlledExperimentConfig` raises `ValueError` on unrecognized `scheduler_mode` (AC 2)
Proof: `REMOTE_SERVER=False PYTHONPATH=src .venv/bin/python3 -m unittest tests.test_s17_training_scheduler_api_hardening.TestS17TrainingSchedulerApiHardening.test_invalid_scheduler_mode_raises_value_error -k test_invalid_scheduler_mode_raises_value_error`

**C3** - `ControlledExperimentConfig` raises `ValueError` when `warmup_steps > 0` under `scheduler_mode == "constant"` in controlled parity mode (AC 3)
Proof: `REMOTE_SERVER=False PYTHONPATH=src .venv/bin/python3 -m unittest tests.test_s17_training_scheduler_api_hardening.TestS17TrainingSchedulerApiHardening.test_warmup_constant_conflict_raises_value_error -k test_warmup_constant_conflict_raises_value_error`

**C4** - `ControlledTrainer.train_epoch` raises `ValueError` on conflicting external scheduler when `scheduler_mode` is `"internal_step"` or `"constant"` (AC 4)
Proof: `REMOTE_SERVER=False PYTHONPATH=src .venv/bin/python3 -m unittest tests.test_s17_training_scheduler_api_hardening.TestS17TrainingSchedulerApiHardening.test_conflicting_external_scheduler_raises_value_error -k test_conflicting_external_scheduler_raises_value_error`

**C5** - `ControlledTrainer.train_epoch` raises `ValueError` on missing external scheduler when `scheduler_mode` is `"external_batch"` or `"external_epoch"` (AC 5)
Proof: `REMOTE_SERVER=False PYTHONPATH=src .venv/bin/python3 -m unittest tests.test_s17_training_scheduler_api_hardening.TestS17TrainingSchedulerApiHardening.test_missing_external_scheduler_raises_value_error -k test_missing_external_scheduler_raises_value_error`

**C6** - `ControlledTrainer.train_step` under `scheduler_mode == "internal_step"` dynamically updates learning rate according to warmup/cosine schedule (AC 6)
Proof: `REMOTE_SERVER=False PYTHONPATH=src .venv/bin/python3 -m unittest tests.test_s17_training_scheduler_api_hardening.TestS17TrainingSchedulerApiHardening.test_train_step_internal_step_schedules_learning_rate -k test_train_step_internal_step_schedules_learning_rate`

**C7** - `ControlledTrainer.train_step` under `scheduler_mode == "constant"` preserves initial learning rate across consecutive steps (AC 7)
Proof: `REMOTE_SERVER=False PYTHONPATH=src .venv/bin/python3 -m unittest tests.test_s17_training_scheduler_api_hardening.TestS17TrainingSchedulerApiHardening.test_train_step_constant_mode_preserves_learning_rate -k test_train_step_constant_mode_preserves_learning_rate`

## Coverage

| Set (size) | Member -> proof | Unproven |
| --- | --- | --- |
| scheduler modes (4) | `internal_step` C1, C4, C6 · `external_batch` C1, C5 · `external_epoch` C1, C5 · `constant` C1, C3, C4, C7 | - |
| validation errors (3) | `unrecognized_mode` C2 · `warmup_constant_conflict` C3 · `missing_external` C5 | - |
| conflict detection (1) | `conflicting_external` C4 | - |
| LR stepping dynamics (2) | `dynamic_step_schedule` C6 · `constant_rate_preservation` C7 | - |

- No check claims more than the single case its proof exercises

## Swept

- validation : C1, C2, C3
- failure modes : C2, C3, C4, C5
- idempotency : C1, C7
- authorization : n/a - local optimization and training loop scheduling
- concurrency : n/a - sequential step and epoch optimization
- data lifecycle : n/a - ephemeral in-memory optimizer parameter updates
- dependency failure : n/a - pure PyTorch optimizer routines
- state transitions : C4, C5, C6, C7
- observability : C6, C7

## Handoff

Files touched:
- `src/viral_cnn/training/config.py` (7 KB)
- `src/viral_cnn/training/trainer.py` (22 KB)
- `tests/test_s17_training_scheduler_api_hardening.py` (8 KB)

Total estimate: ~37 KB · ~9k tokens.
Budget: 150k tokens. Estimate is well within budget: one builder.
