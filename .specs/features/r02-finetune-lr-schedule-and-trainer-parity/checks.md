# r02-finetune-lr-schedule-and-trainer-parity checks

Profile: light
Plan: `.specs/features/r02-finetune-lr-schedule-and-trainer-parity/plan.md`

## Checks

### S1 - Learning rate parity, scheduler warning suppression, and resume contract · 2 files · 20 KB · ~5k

**C1** - `ControlledTrainer` with `learning_rate=5e-5` and `mode="historical_reproduction"` (or `warmup_steps=0`) initializes `optimizer.param_groups[0]["lr"] == 5e-5` at Epoch 1 when coupled with `CosineAnnealingLR`
Proof: `.venv/bin/python3 -m unittest tests/test_n03_finetune_training.py -k test_r02_initial_learning_rate_parity`

**C2** - `ControlledTrainer.train_epoch` with `scheduler_step_unit="batch"` (and `"epoch"`) emits zero `UserWarning` when `GradScaler` encounters infinite/NaN gradients in the first step causing `optimizer.step()` to be skipped
Proof: `.venv/bin/python3 -m unittest tests/test_n03_finetune_training.py -k test_r02_scheduler_step_unit_and_gradscaler_warning_suppression`

**C3** - `ControlledTrainer.load_resume_checkpoint` restores model, optimizer, scaler, and scheduler states from a checkpoint saved at epoch $k$, resumes iteration from epoch $k + 1$, and preserves prior training history of epochs $1..k$
Proof: `.venv/bin/python3 -m unittest tests/test_n03_finetune_training.py -k test_r02_load_resume_checkpoint_preserves_history_and_next_epoch`

### S2 - Epoch 0 pre-training evaluation, resume automation, and history printing in notebooks 03 and 04 · 3 files · 35 KB · ~7k

**C4** - `notebooks/reproduction/short/03_finetune_resnet50_species.ipynb` and `04_finetune_convnext_species.ipynb` implement pre-training validation evaluation (Epoch 0) on clean ($\epsilon=0.0$) and noise ($\epsilon=0.05$) in Cell 3
Proof: `.venv/bin/python3 -m unittest tests/test_n03_finetune_training.py -k test_r02_notebooks_pretrain_evaluation_epoch0`

**C5** - `notebooks/reproduction/short/03_finetune_resnet50_species.ipynb` and `04_finetune_convnext_species.ipynb` include automated resume logic via `load_resume_checkpoint` in Cell 4 and print full `history_df` with Epoch 1 learning rate of $5\times 10^{-5}$ in Cell 5
Proof: `.venv/bin/python3 -m unittest tests/test_n03_finetune_training.py -k test_r02_notebooks_resume_logic_and_history_output`

## Coverage

| Set (size) | Member -> proof | Unproven |
| --- | --- | --- |
| Scheduler step units in ControlledTrainer.train_epoch (2) | `epoch` C2 · `batch` C2 | - |
| Fine-tuning reproduction notebooks updated for Epoch 0 and resume (2) | `03_finetune_resnet50_species.ipynb` C4, C5 · `04_finetune_convnext_species.ipynb` C4, C5 | - |
| Trainer state components restored by load_resume_checkpoint (6) | `model_state_dict` C3 · `optimizer_state_dict` C3 · `scheduler_state_dict` C3 · `scaler_state_dict` C3 · `start_epoch` C3 · `training_history` C3 | - |

- No other check claims more than the single case its proof exercises

## Swept

Where each unwritten requirement landed. All nine, one line each, every time.

- validation: C1, C2 (`ControlledTrainer` validates initial learning rate without warmup reduction and ensures safe stepping conditions under `GradScaler`)
- failure modes: C2 (prevents invalid PyTorch scheduler stepping before optimizer step when gradients contain non-finite values)
- idempotency: C3 (repeated loading and resume from a saved checkpoint produces identical restored parameter weights and scheduler counters)
- authorization: n/a - local and cluster offline training scripts without network authentication
- concurrency: n/a - single-process GPU/CPU model training loops
- data lifecycle: C3, C5 (manages checkpoint persistence, atomic copies, and training history JSON/CSV serialization)
- dependency failure: C4, C5 (handles missing or present resume checkpoints gracefully in notebooks)
- state transitions: C3 (correctly increments epoch index from $k$ to $k + 1$ upon resuming)
- observability: C4, C5 (prints pre-training validation metrics, live epoch progress, and full historical training dataframes)

## Handoff

- S1 = ~5k tokens, S2 = ~7k tokens, total ~12k tokens across 5 files (`src/viral_cnn/training/config.py`, `src/viral_cnn/training/trainer.py`, `notebooks/reproduction/short/03_finetune_resnet50_species.ipynb`, `notebooks/reproduction/short/04_finetune_convnext_species.ipynb`, `tests/test_n03_finetune_training.py`), well under the 150k budget - one builder
- **Boundary:** not yet closed
- **Settled mid-build:** none
- **Abandoned:** none

