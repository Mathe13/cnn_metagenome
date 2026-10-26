# n03-short-finetune-training-notebooks checks

Profile: light
Plan: `.specs/features/n03-short-finetune-training-notebooks/plan.md`

14 checks in 2 slices · 2 one-way doors · 0 open, of which 0 block

## Checks

Grouped by the spec's slices; numbering runs across the whole feature.

### S1 - Noise-Aware Epoch Training, Validation, and Layer Freezing in ControlledTrainer · 3 files · 45 KB · ~15k

**C1** - `ControlledTrainer.train_epoch` with `config.noise_rate > 0.0` corrupts training input tokens using `inject_noise` at effective rate $3\epsilon/4$ before computing forward loss
Proof: `MPLCONFIGDIR=/tmp PYTHONPATH=src SMOKE_TEST=1 .venv/bin/python3 -m unittest -k test_c1_train_epoch_injects_noise tests/test_n03_finetune_training.py`

**C2** - `ControlledTrainer.evaluate_epoch` with `config.noise_rate > 0.0` corrupts validation input tokens using `inject_noise` at effective rate $3\epsilon/4$
Proof: `MPLCONFIGDIR=/tmp PYTHONPATH=src SMOKE_TEST=1 .venv/bin/python3 -m unittest -k test_c2_evaluate_epoch_injects_noise tests/test_n03_finetune_training.py`

**C3** - `ControlledTrainer` with `freeze_layers` parameter prefixes disables gradients (`requires_grad=False`) and locks matching BatchNorm modules to `eval()` mode during training
Proof: `MPLCONFIGDIR=/tmp PYTHONPATH=src SMOKE_TEST=1 .venv/bin/python3 -m unittest -k test_c3_freeze_layers_and_bn_eval tests/test_n03_finetune_training.py`

**C4** - `ControlledTrainer.train_epoch` returns metrics dictionary containing `train_loss`, `train_acc`, `train_f1`, `total_samples`, and `learning_rate`
Proof: `MPLCONFIGDIR=/tmp PYTHONPATH=src SMOKE_TEST=1 .venv/bin/python3 -m unittest -k test_c4_train_epoch_returns_metrics tests/test_n03_finetune_training.py`

**C5** - `ControlledTrainer.evaluate_epoch` returns metrics dictionary containing `val_loss`, `val_acc`, `val_f1_macro`, `val_f1_weighted`, and `total_samples`
Proof: `MPLCONFIGDIR=/tmp PYTHONPATH=src SMOKE_TEST=1 .venv/bin/python3 -m unittest -k test_c5_evaluate_epoch_returns_metrics tests/test_n03_finetune_training.py`

**C6** - `ControlledTrainer.save_epoch_checkpoint` serializes checkpoint file containing `epoch`, `model_state_dict`, `optimizer_state_dict`, `scheduler_state_dict`, `scaler_state_dict`, `metrics`, and `config`
Proof: `MPLCONFIGDIR=/tmp PYTHONPATH=src SMOKE_TEST=1 .venv/bin/python3 -m unittest -k test_c6_save_epoch_checkpoint_payload tests/test_n03_finetune_training.py`

**C7** - `ControlledTrainer.resume_checkpoint` restores weights, optimizer, scheduler, scaler states, and resumes from target epoch
Proof: `MPLCONFIGDIR=/tmp PYTHONPATH=src SMOKE_TEST=1 .venv/bin/python3 -m unittest -k test_c7_resume_checkpoint_restoration tests/test_n03_finetune_training.py`

**C8** - Historical directories `short/` and `long/` remain strictly unmodified (`git status --porcelain short/ long/` empty)
Proof: `MPLCONFIGDIR=/tmp PYTHONPATH=src SMOKE_TEST=1 .venv/bin/python3 -m unittest -k test_c8_historical_preservation tests/test_n03_finetune_training.py`

### S2 - Reproduction Notebooks for ResNet50 and ConvNeXt Species Fine-Tuning · 4 files · 60 KB · ~20k

**C9** - `03_finetune_resnet50_species.ipynb` contains valid JSON structure, zero imports of `short/utils/*`, and imports exclusively from `viral_cnn.*`
Proof: `MPLCONFIGDIR=/tmp PYTHONPATH=src SMOKE_TEST=1 .venv/bin/python3 -m unittest -k test_c9_notebook_03_imports_and_decoupling tests/test_n03_finetune_training.py`

**C10** - `03_finetune_resnet50_species.ipynb` executes end-to-end under `SMOKE_TEST=1` in under 60 seconds without exceptions
Proof: `MPLCONFIGDIR=/tmp PYTHONPATH=src SMOKE_TEST=1 .venv/bin/python3 -m unittest -k test_c10_notebook_03_execution_smoke_test tests/test_n03_finetune_training.py`

**C11** - `03_finetune_resnet50_species.ipynb` saves metrics CSV/JSON and `manifest.json` under `results_reproduction/n03_finetune/resnet50/`
Proof: `MPLCONFIGDIR=/tmp PYTHONPATH=src SMOKE_TEST=1 .venv/bin/python3 -m unittest -k test_c11_notebook_03_artifacts_generated tests/test_n03_finetune_training.py`

**C12** - `04_finetune_convnext_species.ipynb` contains valid JSON structure, zero imports of `short/utils/*`, and imports exclusively from `viral_cnn.*`
Proof: `MPLCONFIGDIR=/tmp PYTHONPATH=src SMOKE_TEST=1 .venv/bin/python3 -m unittest -k test_c12_notebook_04_imports_and_decoupling tests/test_n03_finetune_training.py`

**C13** - `04_finetune_convnext_species.ipynb` executes end-to-end under `SMOKE_TEST=1` in under 60 seconds without exceptions
Proof: `MPLCONFIGDIR=/tmp PYTHONPATH=src SMOKE_TEST=1 .venv/bin/python3 -m unittest -k test_c13_notebook_04_execution_smoke_test tests/test_n03_finetune_training.py`

**C14** - `04_finetune_convnext_species.ipynb` saves metrics CSV/JSON and `manifest.json` under `results_reproduction/n03_finetune/convnext/`
Proof: `MPLCONFIGDIR=/tmp PYTHONPATH=src SMOKE_TEST=1 .venv/bin/python3 -m unittest -k test_c14_notebook_04_artifacts_generated tests/test_n03_finetune_training.py`

## Coverage

| Set (size) | Member -> proof | Unproven |
| --- | --- | --- |
| Checkpoint dictionary serialized state keys (7) | `epoch` C6 · `model_state_dict` C6 · `optimizer_state_dict` C6 · `scheduler_state_dict` C6 · `scaler_state_dict` C6 · `metrics` C6 · `config` C6 | - |
| Training execution modes (2) | `historical_reproduction` C10, C13 · `controlled_parity` C10, C13 | - |
| Architecture reproduction notebooks (2) | `03_finetune_resnet50_species.ipynb` C9, C10, C11 · `04_finetune_convnext_species.ipynb` C12, C13, C14 | - |
| Noise application split stages (2) | training batch corruption C1 · validation batch corruption C2 | - |
| Layer freezing policy components (2) | parameter gradient disabling C3 · batch normalization eval mode retention C3 | - |
| Checkpoint selection strategies (2) | final training epoch C6 · best validation metric C6 | - |

- No check claims more than the single case its proof exercises

## Swept

Where each unwritten requirement landed. All nine, one line each, every time.

- validation: C3, C6 (invalid state dict or unrecognized freeze prefix raises ValueError)
- failure modes: C7 (corrupted resume checkpoint logs error or raises RuntimeError cleanly)
- idempotency: C8 (`git status --porcelain short/ long/` empty)
- authorization: n/a - local scientific code with no network boundary or access control
- concurrency: n/a - sequential notebook cells and PyTorch evaluation loops
- data lifecycle: C11 (`results_reproduction/n03_finetune/resnet50/`), C14 (`results_reproduction/n03_finetune/convnext/`)
- dependency failure: C10, C13 (smoke test runs entirely offline with synthetic data)
- state transitions: C7 (model training, evaluation, checkpointing, and resume)
- observability: C11 (`manifest.json` under resnet50), C14 (`manifest.json` under convnext)

## Handoff

S1 = 15k; S2 enters Notebooks at 35k total, under the 150k budget - one builder

