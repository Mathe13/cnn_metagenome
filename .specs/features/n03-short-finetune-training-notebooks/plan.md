# n03-short-finetune-training-notebooks

## Problem

In the original study, fine-tuning 1D CNNs under artificial noise corruption ($\epsilon=0.05$, Notas 21–23) was performed inside monolithic Jupyter notebooks (`short/finetune_resnet50_species.ipynb` and `short/finetune_convnext_species.ipynb`) via an inline loop (`train_noise_aware`). This implementation suffered from several architectural deficiencies:

1. Coupling to legacy paths and in-place checkpoint overwrites (`./checkpoint_finetune_resnet/finetune_noise_last.pt` and `./checkpoint_transfer_species_conv/finetune_noise_last.pt`), without unified path resolution or protection against clobbering historical weights.
2. Inconsistent checkpoint selection rules between architectures: ResNet50 fine-tuning ran 15 epochs and evaluated the final epoch 14 (`finetune_noise_14.pt`), whereas ConvNeXt selected the best validation checkpoint at epoch 7 (`finetune_noise_7.pt`), without explicit programmatic declarations of this selection policy.
3. The modern modular framework `viral_cnn.training.trainer:ControlledTrainer` currently only supports step-based training without noise corruption applied to dataloader batches (`noise_rate=0.0`), lacks epoch-level training and validation loops with weighted F1 and accuracy accumulation, lacks optional backbone layer freezing with BatchNorm `eval` preservation, and lacks resume capabilities matching historical checkpoint structures.
4. There are no decoupled reproduction notebooks in `notebooks/reproduction/short/` for fine-tuning ResNet50 and ConvNeXt on species classification.

When this feature ships, researchers will be able to execute fine-tuning for both ResNet50 and ConvNeXt on species classification using modular `viral_cnn.training` pipelines, with support for both historical reproduction hyperparameter modes and controlled parity modes, transparent remote JupyterHub / local smoke-test execution, and deterministic checkpoint/manifest tracking without modifying historical code or weights.

## Flow

This feature reuses existing models from `viral_cnn.models.registry`, noise corruption from `viral_cnn.eval.noise`, metrics from `viral_cnn.eval.metrics`, and notebook environment management from `viral_cnn.notebook_runtime`.

1. Notebook execution -> `viral_cnn.notebook_runtime:setup_notebook_environment` (exists) - resolves `DATA_ROOT`, sets `seed=42`, checks device, and configures output directories under `results_reproduction/`.
2. Base model loading -> `viral_cnn.models.registry:build_registered_model` (exists) - loads clean ResNet50 species checkpoint or transfer ConvNeXt species checkpoint (`AD-008`).
3. Optimization engine -> `viral_cnn.training.trainer:ControlledTrainer` (exists, extended) - initializes AdamW, CosineAnnealingLR, AMP GradScaler, label smoothing cross-entropy, optional layer freezing (`freeze_backbone`), and epoch-level tracking.
4. Batch iteration -> `viral_cnn.eval.noise:inject_noise` (exists) - applies artificial base substitution ($\epsilon=0.05$) to training and validation batches.
5. Checkpoint & manifest persistence -> `ControlledTrainer.save_epoch_checkpoint` (new, door 1) - serializes `model_state_dict`, `optimizer_state_dict`, `scheduler_state_dict`, `scaler_state_dict`, and epoch metrics to versioned files in `results_reproduction/checkpoints/`.
6. Telemetry & manifest -> `viral_cnn.notebook_runtime:NotebookRunManifest` (exists) - records runtime configuration, execution duration, and metric trajectories to `manifest.json`.

## Impact

| Front | What changes |
| --- | --- |
| domain | new term: `TrainerEpochResult` - typed dataclass holding train/val losses, accuracies, macro/weighted F1s, and sample counts per epoch, lives in `viral_cnn.training.trainer` |
| domain | existing term: `noise_rate` in `ControlledExperimentConfig` - previously unused during dataloader batch passes; now actively triggers `inject_noise` with rate $\epsilon$ during both training and validation epoch iterations |
| stored data | nothing to migrate - all historical checkpoints remain untouched; new fine-tuned checkpoints and manifests are saved exclusively to versioned output directories (`results_reproduction/checkpoints/` and `results_reproduction/n03_finetune/`) |

## Relations

None - no stored-data shape change

## Surface

None - nothing consumed outside (internal Python training engine and reproduction notebooks only)

## Landing

| One-way door | Literal shape | Alternative rejected |
| --- | --- | --- |
| Checkpoint dictionary serialization format | `{'epoch': int, 'model_state_dict': dict, 'optimizer_state_dict': dict, 'scheduler_state_dict': dict, 'scaler_state_dict': dict, 'metrics': dict, 'config': dict}` | Saving only `model_state_dict`; rejected because resuming interrupted fine-tuning runs requires optimizer, scheduler, and scaler states, and historical code relied on the full dictionary. |
| Non-destructive checkpoint output destination | `save_dir = Path("results_reproduction/checkpoints") / experiment_name` | Saving to historical directories `./checkpoint_finetune_resnet` or `./checkpoint_transfer_species_conv`; rejected because modifying or overwriting existing filesystem state violates AGENTS.md invariant. |

- Nothing else in this change is hard to reverse

## Criteria

Grouped by slice - one observable outcome each, never a layer.

### S1: Noise-Aware Epoch Training, Validation, and Layer Freezing in `ControlledTrainer` (P1)

Extends `ControlledTrainer` and `ControlledExperimentConfig` with epoch-based execution, noise corruption on batches, layer freezing, and checkpoint saving/resuming.

**Acceptance Criteria**

1. WHEN `ControlledTrainer.train_epoch` executes with `config.noise_rate > 0.0` THEN the system SHALL inject noise into each input batch using `inject_noise` with rate `config.noise_rate` before computing forward loss.
2. WHEN `ControlledTrainer.evaluate_epoch` executes with `config.noise_rate > 0.0` THEN the system SHALL inject noise into each validation batch using `inject_noise` with rate `config.noise_rate`.
3. WHEN `ControlledTrainer` is initialized with `freeze_layers` containing parameter prefixes THEN the system SHALL set `requires_grad=False` for all matching parameters and set all corresponding BatchNorm modules to `eval()` mode.
4. WHEN `ControlledTrainer.train_epoch` finishes an epoch THEN the system SHALL return a dictionary containing `train_loss`, `train_acc`, `train_f1`, `total_samples`, and `learning_rate`.
5. WHEN `ControlledTrainer.evaluate_epoch` finishes an epoch THEN the system SHALL return a dictionary containing `val_loss`, `val_acc`, `val_f1_macro`, `val_f1_weighted`, and `total_samples`.
6. WHEN `ControlledTrainer.save_epoch_checkpoint` is called THEN the system SHALL persist a checkpoint file containing `epoch`, `model_state_dict`, `optimizer_state_dict`, `scheduler_state_dict`, `scaler_state_dict`, `metrics`, and `config`.
7. WHEN `ControlledTrainer.resume_checkpoint` is invoked with a valid checkpoint dictionary THEN the system SHALL restore model weights, optimizer state, scheduler state, scaler state, and start epoch.
8. The system SHALL keep all historical checkpoints in `cnn_kraken/`, `cnn_kraken_species/`, and `cnn_kraken_genus/` strictly unmodified.

**Independent test:** Run unit tests instantiating synthetic models, performing 2 epochs of training with `noise_rate=0.05`, verifying noise injection, verifying frozen layers do not update weights, verifying checkpoint saving and restoring, and asserting zero modified files in historical directories.

### S2: Reproduction Notebooks for ResNet50 and ConvNeXt Species Fine-Tuning (P2)

Creates `notebooks/reproduction/short/03_finetune_resnet50_species.ipynb` and `notebooks/reproduction/short/04_finetune_convnext_species.ipynb` with smoke-test and remote execution support.

**Acceptance Criteria**

9. WHEN `03_finetune_resnet50_species.ipynb` executes under `SMOKE_TEST=1` THEN the system SHALL complete without errors, train ResNet50 for dry-run batches under `noise_rate=0.05`, and save training metrics and manifest to `results_reproduction/n03_finetune/resnet50/`.
10. WHEN `04_finetune_convnext_species.ipynb` executes under `SMOKE_TEST=1` THEN the system SHALL complete without errors, train ConvNeXt for dry-run batches under `noise_rate=0.05`, and save training metrics and manifest to `results_reproduction/n03_finetune/convnext/`.
11. The system SHALL ensure that both reproduction notebooks contain zero imports of `short/utils/*` and import all training, model, and metric logic exclusively from `viral_cnn.*`.
12. WHILE `SMOKE_TEST=1` is active THEN each reproduction notebook SHALL complete its execution in under 60 seconds on CPU.

**Independent test:** Execute both notebooks sequentially with `nbclient` or cell iteration runner under `SMOKE_TEST=1`, asserting successful exit code, presence of output metric files, and absence of legacy imports.

## Out of scope

Product capabilities only. Process and harness rules live in AGENTS.md or as Observable `n/a`.

| Excluded | Why |
| --- | --- |
| Full 15-epoch training run on 13,683 classes on CPU | Full training of 15 epochs across 1.7M sequences requires multi-day GPU cluster compute; local testing is verified via synthetic smoke tests and contract tests per AD-005 and AD-006. |
| Modifications to historical notebooks in `short/` | Historical notebooks must be preserved verbatim for provenance per AD-004. |
| In-place overwriting of original checkpoint files | Historical checkpoints must remain preserved for baseline evaluation per AGENTS.md. |

## Assumptions

Defaults that are not already a numbered criterion. Drop a row once it is.

| Assumption | Chosen default | Rationale | Confirmed? |
| --- | --- | --- | --- |
| Historical optimizer parameter parity | AdamW with $lr=5\times 10^{-5}$, weight decay $1\times 10^{-4}$, CosineAnnealingLR with $\eta_{\text{min}}=10^{-7}$, label smoothing 0.1 | Directly extracted from historical cells 8–9 of `short/finetune_resnet50_species.ipynb` and `short/finetune_convnext_species.ipynb`. | y |
| Checkpoint selection policy documentation | ResNet50 uses final epoch (epoch 14) and ConvNeXt uses best validation macro-F1 (epoch 7) | Matches published results in Tables 1, 3, 4 and notes 21–23. | y |

**Open questions:** none - all resolved or logged above.

## Observable

Worksheet, not the review. `n/a` needs its reason. One row may group the same decision across several routes.

| Surface | Decision | Landing |
| --- | --- | --- |
| all notebooks `03_finetune_*.ipynb` and `04_finetune_*.ipynb` | execution under SMOKE_TEST=1 without remote GPU cluster | AC 9, AC 10, AC 12 |
| all notebooks `03_finetune_*.ipynb` and `04_finetune_*.ipynb` | error state on missing dataset partitions or missing base weights | existing - handled via `viral_cnn.notebook_runtime:setup_notebook_environment` |
| all notebooks `03_finetune_*.ipynb` and `04_finetune_*.ipynb` | non-destructive checkpoint persistence | AC 6, AC 8 |
| training CLI / scripts | exit codes and logging | n/a - notebook and module runtime only |
| HTTP APIs / webhooks | response/error status | n/a - no network HTTP endpoints |

## Sources

- `short/finetune_resnet50_species.ipynb` - cells 8–11: historical noise-aware fine-tuning loop for ResNet50.
- `short/finetune_convnext_species.ipynb` - cells 8–11: historical noise-aware fine-tuning loop for ConvNeXt.
- `ROADMAP_REPRODUCAO_NOTEBOOKS.md` - section N03 specifying scope, inputs, outputs, and invariants.
- `AGENTS.md` - scientific invariants for noise fine-tuning ($\epsilon=0.05$, preservation of historical notebooks and checkpoints).

