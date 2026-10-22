# S12: Controlled Cross-Architecture Training Experiment

## Problem

The empirical comparison between 1D ResNet50 and ConvNeXt architectures in the published viral metagenomics study relies on models evaluated under disparate, uncontrolled training regimens, introducing confounding factors into the causal claims of the paper:

1. **Confounded Training Configurations**: ResNet50 clean was trained from scratch, while ResNet50 noise-adapted was fine-tuned for 15 epochs under noise taking the final epoch 14 checkpoint with dropout 0.5. Conversely, ConvNeXt clean was evaluated either from scratch (`ConvNeXt_Z`, achieving 59.18%) or via Genus-to-Species transfer learning (`ConvNeXt_T`, achieving 60.83%), and ConvNeXt noise-adapted (`ConvNeXt_F`, achieving 61.14% clean and 17.38% at $\epsilon=0.15$) was fine-tuned from `ConvNeXt_T` with dropout 0.7 taking the best validation checkpoint at epoch 7. These disparate protocols confound architecture (depthwise-separable 7x1 kernels vs standard 3x1 convolutions, LayerNorm vs BatchNorm) with dropout rates (0.7 vs 0.5), transfer learning lineages, and checkpoint selection policies (best-epoch vs last-epoch).
2. **Absence of Unified Training Pipeline**: The codebase currently lacks a unified, reproducible training and validation engine (`viral_cnn.training`) capable of executing both ResNet50 and ConvNeXt under strictly matched optimization conditions: identical step budgets, identical learning rate schedules (AdamW with CosineAnnealingLR and linear warmup), identical label smoothing (0.1), identical data partitions, identical noise rates, and identical checkpoint selection criteria.
3. **Missing GPU Resource Profiling and Telemetry**: The historical training notebooks record no systematic GPU resource profiling (peak VRAM allocation, batch latency, throughput in sequences per second, estimated training FLOPs, wall-clock time per step) to quantify the computational cost of training or retraining high-cardinality 1D convolutional models.
4. **Unsubstantiated Architectural Causal Claims**: Without an isolated, controlled experiment holding training budgets and hyperparameters constant across architectures and random seeds, assertions that ConvNeXt's architecture intrinsically provides superior noise resilience or that ResNet50 intrinsically achieves higher capacity cannot be scientifically substantiated. In the absence of an executed large-scale cluster retrain, all causal claims must be strictly bounded to the evaluated configurations.

When this ships, the repository provides `src/viral_cnn/training/` with a modular controlled training engine and telemetry tracker; `scripts/run_controlled_training.py` providing a reproducible CLI to run pairwise controlled experiments with matched step budgets, seeds, and hyperparameters; `docs/controlled_architecture_experiment.md` delivering an exhaustive comparative audit with budget definitions, GPU cost models, and statistical comparison across seeds; and strict scientific guardrails that demarcate causal architecture claims from training regimen artifacts.

## Flow

Reuses `src/viral_cnn/models/builder.py` (exists) and `src/viral_cnn/models/registry.py` (exists) for architecture instantiation, `src/viral_cnn/data/encoding.py` (exists) for spatial nucleotide one-hot encoding, and `src/viral_cnn/eval/metrics.py` (exists) for macro-F1, top-1, and top-5 metric computations.

1. controlled training configuration YAML/CLI -> `viral_cnn.training.config:ControlledExperimentConfig` (new, door 1) - validates parameter parity, budgets, and random seeds
2. training and validation dataset iterators -> `viral_cnn.training.trainer:ControlledTrainer` (new, door 2) - executes matched optimization loops with AdamW, CosineAnnealingLR, label smoothing, AMP, and step telemetry
3. step execution metrics and GPU memory allocations -> `viral_cnn.training.profiler:ResourceProfiler` (new, door 3) - captures peak VRAM, step latency, throughput (sequences/s), and estimated compute costs
4. trained checkpoints and evaluation metrics across seeds -> `viral_cnn.reporting.controlled_audit:generate_controlled_experiment_report` (new, door 4) - compiles comparative tables and causal demarcation analysis
5. out: `docs/controlled_architecture_experiment.md` and structured experiment run logs (`controlled_runs/*.json`)

## Impact

| Front | What changes |
| --- | --- |
| domain | new term: `ControlledTrainingPair` - a pair of training runs (ResNet50 vs ConvNeXt) executed under strictly identical budgets, data, optimizer, scheduler, loss, and seeds, lives in `viral_cnn.training.config` |
| domain | new term: `ComputeResourceAudit` - systematic profiling of GPU VRAM, batch throughput (sequences/s), FLOPs, and wall-clock costs, lives in `viral_cnn.training.profiler` |
| domain | existing term: `ArchitecturalSuperiority` previously attributed noise resilience exclusively to ConvNeXt's architecture; now qualified as `RegimenConfoundedComparison` documenting that historical differences were influenced by transfer learning, dropout, and checkpoint selection |
| stored data | nothing to migrate: historical checkpoints and evaluation parquets remain intact; controlled training artifacts land in `configs/controlled/`, `runs/controlled/`, and `docs/controlled_architecture_experiment.md` |

## Relations

None - no stored-data shape change

## Surface

None - nothing consumed outside

## Landing

| One-way door | Literal shape | Alternative rejected |
| --- | --- | --- |
| Controlled experiment configuration schema | `ControlledExperimentConfig(architectures=("resnet50_1d", "convnext_tiny_1d"), max_steps=1000, batch_size=64, learning_rate=5e-5, weight_decay=1e-4, warmup_steps=100, label_smoothing=0.1, dropout_rate=0.5, noise_rate=0.0, seed=42, checkpoint_metric="val_macro_f1")` | ad-hoc script arguments without structured schema validation - rejected because uncontrolled parameters silently introduce confounding variables |
| Unified resource profiling schema | `ResourceProfile(device_name=str, peak_vram_mb=float, mean_step_time_ms=float, throughput_seq_per_sec=float, total_flops=float, wall_clock_seconds=float)` | unstructured console printing without persistent telemetry logs - rejected because cost reporting requires standardized JSON/CSV output for publication |
| Checkpoint selection policy unification | `checkpoint_selection_rule: Literal["best_val_loss", "best_val_macro_f1", "last_step"] = "best_val_macro_f1"` | differing selection criteria between models (e.g. best epoch for ConvNeXt vs last epoch for ResNet) - rejected because differing checkpoint rules confound performance comparisons |
| Causal claim demarcation guardrail | Scientific mandate in `docs/controlled_architecture_experiment.md`: "In the absence of multi-seed full-scale retraining across the entire closed-set taxonomy, causal conclusions are strictly confined to the evaluated training configurations and cannot be attributed exclusively to architecture." | claiming architecture-intrinsic superiority without controlled full retrains - rejected because scientific invariants in AGENTS.md forbid conflating configuration differences with intrinsic architecture properties |

- Nothing else in this change is hard to reverse

## Criteria

### S1: Controlled Training Configuration and Parity Validator (P1)

Declarative configuration specification and validation enforcing strict parameter parity between evaluated architectures.

**Acceptance Criteria**

1. The system SHALL provide `ControlledExperimentConfig` enforcing parameter parity (batch size, optimizer, learning rate, weight decay, label smoothing, dropout rate, noise rate, and step budget) between compared architectures.
2. WHEN `ControlledExperimentConfig` is instantiated with mismatched budgets or differing loss definitions between paired architectures THEN the system SHALL raise a `ValueError` identifying the unisolated parameter.
3. The system SHALL support configuring random seeds (e.g., 42, 43, 44) and applying them deterministically to Python random, NumPy, PyTorch CPU, and PyTorch CUDA generators.
4. WHEN loading configuration from YAML (`configs/controlled/baseline_pair.yaml`) THEN the system SHALL instantiate validated configuration objects without personal or absolute filesystem paths.

**Independent test:** unit tests in `tests/test_controlled_training.py` asserting configuration validation, mismatch rejection, and deterministic seed propagation.

### S2: Unified Training Loop and Resource Profiling Engine (P1)

Modular training loop supporting mixed precision, learning rate warmup, cosine decay, checkpointing, and GPU/CPU resource profiling.

**Acceptance Criteria**

1. The system SHALL implement `ControlledTrainer` supporting PyTorch automatic mixed precision (AMP), gradient clipping, and step-based logging.
2. WHEN executing training steps THEN the system SHALL update model weights using `torch.optim.AdamW` and schedule learning rates via linear warmup followed by `CosineAnnealingLR`.
3. WHEN computing classification loss THEN the system SHALL compute `CrossEntropyLoss` with the configured `label_smoothing` (default 0.1).
4. The system SHALL track resource metrics per run via `ResourceProfiler`, recording peak memory allocation in megabytes, mean step execution time in milliseconds, and throughput in sequences per second.
5. WHEN validation is triggered THEN `ControlledTrainer` SHALL compute Top-1 accuracy, Top-5 accuracy, and Macro-F1 across validation batches and update the optimal checkpoint according to the unified `checkpoint_metric`.

**Independent test:** unit tests in `tests/test_controlled_training.py` validating forward-backward steps, loss decrease on synthetic batches, and resource telemetry logging.

### S3: Controlled Training CLI Runner and Pairwise Execution (P1)

Command-line interface to execute pairwise controlled runs on synthetic benchmarks or real dataset partitions across seeds.

**Acceptance Criteria**

1. The system SHALL provide CLI script `scripts/run_controlled_training.py` accepting `--config`, `--steps`, `--seeds`, `--device`, `--dry-run`, and `--output-dir`.
2. WHEN invoking `scripts/run_controlled_training.py --dry-run` THEN the system SHALL validate the configuration, initialize both `resnet50_1d` and `convnext_tiny_1d` architectures, log parameter parity, and exit with status code 0.
3. WHEN executing a controlled benchmark run with multiple seeds THEN the system SHALL run both architectures sequentially under each seed, saving checkpoints and step logs to the specified output directory.
4. IF an invalid architecture name or conflicting parameter is passed via CLI THEN the system SHALL output a descriptive error message and exit with status code 2.

**Independent test:** subprocess tests executing `python3 scripts/run_controlled_training.py --dry-run` and synthetic micro-runs verifying exit codes and generated JSON artifacts.

### S4: Controlled Architecture Audit and Causal Demarcation Report (P1)

Automated generation of the controlled architecture experiment report documenting experimental budgets, resource profiling, and causal claim guardrails.

**Acceptance Criteria**

1. The system SHALL generate `docs/controlled_architecture_experiment.md` documenting the pre-defined budget, hyperparameter parity matrix, and GPU cost estimates for full retraining.
2. WHEN compiling comparative metrics THEN the report SHALL display clean accuracy, noise accuracy ($\epsilon=0.05$), Macro-F1, peak memory, and throughput (seq/s) side-by-side for ResNet50 and ConvNeXt.
3. The system SHALL include an explicit causal demarcation section in `docs/controlled_architecture_experiment.md` stating that in the absence of full-scale multi-seed cluster retraining, comparative differences cannot be asserted as intrinsic architectural superiority.
4. The system SHALL provide an automated verification function asserting that report tables match generated run logs without missing entries.

**Independent test:** unit tests in `tests/test_controlled_training.py` asserting report structure, table consistency, and presence of causal demarcation text.

## Out of scope

Product capabilities only. Process and harness rules live in AGENTS.md or as Observable `n/a`.

| Excluded | Why |
| --- | --- |
| Full-scale multi-week retraining of 13,683 classes across multiple seeds on local machine | Hardware constraint: local environment lacks multi-GPU cluster capacity; large-scale retraining belongs to remote JupyterHub cluster storage (AD-005, AD-006) |
| Architecture ablation of normalization layers (BatchNorm vs LayerNorm) | Distinct scientific experiment: isolated normalization ablation requires separate architectural variants beyond standard ResNet50 and ConvNeXt-Tiny |
| Kernel size ablation (3x1 vs 7x1 depthwise kernels) | Distinct scientific experiment: kernel size modifications alter the published ConvNeXt and ResNet definitions |

## Assumptions

Defaults that are not already a numbered criterion. Drop a row once it is.

| Assumption | Chosen default | Rationale | Confirmed? |
| --- | --- | --- | --- |
| Default step budget for micro-benchmark | 200 optimization steps with validation every 50 steps | Provides sufficient steps to verify optimizer convergence, LR scheduling, and memory stability without exceeding CPU execution time limits | y |
| Default dropout rate in parity experiments | 0.5 for both ResNet50 and ConvNeXt | Eliminates the historical confounder (0.5 in ResNet50 vs 0.7 in ConvNeXt) while maintaining adequate regularization for high-cardinality targets | y |
| Synthetic benchmark batch size | 32 sequences of 512 bp with 4 channels | Matches input window size and channel dimensions while enabling execution in standard environments | y |

**Open questions:** none - all resolved or logged above.

## Observable

Worksheet, not the review. `n/a` needs its reason. One row may group the same decision across several routes.

| Surface | Decision | Landing |
| --- | --- | --- |
| CLI `scripts/run_controlled_training.py` | exit codes, verbosity, and output format | AC 10, AC 11, AC 12, AC 13 |
| CLI `scripts/run_controlled_training.py` | error handling for invalid architectures | AC 13 |
| Report `docs/controlled_architecture_experiment.md` | document structure and causal demarcation | AC 14, AC 15, AC 16, AC 17 |
| CLI `scripts/run_controlled_training.py` | destructive overwrite protection | existing - output directory saves runs by timestamp and seed without overwriting historical checkpoints |
| all internal training APIs | authentication and rate limits | n/a - local scientific training code with no network boundary |

## Sources

- `ROADMAP_SPECS_ARTIGO.md` (Specification S12) - defines objective, scope, budget, and acceptance criteria for controlled architecture experiment
- `AGENTS.md` - defines scientific invariants for noise, checkpoint selection, and causal attribution
- `docs/model_registry.md` - documents historical training hyperparameter discrepancies and gaps between ResNet50 and ConvNeXt

