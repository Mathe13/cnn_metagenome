# S12: Controlled Cross-Architecture Training Experiment - Checks

Profile: light
Plan: `.specs/features/s12-controlled-architecture-experiment/plan.md`

## Checks

### S1: Controlled Training Configuration and Parity Validator · 3 files · 28 KB · ~7k

**C1** - `viral_cnn.training.config.ControlledExperimentConfig` validates and enforces parameter parity across compared architectures (AC 1)
Proof: `.venv/bin/python3 -m unittest discover -s tests -k test_controlled_experiment_config_enforces_parameter_parity`

**C2** - `viral_cnn.training.config.ControlledExperimentConfig` raises ValueError when parameter mismatch (e.g. mismatched step budget, differing label smoothing) is detected between architectures (AC 2)
Proof: `.venv/bin/python3 -m unittest discover -s tests -k test_controlled_experiment_config_rejects_mismatched_parameters`

**C3** - `viral_cnn.training.config.seed_everything` deterministically seeds Python random, NumPy, PyTorch CPU, and PyTorch CUDA generators (AC 3)
Proof: `.venv/bin/python3 -m unittest discover -s tests -k test_seed_everything_deterministic_reproducibility`

**C4** - `viral_cnn.training.config.load_controlled_config` parses declarative YAML (`configs/controlled/baseline_pair.yaml`) into validated config without absolute/personal paths (AC 4)
Proof: `.venv/bin/python3 -m unittest discover -s tests -k test_load_controlled_config_from_yaml`

### S2: Unified Training Loop and Resource Profiling Engine · 3 files · 45 KB · ~11k

**C5** - `viral_cnn.training.trainer.ControlledTrainer` supports PyTorch AMP mixed precision, gradient clipping, and step execution (AC 5)
Proof: `.venv/bin/python3 -m unittest discover -s tests -k test_controlled_trainer_amp_and_grad_clip`

**C6** - `viral_cnn.training.trainer.ControlledTrainer` updates weights with AdamW and schedules learning rates with linear warmup followed by CosineAnnealingLR (AC 6)
Proof: `.venv/bin/python3 -m unittest discover -s tests -k test_controlled_trainer_adamw_cosine_warmup`

**C7** - `viral_cnn.training.trainer.ControlledTrainer` computes classification loss via CrossEntropyLoss with exact label smoothing = 0.1 (AC 7)
Proof: `.venv/bin/python3 -m unittest discover -s tests -k test_controlled_trainer_label_smoothing`

**C8** - `viral_cnn.training.profiler.ResourceProfiler` measures peak VRAM, step execution latency in ms, and sequence throughput in sequences/second (AC 8)
Proof: `.venv/bin/python3 -m unittest discover -s tests -k test_resource_profiler_telemetry_metrics`

**C9** - `viral_cnn.training.trainer.ControlledTrainer.evaluate` computes Top-1 accuracy, Top-5 accuracy, and Macro-F1 across validation batches and saves the optimal checkpoint based on `checkpoint_metric` (AC 9)
Proof: `.venv/bin/python3 -m unittest discover -s tests -k test_controlled_trainer_evaluation_and_checkpoint_selection`

### S3: Controlled Training CLI Runner and Pairwise Execution · 3 files · 32 KB · ~8k

**C10** - CLI `scripts/run_controlled_training.py` accepts `--config`, `--steps`, `--seeds`, `--device`, `--dry-run`, and `--output-dir` arguments (AC 10)
Proof: `.venv/bin/python3 -m unittest discover -s tests -k test_run_controlled_training_cli_arguments`

**C11** - CLI `scripts/run_controlled_training.py --dry-run` validates config, instantiates both ResNet50 and ConvNeXt models, logs parameter parity, and exits with status 0 (AC 11)
Proof: `.venv/bin/python3 -m unittest discover -s tests -k test_run_controlled_training_dry_run_exit_zero`

**C12** - CLI `scripts/run_controlled_training.py` executes sequential training across multiple seeds and saves structured JSON logs and checkpoints to `--output-dir` (AC 12)
Proof: `.venv/bin/python3 -m unittest discover -s tests -k test_run_controlled_training_multi_seed_execution`

**C13** - CLI `scripts/run_controlled_training.py` exits with status 2 when an invalid architecture or conflicting option is supplied (AC 13)
Proof: `.venv/bin/python3 -m unittest discover -s tests -k test_run_controlled_training_cli_invalid_options_exit_two`

### S4: Controlled Architecture Audit and Causal Demarcation Report · 3 files · 40 KB · ~10k

**C14** - `viral_cnn.reporting.controlled_audit.generate_controlled_experiment_report` writes `docs/controlled_architecture_experiment.md` with pre-defined budget, parity matrix, and GPU cost estimates (AC 14)
Proof: `.venv/bin/python3 -m unittest discover -s tests -k test_generate_controlled_experiment_report_budget_and_gpu_cost`

**C15** - `docs/controlled_architecture_experiment.md` contains side-by-side comparative metrics (clean accuracy, noise accuracy $\epsilon=0.05$, Macro-F1, peak memory, throughput) for ResNet50 and ConvNeXt (AC 15)
Proof: `.venv/bin/python3 -m unittest discover -s tests -k test_controlled_experiment_report_side_by_side_metrics`

**C16** - `docs/controlled_architecture_experiment.md` contains the mandatory scientific causal demarcation text restricting causal claims to evaluated configurations (AC 16)
Proof: `.venv/bin/python3 -m unittest discover -s tests -k test_controlled_experiment_report_causal_demarcation`

**C17** - `viral_cnn.reporting.controlled_audit.verify_controlled_report_consistency` asserts that all report tables match executed run artifacts without missing values (AC 17)
Proof: `.venv/bin/python3 -m unittest discover -s tests -k test_verify_controlled_report_consistency`

## Coverage

| Set (size) | Member -> proof | Unproven |
| --- | --- | --- |
| Architectures in parity comparison (2) | `resnet50_1d` C1, C11 · `convnext_tiny_1d` C1, C11 | - |
| Controlled experiment execution modes (2) | `dry_run` C11 · `benchmark_run` C12 | - |
| Checkpoint selection criteria supported (3) | `best_val_loss` C9 · `best_val_macro_f1` C9 · `last_step` C9 | - |
| Resource profiling metrics recorded (4) | `peak_vram_mb` C8 · `step_time_ms` C8 · `throughput_seq_per_sec` C8 · `total_flops` C8 | - |
| CLI status codes (2) | `0` C11 · `2` C13 | - |
| Noise rates evaluated in controlled audit (2) | `epsilon_0.0` C15 · `epsilon_0.05` C15 | - |

- Claims naming a status code, route or response shape: C11, C13 - each has a proof that crosses the boundary
- No other check claims more than the single case its proof exercises

## Swept

Where each unwritten requirement landed. All nine, one line each, every time.

- validation and bounds: C2
- failure and partial failure: C13
- idempotency, retry, duplicates: C3
- authorization and rate limits: n/a - local scientific training and audit CLI
- concurrency and ordering: C12
- data lifecycle: n/a - training logs and checkpoints written to isolated timestamped run directories
- external-dependency failure: C11
- state transitions: C6, C9
- observability: C8, C14

## Handoff

S1 = 7k; S2 = 11k; S3 = 8k; S4 = 10k -> total estimate = 36k tokens, well under the 150k budget - one builder

- **Mechanism:** one builder
