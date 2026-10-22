# S12: Controlled Cross-Architecture Training Experiment verification

**Verdict**: PASS
**Profile**: light
**Diff range**: 4c4e58b..HEAD
**Round**: 1 - full
**Verifier**: independent sub-agent (author != verifier)

## Checks

| Check | Claim | Proof run | Evidence | Result |
| --- | --- | --- | --- | --- |
| C1 | ControlledExperimentConfig validates and enforces parameter parity across compared architectures | `PYTHONPATH=src .venv/bin/python3 -m unittest discover -s tests -k test_controlled_experiment_config_enforces_parameter_parity` exit 0 | `tests/test_controlled_training.py:51` - `self.assertEqual(len(cfg.architectures), 2)` | PASS |
| C2 | ControlledExperimentConfig raises ValueError when parameter mismatch or invalid parameter is detected | `PYTHONPATH=src .venv/bin/python3 -m unittest discover -s tests -k test_controlled_experiment_config_rejects_mismatched_parameters` exit 0 | `tests/test_controlled_training.py:61` - `with self.assertRaises(ValueError):` | PASS |
| C3 | seed_everything deterministically seeds Python random, NumPy, PyTorch CPU, and PyTorch CUDA generators | `PYTHONPATH=src .venv/bin/python3 -m unittest discover -s tests -k test_seed_everything_deterministic_reproducibility` exit 0 | `tests/test_controlled_training.py:104` - `self.assertTrue(torch.allclose(r1, r2))` | PASS |
| C4 | load_controlled_config parses declarative YAML into validated config without absolute/personal paths | `PYTHONPATH=src .venv/bin/python3 -m unittest discover -s tests -k test_load_controlled_config_from_yaml` exit 0 | `tests/test_controlled_training.py:119` - `self.assertEqual(cfg.architectures, ("resnet50_1d", "convnext_tiny_1d"))` | PASS |
| C5 | ControlledTrainer supports PyTorch AMP mixed precision, gradient clipping, and step execution | `PYTHONPATH=src .venv/bin/python3 -m unittest discover -s tests -k test_controlled_trainer_amp_and_grad_clip` exit 0 | `tests/test_controlled_training.py:171` - `self.assertTrue(metrics["loss"] >= 0.0)` | PASS |
| C6 | ControlledTrainer updates weights with AdamW and schedules learning rates with linear warmup followed by CosineAnnealingLR | `PYTHONPATH=src .venv/bin/python3 -m unittest discover -s tests -k test_controlled_trainer_adamw_cosine_warmup` exit 0 | `tests/test_controlled_training.py:188` - `self.assertIsInstance(trainer.optimizer, torch.optim.AdamW)` | PASS |
| C7 | ControlledTrainer computes classification loss via CrossEntropyLoss with exact label smoothing = 0.1 | `PYTHONPATH=src .venv/bin/python3 -m unittest discover -s tests -k test_controlled_trainer_label_smoothing` exit 0 | `tests/test_controlled_training.py:218` - `self.assertEqual(trainer.criterion.label_smoothing, 0.1)` | PASS |
| C8 | ResourceProfiler measures peak VRAM, step execution latency in ms, and sequence throughput in sequences/second | `PYTHONPATH=src .venv/bin/python3 -m unittest discover -s tests -k test_resource_profiler_telemetry_metrics` exit 0 | `tests/test_controlled_training.py:230` - `self.assertGreater(profile.mean_step_time_ms, 0.0)` | PASS |
| C9 | ControlledTrainer.evaluate computes Top-1, Top-5, Macro-F1 across validation batches and saves optimal checkpoint based on checkpoint_metric | `PYTHONPATH=src .venv/bin/python3 -m unittest discover -s tests -k test_controlled_trainer_evaluation_and_checkpoint_selection` exit 0 | `tests/test_controlled_training.py:247` - `self.assertIn("macro_f1", eval_metrics)` | PASS |
| C10 | CLI scripts/run_controlled_training.py accepts --config, --steps, --seeds, --device, --dry-run, and --output-dir arguments | `PYTHONPATH=src .venv/bin/python3 -m unittest discover -s tests -k test_run_controlled_training_cli_arguments` exit 0 | `tests/test_controlled_training.py:268` - `self.assertEqual(proc.returncode, 0)` | PASS |
| C11 | CLI scripts/run_controlled_training.py --dry-run validates config, models, parameter parity, and exits 0 | `PYTHONPATH=src .venv/bin/python3 -m unittest discover -s tests -k test_run_controlled_training_dry_run_exit_zero` exit 0 | `tests/test_controlled_training.py:292` - `self.assertEqual(proc.returncode, 0)` | PASS |
| C12 | CLI scripts/run_controlled_training.py executes sequential training across multiple seeds and saves structured JSON logs and checkpoints | `PYTHONPATH=src .venv/bin/python3 -m unittest discover -s tests -k test_run_controlled_training_multi_seed_execution` exit 0 | `tests/test_controlled_training.py:322` - `self.assertEqual(proc.returncode, 0)` | PASS |
| C13 | CLI scripts/run_controlled_training.py exits with status 2 when an invalid option or conflicting parameter is passed | `PYTHONPATH=src .venv/bin/python3 -m unittest discover -s tests -k test_run_controlled_training_cli_invalid_options_exit_two` exit 0 | `tests/test_controlled_training.py:346` - `self.assertEqual(proc.returncode, 2)` | PASS |
| C14 | generate_controlled_experiment_report writes docs/controlled_architecture_experiment.md with pre-defined budget, parity matrix, and GPU cost estimates | `PYTHONPATH=src .venv/bin/python3 -m unittest discover -s tests -k test_generate_controlled_experiment_report_budget_and_gpu_cost` exit 0 | `tests/test_controlled_training.py:361` - `self.assertTrue(report_file.exists())` | PASS |
| C15 | docs/controlled_architecture_experiment.md contains side-by-side comparative metrics for ResNet50 and ConvNeXt | `PYTHONPATH=src .venv/bin/python3 -m unittest discover -s tests -k test_controlled_experiment_report_side_by_side_metrics` exit 0 | `tests/test_controlled_training.py:378` - `self.assertIn("resnet50_1d", content)` | PASS |
| C16 | docs/controlled_architecture_experiment.md contains mandatory scientific causal demarcation text restricting causal claims to evaluated configurations | `PYTHONPATH=src .venv/bin/python3 -m unittest discover -s tests -k test_controlled_experiment_report_causal_demarcation` exit 0 | `tests/test_controlled_training.py:397` - `self.assertIn(expected_text, content)` | PASS |
| C17 | verify_controlled_report_consistency asserts that all report tables match executed run artifacts without missing values | `PYTHONPATH=src .venv/bin/python3 -m unittest discover -s tests -k test_verify_controlled_report_consistency` exit 0 | `tests/test_controlled_training.py:408` - `self.assertTrue(is_valid)` | PASS |

## Gate

`PYTHONPATH=src .venv/bin/python3 -m unittest discover -s tests -k test_controlled` - 17 passed, 0 failed
