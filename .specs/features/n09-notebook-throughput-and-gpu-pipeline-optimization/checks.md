# n09-notebook-throughput-and-gpu-pipeline-optimization — Checks

**Profile**: `light`

## Checks

### Slice S1 — GPU-resident training/evaluation loop, device-first noise injection, clamp encoding, and persistent DataLoader workers

**C1** — `ControlledTrainer.train_epoch` and `ControlledTrainer.evaluate_epoch` accumulate loss, Top-1, Top-5, and per-class `support_counts`, `pred_counts`, and `tp_counts` via `torch.bincount` on `self.device`, producing exact `EpochMetrics` parity (`loss`, `top1_acc`, `top5_acc`, `macro_f1`, `weighted_f1`) without calling `.detach().cpu().numpy()` inside the batch loop.
Proof: `.venv/bin/python3 -m unittest tests/test_n09_pipeline_optimization.py -k test_c1_trainer_gpu_resident_epoch_metrics`

**C2** — `ControlledTrainer.train_epoch` calls `optimizer.zero_grad(set_to_none=True)` on every training step.
Proof: `.venv/bin/python3 -m unittest tests/test_n09_pipeline_optimization.py -k test_c2_trainer_zero_grad_set_to_none`

**C3** — `ControlledTrainer.train_epoch` and `ControlledTrainer.evaluate_epoch` transfer `batch_x` and `batch_y` to `self.device` with `non_blocking=True` before invoking `inject_noise` when `noise_generator` is `None` or on `self.device`, and preserve deterministic output when a CPU `torch.Generator` is provided.
Proof: `.venv/bin/python3 -m unittest tests/test_n09_pipeline_optimization.py -k test_c3_trainer_device_first_noise_and_generator_compat`

**C4** — `OneHotEmbedding.forward` bounds input token indices via `torch.clamp(x, min=0, max=4)` without calling `.clone()`, mapping `A/C/G/T` (`0/1/2/3`) to unit vectors and `4` or out-of-bounds indices (`-1`, `5`, `99`) to `[0.0, 0.0, 0.0, 0.0]`.
Proof: `.venv/bin/python3 -m unittest tests/test_n09_pipeline_optimization.py -k test_c4_onehot_embedding_clamp_and_padding`

**C5** — `create_optimized_dataloader` sets `persistent_workers=True` whenever `workers > 0` and computes default workers up to `min(8, max(1, cpu_cnt // 2), num_files)`.
Proof: `.venv/bin/python3 -m unittest tests/test_n09_pipeline_optimization.py -k test_c5_dataloader_persistent_workers_and_cap`

### Slice S2 — Numerical stability in activation entropy, vectorized confused-sample lookup, and batched SQLite audit

**C6** — `TaxonomyEntropyAnalyzer.calculate_activation_entropy` casts flattened `float16` and `bfloat16` activations to `torch.float32` prior to squaring and summing, returning finite non-NaN entropy values for `D=2048` tensors with magnitudes up to `10.0` that would overflow FP16 (`65504`).
Proof: `.venv/bin/python3 -m unittest tests/test_n09_pipeline_optimization.py -k test_c6_activation_entropy_fp16_no_nan`

**C7** — `get_confused_class_samples` filters target `(true_class_idx, pred_class_idx)` pairs per batch using vectorized tensor membership (`torch.isin`) and respects `max_batches`, returning exact matching sequence tensors and stopping early once all pairs reach `num_samples`.
Proof: `.venv/bin/python3 -m unittest tests/test_n09_pipeline_optimization.py -k test_c7_vectorized_confused_class_samples`

**C8** — `audit_retraining_need` queries the SQLite hash table in chunks of up to 900 hashes per `SELECT hash FROM hashes WHERE hash IN (...)` query and produces exact `overlap_records` and `overlap_rate` counts.
Proof: `.venv/bin/python3 -m unittest tests/test_n09_pipeline_optimization.py -k test_c8_batched_sqlite_retraining_audit`

### Slice S3 — Fine-tuning and calibration reproduction notebook optimizations

**C9** — `notebooks/reproduction/short/03_finetune_resnet50_species.ipynb` and `04_finetune_convnext_species.ipynb` configure `batch_size = 32 if ctx.is_smoke_test else 4096` and `torch.compile(model, dynamic=True)`.
Proof: `.venv/bin/python3 -m unittest tests/test_n09_pipeline_optimization.py -k test_c9_finetune_notebooks_batch_size_and_dynamic_compile`

**C10** — `notebooks/reproduction/short/03_finetune_resnet50_species.ipynb` and `04_finetune_convnext_species.ipynb` call `trainer.save_epoch_checkpoint` once per epoch to `finetune_noise_last.pt` and copy to `finetune_noise_{epoch}.pt` and `finetune_noise_best.pt` via `shutil.copyfile`, while logging per-epoch elapsed time.
Proof: `.venv/bin/python3 -m unittest tests/test_n09_pipeline_optimization.py -k test_c10_finetune_notebooks_single_save_and_copyfile`

**C11** — `notebooks/reproduction/short/02_analyze_confidence_and_calibration.ipynb` invokes `get_confused_class_samples` with `max_batches=2 if ctx.is_smoke_test else 50`.
Proof: `.venv/bin/python3 -m unittest tests/test_n09_pipeline_optimization.py -k test_c11_calibration_notebook_bounded_confused_samples`

## Coverage

| Set | Members -> check | Unproven |
|---|---|---|
| Optimizations across pipeline modules (7) | `trainer.py` GPU-resident metrics -> C1 · `trainer.py` `zero_grad(set_to_none=True)` -> C2 · `trainer.py` device-first noise -> C3 · `encoding.py` `torch.clamp` -> C4 · `notebook_runtime.py` `persistent_workers` -> C5 · `calibration.py` FP32 activation entropy -> C6 · `confusion.py` + `retraining_audit.py` batching -> C7 | — |
| Affected reproduction notebooks (3) | `03_finetune_resnet50_species.ipynb` -> C9 · `04_finetune_convnext_species.ipynb` -> C10 · `02_analyze_confidence_and_calibration.ipynb` -> C11 | — |
| Out-of-bounds token values in `OneHotEmbedding` (4) | `-1` -> C4 · `4` (padding) -> C4 · `5` -> C4 · `99` -> C4 | — |

## Swept

- **Validation**: C4 (`OneHotEmbedding` clamps negative and `>=4` indices to index `4` zero vector).
- **Failure modes**: C6 (prevents `float16` overflow `NaN` in `calculate_activation_entropy` under AMP `autocast`).
- **Idempotency**: C3 (explicit CPU `torch.Generator` produces deterministic noise across repeated calls) and C10 (checkpoint copy via `shutil.copyfile` overwrites target atomically on rerun).
- **Authorization**: n/a - offline scientific PyTorch pipeline with no user authentication or network endpoints.
- **Concurrency**: C5 (`create_optimized_dataloader` preserves worker shard safety with `persistent_workers=True` across `ParquetSequenceDataset` workers).
- **Data lifecycle**: C10 (fine-tuning loop writes `finetune_noise_last.pt` once per epoch and copies to `finetune_noise_best.pt` without redundant serialization).
- **Dependency failure**: existing - `torch.compile` in notebooks `03` and `04` is wrapped in `try/except Exception` falling back to eager model execution.
- **State transitions**: C1 and C2 (`ControlledTrainer` state transition across batches resets gradients via `set_to_none=True` and synchronizes metrics once at epoch completion).
- **Observability**: C10 (fine-tuning loop prints per-epoch wall-clock duration in seconds alongside train/val loss, Top-1, Top-5, and Macro-F1).

## Handoff

- **Slices**: 3 (`S1`, `S2`, `S3`)
- **Checks**: 11 (`C1`–`C11`)
- **Files in play**: 10 (`src/viral_cnn/training/trainer.py`, `src/viral_cnn/data/encoding.py`, `src/viral_cnn/notebook_runtime.py`, `src/viral_cnn/eval/runner.py`, `src/viral_cnn/eval/calibration.py`, `src/viral_cnn/eval/confusion.py`, `src/viral_cnn/data/retraining_audit.py`, `notebooks/reproduction/short/03_finetune_resnet50_species.ipynb`, `notebooks/reproduction/short/04_finetune_convnext_species.ipynb`, `notebooks/reproduction/short/02_analyze_confidence_and_calibration.ipynb`)
- **Estimated load**: `10 files × ~2k + 11 checks × ~2.5k + ~15k = ~62.5k tokens < 150k` -> **single builder pass**.
