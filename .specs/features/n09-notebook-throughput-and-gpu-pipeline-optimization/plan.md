# n09-notebook-throughput-and-gpu-pipeline-optimization — Plan

## Problem

Reproduction notebooks in `notebooks/reproduction/short/` suffer from severe GPU/CPU synchronization, worker respawn, and disk I/O bottlenecks when executed on full cluster datasets (`~13.65M` training sequences across 256 Parquet files and `~1.71M` validation/test sequences):

1. `ControlledTrainer.train_epoch` and `ControlledTrainer.evaluate_epoch` in `src/viral_cnn/training/trainer.py` execute four blocking GPU-to-CPU stream synchronizations (`loss.item()`, `(preds == batch_y).sum().item()`, `preds.detach().cpu().numpy()`, `batch_y.detach().cpu().numpy()`) and three CPU `np.bincount(..., minlength=13683)` allocations per batch (~26,664 CUDA sync barriers per epoch).
2. `inject_noise` is invoked on CPU tensors before `.to(self.device, non_blocking=True)`, running random number generation on the CPU and invalidating the `pin_memory` buffer returned by the `DataLoader`.
3. `create_optimized_dataloader` in `src/viral_cnn/notebook_runtime.py` caps `num_workers` at 4 and forces `persistent_workers=False`, respawning worker processes 30 times across 15 epochs.
4. `OneHotEmbedding.forward` in `src/viral_cnn/data/encoding.py` clones input tensors and evaluates two boolean mask assignments on every forward pass instead of using `torch.clamp`.
5. Notebooks `03_finetune_resnet50_species.ipynb` and `04_finetune_convnext_species.ipynb` use `batch_size=2048` instead of the historical `4096`, call `torch.compile(model)` without `dynamic=True` (triggering TorchDynamo recompilation stalls on partial tail batches), and serialize identical checkpoints up to three times per epoch via `torch.save` (~5.7 GB/epoch for ConvNeXt).
6. In `02_analyze_confidence_and_calibration.ipynb`, `TaxonomyEntropyAnalyzer.calculate_activation_entropy` (`src/viral_cnn/eval/calibration.py`) computes activation energy in FP16 under `autocast`, overflowing FP16 (`> 65504`) for ResNet50 (`D=2048`) and returning `H_act: nan`; additionally, `get_confused_class_samples` (`src/viral_cnn/eval/confusion.py`) iterates sample-by-sample in Python with `.item()` across 1.71M sequences.
7. In `src/viral_cnn/data/retraining_audit.py`, `audit_retraining_need` executes 1.71M individual SQLite `SELECT 1` statements instead of batched queries.

## Sources

None - internal reproduction codebase (`src/viral_cnn/`, `notebooks/reproduction/short/`, and historical notebook `short/finetune_resnet50_species.ipynb`).

## Flow

1. `notebooks/reproduction/short/03_finetune_resnet50_species.ipynb`, `04_finetune_convnext_species.ipynb`, `02_analyze_confidence_and_calibration.ipynb` (existing) -> invoke runtime helpers and training/evaluation pipelines.
2. `src/viral_cnn/notebook_runtime.py` (`create_optimized_dataloader`, existing) -> builds `DataLoader` with `persistent_workers=True` and up to 8 workers.
3. `src/viral_cnn/training/trainer.py` (`ControlledTrainer`, existing) -> transfers batch to device (`non_blocking=True`), applies `inject_noise` on device, runs `OneHotEmbedding` (`src/viral_cnn/data/encoding.py`, existing) with `torch.clamp`, accumulates loss/Top-1/Top-5/per-class `torch.bincount` tensors on device, and synchronizes once at epoch end.
4. `src/viral_cnn/eval/runner.py` (`evaluate_dataloader`, existing), `src/viral_cnn/eval/calibration.py` (`TaxonomyEntropyAnalyzer`, existing), `src/viral_cnn/eval/confusion.py` (`get_confused_class_samples`, existing), and `src/viral_cnn/data/retraining_audit.py` (`audit_retraining_need`, existing) -> execute vectorized GPU/CPU evaluation and batched SQLite lookups.
5. out: epoch metrics CSV, deduplicated checkpoints (`finetune_noise_last.pt`, `finetune_noise_best.pt`), and finite FP32 activation entropy metrics.

## Impact

- **Config / env**: `None - existing REMOTE_SERVER and SMOKE_TEST environment flags remain unchanged`
- **Flags**: `None - no CLI flag changes`
- **Events / logs**: `notebooks/reproduction/short/03_finetune_resnet50_species.ipynb` and `04_finetune_convnext_species.ipynb` log per-epoch elapsed time in seconds (`dt`).
- **Background work**: `create_optimized_dataloader` sets `persistent_workers=(workers > 0)` to keep worker processes alive across epochs.
- **Compat**: `ControlledTrainer.train_epoch` and `evaluate_epoch` return the exact same `EpochMetrics` dataclass fields and values; `inject_noise` preserves deterministic CPU generator compatibility when a CPU `torch.Generator` is explicitly supplied.
- **Neighbors**: `notebooks/reproduction/short/01_benchmark_clean_and_noise.ipynb`, `02_analyze_confidence_and_calibration.ipynb`, `03_finetune_resnet50_species.ipynb`, `04_finetune_convnext_species.ipynb`, and `10_decisao_retreinamento_por_genoma.ipynb` all benefit from the shared runtime and evaluation optimizations.

## Relations

None - no stored-data shape change.

## Surface

None - nothing consumed outside.

## Landing

| Decision | Literal shape | Rejected alternative |
|---|---|---|
| Door 1: GPU-resident epoch metric accumulation in `ControlledTrainer` | `support_counts = torch.zeros(num_c, dtype=torch.int64, device=self.device)` updated via `torch.bincount(batch_y, minlength=num_c)` on `self.device`, synced once after the batch loop | Per-batch `preds.detach().cpu().numpy()` and `np.bincount` on CPU (causes 26,664 CUDA stream syncs per epoch) |
| Door 2: FP32 cast before activation energy summation in `TaxonomyEntropyAnalyzer.calculate_activation_entropy` | `z = activations.flatten(1).float()` before `energy = z ** 2` | Computing `z ** 2` and `energy.sum(dim=1)` in `float16` under `autocast` (overflows 65504 and returns `nan` for ResNet50) |
| Door 3: Single `torch.save` per epoch with filesystem copy in fine-tuning notebooks `03` and `04` | `trainer.save_epoch_checkpoint(last_ckpt, ...)` followed by `shutil.copyfile(last_ckpt, best_ckpt)` | Calling `trainer.save_epoch_checkpoint` three times per epoch for identical state dicts (writes up to 5.7 GB/epoch) |

## Criteria

### Slice S1 — GPU-resident training/evaluation loop, device-first noise injection, clamp encoding, and persistent DataLoader workers

**Acceptance Criteria**:

1. WHEN `ControlledTrainer.train_epoch` or `ControlledTrainer.evaluate_epoch` processes batches, the trainer SHALL accumulate batch loss, Top-1 correct count, Top-5 correct count, and per-class `support_counts`, `pred_counts`, and `tp_counts` as tensors on `self.device` and synchronize to CPU only once after the batch loop completes.
2. WHEN `ControlledTrainer.train_epoch` resets gradients before each batch, the trainer SHALL invoke `self.optimizer.zero_grad(set_to_none=True)`.
3. WHEN `ControlledTrainer.train_epoch` or `ControlledTrainer.evaluate_epoch` receives a batch and `noise_generator` is `None` or resides on `self.device`, the trainer SHALL move `batch_x` and `batch_y` to `self.device` with `non_blocking=True` prior to calling `inject_noise`.
4. IF an explicit `noise_generator` on CPU is passed to `ControlledTrainer.train_epoch` or `ControlledTrainer.evaluate_epoch`, THEN the trainer SHALL apply `inject_noise` using that generator before transferring `batch_x` to `self.device` so deterministic CPU-seeded tests produce identical sequences.
5. WHEN `OneHotEmbedding.forward` encodes an integer sequence tensor `x`, the layer SHALL bound indices via `torch.clamp(x, min=0, max=self.num_features)` without cloning `x` and SHALL map index `4` and out-of-range indices to the zero vector `[0.0, 0.0, 0.0, 0.0]`.
6. WHEN `create_optimized_dataloader` constructs a `DataLoader` with `workers > 0`, the function SHALL set `persistent_workers=True` and cap default worker count at `min(8, max(1, cpu_cnt // 2), num_files)`.

### Slice S2 — Numerical stability in activation entropy, vectorized confused-sample lookup, and batched SQLite audit

**Acceptance Criteria**:

7. WHEN `TaxonomyEntropyAnalyzer.calculate_activation_entropy` computes normalized activation energy from `float16` or `bfloat16` feature tensors, the method SHALL cast flattened activations to `torch.float32` prior to squaring and summing so that high-dimensional embeddings (`D=2048`) produce finite entropy values with zero `NaN` entries.
8. WHEN `get_confused_class_samples` searches a `DataLoader` for `(true_class_idx, pred_class_idx)` pairs, the function SHALL filter candidate indices per batch using vectorized tensor comparison (`torch.isin` or boolean mask) instead of iterating over every sample in Python.
9. WHEN `audit_retraining_need` queries the SQLite hash index during test-split scanning, the function SHALL query candidate hashes in chunks of up to 900 hashes per SQL statement instead of issuing one `SELECT 1` query per sequence.

### Slice S3 — Fine-tuning and calibration reproduction notebook optimizations

**Acceptance Criteria**:

10. WHEN `notebooks/reproduction/short/03_finetune_resnet50_species.ipynb` or `04_finetune_convnext_species.ipynb` runs with `ctx.is_smoke_test == False`, the notebook SHALL configure `batch_size = 4096` and `torch.compile(model, dynamic=True)`.
11. WHEN `notebooks/reproduction/short/03_finetune_resnet50_species.ipynb` or `04_finetune_convnext_species.ipynb` completes an epoch, the notebook SHALL serialize the checkpoint once via `trainer.save_epoch_checkpoint` to `finetune_noise_last.pt` and update `finetune_noise_{epoch}.pt` and `finetune_noise_best.pt` via `shutil.copyfile`.
12. WHEN `notebooks/reproduction/short/02_analyze_confidence_and_calibration.ipynb` calls `get_confused_class_samples`, the notebook SHALL pass `max_batches=2 if ctx.is_smoke_test else 50` to bound sample-pair lookup time.

## Out of scope

- Modifying historical notebooks in `short/` or `long/` (prohibited by `AGENTS.md` and `AD-004`).
- Changing model architectures, loss functions (`CrossEntropyLoss`), optimizer hyperparameters (`lr=1e-3`, `weight_decay=0.01`, `ReduceLROnPlateau`), or taxonomic class counts (13,683 species, 2,702 genera).
- Multi-GPU `DistributedDataParallel` training.

## Assumptions

| Assumption | Chosen default | Rationale |
|---|---|---|
| Single-GPU VRAM capacity on cluster nodes for fine-tuning | `batch_size = 4096` when `not ctx.is_smoke_test` | Matches historical notebook `short/finetune_resnet50_species.ipynb` cell 4 (`batch_size=4096`) and halves gradient steps per epoch |
| `torch.bincount` support on target PyTorch device | `torch.bincount` on `self.device` with `minlength=num_c` | Supported natively on both CPU and CUDA for 1D non-negative integer tensors |
| SQLite host parameter limit for batched hash lookup | Chunk size of 900 parameters per `WHERE hash IN (...)` query | Safely below SQLite's default `SQLITE_MAX_VARIABLE_NUMBER` (999 / 32766) |

**Open questions:** none

## Observable

None - no user-facing surface (internal Python scientific library and Jupyter reproduction notebooks).
