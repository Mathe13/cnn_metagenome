# fix-compiled-model-checkpoint-resume checks

Profile: light

## Intent

When `model` is wrapped with `torch.compile(model, dynamic=True)` (producing a `torch._dynamo.eval_frame.OptimizedModule` whose outer `state_dict()` prefixes every parameter/buffer key with `"_orig_mod."` while its inner uncompiled module resides at `model._orig_mod`), calling `trainer.load_resume_checkpoint` in `03_finetune_resnet50_species.ipynb` or `04_finetune_convnext_species.ipynb` strips `"_orig_mod."` from the checkpoint keys (`cleaned_state`) and then calls `self.model.load_state_dict(cleaned_state)` directly on the outer `OptimizedModule`, raising `RuntimeError: Error(s) in loading state_dict for OptimizedModule: Missing key(s) in state_dict: "_orig_mod.conv1.weight"... Unexpected key(s) in state_dict: "conv1.weight"...`. Furthermore, `ControlledTrainer.__init__` checks `name.startswith(prefix)` on `self.model.named_parameters()` without unwrapping `_orig_mod`, which fails to match layer freeze prefixes like `"stem"` or `"stages.0"` on compiled models, and `load_model_checkpoint` in `registry.py` similarly calls `model.load_state_dict(clean_state_dict)` on the outer model. When this fix ships, `ControlledTrainer` and `load_model_checkpoint` unwrap `getattr(model, "_orig_mod", model)` for parameter freezing, clean unprefixed checkpoint serialization (`save_epoch_checkpoint` and `check_and_save_checkpoint`), and checkpoint loading (`load_resume_checkpoint` and `load_model_checkpoint`), seamlessly supporting both compiled (`OptimizedModule`) and uncompiled `nn.Module` instances and checkpoints saved with or without `"_orig_mod."` prefixes.

4 checks in 1 slice · 0 one-way doors · 0 open, of which 0 block

## Checks

### S1 - Unwrapped `_orig_mod` handling for compiled models in trainer and registry · 3 files · 42 KB · ~11k

**C1** - `ControlledTrainer.load_resume_checkpoint` and `ControlledTrainer.resume_checkpoint` restore `cleaned_state` into `getattr(self.model, "_orig_mod", self.model)` without raising `RuntimeError` when `self.model` is wrapped with `torch.compile` (or an `_orig_mod` wrapper), whether the checkpoint on disk was saved with `"_orig_mod."` key prefixes or bare keys
Proof: `PYTHONPATH=src .venv/bin/python3 -m unittest tests/test_n03_finetune_training.py -k test_compiled_model_load_resume_checkpoint`

**C2** - `ControlledTrainer.save_epoch_checkpoint` and `ControlledTrainer.check_and_save_checkpoint` serialize `model_state_dict` from `getattr(self.model, "_orig_mod", self.model).state_dict()` with clean unprefixed keys (no `"_orig_mod."` prefix), even when `self.model` is wrapped in an `_orig_mod` container
Proof: `PYTHONPATH=src .venv/bin/python3 -m unittest tests/test_n03_finetune_training.py -k test_compiled_model_save_checkpoint_strips_orig_mod_prefix`

**C3** - `ControlledTrainer.__init__` resolves `freeze_layers` prefixes against `getattr(self.model, "_orig_mod", self.model)` so that frozen parameters (`requires_grad=False`) and frozen `BatchNorm` modules (`eval()`) are enforced identically whether `model` is compiled or uncompiled
Proof: `PYTHONPATH=src .venv/bin/python3 -m unittest tests/test_n03_finetune_training.py -k test_compiled_model_freeze_layers_and_bn_eval`

**C4** - `load_model_checkpoint` in `src/viral_cnn/models/registry.py` loads `clean_state_dict` into `getattr(model, "_orig_mod", model)` so that loading either a bare or `"_orig_mod."`-prefixed checkpoint into a compiled or `_orig_mod`-wrapped model succeeds with `strict=True`
Proof: `PYTHONPATH=src .venv/bin/python3 -m unittest tests/test_model_registry.py -k test_load_model_checkpoint_into_compiled_wrapper`

## Coverage

| Set (size) | Member -> proof | Unproven |
| --- | --- | --- |
| Checkpoint key formats x model wrapper states (4) | bare keys into compiled model C1 · `_orig_mod.` keys into compiled model C1 · bare keys into uncompiled model C1 · `_orig_mod.` keys into uncompiled model C4 | - |
| Trainer & registry `_orig_mod` surfaces (4) | `ControlledTrainer.load_resume_checkpoint` C1 · `ControlledTrainer.save_epoch_checkpoint` & `check_and_save_checkpoint` C2 · `ControlledTrainer.__init__` (`freeze_layers`) C3 · `registry.load_model_checkpoint` C4 | - |

- No other check claims more than the single case its proof exercises

## Swept

Where each unwritten requirement landed. All nine, one line each, every time.

- validation: C1, C4 (validates strict state_dict key compatibility on `getattr(model, "_orig_mod", model)` and class head dimension matching)
- failure modes: C1 (eliminates `RuntimeError: Error(s) in loading state_dict for OptimizedModule: Missing key(s) in state_dict: "_orig_mod..."`)
- idempotency: C1, C2 (saving and resuming a compiled model across multiple epochs produces identical unprefixed checkpoint keys and parameter tensors)
- authorization: n/a - local and cluster PyTorch model state serialization with no authentication boundary
- concurrency: n/a - single-process checkpoint serialization and restoration
- data lifecycle: C2 (serializes canonical unprefixed state dicts to disk so saved checkpoints are portable across compiled and uncompiled runtimes)
- dependency failure: existing - `load_resume_checkpoint` and `load_model_checkpoint` raise `FileNotFoundError` when checkpoint path is missing
- state transitions: C1, C3 (preserves `epoch + 1` continuation, `training_history`, optimizer/scheduler/scaler states, and frozen BatchNorm `eval()` state across resume)
- observability: existing - `load_model_checkpoint` logs `[LOG_PATH] Loading checkpoint from: ...` via `log_artifact_path`

## Out of scope

- Modifying historical checkpoints in `cnn_kraken/` or `cnn_kraken_species/` - read-only reference artifacts per `AGENTS.md`

## Handoff

- S1 = ~11k tokens across 4 files (`src/viral_cnn/training/trainer.py`, `src/viral_cnn/models/registry.py`, `tests/test_n03_finetune_training.py`, `tests/test_model_registry.py`), well under the 150k budget - one builder
- **Boundary:** C1-C4 closed at working-tree (`HEAD`)
- **Settled mid-build:** none
- **Abandoned:** none
