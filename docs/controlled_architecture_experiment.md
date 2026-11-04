# C05 fine-tuning audit and controlled comparison status

## Evidence status

**Controlled comparison: `NOT_RUN`.** This repository snapshot has no local canonical base checkpoints or empirical cluster data at `/home/mlima/workspace`. No C05 fine-tuning, test evaluation, controlled run, or resource measurement is claimed by this document. Synthetic contracts validate mechanics only.

The historical notebook code and model registry disagree on ConvNeXt dropout (notebook 0.5; registry 0.7). Historical selection is also not established by checkpoint names: the current reproduction loop selects ResNet `last` and ConvNeXt `best_val_macro_f1`, while an older report claimed best validation loss at epoch 7. Equivalence to published weights is unknown without the original configuration and logs. The historical scheduler is stepped per epoch although `T_max` was set from the loader length times epochs; this behavior is recorded, not silently corrected.

## Minimum C05 protocol

The reproducible minimum uses the existing fine-tuning starts `resnet_clean_species` and `convnext_transfer_species`, 15 epochs, AdamW (`lr=5e-5`, `weight_decay=1e-4`), label smoothing 0.1 and epsilon 0.05. The protocol is an audit of two distinct initializations, not an architecture-only intervention. Every execution must carry a unique run namespace, input/checkpoint hashes, the ordered Species mapping hash, effective model dropout, observed scheduler unit, selection rule and per-epoch metrics/support.

The notebooks evaluate a limited clean/noisy validation preview before training. The selected checkpoint is frozen using validation alone; o sistema **não seleciona checkpoints pelo teste**. Then it evaluates the frozen checkpoint on the split de teste identificado for epsilon values 0, 0.02, 0.05, 0.10 and 0.15. A preview, smoke run, incomplete epoch budget or absent artifact cannot be labeled a full empirical result.

Checkpoint roles remain distinct: `last`, `best` and `selected`. Resume is opt-in through `RESUME_CHECKPOINT`; C05 exact resume validates run identity and restores model, optimizer, scheduler, scaler, Python/NumPy/Torch/noise RNG state and history. Legacy checkpoints may seed a new identified adaptation, but they do not prove exact resume continuity.

## Optional controlled experiment

The separate protocol `c05_controlled_scratch_v1` remains **`NOT_RUN` and was not activated**. Conclusão causal indisponível. It requires both architectures initialized from scratch, identical effective data/order, corruption, optimizer, scheduler, loss, dropout, batch/precision, selection and a predeclared full budget for seeds 42, 43 and 44. The existing 200-step configuration is a pilot budget, not a complete experiment. No six-run comparison or causal estimate is available.

If later activated, status must distinguish `PILOT`, `PARTIAL` and `COMPLETE`; `COMPLETE` requires all six model/seed runs at the declared budget and evaluation on the same identified test split. Compute costs are measured only from recorded hardware, elapsed time, processed sequences and GPU utilization; parameter-count FLOPs or extrapolations remain estimates.

## Allowed interpretation

Until the optional experiment has complete, manifest-backed executions, C06 may use identified historical checkpoints only with configuration-specific wording and the explicit limitation that há uma execução por configuração, com origens distintas; não atribui diferenças exclusivamente à arquitetura. The primary split remains fragment-level and closed-set; it does not establish generalization to unseen genomes or species.
