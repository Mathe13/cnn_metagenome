# S17 — Training Scheduler API Hardening and Ownership Contract

## Problem

In `src/viral_cnn/training/trainer.py`, two optimization interfaces currently handle learning rate updates:
- `train_step` automatically recomputes and overwrites parameter group learning rates via an internal warmup/cosine schedule (`_compute_scheduled_lr(current_step)`);
- `train_epoch` accepts an optional external scheduler instance (`scheduler`) stepped either per-batch or per-epoch (`scheduler_step_unit`).

Without explicit scheduler ownership, a training pipeline or custom script could accidentally pass an external scheduler to `train_epoch` while `train_step` simultaneously mutates learning rates, or configure `warmup_steps > 0` under constant learning rate mode without any active schedule stepping. This can cause silent optimization anomalies, conflicting learning rate progressions, and irreproducible training curves.

When this ships, `ControlledExperimentConfig` declares explicit scheduler ownership via `scheduler_mode: Literal["internal_step", "external_batch", "external_epoch", "constant"]`. Configurations enforce single ownership fail-fast: conflicting dual schedules (internal step schedule + external scheduler) or active warmup under constant mode raise `ValueError`. `ControlledTrainer` respects the declared owner without altering historical checkpoint reproduction semantics.

## Flow

This change hardens `viral_cnn.training.config` (exists) and `viral_cnn.training.trainer` (exists) while maintaining backward compatibility with historical reproduction runs.

1. Configuration options enter `ControlledExperimentConfig` (door 1) - validates that `scheduler_mode` is one of `{"internal_step", "external_batch", "external_epoch", "constant"}` and rejects `warmup_steps > 0` under `scheduler_mode == "constant"`.
2. `ControlledTrainer.__init__` receives config and resolves the active scheduler ownership mode (door 1).
3. If `train_epoch` receives an external scheduler while `scheduler_mode == "internal_step"` or `scheduler_mode == "constant"` (door 2), raises `ValueError` flagging conflicting scheduler ownership.
4. If `train_epoch` is called with `scheduler_mode in ("external_batch", "external_epoch")` but `scheduler is None` (door 2), raises `ValueError` flagging missing scheduler owner.
5. In `train_step` (exists), internal warmup/cosine scheduling is applied if and only if `scheduler_mode == "internal_step"`, leaving external schedules or constant rates undisturbed.

## Impact

| Front | What changes |
| --- | --- |
| domain | `src/viral_cnn/training/config.py` defines `scheduler_mode: Literal["internal_step", "external_batch", "external_epoch", "constant"]` with validation |
| domain | `src/viral_cnn/training/trainer.py` enforces exclusive scheduler ownership in `train_step` and `train_epoch` |
| stored data | no stored files or checkpoints are modified |

## Relations

None - no stored-data shape change

## Surface

None - nothing consumed outside

## Landing

| One-way door | Literal shape | Alternative rejected |
| --- | --- | --- |
| 1 — posse explícita de agendador via scheduler_mode | `scheduler_mode: Literal["internal_step", "external_batch", "external_epoch", "constant"]` em `ControlledExperimentConfig` | Permitir agendador implícito deduzido por heurísticas silenciosas: mascara erros onde tanto `_compute_scheduled_lr` quanto `scheduler.step()` alteram o LR simultaneamente. |
| 2 — fail-fast para conflitos e agendadores ausentes | `raise ValueError("Conflicting schedulers: ...")` e `raise ValueError("Missing external scheduler: ...")` | Ignorar silenciosamente o agendador externo ou sobrescrever o LR sem aviso. |

- Nothing else in this change is hard to reverse

## Criteria

### S1: Training scheduler ownership enforcement (P1)

A configuração e o loop de treinamento devem exigir e validar a posse exclusiva do agendador de taxa de aprendizado, impedindo agendadores concorrentes ou modos mal configurados.

**Acceptance Criteria**

1. WHEN `ControlledExperimentConfig` is created with valid `scheduler_mode` THEN the config SHALL store the declared mode without error.
2. IF `ControlledExperimentConfig` receives an invalid `scheduler_mode` THEN the system SHALL raise a `ValueError` listing valid modes.
3. IF `ControlledExperimentConfig` specifies `warmup_steps > 0` with `scheduler_mode == "constant"` in controlled parity mode THEN the system SHALL raise a `ValueError`.
4. IF `train_epoch` is called with an external `scheduler` while `scheduler_mode == "internal_step"` or `"constant"` THEN `ControlledTrainer` SHALL raise a `ValueError` detailing conflicting scheduler ownership.
5. IF `train_epoch` is called without an external `scheduler` while `scheduler_mode == "external_batch"` or `"external_epoch"` THEN `ControlledTrainer` SHALL raise a `ValueError` detailing the missing external scheduler.
6. WHEN `train_step` is executed under `scheduler_mode == "internal_step"` THEN the trainer SHALL update optimizer learning rate according to the warmup/cosine formula.
7. WHEN `train_step` is executed under `scheduler_mode == "constant"` THEN the trainer SHALL preserve the initial learning rate without scheduled decay.

**Independent test:** Execute `python3 -m unittest tests/test_s17_training_scheduler_api_hardening.py` verifying config mode validation, warmup-constant conflict rejection, conflicting scheduler detection in `train_epoch`, missing scheduler detection, and LR dynamics under `internal_step` vs `constant`.

## Out of scope

| Excluded | Why |
| --- | --- |
| Alteração de rotinas de treinamento no JupyterHub remoto | S17 formaliza a API e a segurança do agendador em código Python sem exigir novo treinamento completo de 14 épocas no cluster. |
| Alteração de checkpoints históricos | Preserva o comportamento de reprodução histórica sem retroagir sobre checkpoints existentes. |

## Assumptions

| Assumption | Chosen default | Rationale | Confirmed? |
| --- | --- | --- | --- |
| scheduler_mode padrão para controlled_parity | `"internal_step"` quando `warmup_steps > 0` | Mantém paridade exata com o comportamento existente de step-level warmup/cosine. | y |
| scheduler_mode padrão para historical_reproduction | `"constant"` quando `warmup_steps == 0` | Mantém paridade exata com reproduções históricas sem warmup. | y |

**Open questions:** none - all resolved or logged above.

## Observable

None - no user-facing surface

## Sources

- ROADMAP_IMPROVEMENTS.md (Section 11: S17 — Training scheduler API hardening) - define os quatro modos de scheduler_mode, as regras de falha para múltiplos agendadores ou warmup sem agendador, e o invariante de não alterar checkpoints históricos.
- ROADMAP_IMPROVEMENTS.md (Section 20: Milestone D — Defensive hardening) - previne falhas silenciosas em experimentos futuros.
- AGENTS.md (Invariantes científicos) - preservar origem, épocas, otimizador e scheduler sem alterações silenciosas.
