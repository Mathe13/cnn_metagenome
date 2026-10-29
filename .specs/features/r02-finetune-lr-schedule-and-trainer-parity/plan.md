# r02-finetune-lr-schedule-and-trainer-parity — Plan

## Problem

1. **Taxa de aprendizado inicial 20x menor em `ControlledTrainer`:** Em `src/viral_cnn/training/trainer.py`, `ControlledTrainer.__init__` inicializa o otimizador com `initial_lr = self._compute_scheduled_lr(0)`. Quando `ControlledExperimentConfig.warmup_steps` vale `20` (padrão), `initial_lr` se torna $5\times 10^{-5} / 20 = 2{,}5\times 10^{-6}$. Quando os notebooks de reprodução `notebooks/reproduction/short/03_finetune_resnet50_species.ipynb` e `04_finetune_convnext_species.ipynb` instanciam `CosineAnnealingLR(trainer.optimizer, ...)`, o scheduler do PyTorch adota `base_lrs = [2.5e-6]`, fazendo o decaimento ocorrer a partir de um valor 20x inferior ao pretendido, estagnando a acurácia sob ruído.
2. **`UserWarning` de `lr_scheduler.step()` chamado antes de `optimizer.step()` sob `GradScaler`:** Em `ControlledTrainer.train_epoch`, o método chama `scheduler.step()` a cada lote sem verificar se `GradScaler.step(optimizer)` pulou o passo do otimizador devido a gradientes `inf`/`NaN` (comum no primeiro lote de precisão mista FP16), gerando advertências no PyTorch e pulando o primeiro valor da grade de taxa de aprendizado. Além disso, não há suporte explícito para alternar entre escalonamento por lote (`"batch"`) e por época (`"epoch"`).
3. **Ausência de retomada automática de checkpoint (`load_resume_checkpoint`) e avaliação pré-treino:** Uma interrupção de sessão no cluster durante o treinamento de 15 épocas em `03` e `04` obriga o reinício do zero (~10h de GPU). Além disso, a Célula 3 não realiza avaliação pré-treino (Época 0) para demonstrar que o checkpoint base carregou com a acurácia esperada antes da adaptação ao ruído.

Quando esta especificação for concluída, `ControlledTrainer` inicializará `AdamW` diretamente com `config.learning_rate` ($5\times 10^{-5}$) sem warmup quando em modo de reprodução histórica ou com scheduler externo, suportará `scheduler_step_unit: Literal["epoch", "batch"]` com verificação estrita de `scaler.get_scale()` eliminando `UserWarning`, fornecerá `load_resume_checkpoint` capaz de restaurar pesos e histórico para prosseguir de `k + 1`, e os notebooks `03` e `04` incluirão avaliação na Época 0 e retomada automática.

## Flow

Reutiliza `ControlledTrainer`, `ControlledExperimentConfig`, `build_registered_model` e `write_notebook_manifest` existentes em `src/viral_cnn/`.

1. `src/viral_cnn/training/config.py` (existing) -> aceita `mode: Literal["historical_reproduction", "controlled_parity"] = "historical_reproduction"` e `scheduler_step_unit: Literal["epoch", "batch"] = "epoch"` (door 1).
2. `src/viral_cnn/training/trainer.py` (`ControlledTrainer`, existing) -> inicializa `optimizer` com `initial_lr = config.learning_rate` quando `mode == "historical_reproduction"` ou `warmup_steps == 0` (door 1), protege chamadas de `scheduler.step()` em `train_epoch` contra gradientes `inf` e suporta passo por época vs. por lote (door 2), e implementa `load_resume_checkpoint` (door 3).
3. `notebooks/reproduction/short/03_finetune_resnet50_species.ipynb` e `04_finetune_convnext_species.ipynb` (existing) -> Célula 3 executa avaliação pré-treino (Época 0) limpa e com ruído em subconjunto de validação, Célula 4 retoma automaticamente de `finetune_noise_last.pt` se existente, e Célula 5 imprime `history_df` completo com LR real de $5\times 10^{-5}$ na Época 1.
4. out: checkpoints versionados em `results_reproduction/checkpoints/` com histórico íntegro e telemetria de treino reproduzível.

## Impact

| Front | What changes |
| --- | --- |
| domain | `ControlledTrainer` em `trainer.py` suporta `load_resume_checkpoint`, inicialização de LR em $5\times 10^{-5}$ sem warmup para reprodução histórica, e `scheduler_step_unit` em `train_epoch` |
| configuration | `ControlledExperimentConfig` em `config.py` suporta parâmetro `mode` e valida valores de `warmup_steps` e unidades de scheduler |
| notebook runtime | `03_finetune_resnet50_species.ipynb` e `04_finetune_convnext_species.ipynb` ganham avaliação na Época 0 e capacidade de retomar de `finetune_noise_last.pt` |

## Relations

None - no stored-data shape change

## Surface

None - nothing consumed outside

## Landing

| One-way door | Literal shape | Alternative rejected |
| --- | --- | --- |
| Door 1: Inicialização de LR base no otimizador sem divisão espúria por warmup | Em `ControlledTrainer.__init__`: `is_historical = getattr(self.config, "mode", "") == "historical_reproduction" or getattr(self.config, "warmup_steps", 0) == 0`; `initial_lr = float(self.config.learning_rate) if is_historical else self._compute_scheduled_lr(0)` | Forçar `warmup_steps=0` manualmente em cada chamada de script (frágil e permite regressão silenciosa para LR dividida por 20) |
| Door 2: Despacho condicional de scheduler com guarda de `GradScaler` e unidade época/lote | Em `train_epoch`: se `scheduler_step_unit == "batch"`, chamar `scheduler.step()` apenas se `optimizer_stepped` (`scale_after >= scale_before`); se `scheduler_step_unit == "epoch"`, chamar `scheduler.step()` ao final da época | Chamar `scheduler.step()` incondicionalmente a cada lote (provoca `UserWarning: Detected call of lr_scheduler.step() before optimizer.step()` sob FP16 AMP) |
| Door 3: Assinatura e contrato de retomada `load_resume_checkpoint` | `load_resume_checkpoint(resume_path: Path \| str, scheduler: Optional[Any] = None) -> tuple[int, list[dict[str, Any]]]`, restaurando pesos do modelo, otimizador, scaler, scheduler e retornando `(int(checkpoint["epoch"]) + 1, checkpoint.get("training_history", []))` | Retornar apenas a época sem recuperar histórico de métricas anteriores, truncando os CSVs de treinamento |

- Nothing else in this change is hard to reverse

## Criteria

### S1: Paridade de taxa de aprendizado e ausência de advertências de scheduler (P1)

Garante taxa de aprendizado inicial correta de $5\times 10^{-5}$ e compatibilidade robusta com `GradScaler` e `CosineAnnealingLR`.

**Acceptance Criteria**

1. WHEN `ControlledTrainer` is instantiated with `learning_rate=5e-5` and `mode="historical_reproduction"` (or `warmup_steps=0`) and coupled with `CosineAnnealingLR` THEN the optimizer SHALL initialize with `trainer.optimizer.param_groups[0]["lr"] == 5e-5` at the start of Epoch 1.
2. WHEN `ControlledTrainer.train_epoch` runs with `amp=True`, `scheduler_step_unit="batch"` (or `"epoch"`), and `GradScaler` encounters infinite/NaN gradients in the first step causing `optimizer.step()` to be skipped THEN the function SHALL not trigger any `UserWarning` regarding `lr_scheduler.step() before optimizer.step()`.
3. WHEN `ControlledTrainer.save_epoch_checkpoint` saves an epoch and `load_resume_checkpoint` restores from it at epoch $k$ THEN the method SHALL restore model, optimizer, scaler, and scheduler states and return start epoch $k + 1$ along with the exact training history for epochs $1..k$.

**Independent test:** Executar testes unitários em `tests/test_n03_finetune_training.py` cobrindo LR inicial, supressão de advertências sob inf gradients em AMP, e integridade de retomada de checkpoints.

### S2: Avaliação pré-treino e retomada automática nos notebooks 03 e 04 (P1)

Adiciona Célula 3 de avaliação pré-treino (Época 0) e guarda de retomada automática de checkpoint em `03` e `04`.

**Acceptance Criteria**

1. WHEN `notebooks/reproduction/short/03_finetune_resnet50_species.ipynb` and `04_finetune_convnext_species.ipynb` are executed at Célula 3 THEN the notebook SHALL evaluate the loaded base checkpoint on validation clean ($\epsilon=0.0$) and validation noise ($\epsilon=0.05$) to establish Epoch 0 baseline accuracy prior to fine-tuning.
2. WHEN `finetune_noise_last.pt` exists in the checkpoint directory during Célula 4 execution THEN the notebook SHALL call `load_resume_checkpoint` and resume iteration from epoch $k + 1$ preserving prior history records.
3. WHEN Célula 5 executes in notebooks `03` and `04` THEN the notebook SHALL print the full `history_df` table and verify that `learning_rate` at Epoch 1 equals $5\times 10^{-5}$.

**Independent test:** Executar verificação estrutural e estática sobre as células dos notebooks `03` e `04` em `tests/test_n03_finetune_training.py`.

## Out of scope

| Excluded | Why |
| --- | --- |
| Execução de 15 épocas completas de treinamento em GPU local | Restrito a cluster JupyterHub remoto conforme AD-006 e AD-017; ambiente local valida via smoke test determinístico |
| Modificação de notebooks históricos em `short/` ou `long/` | Proibido por `AGENTS.md` e `AD-004` |
| Reescrita de otimizadores para modelos além de `ResNet50_1D` e `ConvNeXt_Tiny_1D` | O escopo do artigo foca exclusivamente nas duas famílias taxonômicas virais |

## Assumptions

| Assumption | Chosen default | Rationale | Confirmed? |
| --- | --- | --- | --- |
| Modo de aprendizado padrão em reprodução histórica | Quando `mode == "historical_reproduction"`, inicializar optimizer com `config.learning_rate` sem warmup | Notebooks históricos usavam $5\times 10^{-5}$ direto com CosineAnnealingLR sem divisão por 20 | y |
| Unidade de passo do scheduler | Suportar `"epoch"` (padrão) e `"batch"`, verificando `scaler.get_scale()` em `"batch"` | Elimina `UserWarning` do PyTorch e assegura compatibilidade tanto com loops por lote quanto por época | y |
| Formato de retorno de `load_resume_checkpoint` | Retornar `(resume_epoch, training_history)` com `resume_epoch = saved_epoch + 1` | Permite retomada transparente a partir do ponto de interrupção e continuidade do histórico | y |

**Open questions:** none - all resolved or logged above.

## Observable

None - no user-facing surface

## Sources

- `ROADMAP_CORRECAO_AUDITORIA.md` (Seção 3, `R02 — r02-finetune-lr-schedule-and-trainer-parity`) - define as regras de paridade de LR, scheduler, supressão de `UserWarning` e retomada de checkpoint.
- `src/viral_cnn/training/trainer.py` e `src/viral_cnn/training/config.py` - implementação de treino controlado e configuração de experimentos.
- `notebooks/reproduction/short/03_finetune_resnet50_species.ipynb` e `04_finetune_convnext_species.ipynb` - pipelines de fine-tuning para ResNet50 e ConvNeXt sob ruído.

