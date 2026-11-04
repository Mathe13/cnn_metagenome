# C05 checks

Profile: light
Plan: `.specs/features/c05-finetune-e-comparacao-controlada/plan.md`

## Checks

### S1 - Trilha mínima auditável · 6 arquivos · ~80 KB · ~20k

**C1** - Preparação empírica rejeita dados, mapeamento de classes ou checkpoint base ausentes antes de construir/atualizar o modelo, sem fallback sintético
Proof: `.venv/bin/python3 -m unittest tests.test_c05_finetune_audit -k test_empirical_preflight_requires_all_inputs`

**C2** - Manifesto C05 identifica protocolo, variante base, SHA-256 do checkpoint e das entradas, ordem/rank de classes, dropout efetivo/desconhecido e papéis last/best/selected
Proof: `.venv/bin/python3 -m unittest tests.test_c05_finetune_audit -k test_manifest_records_lineage_and_checkpoint_roles`

**C3** - Consolidação de protocolo declara scheduler por época, dropout divergente, regra de seleção e divergência histórica documentada sem inferir estado a partir do nome do peso
Proof: `.venv/bin/python3 -m unittest tests.test_c05_finetune_audit -k test_protocol_audit_reports_historical_divergences`

**C4** - A validação de fine-tuning usa `noise_rate=0.05`, e a injeção uniforme inclui a base original, registrando seed e amostras/contagens por época
Proof: `.venv/bin/python3 -m unittest tests.test_c05_finetune_audit -k test_train_and_validation_noise_are_audited`

**C5** - Avaliação pré-treino identifica clean e `epsilon=0.05`, amostra limitada e completa sem confundir seus suportes
Proof: `.venv/bin/python3 -m unittest tests.test_c05_finetune_audit -k test_pretraining_evaluation_labels_support_and_noise`

**C6** - Seleção `last` escolhe a última época e `best_val_macro_f1` escolhe a maior métrica, com empate pela primeira observação inclusive quando o valor inicial é zero
Proof: `.venv/bin/python3 -m unittest tests.test_c05_finetune_audit -k test_checkpoint_selection_last_best_and_zero_tie`

### S2 - Isolamento e retomada · 3 arquivos · ~60 KB · ~15k

**C7** - Checkpoint C05 grava atomicamente modelo, otimizador, scheduler, scaler, progresso, histórico, métrica best, configuração/dados e estados RNG de Python, NumPy, Torch CPU/CUDA e ruído
Proof: `.venv/bin/python3 -m unittest tests.test_c05_finetune_audit -k test_checkpoint_contains_full_versioned_state_and_rng`

**C8** - Retomada exige caminho explícito e identidade compatível e restaura os seis componentes `model`, `optimizer`, `scheduler`, `scaler`, `RNG`, `history`; qualquer incompatibilidade/falha encerra antes da atualização
Proof: `.venv/bin/python3 -m unittest tests.test_c05_finetune_audit -k test_resume_validates_identity_and_restores_required_state`
Proof: `.venv/bin/python3 -m unittest tests.test_c05_finetune_audit -k test_resume_rejects_incompatible_or_missing_state`

**C9** - Interrupção antes do rename mantém íntegro o checkpoint publicado anteriormente
Proof: `.venv/bin/python3 -m unittest tests.test_c05_finetune_audit -k test_atomic_checkpoint_preserves_previous_file_on_replace_failure`

### S3 - Avaliação e limites · 4 arquivos · ~55 KB · ~14k

**C10** - Notebooks 03/04 reservam diretório exclusivo por execução, carregam exatamente o checkpoint pré-validado, não retomam por existência de `last`, pedem resume explicitamente e registram inputs/checkpoints por caminho e hash
Proof: `.venv/bin/python3 -m unittest tests.test_c05_finetune_audit -k test_notebooks_isolate_runs_and_require_explicit_resume`

**C11** - Manifestos de notebooks distinguem protocolo C05, execução sintética, prévia limitada, execução parcial/falha e conclusão, com etapas reais, hashes e seleção comprovada
Proof: `.venv/bin/python3 -m unittest tests.test_c05_finetune_audit -k test_manifest_never_promotes_smoke_or_incomplete_run`

**C12** - Relatório C05 separa evidência mínima/execução única de comparação controlada `NOT_RUN` e declara indisponível qualquer conclusão causal
Proof: `.venv/bin/python3 -m unittest tests.test_c05_finetune_audit -k test_report_marks_optional_comparison_not_run_and_limits_claims`

**C13** - Avaliação final usa checkpoints congelados e split de teste identificado, sem usar teste na seleção; os dois fine-tunings declaram origens distintas e uma execução por configuração
Proof: `.venv/bin/python3 -m unittest tests.test_c05_finetune_audit -k test_report_separates_test_evaluation_from_selection_and_architecture_claims`

**C15** - Gerador de relatório sem manifestos controlados completos escreve `NOT_RUN`, custo `NOT_MEASURED` e nenhuma linha numérica de resultado/inferência experimental
Proof: `.venv/bin/python3 -m unittest tests.test_c05_finetune_audit -k test_generated_report_never_invents_controlled_results`

**C14** - `ParquetBatchIterableDataset` sem `include_identity` entrega batches de sequência/rótulo sem consultar metadados de partição inexistentes
Proof: `.venv/bin/python3 -m unittest tests.test_c05_finetune_audit -k test_parquet_loader_yields_without_identity_tracking`

### S4 - Extensão controlada P2 · NOT ACTIVATED

A trilha controlada não foi explicitamente ativada. C14–C20 (plan AC 19–25) permanecem fora do build e o estado exigido é `NOT_RUN`; loaders empíricos pareados, seis treinamentos completos, medição de recursos e comparação por semente não são declarados implementados por esta entrega.

## Coverage

| Set (size) | Member -> proof | Unproven |
| --- | --- | --- |
| Landing doors (3) | identidade/seleção C2,C6 · retomada/atomicidade C7-C9 · protocolo controlado condicional C12 | - |
| Relations (6) | execução/base/config C2,C3 · época/checkpoint C4,C6,C7 · seleção C6 · resume C8 · experimento opcional C12 · teste C13 | - |
| Observable surfaces (12) | preflight C1 · linhagem C2 · protocolo C3 · treino C4 · avaliação prévia C5 · seleção C6 · checkpoints C7,C9 · resume C8 · manifesto C11 · avaliação C13 · relatório C12 · relatório gerado C15 · trilha opcional C12 | - |
| Checkpoint state restored (6) | model C8 · optimizer C8 · scheduler C8 · scaler C8 · RNG C8 · history C8 | - |
| Notebook runs (2) | `03_finetune_resnet50_species.ipynb` C10 · `04_finetune_convnext_species.ipynb` C10 | - |
| Run evidence states (5) | smoke/synthetic C11 · preview C5,C11 · partial/failure C11 · complete C11 · optional NOT_RUN C12 | - |
| Parquet dataset mode (1) | identity disabled C14 | - |

- Nenhum check declara comparação controlada concluída; não há membros empíricos de sementes para a trilha opcional.
- Nenhuma rota HTTP: Surface = None.

## Swept

- validation: C1-C3,C8 (entradas empíricas, protocolo e identidade de retomada)
- failure modes: C1,C8,C9,C11 (ausência, incompatibilidade, falha de publicação e execução incompleta)
- idempotency: C10 (execução nova sempre reserva namespace exclusivo; resume só por escolha explícita)
- authorization: n/a - processamento local sem autenticação externa
- concurrency: C10 (namespace exclusivo); C9 (publicação atômica)
- data lifecycle: C2,C7,C10,C11 (hashes, estado persistido, saída isolada e evidência)
- dependency failure: C1,C8 (ausência/falha de dados, pesos ou restauração interrompe a trilha)
- state transitions: C6-C8,C11,C12 (last/best/selected; fresh/resume/failure; NOT_RUN/partial/complete)
- observability: C2-C5,C7,C10-C15 (caminhos, hashes, contagens, métricas, estado e limites)

## Handoff

- S1 ~20k + S2 ~15k + S3 ~14k = ~49k tokens estimados; arquivos inspecionados e artefatos de teste somam aproximadamente 230 KB / 4 ≈ 57.5k tokens, abaixo do budget light de 150k. Uma builder.
- Mecanismo: um builder; escopo opcional S4 não ativado.
- **Boundary:** implementação S1-S3 concluída; S4 permanece NOT ACTIVATED. Verificação independente pendente após commit.
- **Settled mid-build:** nenhum.
- **Abandoned:** nenhum.
