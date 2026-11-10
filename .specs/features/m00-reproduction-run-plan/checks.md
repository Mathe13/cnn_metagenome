# M00 checks

Profile: light
Plan: `.specs/features/m00-reproduction-run-plan/plan.md` — aprovado pelo usuário (“prossiga”).

18 checks, três slices, duas portas; nenhuma questão bloqueante.

## Checks

### S1

**C1** — Inventário ordenado e deduplicado registra caminhos, tamanhos e estados (AC 1; door 1).
Proof: `REMOTE_SERVER=False PYTHONPATH=src .venv/bin/python3 -m unittest tests.test_m00_run_plan.RunPlanTests.test_inventory_records -k test_inventory_records`

**C2** — SHA-256 integral é registrado até o limite configurado (AC 2).
Proof: `REMOTE_SERVER=False PYTHONPATH=src .venv/bin/python3 -m unittest tests.test_m00_run_plan.RunPlanTests.test_inventory_hash_boundary -k test_inventory_hash_boundary`

**C3** — Arquivo grande/ausente/ilegível e symlink são explícitos, sem cópia nem travessia externa (AC 3).
Proof: `REMOTE_SERVER=False PYTHONPATH=src .venv/bin/python3 -m unittest tests.test_m00_run_plan.RunPlanTests.test_inventory_exceptions -k test_inventory_exceptions`

**C4** — Duas rodadas reservam destinos distintos e preservam fontes (AC 4; door 2).
Proof: `REMOTE_SERVER=False PYTHONPATH=src .venv/bin/python3 -m unittest tests.test_m00_run_plan.RunPlanTests.test_exclusive_rounds -k test_exclusive_rounds`

**C5** — Colisão, concorrência e traversal são recusados preservando sentinelas (AC 5).
Proof: `REMOTE_SERVER=False PYTHONPATH=src .venv/bin/python3 -m unittest tests.test_m00_run_plan.RunPlanTests.test_reservation_rejection -k test_reservation_rejection`

### S2

**C6** — Plano persiste identidade, modo, config/hash e recursos/unknown (AC 6).
Proof: `REMOTE_SERVER=False PYTHONPATH=src .venv/bin/python3 -m unittest tests.test_m00_run_plan.RunPlanTests.test_plan_identity -k test_plan_identity`

**C7** — Configuração padrão inclui 16 notebooks e skip justificado de 03/04 (AC 7).
Proof: `REMOTE_SERVER=False PYTHONPATH=src .venv/bin/python3 -m unittest tests.test_m00_run_plan.RunPlanTests.test_default_stages -k test_default_stages`

**C8** — Plano fixa manifestos das quatro cadeias exigidas, sem descoberta por data (AC 8).
Proof: `REMOTE_SERVER=False PYTHONPATH=src .venv/bin/python3 -m unittest tests.test_m00_run_plan.RunPlanTests.test_dependency_paths -k test_dependency_paths`

**C9** — Configuração com seleção ausente/duplicata/ciclo/dependência desconhecida retorna 2 sem plano válido (AC 9).
Proof: `REMOTE_SERVER=False PYTHONPATH=src .venv/bin/python3 -m unittest tests.test_m00_run_plan.RunPlanTests.test_invalid_configuration_cli -k test_invalid_configuration_cli`

**C10** — Manifesto externo é identificado por caminho/hash e permanece somente leitura (AC 10).
Proof: `REMOTE_SERVER=False PYTHONPATH=src .venv/bin/python3 -m unittest tests.test_m00_run_plan.RunPlanTests.test_external_identity -k test_external_identity`

**C11** — Retomada 06/06b tem nova reserva e validação C09 pendente (AC 11).
Proof: `REMOTE_SERVER=False PYTHONPATH=src .venv/bin/python3 -m unittest tests.test_m00_run_plan.RunPlanTests.test_resume_pending -k test_resume_pending`

### S3

**C12** — Preflight enumera requisitos/esquemas e inclui 06b (AC 12).
Proof: `REMOTE_SERVER=False PYTHONPATH=src .venv/bin/python3 -m unittest tests.test_m00_run_plan.RunPlanTests.test_metadata_requirements -k test_metadata_requirements`

**C13** — Dependência planejada/incompleta/sintética/adulterada bloqueia etapa empírica (AC 13).
Proof: `REMOTE_SERVER=False PYTHONPATH=src .venv/bin/python3 -m unittest tests.test_m00_run_plan.RunPlanTests.test_dependency_gate -k test_dependency_gate`

**C14** — Inspeção empírica não cria fixtures nem lê sequências/pesos nem executa notebooks (AC 14).
Proof: `REMOTE_SERVER=False PYTHONPATH=src .venv/bin/python3 -m unittest tests.test_m00_run_plan.RunPlanTests.test_metadata_only -k test_metadata_only`

**C15** — Falha de escrita não publica plano parcial como pronto (AC 15).
Proof: `REMOTE_SERVER=False PYTHONPATH=src .venv/bin/python3 -m unittest tests.test_m00_run_plan.RunPlanTests.test_atomic_failure -k test_atomic_failure`

**C16** — Skip é SKIPPED e não cria resultado científico (AC 16).
Proof: `REMOTE_SERVER=False PYTHONPATH=src .venv/bin/python3 -m unittest tests.test_m00_run_plan.RunPlanTests.test_skip_no_products -k test_skip_no_products`

**C17** — CLI preflight com/sem plano conserva opções e códigos 0/2 (AC 17).
Proof: `REMOTE_SERVER=False PYTHONPATH=src .venv/bin/python3 -m unittest tests.test_m00_run_plan.RunPlanTests.test_preflight_cli -k test_preflight_cli`

**C18** — README fornece comandos e distingue local/remoto/empírico (AC 18).
Proof: `REMOTE_SERVER=False PYTHONPATH=src .venv/bin/python3 -m unittest tests.test_m00_run_plan.RunPlanTests.test_documented_limits -k test_documented_limits`

## Coverage

| Set (size) | Member -> proof | Unproven |
| --- | --- | --- |
| Inventário (5) | hashed C2 · skipped_size C3 · missing C3 · unreadable C3 · symlink C3 | - |
| Etapas (16) | C7, table-driven over all 16 | - |
| Cadeias (4) | 07→08 C8 · 06→06b C8 · baseline→09 C8 · long01→02→03→04 C8 | - |
| Erros DAG (4) | seleção-ausente C9 · duplicata C9 · ciclo C9 · desconhecida C9 | - |
| Dependências (5) | COMPLETE C10 · planned C13 · incomplete C13 · synthetic C13 · hash-mismatch C13 | - |
| Status (3) | READY C17 · BLOCKED C13 · SKIPPED C16 | - |
| Portas (2) | inventário C1 · plano-versionado C6 | - |
| Entidades (4) | rodada C4 · inventário C1 · etapa C7 · dependência C8 | - |
| Assemblies (2) | planner-CLI C9 · preflight-CLI C17 | - |

## Swept

- validation: C5, C9, C12
- failure modes: C3, C15
- idempotency: C4, C5
- authorization: n/a — CLIs locais sob permissões do sistema
- concurrency: C5
- data lifecycle: C1, C4, C10
- dependency failure: C12, C13, C14
- state transitions: C13, C15, C16
- observability: C1, C6, C12, C18

## Handoff

Um builder. Arquivos existentes runtime/CLI/teste C00/README = 71.060 bytes / 4 = 17.765 tokens; auxiliares, configuração e testes estimados em 64.000 bytes / 4 = 16.000 tokens; leitura adicional da skill/plano ~20.000 tokens. Total ~53.765 < orçamento padrão 150.000. Sem handoff de implementação. Verifier independente após o último commit de código, perfil light. Preflight remoto e execução empírica não são substituídos pelas provas locais.

- **Boundary:** C1–C18 implementados e 31 testes M00/C00/runtime verdes; 60 testes de regressão C02/C06/C09 verdes.
- **Settled mid-build:** plano administrativo não altera os produtores; 06b é aceito pela CLI com plano sem mudar o registro legado de 15 notebooks. A configuração padrão enumera explicitamente as 16 etapas.
- **Abandoned:** nenhuma execução empírica, treinamento, migração de manifestos antigos ou ajuste dos resultados históricos.

- **Review round 1:** Verifier no commit `e560655` comprovou 15/18 checks e apontou falta dos casos 06b e READY legado, além de BadZipFile não tratado. As provas C11/C12/C17 foram ampliadas sem remover asserções ou alterar claims; captura de BadZipFile mantém a enumeração de requisitos. Reverificação independente pendente após commit do reparo.
