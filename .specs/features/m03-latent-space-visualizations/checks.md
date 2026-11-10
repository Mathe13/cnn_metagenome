# M03 checks

Profile: light
Plan: `.specs/features/m03-latent-space-visualizations/plan.md` — aprovado pelo usuário (“prossiga”).

16 checks em quatro slices; quatro portas; nenhuma questão bloqueante.

## Checks

### S1

**C1** — WHEN executa 06 ou 06b THEN o fluxo SHALL resolver os caminhos e parâmetros a partir do plano M00 ou variáveis de ambiente sem depender de diretórios absolutos fixos de usuários. (AC 1).
Proof: `REMOTE_SERVER=False PYTHONPATH=src .venv/bin/python3 -m unittest tests.test_m03_latent_space_visualizations.M03Tests.test_paths -k test_paths`

**C2** — WHEN `preflight_notebook("short/06b", ...)` é executado THEN o runtime SHALL validar os requisitos declarados da etapa 06b e reportar status `READY` ou `BLOCKED`. (AC 2).
Proof: `REMOTE_SERVER=False PYTHONPATH=src .venv/bin/python3 -m unittest tests.test_m03_latent_space_visualizations.M03Tests.test_preflight_06b -k test_preflight_06b`

**C3** — IF o diretório de destino reservado já contiver produtos ou houver colisão de run THEN o fluxo SHALL falhar antes de iniciar o processamento ou modificar os arquivos existentes. (AC 3).
Proof: `REMOTE_SERVER=False PYTHONPATH=src .venv/bin/python3 -m unittest tests.test_m03_latent_space_visualizations.M03Tests.test_collision_rejection -k test_collision_rejection`

**C4** — WHEN uma execução 06 ou 06b falha por exceção THEN o fluxo SHALL registrar manifesto atômico com status `FAILED` e causa do erro, nunca `COMPLETE`. (AC 4).
Proof: `REMOTE_SERVER=False PYTHONPATH=src .venv/bin/python3 -m unittest tests.test_m03_latent_space_visualizations.M03Tests.test_failed_manifest -k test_failed_manifest`

**C5** — WHEN 06 ou 06b conclui com sucesso THEN o fluxo SHALL publicar manifesto individual `COMPLETE` contendo protocolo `m03_latent_audit_v1`, `is_smoke_test`, semente 42, hashes das entradas e produtos. (AC 5).
Proof: `REMOTE_SERVER=False PYTHONPATH=src .venv/bin/python3 -m unittest tests.test_m03_latent_space_visualizations.M03Tests.test_complete_manifest -k test_complete_manifest`

### S2

**C6** — WHEN executa 06 no modo `reuse` THEN o fluxo SHALL validar a integridade dos embeddings salvos através de SHA-256 e identidade C09 antes de prosseguir. (AC 6).
Proof: `REMOTE_SERVER=False PYTHONPATH=src .venv/bin/python3 -m unittest tests.test_m03_latent_space_visualizations.M03Tests.test_c09_reuse_validation -k test_c09_reuse_validation`

**C7** — WHEN executa 06 no modo empírico (`SMOKE_TEST=0`) THEN o fluxo SHALL rejeitar dependências ou embeddings marcados como sintéticos (`is_smoke_test=true`). (AC 7).
Proof: `REMOTE_SERVER=False PYTHONPATH=src .venv/bin/python3 -m unittest tests.test_m03_latent_space_visualizations.M03Tests.test_reject_synthetic_in_empirical -k test_reject_synthetic_in_empirical`

**C8** — WHEN calcula KNN Cosseno ($k=5$), Silhouette Score e Sonda Linear THEN o código SHALL computar todas as métricas no espaço original de alta dimensão pré-classificador, sem redução prévia por UMAP ou PCA. (AC 8).
Proof: `REMOTE_SERVER=False PYTHONPATH=src .venv/bin/python3 -m unittest tests.test_m03_latent_space_visualizations.M03Tests.test_original_space_metrics -k test_original_space_metrics`

**C9** — WHEN publica as métricas quantitativas de alta dimensão THEN o fluxo SHALL gerar `latent_quantitative_summary.csv` e `latent_quantitative_summary.png` confrontando os cinco modelos com indicação explícita de $N$ e cobertura de classes válidas. (AC 9).
Proof: `REMOTE_SERVER=False PYTHONPATH=src .venv/bin/python3 -m unittest tests.test_m03_latent_space_visualizations.M03Tests.test_quantitative_summary_artifacts -k test_quantitative_summary_artifacts`

### S3

**C10** — WHEN seleciona fragmentos para projeção 2D THEN o fluxo SHALL utilizar o protocolo determinístico pareado C10, garantindo que os cinco modelos recebam exatamente as mesmas sequências (`row_id`), espécies e gêneros. (AC 10).
Proof: `REMOTE_SERVER=False PYTHONPATH=src .venv/bin/python3 -m unittest tests.test_m03_latent_space_visualizations.M03Tests.test_paired_selection_identity -k test_paired_selection_identity`

**C11** — WHEN exporta a seleção pareada THEN o fluxo SHALL salvar `selected_genera.csv` e `sampled_query_ids.csv` contendo a lista completa de gêneros, contagens, status de destaque e identificadores de sequência. (AC 11).
Proof: `REMOTE_SERVER=False PYTHONPATH=src .venv/bin/python3 -m unittest tests.test_m03_latent_space_visualizations.M03Tests.test_exported_selection_csvs -k test_exported_selection_csvs`

**C12** — WHEN renderiza o painel seletivo `umap_selective_panel.png` THEN o renderizador SHALL colorir os gêneros em destaque com a paleta contrastante e desenhar todos os demais fragmentos da amostra em cinza neutro ao fundo com opacidade atenuada. (AC 12).
Proof: `REMOTE_SERVER=False PYTHONPATH=src .venv/bin/python3 -m unittest tests.test_m03_latent_space_visualizations.M03Tests.test_selective_panel_rendering -k test_selective_panel_rendering`

**C13** — WHEN renderiza qualquer visualização UMAP ou PCA THEN o gráfico e o arquivo gerado SHALL incluir aviso proeminente de que a representação é estritamente qualitativa e que os eixos são independentes e não-filogenéticos. (AC 13).
Proof: `REMOTE_SERVER=False PYTHONPATH=src .venv/bin/python3 -m unittest tests.test_m03_latent_space_visualizations.M03Tests.test_qualitative_disclaimers -k test_qualitative_disclaimers`

**C14** — WHEN executa 06b a partir de embeddings ou coordenadas salvas de 06 THEN o notebook 06b SHALL regenerar tanto o painel completo quanto o painel seletivo e o HTML interativo preservando estritamente as coordenadas originais. (AC 14).
Proof: `REMOTE_SERVER=False PYTHONPATH=src .venv/bin/python3 -m unittest tests.test_m03_latent_space_visualizations.M03Tests.test_06b_regeneration_preserves_coordinates -k test_06b_regeneration_preserves_coordinates`

### S4

**C15** — WHEN artefatos persistentes são lidos ou gravados nos fluxos M03 THEN o código SHALL emitir logs com prefixo `[LOG_PATH]` contendo o caminho do arquivo, respeitando `LOG_PATH=0/false`. (AC 15).
Proof: `REMOTE_SERVER=False PYTHONPATH=src .venv/bin/python3 -m unittest tests.test_m03_latent_space_visualizations.M03Tests.test_logging_path -k test_logging_path`

**C16** — WHEN documenta M03 THEN a documentação SHALL explicitar comandos de execução para 06 e 06b, parâmetros suportados, distinção entre smoke local e reprodução empírica no cluster, e demarcação de que projeções UMAP 2D não medem acurácia nem proximidade filogenética. (AC 16).
Proof: `REMOTE_SERVER=False PYTHONPATH=src .venv/bin/python3 -m unittest tests.test_m03_latent_space_visualizations.M03Tests.test_documentation_and_limits -k test_documentation_and_limits`

## Coverage

| Set (size) | Member -> proof | Unproven |
| --- | --- | --- |
| Assemblies (2) | short/06 C5 · short/06b C14 | - |
| Manifest states (2) | COMPLETE C5 · FAILED C4 | - |
| Execution modes (2) | reuse C6 · rerun C5 | - |
| Metric scopes (2) | high_dimensional C8 · qualitative_2d C10 | - |
| Panels (2) | full_overview C14 · selective_highlight C12 | - |
| Preflight states (2) | READY C2 · BLOCKED C2 | - |
| Doors (4) | lifecycle C5 · c09-validation C6 · quantitative-summary C9 · selective-panel C12 | - |
| Entities (4) | model C8 · embedding C6 · selection C11 · projection C10 | - |

## Test policy

| Target | Policy |
| --- | --- |
| m03_workflow | unit: resolução de entradas, reservas atômicas, captura de falhas e metadados |
| preflight_06b | unit: inspeção de requisitos declarados e status READY/BLOCKED |
| high_dim_metrics | unit: conformidade de KNN Cosseno, Silhouette e Sonda Linear em alta dimensão |
| selective_panel | unit: integridade de seleção determinística, fundo neutro e exportação de CSVs |
| notebooks_06_06b | integration: execução de ponta a ponta sobre fixtures e preservação de coordenadas |

## Swept

- validation: C2, C6, C7
- failure modes: C3, C4
- idempotency: C1, C3, C10
- authorization: n/a — execução local sob permissões do sistema
- concurrency: C3, reserva exclusiva com claim de execução atômico
- data lifecycle: C5, C6, C10, C11
- dependency failure: C2, C4, C7
- state transitions: C4, C5, C14
- observability: C5, C9, C11, C13, C15, C16

## Handoff

Um builder. Arquivos existentes (`notebook_runtime.py` ~40 KB, `paired_umap.py` ~14 KB, `latent_plotting.py` ~10 KB, `short/06` ~25 KB, `short/06b` ~15 KB) somam ~104 KB / 4 = ~26.000 tokens; módulos novos (`m03_workflow.py` ~15 KB, `configs/reproduction_m03.yaml` ~4 KB, `test_m03_latent_space_visualizations.py` ~25 KB, docs ~15 KB) somam ~59 KB / 4 = ~15.000 tokens; plano/checks ~15.000 tokens. Total estimado ~56.000 tokens < orçamento padrão de 150.000 tokens. Sem necessidade de handoff. Verifier independente será disparado após o último commit da feature. Perfil light. Testes sintéticos executam em CPU offline e não substituem preflight nem execução empírica real no cluster.
