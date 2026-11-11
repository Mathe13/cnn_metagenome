# M05 checks

Profile: light
Plan: `.specs/features/m05-triplet-composition-and-cooccurrence/plan.md` — aprovado pelo usuário (“aprovado”).

17 checks em três slices; cinco portas; nenhuma questão bloqueante.

## Checks

### S1

**C1** — WHEN executa 08 THEN o fluxo SHALL resolver os caminhos de entrada e saída a partir do plano M00 ou variáveis de ambiente sem depender de diretórios fixos legados (`results_reproduction/motifs`). (AC 1).
Proof: `REMOTE_SERVER=False PYTHONPATH=src .venv/bin/python3 -m unittest tests.test_m05_triplets_and_cooccurrence.M05Tests.test_paths -k test_paths`

**C2** — WHEN `preflight_notebook("short/08", ...)` é executado no modo empírico (`SMOKE_TEST=0`) THEN o runtime SHALL validar a presença e legibilidade do manifesto de 07, do sumário de motivos (`motif_summary.parquet`) e das matrizes Jaccard das cinco variantes, reportando `READY` apenas se todos existirem e `BLOCKED` se qualquer um faltar ou estiver incompleto. (AC 2).
Proof: `REMOTE_SERVER=False PYTHONPATH=src .venv/bin/python3 -m unittest tests.test_m05_triplets_and_cooccurrence.M05Tests.test_preflight_08_inputs -k test_preflight_08_inputs`

**C3** — IF o diretório reservado de destino já contiver produtos ou colidir com run existente THEN o fluxo SHALL abortar antes de modificar qualquer arquivo. (AC 3).
Proof: `REMOTE_SERVER=False PYTHONPATH=src .venv/bin/python3 -m unittest tests.test_m05_triplets_and_cooccurrence.M05Tests.test_collision_rejection -k test_collision_rejection`

**C4** — WHEN uma execução de 08 falha por erro inesperado THEN o fluxo SHALL emitir manifesto atômico com status `FAILED` e detalhe da falha, nunca `COMPLETE`. (AC 4).
Proof: `REMOTE_SERVER=False PYTHONPATH=src .venv/bin/python3 -m unittest tests.test_m05_triplets_and_cooccurrence.M05Tests.test_failed_manifest -k test_failed_manifest`

**C5** — WHEN a etapa 08 conclui com sucesso THEN o fluxo SHALL publicar manifesto `COMPLETE` contendo protocolo `m05_triplets_audit_v1`, `is_smoke_test`, semente 42, hashes das entradas e lista completa de artefatos gerados. (AC 5).
Proof: `REMOTE_SERVER=False PYTHONPATH=src .venv/bin/python3 -m unittest tests.test_m05_triplets_and_cooccurrence.M05Tests.test_complete_manifest -k test_complete_manifest`

### S2

**C6** — WHEN amostra sequências de teste para cálculo composicional de tripletos THEN o extrator SHALL selecionar deterministicamente até 1000 fragmentos com semente 42 fixa e cobertura conhecida, sem depender de cabeçalhos arbitrários ou primeiras linhas não documentadas. (AC 6).
Proof: `REMOTE_SERVER=False PYTHONPATH=src .venv/bin/python3 -m unittest tests.test_m05_triplets_and_cooccurrence.M05Tests.test_deterministic_sampling -k test_deterministic_sampling`

**C7** — WHEN calcula a composição de tripletos e RSCU exploratório THEN o fluxo SHALL avaliar os 64 códons sob os modos `frame0`, `forward_3frames` e `six_frames` contra o modelo nulo multinomial de bases e exportar `table_09_exploratory_rscu.csv`. (AC 7).
Proof: `REMOTE_SERVER=False PYTHONPATH=src .venv/bin/python3 -m unittest tests.test_m05_triplets_and_cooccurrence.M05Tests.test_rscu_three_frames_export -k test_rscu_three_frames_export`

**C8** — WHEN renderiza o perfil de tripletos (Figura 4) THEN o renderizador SHALL exportar `figure_04_rscu_profile.png` contrastando as frequências de tripletos entre os modos de leitura com linha de referência unitária. (AC 8).
Proof: `REMOTE_SERVER=False PYTHONPATH=src .venv/bin/python3 -m unittest tests.test_m05_triplets_and_cooccurrence.M05Tests.test_figure_04_profile_rendering -k test_figure_04_profile_rendering`

**C9** — WHEN audita a ocorrência de sinais de início (ATG) e parada (TAA, TAG, TGA) em motivos de 7 bp THEN o fluxo SHALL confrontar contagens observadas contra probabilidades esperadas sob composição de bases e exportar `table_10_translation_boundary_signals.csv`. (AC 9).
Proof: `REMOTE_SERVER=False PYTHONPATH=src .venv/bin/python3 -m unittest tests.test_m05_triplets_and_cooccurrence.M05Tests.test_boundary_signals_audit -k test_boundary_signals_audit`

**C10** — WHEN executa 08 no modo empírico (`SMOKE_TEST=0`) THEN o fluxo SHALL rejeitar artefatos de 07 ou fixtures marcados como sintéticos (`is_smoke_test=true`). (AC 10).
Proof: `REMOTE_SERVER=False PYTHONPATH=src .venv/bin/python3 -m unittest tests.test_m05_triplets_and_cooccurrence.M05Tests.test_reject_synthetic_in_empirical -k test_reject_synthetic_in_empirical`

### S3

**C11** — WHEN analisa a coocorrência de motivos (Tabela 11) THEN o fluxo SHALL computar a correlação bidirecional ($r_{\text{bidirectional}}$), ICC simétrico e teste de permutação de nós preservando a topologia de arestas, exportando `table_11_cooccurrence_synergy.csv` com contagem explícita de `n_valid_edges` e $p_{\text{perm}}$. (AC 11).
Proof: `REMOTE_SERVER=False PYTHONPATH=src .venv/bin/python3 -m unittest tests.test_m05_triplets_and_cooccurrence.M05Tests.test_cooccurrence_symmetric_synergy -k test_cooccurrence_symmetric_synergy`

**C12** — The fluxo SHALL garantir que a estatística de associação de coocorrência seja invariante à orientação das arestas, produzindo $\Delta r_{\text{orientation}} = 0$ e $r_{\text{canonical}} = r_{\text{reversed}}$. (AC 12).
Proof: `REMOTE_SERVER=False PYTHONPATH=src .venv/bin/python3 -m unittest tests.test_m05_triplets_and_cooccurrence.M05Tests.test_orientation_invariance -k test_orientation_invariance`

**C13** — WHEN exporta a tabela de arestas da rede de coocorrência THEN o fluxo SHALL salvar `edge_table.csv` contendo colunas `x1`, `y1`, `x2`, `y2`, `motif_a`, `motif_b` e `jaccard` derivadas por posicionamento determinístico, compatível com o validador de publicação C06. (AC 13).
Proof: `REMOTE_SERVER=False PYTHONPATH=src .venv/bin/python3 -m unittest tests.test_m05_triplets_and_cooccurrence.M05Tests.test_edge_table_c06_compatibility -k test_edge_table_c06_compatibility`

**C14** — WHEN renderiza a Figura 5 THEN o renderizador SHALL exportar `figure_05_cooccurrence_synergy.png`, o gráfico de rede `figure_05_network.png` baseado em `edge_table.csv`, e a figura `figure_05_permutation_null.png` com a distribuição nula empírica e a estatística observada demarcada. (AC 14).
Proof: `REMOTE_SERVER=False PYTHONPATH=src .venv/bin/python3 -m unittest tests.test_m05_triplets_and_cooccurrence.M05Tests.test_figure_05_network_and_null_rendering -k test_figure_05_network_and_null_rendering`

**C15** — WHEN compara o conjunto de motivos e matrizes Jaccard entre variantes de modelos THEN o fluxo SHALL exportar `table_11b_jaccard_variant_diagnostics.csv` demarcando explicitamente a sobreposição descritiva sem atribuir correlações perfeitas ($r=1$) a concordância biológica independente. (AC 15).
Proof: `REMOTE_SERVER=False PYTHONPATH=src .venv/bin/python3 -m unittest tests.test_m05_triplets_and_cooccurrence.M05Tests.test_inter_variant_diagnostics -k test_inter_variant_diagnostics`

**C16** — WHEN documenta os escores de tripletos e coocorrência THEN o texto e os metadados SHALL registrar formalmente a advertência de composição exploratória, refutando inferência causal de tradução ou códons de ORFs ativas. (AC 16).
Proof: `REMOTE_SERVER=False PYTHONPATH=src .venv/bin/python3 -m unittest tests.test_m05_triplets_and_cooccurrence.M05Tests.test_exploratory_demarcation -k test_exploratory_demarcation`

**C17** — WHEN artefatos são lidos ou gravados nos fluxos M05 THEN o código SHALL emitir logs com prefixo `[LOG_PATH]` contendo o caminho do arquivo, respeitando `LOG_PATH=0/false`. (AC 17).
Proof: `REMOTE_SERVER=False PYTHONPATH=src .venv/bin/python3 -m unittest tests.test_m05_triplets_and_cooccurrence.M05Tests.test_logging_path -k test_logging_path`

## Coverage

| Set (size) | Member -> proof | Unproven |
| --- | --- | --- |
| Assemblies (1) | short/08 C5 | - |
| Manifest states (2) | COMPLETE C5 · FAILED C4 | - |
| Reading frames (3) | frame0 C7 · forward_3frames C7 · six_frames C7 | - |
| Preflight states (2) | READY C2 · BLOCKED C2 | - |
| Doors (5) | lifecycle C5 · 07-input-resolution C2 · deterministic-sampling C6 · network-edges-c06 C13 · descriptive-inter-variant C15 | - |
| Entities (5) | triplet_rscu C7 · boundary_signal C9 · cooccurrence_synergy C11 · edge_network C13 · inter_variant_diagnostic C15 | - |
| Figures (4) | figure_04_rscu_profile C8 · figure_05_cooccurrence_synergy C14 · figure_05_network C14 · figure_05_permutation_null C14 | - |

## Test policy

| Target | Policy |
| --- | --- |
| m05_workflow | unit: resolução de entradas, isolamento de saídas, amostragem determinística, captura de falhas e manifesto atômico |
| preflight_08 | unit: validação da presença física e legibilidade do manifesto e artefatos de 07 |
| triplet_composition | unit: cálculo de frequências e RSCU em 3 regimes contra modelo multinomial nulo de bases |
| cooccurrence_synergy | unit: correlação bidirecional, ICC simétrico, invariância de orientação e teste de permutação de nós |
| network_edge_table | unit: derivação de coordenadas x1/y1/x2/y2 e compatibilidade com C06 network_edges |
| notebook_08 | integration: execução de ponta a ponta sobre fixtures e exportação das tabelas 9, 10, 11, 11b e figuras 4 e 5 |

## Swept

- validation: C2, C6, C10
- failure modes: C3, C4
- idempotency: C1, C3, C5
- authorization: n/a — execução local sob permissões de arquivo do sistema operacional
- concurrency: C3, reserva atômica de diretório exclusivo impedindo colisões
- data lifecycle: C5, C7, C9, C11, C13, C15
- dependency failure: C2, C4, C10
- state transitions: C4, C5
- observability: C5, C7, C8, C9, C11, C13, C14, C15, C16, C17

## Handoff

S1–S3 = ~38k total, dentro do orçamento de 150k. Mecanismo: um builder.
