# M05 — Composição Exploratória de Tripletos e Coocorrência (08)

## Problem

O notebook de reprodução `short/08_triplet_composition_and_cooccurrence.ipynb` grava suas saídas diretamente em um diretório fixo e compartilhado (`results_reproduction/motifs`), violando o isolamento de execuções imutáveis e o ciclo de vida M00 (`m00_run_plan_v1`). Em `notebook_runtime.py:NOTEBOOK_REQUIREMENTS`, a etapa `short/08` aponta para um caminho estático legado (`results_reproduction/motifs/manifest_07_motifs.json`), impedindo que o fluxo consuma dinamicamente a saída selecionada de 07 (M04) e gerando preflights que falham silenciosamente ou aceitam artefatos de execuções desconectadas.

Cientificamente, o cálculo da composição de tripletos e RSCU no notebook 08 seleciona as primeiras 1.000 sequências lidas da pasta de Parquet (`df_test.head(1000)`), uma amostragem não determinística e sensível à ordenação física dos arquivos no filesystem. Os fragmentos virais de 512 bp avaliados não contêm anotação biológica de fase de leitura (CDS/ORF), de modo que a contagem em `frame0` constitui uma propriedade composicional local de bases e não tradução biológica, uso funcional de códons ou mecanismo causal de iniciação (ATG) e parada (STOP).

Na coocorrência de motivos (Tabela 11 e Figura 5), a versão histórica sofria de duas divergências críticas:
1. Uma alegação espúria de correlação entre arquiteturas ($r=1.0, p=0.0$), que decorria trivialmente da comparação de matrizes de Jaccard calculadas sobre as mesmíssimas sequências de entrada para motivos compartilhados, confundindo invariância de entrada com concordância independente entre modelos;
2. A Tabela 11 do notebook computa sinergia intra-modelo de escores de tripletos (`table_11_cooccurrence_synergy.csv`), mas a prévia observada no cluster revelou apenas 37 arestas válidas ($r \approx 0.12, p \approx 0.47$), refutando qualquer hipótese de sinergia estatisticamente significativa. Além disso, o validador de publicação C06 (`publication_c06.yaml`) exige para a Figura 5 o tipo `network_edges` com colunas de coordenadas (`x1`, `y1`, `x2`, `y2`), enquanto o notebook 08 gerava apenas um gráfico de barras estático.

Esta entrega implementa a fase M05 do roadmap de 09/11/2026, isolando o notebook 08 sob o protocolo `m05_triplets_audit_v1`, consumindo os artefatos verificados de 07 (M04), introduzindo amostragem determinística de sequências, reconciliando T9, T10, T11, F4 e F5 com C06, e demarcando formalmente a natureza exploratória dos tripletos.

## Flow

Reutilizar o runtime M00/M01/M04, o manifesto atômico e as matrizes de Jaccard geradas por 07, o extrator de RSCU e o auditor de sinais de fronteira de `interpretability.motifs`, sem alterar pesos de modelos nem inventar ORFs não verificadas.

1. Ambiente, configuração e plano M00 entram em `m05_workflow` (new), que reutiliza `notebook_runtime` e `run_planning` (exists): resolver entradas explícitas, parâmetros de controle, semente 42 e reserva de diretório exclusivo para 08 (door 1).
2. Manifesto e matrizes Jaccard de 07 e Parquets de teste entram em `preflight_notebook("short/08", ...)` (exists/updated): validar a presença física, integridade e permissão de leitura dos artefatos de 07 (`manifest.json` / `manifest_07_motifs.json`, `motif_summary.parquet`, e matrizes `motif_interaction_jaccard_<variant>.parquet`) e do diretório de teste (door 2).
3. Sequências de teste entram no amostrador determinístico em `m05_workflow` / `notebook_runtime` (exists/updated): selecionar deterministicamente até 1.000 sequências com semente 42 fixa e registrar contagens e cobertura (door 3).
4. Sequências amostradas entram em `compute_exploratory_triplet_rscu` (exists): avaliar frequências de tripletos e RSCU nos regimes `frame0`, `forward_3frames` e `six_frames` contra o nulo multinomial de bases; exportar `table_09_exploratory_rscu.csv` e `figure_04_rscu_profile.png` (door 3).
5. Motivos de 7 bp de 07 e frequências de bases entram em `audit_translation_boundary_signals` (exists): quantificar enriquecimento de ATG e STOP (TAA, TAG, TGA) contra modelo nulo de composição; exportar `table_10_translation_boundary_signals.csv` com advertência de ausência de ORF biológica (door 3).
6. Matriz Jaccard de ResNet50 Species e escores de tripletos entram em `analyze_cooccurrence_pairs` (exists): calcular correlação bidirecional ($r_{\text{bidirectional}}$), ICC simétrico e teste de permutação de nós; gerar `table_11_cooccurrence_synergy.csv` e figura nula de permutação `figure_05_permutation_null.png` (door 4).
7. Pares de coocorrência entram no layout de rede determinístico (new/updated): exportar `edge_table.csv` com coordenadas (`x1`, `y1`, `x2`, `y2`) compatível com C06 `network_edges`, renderizar `figure_05_network.png` e `figure_05_cooccurrence_synergy.png` (door 4).
8. Matrizes Jaccard das cinco variantes entram no comparador descritivo inter-modelos (exists/updated): quantificar sobreposição de motivos e identidade fora da diagonal; exportar `table_11b_jaccard_variant_diagnostics.csv` rotulado como verificação descritiva sem falso teste de hipótese inferencial (door 5).
9. Runtime grava manifesto individual atômico `manifest.json` com protocolo `m05_triplets_audit_v1`, status `COMPLETE` ou `FAILED`, hashes das entradas de 07 e sequências, e vínculo M00 (door 1).

## Impact

| Front | What changes |
| --- | --- |
| arquivos previstos | Notebook `short/08_triplet_composition_and_cooccurrence.ipynb`; novos módulos `src/viral_cnn/m05_workflow.py` e configuração `configs/reproduction_m05.yaml`; atualizações em `notebook_runtime.py`, `interpretability/motifs.py`, `reporting/figures.py` e `reporting/tables.py`; testes em `tests/test_m05_triplets_and_cooccurrence.py`. |
| saídas | Cada execução escreve em subdiretório exclusivo `results_reproduction/runs/<run_id>/short/08/`. Nenhuma pasta histórica legada (`results_reproduction/motifs`) é sobrescrita. |
| números afetados | Tabelas 9, 10, 11 e 11b passam a conter dados auditados com rastreabilidade total a partir do 07 selecionado; Figura 4 reflete RSCU multiquadro; Figura 5 disponibiliza rede com coordenadas C06 e distribuição nula de permutação. |
| cadeia científica | Saídas de 07 (M04) + split de teste → amostragem determinística → RSCU exploratório (3 regimes) + Tabela 9 + Figura 4 → fronteiras ATG/STOP + Tabela 10 → coocorrência de Jaccard + teste simétrico de permutação de nós + Tabela 11 + `edge_table.csv` + Figura 5 (rede e distribuição nula) → diagnóstico descritivo inter-modelos + Tabela 11b. |
| compatibilidade | O validador de publicação C06 consome `table_09.csv`, `table_10.csv`, `table_11.csv`, `fig_4.png` e `fig_5.png` via `edge_table.csv` sem erros de esquema de colunas. |
| demarcação biológica | Escores de tripletos, RSCU e coocorrência de motivos são explicitamente documentados e rotulados como composição exploratória de nucleotídeos, refutando hipóteses causais de tradução biológica, códons de ORFs ativas ou fatores de transcrição. |

## Relations

`None - no stored-data shape change`

## Surface

None - nenhuma API HTTP ou serviço de rede. Interfaces locais dos notebooks e CLI:

- `short/08_triplet_composition_and_cooccurrence.ipynb`: consome variáveis de ambiente ou configuração para `DATA_ROOT`, `SMOKE_TEST`, plano M00 opcional, semente (padrão 42), e manifesto/diretório de 07. Produz `manifest.json`, `table_09_exploratory_rscu.csv`, `table_10_translation_boundary_signals.csv`, `table_11_cooccurrence_synergy.csv`, `table_11b_jaccard_variant_diagnostics.csv`, `edge_table.csv`, `figure_04_rscu_profile.png`, `figure_05_cooccurrence_synergy.png`, `figure_05_network.png`, e `figure_05_permutation_null.png`. Status de manifesto: `COMPLETE` ou `FAILED`.
- `preflight_notebook("short/08", ...)`: inspeção de preflight em `notebook_runtime.py`, validando a presença física e legibilidade dos artefatos de 07 e do split de sequências antes de iniciar o processamento.
- `configs/reproduction_m05.yaml`: configuração YAML parametrizando o protocolo `m05_triplets_audit_v1`, regimes de leitura de tripletos, sementes, repetições de permutação (padrão 1000) e regras de coocorrência.

Precedência de resolução: (1) plano M00 explícito quando fornecido; (2) variáveis de ambiente explícitas; (3) arquivo de configuração `configs/reproduction_m05.yaml`; (4) topologia padrão do cluster / diretório de dados.

## Landing

| One-way door | Literal shape | Alternative rejected |
| --- | --- | --- |
| 1 — ciclo de vida e adoção M00 em 08 | `protocol_id=m05_triplets_audit_v1`, `notebook_id=short/08`, `run_id`, `is_smoke_test`, semente 42, `git_commit`, hashes dos artefatos de 07 e produtos. Reserva atômica via `reserve_run_directory`. | Usar pasta estática `results_reproduction/motifs`: sobrescreve execuções e perde rastreabilidade. |
| 2 — resolução estrita de dependências de 07 | Exigir em `NOTEBOOK_REQUIREMENTS["short/08"]` e no preflight empírico o manifesto de 07 (`manifest.json` ou `manifest_07_motifs.json`), `motif_summary.parquet` e as cinco matrizes `motif_interaction_jaccard_<variant>.parquet`. | Permitir fallback para pasta vazia ou preflight cego: gera erro no meio do notebook após minutos de execução. |
| 3 — amostragem determinística de sequências de teste | Amostragem determinística de até 1000 fragmentos do split de teste com semente 42 fixa e registro de integridade, suportando avaliação de 64 códons em `frame0`, `forward_3frames` e `six_frames`. | Ler as primeiras 1.000 linhas (`head(1000)`): sensível à ordem física dos arquivos nos workers e sem representatividade documentada. |
| 4 — Tabela 11 e F5 com rede, nulo por permutação e tabela de arestas C06 | Exportar `table_11_cooccurrence_synergy.csv` com $r_{\text{bidirectional}}$, ICC e $p_{\text{perm}}$, e `edge_table.csv` com colunas `x1, y1, x2, y2, motif_a, motif_b, jaccard` satisfazendo o tipo `network_edges` de C06, gerando `figure_05_permutation_null.png`. | Manter apenas barras sem arestas `network_edges` e sem distribuição nula: falha a montagem C06 e oculta que $p \approx 0.47$ não sustenta sinergia biológica. |
| 5 — diagnóstico descritivo inter-modelos sem falso teste inferencial | `table_11b_jaccard_variant_diagnostics.csv` registrando `n_common_motifs`, `motif_set_jaccard`, `n_positive_pairs_first`, `n_positive_pairs_second`, `off_diagonal_identical`, `status=DESCRIPTIVE_SHARED_INPUT_CHECK`. | Calcular $r=1$ e $p=0$ em matrizes idênticas geradas pelas mesmas sequências de entrada: alegação estatisticamente falsa e metodologicamente inválida. |

## Criteria

### S1: Isolamento de execução, adoção M00 e resolução de insumos de 07 (P1)

**Acceptance Criteria**

1. WHEN executa 08 THEN o fluxo SHALL resolver os caminhos de entrada e saída a partir do plano M00 ou variáveis de ambiente sem depender de diretórios fixos legados (`results_reproduction/motifs`).
2. WHEN `preflight_notebook("short/08", ...)` é executado no modo empírico (`SMOKE_TEST=0`) THEN o runtime SHALL validar a presença e legibilidade do manifesto de 07, do sumário de motivos (`motif_summary.parquet`) e das matrizes Jaccard das cinco variantes, reportando `READY` apenas se todos existirem e `BLOCKED` se qualquer um faltar ou estiver incompleto.
3. IF o diretório reservado de destino já contiver produtos ou colidir com run existente THEN o fluxo SHALL abortar antes de modificar qualquer arquivo.
4. WHEN uma execução de 08 falha por erro inesperado THEN o fluxo SHALL emitir manifesto atômico com status `FAILED` e detalhe da falha, nunca `COMPLETE`.
5. WHEN a etapa 08 conclui com sucesso THEN o fluxo SHALL publicar manifesto `COMPLETE` contendo protocolo `m05_triplets_audit_v1`, `is_smoke_test`, semente 42, hashes das entradas e lista completa de artefatos gerados.

**Independent test:** executar preflight com insumos completos de 07 e com artefato faltante; testar aborto por colisão de diretório e persistência de manifesto FAILED sob exceção induzida.

### S2: Amostragem determinística, composição de tripletos e sinais ATG/STOP (P1)

**Acceptance Criteria**

6. WHEN amostra sequências de teste para cálculo composicional de tripletos THEN o extrator SHALL selecionar deterministicamente até 1000 fragmentos com semente 42 fixa e cobertura conhecida, sem depender de cabeçalhos arbitrários ou primeiras linhas não documentadas.
7. WHEN calcula a composição de tripletos e RSCU exploratório THEN o fluxo SHALL avaliar os 64 códons sob os modos `frame0`, `forward_3frames` e `six_frames` contra o modelo nulo multinomial de bases e exportar `table_09_exploratory_rscu.csv`.
8. WHEN renderiza o perfil de tripletos (Figura 4) THEN o renderizador SHALL exportar `figure_04_rscu_profile.png` contrastando as frequências de tripletos entre os modos de leitura com linha de referência unitária.
9. WHEN audita a ocorrência de sinais de início (ATG) e parada (TAA, TAG, TGA) em motivos de 7 bp THEN o fluxo SHALL confrontar contagens observadas contra probabilidades esperadas sob composição de bases e exportar `table_10_translation_boundary_signals.csv`.
10. WHEN executa 08 no modo empírico (`SMOKE_TEST=0`) THEN o fluxo SHALL rejeitar artefatos de 07 ou fixtures marcados como sintéticos (`is_smoke_test=true`).

**Independent test:** verificar amostragem determinística repetida sob semente 42; checar cálculo de RSCU nos 3 regimes contra nulo multinomial; validar contagens e enriquecimento de ATG/STOP em motivos conhecidos; testar rejeição de fixture sintética em modo empírico.

### S3: Coocorrência, rede, nulo por permutação e demarcação de evidência (P1)

**Acceptance Criteria**

11. WHEN analisa a coocorrência de motivos (Tabela 11) THEN o fluxo SHALL computar a correlação bidirecional ($r_{\text{bidirectional}}$), ICC simétrico e teste de permutação de nós preservando a topologia de arestas, exportando `table_11_cooccurrence_synergy.csv` com contagem explícita de `n_valid_edges` e $p_{\text{perm}}$.
12. The fluxo SHALL garantir que a estatística de associação de coocorrência seja invariante à orientação das arestas, produzindo $\Delta r_{\text{orientation}} = 0$ e $r_{\text{canonical}} = r_{\text{reversed}}$.
13. WHEN exporta a tabela de arestas da rede de coocorrência THEN o fluxo SHALL salvar `edge_table.csv` contendo colunas `x1`, `y1`, `x2`, `y2`, `motif_a`, `motif_b` e `jaccard` derivadas por posicionamento determinístico, compatível com o validador de publicação C06.
14. WHEN renderiza a Figura 5 THEN o renderizador SHALL exportar `figure_05_cooccurrence_synergy.png`, o gráfico de rede `figure_05_network.png` baseado em `edge_table.csv`, e a figura `figure_05_permutation_null.png` com a distribuição nula empírica e a estatística observada demarcada.
15. WHEN compara o conjunto de motivos e matrizes Jaccard entre variantes de modelos THEN o fluxo SHALL exportar `table_11b_jaccard_variant_diagnostics.csv` demarcando explicitamente a sobreposição descritiva sem atribuir correlações perfeitas ($r=1$) a concordância biológica independente.
16. WHEN documenta os escores de tripletos e coocorrência THEN o texto e os metadados SHALL registrar formalmente a advertência de composição exploratória, refutando inferência causal de tradução ou códons de ORFs ativas.
17. WHEN artefatos são lidos ou gravados nos fluxos M05 THEN o código SHALL emitir logs com prefixo `[LOG_PATH]` contendo o caminho do arquivo, respeitando `LOG_PATH=0/false`.

**Independent test:** verificar simetria e invariância à orientação das arestas; checar esquema e coordenadas de `edge_table.csv`; testar geração de `figure_05_network.png` e `figure_05_permutation_null.png`; validar presença de `table_11b_jaccard_variant_diagnostics.csv`; verificar emissão de logs `[LOG_PATH]`.

## Out of scope

Product capabilities only. Process and harness rules live in AGENTS.md or as Observable `n/a`.

| Excluded | Why |
| --- | --- |
| Predição ou anotação de ORFs biológicas reais | Fragmentos virais de 512 bp não possuem quadro aberto de leitura verificado; tripletos são estritamente exploratórios e composicionais. |
| Treinamento de modelos ou ajuste de pesos em 08 | A etapa 08 analisa saídas estáticas prévias congeladas dos modelos auditados. |
| Testes inferenciais de hipótese entre matrizes Jaccard geradas pelas mesmas sequências | Entradas compartilhadas determinam a igualdade dos motivos comuns; $r=1$ decorre de determinismo de entrada e não de concordância biológica independente. |

## Assumptions

Defaults that are not already a numbered criterion. Drop a row once it is.

| Assumption | Chosen default | Rationale | Confirmed? |
| --- | --- | --- | --- |
| Número de permutações para o teste nulo de nós | 1.000 permutações | Equilíbrio adequado entre precisão empírica do valor de p ($\pm 0.001$) e tempo de execução. | y |
| Amostra de sequências para composição de tripletos | 1.000 fragmentos | Paridade com a amostragem histórica, mas com seleção determinística controlada por semente fixa. | y |
| Limite de pares com Jaccard positivo analisados | Até 200 pares ordenados | Cobre os pares mais fortemente coocorrentes mantendo a representatividade da rede. | y |

**Open questions:** none - all resolved or logged above.

## Observable

Worksheet, not the review. `n/a` needs its reason. One row may group the same decision across several routes.

| Surface | Decision | Landing |
| --- | --- | --- |
| notebook `short/08_triplet_composition_and_cooccurrence.ipynb` | empty / missing 07 inputs state | AC 2 |
| notebook `short/08_triplet_composition_and_cooccurrence.ipynb` | error state / atomic failure manifest | AC 4 |
| notebook `short/08_triplet_composition_and_cooccurrence.ipynb` | successful execution / manifest complete | AC 5 |
| preflight `short/08` | validation of 07 manifest, summary and Jaccard parquets | AC 2 |
| sampling of test sequences | deterministic selection under seed 42 | AC 6 |
| artifacts `table_09_exploratory_rscu.csv` & `figure_04_rscu_profile.png` | multi-frame RSCU profile and null model | AC 7, AC 8 |
| artifact `table_10_translation_boundary_signals.csv` | ATG/STOP counts and fold enrichment | AC 9 |
| artifacts `table_11_cooccurrence_synergy.csv` & `edge_table.csv` | symmetric association, permutation p-value and edge coordinates | AC 11, AC 12, AC 13 |
| figures `figure_05_network.png` & `figure_05_permutation_null.png` | network visualization and null permutation distribution | AC 14 |
| artifact `table_11b_jaccard_variant_diagnostics.csv` | descriptive shared motif overlap | AC 15 |
| all new CLI / workflow invocations | logging path traceability | AC 17 |

## Sources

- `ROADMAP_MELHORIAS_REPRODUCTION_2026-11-09.md` (M05 — Tripletos, T11 e F5) - define requisitos de consumo de 07, amostragem determinística, reconciliação de T11/F5 e advertência exploratória.
- `AGENTS.md` e `AD-014` - estabelecem invariantes de análise de coocorrência simétrica, nulo de permutação de nós e advertência de ausência de ORFs biológicas em 512 bp.
- `configs/publication_c06.yaml` - define requisitos de esquema para T9, T10, T11, F4 e F5 (formato `network_edges`).
