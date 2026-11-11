# M04 — Motivos de Primeira Camada e Perturbação Pareada (07)

## Problem

O notebook de reprodução `short/07_first_layer_motifs_and_perturbation.ipynb` grava suas saídas em um diretório fixo e hardcoded (`results_reproduction/motifs`), violando o isolamento de execuções imutáveis e o ciclo de vida definido por M00 (`m00_run_plan_v1`). Em `notebook_runtime.py:NOTEBOOK_REQUIREMENTS`, a etapa `short/07` declara apenas o checkpoint `resnet_clean_species`, ignorando que a análise audita simultaneamente as cinco variantes (`resnet_clean_species`, `resnet_noise_species`, `convnext_zero_species`, `convnext_transfer_species`, `convnext_noise_species`), permitindo que preflights aprovem execuções mesmo quando faltam quatro dos cinco modelos.

Cientificamente, a Figura 3 gerada atualmente por 07 (`generate_motif_audit_figures`) produz apenas dois painéis: um gráfico de dispersão com todos os pontos agregados e um gráfico de barras horizontal medindo trivialmente a presença do checkpoint (`status=1`). O manuscrito e a especificação de publicação C06 exigem uma Figura 3 com cinco painéis (a: ResNet50 Clean, b: ConvNeXt_Z, c: ConvNeXt_T, d: ResNet50_F, e: ConvNeXt_F), onde cada painel confronta a queda de ativação do motivo contra o controle pareado da mesma sequência, com marcação de significância sob controle GC e suporte $N$. O painel (e) deve refletir avaliações genuínas de `convnext_noise_species`, nunca replicando ou simulando o painel (d).

Adicionalmente, os nomes de colunas gerados em `table_08_motif_perturbation.csv` divergem dos esquemas requeridos pelo validador de publicação C06 e por `reporting/tables.py`. As matrizes de coocorrência de Jaccard e os resumos de motivos (`motif_summary.parquet`) precisam ser persistidos no namespace exclusivo da execução para que o notebook subsequente (08 / M05) possa consumi-los via manifesto explícito, sem depender de caminhos estáticos legados. Por fim, conforme AD-014 e AGENTS.md, os motivos de 7-mers atuam primordialmente como detectores de composição de bases (homopolímeros caem a sinal ~0 sob pareamento por GC), exigindo demarcação explícita de que os motivos não constituem sítios de ligação de fatores de transcrição biologicamente funcionais nem códons funcionais comprovados.

Esta entrega cobre a fase M04 do roadmap de 09/11/2026. A população continua sendo o split fechado de fragmentos virais de 512 bp com 13.683 espécies.

## Flow

Reutilizar o runtime M00/M01/M02/M03, os filtros convolucionais de 1ª camada dos cinco modelos registrados, o mecanismo de escaneamento e perturbação pareada de `interpretability.motifs`, e os seletores de checkpoints de `models.registry`, sem alterar pesos de modelos nem inventar anotações biológicas não verificadas.

1. Ambiente, configuração e plano M00 entram em `m04_workflow` (new), que reutiliza `notebook_runtime` e `run_planning` (exists): resolver entradas explícitas, parâmetros de controle, semente 42 e reserva de diretório exclusivo para 07 (door 1).
2. Checkpoints das cinco variantes e Parquets de teste entram em `preflight_notebook("short/07", ...)` (exists/updated): validar a presença física, integridade e permissão de leitura dos cinco checkpoints (`resnet_clean_species`, `resnet_noise_species`, `convnext_zero_species`, `convnext_transfer_species`, `convnext_noise_species`) e do diretório de sequências de teste (door 2).
3. Checkpoints entram na extração de 7-mers em `extract_first_layer_motifs` (exists): inspecionar pesos convolucionais da 1ª camada (conv1), construir motivos consenso de 7 bp e selecionar até 10 motivos mais ativadores por variante (door 3).
4. Motivos selecionados e sequências de teste entram em `evaluate_motif_perturbation` (exists/updated): para cada ocorrência de motivo, executar perturbação pareada na *mesma sequência* sob controle aleatório uniforme e controle pareado por GC (|ΔGC|=0); calcular quedas de ativação média, teste de postos sinalizados de Wilcoxon pareado e correção FDR Benjamini-Hochberg (door 3).
5. Resultados de perturbação entram no exportador tabular (exists/updated): gerar `table_08_motif_perturbation.csv` e `motif_perturbation_by_variant.csv` com esquema reconciliado compatível com C06 e T8 contendo colunas uniformes e GC, FDRs e suportes $N$ (door 4).
6. Dados tabulares e ativações entram nos geradores de figuras (exists/updated): produzir a Figura 3 científica principal com cinco painéis (a–e) atribuídos a cada variante, a figura complementar de dispersão uniforme × GC com diagonal de robustez, e salvar matrizes Jaccard em Parquet (`motif_interaction_jaccard_<variant>.parquet`) para consumo por M05 (door 5).
7. Runtime grava manifesto individual atômico `manifest.json` com protocolo `m04_motifs_audit_v1`, status `COMPLETE` ou `FAILED`, hashes de checkpoints e dados, e vínculo M00 (door 1).

## Impact

| Front | What changes |
| --- | --- |
| arquivos previstos | Notebook `short/07_first_layer_motifs_and_perturbation.ipynb`; novos módulos `src/viral_cnn/m04_workflow.py` e configuração `configs/reproduction_m04.yaml`; atualizações em `notebook_runtime.py`, `interpretability/motifs.py`, `reporting/figures.py` e `reporting/tables.py`; testes em `tests/test_m04_motifs_and_first_layer_perturbation.py`. |
| saídas | Cada execução escreve em subdiretório exclusivo `results_reproduction/runs/<run_id>/short/07/`. Nenhuma pasta histórica legada (`results_reproduction/motifs`) é sobrescrita. |
| números afetados | Tabela 8 (perturbação pareada de motivos) passa a conter dados auditados e esquema padronizado para as cinco variantes; Figura 3 exibe cinco painéis científicos reais sem barras de status artificiais. |
| cadeia científica | 5 checkpoints + split de teste → extração de 7-mers de 1ª camada → perturbação pareada na mesma sequência (uniforme e GC-matched) → Wilcoxon pareado + FDR → Tabela 8 + Figura 3 (5 painéis) + matrizes Jaccard para 08 / M05. |
| compatibilidade | O runner CLI `scripts/run_motif_analysis.py` e notebooks legados continuam funcionando importando as rotinas de `interpretability/motifs.py`. |
| demarcação biológica | Motivos 7-mers de 1ª camada são explicitamente documentados e rotulados como detectores de composição de bases nucleotídicas, rejeitando atribuição causal a fatores de transcrição ou códons biológicos sem evidência externa. |

## Relations

`None - no stored-data shape change`

## Surface

None - nenhuma API HTTP ou serviço de rede. Interfaces locais dos notebooks e CLI:

- `short/07_first_layer_motifs_and_perturbation.ipynb`: consome variáveis de ambiente ou configuração para `DATA_ROOT`, `SMOKE_TEST`, plano M00 opcional, semente (padrão 42), número de repetições e caminhos dos cinco checkpoints. Produz `manifest.json`, `table_08_motif_perturbation.csv`, `motif_perturbation_by_variant.csv`, `figure_03_motifs.png`, `figure_03_scatter_gc_uniform.png`, `motif_summary.parquet`, e matrizes `motif_interaction_jaccard_<variant>.parquet`. Status de manifesto: `COMPLETE` ou `FAILED`.
- `preflight_notebook("short/07", ...)`: inspeção de preflight em `notebook_runtime.py`, validando a presença física e legibilidade dos cinco checkpoints de espécies e dos dados de sequências antes de iniciar o processamento.
- `configs/reproduction_m04.yaml`: configuração YAML parametrizando o protocolo `m04_motifs_audit_v1`, variantes auditadas, sementes e regras de controle pareado.

Precedência de resolução: (1) plano M00 explícito quando fornecido; (2) variáveis de ambiente explícitas; (3) arquivo de configuração `configs/reproduction_m04.yaml`; (4) topologia padrão do cluster / diretório de dados.

## Landing

| One-way door | Literal shape | Alternative rejected |
| --- | --- | --- |
| 1 — ciclo de vida e adoção M00 em 07 | `protocol_id=m04_motifs_audit_v1`, `notebook_id=short/07`, `run_id`, `is_smoke_test`, semente 42, `git_commit`, hashes de 5 checkpoints/splits e produtos. Reserva atômica via `reserve_run_directory`. | Usar pasta estática `results_reproduction/motifs`: sobrescreve execuções e perde rastreabilidade. |
| 2 — preflight das cinco variantes de modelo | Exigir explicitamente em `NOTEBOOK_REQUIREMENTS["short/07"]` os cinco checkpoints: `resnet_clean_species`, `resnet_noise_species`, `convnext_zero_species`, `convnext_transfer_species`, `convnext_noise_species`. | Manter apenas `resnet_clean_species`: aprova preflight falso quando faltam os modelos de ConvNeXt ou Noise. |
| 3 — controle pareado estrito na mesma sequência | Controle pareado uniforme e controle pareado por GC (|ΔGC|=0) amostrados obrigatoriamente da *mesma sequência* portadora do motivo, com semente 42 e Wilcoxon pareado bicaudal com correção FDR Benjamini-Hochberg. | Comparar contra sequências aleatórias de outro genoma/fragmento: introduz viés de GC global e infla falsamente o efeito do motivo. |
| 4 — esquema de colunas reconciliado para T8 | `motif_perturbation_by_variant.csv` contendo: `panel`, `variant`, `motif`, `mean_random_drop_uniform`, `mean_motif_drop_uniform`, `mean_specific_drop_uniform`, `mean_random_drop_gc`, `mean_motif_drop_gc`, `mean_specific_drop_gc`, `fdr_uniform`, `fdr_gc`, `n_occurrences`, `is_gc_significant`. | Manter colunas divergentes entre notebook e C06: quebra a regeneração automática de tabelas e falha o validador C06. |
| 5 — Figura 3 científica com cinco painéis (a–e) | Renderizar `figure_03_motifs.png` contendo cinco subpainéis estruturados (a: ResNet50 Clean, b: ConvNeXt_Z, c: ConvNeXt_T, d: ResNet50_F, e: ConvNeXt_F) com quedas de ativação e indicadores de suporte e significância, removendo o gráfico de barras de status. | Gráfico de 2 painéis com barras `status=1`: gráfico não-científico que não atende a Fig. 3 do manuscrito. |

## Criteria

### S1: Isolamento de execução, adoção M00 e preflight das 5 variantes (P1)

**Acceptance Criteria**

1. WHEN executa 07 THEN o fluxo SHALL resolver os caminhos de entrada e saída a partir do plano M00 ou variáveis de ambiente sem depender de diretórios fixos legados.
2. WHEN `preflight_notebook("short/07", ...)` é executado no modo empírico (`SMOKE_TEST=0`) THEN o runtime SHALL validar a presença e legibilidade dos cinco checkpoints de espécies e reportar `READY` apenas se todos existirem, ou `BLOCKED` se qualquer um faltar.
3. IF o diretório reservado de destino já contiver produtos ou colidir com run existente THEN o fluxo SHALL abortar antes de modificar qualquer arquivo.
4. WHEN uma execução de 07 falha por erro inesperado THEN o fluxo SHALL emitir manifesto atômico com status `FAILED` e detalhe da falha, nunca `COMPLETE`.
5. WHEN a etapa 07 conclui com sucesso THEN o fluxo SHALL publicar manifesto `COMPLETE` contendo protocolo `m04_motifs_audit_v1`, `is_smoke_test`, semente 42, hashes das entradas e lista completa de artefatos gerados.

**Independent test:** executar preflight com checkpoints completos e com um checkpoint faltante; testar aborto por colisão de diretório e persistência de manifesto FAILED sob exceção induzida.

### S2: Extração de 7-mers e avaliação pareada com controle de GC (P1)

**Acceptance Criteria**

6. WHEN executa a extração de motivos de 1ª camada THEN o extrator SHALL analisar os pesos convolucionais de entrada e extrair até 10 motivos consenso de 7 bp para cada um dos cinco modelos.
7. WHEN avalia a perturbação de cada motivo THEN o fluxo SHALL selecionar controles nulos (uniforme aleatório e pareado por GC com |ΔGC|=0) obrigatoriamente na mesma sequência hospedeira do motivo sem sobreposição.
8. WHEN calcula o impacto da perturbação THEN o fluxo SHALL computar a queda específica de ativação, o teste bicaudal de Wilcoxon pareado e a correção para múltiplos testes por FDR Benjamini-Hochberg.
9. WHEN exporta os resultados tabulares de perturbação THEN o fluxo SHALL salvar `table_08_motif_perturbation.csv` e `motif_perturbation_by_variant.csv` contendo esquema de colunas padronizado com quedas uniformes e GC, FDRs e contagens de ocorrências.
10. WHEN executa 07 no modo empírico (`SMOKE_TEST=0`) THEN o fluxo SHALL rejeitar checkpoints ou fixtures marcados como sintéticos (`is_smoke_test=true`).

**Independent test:** fixture com sequências sintéticas conhecidas e modelo determinístico; verificar cálculo de Wilcoxon e FDR; testar que homopolímeros produzem queda específica próxima de zero sob controle pareado por GC; verificar rejeição de modelo sintético sob `SMOKE_TEST=0`.

### S3: Figura 3 de cinco painéis, coocorrência e demarcação de evidência (P1)

**Acceptance Criteria**

11. WHEN renderiza a Figura 3 científica principal THEN o renderizador SHALL produzir painel com exatamente cinco subgráficos (a: ResNet50 Clean, b: ConvNeXt_Z, c: ConvNeXt_T, d: ResNet50_F, e: ConvNeXt_F) exibindo quedas de ativação e significância por GC, sem barras de status de checkpoint.
12. The renderizador SHALL garantir que o painel (e) utilize os dados e ativações genuínos de ConvNeXt_F (`convnext_noise_species`), nunca clonando bytes ou dados do painel (d).
13. WHEN renderiza a figura de sensibilidade composicional THEN o renderizador SHALL exportar gráfico de dispersão confrontando quedas sob controle uniforme versus GC com indicação de linha de identidade e limiares de robustez.
14. WHEN calcula a coocorrência entre motivos THEN o fluxo SHALL exportar matrizes de presença e matrizes Jaccard em Parquet (`motif_interaction_jaccard_<variant>.parquet`) no namespace da execução para consumo por 08 / M05.
15. WHEN documenta ou sumariza os motivos de 1ª camada THEN o texto e os metadados SHALL demarcar explicitamente que os motivos representam detectores de composição de bases nucleotídicas e não sítios funcionais biológicos comprovados de fatores de transcrição nem códons biológicos funcionais.
16. WHEN artefatos são lidos ou gravados nos fluxos M04 THEN o código SHALL emitir logs com prefixo `[LOG_PATH]` contendo o caminho do arquivo, respeitando `LOG_PATH=0/false`.

**Independent test:** verificar geração de `figure_03_motifs.png` e comprovar que os bytes e dados do painel (e) são distintos do painel (d); inspecionar matrizes Jaccard geradas; testar emissão e supressão de logs `[LOG_PATH]`.

## Out of scope

Product capabilities only. Process and harness rules live in AGENTS.md or as Observable `n/a`.

| Excluded | Why |
| --- | --- |
| Retreinamento de pesos ou fine-tuning de modelos em 07 | A etapa 07 audita checkpoints existentes congelados; retreinamento pertence ao cluster conforme AD-016. |
| Inclusão de ORFs ou anotações biológicas de códons | Fragmentos de 512 bp não possuem quadro aberto de leitura verificado; tripletos são estritamente exploratórios e pertencem ao M05 / 08. |
| Alinhamento ou pareamento com bases de dados de fatores de transcrição (JASPAR/UniBind) | Motivos de 1ª camada de DNA de vírus sem anotação experimental não suportam atribuição funcional direta. |

## Assumptions

Defaults that are not already a numbered criterion. Drop a row once it is.

| Assumption | Chosen default | Rationale | Confirmed? |
| --- | --- | --- | --- |
| Número máximo de motivos selecionados por variante | 10 motivos por modelo | Paridade com o método publicado e cobertura dos filtros mais ativadores da camada convolucional inicial. | y |
| Comprimento dos motivos de 1ª camada | 7 bp (7-mers) | Corresponde ao kernel_size=7 da primeira camada de convolução 1D do artigo. | y |
| Unidade de pareamento estatístico | Mesma sequência de 512 bp | Pareamento dentro da mesma sequência controla o GC de fundo e a complexidade local do genoma viral. | y |

**Open questions:** none - all resolved or logged above.

## Observable

Worksheet, not the review. `n/a` needs its reason. One row may group the same decision across several routes.

| Surface | Decision | Landing |
| --- | --- | --- |
| notebook `short/07_first_layer_motifs_and_perturbation.ipynb` | empty / missing checkpoints state | AC 2 |
| notebook `short/07_first_layer_motifs_and_perturbation.ipynb` | error state / atomic failure manifest | AC 4 |
| notebook `short/07_first_layer_motifs_and_perturbation.ipynb` | successful execution / manifest complete | AC 5 |
| preflight `short/07` | validation of 5 model variants and split Parquets | AC 2 |
| artifacts `table_08_motif_perturbation.csv` & `motif_perturbation_by_variant.csv` | standardized schema and support columns | AC 9 |
| figure `figure_03_motifs.png` | 5 panels (a–e) without artificial status bars | AC 11, AC 12 |
| all new CLI / workflow invocations | logging path traceability | AC 16 |

## Sources

- `ROADMAP_MELHORIAS_REPRODUCTION_2026-11-09.md` (M04 — Motivos e F3) - define requisitos para isolamento de 07, T8, F3 e integridade de ConvNeXt_F.
- `AGENTS.md` e `AD-014` - estabelecem invariantes de controle pareado por GC, demarcação composicional de 7-mers e preservação de paridade metodológica.
- `configs/publication_c06.yaml` - define requisitos de esquema e painéis esperados para T8 e F3.
