# M03 — Espaço latente e visualizações úteis (06, 06b)

## Problem

Os notebooks de reprodução `short/06_ictv_latent_space_and_hierarchy.ipynb` e `short/06b_latent_visualizations.ipynb` contêm caminhos absolutos codificados para sessões anteriores no JupyterHub (ex.: `/home/mlima/workspace/cnn_metagenome/results_reproduction/runs/n06b-20261106T193632821026/06b_latent_visualizations/umap_comparison.png` em 06b) e não estão plenamente integrados ao gerenciamento de ciclo de vida e adoção de reservas imutáveis M00 (`m00_run_plan_v1`). A etapa `short/06b` sequer está registrada em `NOTEBOOK_REQUIREMENTS` de `notebook_runtime.py`, impedindo pré-verificação padronizada via preflight.

Embora `c10_paired_umap_v1` garanta que os cinco modelos recebam as mesmas sequências e rótulos para projeções 2D independentes, o painel estático atual tenta renderizar todos os gêneros com paletas de até centenas de categorias superpostas, tornando a inspeção visual ineficaz sem o HTML explorável. Falta um painel secundário seletivo, com gêneros escolhidos por regra estatística objetiva e pré-definida (quotas mínimas de suporte e amostragem determinística estratificada), destacando os gêneros de interesse e projetando todos os demais pontos em cinza neutro ao fundo. Não são exportados formalmente `selected_genera.csv` e `sampled_query_ids.csv` para auditoria independente da seleção. Além disso, as figuras qualitativas UMAP não são acompanhadas por um painel quantitativo no espaço original com as métricas auditadas (KNN Cosseno k=5, Silhouette Score e acurácia da sonda linear selada), arriscando que leitores interpretem agrupamentos UMAP 2D como proximidade filogenética ou métrica direta de separabilidade.

Esta entrega cobre a fase M03 do roadmap de 09/11/2026. A população continua sendo o split fechado de fragmentos virais de 512 bp com 13.683 espécies e 2.702 gêneros.

## Flow

Reutilizar o runtime M00/M01/M02, a extração e auditoria de embeddings de `eval.latent`, o cache de retomada C09 em `latent_resume`, a amostragem pareada C10 em `paired_umap` e a renderização C12 em `latent_plotting`, sem alterar pesos de modelos nem recalcular dados históricos fora do namespace de execução.

1. Ambiente, configuração e plano M00 entram em `m03_workflow` (new), que reutiliza `notebook_runtime` e `run_planning` (exists): resolver entradas explícitas, modo de execução (`reuse` ou `rerun`), IDs e reserva de diretório exclusivo para 06 e 06b (door 1).
2. Pré-requisitos de splits, taxonomia com hierarquia validada M01 (`hierarchy_ready=true`) e checkpoints entram em `preflight_notebook` / `preflight_latent_inputs` (exists): validar integridade de esquemas, partições disjuntas e presença dos cinco modelos antes de iniciar extração (door 2).
3. Checkpoints e splits entram no pipeline de extração de embeddings de 06: em `reuse`, validar identidade e integridade de caches C09 existentes; em `rerun`, extrair embeddings dos cinco modelos no espaço original pré-classificador, salvar tensores e metadados com hashes atômicos (door 2).
4. Embeddings pré-classificadores entram na avaliação quantitativa no espaço original (exists): calcular KNN Cosseno ($k=5$) com workspace limitado C08, Silhouette Score amostral com controle de classes válidas e Sonda Linear treinada em train/val e avaliada em test selado (AD-012, AD-019, AD-025). Gerar figura quantitativa do espaço original (door 3).
5. Embeddings e taxonomia de test entram em `paired_umap` e `latent_plotting` (exists/updated): selecionar amostra balanceada pareada, gerar projeções UMAP (ou fallback PCA) independentes para as cinco variantes, mantendo o painel UMAP completo e o HTML explorável (exists), e adicionando o novo painel seletivo com gêneros destacados e fundo neutro, exportando `selected_genera.csv` e `sampled_query_ids.csv` (door 4).
6. Notebook 06b (exists/updated) consome os embeddings ou projeções publicadas por 06 através de seleção explícita de `source_dir` ou run M00, gera as figuras regeneráveis e publica manifesto independente `COMPLETE` com vínculo `parent_run_id` sem caminhos fixos (door 1).

## Impact

| Front | What changes |
| --- | --- |
| arquivos previstos | Notebooks `short/06_ictv_latent_space_and_hierarchy.ipynb` e `short/06b_latent_visualizations.ipynb`; novos módulos `src/viral_cnn/m03_workflow.py` e configuração `configs/reproduction_m03.yaml`; atualizações em `notebook_runtime.py`, `paired_umap.py` e `latent_plotting.py`; testes em `tests/test_m03_latent_space_visualizations.py`. |
| saídas | Cada execução escreve em subdiretórios exclusivos `results_reproduction/runs/<run_id>/short/06` e `.../06b`. Nenhuma pasta histórica de cluster é sobrescrita. |
| números afetados | Métricas quantitativas do espaço original (Tabela 5 e Tabela 6) são preservadas e auditadas; novo CSV de gêneros selecionados e novo painel comparativo seletivo. Nenhum número histórico é modificado silenciosamente. |
| cadeia científica | Split de fragmentos + taxonomia M01 + 5 checkpoints → embeddings pré-classificadores → KNN / Silhouette / Sonda Linear no espaço original → Projeções 2D qualitativas pareadas (completa e seletiva) → Figuras e manifestos rastreáveis. |
| compatibilidade | O modo `reuse` C09 continua funcional para reaproveitar embeddings legítimos pré-computados com verificação de SHA-256; `rerun` executa em novo namespace limpo. |
| evidência qualitativa | UMAP e PCA são estritamente demarcados como qualitativos 2D; eixos e distâncias 2D não constituem evidência filogenética nem substituem métricas de alta dimensão. |

## Relations

Uma execução de 06 produz embeddings de teste para exatamente cinco modelos, tabelas quantitativas de KNN/Silhouette/Sonda e opcionalmente as projeções pareadas. A etapa 06b consome exatamente uma execução 06 (ou diretório de coordenadas salvas) identificada e imutável. Cada fragmento projetado pertence a exatamente uma espécie e um gênero válido. Gêneros ausentes ou inválidos na taxonomia são formalmente excluídos da população elegível com registro de auditoria, nunca imputados.

## Surface

None - nenhuma API HTTP ou serviço de rede. Interfaces locais dos notebooks e CLI:

- `short/06_ictv_latent_space_and_hierarchy.ipynb`: recebe variáveis de ambiente ou configuração para `DATA_ROOT`, `SMOKE_TEST`, `LATENT_EXECUTION_MODE` (`reuse` ou `rerun`), `LATENT_REUSE_RUN_ID`, plano M00 opcional, limites de UMAP (`LATENT_UMAP_MAX_SAMPLES`, `LATENT_UMAP_MAX_GENERA`) e caminho da taxonomia. Produz `manifest.json`, `table5_knn_silhouette.csv`, `table6_correlations.csv`, `probe_audit.json`, embeddings por variante, e sob o protocolo de UMAP pareado: `umap_comparison.png`, `umap_selective_panel.png`, `latent_quantitative_summary.png`, `selected_genera.csv`, `sampled_query_ids.csv`, `umap_comparison.json`. Status de manifesto: `COMPLETE` ou `FAILED`.
- `short/06b_latent_visualizations.ipynb`: recebe `LATENT_SOURCE_DIR` (diretório de execução de 06 contendo embeddings ou `umap_comparison.json`) ou plano M00 via ambiente. Não aceita caminhos absolutos hardcoded no código. Produz `manifest.json`, `umap_comparison.png`, `umap_selective_panel.png`, `umap_comparison.html`, `umap_genus_colors.csv` e cópias auditadas das coordenadas CSV. Status de manifesto: `COMPLETE` ou `FAILED`.
- `preflight_notebook("short/06b", ...)`: nova inspeção de preflight em `notebook_runtime.py`, verificando se a fonte declarada de 06 existe e contém os manifestos/arquivos necessários antes de iniciar a execução.

Precedência de resolução: (1) plano M00 explícito quando fornecido; (2) variáveis de ambiente explícitas; (3) arquivo de configuração `configs/reproduction_m03.yaml`; (4) topologia padrão do cluster / diretório de dados.

## Landing

| One-way door | Literal shape | Alternative rejected |
| --- | --- | --- |
| 1 — ciclo de vida e adoção M00 em 06 e 06b | `protocol_id=m03_latent_audit_v1`, `notebook_id=short/06\|short/06b`, `run_id`, `parent_run_id`, `is_smoke_test`, semente 42, `git_commit`, hashes de checkpoints/splits/taxonomia e produtos. Registro de `short/06b` em `NOTEBOOK_REQUIREMENTS`. Preflight específico para 06 e 06b. | Usar pastas fixas e caminhos codificados `/home/mlima/...`: sobrescreve execuções e impede reprodutibilidade em novos ambientes. |
| 2 — validação e demarcação de embeddings C09 | Em `reuse`, exigir correspondência exata de `embedding_sha256` e código; em `rerun`, criar novo diretório reservado. Rejeitar embeddings sintéticos em execuções empíricas (`SMOKE_TEST=0`). | Reutilizar arquivos por data de modificação ou nome: permite contaminação por caches desatualizados ou sintéticos. |
| 3 — figura quantitativa no espaço original | Exportar `latent_quantitative_summary.png` e `.csv` contendo as métricas reais em alta dimensão: KNN Cosseno Accuracy por modelo, Silhouette Score (amostragem aleatória e balanceada), taxa de classes válidas ($\ge 2$ amostras) e Sonda Linear Accuracy em test. | Apresentar apenas UMAP 2D: induz a falsa interpretação de que o gráfico 2D demonstra separabilidade ou filogenia. |
| 4 — painel seletivo e exportação de amostra pareada | `umap_selective_panel.png` destacando os $K$ gêneros selecionados deterministicamente (ex.: por quota mínima de suporte e ordenação canônica) com cores contrastantes, e renderizando todos os demais pontos em cinza neutro (`#d0d0d0`, alpha reduzido). Exportar `selected_genera.csv` (colunas: `Genus`, `color`, `sample_count`, `selection_rank`, `is_highlighted`) e `sampled_query_ids.csv` (`query_id`, `label`, `Genus`, `is_highlighted`). | Escolher gêneros manualmente pela estética do gráfico UMAP: viés de confirmação e ausência de reprodutibilidade científica. |

## Criteria

### S1: Isolamento de execução, adoção M00 e preflight 06/06b (P1)

**Acceptance Criteria**

1. WHEN executa 06 ou 06b THEN o fluxo SHALL resolver os caminhos e parâmetros a partir do plano M00 ou variáveis de ambiente sem depender de diretórios absolutos fixos de usuários.
2. WHEN `preflight_notebook("short/06b", ...)` é executado THEN o runtime SHALL validar os requisitos declarados da etapa 06b e reportar status `READY` ou `BLOCKED`.
3. IF o diretório de destino reservado já contiver produtos ou houver colisão de run THEN o fluxo SHALL falhar antes de iniciar o processamento ou modificar os arquivos existentes.
4. WHEN uma execução 06 ou 06b falha por exceção THEN o fluxo SHALL registrar manifesto atômico com status `FAILED` e causa do erro, nunca `COMPLETE`.
5. WHEN 06 ou 06b conclui com sucesso THEN o fluxo SHALL publicar manifesto individual `COMPLETE` contendo protocolo `m03_latent_audit_v1`, `is_smoke_test`, semente 42, hashes das entradas e produtos.

**Independent test:** simular execuções com destino limpo, destino colidente, preflight de 06b com fonte ausente/presente, e falha induzida para validar persistência de `FAILED`.

### S2: Auditoria e integridade do espaço latente original (P1)

**Acceptance Criteria**

6. WHEN executa 06 no modo `reuse` THEN o fluxo SHALL validar a integridade dos embeddings salvos através de SHA-256 e identidade C09 antes de prosseguir.
7. WHEN executa 06 no modo empírico (`SMOKE_TEST=0`) THEN o fluxo SHALL rejeitar dependências ou embeddings marcados como sintéticos (`is_smoke_test=true`).
8. WHEN calcula KNN Cosseno ($k=5$), Silhouette Score e Sonda Linear THEN o código SHALL computar todas as métricas no espaço original de alta dimensão pré-classificador, sem redução prévia por UMAP ou PCA.
9. WHEN publica as métricas quantitativas de alta dimensão THEN o fluxo SHALL gerar `latent_quantitative_summary.csv` e `latent_quantitative_summary.png` confrontando os cinco modelos com indicação explícita de $N$ e cobertura de classes válidas.

**Independent test:** fixture com tensores sintéticos de embeddings de alta dimensão e rótulos conhecidos; comparar cálculos de KNN, Silhouette e acurácia da sonda contra valores de referência; testar rejeição de hash divergente em `reuse`.

### S3: Projeção pareada C10 e novo painel seletivo (P1)

**Acceptance Criteria**

10. WHEN seleciona fragmentos para projeção 2D THEN o fluxo SHALL utilizar o protocolo determinístico pareado C10, garantindo que os cinco modelos recebam exatamente as mesmas sequências (`row_id`), espécies e gêneros.
11. WHEN exporta a seleção pareada THEN o fluxo SHALL salvar `selected_genera.csv` e `sampled_query_ids.csv` contendo a lista completa de gêneros, contagens, status de destaque e identificadores de sequência.
12. WHEN renderiza o painel seletivo `umap_selective_panel.png` THEN o renderizador SHALL colorir os gêneros em destaque com a paleta contrastante e desenhar todos os demais fragmentos da amostra em cinza neutro ao fundo com opacidade atenuada.
13. WHEN renderiza qualquer visualização UMAP ou PCA THEN o gráfico e o arquivo gerado SHALL incluir aviso proeminente de que a representação é estritamente qualitativa e que os eixos são independentes e não-filogenéticos.
14. WHEN executa 06b a partir de embeddings ou coordenadas salvas de 06 THEN o notebook 06b SHALL regenerar tanto o painel completo quanto o painel seletivo e o HTML interativo preservando estritamente as coordenadas originais.

**Independent test:** fixture com projeções 2D conhecidas para 5 modelos; verificar existência, colunas e consistência de `selected_genera.csv` e `sampled_query_ids.csv`; verificar que pontos não destacados recebem cor neutra; verificar que as coordenadas não são alteradas no 06b.

### S4: Rastreabilidade, logs e limites da evidência (P1)

**Acceptance Criteria**

15. WHEN artefatos persistentes são lidos ou gravados nos fluxos M03 THEN o código SHALL emitir logs com prefixo `[LOG_PATH]` contendo o caminho do arquivo, respeitando `LOG_PATH=0/false`.
16. WHEN documenta M03 THEN a documentação SHALL explicitar comandos de execução para 06 e 06b, parâmetros suportados, distinção entre smoke local e reprodução empírica no cluster, e demarcação de que projeções UMAP 2D não medem acurácia nem proximidade filogenética.

**Independent test:** capturar saída de console com `LOG_PATH=1` e `LOG_PATH=0`; inspecionar documentação e notas de evidência de M03.

## Out of scope

| Excluded | Why |
| --- | --- |
| Retreinar ou ajustar hiperparâmetros dos cinco modelos | Os checkpoints são insumos pré-existentes imutáveis e avaliados closed-set. |
| Alinhar eixos ou forçar rotação entre projeções UMAP | Cada espaço latente tem dinâmica própria; forçar alinhamento induziria falsas correlações espaciais. |
| Reconstruir taxonomia ou imputar gêneros faltantes | Taxonomia validada provém de M01; classes sem gênero válido são excluídas da análise de gênero com registro de auditoria. |
| Treinar sonda linear diretamente sobre conjunto test | Viola o contrato científico AD-012/AD-019; o treino ocorre estritamente em train/val, avaliando apenas a inferência em test. |

## Assumptions

| Assumption | Chosen default | Rationale | Confirmed? |
| --- | --- | --- | --- |
| Perfil de verificação | light | Padrão da skill; verificação independente após implementação. | n |
| Quantidade de gêneros em destaque no painel seletivo | 10 gêneros | Permite distinção clara de cores em gráfico estático de cinco colunas sem sobrecarga perceptiva. | n |
| Critério de seleção para destaque | Top gêneros por suporte amostral elegível com desempate determinístico | Assegura estabilidade estatística e cobertura suficiente de pontos em cada classe para visualização. | n |
| Backend de projeção sem biblioteca UMAP instalada | Fallback determinístico para PCA (2 componentes) | Mantém compatibilidade com ambientes de teste leves onde umap-learn não está instalado. | n |

**Open questions:** none — os padrões adotados acima preservam os contratos de AD-012, AD-019, AD-025, AD-026 e AD-027.

## Observable

| Surface | Decision | Landing |
| --- | --- | --- |
| Notebooks 06 e 06b | Parâmetros de ambiente/configuração, eliminação de caminhos fixos | Surface; Criteria 1, 3, 4, 5, 14 |
| Preflight de requisitos | Inspeção de 06 e novo suporte a 06b em `notebook_runtime` | Criteria 2, 4 |
| Resumo quantitativo | Gráfico e CSV com KNN, Silhouette e Sonda em alta dimensão | Landing 3; Criteria 8, 9 |
| Painel seletivo e CSVs | Novo painel com destaque e fundo neutro; `selected_genera.csv` e IDs | Landing 4; Criteria 10, 11, 12 |
| Demarcação qualitativa | Textos de aviso e metadados sobre limitações do UMAP | Landing 4; Criteria 13, 16 |
| Manifestos e logs | Manifestos atômicos `COMPLETE`/`FAILED` e logs `[LOG_PATH]` | Landing 1; Criteria 5, 15 |
| Tela/API de rede | Sem interface web ou serviço HTTP novo | n/a — notebooks e scripts locais |
| Ações destrutivas | Sem exclusão de arquivos de origem ou substituição em pastas antigas | n/a — reservas exclusivas em namespace de run |

Nove dimensões para derivar em checks: validação (Criteria 2, 6, 7); falha parcial (Criteria 3, 4); idempotência/duplicatas (Criteria 1, 3, 10); autorização/rate limits (n/a — execução local sem serviço); concorrência/ordenação (reserva exclusiva e dependência de stage 1, 5, 14); ciclo de dados (Criteria 5, 6, 10, 11); falha de dependência externa (Criteria 2, 4, 7); transições (Criteria 4, 5, 14); observabilidade (Criteria 5, 9, 11, 13, 15, 16).

## Sources

- `ROADMAP_MELHORIAS_REPRODUCTION_2026-11-09.md`: seção 31 (M03), contratos comuns de execução e integridade científica.
- `.specs/STATE.md`: decisões AD-001, AD-004, AD-005, AD-006, AD-012, AD-017, AD-019, AD-025, AD-026, AD-027, AD-028, AD-029, AD-030.
- `src/viral_cnn/paired_umap.py`, `src/viral_cnn/latent_plotting.py`, `src/viral_cnn/latent_resume.py` e `src/viral_cnn/eval/latent.py`.
- Notebooks históricos e de reprodução: `notebooks/reproduction/short/06_ictv_latent_space_and_hierarchy.ipynb` e `06b_latent_visualizations.ipynb`.
