# C01 — Inventário e Contratos Executáveis de Dados de Publicação

## Problem

A cadeia de publicação e diagnóstico local (`configs/reproduction_m07.yaml`, `src/viral_cnn/reporting/publication.py`, `src/viral_cnn/reporting/transformations.py`) aponta para caminhos de origem incorretos e assume esquemas tabulares incompatíveis com os dados que os notebooks de reprodução (`short/01–09`, `long/01–03`) efetivamente gravam:

1. **Caminhos propostos incorretos em `configs/reproduction_m07.yaml`:**
   - T2 está associada a `short/02/`, mas é gerada por `short/01_benchmark_clean_and_noise.ipynb` (`table2_profiling.csv`).
   - T7 está associada a `short/06b/`, mas é gerada por `short/02_analyze_confidence_and_calibration.ipynb` (`table7_hierarchical_entropy.csv`).
   - F2 aponta para `short/06b/selected_genera.csv`, que armazena apenas gêneros selecionados sem coordenadas de projeção (`umap_1`, `umap_2`) nem identidade de variantes.
   - T11 possui lista vazia de colunas esperadas (`columns: []`), bloqueando a verificação C06.

2. **Divergência entre esquemas reais e adaptadores:**
   - As transformações atuais preenchem medições ausentes com `0.0` (ex.: em T1 e T2), violando o contrato contra valores inventados.
   - Os nomes e unidades exportados diferem dos esperados: `noise_rate` vs `epsilon` (T3/F1), `Transition` vs `level_pair` (T6), `codon` vs `triplet` e séries de RSCU (T9/F4).
   - T12 em `long/03` é exportada por linha de variante e agregação (`build_c02_table12`), enquanto C06 espera colunas agregadas (`majority_vote_acc`, `mean_logits_acc`).
   - F2 possui comentários de cabeçalho (`# QUALITATIVE ONLY...`) gravados por `paired_umap.py` que causam falhas ou corrupção de cabeçalho ao serem consumidos diretamente.

3. **Fixtures de teste artificiais:**
   - Os testes existentes em `tests/test_m07_publication_contracts.py` utilizam DataFrames sintéticos já no esquema final de publicação, não exercitando os formatos reais exportados pelos notebooks.

## Flow

Reutiliza os exportadores existentes dos notebooks e o motor C06 (`publication.py`, `transformations.py`, `paired_umap.py`), sem alterar a integridade científica dos dados observados.

1. Configuração de reprodução e saídas reais dos notebooks entram em `diagnose_publication_readiness` (exists) - inspecionar caminhos corrigidos e cabeçalhos reais.
2. Saídas com cabeçalhos reais entram em `transformations.py` (exists) - converter colunas (`noise_rate -> epsilon`, `Transition -> level_pair`, `codon -> triplet`), pivotar agregações de T12 e validar campos obrigatórios sem preenchimento artificial por zero (door 1, door 3).
3. Arquivos de projeção de `short/06b` com comentários de aviso qualitativo entram em `paired_umap.py` / adaptador F2 (exists) - ler coordenadas e metadados ignorando linhas iniciadas por `#` e preservando a declaração qualitativa (door 2).
4. Dados do teste de permutação e coocorrência de `short/08` entram no contrato de T11 (exists) - validar lista explícita de colunas e significância estatística empírica (door 4).
5. Tabela consolidada de contratos entra em `docs/publication_data_contracts.md` e `configs/reproduction_m07.yaml` (new/updated) - fornecer referência única para C02–C06.

## Impact

| Front | What changes |
| --- | --- |
| domain | new term: `publication_data_contract_matrix` - matriz de contratos especificando notebook produtor, caminho relativo, colunas reais, cardinalidade, denominador e destino C06, documentada em `docs/publication_data_contracts.md` |
| domain | existing term: `T11` significava correlação entre arquiteturas com colunas vazias, agora significa associação de coocorrência exploratória e orientação com teste de permutação de nós |
| stored data | nada a migrar |

## Relations

`None - no stored-data shape change`

## Surface

| Route | In | Out | Status |
| --- | --- | --- | --- |
| `viral_cnn.reporting.publication:diagnose_publication_readiness(config_or_path, project_root)` | `config_path` | `DataFrame[item_id, status, diagnostic, proposed_source, recommendation]` | `200`, `400` |
| `viral_cnn.reporting.transformations:execute_publication_transformation(item_id, source_path, output_dir, parent_manifest_path)` | `item_id`, `source_path`, `output_dir`, `parent_manifest_path` | `tuple[Path, Path]` | `200`, `400` |
| `viral_cnn.paired_umap:load_projection_coordinates(csv_path)` | `csv_path` | `DataFrame[umap_1, umap_2, row_id, label, Genus, projection_method, is_qualitative_only]` | `200`, `400` |

## Landing

| One-way door | Literal shape | Alternative rejected |
| --- | --- | --- |
| 1 — Matriz canônica de contratos de dados em documento versionado | Tabela estruturada em `docs/publication_data_contracts.md` e espelhada em `configs/reproduction_m07.yaml` com mapeamento exato dos 17 itens T1–T12 e F1–F5. | Manter mapeamentos implícitos em scripts ou fixtures dispersas. |
| 2 — Leitura resiliente de projeções F2 com comentários qualitativos | Parâmetro `comment='#'` na ingestão de CSVs de projeção UMAP/PCA, preservando o disclaimer físico no arquivo e permitindo parsing tabular. | Remover comentários dos arquivos físicos ou ignorar arquivos com avisos qualitativos. |
| 3 — Proibição de preenchimento artificial com zeros | Lançar `PublicationValidationError` quando colunas obrigatórias estiverem ausentes em vez de atribuir `df[c] = 0.0`. | Manter zeros fabricados mascarando medições não computadas. |
| 4 — Contrato explícito e colunas esperadas para T11 | Definir colunas obrigatórias em `configs/reproduction_m07.yaml`: `["status", "n_valid_edges", "n_valid_permutations", "r_canonical", "r_reversed", "delta_r_orientation", "icc_symmetric", "r_bidirectional", "p_permutation_null"]`. | Manter `columns: []` que bloqueia C06 ou assumir correlação entre modelos sem teste de permutação. |

## Criteria

### S1: Matriz versionada de contratos e correção de caminhos propostos (P1)

Publicação da matriz de contratos e alinhamento dos caminhos propostos em `configs/reproduction_m07.yaml`.

**Acceptance Criteria**

1. The system SHALL document the data contract matrix for all 17 items (T1–T12, F1–F5) in `docs/publication_data_contracts.md` specifying producer notebook, output path, real columns, cardinality, denominator, transformation, and target file.
2. The system SHALL configure the proposed source for T2 as `results_reproduction/runs/<run_id>/short/01/table2_profiling.csv` in `configs/reproduction_m07.yaml`.
3. The system SHALL configure the proposed source for T7 as `results_reproduction/runs/<run_id>/short/02/table7_hierarchical_entropy.csv` in `configs/reproduction_m07.yaml`.
4. The system SHALL configure the proposed source for F2 as `results_reproduction/runs/<run_id>/short/06b/umap_projection_convnext_f.csv` or projection folder in `configs/reproduction_m07.yaml`.
5. The system SHALL define non-empty expected columns for T11 matching `short/08` co-occurrence synergy output in `configs/reproduction_m07.yaml`.

**Independent test:** verificar integridade de `docs/publication_data_contracts.md` e validação sintática de `configs/reproduction_m07.yaml`.

### S2: Adaptadores fiéis aos esquemas reais e rejeição de preenchimentos nulos (P1)

Adaptação robusta das colunas reais exportadas para o formato canônico de publicação sem fabricação de valores.

**Acceptance Criteria**

6. WHEN transforming T1 THEN the system SHALL require verified columns for parameters, accuracy, top-5, and macro-F1, and SHALL raise a PublicationValidationError IF any required measurement column is missing, without filling missing values with 0.0.
7. WHEN transforming T2 THEN the system SHALL map parameters and compute latency and throughput from `table2_profiling.csv`, and SHALL raise a PublicationValidationError IF profiling columns are absent, without filling missing values with 0.0.
8. WHEN transforming T3 or F1 THEN the system SHALL convert `noise_rate` to `epsilon` while preserving all evaluated methods, strata, and sample size $N$.
9. WHEN transforming T6 THEN the system SHALL convert `Transition` to `level_pair` and map `Model` to canonical model names.
10. WHEN transforming T9 or F4 THEN the system SHALL map `codon` to `triplet` and preserve the exploratory RSCU reading frames (`rscu_frame0`, `rscu_forward_3frames`, `rscu_six_frames`).
11. WHEN transforming T10 THEN the system SHALL preserve `signal`, `codon`, and observed/expected counts without claiming confirmed ORF boundaries.
12. WHEN transforming T12 THEN the system SHALL pivot rows from `build_c02_table12` (`variant × aggregation`) into a consolidated row per variant with explicit `majority_vote_acc` and `mean_logits_acc`.

**Independent test:** executar transformações sobre fixtures sintéticas que reproduzem os cabeçalhos exatos de `short/01`, `02`, `06`, `07`, `08`, `09` e `long/03`.

### S3: Consumo de projeções F2 com comentários qualitativos (P1)

Leitura de coordenadas de projeção sem falhas provocadas por linhas de comentário.

**Acceptance Criteria**

13. WHEN loading F2 projection data containing `#` comment lines THEN the system SHALL parse tabular coordinates (`umap_1`, `umap_2`), `label`, `row_id`, and `Genus` while ignoring header comment lines.
14. IF an F2 source artifact lacks coordinates `umap_1` and `umap_2` THEN the system SHALL reject it with a PublicationValidationError.

**Independent test:** carregar CSV gerado por `paired_umap.py` contendo disclaimer qualitativo e verificar extração de coordenadas e metadados.

### S4: Diagnóstico granular de preflight contra esquemas reais (P1)

Classificação exata dos estados de prontidão no diagnóstico prévio C06.

**Acceptance Criteria**

15. WHEN `diagnose_publication_readiness` evaluates an artifact whose columns do not match the expected real schema THEN the system SHALL assign status `SCHEMA_MISMATCH` with actionable details identifying missing columns.
16. IF an artifact or manifest is missing THEN the system SHALL assign status `FALTA`.

**Independent test:** executar diagnóstico contra fixtures deliberadamente incompletas e validar taxonomia dos status.

## Out of scope

| Excluded | Why |
| --- | --- |
| Execução de dados empíricos no cluster | Reservada para a etapa C07 após certificação dos testes locais |
| Mudança no modelo de proveniência de manifestos derivados | Objeto da spec C02 conforme cronograma de dependências |
| Alteração estrutural nos notebooks de reprodução | Invariante do AGENTS.md para preservação do histórico |

## Assumptions

| Assumption | Chosen default | Rationale | Confirmed? |
| --- | --- | --- | --- |
| Ingestão de CSVs comentados de F2 | Utilizar `pd.read_csv(..., comment='#')` | Mantém os avisos qualitativos obrigatórios no arquivo físico sem corromper a leitura tabular | y |
| Pivotagem de T12 | O adaptador T12 consolida as linhas `mean_logits` e `majority_vote` de `table_12_long_voter_comparison.csv` em colunas | Atende ao esquema C06 sem exigir nova execução de inferência em contigs longos | y |

**Open questions:** none - all resolved or logged above.

## Observable

| Surface | Decision | Landing |
| --- | --- | --- |
| Config `configs/reproduction_m07.yaml` | paths and schemas | AC 2, AC 3, AC 4, AC 5 |
| Module `viral_cnn.reporting.transformations` | error shape on missing columns | AC 6, AC 7, AC 14 |
| Module `viral_cnn.reporting.transformations` | schema mappings and conversions | AC 8, AC 9, AC 10, AC 11, AC 12 |
| Module `viral_cnn.paired_umap` / adapter | comment handling in projections | AC 13 |
| Document `docs/publication_data_contracts.md` | contract matrix specification | AC 1 |
| Function `diagnose_publication_readiness` | status classification | AC 15, AC 16 |

## Sources

- `PLANO_CORRECOES_CNN_METAGENOME_AGENTS_2026-11-15.md` - Planejamento de correções pré-empíricas e definição de escopo da spec C01
- `AGENTS.md` - Invariantes científicos, rastreabilidade de caminhos e proibição de medições inventadas
- `notebooks/reproduction/` - Cabeçalhos e formatos reais exportados pelos experimentos
