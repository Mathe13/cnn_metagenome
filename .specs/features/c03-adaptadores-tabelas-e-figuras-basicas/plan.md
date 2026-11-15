# C03 — Adaptadores de Tabelas e Figuras Basicas

## Problem

Os dados intermediários e tabelas consolidadas geradas pelos notebooks de reprodução (`short/01`, `short/02`, `short/06`, `short/06b`, `short/07`, `short/08`, `short/09`, `long/03`) apresentam variações de esquema, nomenclaturas e cardinalidades em relação aos contratos de publicação estritos estabelecidos em C01 e C02:

1. **T1 e T2 (Zero-filling e métricas de inferência):**
   - Historicamente, transformadores de T1 e T2 preenchiam medições faltantes com `0.0` (como contagens de parâmetros, throughput ou latência).
   - O profiler de `short/01` exporta `table2_profiling.csv` com `total_params`, `backbone_params`, `head_params`, `throughput_seq_sec` e `avg_latency_ms`. O transformador deve mapear fielmente esses nomes reais para as colunas publicadas (`backbone_m`, `fc_params_m`, `parameters_m`, `throughput_seq_sec`, `latency_ms`), exigir a presença real de medições empíricas e rejeitar preenchimentos artificiais com zero.
   - Os limites de acurácia, top-5 e macro-F1 em T1 devem respeitar o intervalo $[0, 1]$ (ou $[0, 100]$ com normalização explícita).

2. **T3 e F1 (Perturbação de ruído):**
   - `short/03` e `short/09` exportam colunas com o identificador `noise_rate`. O contrato exige mapeamento canônico para `epsilon` ($\epsilon \in \{0, 0.02, 0.05, 0.10, 0.15\}$), preservando os estratos por arquitetura, cobertura e o suporte $N=1.706.270$ fragmentos.

3. **T6 (Correlações hierárquicas):**
   - O exportador em `short/02` emite `Transition` para pares taxonômicos (ex.: `Species -> Genus`, `Genus -> Order`). O contrato de publicação exige a coluna padronizada `level_pair`, conservando os coeficientes de correlação de Pearson e Spearman e o número de táxons avaliados.

4. **T9, T10 e F4 (Análise de tripletos e motivos):**
   - `short/08` exporta frequências com a coluna `codon` e séries relativas `rscu_frame0`, `rscu_3frames`, `rscu_6frames`.
   - Como fragmentos virais de 512 bp sem anotação de CDS não comprovam quadro de leitura aberto (ORF), os transformadores de T9 e F4 devem rotular explicitamente a métrica como **tripleto exploratório** (`triplet`), mantendo as séries de leitura distinguíveis.
   - T10 deve preservar `signal`, `codon`, contagens observadas e fold enrichment, demarcando ATG e códons de terminação como contagens locais sem atribuição funcional de início/fim de tradução biológica.

5. **T12 (Consenso em sequências longas):**
   - `long/03` (`build_c02_table12()`) exporta dados tabulares desnormalizados em múltiplas linhas por variante e por modo de agregação (`majority_vote` e `mean_logits`). O contrato de publicação de T12 exige uma linha única por variante com colunas pivotadas (`majority_vote_acc`, `mean_logits_acc`), validando completude dos modos e rejeitando agregações duplicadas ou faltantes.

6. **F2 (Visualização UMAP e coordenadas 2D):**
   - O artefato de projeção F2 gerado em `short/06b` (`umap_projection_convnext_f.csv`) contém coordenadas `umap_1`, `umap_2`, identificador de classe/táxon, modelo e metadados qualitativos. O transformador deve processar essas projeções, estruturar os pontos e alertar que projeções UMAP independentes não sustentam comparações de distâncias globais entre painéis.

7. **T4, T5, T7 e T8 (Validação de esquemas e contratos):**
   - T4 (F1 por faixa de espécies e comparações pareadas), T5 (métricas latentes KNN/Silhouette), T7 (entropia hierárquica por nível) e T8 (perturbação pareada de motivos com controle de GC |ΔGC|=0) devem ter seus adaptadores auditados e confirmados contra o inventário C01.

## Flow

1. Arquivo tabular do run selecionado entra em `execute_publication_transformation` (exists) - ler CSV/Parquet tratando comentários `#` no cabeçalho e validar formato tabular (door 1, door 2).
2. Dados brutos entram no adaptador específico registrado (`TRANSFORMATION_REGISTRY[item_id]`) (exists/updated) - aplicar conversão de nomes de colunas, verificar tipos, checar limites numéricos e rejeitar medições nulas ou preenchimentos fabricados (door 1, door 2, door 3).
3. DataFrame transformado e metadados entram em `emit_derived_manifest` (exists) - emitir artefato final em `output_dir` e manifesto derivado C02 com `derivation` completo e preservação de denominadores upstream (door 1, door 3).
4. Manifestos derivados entram em `validate_selection` (exists) - validar admissão estrita como `DERIVED_VERIFIED` e correspondência com as colunas esperadas em `publication_data_contracts.md` (door 1, door 2).

## Impact

| Front | What changes |
| --- | --- |
| domain | new term: `tabular_publication_adapters` - conjunto canônico de funções puras determinísticas mapeando formatos dos notebooks para esquemas do manuscrito em `src/viral_cnn/reporting/transformations.py` |
| domain | updated term: `table12_pivot` - pivot bidimensional de `(variant, aggregation) -> (variant, majority_vote_acc, mean_logits_acc)` com validação de unicidade de pares |
| stored data | nada a migrar |

## Relations

`None - no stored-data shape change`

## Surface

| Route | In | Out | Status |
| --- | --- | --- | --- |
| `viral_cnn.reporting.transformations:transform_table_1_metrics(df, meta)` | `pd.DataFrame`, `dict` | `pd.DataFrame` (T1) | `200`, `400` |
| `viral_cnn.reporting.transformations:transform_table_2_profiling(df, meta)` | `pd.DataFrame`, `dict` | `pd.DataFrame` (T2) | `200`, `400` |
| `viral_cnn.reporting.transformations:transform_table_3_noise_degradation(df, meta)` | `pd.DataFrame`, `dict` | `pd.DataFrame` (T3) | `200`, `400` |
| `viral_cnn.reporting.transformations:transform_table_6_hierarchy_correlations(df, meta)` | `pd.DataFrame`, `dict` | `pd.DataFrame` (T6) | `200`, `400` |
| `viral_cnn.reporting.transformations:transform_table_9_rscu_exploratory(df, meta)` | `pd.DataFrame`, `dict` | `pd.DataFrame` (T9) | `200`, `400` |
| `viral_cnn.reporting.transformations:transform_table_10_triplet_composition(df, meta)` | `pd.DataFrame`, `dict` | `pd.DataFrame` (T10) | `200`, `400` |
| `viral_cnn.reporting.transformations:transform_table_12_long_sequence_consensus(df, meta)` | `pd.DataFrame`, `dict` | `pd.DataFrame` (T12) | `200`, `400` |
| `viral_cnn.reporting.transformations:transform_figure_1_noise_robustness(df, meta)` | `pd.DataFrame`, `dict` | `pd.DataFrame` (F1) | `200`, `400` |
| `viral_cnn.reporting.transformations:transform_figure_2_umap_visualization(df, meta)` | `pd.DataFrame`, `dict` | `pd.DataFrame` (F2) | `200`, `400` |
| `viral_cnn.reporting.transformations:transform_figure_4_rscu_profile(df, meta)` | `pd.DataFrame`, `dict` | `pd.DataFrame` (F4) | `200`, `400` |

## Landing

| One-way door | Literal shape | Alternative rejected |
| --- | --- | --- |
| 1 — Proibição estrita de preenchimento artificial e validação de limites | Adaptadores de T1 e T2 rejeitam peremptoriamente a falta de medições reais de latência, VRAM, throughput ou parâmetros com PublicationValidationError; limites de acurácia/F1 são estritamente validados em [0, 1] ou [0, 100]. | Preencher valores não medidos com 0.0 para compatibilizar tabelas sintéticas incompletas. |
| 2 — Pivot bidimensional determinístico de T12 | O adaptador de T12 transforma pares de linhas (variant × aggregation) em uma linha única por variante com colunas `majority_vote_acc` e `mean_logits_acc`, exigindo ambos os agregadores e rejeitando duplicatas. | Tratar cada linha de agregação como uma entrada independente com esquemas divergentes. |
| 3 — Rotulagem exploratória explícita de composição de tripletos em T9/F4 e motivos em T10 | O adaptador de T9 e F4 renomeia `codon` para `triplet` preservando as séries de leitura de frame 0, 3 frames e 6 frames; T10 preserva `signal` e `codon` com demarcação observacional de contagem de bases. | Apresentar contagens de tripletos como códons em quadro de leitura confirmado ou inferir anotação gênica em fragmentos de 512 bp. |

## Criteria

### S1: Adaptadores de desempenho e perfilamento computacional T1 e T2 (P1)

Garantir integridade métrica e ausência de dados fabricados para T1 e T2.

**Acceptance Criteria**

1. WHEN `transform_table_1_metrics` receives a DataFrame with real performance measurements THEN the system SHALL output columns `["model", "taxonomic_level", "backbone_m", "fc_params_m", "parameters_m", "accuracy", "top_5", "macro_f1", "latency_ms"]` without adding zero-filled values.
2. IF any required performance metric in T1 is missing or contains null values THEN the system SHALL raise `PublicationValidationError`.
3. IF any accuracy, top-5 or macro-F1 value in T1 is outside the closed interval $[0.0, 1.0]$ (or $[0.0, 100.0]$) THEN the system SHALL raise `PublicationValidationError`.
4. WHEN `transform_table_2_profiling` executes THEN the system SHALL map profiler columns (`total_params`, `backbone_params`, `head_params`, `throughput_seq_sec`, `avg_latency_ms`) to published profiling columns, rejecting missing measurements.

**Independent test:** testar T1 e T2 com dados completos do profiler e com dados faltantes/fora de limites, verificando conformidade e exceções.

### S2: Adaptadores de ruído, transição hierárquica e consenso longo T3, T6, T12 e F1 (P1)

Mapeamento canônico de variáveis e pivot estruturado.

**Acceptance Criteria**

5. WHEN `transform_table_3_noise_degradation` and `transform_figure_1_noise_robustness` execute THEN the system SHALL convert `noise_rate` to `epsilon` preserving evaluated models, error rate intervals ($3\epsilon/4$), and sample support $N$.
6. WHEN `transform_table_6_hierarchy_correlations` executes THEN the system SHALL map `Transition` to `level_pair` and preserve Pearson and Spearman correlation coefficients.
7. WHEN `transform_table_12_long_sequence_consensus` executes THEN the system SHALL pivot rows by `variant` combining `majority_vote` and `mean_logits` into columns `majority_vote_acc` and `mean_logits_acc`.
8. IF a variant in T12 is missing either `majority_vote` or `mean_logits`, or has duplicated aggregations THEN the system SHALL raise `PublicationValidationError`.

**Independent test:** executar T3, T6, T12 e F1 sobre fixtures com esquemas brutos e verificar colunas mapeadas e integridade do pivot.

### S3: Adaptadores de tripletos, motivos e projeções UMAP T9, T10, F2 e F4 (P1)

Demarcação de composição exploratória e consumo de projeções.

**Acceptance Criteria**

9. WHEN `transform_table_9_rscu_exploratory` and `transform_figure_4_rscu_profile` execute THEN the system SHALL map `codon` to `triplet` and distinguish reading series `rscu_frame0`, `rscu_3frames`, and `rscu_6frames`.
10. WHEN `transform_table_10_triplet_composition` executes THEN the system SHALL preserve `signal`, `codon`, observed counts, and fold enrichment without biological translation claims.
11. WHEN `transform_figure_2_umap_visualization` executes THEN the system SHALL ingest 2D coordinates `umap_1`, `umap_2`, model name, and genus labels, stripping initial `#` comments if present.
12. WHEN validating T4, T5, T7, and T8 THEN the system SHALL verify that their schema transformations satisfy the data contract inventory established in C01.

**Independent test:** verificar a saída de T9, T10, F2, F4 e auditoria de T4/T5/T7/T8 contra fixtures representativas.

## Out of scope

| Excluded | Why |
| --- | --- |
| Decisão e contrato específico de T11 | Objeto exclusivo da spec C04 |
| Renderização final e legendas de F3/F5 | Objeto exclusivo da spec C05 |
| Execução de inferência no cluster | Objeto exclusivo da spec C07 |


## Assumptions

| Assumption | Chosen default | Rationale | Confirmed? |
| --- | --- | --- | --- |
| Limites de acurácia em T1 | Aceitar intervalo $[0.0, 1.0]$ ou $[0.0, 100.0]$ | Permite valores em frações unitárias ou porcentagens inteiras sem distorção numérica | y |
| Pivotagem estrita de T12 | Exigir exatamente as duas agregações (`majority_vote` e `mean_logits`) por variante | Garante paridade comparativa entre voto e média de logits pré-softmax conforme método publicado | y |
| Rotulagem de tripletos em T9 e F4 | Rótulo explícito `triplet` com preservação das séries de leitura | Alinha o vocabulário à demarcação científica de que fragmentos de 512 bp são exploratórios | y |

**Open questions:** none - all resolved or logged above.

## Observable

| Surface | Decision | Landing |
| --- | --- | --- |
| Module `viral_cnn.reporting.transformations:transform_table_1_metrics` | rejeição de valores nulos e validação de limites $[0, 1]$ ou $[0, 100]$ | AC 1, AC 2, AC 3 |
| Module `viral_cnn.reporting.transformations:transform_table_2_profiling` | mapeamento de colunas do profiler e rejeição de preenchimento zero | AC 4 |
| Module `viral_cnn.reporting.transformations:transform_table_3_noise_degradation` / F1 | conversão canônica de `noise_rate` para `epsilon` | AC 5 |
| Module `viral_cnn.reporting.transformations:transform_table_6_hierarchy_correlations` | mapeamento de `Transition` para `level_pair` | AC 6 |
| Module `viral_cnn.reporting.transformations:transform_table_12_long_sequence_consensus` | pivot determinístico por variante de `majority_vote` e `mean_logits` | AC 7, AC 8 |
| Module `viral_cnn.reporting.transformations:transform_table_9_rscu_exploratory` / F4 | mapeamento `codon -> triplet` e séries de leitura | AC 9 |
| Module `viral_cnn.reporting.transformations:transform_table_10_triplet_composition` | conservação de `signal` e `codon` com contagens locais | AC 10 |
| Module `viral_cnn.reporting.transformations:transform_figure_2_umap_visualization` | ingestão e limpeza de coordenadas 2D de projeção F2 | AC 11 |
| Module `viral_cnn.reporting.transformations` (T4, T5, T7, T8) | auditoria e conformidade com inventário C01 | AC 12 |

## Sources

- `PLANO_CORRECOES_CNN_METAGENOME_AGENTS_2026-11-15.md` - Planejamento de correções pré-empíricas e definição de escopo da spec C03
- `docs/publication_data_contracts.md` - Matriz executável de contratos de dados de publicação
- `.specs/features/c01-inventario-contratos-dados/plan.md` - Especificação de contratos executáveis C01
- `.specs/features/c02-manifestos-derivados-e-suporte/plan.md` - Especificação de manifestos derivados e suporte experimental C02
