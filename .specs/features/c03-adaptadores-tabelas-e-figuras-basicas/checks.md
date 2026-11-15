# C03 — Adaptadores de Tabelas e Figuras Basicas checks

Profile: light
Plan: `.specs/features/c03-adaptadores-tabelas-e-figuras-basicas/plan.md`

12 checks in 3 slices · 3 one-way doors · 0 open, of which 0 block

## Checks

Grouped by the plan's slices; numbering runs across the whole feature.

### S1 - Adaptadores de desempenho e perfilamento computacional T1 e T2 · 2 files · 40 KB · ~10k

**C1** - A funcao `transform_table_1_metrics` produz as colunas canônicas completas a partir de medicoes reais sem preenchimento zero (AC 1)
Proof: `.venv/bin/python -m unittest -k test_t1_produces_complete_columns_without_zero_fill tests/test_c03_publication_adapters.py`

**C2** - A funcao `transform_table_1_metrics` rejeita medicoes faltantes ou nulas com PublicationValidationError (AC 2)
Proof: `.venv/bin/python -m unittest -k test_t1_rejects_missing_or_null_measurements tests/test_c03_publication_adapters.py`

**C3** - A funcao `transform_table_1_metrics` valida que valores de acuracia, top_5 e macro_f1 estejam contidos em [0.0, 1.0] ou [0.0, 100.0] (AC 3)
Proof: `.venv/bin/python -m unittest -k test_t1_validates_metric_bounds tests/test_c03_publication_adapters.py`

**C4** - A funcao `transform_table_2_profiling` mapeia as colunas reais do profiler (`total_params`, `backbone_params`, `head_params`, `throughput_seq_sec`, `avg_latency_ms`) rejeitando medicoes faltantes (AC 4)
Proof: `.venv/bin/python -m unittest -k test_t2_maps_profiler_columns_strictly tests/test_c03_publication_adapters.py`

### S2 - Adaptadores de ruido, transicao hierarquica e consenso longo T3, T6, T12 e F1 · 2 files · 40 KB · ~10k

**C5** - As funcoes `transform_table_3_noise_degradation` e `transform_figure_1_noise_robustness` convertem `noise_rate` para `epsilon` conservando modelos, estratos e suporte amostral N (AC 5)
Proof: `.venv/bin/python -m unittest -k test_t3_f1_noise_rate_conversion tests/test_c03_publication_adapters.py`

**C6** - A funcao `transform_table_6_hierarchy_correlations` mapeia `Transition` para `level_pair` preservando coeficientes de Pearson e Spearman (AC 6)
Proof: `.venv/bin/python -m unittest -k test_t6_transition_to_level_pair tests/test_c03_publication_adapters.py`

**C7** - A funcao `transform_table_12_long_sequence_consensus` pivota linhas por variante combinando `majority_vote` e `mean_logits` em colunas distintas (AC 7)
Proof: `.venv/bin/python -m unittest -k test_t12_pivots_variant_aggregations tests/test_c03_publication_adapters.py`

**C8** - A funcao `transform_table_12_long_sequence_consensus` rejeita variantes com agregacoes faltantes ou duplicadas com PublicationValidationError (AC 8)
Proof: `.venv/bin/python -m unittest -k test_t12_rejects_missing_or_duplicated_aggregations tests/test_c03_publication_adapters.py`

### S3 - Adaptadores de tripletos, motivos e projecoes UMAP T9, T10, F2 e F4 · 2 files · 40 KB · ~10k

**C9** - As funcoes `transform_table_9_rscu_exploratory` e `transform_figure_4_rscu_profile` mapeiam `codon` para `triplet` distinguindo as series de frame 0, 3 frames e 6 frames (AC 9)
Proof: `.venv/bin/python -m unittest -k test_t9_f4_triplet_and_frames tests/test_c03_publication_adapters.py`

**C10** - A funcao `transform_table_10_triplet_composition` preserva `signal`, `codon`, contagens observadas e fold enrichment (AC 10)
Proof: `.venv/bin/python -m unittest -k test_t10_signal_codon_preservation tests/test_c03_publication_adapters.py`

**C11** - A funcao `transform_figure_2_umap_visualization` extrai coordenadas 2D e rotulos descartando comentarios `#` no cabecalho (AC 11)
Proof: `.venv/bin/python -m unittest -k test_f2_coordinates_and_comments tests/test_c03_publication_adapters.py`

**C12** - As transformacoes de T4, T5, T7 e T8 validam esquemas contra fixtures em conformidade com os contratos C01 (AC 12)
Proof: `.venv/bin/python -m unittest -k test_t4_t5_t7_t8_conformity tests/test_c03_publication_adapters.py`

## Coverage

| Set (size) | Member -> proof | Unproven |
| --- | --- | --- |
| Desempenho e profiling (4) | t1_colunas_completas C1 · t1_rejeita_nulos C2 · t1_limites_metricas C3 · t2_profiler_estrito C4 | - |
| Ruido, transicao e consenso (4) | ruido_conversao C5 · transicao_hierarquica C6 · consenso_pivot C7 · consenso_rejeicao_incompleto C8 | - |
| Tripletos, motivos e UMAP (4) | rscu_tripletos C9 · motivos_sinais C10 · umap_projecoes C11 · auditoria_t4_t5_t7_t8 C12 | - |

## Swept

Where each unwritten requirement landed. All nine, one line each, every time.

- validation: C2, C3, C4, C8
- failure modes: C2, C3, C8
- idempotency: existing - adaptadores sao funcoes puras deterministas sem efeitos colaterais
- authorization: n/a - processamento em memoria e disco local sem autenticacao de rede
- concurrency: n/a - pipeline sequencial e monotarefa
- data lifecycle: n/a - tabelas transformadas sao salvas em diretorios de execucao sem purga
- dependency failure: C2, C8
- state transitions: C7
- observability: existing - registro em log dos artefatos processados pelo logger de publicacao

## Handoff

S1-S3 = ~35k tokens acumulados (transformations, publication, tests), bem abaixo do orcamento de 150k - one builder.

- Boundary: C1-C12 fechados em conjunto por um unico builder
