# C03 — Adaptadores de Tabelas e Figuras Basicas verification

**Verdict**: PASS
**Profile**: light
**Diff range**: HEAD
**Round**: 1 - full
**Verifier**: independent sub-agent (author != verifier)
**Checks**: `.specs/features/c03-adaptadores-tabelas-e-figuras-basicas/checks.md`

## Summary

- **Total checks**: 12 passed, 0 failed, 0 skipped
- **Suite duration**: 0.053s across 12 unit tests in `tests/test_c03_publication_adapters.py`
- **Regression verification**:
  - `tests/test_c01_data_contracts.py` and `tests/test_c02_derived_manifests.py`: 27 passed, 0 failed (0.444s)
  - `tests/test_m07_publication_contracts.py`: 16 passed, 0 failed (0.808s)

## Checks

| Check | Claim | Command Executed | Duration | Evidence | Result |
| --- | --- | --- | --- | --- | --- |
| C1 | A funcao `transform_table_1_metrics` produz as colunas canônicas completas a partir de medicoes reais sem preenchimento zero (AC 1) | `.venv/bin/python -m unittest -k test_t1_produces_complete_columns_without_zero_fill tests/test_c03_publication_adapters.py` | 0.006s | `tests/test_c03_publication_adapters.py:41` | PASS |
| C2 | A funcao `transform_table_1_metrics` rejeita medicoes faltantes ou nulas com PublicationValidationError (AC 2) | `.venv/bin/python -m unittest -k test_t1_rejects_missing_or_null_measurements tests/test_c03_publication_adapters.py` | 0.005s | `tests/test_c03_publication_adapters.py:65` | PASS |
| C3 | A funcao `transform_table_1_metrics` valida que valores de acuracia, top_5 e macro_f1 estejam contidos em [0.0, 1.0] ou [0.0, 100.0] (AC 3) | `.venv/bin/python -m unittest -k test_t1_validates_metric_bounds tests/test_c03_publication_adapters.py` | 0.009s | `tests/test_c03_publication_adapters.py:85` | PASS |
| C4 | A funcao `transform_table_2_profiling` mapeia as colunas reais do profiler (`total_params`, `backbone_params`, `head_params`, `throughput_seq_sec`, `avg_latency_ms`) rejeitando medicoes faltantes (AC 4) | `.venv/bin/python -m unittest -k test_t2_maps_profiler_columns_strictly tests/test_c03_publication_adapters.py` | 0.010s | `tests/test_c03_publication_adapters.py:116` | PASS |
| C5 | As funcoes `transform_table_3_noise_degradation` e `transform_figure_1_noise_robustness` convertem `noise_rate` para `epsilon` conservando modelos, estratos e suporte amostral N (AC 5) | `.venv/bin/python -m unittest -k test_t3_f1_noise_rate_conversion tests/test_c03_publication_adapters.py` | 0.003s | `tests/test_c03_publication_adapters.py:144` | PASS |
| C6 | A funcao `transform_table_6_hierarchy_correlations` mapeia `Transition` para `level_pair` preservando coeficientes de Pearson e Spearman (AC 6) | `.venv/bin/python -m unittest -k test_t6_transition_to_level_pair tests/test_c03_publication_adapters.py` | 0.003s | `tests/test_c03_publication_adapters.py:162` | PASS |
| C7 | A funcao `transform_table_12_long_sequence_consensus` pivota linhas por variante combinando `majority_vote` e `mean_logits` em colunas distintas (AC 7) | `.venv/bin/python -m unittest -k test_t12_pivots_variant_aggregations tests/test_c03_publication_adapters.py` | 0.014s | `tests/test_c03_publication_adapters.py:179` | PASS |
| C8 | A funcao `transform_table_12_long_sequence_consensus` rejeita variantes com agregacoes faltantes ou duplicadas com PublicationValidationError (AC 8) | `.venv/bin/python -m unittest -k test_t12_rejects_missing_or_duplicated_aggregations tests/test_c03_publication_adapters.py` | 0.007s | `tests/test_c03_publication_adapters.py:197` | PASS |
| C9 | As funcoes `transform_table_9_rscu_exploratory` e `transform_figure_4_rscu_profile` mapeiam `codon` para `triplet` distinguindo as series de frame 0, 3 frames e 6 frames (AC 9) | `.venv/bin/python -m unittest -k test_t9_f4_triplet_and_frames tests/test_c03_publication_adapters.py` | 0.003s | `tests/test_c03_publication_adapters.py:220` | PASS |
| C10 | A funcao `transform_table_10_triplet_composition` preserva `signal`, `codon`, contagens observadas e fold enrichment (AC 10) | `.venv/bin/python -m unittest -k test_t10_signal_codon_preservation tests/test_c03_publication_adapters.py` | 0.002s | `tests/test_c03_publication_adapters.py:241` | PASS |
| C11 | A funcao `transform_figure_2_umap_visualization` extrai coordenadas 2D e rotulos descartando comentarios `#` no cabecalho (AC 11) | `.venv/bin/python -m unittest -k test_f2_coordinates_and_comments tests/test_c03_publication_adapters.py` | 0.004s | `tests/test_c03_publication_adapters.py:257` | PASS |
| C12 | As transformacoes de T4, T5, T7 e T8 validam esquemas contra fixtures em conformidade com os contratos C01 (AC 12) | `.venv/bin/python -m unittest -k test_t4_t5_t7_t8_conformity tests/test_c03_publication_adapters.py` | 0.004s | `tests/test_c03_publication_adapters.py:281` | PASS |

## One-Way Doors Verified

1. **Door 1 — Proibição estrita de preenchimento artificial e validação de limites**:
   - `transform_table_1_metrics` e `transform_table_2_profiling` rejeitam peremptoriamente a falta de medições reais de latência, VRAM, throughput ou parâmetros com `PublicationValidationError`, proibindo preenchimento com zero artificial (`0.0`).
   - Limites de acurácia, top-5 e macro-F1 em T1 são rigorosamente validados no intervalo $[0.0, 1.0]$ ou $[0.0, 100.0]$, rejeitando anomalias ou valores fora de escala.
   - Verificado em C1 (`tests/test_c03_publication_adapters.py:41`), C2 (`tests/test_c03_publication_adapters.py:65`), C3 (`tests/test_c03_publication_adapters.py:85`) e C4 (`tests/test_c03_publication_adapters.py:116`).

2. **Door 2 — Pivot bidimensional determinístico de T12**:
   - `transform_table_12_long_sequence_consensus` mapeia linhas unpivoted `(variant, aggregation)` para linha única por variante com colunas canônicas `majority_vote_acc` e `mean_logits_acc`.
   - Rejeita variantes com agregações faltantes ou duplicadas com `PublicationValidationError`.
   - Verificado em C7 (`tests/test_c03_publication_adapters.py:179`) e C8 (`tests/test_c03_publication_adapters.py:197`).

3. **Door 3 — Rotulagem exploratória explícita de composição de tripletos em T9/F4 e motivos em T10**:
   - `transform_table_9_rscu_exploratory` e `transform_figure_4_rscu_profile` renomeiam explicitamente `codon` para `triplet`, distinguindo séries de leitura `rscu_frame0`, `rscu_3frames` e `rscu_6frames` sem atribuição biológica de CDS/ORF em fragmentos de 512 bp.
   - `transform_table_10_triplet_composition` preserva `signal`, `codon`, contagens observadas e fold enrichment como observações de frequência local.
   - Verificado em C9 (`tests/test_c03_publication_adapters.py:220`) e C10 (`tests/test_c03_publication_adapters.py:241`).

## Gate Execution

- Execução da suíte completa de testes de adaptadores C03:
  `.venv/bin/python -m unittest tests/test_c03_publication_adapters.py` — 12 passed, 0 failed in 0.053s.
- Execução das suítes de regressão C01, C02 e M07:
  - `tests/test_c01_data_contracts.py` e `tests/test_c02_derived_manifests.py` — 27 passed, 0 failed in 0.444s.
  - `tests/test_m07_publication_contracts.py` — 16 passed, 0 failed in 0.808s.
