# C01 — Inventário e Contratos Executáveis de Dados de Publicação checks

Profile: light
Plan: `.specs/features/c01-inventario-contratos-dados/plan.md`

16 checks in 4 slices · 4 one-way doors · 0 open, of which 0 block

## Checks

Grouped by the plan's slices; numbering runs across the whole feature.

### S1 - Matriz versionada de contratos e correção de caminhos propostos · 2 files · 23 KB · ~6k

**C1** - A matriz de contratos cobre todos os 17 itens (T1–T12, F1–F5) com notebook produtor, caminho relativo no run, manifesto pai, colunas realmente exportadas e denominadores em `docs/publication_data_contracts.md` (AC 1)
Proof: `.venv/bin/python -m unittest -k test_contract_matrix_completeness tests/test_c01_data_contracts.py`

**C2** - A fonte proposta para T2 aponta para `results_reproduction/runs/<run_id>/short/01/table2_profiling.csv` em `configs/reproduction_m07.yaml` (AC 2)
Proof: `.venv/bin/python -m unittest -k test_t2_proposed_source_path tests/test_c01_data_contracts.py`

**C3** - A fonte proposta para T7 aponta para `results_reproduction/runs/<run_id>/short/02/table7_hierarchical_entropy.csv` em `configs/reproduction_m07.yaml` (AC 3)
Proof: `.venv/bin/python -m unittest -k test_t7_proposed_source_path tests/test_c01_data_contracts.py`

**C4** - A fonte proposta para F2 aponta para `results_reproduction/runs/<run_id>/short/06b/umap_projection_convnext_f.csv` em `configs/reproduction_m07.yaml` (AC 4)
Proof: `.venv/bin/python -m unittest -k test_f2_proposed_source_path tests/test_c01_data_contracts.py`

**C5** - A lista de colunas esperadas para T11 é não vazia e reflete as colunas da análise de coocorrência de `short/08` em `configs/reproduction_m07.yaml` (AC 5)
Proof: `.venv/bin/python -m unittest -k test_t11_expected_columns_non_empty tests/test_c01_data_contracts.py`

### S2 - Adaptadores fiéis aos esquemas reais e rejeição de preenchimentos nulos · 2 files · 40 KB · ~10k

**C6** - A transformação de T1 rejeita esquemas sem as métricas obrigatórias com PublicationValidationError em vez de preencher valores com 0.0 (AC 6)
Proof: `.venv/bin/python -m unittest -k test_t1_rejects_missing_without_zero_fill tests/test_c01_data_contracts.py`

**C7** - A transformação de T2 mapeia as colunas reais do profiler (`total_params`, `backbone_params`, `head_params`, `throughput_seq_sec`, `avg_latency_ms`) e rejeita medições faltantes sem preencher com 0.0 (AC 7)
Proof: `.venv/bin/python -m unittest -k test_t2_maps_profiling_columns_without_zero_fill tests/test_c01_data_contracts.py`

**C8** - As transformações de T3 e F1 convertem `noise_rate` para `epsilon` conservando os métodos avaliados, estratos e suporte amostral N (AC 8)
Proof: `.venv/bin/python -m unittest -k test_t3_f1_converts_noise_rate_to_epsilon tests/test_c01_data_contracts.py`

**C9** - A transformação de T6 converte `Transition` para `level_pair` e preserva correlações de Pearson e Spearman (AC 9)
Proof: `.venv/bin/python -m unittest -k test_t6_converts_transition_to_level_pair tests/test_c01_data_contracts.py`

**C10** - A transformação de T9 e F4 mapeia `codon` para `triplet` preservando as séries de leitura de frame 0, 3 frames e 6 frames (AC 10)
Proof: `.venv/bin/python -m unittest -k test_t9_maps_codon_to_triplet_and_frames tests/test_c01_data_contracts.py`

**C11** - A transformação de T10 preserva as colunas `signal`, `codon`, contagens observadas e fold enrichment (AC 11)
Proof: `.venv/bin/python -m unittest -k test_t10_preserves_signal_and_codon tests/test_c01_data_contracts.py`

**C12** - A transformação de T12 pivota as linhas de variante por agregação de `build_c02_table12` em uma linha por variante com colunas `majority_vote_acc` e `mean_logits_acc` (AC 12)
Proof: `.venv/bin/python -m unittest -k test_t12_pivots_variant_aggregation_rows tests/test_c01_data_contracts.py`

### S3 - Consumo de projeções F2 com comentários qualitativos · 2 files · 27 KB · ~7k

**C13** - A ingestão de projeção F2 processa arquivos CSV com comentários `#` no cabeçalho extraindo coordenadas `umap_1`, `umap_2` e metadados (AC 13)
Proof: `.venv/bin/python -m unittest -k test_f2_parses_commented_projection_csv tests/test_c01_data_contracts.py`

**C14** - A ingestão de projeção F2 rejeita tabelas sem colunas de coordenadas com PublicationValidationError (AC 14)
Proof: `.venv/bin/python -m unittest -k test_f2_rejects_missing_coordinates tests/test_c01_data_contracts.py`

### S4 - Diagnóstico granular de preflight contra esquemas reais · 2 files · 40 KB · ~10k

**C15** - A função `diagnose_publication_readiness` classifica artefatos com colunas incompatíveis com o status SCHEMA_MISMATCH identificando a coluna ausente (AC 15)
Proof: `.venv/bin/python -m unittest -k test_diagnose_readiness_schema_mismatch tests/test_c01_data_contracts.py`

**C16** - A função `diagnose_publication_readiness` classifica artefatos ausentes com o status FALTA (AC 16)
Proof: `.venv/bin/python -m unittest -k test_diagnose_readiness_falta tests/test_c01_data_contracts.py`

## Coverage

| Set (size) | Member -> proof | Unproven |
| --- | --- | --- |
| Itens de publicação (17) | T1 C6 · T2 C7 · T3 C8 · T4 C15 · T5 C15 · T6 C9 · T7 C3 · T8 C15 · T9 C10 · T10 C11 · T11 C5 · T12 C12 · F1 C8 · F2 C13 · F3 C15 · F4 C10 · F5 C5 | - |
| Caminhos corrigidos de propostas (3) | T2 C2 · T7 C3 · F2 C4 | - |
| Mapeamento de colunas reais (4) | noise_rate->epsilon C8 · Transition->level_pair C9 · codon->triplet C10 · pivot_t12 C12 | - |
| Ingestão de projeções F2 (2) | cabecalho_comentado C13 · coordenadas_ausentes C14 | - |
| Estados de preflight diagnosticados (2) | SCHEMA_MISMATCH C15 · FALTA C16 | - |
| Rejeição de zero inventado (2) | T1 C6 · T2 C7 | - |

- Claims naming a status code, route or response shape: C15, C16 - each has a proof that crosses the boundary
- No other check claims more than the single case its proof exercises

## Swept

Where each unwritten requirement landed. All nine, one line each, every time.

- validation: C6, C7, C14
- failure modes: C6, C7, C14
- idempotency: existing - as transformações de dados são funções puras com saídas determinísticas
- authorization: n/a - execução de pipeline local sem autenticação de rede
- concurrency: n/a - pipeline síncrono e monotarefa
- data lifecycle: n/a - artefatos intermediários persistem no diretório de execução sem purga temporal
- dependency failure: C16
- state transitions: C15, C16
- observability: existing - chamadas a log_artifact_path registram todos os caminhos consumidos e gerados

## Handoff

S1-S4 = 33k tokens acumulados (configs, transformations, publication, paired_umap, docs, tests), bem abaixo do orçamento de 150k - one builder.

- Boundary: C1-C16 fechados em conjunto por um único builder
- Settled mid-build: none
- Abandoned: none
