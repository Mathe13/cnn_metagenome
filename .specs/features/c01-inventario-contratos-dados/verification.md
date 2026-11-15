# C01 — Inventário e Contratos Executáveis de Dados de Publicação verification

**Verdict**: PASS
**Profile**: light
**Diff range**: f427723..HEAD
**Round**: 1 - full
**Verifier**: independent sub-agent (author != verifier)

## Checks

| Check | Claim | Proof run | Evidence | Result |
| --- | --- | --- | --- | --- |
| C1 | Matriz em docs/publication_data_contracts.md cobre 17 itens com notebooks, caminhos e esquemas | `.venv/bin/python -m unittest -k test_contract_matrix_completeness tests/test_c01_data_contracts.py` exit 0 | `tests/test_c01_data_contracts.py:67` | PASS |
| C2 | Fonte proposta para T2 aponta para short/01/table2_profiling.csv em configs/reproduction_m07.yaml | `.venv/bin/python -m unittest -k test_t2_proposed_source_path tests/test_c01_data_contracts.py` exit 0 | `tests/test_c01_data_contracts.py:78` | PASS |
| C3 | Fonte proposta para T7 aponta para short/02/table7_hierarchical_entropy.csv em configs/reproduction_m07.yaml | `.venv/bin/python -m unittest -k test_t7_proposed_source_path tests/test_c01_data_contracts.py` exit 0 | `tests/test_c01_data_contracts.py:85` | PASS |
| C4 | Fonte proposta para F2 aponta para short/06b/umap_projection_convnext_f.csv em configs/reproduction_m07.yaml | `.venv/bin/python -m unittest -k test_f2_proposed_source_path tests/test_c01_data_contracts.py` exit 0 | `tests/test_c01_data_contracts.py:92` | PASS |
| C5 | Lista de colunas esperadas para T11 não vazia com métricas de coocorrência de short/08 | `.venv/bin/python -m unittest -k test_t11_expected_columns_non_empty tests/test_c01_data_contracts.py` exit 0 | `tests/test_c01_data_contracts.py:99` | PASS |
| C6 | Transformação de T1 rejeita esquemas sem métricas com PublicationValidationError sem zero artificial | `.venv/bin/python -m unittest -k test_t1_rejects_missing_without_zero_fill tests/test_c01_data_contracts.py` exit 0 | `tests/test_c01_data_contracts.py:116` | PASS |
| C7 | Transformação de T2 mapeia colunas reais do profiler e rejeita faltantes sem preencher com 0.0 | `.venv/bin/python -m unittest -k test_t2_maps_profiling_columns_without_zero_fill tests/test_c01_data_contracts.py` exit 0 | `tests/test_c01_data_contracts.py:164` | PASS |
| C8 | Transformações de T3 e F1 convertem noise_rate para epsilon conservando métodos, estratos e N | `.venv/bin/python -m unittest -k test_t3_f1_converts_noise_rate_to_epsilon tests/test_c01_data_contracts.py` exit 0 | `tests/test_c01_data_contracts.py:179` | PASS |
| C9 | Transformação de T6 converte Transition para level_pair e preserva correlações | `.venv/bin/python -m unittest -k test_t6_converts_transition_to_level_pair tests/test_c01_data_contracts.py` exit 0 | `tests/test_c01_data_contracts.py:197` | PASS |
| C10 | Transformação de T9 e F4 mapeia codon para triplet preservando séries de leitura de frame | `.venv/bin/python -m unittest -k test_t9_maps_codon_to_triplet_and_frames tests/test_c01_data_contracts.py` exit 0 | `tests/test_c01_data_contracts.py:214` | PASS |
| C11 | Transformação de T10 preserva signal, codon, contagens observadas e fold enrichment | `.venv/bin/python -m unittest -k test_t10_preserves_signal_and_codon tests/test_c01_data_contracts.py` exit 0 | `tests/test_c01_data_contracts.py:236` | PASS |
| C12 | Transformação de T12 pivota linhas de agregação de build_c02_table12 em colunas agregadas | `.venv/bin/python -m unittest -k test_t12_pivots_variant_aggregation_rows tests/test_c01_data_contracts.py` exit 0 | `tests/test_c01_data_contracts.py:255` | PASS |
| C13 | Ingestão de projeção F2 processa CSV com comentários '#' extraindo coordenadas umap_1 e umap_2 | `.venv/bin/python -m unittest -k test_f2_parses_commented_projection_csv tests/test_c01_data_contracts.py` exit 0 | `tests/test_c01_data_contracts.py:272` | PASS |
| C14 | Ingestão de projeção F2 rejeita tabelas sem coordenadas com PublicationValidationError | `.venv/bin/python -m unittest -k test_f2_rejects_missing_coordinates tests/test_c01_data_contracts.py` exit 0 | `tests/test_c01_data_contracts.py:291` | PASS |
| C15 | diagnose_publication_readiness classifica colunas incompatíveis com status SCHEMA_MISMATCH | `.venv/bin/python -m unittest -k test_diagnose_readiness_schema_mismatch tests/test_c01_data_contracts.py` exit 0 | `tests/test_c01_data_contracts.py:335` | PASS |
| C16 | diagnose_publication_readiness classifica artefatos ausentes com status FALTA | `.venv/bin/python -m unittest -k test_diagnose_readiness_falta tests/test_c01_data_contracts.py` exit 0 | `tests/test_c01_data_contracts.py:353` | PASS |

## Gate

`.venv/bin/python -m unittest tests/test_c01_data_contracts.py` - 16 passed, 0 failed
