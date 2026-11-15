# C04 — Decisao Cientifica e Contrato Executavel de T11 checks

Profile: light
Plan: `.specs/features/c04-decisao-cientifica-t11/plan.md`

10 checks in 3 slices · 3 one-way doors · 0 open, of which 0 block

## Checks

Grouped by the plan's slices; numbering runs across the whole feature.

### S1 - Formalizacao da decisao cientifica e colunas esperadas de T11 · 2 files · 40 KB · ~10k

**C1** - A lista `expected.columns` para T11 em `configs/reproduction_m07.yaml` e `docs/publication_data_contracts.md` define as 10 colunas canônicas da coocorrência de motivos (AC 1)
Proof: `.venv/bin/python -m unittest -k test_t11_expected_columns_canonical tests/test_c04_t11_scientific_decision.py`

**C2** - A funcao `transform_table_11_architecture_correlations` emite as colunas completas da sinergia de coocorrência a partir de dados válidos de `short/08` (AC 2)
Proof: `.venv/bin/python -m unittest -k test_t11_transforms_valid_cooccurrence_data tests/test_c04_t11_scientific_decision.py`

**C3** - A documentacao em `docs/publication_data_contracts.md` registra a decisao metodologica demarcando T11 como coocorrência com modelo nulo de nós e nao correlacao entre modelos (AC 3)
Proof: `.venv/bin/python -m unittest -k test_t11_documentation_decision_contract tests/test_c04_t11_scientific_decision.py`

### S2 - Rejeicao de correlacoes espurias e exigencia do teste nulo · 2 files · 40 KB · ~10k

**C4** - A funcao `transform_table_11_architecture_correlations` rejeita entradas sem `p_permutation_null` e sem `is_permutation_tested` com PublicationValidationError (AC 4)
Proof: `.venv/bin/python -m unittest -k test_t11_rejects_missing_permutation_test tests/test_c04_t11_scientific_decision.py`

**C5** - A funcao `transform_table_11_architecture_correlations` rejeita alegacoes de $r=1.0$ e $p=0.0$ oriundas de matrizes identicas ou sem variância com PublicationValidationError (AC 5)
Proof: `.venv/bin/python -m unittest -k test_t11_rejects_spurious_r1_zero_variance tests/test_c04_t11_scientific_decision.py`

**C6** - A funcao `transform_table_11_architecture_correlations` rejeita entradas onde `is_permutation_tested` contem valores falsos com PublicationValidationError (AC 6)
Proof: `.venv/bin/python -m unittest -k test_t11_rejects_false_permutation_flag tests/test_c04_t11_scientific_decision.py`

**C7** - A funcao `transform_table_11_architecture_correlations` rejeita entradas com contagem de permutacoes validas inferior a 100 com PublicationValidationError (AC 7)
Proof: `.venv/bin/python -m unittest -k test_t11_rejects_insufficient_permutations tests/test_c04_t11_scientific_decision.py`

### S3 - Integracao no preflight de publicacao e no validador C06 · 2 files · 40 KB · ~10k

**C8** - A funcao `diagnose_publication_readiness` classifica artefato de T11 sem teste nulo de permutacao com status SCIENTIFIC_MISMATCH (AC 8)
Proof: `.venv/bin/python -m unittest -k test_diagnose_readiness_t11_missing_permutation tests/test_c04_t11_scientific_decision.py`

**C9** - A funcao `diagnose_publication_readiness` classifica artefato de T11 valido acompanhado de teste nulo com status OK (AC 9)
Proof: `.venv/bin/python -m unittest -k test_diagnose_readiness_t11_valid_cooccurrence tests/test_c04_t11_scientific_decision.py`

**C10** - A funcao `validate_selection` aceita T11 com status DERIVED_VERIFIED quando acompanhada de evidencia empirica valida de coocorrência e teste nulo (AC 10)
Proof: `.venv/bin/python -m unittest -k test_validate_selection_t11_derived_verified tests/test_c04_t11_scientific_decision.py`

## Coverage

| Set (size) | Member -> proof | Unproven |
| --- | --- | --- |
| Contrato e colunas T11 (3) | colunas_canonicas C1 · transformacao_valida C2 · documentacao_decisao C3 | - |
| Rejeicao e integridade inferencial (4) | rejeicao_sem_teste_nulo C4 · rejeicao_r1_espúrio C5 · rejeicao_flag_falso C6 · rejeicao_poucas_permutacoes C7 | - |
| Preflight e C06 (3) | diagnose_mismatch C8 · diagnose_ok C9 · validate_selection_derived C10 | - |

- Claims naming a status code, route or response shape: C8, C9, C10 - each has a proof that crosses the boundary
- No other check claims more than the single case its proof exercises

## Swept

Where each unwritten requirement landed. All nine, one line each, every time.

- validation: C4, C5, C6, C7
- failure modes: C4, C5, C6, C7
- idempotency: existing - transformacao deterministica sem efeitos colaterais
- authorization: n/a - processamento em memoria e disco local sem autenticacao de rede
- concurrency: n/a - pipeline sequencial e monotarefa
- data lifecycle: n/a - tabelas transformadas sao salvas em diretorios de execucao sem purga
- dependency failure: C4, C8
- state transitions: C8, C9, C10
- observability: existing - registro em log dos artefatos processados pelo logger de publicacao

## Handoff

S1-S3 = ~30k tokens acumulados (configs, documentation, transformations, publication, tests), bem abaixo do orcamento de 150k - one builder.

- Boundary: C1-C10 fechados em conjunto por um unico builder
