# C04 — Decisao Cientifica e Contrato Executavel de T11 verification

**Verdict**: PASS
**Profile**: light
**Diff range**: HEAD
**Round**: 1 - full
**Verifier**: independent sub-agent (author != verifier)
**Checks**: `.specs/features/c04-decisao-cientifica-t11/checks.md`

## Summary

- **Total checks**: 10 passed, 0 failed, 0 skipped
- **Suite duration**: 0.088s across 10 unit tests in `tests/test_c04_t11_scientific_decision.py`
- **Regression verification**:
  - `tests/test_c01_data_contracts.py`, `tests/test_c02_derived_manifests.py`, and `tests/test_c03_publication_adapters.py`: 39 passed, 0 failed (0.470s)

## Checks

| Check | Claim | Command Executed | Duration | Evidence | Result |
| --- | --- | --- | --- | --- | --- |
| C1 | A lista `expected.columns` para T11 em `configs/reproduction_m07.yaml` e `docs/publication_data_contracts.md` define as 10 colunas canônicas da coocorrência de motivos (AC 1) | `.venv/bin/python -m unittest -k test_t11_expected_columns_canonical tests/test_c04_t11_scientific_decision.py` | 0.024s | `tests/test_c04_t11_scientific_decision.py:64` | PASS |
| C2 | A funcao `transform_table_11_architecture_correlations` emite as colunas completas da sinergia de coocorrência a partir de dados válidos de `short/08` (AC 2) | `.venv/bin/python -m unittest -k test_t11_transforms_valid_cooccurrence_data tests/test_c04_t11_scientific_decision.py` | 0.005s | `tests/test_c04_t11_scientific_decision.py:75` | PASS |
| C3 | A documentacao em `docs/publication_data_contracts.md` registra a decisao metodologica demarcando T11 como coocorrência com modelo nulo de nós e nao correlacao entre modelos (AC 3) | `.venv/bin/python -m unittest -k test_t11_documentation_decision_contract tests/test_c04_t11_scientific_decision.py` | 0.001s | `tests/test_c04_t11_scientific_decision.py:89` | PASS |
| C4 | A funcao `transform_table_11_architecture_correlations` rejeita entradas sem `p_permutation_null` e sem `is_permutation_tested` com PublicationValidationError (AC 4) | `.venv/bin/python -m unittest -k test_t11_rejects_missing_permutation_test tests/test_c04_t11_scientific_decision.py` | 0.002s | `tests/test_c04_t11_scientific_decision.py:100` | PASS |
| C5 | A funcao `transform_table_11_architecture_correlations` rejeita alegacoes de $r=1.0$ e $p=0.0$ oriundas de matrizes identicas ou sem variância com PublicationValidationError (AC 5) | `.venv/bin/python -m unittest -k test_t11_rejects_spurious_r1_zero_variance tests/test_c04_t11_scientific_decision.py` | 0.002s | `tests/test_c04_t11_scientific_decision.py:112` | PASS |
| C6 | A funcao `transform_table_11_architecture_correlations` rejeita entradas onde `is_permutation_tested` contem valores falsos com PublicationValidationError (AC 6) | `.venv/bin/python -m unittest -k test_t11_rejects_false_permutation_flag tests/test_c04_t11_scientific_decision.py` | 0.003s | `tests/test_c04_t11_scientific_decision.py:126` | PASS |
| C7 | A funcao `transform_table_11_architecture_correlations` rejeita entradas com contagem de permutacoes validas inferior a 100 com PublicationValidationError (AC 7) | `.venv/bin/python -m unittest -k test_t11_rejects_insufficient_permutations tests/test_c04_t11_scientific_decision.py` | 0.003s | `tests/test_c04_t11_scientific_decision.py:137` | PASS |
| C8 | A funcao `diagnose_publication_readiness` classifica artefato de T11 sem teste nulo de permutacao com status SCIENTIFIC_MISMATCH (AC 8) | `.venv/bin/python -m unittest -k test_diagnose_readiness_t11_missing_permutation tests/test_c04_t11_scientific_decision.py` | 0.009s | `tests/test_c04_t11_scientific_decision.py:146` | PASS |
| C9 | A funcao `diagnose_publication_readiness` classifica artefato de T11 valido acompanhado de teste nulo com status OK (AC 9) | `.venv/bin/python -m unittest -k test_diagnose_readiness_t11_valid_cooccurrence tests/test_c04_t11_scientific_decision.py` | 0.006s | `tests/test_c04_t11_scientific_decision.py:185` | PASS |
| C10 | A funcao `validate_selection` aceita T11 com status DERIVED_VERIFIED quando acompanhada de evidencia empirica valida de coocorrência e teste nulo (AC 10) | `.venv/bin/python -m unittest -k test_validate_selection_t11_derived_verified tests/test_c04_t11_scientific_decision.py` | 0.030s | `tests/test_c04_t11_scientific_decision.py:219` | PASS |

## One-Way Doors Verified

1. **Door 1 — Demarcação científica de T11 como coocorrência com nulo de permutação**:
   - T11 é formalmente definida como tabela de coocorrência de pares de 7-mers e escore RSCU controlada por distribuição nula de permutação de nós do grafo (`short/08`), rejeitando a alegação histórica infundada de correlação inter-modelos baseada em sequências compartilhadas ou matrizes idênticas.
   - O esquema canônico exige as 10 colunas canônicas: `[status, motif_score, n_valid_edges, n_valid_permutations, r_canonical, r_reversed, delta_r_orientation, icc_symmetric, r_bidirectional, p_permutation_null]`.
   - Diagnósticos de sobreposição de Jaccard entre variantes permanecem estritamente descritivos em `table_11b_jaccard_variant_diagnostics.csv`.
   - Verificado em C1 (`tests/test_c04_t11_scientific_decision.py:64`), C2 (`tests/test_c04_t11_scientific_decision.py:75`) e C3 (`tests/test_c04_t11_scientific_decision.py:89`).

2. **Door 2 — Rejeição estrita de matrizes com variância nula e correlações espúrias**:
   - `transform_table_11_architecture_correlations` rejeita peremptoriamente entradas onde $r=1.0$ e $p=0.0$ decorram de variância nula (`has_variance == False`) ou matrizes idênticas, levantando `PublicationValidationError`.
   - Verificado em C5 (`tests/test_c04_t11_scientific_decision.py:112`).

3. **Door 3 — Obrigatoriedade incondicional do teste de permutação de nós**:
   - É mandatório fornecer evidência de teste nulo de permutação (`p_permutation_null` ou `is_permutation_tested: True`) com contagem mínima de permutações $N_{\text{perm}} \ge 100$. Entradas omitindo o teste nulo, com flags falsas ou com contagem insuficiente de permutações são rejeitadas com `PublicationValidationError`.
   - No preflight de prontidão de publicação (`diagnose_publication_readiness`), itens sem teste nulo recebem status `SCIENTIFIC_MISMATCH`.
   - Sob dados válidos de coocorrência e teste nulo, `validate_selection` admite o artefato com status `DERIVED_VERIFIED`.
   - Verificado em C4 (`tests/test_c04_t11_scientific_decision.py:100`), C6 (`tests/test_c04_t11_scientific_decision.py:126`), C7 (`tests/test_c04_t11_scientific_decision.py:137`), C8 (`tests/test_c04_t11_scientific_decision.py:146`), C9 (`tests/test_c04_t11_scientific_decision.py:185`) e C10 (`tests/test_c04_t11_scientific_decision.py:219`).

## Gate Execution

- Execução da suíte completa de testes de decisão científica de T11 (C04):
  `.venv/bin/python -m unittest tests/test_c04_t11_scientific_decision.py` — 10 passed, 0 failed in 0.088s.
- Execução das suítes de regressão (C01, C02, C03):
  `.venv/bin/python -m unittest tests/test_c01_data_contracts.py tests/test_c02_derived_manifests.py tests/test_c03_publication_adapters.py` — 39 passed, 0 failed in 0.470s.
