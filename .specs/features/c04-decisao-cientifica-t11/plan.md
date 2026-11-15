# C04 — Decisao Cientifica e Contrato Executavel de T11

## Problem

A Tabela 11 histórica e o validador de publicação C06 apresentavam uma grave inconsistência científica e técnica:

1. **Ambiguidade entre correlação inter-modelos e coocorrência de motivos:**
   - O manuscrito preliminar sugeria que a Tabela 11 representava uma correlação de motivos "entre arquiteturas" (ResNet50 vs ConvNeXt). No entanto, o pipeline histórico avaliava sequências idênticas entre variantes ou matrizes degeneradas com variância nula, gerando correlações espúrias $r=1$ ($p=0$).
   - Em contrapartida, o notebook de reprodução `short/08` implementou a análise empírica real: associação exploratória de coocorrência e escore RSCU em pares de 7-mers, controlada por modelo nulo de **permutação de nós** (1.000 permutações determinísticas com semente 42) gerando `table_11_cooccurrence_synergy.csv`.
   - Os diagnósticos de sobreposição de Jaccard entre variantes foram isolados em `table_11b_jaccard_variant_diagnostics.csv` com status puramente descritivo, não inferencial.

2. **Lacunas no validador e bloqueio em C06:**
   - O validador de C06 continha `expected.columns: []` vazio em versões legadas de configuração, bloqueando a certificação de prontidão de publicação.
   - O adaptador `transform_table_11_architecture_correlations` só executava a verificação do teste nulo se a coluna `is_permutation_tested` estivesse presente, permitindo que entradas sem teste nulo ou com dados espúrios passassem desapercebidas se os cabeçalhos fossem alterados.
   - Faltava um contrato executável que formalizasse a decisão metodológica e impedisse a publicação de T11 sem comprovação empírica do teste de permutação de nós.

## Flow

1. Arquivo de coocorrência de `short/08` entra em `execute_publication_transformation` (exists) - ler `table_11_cooccurrence_synergy.csv` validando formato tabular (door 1, door 2).
2. Dados brutos entram em `transform_table_11_architecture_correlations` (exists/updated) - mapear colunas equivalentes (`p_perm -> p_permutation_null`), verificar presença de `p_permutation_null` e `n_valid_permutations`, validar ausência de correlações espúrias $r=1$ com variância nula, e exigir teste nulo de permutação (door 1, door 2, door 3).
3. Dados validados e manifesto pai entram em `emit_derived_manifest` (exists) - emitir artefato derivado com bloco `derivation` C02 e registrar colunas completas da decisão C04 (door 1, door 2).
4. Configuração de publicação e manifesto entram em `validate_selection` (exists) - validar colunas esperadas contra a lista canônica não vazia de `configs/reproduction_m07.yaml` e admitir com status `DERIVED_VERIFIED` (door 1, door 2).

## Impact

| Front | What changes |
| --- | --- |
| domain | new term: `t11_cooccurrence_synergy_contract` - contrato científico formal definindo T11 como análise de coocorrência de 7-mers controlada por permutação de nós, rejeitando alegações de correlação inter-modelos sem pareamento independente |
| domain | updated term: `p_permutation_null` - valor-p bicaudal do teste de permutação de nós em grafo de coocorrência de 7-mers |
| stored data | nada a migrar |

## Relations

`None - no stored-data shape change`

## Surface

| Route | In | Out | Status |
| --- | --- | --- | --- |
| `viral_cnn.reporting.transformations:transform_table_11_architecture_correlations(df, meta)` | `pd.DataFrame`, `dict` | `pd.DataFrame` (T11) | `200`, `400` |
| `viral_cnn.reporting.publication:diagnose_publication_readiness(config)` | `dict`, `Path` | `pd.DataFrame` (diagnostics) | `200`, `400` |
| `viral_cnn.reporting.publication:validate_selection(config, strict_empirical)` | `dict` | `list[dict]` (selection status) | `200`, `400` |

## Landing

| One-way door | Literal shape | Alternative rejected |
| --- | --- | --- |
| 1 — Demarcação científica de T11 como coocorrência com nulo de permutação | Definir formalmente T11 como tabela de coocorrência e escore RSCU de 7-mers com colunas `[status, motif_score, n_valid_edges, n_valid_permutations, r_canonical, r_reversed, delta_r_orientation, icc_symmetric, r_bidirectional, p_permutation_null]`. | Manter a denominação de "correlação entre arquiteturas" baseada em dados degenerados com $r=1$. |
| 2 — Rejeição estrita de matrizes com variância nula e correlações espúrias | Rejeitar com `PublicationValidationError` qualquer entrada onde $r=1.0$ e $p=0.0$ advenham de matrizes idênticas ou sem variância (`has_variance == False`). | Aceitar $r=1$ acriticamente no pipeline de publicação. |
| 3 — Obrigatoriedade incondicional do teste de permutação de nós | Exigir obrigatoriamente a coluna `p_permutation_null` (ou `is_permutation_tested: True`) com $N_{\text{perm}} \ge 100$, falhando peremptoriamente se o teste nulo for omitido. | Tornar o teste de permutação opcional. |

## Criteria

### S1: Formalização da decisão científica e colunas esperadas de T11 (P1)

Contrato canônico e validação da lista de colunas em configuração e documentação.

**Acceptance Criteria**

1. The system SHALL define `expected.columns` in `configs/reproduction_m07.yaml` and `docs/publication_data_contracts.md` as `["status", "motif_score", "n_valid_edges", "n_valid_permutations", "r_canonical", "r_reversed", "delta_r_orientation", "icc_symmetric", "r_bidirectional", "p_permutation_null"]`.
2. WHEN `transform_table_11_architecture_correlations` executes THEN the system SHALL output a DataFrame containing all required co-occurrence synergy columns.
3. The system SHALL document in `docs/publication_data_contracts.md` and `configs/reproduction_m07.yaml` that T11 represents 7-mer motif co-occurrence synergy with node-permutation testing, not inter-architecture correlation.

**Independent test:** validar que `configs/reproduction_m07.yaml` e o adaptador T11 expõem a lista completa e não vazia de colunas esperadas.

### S2: Rejeição de correlações espúrias e exigência do teste nulo (P1)

Validação matemática e integridade inferencial.

**Acceptance Criteria**

4. IF an input to `transform_table_11_architecture_correlations` lacks `p_permutation_null` and `is_permutation_tested` THEN the system SHALL raise `PublicationValidationError`.
5. IF an input claims $r=1.0$ and $p=0.0$ with zero variance or identical matrices THEN the system SHALL raise `PublicationValidationError`.
6. IF `is_permutation_tested` is present but contains `False` values THEN the system SHALL raise `PublicationValidationError`.
7. IF `n_valid_permutations` is less than 100 THEN the system SHALL raise `PublicationValidationError`.

**Independent test:** testar o adaptador com entradas sem teste nulo, entradas degeneradas com $r=1$ e entradas com permutações insuficientes, verificando rejeição.

### S3: Integração no preflight de publicação e no validador C06 (P1)

Classificação precisa de prontidão e admissão estrita.

**Acceptance Criteria**

8. WHEN `diagnose_publication_readiness` evaluates an artifact for T11 without permutation null test THEN the system SHALL classify the item with status `SCIENTIFIC_MISMATCH`.
9. WHEN `diagnose_publication_readiness` evaluates a valid co-occurrence artifact for T11 THEN the system SHALL classify the item with status `OK` (or `FALTA` if path unset).
10. WHEN `validate_selection` evaluates T11 under strict empirical mode with valid co-occurrence evidence THEN the system SHALL assign status `DERIVED_VERIFIED`.

**Independent test:** executar diagnóstico e validação de seleção sobre T11 em cenários conformes e não conformes.

## Out of scope

| Excluded | Why |
| --- | --- |
| Execução de novo treinamento multi-semente entre arquiteturas | Reservada para specs futuras dedicadas fora do escopo pré-empírico |
| Análise de rede de arestas de F5 | Objeto de verificação de F5 na spec C05 |
| Execução pesada de inferência no cluster | Objeto exclusivo da spec C07 |

## Assumptions

| Assumption | Chosen default | Rationale | Confirmed? |
| --- | --- | --- | --- |
| Escopo científico de T11 | Coocorrência de 7-mers com escore RSCU e nulo de permutação | Reflete com precisão os dados calculados em `short/08` sem alegações espúrias de correlação inter-modelos | y |
| Mapeamento de `p_perm` para `p_permutation_null` | Mapear automaticamente se presente | Garante compatibilidade direta com a nomenclatura interna de `analyze_cooccurrence_pairs()` | y |
| Mínimo de permutações válidas | Exigir $N_{\text{perm}} \ge 100$ | Garante poder estatístico mínimo para resolução de valores-p empíricos | y |

**Open questions:** none - all resolved or logged above.

## Observable

| Surface | Decision | Landing |
| --- | --- | --- |
| Config `configs/reproduction_m07.yaml` | lista de colunas esperadas de T11 | AC 1, AC 3 |
| Document `docs/publication_data_contracts.md` | contrato executável e decisão metodológica | AC 1, AC 3 |
| Module `viral_cnn.reporting.transformations:transform_table_11_architecture_correlations` | rejeição de $r=1$ espúrio e validação do teste nulo | AC 2, AC 4, AC 5, AC 6, AC 7 |
| Module `viral_cnn.reporting.publication:diagnose_publication_readiness` | diagnóstico granular de `SCIENTIFIC_MISMATCH` para T11 | AC 8, AC 9 |
| Module `viral_cnn.reporting.publication:validate_selection` | status `DERIVED_VERIFIED` para T11 sob evidência válida | AC 10 |

## Sources

- `PLANO_CORRECOES_CNN_METAGENOME_AGENTS_2026-11-15.md` - Planejamento de correções pré-empíricas e definição de escopo da spec C04
- `docs/publication_data_contracts.md` - Matriz executável de contratos de dados de publicação
- `notebooks/reproduction/short/08_triplet_composition_and_cooccurrence.ipynb` - Exportador de `table_11_cooccurrence_synergy.csv`
- `.specs/STATE.md` - Decisões AD-014, AD-033, AD-043, AD-044, AD-045
