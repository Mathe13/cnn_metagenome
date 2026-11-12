# S11 — Strict Matched-Control Policies

## Problem

In `src/viral_cnn/interpretability/motifs.py:335-370`, `sample_matched_control_region` returns a bare integer `Optional[int]`. When `control_policy="gc_matched"` fails to find an exact GC match ($|\Delta\text{GC}|=0$), it silently falls back to a $\pm 1$ GC base candidate without informing the caller ($1/7 = 14.3\%$ GC deviation for 7-mers). When `control_policy="complexity_matched"` fails to find an exact homopolymer run length match, it silently falls back to $\pm 1$ run length and ultimately to an arbitrary non-overlapping candidate, returning an integer that is still treated as complexity matched. Downstream perturbation records and summary exports do not record match quality (`exact`, `relaxed_1`, `unmatched_fallback`, `unavailable`) or matching distance, concealing control relaxation and preventing post-hoc auditing. Furthermore, `evaluate_motif_perturbation` lacks a strict mode to restrict primary scientific analyses to exact controls and lacks matching coverage quantification (`exact_match_rate`).

When this ships, control sampling returns structured `MatchedControl` metadata, records matching quality explicitly, supports strict-mode enforcement (`quality == 'exact'`), and reports exact matching coverage.

## Flow

This change reuses `viral_cnn.interpretability.motifs` (exists), `viral_cnn.models.short.utils.models.OneHotEmbedding` (exists), and `notebook_runtime` (exists) without retraining models or modifying model weights.

1. `sequence`, `motif_starts`, `k`, `query_motif`, `control_policy`, `rng` enter `sample_matched_control_region` (door 1) - filters non-overlapping windows with valid bases $\{0, 1, 2, 3\}$, evaluates GC count or homopolymer run length, and returns `MatchedControl` (door 1).
2. `MatchedControl` enters `evaluate_motif_perturbation` in `motifs` (exists) - unpacks `ctrl.start` while logging `control_match_quality`, `control_match_distance`, and `control_policy_requested` in emitted records (door 2).
3. If `strict_matching=True` enters `evaluate_motif_perturbation` in `motifs` (exists), occurrences where `ctrl.quality != "exact"` are skipped from perturbation testing (door 3).
4. Perturbation records enter `summarize_matched_control_coverage` (door 4) - aggregates `n_requested`, `n_exact`, `n_relaxed`, `n_unmatched`, `n_unavailable`, and `exact_match_rate` per model and motif.
5. `MatchedControl` implements `__index__`, `__int__`, and comparison methods so existing integer indexing and assertions remain backward compatible.

## Impact

| Front | What changes |
| --- | --- |
| domain | new class `MatchedControl` and new function `summarize_matched_control_coverage` in `src/viral_cnn/interpretability/motifs.py` |
| domain | `sample_matched_control_region` in `src/viral_cnn/interpretability/motifs.py` returns `MatchedControl` with explicit `quality` (`exact`, `relaxed_1`, `unmatched_fallback`, `unavailable`) and `distance` |
| domain | `evaluate_motif_perturbation` in `src/viral_cnn/interpretability/motifs.py` records `control_match_quality`, `control_match_distance`, and `control_policy_requested` and adds `strict_matching` parameter |
| stored data | perturbation result records and summary CSVs include `control_match_quality` and `control_match_distance`; no historical snapshots are altered in-place |

## Relations

None - no stored-data shape change

## Surface

None - nothing consumed outside

## Landing

| One-way door | Literal shape | Alternative rejected |
| --- | --- | --- |
| 1 — representação estruturada de controle pareado | `MatchedControl(start: Optional[int], policy: str, quality: Literal['exact', 'relaxed_1', 'unmatched_fallback', 'unavailable'], distance: int)` | Retornar `Optional[int]`: descarta a qualidade do pareamento e oculta a degradação silenciosa. |
| 2 — vocabulário canônico de qualidade de pareamento | Qualidade restrita a `exact` ($|\Delta|=0$), `relaxed_1` ($|\Delta|=1$), `unmatched_fallback` ($|\Delta|>1$), `unavailable` (nenhum controle viável) | Usar booleano `is_matched`: não distingue relaxamento $\pm 1$ de fallback totalmente não pareado. |
| 3 — modo estrito para análise científica primária | Parâmetro `strict_matching: bool = False` em `evaluate_motif_perturbation` que exige `quality == 'exact'` quando ativo, com rastreamento de cobertura | Relaxar silenciosamente controles sob escassez de candidatos: introduz viés composicional de GC na análise primária. |
| 4 — métricas de cobertura de pareamento | `summarize_matched_control_coverage(df: pd.DataFrame) -> pd.DataFrame` calculando `n_requested`, `n_exact`, `n_relaxed`, `n_unmatched`, `n_unavailable`, `exact_match_rate` | Não quantificar cobertura: impede auditar se conclusões científicas dependem de amostras com pareamento relaxado. |

- Nothing else in this change is hard to reverse

## Criteria

### S1: Strict matched-control policy and quality tracking (P1)

Controles pareados possuem metadados estruturados de qualidade, distância e proveniência sem degradação silenciosa.

**Acceptance Criteria**

1. WHEN `sample_matched_control_region` is called with `control_policy="gc_matched"` and an exact GC candidate exists THEN the system SHALL return `MatchedControl` with `quality="exact"`, `distance=0`, and the matching start position.
2. WHEN `sample_matched_control_region` is called with `control_policy="gc_matched"` and only $\pm 1$ GC candidate exists THEN the system SHALL return `MatchedControl` with `quality="relaxed_1"` and `distance=1`.
3. IF no candidate window in the sequence satisfies GC difference $\le 1$ or all candidates are invalid THEN the system SHALL return `MatchedControl` with `start=None`, `quality="unavailable"`, and `distance=-1`.
4. WHEN `sample_matched_control_region` is called with `control_policy="complexity_matched"` and an exact run length candidate exists THEN the system SHALL return `MatchedControl` with `quality="exact"` and `distance=0`.
5. WHEN `sample_matched_control_region` is called with `control_policy="complexity_matched"` and only fallback candidates exist THEN the system SHALL NOT label the quality as `exact`.
6. WHILE selecting control candidates in `sample_matched_control_region`, the system SHALL reject any window that overlaps any index in `motif_starts` within $[start, start+K)$.
7. WHILE selecting control candidates in `sample_matched_control_region`, the system SHALL reject any window containing invalid bases or padding ($< 0$ or $\ge 4$).
8. WHEN `evaluate_motif_perturbation` evaluates occurrences THEN the system SHALL record `control_match_quality`, `control_match_distance`, and `control_policy_requested` in each emitted record.
9. WHERE `strict_matching=True` is provided to `evaluate_motif_perturbation`, IF a candidate control has `quality != "exact"` THEN the system SHALL skip perturbation evaluation for that occurrence.
10. WHEN `summarize_matched_control_coverage` is called on perturbation records THEN the system SHALL compute `exact_match_rate` as the ratio of exact matches over total requests.

**Independent test:** Execute `pytest tests/test_s11_strict_matched_control_policies.py` verifying exact GC match, relaxed GC label, unavailable GC response, complexity exact match, complexity fallback labeling, overlap avoidance, ambiguous base rejection, record metadata presence, strict-mode filtering, and coverage summary metrics.

## Out of scope

| Excluded | Why |
| --- | --- |
| Retreino de modelos convolucionais | S11 é uma alteração metodológica no protocolo de controle pareado de motivos, sem impacto nos pesos de redes treinadas. |
| Re-execução empírica integral no cluster | O re-processamento em larga escala do dataset completo de teste pertence às etapas finais de reprodução (S20). |
| Mudança no modelo nulo de coocorrência de nós | A análise de rede e permutação de nós é escopo de M05/S13. |

## Assumptions

| Assumption | Chosen default | Rationale | Confirmed? |
| --- | --- | --- | --- |
| Compatibilidade de `MatchedControl` com inteiros | `MatchedControl` implementa `__index__`, `__int__`, e comparações com inteiros | Garante retrocompatibilidade com código existente que usa `seq[start:start+k]` e `start == 10`. | y |
| Padrão de `strict_matching` em `evaluate_motif_perturbation` | `strict_matching: bool = False` por padrão com aviso observável, ativado explicitamente na análise primária | Preserva compatibilidade com rotinas exploratórias enquanto viabiliza o modo estrito sem falhas silenciosas. | y |

**Open questions:** none - all resolved or logged above.

## Observable

None - no user-facing surface

## Sources

- ROADMAP_IMPROVEMENTS.md (Section 5: S11 — Strict matched-control policies) - define o vocabulário de qualidade de pareamento (`exact`, `relaxed_1`, `unmatched_fallback`, `unavailable`), a dataclass `MatchedControl` e as métricas de cobertura.
- ROADMAP_IMPROVEMENTS.md (Section 8: S14 — Scientific regression tests for motif controls) - define os 10 testes de regressão científica para controle pareado.
- AGENTS.md (Invariantes científicos e motivos) - exige controle pareado de mesma sequência com $|\Delta\text{GC}|=0$ estrito para afastar viés composicional de homopolímeros.
