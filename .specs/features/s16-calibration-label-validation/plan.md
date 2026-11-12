# S16 — Calibration Label Fail-Fast Validation

## Problem

In `src/viral_cnn/eval/calibration.py:215-218`, `compute_calibration_metrics` historically tolerated ground-truth labels outside `[0, n_classes)` by clamping invalid targets and assigning `p_true=0.0`. This mask (`valid_target_mask = (y_np >= 0) & (y_np < n_classes)`) allowed out-of-vocabulary indices (such as abstention labels `-1` or 1-based indexing offsets `NUM_CLASSES`) to pass silently into evaluation, resulting in inflated Negative Log-Likelihood (NLL) and distorted Brier scores instead of immediately flagging a malformed evaluation set.

When this ships, `compute_calibration_metrics` and `compute_reliability_diagram` enforce closed-set label validity (`0 <= y_true < n_classes`) fail-fast by default (`invalid_label_policy="raise"`), reporting the count and representative examples of invalid labels. An explicit opt-in policy (`invalid_label_policy="exclude"`) allows filtering invalid labels only when scientifically justified, adjusting sample counts transparently without silent denominator modification or artificial zero-probability penalty.

## Flow

This change reuses `viral_cnn.eval.calibration` (exists) without altering network architectures or checkpoint representations.

1. `probs`, `y_true`, `invalid_label_policy` enter `compute_calibration_metrics` (door 1) - converts inputs to numpy arrays via `_to_numpy_probs_and_labels` (exists), discovers `n_samples` and `n_classes`, and evaluates label boundaries `[0, n_classes)`.
2. If invalid labels exist and `invalid_label_policy == "raise"` (door 1), raises `ValueError` detailing the invalid count, `n_classes`, and sample invalid label values (door 2).
3. If invalid labels exist and `invalid_label_policy == "exclude"` (door 3), filters probability and label arrays to valid samples, updates `n_samples`, and raises `ValueError` if 0 valid samples remain.
4. If an unsupported policy string enters, raises `ValueError`.
5. Filtered/validated arrays proceed to `compute_reliability_diagram` (exists), ECE/MCE, NLL, and Brier calculations with an accurate, consistent denominator.

## Impact

| Front | What changes |
| --- | --- |
| domain | `compute_calibration_metrics` and `compute_reliability_diagram` in `src/viral_cnn/eval/calibration.py` receive `invalid_label_policy: Literal["raise", "exclude"] = "raise"` |
| domain | closed-set label violations (`y < 0` or `y >= n_classes`) raise `ValueError` by default instead of assigning `p_true=0` |
| domain | `invalid_label_policy="exclude"` updates sample size and denominator explicitly |
| stored data | no stored files or checkpoints are modified |

## Relations

None - no stored-data shape change

## Surface

None - nothing consumed outside

## Landing

| One-way door | Literal shape | Alternative rejected |
| --- | --- | --- |
| 1 — fail-fast por padrão para rótulos inválidos | `invalid_label_policy: Literal["raise", "exclude"] = "raise"` em `compute_calibration_metrics` | Tolerar silenciosamente rótulos fora do intervalo atribuindo `p_true=0`: mascara erros de indexação (como 1-based vs 0-based) e infla artificialmente o NLL. |
| 2 — mensagem de erro acionável com contagem e exemplos | `raise ValueError(f"Found {n_invalid} invalid labels outside [0, {n_classes}). Examples: {examples}.")` | Erro genérico `AssertionError` sem indicar quais rótulos violaram o domínio da taxonomia. |
| 3 — exclusão explícita com conservação do denominador | Filtragem preliminar de amostras antes do cálculo de acurácia, ECE, NLL e Brier quando `invalid_label_policy="exclude"` | Manter denominador original com probabilidade zero para amostras excluídas. |

- Nothing else in this change is hard to reverse

## Criteria

### S1: Calibration label fail-fast validation and policy enforcement (P1)

A avaliação probabilística de calibração em closed-set deve validar estritamente se todos os rótulos pertencem a `[0, n_classes)` e falhar imediatamente em caso de rótulo inválido.

**Acceptance Criteria**

1. WHEN `compute_calibration_metrics` is called with all ground-truth labels in `[0, n_classes)` under default `invalid_label_policy="raise"` THEN the system SHALL return `CalibrationMetrics` with `n_samples` equal to input length.
2. IF `compute_calibration_metrics` receives ground-truth labels outside `[0, n_classes)` with default `invalid_label_policy="raise"` THEN the system SHALL raise a `ValueError` reporting the invalid count, `n_classes`, and sample invalid values.
3. WHEN `compute_calibration_metrics` is called with `invalid_label_policy="exclude"` on inputs containing invalid labels THEN the system SHALL calculate metrics over valid samples only, and `n_samples` SHALL reflect the filtered count.
4. IF `compute_calibration_metrics` with `invalid_label_policy="exclude"` receives inputs with zero valid samples THEN the system SHALL raise `ValueError("All samples were excluded due to invalid labels.")`.
5. IF `compute_calibration_metrics` receives an unrecognized policy string THEN the system SHALL raise `ValueError` stating the policy must be 'raise' or 'exclude'.
6. WHEN `compute_reliability_diagram` is called with invalid labels under `invalid_label_policy="raise"` THEN the system SHALL raise `ValueError`.
7. WHEN `compute_reliability_diagram` is called with `invalid_label_policy="exclude"` THEN the system SHALL return bins whose total count strictly matches the filtered sample count.

**Independent test:** Execute `python3 -m unittest tests/test_s16_calibration_label_validation.py` verifying default fail-fast on negative and out-of-bound labels, descriptive error reporting, explicit exclude filtering, all-invalid error handling, invalid policy rejection, and reliability diagram policy parity.

## Out of scope

| Excluded | Why |
| --- | --- |
| Retreinamento de modelos em GPU | S16 trata exclusivamente da validação de rótulos no cálculo de métricas probabilísticas de calibração. |
| Alteração de rotulagem do dataset Parquet | A convenção do repositório já adota 0-based para tensores de entrada conforme AD-002. |

## Assumptions

| Assumption | Chosen default | Rationale | Confirmed? |
| --- | --- | --- | --- |
| Comportamento padrão | `invalid_label_policy="raise"` | Avaliação closed-set de redes neurais requer que todas as amostras tenham classe conhecida no vocabulário. | y |
| Política alternativa | `"exclude"` | Permite que pipelines com rótulos de abstenção (-1) ou classes não anotadas excluam explicitamente amostras sem mascarar erros. | y |

**Open questions:** none - all resolved or logged above.

## Observable

None - no user-facing surface

## Sources

- ROADMAP_IMPROVEMENTS.md (Section 10: S16 — Calibration label fail-fast validation) - define a obrigatoriedade de `0 <= y_true < n_classes`, o parâmetro `invalid_label_policy="raise"`, e a proibição de tolerar silenciosamente `p_true=0`.
- ROADMAP_IMPROVEMENTS.md (Section 18: Milestone D — Defensive hardening) - prescreve menos modos silenciosos de falha em experimentos futuros.
- AGENTS.md (Invariantes científicos) - define closed-set estrito e ordem de classes consistente.
