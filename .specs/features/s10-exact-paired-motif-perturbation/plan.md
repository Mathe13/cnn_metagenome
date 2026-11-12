# S10 — Exact Paired Motif Perturbation

## Problem

In `src/viral_cnn/interpretability/motifs.py:449-450`, the perturbation evaluation loop executes:
```python
for _ in range(n_repeats):
    m_pert = perturb_sequence_mask(seq_batch, motif_mask.unsqueeze(0), generator=generator)
    c_pert = perturb_sequence_mask(seq_batch, ctrl_mask.unsqueeze(0), generator=generator)
```
Although both calls share the same `torch.Generator`, each call advances the RNG stream independently to sample random substitution integers (`torch.randint(1, 4, ...)`). Consequently, the motif region and the matched control region undergo different nucleotide substitutions. The manuscript method and the scientific contract require that in each paired repeat, the motif region and the control region receive the exact same substitution offset pattern ($\delta_i \in \{1, 2, 3\}$ for positions $i=1 \dots K$), so that only genomic location differs. Without identical offsets, the differential probability drop reflects random base sampling divergence rather than pure sequence-context specificity. Furthermore, output records omit mutation pattern provenance, preventing post-hoc mechanical verification of the pairing invariant.

When this ships, motif perturbation runs apply the exact same substitution offsets to both regions in every paired repeat and persist a deterministic hash of the mutation pattern in downstream records.

## Flow

This change reuses `viral_cnn.interpretability.motifs` (exists), `viral_cnn.models.short.utils.models.OneHotEmbedding` (exists), and `notebook_runtime` (exists) without altering model weights or taxonomy structures.

1. `n_positions`, `generator`, and `device` enter `sample_substitution_offsets` (door 1) - samples 1D tensor of offsets $\delta_i \in \{1, 2, 3\}$ of length $K$, hands offsets to callers.
2. `windows`, `start`, and `offsets` enter `perturb_region_with_offsets` (door 2) - applies $(x + \delta_i) \pmod 4$ to $[start, start+K)$ for valid bases $x \in \{0, 1, 2, 3\}$, leaving invalid/padding bases untouched; hands perturbed sequence to model.
3. In `evaluate_motif_perturbation` in `motifs` (exists), for each paired repeat, `sample_substitution_offsets` (door 1) samples `offsets` once; `perturb_region_with_offsets` (door 2) mutates `motif_start` and `control_start` using the identical `offsets`.
4. `offsets` enters `hash_mutation_offsets` (door 3) - computes a deterministic 16-hex SHA-256 fingerprint of the offsets array.
5. In `evaluate_motif_perturbation` in `motifs` (exists), result records persist `mutation_pattern_hash` alongside `motif_start` and `control_start` for provenance (door 3).
6. Existing `perturb_sequence_mask` in `motifs` (exists) remains available for backward compatibility, optionally delegating to `perturb_region_with_offsets`.

## Impact

| Front | What changes |
| --- | --- |
| domain | new functions: `sample_substitution_offsets`, `perturb_region_with_offsets`, `hash_mutation_offsets` in `src/viral_cnn/interpretability/motifs.py` |
| domain | existing function `evaluate_motif_perturbation` in `src/viral_cnn/interpretability/motifs.py` applies identical offsets to motif and control within each repeat, and emits `mutation_pattern_hash` in result records |
| stored data | perturbation result records and exported CSVs (`motif_perturbation_by_variant.csv`, `table_08_motif_perturbation.csv`) include provenance column `mutation_pattern_hash`; no historical runs are modified in-place |

## Relations

None - no stored-data shape change

## Surface

None - nothing consumed outside

## Landing

| One-way door | Literal shape | Alternative rejected |
| --- | --- | --- |
| 1 — separação de amostragem de offsets | `sample_substitution_offsets(n_positions: int, generator: Optional[torch.Generator] = None, device: torch.device = torch.device('cpu')) -> torch.Tensor` gerando valores em `{1, 2, 3}` | Reutilizar `perturb_sequence_mask` duas vezes com o mesmo generator: avança o RNG e gera mutações distintas para motivo e controle. |
| 2 — perturbação determinística por offsets | `perturb_region_with_offsets(windows: torch.Tensor, start: int, offsets: torch.Tensor) -> torch.Tensor` aplicando $(x + \delta_i) \pmod 4$ | Passar máscara booleana e re-amostrar internamente: impossibilita garantia externa de igualdade de offsets entre regiões distintas. |
| 3 — hash de proveniência do padrão de mutação | `hash_mutation_offsets(offsets: torch.Tensor) -> str` retornando digest SHA-256 em hex (16 caracteres) registrado como `mutation_pattern_hash` | Omitir hash do padrão: impede auditoria e verificação mecânica de que motivo e controle receberam o mesmo padrão na mesma repetição. |

- Nothing else in this change is hard to reverse

## Criteria

### S1: Exact paired motif and control perturbation mechanics (P1)

Substituições de nucleotídeos em repetições pareadas compartilham o mesmo vetor de offsets $\delta \in \{1, 2, 3\}^K$.

**Acceptance Criteria**

1. WHEN `sample_substitution_offsets` is called with length $K$ and a `torch.Generator` THEN the system SHALL return a 1D long tensor of length $K$ where every element is in $\{1, 2, 3\}$.
2. WHEN `perturb_region_with_offsets` is called with sequence tensor, start position, and offsets THEN the system SHALL replace each nucleotide $x \in \{0, 1, 2, 3\}$ at index $start + i$ with $(x + \delta_i) \pmod 4$.
3. WHEN `perturb_region_with_offsets` applies substitutions THEN the system SHALL keep all positions outside $[start, start+K)$ unchanged.
4. IF a base at $start + i$ is invalid or padding ($< 0$ or $\ge 4$) THEN the system SHALL leave that position untouched without modulo addition.
5. WHEN `evaluate_motif_perturbation` executes a repeat for a paired motif and control THEN the system SHALL supply the identical `offsets` tensor to both the motif region and the control region.
6. WHILE evaluating paired motif-control repeats, WHEN the motif subsequence and control subsequence are identical prior to perturbation THEN the system SHALL produce identical perturbed subsequences in both regions.
7. WHEN `hash_mutation_offsets` is called with an offsets tensor THEN the system SHALL return a deterministic 16-character hexadecimal SHA-256 digest string.
8. WHEN `evaluate_motif_perturbation` records perturbation results THEN the system SHALL include `mutation_pattern_hash`, `motif_start`, and `control_start` in every emitted record.

**Independent test:** Execute `pytest tests/test_paired_motif_perturbation.py` passing identical synthetic 7-bp motif and control regions and verifying that mutated subsequences are identical and mutation pattern hashes match.

## Out of scope

| Excluded | Why |
| --- | --- |
| Retreino de modelos convolucionais | S10 é uma correção puramente metodológica de inferência/interpretabilidade em modelos já treinados. |
| Re-execução empírica completa no cluster | Execução em larga escala com Parquets reais ocorre nas etapas finais de reprodução (S20). |
| Mudança nas regras de controle pareado GC/complexidade | Política estrita de seleção de controle é escopo específico de S11. |

## Assumptions

| Assumption | Chosen default | Rationale | Confirmed? |
| --- | --- | --- | --- |
| Compatibilidade de `perturb_sequence_mask` | Manter a assinatura original de `perturb_sequence_mask` | Outros módulos e testes existentes podem chamar `perturb_sequence_mask` diretamente. | y |
| Formato do hash de mutação | SHA-256 truncado em 16 caracteres hexadecimais | Suficiente para garantir unicidade e integridade no log sem sobrecarregar tabelas CSV/Parquet. | y |

**Open questions:** none - all resolved or logged above.

## Observable

None - no user-facing surface

## Sources

- ROADMAP_IMPROVEMENTS.md (Section 4: S10 — Exact paired motif perturbation) - define a separação entre amostragem de offsets e aplicação, e a exigência de igualdade de offsets entre motivo e controle.
- ROADMAP_IMPROVEMENTS.md (Section 8: S14 — Scientific regression tests for motif controls) - estabelece o teste de regressão `test_paired_motif_control_reuses_identical_substitution_offsets`.
- AGENTS.md (Invariantes científicos e motivos) - prescreve o mesmo padrão de substituições entre motivo e controle pareado da mesma sequência.
