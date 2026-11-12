# S13 — Dataset Overlap and Manuscript Reconciliation Contract

## Problem

Historical drafts of the manuscript asserted or implied sequence-level disjointness between partitions (e.g. "While fragment sequences are disjunct", "While no identical 512-bp sequence fragment appears across partitions"). Empirical audits (AD-007, S13) demonstrate that approximately ~95.91% of evaluated test queries exhibit exact or reverse-complement sequence identity with training references due to reference genome redundancy. While the experiment remains scientifically sound as a closed-set reference-fragment recognition benchmark, imprecise claims of sequence-level disjointness risk misleading readers regarding unseen-genome generalization.

When this ships, the codebase and manuscript formally distinguish closed-set class membership (`Species_test ⊆ Species_train`), row-disjoint partitioning (`row_train ∩ row_test = ∅`), and sequence redundancy (~95.91% exact/RC overlap). Stale assertions of sequence disjointness are eliminated from the manuscript, mandatory terminology (`fragment-level closed-set classification`, `known-reference distribution`, `no-exact-overlap subset`) is adopted, an automated terminology audit validator is introduced, and a canonical machine-readable overlap audit artifact is generated.

## Flow

This change reuses `viral_cnn.baselines.overlap` (exists), `viral_cnn.reporting.manuscript` (exists), and `viral_cnn.utils.logging_utils` (exists) without altering model architectures, checkpoints, or numerical evaluation results.

1. `train_sequences`, `test_sequences`, `canonicalize_rc` enter `build_dataset_overlap_audit` (door 1) - evaluates canonical sequence digests, computes exact/RC overlap counts and fraction, asserts row disjointness, and returns structured audit dictionary.
2. `audit_dict`, `output_path` enter `write_dataset_overlap_audit_artifact` (door 2) - persists JSON matching the canonical S13 schema.
3. `latex_content` enters `audit_manuscript_terminology` (door 3) - validates that forbidden claims ("fragment sequences are disjunct", "no identical 512-bp sequence fragment", unqualified "unseen-genome generalization") are absent and required terms (`fragment-level closed-set classification`, `known-reference distribution`, `no-exact-overlap subset`) are present.
4. `generate_manuscript_content` in `src/viral_cnn/reporting/manuscript.py` (door 4) emits LaTeX reconciled with S13 terminology and passing `audit_manuscript_terminology`.

## Impact

| Front | What changes |
| --- | --- |
| domain | new module `src/viral_cnn/data/overlap_contract.py` implementing `build_dataset_overlap_audit`, `write_dataset_overlap_audit_artifact`, and `audit_manuscript_terminology` |
| domain | `src/viral_cnn/reporting/manuscript.py` replaces misleading sequence-disjointness assertions with explicit row-disjoint and ~95.91% overlap description |
| stored data | machine-readable overlap audit artifact JSON emitted following canonical S13 specification |

## Relations

None - no stored-data shape change

## Surface

None - nothing consumed outside

## Landing

| One-way door | Literal shape | Alternative rejected |
| --- | --- | --- |
| 1 — contrato formal de sobreposição | `build_dataset_overlap_audit(train_sequences, test_queries, canonicalize_rc=True) -> dict[str, Any]` retornando o esquema canônico S13 | Deixar a sobreposição sem schema canônico estruturado ou apenas em comentários de código. |
| 2 — artefato JSON de auditoria de sobreposição rastreável | `write_dataset_overlap_audit_artifact(audit_data: dict, output_path: Path) -> Path` | Gerar relatórios ad-hoc não parseáveis por pipelines de publicação. |
| 3 — validador automatizado de terminologia do manuscrito | `audit_manuscript_terminology(latex_text: str) -> dict[str, Any]` levantando ou reportando violações de termos proibidos e ausência de termos exigidos | Confiar apenas em revisão manual de texto suscetível a reintroduzir alegações de disjunção de sequências. |
| 4 — reconciliação definitiva do texto do manuscrito | Atualização das seções de Dataset Partitioning e Discussion em `src/viral_cnn/reporting/manuscript.py` | Manter alegações históricas como "While fragment sequences are disjunct" que contradizem a evidência empírica de 95,91% de overlap. |

- Nothing else in this change is hard to reverse

## Criteria

### S1: Dataset overlap scientific contract and manuscript reconciliation (P0)

A auditoria de sobreposição de fragmentos e o texto do manuscrito devem distinguir rigorosamente disjunção de linhas de disjunção de sequências, eliminando alegações enganosas de generalização a genomas não vistos.

**Acceptance Criteria**

1. WHEN `build_dataset_overlap_audit` is called with train and test sequences THEN the system SHALL return a dictionary containing `split_unit="fragment"`, `row_disjoint=True`, `genome_disjoint=False`, and `sequence_identity_audit`.
2. WHEN `sequence_identity_audit` is generated THEN `canonicalizes_reverse_complement` SHALL be True, evaluating sequences against `min(seq, reverse_complement(seq))`.
3. WHEN `write_dataset_overlap_audit_artifact` is called THEN it SHALL serialize the audit dictionary into valid, formatted JSON.
4. IF a text asserts sequence disjointness (e.g. "fragment sequences are disjunct" or "no identical 512-bp sequence fragment appears across partitions") THEN `audit_manuscript_terminology` SHALL report a violation.
5. IF a text asserts unqualified "unseen-genome generalization" as an established property of the fragment test split THEN `audit_manuscript_terminology` SHALL report a violation.
6. WHEN `generate_manuscript_content()` is called THEN both review and submission outputs SHALL pass `audit_manuscript_terminology` with zero violations.
7. WHEN `generate_manuscript_content()` is inspected THEN it SHALL contain the required terms: `fragment-level closed-set classification`, `known-reference distribution`, and `no-exact-overlap subset`.
8. WHEN `compile_manuscript` is executed on the reconciled `manuscript_submission.tex` THEN compilation SHALL succeed with returncode 0.

**Independent test:** Execute `python3 -m unittest tests/test_s13_dataset_overlap_contract.py` verifying canonical audit artifact structure, reverse-complement hashing, terminology auditor detection of forbidden claims, reconciled manuscript compliance, and LaTeX compilation clean pass.

## Out of scope

| Excluded | Why |
| --- | --- |
| Retreinamento de modelos em GPU | S13 reconcilia o escopo científico, a terminologia e emite artefatos de auditoria de overlap; não retreina modelos. |
| Construção de split desincorporado por genoma | Sem identificadores confiáveis de accession para todas as espécies, um split por genoma closed-set é inviável (AD-007). |

## Assumptions

| Assumption | Chosen default | Rationale | Confirmed? |
| --- | --- | --- | --- |
| Fração representativa de overlap | ~95.91% conforme demonstrado pela auditoria das consultas test do benchmark e documentado no roadmap | Consistente com a redundância genômica das referências Kraken2 e ICTV. | y |
| Tratamento de `no_exact_overlap` | Permanece como estrato de consultas sem hit exato no treino, sem promover a alegação de genomas novos | Preserva a integridade da análise comparativa com Kraken2/BLASTN. | y |

**Open questions:** none - all resolved or logged above.

## Observable

None - no user-facing surface

## Sources

- ROADMAP_IMPROVEMENTS.md (Section 7: S13 — Dataset overlap scientific contract) - prescreve as distinções entre pertinência de classe, disjunção de linhas e redundância de sequências, a terminologia obrigatória e o artefato JSON canônico.
- ROADMAP_IMPROVEMENTS.md (Section 17: S13 invalidates) - especifica que S13 não invalida números experimentais, mas corrige interpretação, escopo e texto do manuscrito.
- AGENTS.md (Invariantes científicos) - define que o split principal é por fragmentos e veda descrever os resultados como generalização para genomas ou espécies não vistos.
