# C02 — Manifestos Derivados, Hashes e Suporte Experimental em C06

## Problem

A cadeia de publicação e auditoria estrita C06 (`src/viral_cnn/reporting/publication.py` e `src/viral_cnn/reporting/transformations.py`) falha ao integrar artefatos derivados gerados por transformações determinísticas:

1. **Rejeição de `evidence_kind: derived`:**
   - `emit_derived_manifest()` grava `evidence_kind: "derived"`, mas `validate_selection()` aceita estritamente apenas `{"empirical", "structural"}`. Ao validar um artefato derivado sob o protocolo estrito, o validador emite `PublicationValidationError` e marca o item como `INVALID`.
   - Apenas artefatos estruturais têm regras de derivação registradas (`derivation: {source, config}`), deixando transformações tabulares sem validação de linhagem.

2. **Inversão de suporte amostral e denominadores ($N$ artificial):**
   - Atualmente, `emit_derived_manifest()` atribui `n = len(transformed_df)` e `support = len(transformed_df)`. Em tabelas agregadas e resumos (como T1 com 5 linhas de modelos ou T11 com 1 linha), $N$ passa a ser gravado como $1$ ou $5$, em vez de herdar e propagar o suporte real avaliado (ex.: $N=1.706.270$ fragmentos em T1 ou $N=4.514$ contigs em T12).
   - Não há distinção estruturada entre `output_rows`, `n_queries`, `n_fragments`, `n_taxa` e `denominator_n`.

3. **Lacunas de rastreabilidade e integridade transitiva:**
   - O manifesto derivado registra o manifesto pai, mas não registra formalmente o caminho e o hash SHA-256 do arquivo tabular fonte (`source_artifact_path`), nem valida se os dados de entrada foram adulterados.
   - Não há validação transitiva que impeça dependências sintéticas disfarçadas ou verifique a compatibilidade dos atributos de identidade científica herdados (`dataset_sha256`, `seed`, `split_id`, `checkpoints`, `model_variants`).

## Flow

Reutiliza o motor de manifestos e validação C06 existente (`publication.py`, `transformations.py`), preservando a distinção formal entre evidência empírica bruta e derivação determinística.

1. Artefato fonte (CSV/Parquet) e manifesto pai entram em `execute_publication_transformation` (exists/updated) - aplicar a transformação registrada, calcular `output_rows` e inspecionar suporte no pai (door 1, door 2).
2. Metadados upstream e código de transformação entram em `emit_derived_manifest` (exists/updated) - emitir manifesto com `evidence_kind: "derived"`, bloco `derivation: {source_artifact, parent_manifest, transformation_code}`, suportes separados (`output_rows`, `n_fragments`, `n_queries`, `n_taxa`, `denominator_n`) e identidade científica herdada verificada (door 1, door 2, door 3).
3. Configuração de publicação e manifestos derivados entram em `validate_selection` (exists/updated) - aceitar `evidence_kind: "derived"` com status `DERIVED_VERIFIED`, validar transitivamente hashes do arquivo fonte e do manifesto pai, e verificar que o pai não é sintético (door 1, door 3).
4. Manifestos com fonte adulterada, pai sintético ou hash divergente entram em `validate_selection` (exists) - rejeitar com `PublicationValidationError`, atribuindo status `INVALID` ou `SYNTHETIC` com diagnóstico detalhado (door 1, door 3).

## Impact

| Front | What changes |
| --- | --- |
| domain | new term: `derived_evidence_contract` - derivação formal de evidência empírica validando arquivo fonte, manifesto pai e código determinístico com status `DERIVED_VERIFIED`, vive em `src/viral_cnn/reporting/` |
| domain | existing term: `support` significava contagem de linhas do CSV (`len(df)`), agora distingue `output_rows` do suporte experimental herdado (`denominator_n`, `n_fragments`, `n_taxa`) |
| stored data | nada a migrar |

## Relations

`None - no stored-data shape change`

## Surface

| Route | In | Out | Status |
| --- | --- | --- | --- |
| `viral_cnn.reporting.transformations:emit_derived_manifest(item_id, output_artifact_path, parent_manifest_path, source_artifact_path, transformation_name, ...)` | `item_id`, `output_artifact_path`, `parent_manifest_path`, `source_artifact_path`, `transformation_name` | `Path` (manifest_path) | `200`, `400` |
| `viral_cnn.reporting.publication:validate_selection(config, strict_empirical, strict_provenance)` | `config`, `strict_empirical` | `list[dict[item_id, status, n, support, denominator, ...]]` | `200`, `400` |

## Landing

| One-way door | Literal shape | Alternative rejected |
| --- | --- | --- |
| 1 — Admissão formal de `derived` com status `DERIVED_VERIFIED` em C06 | Aceitar `evidence_kind == "derived"` em `validate_selection()`, exigindo bloco `derivation` completo com validação de hashes do arquivo fonte e do manifesto pai, e atribuir status `DERIVED_VERIFIED`. | Forçar `evidence_kind: "empirical"` ou `"structural"` para artefatos gerados por adaptadores de tabela. |
| 2 — Separação estruturada de suportes e conservação do denominador | Manifestos derivados registram `output_rows` para o tamanho da tabela, e herdam ou calculam `n`, `support` e `denominator_n` a partir do pai auditado (ex.: $N=1.706.270$ em T1), proibindo inferir $N$ a partir de `len(transformed_df)`. | Manter $N=5$ ou $N=1$ para tabelas de resumo e resumos agregados. |
| 3 — Rastreabilidade completa da fonte transformada no manifesto | Gravar em `derivation.source_artifact` o caminho relativo/absoluto e o SHA-256 do CSV/Parquet de entrada, além do SHA-256 do código da transformação. | Registrar apenas o manifesto pai sem vincular o hash do arquivo tabular insumo. |

## Criteria

### S1: Admissão de `evidence_kind: derived` e status `DERIVED_VERIFIED` em C06 (P1)

Validação formal de artefatos derivados determinísticos no validador C06.

**Acceptance Criteria**

1. WHEN `validate_selection` evaluates a manifest with `evidence_kind: "derived"` THEN the system SHALL validate the entry and assign status `DERIVED_VERIFIED` IF all derivation inputs and scientific identity checks pass.
2. WHEN `validate_selection` inspects a derived manifest THEN the system SHALL verify that `derivation` contains valid `source_artifact`, `parent_manifest`, and `transformation`.
3. IF any derivation input file (source artifact or parent manifest) is missing or its SHA-256 does not match the recorded hash THEN the system SHALL reject the item with status `INVALID`.
4. IF a derived manifest's parent manifest declares `is_smoke_test: true` or `synthetic: true` under strict empirical mode THEN the system SHALL assign status `SYNTHETIC` and reject the item.

**Independent test:** validar seleção C06 sobre manifestos derivados com pais válidos, pais adulterados e pais sintéticos.

### S2: Suporte experimental explícito e distinção de denominadores (P1)

Propagação fiel do suporte amostral dos dados observados sem confundir tamanho de tabela com tamanho de amostra.

**Acceptance Criteria**

5. WHEN `emit_derived_manifest` executes THEN the system SHALL record `output_rows` as the row count of the transformed DataFrame.
6. WHEN transforming an item with audited parent sample support THEN the system SHALL propagate `n`, `support`, and `denominator_n` from the parent manifest into the derived manifest without overriding them with `output_rows`.
7. IF support is not applicable for a specific item THEN the system SHALL require an explicit `support_not_applicable_reason` in the derived manifest.
8. The system SHALL verify that for T1 the propagated denominator is $N=1.706.270$ even when the transformed table has 5 rows.

**Independent test:** emitir manifesto derivado de T1 a partir de `benchmark_clean_noise_summary.csv` e verificar que `n` e `denominator_n` permanecem $1.706.270$ enquanto `output_rows` é $5$.

### S3: Linhagem completa e herança de identidade científica (P1)

Rastreabilidade ponta a ponta vinculando fonte, código e atributos científicos.

**Acceptance Criteria**

9. WHEN `emit_derived_manifest` executes THEN the system SHALL record the source artifact path, source artifact SHA-256, parent manifest path, parent manifest SHA-256, and transformation code hash.
10. WHEN deriving an artifact THEN the system SHALL inherit and validate the scientific identity dictionary (`dataset_sha256`, `seed`, `split_id`, `taxonomy_version`, `class_order_sha256`, `checkpoints`) from the parent manifest.
11. IF a figure artifact is derived THEN the system SHALL record selection parameters including `seed`, `ids`, `sampling`, and `filters`.

**Independent test:** inspecionar JSON de manifesto derivado emitido e verificar presença de todos os campos de derivação e integridade dos hashes.

## Out of scope

| Excluded | Why |
| --- | --- |
| Execução pesada de inferência no cluster | Objeto exclusivo da spec C07 |
| Implementação completa dos adaptadores específicos C03 | Foco de C02 é a infraestrutura de manifestos e validação C06 |
| Alteração estrutural nos notebooks de reprodução | Invariante de AGENTS.md |

## Assumptions

| Assumption | Chosen default | Rationale | Confirmed? |
| --- | --- | --- | --- |
| Propagação de suporte de pai para filho | Se o manifesto pai contiver `n` ou `support`, o manifesto derivado herda esses valores prioritariamente | Preserva o denominador real da avaliação empírica | y |
| Formato do bloco de derivação | `derivation: {"source_artifact": {"path", "sha256"}, "parent_manifest": {"path", "sha256"}, "transformation": {"name", "code_sha256"}}` | Compatível com a semântica de derivation existente para dados estruturais | y |

**Open questions:** none - all resolved or logged above.

## Observable

| Surface | Decision | Landing |
| --- | --- | --- |
| Function `emit_derived_manifest` | manifest structure and support fields | AC 5, AC 6, AC 7, AC 8, AC 9, AC 10, AC 11 |
| Function `validate_selection` | derived acceptance and status assignment | AC 1, AC 2, AC 3, AC 4 |

## Sources

- `PLANO_CORRECOES_CNN_METAGENOME_AGENTS_2026-11-15.md` - Especificação da demanda C02 e regras de aceitação
- `AGENTS.md` - Invariantes científicos, rastreabilidade de linhagem e proibição de valores inventados
- `src/viral_cnn/reporting/publication.py` - Protocolo `c06_strict_v1` e validador de seleção
