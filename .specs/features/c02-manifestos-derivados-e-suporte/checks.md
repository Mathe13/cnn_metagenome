# C02 — Manifestos Derivados, Hashes e Suporte Experimental em C06 checks

Profile: light
Plan: `.specs/features/c02-manifestos-derivados-e-suporte/plan.md`

11 checks in 3 slices · 3 one-way doors · 0 open, of which 0 block

## Checks

Grouped by the plan's slices; numbering runs across the whole feature.

### S1 - Admissao de evidence_kind: derived e status DERIVED_VERIFIED em C06 · 2 files · 40 KB · ~10k

**C1** - A funcao `validate_selection` aceita manifestos com `evidence_kind: "derived"` e atribui status `DERIVED_VERIFIED` quando insumos e hashes conferem (AC 1)
Proof: `.venv/bin/python -m unittest -k test_validate_selection_accepts_derived tests/test_c02_derived_manifests.py`

**C2** - A funcao `validate_selection` inspeciona o bloco `derivation` exigindo `source_artifact`, `parent_manifest` e `transformation` validos (AC 2)
Proof: `.venv/bin/python -m unittest -k test_validate_selection_checks_derivation_block tests/test_c02_derived_manifests.py`

**C3** - A funcao `validate_selection` rejeita manifesto derivado com status `INVALID` se o arquivo fonte ou o manifesto pai estiver ausente ou com hash divergente (AC 3)
Proof: `.venv/bin/python -m unittest -k test_validate_selection_rejects_tampered_or_missing_source tests/test_c02_derived_manifests.py`

**C4** - A funcao `validate_selection` atribui status `SYNTHETIC` e rejeita o item derivado em modo estrito se o manifesto pai for marcado como sintetico ou smoke test (AC 4)
Proof: `.venv/bin/python -m unittest -k test_validate_selection_rejects_synthetic_parent_under_strict tests/test_c02_derived_manifests.py`

### S2 - Suporte experimental explicito e distincao de denominadores · 2 files · 40 KB · ~10k

**C5** - A funcao `emit_derived_manifest` registra `output_rows` correspondente a contagem de linhas do DataFrame transformado (AC 5)
Proof: `.venv/bin/python -m unittest -k test_emit_derived_manifest_records_output_rows tests/test_c02_derived_manifests.py`

**C6** - A funcao `emit_derived_manifest` propaga `n`, `support` e `denominator_n` do manifesto pai sem sobrescreve-los com `output_rows` (AC 6)
Proof: `.venv/bin/python -m unittest -k test_emit_derived_manifest_propagates_parent_support tests/test_c02_derived_manifests.py`

**C7** - A funcao `emit_derived_manifest` exige `support_not_applicable_reason` quando suporte experimental nao se aplicar ao item (AC 7)
Proof: `.venv/bin/python -m unittest -k test_emit_derived_manifest_requires_reason_when_not_applicable tests/test_c02_derived_manifests.py`

**C8** - O manifesto derivado de T1 propaga o denominador $N=1.706.270$ mesmo quando a tabela transformada possui 5 linhas (AC 8)
Proof: `.venv/bin/python -m unittest -k test_t1_derived_manifest_preserves_large_denominator tests/test_c02_derived_manifests.py`

### S3 - Linhagem completa e heranca de identidade cientifica · 2 files · 40 KB · ~10k

**C9** - A funcao `emit_derived_manifest` registra o caminho e SHA-256 do arquivo fonte, caminho e SHA-256 do manifesto pai e hash do codigo da transformacao (AC 9)
Proof: `.venv/bin/python -m unittest -k test_emit_derived_manifest_records_lineage_hashes tests/test_c02_derived_manifests.py`

**C10** - A derivacao de artefato herda e preserva os atributos de identidade cientifica (`dataset_sha256`, `seed`, `split_id`, `taxonomy_version`, `class_order_sha256`, `checkpoints`) (AC 10)
Proof: `.venv/bin/python -m unittest -k test_emit_derived_manifest_inherits_scientific_identity tests/test_c02_derived_manifests.py`

**C11** - A derivacao de artefatos de figuras registra parametros de selecao incluindo `seed`, `ids`, `sampling` e `filters` (AC 11)
Proof: `.venv/bin/python -m unittest -k test_emit_derived_manifest_records_figure_selection tests/test_c02_derived_manifests.py`

## Coverage

| Set (size) | Member -> proof | Unproven |
| --- | --- | --- |
| Validacao C06 de derivados (4) | admissao_derived C1 · checagem_bloco_derivation C2 · rejeicao_adulterado_ou_ausente C3 · rejeicao_pai_sintetico C4 | - |
| Suporte e denominadores (4) | output_rows C5 · propagacao_suporte_pai C6 · justificativa_inaplicavel C7 · preservacao_t1_grande_n C8 | - |
| Linhagem e identidade (3) | hashes_linhagem C9 · identidade_cientifica C10 · parametros_figura C11 | - |

## Swept

Where each unwritten requirement landed. All nine, one line each, every time.

- validation: C2, C3, C7
- failure modes: C3, C4
- idempotency: existing - emissao de manifestos deterministica com hashes estaveis
- authorization: n/a - pipeline local sem controle de acesso de rede
- concurrency: n/a - execucao sequencial e monotarefa
- data lifecycle: n/a - manifestos salvos junto aos artefatos sem politica de expiracao
- dependency failure: C3
- state transitions: C1, C4
- observability: existing - registro em log dos caminhos e hashes pelo logger de publicacao

## Handoff

S1-S3 = ~30k tokens acumulados (publication, transformations, tests), bem abaixo do orcamento de 150k - one builder.

- Boundary: C1-C11 fechados em conjunto por um unico builder
