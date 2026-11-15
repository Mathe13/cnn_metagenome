# C02 — Manifestos Derivados, Hashes e Suporte Experimental em C06 verification

**Verdict**: PASS
**Profile**: light
**Diff range**: HEAD
**Round**: 1 - full
**Verifier**: independent sub-agent (author != verifier)
**Checks**: `.specs/features/c02-manifestos-derivados-e-suporte/checks.md`

## Summary

- **Total checks**: 11 passed, 0 failed, 0 skipped
- **Suite duration**: 0.235s across 11 unit tests in `tests/test_c02_derived_manifests.py`
- **Regression verification**:
  - `tests/test_c01_data_contracts.py`: 16 passed, 0 failed (0.143s)
  - `tests/test_m07_publication_contracts.py`: 16 passed, 0 failed (0.777s)

## Checks

| Check | Claim | Command Executed | Duration | Evidence | Result |
| --- | --- | --- | --- | --- | --- |
| C1 | validate_selection aceita manifestos com evidence_kind: "derived" e atribui status DERIVED_VERIFIED quando insumos e hashes conferem (AC 1) | `.venv/bin/python -m unittest -k test_validate_selection_accepts_derived tests/test_c02_derived_manifests.py` | 0.028s | `tests/test_c02_derived_manifests.py:115` | PASS |
| C2 | validate_selection inspeciona o bloco derivation exigindo source_artifact, parent_manifest e transformation válidos (AC 2) | `.venv/bin/python -m unittest -k test_validate_selection_checks_derivation_block tests/test_c02_derived_manifests.py` | 0.026s | `tests/test_c02_derived_manifests.py:147` | PASS |
| C3 | validate_selection rejeita manifesto derivado com status INVALID se arquivo fonte ou manifesto pai estiver ausente ou com hash divergente (AC 3) | `.venv/bin/python -m unittest -k test_validate_selection_rejects_tampered_or_missing_source tests/test_c02_derived_manifests.py` | 0.026s | `tests/test_c02_derived_manifests.py:174` | PASS |
| C4 | validate_selection atribui status SYNTHETIC e rejeita item derivado em modo estrito se manifesto pai for marcado como sintético ou smoke test (AC 4) | `.venv/bin/python -m unittest -k test_validate_selection_rejects_synthetic_parent_under_strict tests/test_c02_derived_manifests.py` | 0.026s | `tests/test_c02_derived_manifests.py:201` | PASS |
| C5 | emit_derived_manifest registra output_rows correspondente à contagem de linhas do DataFrame transformado (AC 5) | `.venv/bin/python -m unittest -k test_emit_derived_manifest_records_output_rows tests/test_c02_derived_manifests.py` | 0.024s | `tests/test_c02_derived_manifests.py:233` | PASS |
| C6 | emit_derived_manifest propaga n, support e denominator_n do manifesto pai sem sobrescrevê-los com output_rows (AC 6) | `.venv/bin/python -m unittest -k test_emit_derived_manifest_propagates_parent_support tests/test_c02_derived_manifests.py` | 0.023s | `tests/test_c02_derived_manifests.py:255` | PASS |
| C7 | emit_derived_manifest exige support_not_applicable_reason quando suporte experimental não se aplicar ao item (AC 7) | `.venv/bin/python -m unittest -k test_emit_derived_manifest_requires_reason_when_not_applicable tests/test_c02_derived_manifests.py` | 0.045s | `tests/test_c02_derived_manifests.py:275` | PASS |
| C8 | Manifesto derivado de T1 propaga denominador N=1.706.270 mesmo quando a tabela transformada possui 5 linhas (AC 8) | `.venv/bin/python -m unittest -k test_t1_derived_manifest_preserves_large_denominator tests/test_c02_derived_manifests.py` | 0.024s | `tests/test_c02_derived_manifests.py:311` | PASS |
| C9 | emit_derived_manifest registra caminho e SHA-256 do arquivo fonte, caminho e SHA-256 do manifesto pai e hash do código da transformação (AC 9) | `.venv/bin/python -m unittest -k test_emit_derived_manifest_records_lineage_hashes tests/test_c02_derived_manifests.py` | 0.024s | `tests/test_c02_derived_manifests.py:334` | PASS |
| C10 | Derivação de artefato herda e preserva os atributos de identidade científica (dataset_sha256, seed, split_id, taxonomy_version, class_order_sha256, checkpoints) (AC 10) | `.venv/bin/python -m unittest -k test_emit_derived_manifest_inherits_scientific_identity tests/test_c02_derived_manifests.py` | 0.027s | `tests/test_c02_derived_manifests.py:355` | PASS |
| C11 | Derivação de artefatos de figuras registra parâmetros de seleção incluindo seed, ids, sampling e filters (AC 11) | `.venv/bin/python -m unittest -k test_emit_derived_manifest_records_figure_selection tests/test_c02_derived_manifests.py` | 0.025s | `tests/test_c02_derived_manifests.py:376` | PASS |

## One-Way Doors Verified

1. **Door 1 — Admissão formal de `derived` com status `DERIVED_VERIFIED` em C06**:
   - `validate_selection()` aceita explicitamente `evidence_kind == "derived"`, validando integridade estrutural do bloco `derivation`, conferindo existência e SHA-256 dos artefatos insumo (arquivo fonte e manifesto pai), e checando código de transformação registrado.
   - Atribui status `DERIVED_VERIFIED` quando íntegro, ou `INVALID`/`SYNTHETIC` quando corrompido, ausente ou derivado de dados sintéticos/smoke test.
   - Verificado em C1 (`tests/test_c02_derived_manifests.py:115`), C2 (`tests/test_c02_derived_manifests.py:147`), C3 (`tests/test_c02_derived_manifests.py:174`) e C4 (`tests/test_c02_derived_manifests.py:201`).

2. **Door 2 — Separação estruturada de suportes e conservação do denominador**:
   - `emit_derived_manifest()` grava `output_rows` para representar as dimensões físicas da tabela transformada, preservando `n`, `support` e `denominator_n` herdados do manifesto pai auditado.
   - Proíbe a geração de denominadores artificiais (como $N=5$ em resumos de benchmarks).
   - Verificado em C5 (`tests/test_c02_derived_manifests.py:233`), C6 (`tests/test_c02_derived_manifests.py:255`), C7 (`tests/test_c02_derived_manifests.py:275`) e C8 (`tests/test_c02_derived_manifests.py:311`).

3. **Door 3 — Rastreabilidade completa da fonte e hashes de código no manifesto**:
   - Registra explicitamente em `derivation.source_artifact` o caminho e o hash SHA-256 do arquivo tabular de entrada, em `derivation.parent_manifest` o caminho e o hash do manifesto upstream, e em `derivation.transformation` o nome e o SHA-256 do código fonte da função de transformação.
   - Propaga e valida o dicionário de identidade científica herdado (`dataset_sha256`, `seed`, `split_id`, `taxonomy_version`, `class_order_sha256`, `checkpoints`).
   - Verificado em C9 (`tests/test_c02_derived_manifests.py:334`), C10 (`tests/test_c02_derived_manifests.py:355`) e C11 (`tests/test_c02_derived_manifests.py:376`).

## Gate Execution

- Suite completa C02: `.venv/bin/python -m unittest tests/test_c02_derived_manifests.py` — 11 passed, 0 failed, 0 errors in 0.235s.
- Validação estrita de completude TLC: `python3 .agent/skills/tlc-spec-lean/scripts/validate_verification.py c02-manifestos-derivados-e-suporte --strict`
