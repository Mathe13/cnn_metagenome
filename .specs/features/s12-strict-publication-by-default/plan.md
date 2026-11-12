# S12 — Strict Publication by Default

## Problem

In `src/viral_cnn/reporting/tables.py:299`, `generate_all_tables` has parameter `strict_empirical: bool = False` by default. When invoked without explicit arguments or from default scripts, it silently emits hardcoded historical snapshot constants for Tables 1 through 12 into output directories, creating an imminent risk of publishing stale or synthetic numbers as authentic empirical findings. Furthermore, `generate_all_tables` contains an unreachable dead code block (lines 332-354) with a secondary `if strict_empirical:` check that can never be reached because lines 315-329 already return or raise. There is no explicit separation between the verifiable empirical publication pipeline and the legacy historical snapshot generator, and no fail-closed guard preventing execution when empirical artifact selection is missing.

When this ships, `generate_all_tables` fails closed by default (`strict_empirical=True`), requires an explicit admissible publication configuration, isolates historical snapshots into `generate_historical_snapshot_tables` with non-admissible metadata, and eliminates unreachable dead code.

## Flow

This change reuses `viral_cnn.reporting.publication` (exists), `viral_cnn.notebook_runtime.reserve_run_directory` (exists), and `viral_cnn.reporting.tables` (exists) without altering historical table formatting logic or models.

1. `output_dir`, `publication_config`, `formats` enter `generate_publication_tables` (door 1) - validates configuration presence and admissibility, executes `render_tables`, and returns publication DataFrames.
2. `output_dir` entering `generate_all_tables` (exists) defaults to `strict_empirical=True` (door 2); if `publication_config is None`, raises `RuntimeError("Publication generation requires an explicit admissible empirical artifact selection.")`.
3. If `strict_empirical=False` enters `generate_all_tables` (exists), execution delegates to `generate_historical_snapshot_tables` (door 3).
4. `output_dir`, `formats` enter `generate_historical_snapshot_tables` (door 3) - generates legacy snapshot tables 1-12, removes dead branch (door 4), and returns tables alongside metadata dictionary (`evidence_class="historical_snapshot"`, `publication_admissible=False`).
5. CLI script `scripts/generate_paper_tables.py` (exists) defaults to `--mode strict` using C06 namespace and delegates to `generate_historical_snapshot_tables` only under explicit `--mode legacy`.

## Impact

| Front | What changes |
| --- | --- |
| domain | `strict_empirical: bool = True` becomes default in `generate_all_tables` in `src/viral_cnn/reporting/tables.py` |
| domain | new functions `generate_publication_tables` and `generate_historical_snapshot_tables` in `src/viral_cnn/reporting/tables.py` |
| domain | unreachable dead branch in `generate_all_tables` lines 332-354 is deleted |
| domain | `generate_historical_snapshot_tables` returns metadata with `evidence_class="historical_snapshot"` and `publication_admissible=False` |
| stored data | no historical snapshots or runs are modified in-place; legacy snapshot tables are written only under explicit opt-in |

## Relations

None - no stored-data shape change

## Surface

None - nothing consumed outside

## Landing

| One-way door | Literal shape | Alternative rejected |
| --- | --- | --- |
| 1 — fail-closed por padrão na geração de tabelas | `generate_all_tables(..., strict_empirical: bool = True)` com `RuntimeError("Publication generation requires an explicit admissible empirical artifact selection.")` se `publication_config is None` | Manter `strict_empirical=False` por padrão: permite que comandos acidentais emitam constantes históricas como tabelas do artigo. |
| 2 — API dedicada de publicação empírica | `generate_publication_tables(output_dir: Path, publication_config: Path, formats: tuple[str, ...]) -> dict[int, pd.DataFrame]` | Misturar geração empírica e legada na mesma função sem validação obrigatória de configuração. |
| 3 — API dedicada e isolada de snapshots históricos | `generate_historical_snapshot_tables(output_dir: Path, formats: tuple[str, ...]) -> tuple[dict[int, pd.DataFrame], dict[str, Any]]` com metadados `evidence_class="historical_snapshot"` e `publication_admissible=False` | Permitir que tabelas legadas sejam geradas sem aviso observável de inadmissibilidade para publicação. |
| 4 — remoção de código morto estrito | Exclusão do bloco inalcançável `if strict_empirical:` em `tables.py:332-354` | Manter branch redundante: cria falsa impressão de que `results_dir.rglob("manifest*.json")` é executado no modo estrito. |

- Nothing else in this change is hard to reverse

## Criteria

### S1: Fail-closed publication reporting and historical isolation (P1)

A emissão de tabelas do artigo exige seleção empírica explícita e isola snapshots históricos sob metadados de inadmissibilidade.

**Acceptance Criteria**

1. WHEN `generate_all_tables` is called without explicit `strict_empirical` argument THEN the system SHALL execute with `strict_empirical=True`.
2. IF `generate_all_tables` is executed with `strict_empirical=True` and `publication_config is None` THEN the system SHALL raise a `RuntimeError` stating `"Publication generation requires an explicit admissible empirical artifact selection."`.
3. IF `generate_publication_tables` is called with a missing or non-existent configuration file THEN the system SHALL raise an exception without writing unverified tables.
4. WHEN `generate_publication_tables` is called with an admissible empirical configuration THEN the system SHALL render all publication tables and return a dictionary of DataFrames.
5. WHEN `generate_historical_snapshot_tables` is called THEN the system SHALL generate Tables 1 through 12 and return metadata containing `evidence_class="historical_snapshot"` and `publication_admissible=False`.
6. WHEN `generate_all_tables` is called with explicit `strict_empirical=False` THEN the system SHALL generate legacy snapshot tables and return a dictionary of 12 DataFrames.
7. WHILE executing `generate_all_tables` or `generate_historical_snapshot_tables`, the system SHALL NOT contain or execute any secondary dead `if strict_empirical:` directory globbing branch.
8. WHEN `scripts/generate_paper_tables.py` is called with `--mode strict` (default) THEN the system SHALL fail closed if empirical artifacts are blocked or missing.
9. WHEN `scripts/generate_paper_tables.py` is called with `--mode legacy` THEN the system SHALL explicitly invoke the historical snapshot generator.

**Independent test:** Execute `pytest tests/test_s12_strict_publication_by_default.py` verifying fail-closed default, exception on missing config, publication table rendering, historical snapshot metadata, legacy opt-in delegation, dead code removal, and CLI mode dispatch.

## Out of scope

| Excluded | Why |
| --- | --- |
| Re-execução empírica integral no cluster | S12 trata da política fail-closed e contratos da API de publicação; a execução completa de todas as etapas no cluster ocorre em S20. |
| Alteração nas rotinas de renderização LaTeX/TikZ de figuras | Figuras históricas e C06 já contam com isolamento de namespace C06. |
| Modificação nos valores numéricos das tabelas históricas | Os valores históricos de 1-12 permanecem preservados para auditoria de regressão em `historical_snapshot`. |

## Assumptions

| Assumption | Chosen default | Rationale | Confirmed? |
| --- | --- | --- | --- |
| Assinatura de `generate_all_tables` | Manter compatibilidade com chamadas passando `strict_empirical=False` | Permite que testes e scripts legados que declarem explicitamente `strict_empirical=False` continuem funcionando. | y |
| Retorno de `generate_all_tables(..., strict_empirical=False)` | Retorna `dict[int, pd.DataFrame]` com as 12 tabelas | Garante compatibilidade direta com chamadas legadas que desempacotavam o dicionário retornado. | y |

**Open questions:** none - all resolved or logged above.

## Observable

None - no user-facing surface

## Sources

- ROADMAP_IMPROVEMENTS.md (Section 6: S12 — Publication reporting must fail closed by default) - prescreve fail-closed por padrão (`strict_empirical=True`), mensagem de erro e APIs dedicadas `generate_publication_tables` e `generate_historical_snapshot_tables`.
- ROADMAP_IMPROVEMENTS.md (Section 13: S19 — Isolate historical reporting code) - prescreve o isolamento de snapshots históricos sob `evidence_class="historical_snapshot"` e a remoção do branch estrito morto em `generate_all_tables`.
- AGENTS.md (Invariantes científicos e tabelas) - proíbe promover constantes históricas a evidência empírica por padrão.
