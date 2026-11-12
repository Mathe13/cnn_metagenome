# S15 — Empirical Artifact Provenance Hardening

## Problem

In `src/viral_cnn/reporting/publication.py` and `src/viral_cnn/reporting/transformations.py`, empirical artifacts and derived publication manifests track subsets of metadata, but do not uniformly validate or enforce the full 12-attribute provenance contract required by the roadmap. Prior to this improvement, a manifest could be accepted even if it omitted software/library versions, execution timestamps, explicit git commit hashes, or parent upstream lineage links. Additionally, `emit_derived_manifest` in `transformations.py` did not propagate parent git commits, software environment metadata, or configuration hashes, leaving gaps in end-to-end reproducibility chains from C03 through C06 to published tables and figures.

When this ships, `validate_artifact_provenance` provides fail-closed validation for empirical manifests across all 12 required provenance fields (git commit, dataset SHA256, checkpoint SHA256s, configuration SHA256, seed, model variant/variants, split, noise conditions, execution timestamp, software metadata, parent artifacts, and `is_smoke_test=false`), `validate_selection` enforces this provenance in publication assembly, and `emit_derived_manifest` propagates complete upstream lineage.

## Flow

This change reuses `viral_cnn.reporting.publication` (exists), `viral_cnn.reporting.transformations` (exists), and `viral_cnn.utils.logging_utils` (exists) without altering historical table formatting logic or models.

1. `manifest_data`, `project_root` enter `validate_artifact_provenance` (door 1) - validates that `is_smoke_test` is strictly `False`, all 12 required provenance attributes exist and are non-empty, and verifies all parent artifact hashes and existence on disk.
2. If any required attribute is missing or parent artifact is missing/corrupted, raises `PublicationValidationError` with an actionable diagnostic.
3. If any parent artifact in the transitive dependency chain has `is_smoke_test=True`, execution fails closed (door 2).
4. `item_id`, `output_artifact_path`, `parent_manifest_path`, `transformation_name`, `support_n` enter `emit_derived_manifest` (door 3) - reads parent manifest, resolves git commit, captures environment library versions, propagates configuration and dataset hashes, and emits a validated derived manifest.
5. In `validate_selection` (exists), each selected empirical item executes `validate_artifact_provenance` under `strict_empirical=True` (door 4).

## Impact

| Front | What changes |
| --- | --- |
| domain | new function `validate_artifact_provenance` in `src/viral_cnn/reporting/publication.py` enforcing the 12 required provenance fields |
| domain | `emit_derived_manifest` in `src/viral_cnn/reporting/transformations.py` populates complete lineage including `git_commit`, `library_versions`, `configuration_sha256`, and upstream parent artifacts |
| domain | `validate_selection` in `src/viral_cnn/reporting/publication.py` executes `validate_artifact_provenance` on all empirical entries under `strict_empirical=True` |
| stored data | derived manifests emitted during transformations contain complete 12-point provenance metadata |

## Relations

None - no stored-data shape change

## Surface

None - nothing consumed outside

## Landing

| One-way door | Literal shape | Alternative rejected |
| --- | --- | --- |
| 1 — validação uniforme do contrato de 12 atributos de proveniência | `validate_artifact_provenance(manifest_data: dict[str, Any], project_root: Path) -> dict[str, Any]` levantando `PublicationValidationError` em qualquer omissão | Validações ad-hoc espalhadas em scripts: permite que artefatos incompletos ou sem versão de software entrem em publicação. |
| 2 — rejeição transitiva estrita de smoke tests e falhas na cadeia | Rejeição imediata (`is_smoke_test is not False` ou parent ausente/inválido) sem substituição silenciosa | Tolerar artefatos sintéticos na cadeia quando o artefato final parece empírico. |
| 3 — propagação completa de linhagem em manifestos derivados | `emit_derived_manifest(..., extra_metadata=...)` propagando `git_commit`, `library_versions`, `configuration_sha256`, e `scientific_identity` completos | Deixar manifestos derivados apenas com campos básicos e sem rastreabilidade de código ou ambiente. |
| 4 — integração fail-closed em `validate_selection` | Chamada obrigatória a `validate_artifact_provenance` para todos os manifestos de itens empíricos sob `strict_empirical=True` | Manter `validate_selection` frouxo para manifestos empíricos. |

- Nothing else in this change is hard to reverse

## Criteria

### S1: Complete artifact provenance validation and lineage propagation (P1)

Todo artefato empírico de publicação deve conter ou referenciar os 12 atributos de proveniência e falhar se contiver dados sintéticos ou linhagem rompida.

**Acceptance Criteria**

1. WHEN `validate_artifact_provenance` is called with a complete empirical manifest containing all 12 attributes THEN the system SHALL return a verified provenance dictionary.
2. IF a manifest has `is_smoke_test` set to `True` or missing THEN `validate_artifact_provenance` SHALL raise `PublicationValidationError`.
3. IF a manifest omits `git_commit` or contains an empty string THEN `validate_artifact_provenance` SHALL raise `PublicationValidationError`.
4. IF a manifest omits execution timestamp (`timestamp_utc` or `created_utc`) THEN `validate_artifact_provenance` SHALL raise `PublicationValidationError`.
5. IF a manifest omits software environment metadata (`versions` or `library_versions`) THEN `validate_artifact_provenance` SHALL raise `PublicationValidationError`.
6. IF a manifest references parent artifacts that do not exist on disk or whose SHA-256 does not match THEN `validate_artifact_provenance` SHALL raise `PublicationValidationError`.
7. IF any upstream parent artifact in the transitive lineage declares `is_smoke_test=True` THEN `validate_artifact_provenance` SHALL raise `PublicationValidationError`.
8. WHEN `emit_derived_manifest` is executed THEN the resulting derived manifest SHALL contain `git_commit`, `created_utc`, `library_versions`, `is_smoke_test: false`, `artifact_sha256`, and parent manifest reference.
9. WHEN `validate_selection` is executed with `strict_empirical=True` THEN every empirical manifest lacking required provenance attributes SHALL raise `PublicationValidationError` and be flagged as `INVALID` or `SYNTHETIC`.

**Independent test:** Execute `python3 -m unittest tests/test_s15_artifact_provenance_hardening.py` verifying full provenance validation, smoke test rejection, missing commit/timestamp/versions rejection, broken parent detection, transitive smoke rejection, derived manifest completeness, and strict publication selection integration.

## Out of scope

| Excluded | Why |
| --- | --- |
| Retreino de modelos em GPU | S15 endurece contratos e validações de proveniência de artefatos empíricos; não roda retreinamentos no cluster. |
| Alteração de rotulagem de calibração | Escopo de S16. |

## Assumptions

| Assumption | Chosen default | Rationale | Confirmed? |
| --- | --- | --- | --- |
| Formato dos 12 atributos de proveniência | Suportar nomes canônicos e variações históricas (`versions`/`library_versions`, `created_utc`/`timestamp_utc`, `parent_artifacts`/`inputs`) | Permite interoperabilidade com manifestos já gerados por etapas de treino e avaliação sem perder rigor de validação. | y |
| Validação transitiva | Recursão limitada a manifestos dependentes para garantir que nenhum ancestral é smoke test ou corrompido | Evita aceitar derivados que parecem empíricos mas descendem de etapas sintéticas. | y |

**Open questions:** none - all resolved or logged above.

## Observable

None - no user-facing surface

## Sources

- ROADMAP_IMPROVEMENTS.md (Section 9: S15 — Empirical artifact provenance hardening) - prescreve os 12 atributos de proveniência obrigatórios e o invariante de invalidade se `is_smoke_test==true` ou ancestral ausente.
- ROADMAP_IMPROVEMENTS.md (Section 18: Milestone B — Publication safety) - define a segurança de publicação impedindo que constantes sintéticas ou históricas sejam publicadas acidentalmente.
- AGENTS.md (Invariantes científicos e proveniência) - exige rastreabilidade rigorosa de dados de entrada, split, checkpoint, inferência e métricas.

