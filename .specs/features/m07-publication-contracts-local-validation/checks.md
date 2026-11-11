# M07 checks

Profile: light
Plan: `.specs/features/m07-publication-contracts-local-validation/plan.md` — aprovado pelo usuário (“prossiga”).

16 checks em três slices; cinco portas; nenhuma questão bloqueante.

## Checks

### S1

**C1** — The system SHALL provide a centralized registry of publication transformations mapping each item in T1–T12 and F1–F5 to an explicit transformation function. (AC 1).
Proof: `REMOTE_SERVER=False PYTHONPATH=src .venv/bin/python3 -m unittest tests.test_m07_publication_contracts.M07Tests.test_transformation_registry -k test_transformation_registry`

**C2** — WHEN a table transformation executes THEN the system SHALL validate the input columns, convert formats, compute support $N$, and output a DataFrame matching the publication column contract. (AC 2).
Proof: `REMOTE_SERVER=False PYTHONPATH=src .venv/bin/python3 -m unittest tests.test_m07_publication_contracts.M07Tests.test_table_transformations_execution -k test_table_transformations_execution`

**C3** — WHEN a transformation finishes THEN the system SHALL emit a derived manifest recording the parent manifest path and SHA-256, transformation name, code hash, and support $N$ with `evidence_kind="derived"`. (AC 3).
Proof: `REMOTE_SERVER=False PYTHONPATH=src .venv/bin/python3 -m unittest tests.test_m07_publication_contracts.M07Tests.test_derived_manifest_emission -k test_derived_manifest_emission`

**C4** — IF an input artifact is a PNG or non-tabular image file THEN the system SHALL reject it with a PublicationValidationError, requiring structured CSV or Parquet data. (AC 4).
Proof: `REMOTE_SERVER=False PYTHONPATH=src .venv/bin/python3 -m unittest tests.test_m07_publication_contracts.M07Tests.test_reject_png_source -k test_reject_png_source`

**C5** — WHILE executing in smoke test mode (`SMOKE_TEST=1`) THEN all 17 transformation functions SHALL run completely on synthetic fixtures within 15 seconds. (AC 5).
Proof: `REMOTE_SERVER=False PYTHONPATH=src .venv/bin/python3 -m unittest tests.test_m07_publication_contracts.M07Tests.test_smoke_mode_transformations_speed -k test_smoke_mode_transformations_speed`

### S2

**C6** — WHEN `diagnose_publication_readiness` executes THEN the system SHALL evaluate all 17 items (T1–T12, F1–F5) and return a diagnostic matrix with columns `item_id`, `status`, `diagnostic`, `proposed_source`, and `recommendation`. (AC 6).
Proof: `REMOTE_SERVER=False PYTHONPATH=src .venv/bin/python3 -m unittest tests.test_m07_publication_contracts.M07Tests.test_batch_diagnosis_matrix -k test_batch_diagnosis_matrix`

**C7** — IF an artifact or manifest path is unset or does not exist on disk THEN the system SHALL assign status `FALTA`. (AC 7).
Proof: `REMOTE_SERVER=False PYTHONPATH=src .venv/bin/python3 -m unittest tests.test_m07_publication_contracts.M07Tests.test_diagnosis_status_falta -k test_diagnosis_status_falta`

**C8** — IF an artifact exists but lacks required columns or plot coordinate parameters THEN the system SHALL assign status `SCHEMA_MISMATCH`. (AC 8).
Proof: `REMOTE_SERVER=False PYTHONPATH=src .venv/bin/python3 -m unittest tests.test_m07_publication_contracts.M07Tests.test_diagnosis_status_schema_mismatch -k test_diagnosis_status_schema_mismatch`

**C9** — IF an artifact exists but exhibits conflicting seed, split_id, or taxonomic class ordering THEN the system SHALL assign status `IDENTITY_MISMATCH`. (AC 9).
Proof: `REMOTE_SERVER=False PYTHONPATH=src .venv/bin/python3 -m unittest tests.test_m07_publication_contracts.M07Tests.test_diagnosis_status_identity_mismatch -k test_diagnosis_status_identity_mismatch`

**C10** — IF an artifact violates scientific invariants (such as F3 lacking 5 models, T11 claiming architecture correlation without permutation null test, or F5 lacking edge coordinates) THEN the system SHALL assign status `SCIENTIFIC_MISMATCH`. (AC 10).
Proof: `REMOTE_SERVER=False PYTHONPATH=src .venv/bin/python3 -m unittest tests.test_m07_publication_contracts.M07Tests.test_diagnosis_status_scientific_mismatch -k test_diagnosis_status_scientific_mismatch`

### S3

**C11** — WHEN F3 is evaluated THEN the system SHALL require exactly five species model variants (`ResNet50_Clean`, `ConvNeXt_Z`, `ConvNeXt_T`, `ResNet50_F`, `ConvNeXt_F`), rejecting inputs missing ConvNeXt_F with status `SCIENTIFIC_MISMATCH`. (AC 11).
Proof: `REMOTE_SERVER=False PYTHONPATH=src .venv/bin/python3 -m unittest tests.test_m07_publication_contracts.M07Tests.test_f3_requires_five_models -k test_f3_requires_five_models`

**C12** — WHEN T11 is evaluated THEN the system SHALL verify that co-occurrence matrices without permutation null testing or without variance are rejected as architecture correlation tables with status `SCIENTIFIC_MISMATCH`. (AC 12).
Proof: `REMOTE_SERVER=False PYTHONPATH=src .venv/bin/python3 -m unittest tests.test_m07_publication_contracts.M07Tests.test_t11_rejects_raw_cooccurrence_without_permutation -k test_t11_rejects_raw_cooccurrence_without_permutation`

**C13** — WHEN F5 is evaluated as a network graph THEN the system SHALL verify that node and edge coordinates (`x1`, `y1`, `x2`, `y2`) are present, rejecting bar chart data with status `SCHEMA_MISMATCH` or `SCIENTIFIC_MISMATCH`. (AC 13).
Proof: `REMOTE_SERVER=False PYTHONPATH=src .venv/bin/python3 -m unittest tests.test_m07_publication_contracts.M07Tests.test_f5_requires_network_coordinates -k test_f5_requires_network_coordinates`

**C14** — WHEN `assemble_publication` runs with `strict_empirical=True` THEN the system SHALL reject any artifact or dependency with `is_smoke_test: true`, raising PublicationValidationError and marking gate `empirical="BLOCKED"`. (AC 14).
Proof: `REMOTE_SERVER=False PYTHONPATH=src .venv/bin/python3 -m unittest tests.test_m07_publication_contracts.M07Tests.test_strict_rejection_of_synthetic -k test_strict_rejection_of_synthetic`

**C15** — WHEN `00_regenerate_all_paper_artifacts.ipynb` executes THEN the system SHALL render the batch diagnostic table and invoke `assemble_publication` isolating outputs in `results_reproduction/runs/<run_id>/publication/`. (AC 15).
Proof: `REMOTE_SERVER=False PYTHONPATH=src .venv/bin/python3 -m unittest tests.test_m07_publication_contracts.M07Tests.test_regenerate_notebook_diagnostic_and_isolation -k test_regenerate_notebook_diagnostic_and_isolation`

**C16** — WHEN artifacts are inspected, loaded, or written across M07 workflows THEN the system SHALL log paths with prefix `[LOG_PATH]`, respecting `LOG_PATH=0/false`. (AC 16).
Proof: `REMOTE_SERVER=False PYTHONPATH=src .venv/bin/python3 -m unittest tests.test_m07_publication_contracts.M07Tests.test_logging_path_traceability -k test_logging_path_traceability`

## Coverage

| Set (size) | Member -> proof | Unproven |
| --- | --- | --- |
| Items (17) | T1 C1 · T2 C1 · T3 C1 · T4 C1 · T5 C1 · T6 C1 · T7 C1 · T8 C1 · T9 C1 · T10 C1 · T11 C1 · T12 C1 · F1 C1 · F2 C1 · F3 C1 · F4 C1 · F5 C1 | - |
| Diagnostic statuses (5) | OK C6 · FALTA C7 · SCHEMA_MISMATCH C8 · IDENTITY_MISMATCH C9 · SCIENTIFIC_MISMATCH C10 | - |
| Scientific gates (5) | png_rejection C4 · f3_five_models C11 · t11_permutation_null C12 · f5_network_coords C13 · synthetic_rejection C14 | - |
| Doors (5) | transformation_registry C1 · batch_diagnosis C6 · derived_manifests C3 · negative_contracts C11 · notebook_integration C15 | - |
| Manifest modes (3) | COMPLETE C3 · FAILED C14 · DIAGNOSTIC C6 | - |

## Test policy

| Target | Policy |
| --- | --- |
| transformations_registry | unit: mapeamento de T1–T12 e F1–F5 para funções de transformação e execução sobre dados sintéticos |
| derived_manifests | unit: emissão de manifestos com evidence_kind='derived', parent_sha256 e suporte N |
| batch_diagnosis | unit: auditoria dos 17 itens classificando com precisão os 5 estados diagnósticos e fontes propostas |
| negative_contracts | unit: bloqueio estrito de PNGs como fontes, F3 incompleto, T11 sem permutação, F5 em barras e evidência sintética |
| notebook_regenerate | integration: execução de ponta a ponta da célula diagnóstica e montagem de pacote em diretório isolado |

## Swept

Where each unwritten requirement landed. All nine, one line each, every time.

- validation: C1, C2, C4, C6, C7, C8, C9, C10, C11, C12, C13
- failure modes: C4, C14, emissão de PublicationValidationError e bloqueio estrito do portão empírico
- idempotency: C1, C3, C6, C15
- authorization: n/a — execução local sob permissões de arquivo do sistema operacional
- concurrency: C15, reserva atômica de diretório exclusivo sob runs/ impedindo colisões
- data lifecycle: C2, C3, C6, C15
- dependency failure: C4, C7, C8, C9, C10, C14
- state transitions: C3, C6, C14
- observability: C1, C2, C3, C6, C7, C8, C9, C10, C11, C12, C13, C15, C16

## Handoff

S1–S3 = ~45k total, dentro do orçamento de 150k. Mecanismo: um builder.
