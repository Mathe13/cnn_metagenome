# r08-dynamic-paper-tables-and-manuscript-reconciliation checks

Profile: light
Plan: `.specs/features/r08-dynamic-paper-tables-and-manuscript-reconciliation/plan.md`

## Checks

### S1 - Exact parameter reconciliation and LaTeX manuscript synchronization · 3 files · 25 KB · ~6k

**C1** - `generate_table_1_metrics` reports total parameters of 43.99 M for ResNet50 Species, 21.49 M for ResNet50 Genus, 164.01 M for ConvNeXt Species, and 141.51 M for ConvNeXt Genus, with backbone and fc parameter breakdowns
Proof: `PYTHONPATH=src .venv/bin/python3 -m unittest tests/test_r08_reconciliation_and_tables.py -k test_r08_table_1_parameters_reconciliation`

**C2** - `generate_table_2_profiling` reports total species parameters of 43.99 M for ResNet50 1D and 164.01 M for ConvNeXt 1D with exact backbone breakdowns
Proof: `PYTHONPATH=src .venv/bin/python3 -m unittest tests/test_r08_reconciliation_and_tables.py -k test_r08_table_2_parameters_reconciliation`

**C3** - `paper/manuscript_submission.tex` and `paper/manuscript_review.tex` regenerated via `build_manuscript_variants` contain 43.99 M and 164.01 M for Tables 1 and 2, contain the architectural documentation of HybridConvNeXt, and do not contain legacy 23.5 or 28.1 M values
Proof: `PYTHONPATH=src .venv/bin/python3 -m unittest tests/test_r08_reconciliation_and_tables.py -k test_r08_manuscript_latex_parameters_and_text`

**C4** - `compile_manuscript` compiles `paper/manuscript_submission.tex` cleanly to PDF with exit code 0
Proof: `PYTHONPATH=src .venv/bin/python3 -m unittest tests/test_r08_reconciliation_and_tables.py -k test_r08_manuscript_submission_pdf_compilation`

### S2 - Dynamic artifact ingestion, strict_empirical gate, and reconciliation reporting · 2 files · 30 KB · ~7k

**C5** - `generate_all_tables(..., strict_empirical=True)` raises `RuntimeError` when given a results directory containing any manifest with `"is_smoke_test": true`
Proof: `PYTHONPATH=src .venv/bin/python3 -m unittest tests/test_r08_reconciliation_and_tables.py -k test_r08_strict_empirical_rejects_smoke_test`

**C6** - `generate_all_tables` dynamically loads available reproduction artifacts, produces all 12 publication tables in CSV, Markdown, and LaTeX formats, and exports `reconciliation_report.csv` and `reconciliation_report.md`
Proof: `PYTHONPATH=src .venv/bin/python3 -m unittest tests/test_r08_reconciliation_and_tables.py -k test_r08_generate_all_tables_and_reconciliation_report`

## Coverage

| Set (size) | Member -> proof | Unproven |
| --- | --- | --- |
| Architecture parameter calculation levels (2) | `species_level_parameters` C1 · `genus_level_parameters` C1 | - |
| Table generation execution regimes (2) | `strict_empirical_enforcement` C5 · `dynamic_artifact_ingestion` C6 | - |
| Manuscript publication formats (2) | `submission_clean_latex` C3 · `review_dual_track_latex` C3 | - |
| Reconciliation audit deliverables (2) | `reconciliation_report_csv` C6 · `reconciliation_report_markdown` C6 | - |

- No other check claims more than the single case its proof exercises

## Swept

Where each unwritten requirement landed. All nine, one line each, every time.

- validation: C1, C2, C3 (validates exact parameter arithmetic for backbones and fc heads across taxonomic ranks)
- failure modes: C5 (fails fast with RuntimeError when strict_empirical is set and smoke test manifests are detected)
- idempotency: C1, C6 (deterministic table generation and identical markdown/LaTeX outputs across repeated calls)
- authorization: n/a - local and cluster offline scripts with no network credentials
- concurrency: n/a - single-process table generation and document compilation
- data lifecycle: C6 (persists clean publication tables and reconciliation reports without lingering temp files)
- dependency failure: C5 (prevents uncalibrated smoke fixtures from being published as empirical article tables)
- state transitions: C3, C4 (chains table generation into manuscript source regeneration and clean PDF compilation)
- observability: C6 (writes comprehensive reconciliation_report.csv and .md detailing evidence status for each table)

## Handoff

- S1 = ~6k tokens, S2 = ~7k tokens, total ~13k tokens across 5 files (`src/viral_cnn/reporting/tables.py`, `src/viral_cnn/reporting/manuscript.py`, `paper/manuscript_submission.tex`, `paper/manuscript_review.tex`, `tests/test_r08_reconciliation_and_tables.py`), well under the 150k budget - one builder
- **Boundary:** C1-C6 closed at HEAD
- **Settled mid-build:** none
- **Abandoned:** none

