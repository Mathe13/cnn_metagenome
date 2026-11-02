# r08-dynamic-paper-tables-and-manuscript-reconciliation verification

**Verdict**: PASS
**Profile**: light
**Diff range**: 9b86aa8..329ca9b
**Round**: 1 - full
**Verifier**: independent sub-agent (author != verifier)

## Checks

| Check | Claim | Proof run | Evidence | Result |
| --- | --- | --- | --- | --- |
| C1 | `generate_table_1_metrics` reports total parameters of 43.99 M for ResNet50 Species, 21.49 M for ResNet50 Genus, 164.01 M for ConvNeXt Species, and 141.51 M for ConvNeXt Genus, with backbone and fc parameter breakdowns | `PYTHONPATH=src .venv/bin/python3 -m unittest tests/test_r08_reconciliation_and_tables.py -k test_r08_table_1_parameters_reconciliation` exit 0 | `tests/test_r08_reconciliation_and_tables.py:26-58` - `r_sp['backbone_m'] == 15.96`, `r_sp['fc_params_m'] == 28.04`, `r_sp['parameters_m'] == 43.99`, `r_gen['parameters_m'] == 21.49`, `c_sp['parameters_m'] == 164.01`, `c_gen['parameters_m'] == 141.51` | PASS |
| C2 | `generate_table_2_profiling` reports total species parameters of 43.99 M for ResNet50 1D and 164.01 M for ConvNeXt 1D with exact backbone breakdowns | `PYTHONPATH=src .venv/bin/python3 -m unittest tests/test_r08_reconciliation_and_tables.py -k test_r08_table_2_parameters_reconciliation` exit 0 | `tests/test_r08_reconciliation_and_tables.py:59-76` - `r50['backbone_m'] == 15.96`, `r50['fc_species_m'] == 28.04`, `r50['parameters_m'] == 43.99`, `cnx['backbone_m'] == 135.98`, `cnx['parameters_m'] == 164.01` | PASS |
| C3 | `paper/manuscript_submission.tex` and `paper/manuscript_review.tex` regenerated via `build_manuscript_variants` contain 43.99 M and 164.01 M for Tables 1 and 2, contain the architectural documentation of HybridConvNeXt, and do not contain legacy 23.5 or 28.1 M values | `PYTHONPATH=src .venv/bin/python3 -m unittest tests/test_r08_reconciliation_and_tables.py -k test_r08_manuscript_latex_parameters_and_text` exit 0 | `tests/test_r08_reconciliation_and_tables.py:77-100` - `self.assertIn("43.99", content)`, `self.assertIn("164.01", content)`, `self.assertIn("HybridConvNeXt", content)`, `self.assertNotIn("& 23.5 &", content)` | PASS |
| C4 | `compile_manuscript` compiles `paper/manuscript_submission.tex` cleanly to PDF with exit code 0 | `PYTHONPATH=src .venv/bin/python3 -m unittest tests/test_r08_reconciliation_and_tables.py -k test_r08_manuscript_submission_pdf_compilation` exit 0 | `tests/test_r08_reconciliation_and_tables.py:101-111` - `retcode = compile_manuscript(submission_tex)`, `self.assertEqual(retcode, 0)`, `self.assertTrue((paper_dir / "manuscript_submission.pdf").exists())` | PASS |
| C5 | `generate_all_tables(..., strict_empirical=True)` raises `RuntimeError` when given a results directory containing any manifest with `"is_smoke_test": true` | `PYTHONPATH=src .venv/bin/python3 -m unittest tests/test_r08_reconciliation_and_tables.py -k test_r08_strict_empirical_rejects_smoke_test` exit 0 | `tests/test_r08_reconciliation_and_tables.py:112-137` - `with self.assertRaises(RuntimeError): generate_all_tables(..., strict_empirical=True)` on non-existent, empty, and smoke manifest directories | PASS |
| C6 | `generate_all_tables` dynamically loads available reproduction artifacts, produces all 12 publication tables in CSV, Markdown, and LaTeX formats, and exports `reconciliation_report.csv` and `reconciliation_report.md` | `PYTHONPATH=src .venv/bin/python3 -m unittest tests/test_r08_reconciliation_and_tables.py -k test_r08_generate_all_tables_and_reconciliation_report` exit 0 | `tests/test_r08_reconciliation_and_tables.py:138-178` - `self.assertEqual(len(tables), 12)`, table files 1-12 in csv, md, tex exist, `reconciliation_report.csv` and `.md` exist, `status_map[1] == 'RECONCILED_PARAMETERS'`, `status_map[5] == 'DYNAMIC_ARTIFACT_LOADED'` | PASS |

## Gate

- `PYTHONPATH=src .venv/bin/python3 -m unittest tests/test_r08_reconciliation_and_tables.py` - 6 passed, 0 failed
- `PYTHONPATH=src .venv/bin/python3 -m unittest tests/test_manuscript_compilation.py` - 8 passed, 0 failed
