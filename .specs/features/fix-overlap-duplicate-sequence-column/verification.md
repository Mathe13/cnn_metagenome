# fix-overlap-duplicate-sequence-column verification

**Verdict**: PASS
**Profile**: light
**Diff range**: 6566831..working-tree
**Round**: 1 - full
**Verifier**: independent sub-agent (author != verifier)

## Checks

| Check | Claim | Proof run | Evidence | Result |
| --- | --- | --- | --- | --- |
| C1 | `audit_exact_overlap` extracts scalar string sequences from the first matching column when `test_queries` has duplicate `"sequence"` columns, returning `audit_df` of length `N` and boolean `masks` without raising `AttributeError` | `REMOTE_SERVER=False .venv/bin/python3 -m unittest tests/test_baselines.py -v` (`test_audit_exact_overlap_handles_duplicate_sequence_columns`) exit 0 | `tests/test_baselines.py:144` - `self.assertEqual(list(df_dup.columns), ["sequence", "sequence", "label", "query_id"])` · `tests/test_baselines.py:147` - `self.assertEqual(len(audit_df), 2)` · `tests/test_baselines.py:151` - `self.assertEqual(list(masks["overlap"]), [True, False])` | PASS |
| C2 | `audit_exact_overlap` accepts `pd.DataFrame` inputs whose sequence column is named `"sequence_padded"`, `"padded_Representation"`, or `"sequence"` without requiring prior column renaming, resolving in priority order `("sequence_padded", "padded_Representation", "sequence")` | `REMOTE_SERVER=False .venv/bin/python3 -m unittest tests/test_baselines.py -v` (`test_audit_exact_overlap_resolves_canonical_sequence_column_names`) exit 0 | `tests/test_baselines.py:165` - `self.assertEqual(len(audit_df), 2)` · `tests/test_baselines.py:168` - `self.assertEqual(list(masks["overlap"]), [True, False])` · `tests/test_baselines.py:178` - `self.assertEqual(int(audit_both.loc[0, "train_count"]), 1)` | PASS |
| C3 | `notebooks/reproduction/short/09_benchmark_cnns_kraken2_blast_overlap.ipynb` builds `df_test_queries` with unique columns `["query_id", "sequence", "label"]` from the selected priority column `col` without calling `.rename(columns={col: "sequence"})` on multi-column Parquet samples, and executes end-to-end in smoke-test mode | `REMOTE_SERVER=False .venv/bin/python3 -m unittest tests/test_baselines.py -v` (`test_notebook_09_unique_sequence_column_construction`) exit 0 | `tests/test_baselines.py:188` - `self.assertNotIn("rename(columns={col: \"sequence\"})", cell_sources)` · `tests/test_baselines.py:197` - `self.assertEqual(error_outputs, [])` · `tests/test_baselines.py:215` - `self.assertEqual(list(df_test_queries.columns), ["query_id", "sequence", "label"])` | PASS |

## Swept existing

- `dependency failure: existing` — verified at `notebooks/reproduction/short/09_benchmark_cnns_kraken2_blast_overlap.ipynb:107` (`if ctx.is_smoke_test or test_parquet is None or not Path(test_parquet).exists():`) and `src/viral_cnn/notebook_runtime.py:455` (`if not p.exists() or p.is_dir(): continue`).
- `observability: existing` — verified at `src/viral_cnn/notebook_runtime.py:443-447` (`log_artifact_path("ParquetSample", files, extra=f"split={split}, max_rows={max_rows}")`).

## Gate

`REMOTE_SERVER=False .venv/bin/python3 -m unittest tests/test_baselines.py -v` - 25 passed, 0 failed
