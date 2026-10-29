# fix-overlap-duplicate-sequence-column checks

Profile: light

## Intent

When `read_parquet_sample` loads a Parquet partition from `processed_kraken_species_splited_parquet` that contains both `sequence_padded` and `sequence` columns in its schema, renaming `sequence_padded` to `sequence` in `09_benchmark_cnns_kraken2_blast_overlap.ipynb` produces a DataFrame with duplicate `sequence` column labels. Indexing `row["sequence"]` inside `audit_exact_overlap` then yields a 2-element `pandas.Series` instead of a `str`, raising `AttributeError: 'Series' object has no attribute 'strip'` inside `canonical_sequence_hash`, and breaks `df_test_queries["sequence"].tolist()` in downstream cells. When this fix ships, `audit_exact_overlap` extracts a 1D sequence series even when input DataFrames have duplicate column names or canonical `sequence_padded` / `padded_Representation` columns, and `09_benchmark_cnns_kraken2_blast_overlap.ipynb` constructs `df_test_queries` with unique `["query_id", "sequence", "label"]` columns.

3 checks in 1 slice · 0 one-way doors · 0 open, of which 0 block

## Checks

### S1 - Overlap DataFrame column deduplication and notebook query construction · 3 files · 28 KB · ~7k

**C1** - `audit_exact_overlap` extracts scalar string sequences from the first matching column when `test_queries` is a `pd.DataFrame` with duplicate `"sequence"` columns (such as after `.rename(columns={"sequence_padded": "sequence"})` on a schema containing both `"sequence_padded"` and `"sequence"`), returning an `audit_df` of length `N` and boolean `masks` without raising `AttributeError`
Proof: `.venv/bin/python3 -m unittest tests/test_baselines.py -k test_audit_exact_overlap_handles_duplicate_sequence_columns`

**C2** - `audit_exact_overlap` accepts `pd.DataFrame` inputs whose sequence column is named `"sequence_padded"`, `"padded_Representation"`, or `"sequence"` without requiring prior column renaming, resolving in priority order `("sequence_padded", "padded_Representation", "sequence")`
Proof: `.venv/bin/python3 -m unittest tests/test_baselines.py -k test_audit_exact_overlap_resolves_canonical_sequence_column_names`

**C3** - `notebooks/reproduction/short/09_benchmark_cnns_kraken2_blast_overlap.ipynb` builds `df_test_queries` with unique columns `["query_id", "sequence", "label"]` from the selected priority column `col` without calling `.rename(columns={col: "sequence"})` on multi-column Parquet samples, and executes end-to-end in smoke-test mode
Proof: `.venv/bin/python3 -m unittest tests/test_baselines.py -k test_notebook_09_unique_sequence_column_construction`

## Coverage

| Set (size) | Member -> proof | Unproven |
| --- | --- | --- |
| Supported sequence column layouts in `audit_exact_overlap` (4) | duplicate `"sequence"` columns C1 · `"sequence_padded"` C2 · `"padded_Representation"` C2 · single `"sequence"` C2 | - |
| Touched runtime surfaces (2) | `src/viral_cnn/baselines/overlap.py` C1 · `notebooks/reproduction/short/09_benchmark_cnns_kraken2_blast_overlap.ipynb` C3 | - |

## Swept

- validation: C1, C2 (`audit_exact_overlap` validates presence of a sequence column among `("sequence_padded", "padded_Representation", "sequence")` and coerces duplicate column slices to 1D series)
- failure modes: C1 (prevents `AttributeError: 'Series' object has no attribute 'strip'` when duplicate `"sequence"` columns exist)
- idempotency: C1, C3 (constructing `df_test_queries` with explicit `["query_id", "sequence", "label"]` columns produces identical SHA-256 overlap counts across repeated calls)
- authorization: n/a - offline scientific genomics evaluation module with no authentication or network boundaries
- concurrency: n/a - single-process in-memory DataFrame and SHA-256 hash dictionary audit
- data lifecycle: n/a - read-only Parquet sample inspection and deterministic CSV output overwrite
- dependency failure: existing - `read_parquet_sample` and smoke-test fallback in `09_benchmark_cnns_kraken2_blast_overlap.ipynb` guard missing cluster paths
- state transitions: n/a - stateless pure function `audit_exact_overlap`
- observability: existing - `read_parquet_sample` logs `[LOG_PATH] Loading ParquetSample from: ...` via `log_artifact_path`

## Out of scope

- Modifying Parquet files on the cluster - read-only dataset artifacts per `AGENTS.md`

## Handoff

- S1 = ~7k tokens across 3 files (`src/viral_cnn/baselines/overlap.py`, `notebooks/reproduction/short/09_benchmark_cnns_kraken2_blast_overlap.ipynb`, `tests/test_baselines.py`), under the 150k budget - one builder
- **Boundary:** C1-C3 closed at `6566831` (working tree)
- **Settled mid-build:** none
- **Abandoned:** none
