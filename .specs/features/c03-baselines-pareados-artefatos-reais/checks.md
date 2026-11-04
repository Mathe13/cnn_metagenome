# C03 implementation checks

Profile: light  
Plan: `.specs/features/c03-baselines-pareados-artefatos-reais/plan.md`

## Proof map

| Acceptance criteria | Implementation/proof | Status |
| --- | --- | --- |
| S1.1 inventory has no side effects | `scripts/prepare_baseline_benchmark.py --mode inventory`; `inventory_inputs` | Implemented; real inputs absent locally |
| S1.2 deterministic per-class selection and row identity | `preparation.prepare_queries`, `query_identity.csv`, preparation manifest | Implemented; requires full input partitions |
| S1.3 shared `3ε/4` replacement protocol | `_noisy_sequence`, `queries_noise_*.csv/.fasta`, query manifests | Implemented; deterministic per query/rate |
| S1.4 shared content and class identity | Query manifest hash, IDs and label validation during ingestion | Implemented |
| S1.5 canonical exact overlap | `canonical_sequence_hash`, `audit_exact_overlap` | Implemented; forward and reverse complement |
| S1.6 historical substitution scenario isolation | Loader accepts only `resample_including_original_v1` | Implemented |
| S2.7–12 tool outputs, checkpoints, aliases, failures and legacy import | `run_c03_tool_baseline.py`, `run_c03_cnn_inference.py`, `write_method_condition_manifest`, strict loader | Implemented; result commands run individually from the generated 35-command plan; legacy inputs without provenance are rejected |
| S3.13–19 exact joins, abstentions, BLAST ranking, metrics | Strict loader, parsers, `evaluate_predictions`, notebook 09 | Implemented; exact IDs/labels, output hashes, top-k domain, abstention reasons, full-N denominators and explicit Top-5 availability are checked |
| S4.20–26 paired outputs and evidence status | Notebook 09, run-specific C00 manifests, C03 result metadata | Implemented structurally; no real package was available to certify, so empirical result generation remains outstanding |

## Local checks

- `python3 -m py_compile src/viral_cnn/baselines/overlap.py src/viral_cnn/baselines/preparation.py scripts/prepare_baseline_benchmark.py`
- `PYTHONPATH=src python3 -m unittest tests/test_c03_baselines.py`
- `REMOTE_SERVER=False python3 scripts/prepare_baseline_benchmark.py --mode inventory`
- `REMOTE_SERVER=False python3 scripts/prepare_baseline_benchmark.py --mode plan --project-root /tmp --run-id c03-plan-review-001`
- Parse all code cells in notebook 09 with Python `ast` after validating its JSON.

## Evidence boundary

Unit tests cover contracts only. The local `results_reproduction/baselines` artifacts are not C03 empirical evidence. The inventory/plan ran locally and reported 35 required conditions, but remote train/test partitions, Kraken2/BLASTN executables and databases, and all five Species checkpoints are absent from this session. Forward/reverse-complement no-exact-overlap does not establish an unseen genome. Historical forced-substitution outputs and legacy manifests cannot be relabeled as `c03_paired_v1`.
