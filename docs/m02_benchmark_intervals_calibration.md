# M02: benchmark, intervals and calibration

M02 prepares reproduction notebooks 01/02/09 for isolated runs. It preserves the
closed-set fragment split and the C03/C04 estimands. It does not retrain models or
execute Kraken2/BLASTN from notebook 09. A valid prepared benchmark is a prerequisite.

## Run configuration

Use the repository root as the working directory. Set variables in the Jupyter
kernel environment before execution. `REMOTE_SERVER` defaults to remote topology;
set `SMOKE_TEST=0` explicitly for empirical runs. M02 disables automatic synthetic
fallback. `DATA_ROOT` identifies the canonical input root. Set `LOG_PATH=0` only
when the default artifact path logs are unnecessary.

Standalone configuration is optional:

```bash
export REMOTE_SERVER=true SMOKE_TEST=0
export DATA_ROOT=/path/to/verified-data
export M02_CONFIG="$PWD/configs/reproduction_m02.yaml"
export C03_PREPARED_DIR=/path/to/completed-c03/prepared
.venv/bin/python3 scripts/preflight_notebook.py short/09 --data-root "$DATA_ROOT"
```

`C03_PREPARED_DIR` overrides the standalone configuration value. Relative paths
inside that setting use DATA_ROOT. Set `population_id` and `split_id` to verified
identities in a local copy of the config; unverified values remain `unknown`.
`M02_RUN_ID` can set a standalone ID, but an occupied destination fails.
`C04_RUN_ID` remains a compatibility alias when M02_RUN_ID is absent.

For a prepared M00 round, export `RUN_PLAN` instead of M02_CONFIG. The plan fixes
run/parent IDs, reservations, input paths and the C03 manifest/hash. Divergent
DATA_ROOT/C03_PREPARED_DIR/SMOKE_TEST overrides fail. Future/incomplete dependencies
remain blocked. Configure the explicit C03 preparation manifest before planning:

```bash
unset M02_CONFIG
export RUN_PLAN=/path/to/round/m00/run_plan.json
.venv/bin/python3 scripts/preflight_notebook.py short/09 --run-plan "$RUN_PLAN"
```

Notebook 01 uses Species and Genus test splits, five Species checkpoints and two
Genus checkpoints. Notebook 02 uses Species validation/test, the Species mapping,
verified hybrid taxonomy and the five Species checkpoints. Their requirements
are declared in configs/reproduction_m00.yaml. M00 destinations are adopted only
when empty and claimed exclusively. A retry needs a new reservation.

## Execute

After preflight, execute the notebook from the repository root:

```bash
.venv/bin/jupyter nbconvert --execute --to notebook --stdout \
  notebooks/reproduction/short/01_benchmark_clean_and_noise.ipynb \
  > /tmp/m02-01.executed.ipynb
.venv/bin/jupyter nbconvert --execute --to notebook --stdout \
  notebooks/reproduction/short/02_analyze_confidence_and_calibration.ipynb \
  > /tmp/m02-02.executed.ipynb
.venv/bin/jupyter nbconvert --execute --to notebook --stdout \
  notebooks/reproduction/short/09_benchmark_cnns_kraken2_blast_overlap.ipynb \
  > /tmp/m02-09.executed.ipynb
```

Outputs go to `results_reproduction/runs/<run_id>/short/<01|02|09>/`.
The terminal prints the selected paths. Standalone manifests are `manifest.json`;
M00 uses its explicitly selected manifest filename, including
`manifest_09_baselines.json` for 09 in the standard config. Python exceptions in
notebook cells record `FAILED`. Abrupt kernel/process termination can leave
`RUNNING`; it never becomes admissible evidence without a successful completion.

## C03 preflight and inputs

The specialized 09 preflight reads materialized C03 tables and full hashes. This
is more than metadata inspection and can be expensive on the cluster. It loads
no model and starts no search tool. It requires:

- `preparation_manifest.json` with explicit `status=COMPLETE`, empirical mode,
  canonical class order, query identities, sources and overlap audit.
- Database build manifests with `state=COMPLETE`, empirical mode, identified
  references and all six expected Kraken2/BLASTN database outputs.
- Exactly one admissible manifest/output for every method and rate: five CNNs,
  Kraken2-flat and BLASTN × 0/0.02/0.05/0.10/0.15.
- Same query IDs, truth labels and materialized sequence content at each rate,
  shared class order and training reference, identified checkpoints, canonical
  absten­tion handling and matching hashes.

Old preparation exports without explicit COMPLETE remain read-only history;
regenerate an export in a new namespace with the current C03 exporter. Do not add
status retrospectively to claim that an old run completed. Existing method
manifests must still correspond to the exact current query-manifest hashes.
The ResNet presentation alias `resnet50_species` maps explicitly to the verified
variant `resnet_clean_species`; it does not authorize swapping checkpoints.

## Products and statistical scope

01 retains profiling, clean/noisy metrics, F1 bands, specialist comparisons and
C04 per-query evidence/uncertainty. `c04_notebook_manifest.json` retains the C04
report; the M02 run manifest links outputs and their SHA-256 identities.

02 retains confusion, entropy, ECE, Brier (`brier_score` and equivalent `brier`
column), NLL and reliability diagrams. Temperature is selected on identified
clean validation and sealed before test. The tqdm wrapper preserves access to
partition identity. Reliability figures display N and preserve before/after
support. Random GradCAM is a smoke demonstration only; `gradcam_status.json`
records `omitted_empirical` in empirical mode, without invented biological input.

09 writes `exact_overlap_audit.csv`, `benchmark_metrics_stratified.csv` (105 cells),
`paired_discordance.csv`, `species_cluster_bootstrap.csv` (150 cells),
`figure_sources.json`, and accuracy/coverage/paired-difference figures in PNG/PDF.
The CSVs are the figure inputs. All seven methods remain in tables and figures.
The three panels are `all`, `overlap` and `no_exact_overlap`, with N.

Accuracy uses all N queries; label -1 remains an error. Coverage counts classified
queries/N; conditional accuracy uses only classified queries. Accuracy and
coverage intervals are individual Wilson 95% intervals. Deltas are CNN minus
tool accuracy in percentage points, with paired species cluster bootstrap,
2,000 repetitions and seed 42. Empty strata/less than two species retain an
explicit unavailable reason and no fabricated bounds. A point estimate can
remain available when its interval is not.

C04 uses stratified fragment bootstrap with fixed true-taxon support, 95%
confidence, 2,000 repetitions and seed 42. C03 and C04 populations/denominators are
separate. All runs record one noise realization per rate. These intervals are
conditional on the observed population and do not estimate training-to-training
variability or generalization to unseen genomes. No exact overlap is not proof
of genome independence. Genome identity remains `unknown` without accessions.
New figures are M02 products, without automatic assignment to article F IDs.

## Local verification and remaining evidence

```bash
REMOTE_SERVER=False PYTHONPATH=src .venv/bin/python3 -m unittest -v \
  tests.test_m02_benchmark_intervals_calibration
REMOTE_SERVER=False PYTHONPATH=src .venv/bin/python3 -m unittest -v \
  tests.test_c03_baselines tests.test_c04_uncertainty tests.test_calibration \
  tests.test_m00_run_plan tests.test_c00_preflight_manifest tests.test_notebook_runtime
```

The tests use temporary protocol fixtures, including dummy database/checkpoint
bytes that are never executed as models or search databases. Notebook 01/02
integration uses a tiny CPU backbone. All outputs are synthetic contract checks,
not empirical evidence. Smoke fixtures are generated only in the new run's
`fixtures/` directory, with 64 rows/split and distinct validation identity.
They never overwrite DATA_ROOT. The M01 overwritten local Parquets are not used.

Canonical cluster Parquets, real checkpoints, database outputs and completed
35-condition C03 runs are unavailable in this local environment. Remote preflight,
full inference and paper-number reconciliation remain pending. No table/figure
number or metric in the paper was reproduced here. Historical notebooks and
results are preserved. See the feature verification report for exact local
checks, failures encountered and the independent review.
