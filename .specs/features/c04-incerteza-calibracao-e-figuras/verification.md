# C04 verification

## Local verification

Executed in the local CPU environment:

- `.venv/bin/python3 -m unittest tests.test_c04_uncertainty tests.test_calibration tests.test_n02_benchmark_and_calibration.TestN02BenchmarkAndCalibration.test_r03_audit_dataloader_calibration_temperature_scaling` — 27 tests passed.
- Notebook JSON and all code-cell syntax parsed for reproduction notebooks 01/02.
- `python3 -m compileall -q src/viral_cnn/eval tests/test_c04_uncertainty.py` — passed.

These checks validate mechanics on small fixtures only. Synthetic results are not empirical evidence for the manuscript.

## Empirical verification required

Run notebooks `01_benchmark_clean_and_noise.ipynb` and `02_analyze_confidence_and_calibration.ipynb` in the remote JupyterHub environment with the same `C04_RUN_ID` (for example, export a unique value such as `C04_RUN_ID=c04-empirical-20261103` before starting Jupyter). Confirm the manifests identify real (`is_smoke_test: false`) test/validation partitions and all seven required checkpoints, then inspect the expected 35 benchmark and 25 calibration cells. The local `/home/mlima/workspace` data volume and Species checkpoints were absent during implementation; no empirical C04 result or reproduction is claimed.

The C04 bootstrap interval is conditional on the observed fragment population and fixed taxon supports. It does not quantify training-seed, genome, or species-level uncertainty. The existing fragment split and class conventions are retained; historical outputs are not migrated.
