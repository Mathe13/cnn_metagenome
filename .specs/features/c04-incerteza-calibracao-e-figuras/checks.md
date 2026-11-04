# C04 checks

- [x] C04 uncertainty protocol implemented as class-stratified fragment bootstrap (fixed true-class support; 2,000 replicates and seed 42 at notebook call sites).
- [x] Accuracy, Top-5, Macro-F1 and Weighted-F1 estimates retain class universe and units; Top-5 is unavailable if per-query hits are absent.
- [x] Benchmark loader requests partition-hash/row-ordinal identities and runner retains per-query Top-5 hits.
- [x] Compact Parquet evidence shards persist query ID, truth, prediction, confidence and Top-5 hit with SHA-256 sidecars.
- [x] Temperature scaler tracks UNFITTED/FITTED/SEALED state, records validation selection/provenance and rejects refit after sealing.
- [x] Calibration audit defaults to all validation batches; explicit smoke cap is recorded.
- [x] Calibration output includes unrounded before/after reliability bins, count/sums, and Top-1 preservation diagnostics.
- [x] Reliability plotting omits empty-bin accuracy points and failed PNG generation leaves no invalid image placeholder.
- [x] Notebook outputs use run-specific C04 namespaces, completeness matrices and manifests.
- [ ] Full 35 benchmark conditions and 25 calibration panels verified empirically on JupyterHub real partitions/checkpoints.
- [ ] Independent analysis of C04 outputs reconciled against historical notebook and article values.
