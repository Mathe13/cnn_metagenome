# Long-sequence audit: C02 evidence status

## Evidence available in this checkout

The local repository contains the historical and reproduction notebooks, source code, and prior long-voter manifests. The previous manifests include result summaries and fixed claims, but do not prove the input provenance, canonical label ordering, or five-checkpoint paired comparison required by C02. They are not upgraded by this implementation.

At the C02 plan inspection, the five registered Species checkpoint files, the required long Species mapping, and the remote `/home/mlima/workspace` data tree were absent. The small local Parquet and prediction artifacts described in the plan are not a substitute for an empirical run. No C02 remote empirical execution has been performed.

## Protocol and interpretation

The implementation retains 512-bp windows, 256-bp stride, incomplete-tail dropping, deterministic minimum-index voting ties, pre-softmax logit averaging, and separate `pad` and `skip` handling. It records observed window counts and writes window logits only for a declared diagnostic sample. Exact genome identity remains unknown unless an explicit accession-level mapping is available.

AD-010's technical windowing and aggregation parameters remain. Its assertions that the test contigs were seen in training and that sampling disparity or logit pulling explains model performance are not established without the corresponding identified observations and tests. The historical 50/200/500 read-generation rule is recorded separately from observed counts.

## Verification boundary

Contract tests and local CPU smoke execution establish software behavior only. They do not reproduce Table 12 or support empirical diagnostic findings. A remote run requires five identified checkpoints, a mapping whose class order matches those checkpoints, the query source, and the run manifests from long/01–03. Missing or ambiguous inputs block the empirical comparison.
