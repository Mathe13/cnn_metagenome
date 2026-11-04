# AD-020: Long-sequence evidence boundary

- Status: Accepted for C02 implementation
- Date: 2026-11-03
- Supersedes: the causal and genome-presence interpretation of AD-010; it preserves the 512-bp window, 256-bp stride, mean pre-softmax logits, majority vote, and short-sequence `pad`/`skip` policies.

## Decision

Long-sequence results are empirical only when five distinct Species checkpoints, an explicitly identified canonical label order, and exactly aligned query identities and truths are present in the same declared execution. A missing accession or reliable genome identifier has state `unknown`; `origin_len`, taxon, and k-mer overlap do not establish a genome match.

The historical 50/200/500 sequence-generation rule is a provenance fact, not a measurement of surviving data composition. Composition claims are computed from observed rows. Mean-logit and majority-vote differences, window influence, and calibration are diagnostics; they do not establish a cause for architectural or transfer performance.

Evidence records use `SUPPORTED`, `HYPOTHESIS`, or `INCONCLUSIVE`. Synthetic smoke data cannot produce an empirically `SUPPORTED` result. The published aggregation remains mean logits before softmax; calibration uses the softmax of that mean and does not modify predictions.

## Consequences

C02 artifacts use `long_sequence_protocol: c02_audit_v1` and an exclusive run namespace. Comparisons fail closed on missing or ambiguous class mappings, checkpoints, variants, query IDs, truths, or conditions. Historical manifests and paper values remain references, not inferred empirical evidence.
