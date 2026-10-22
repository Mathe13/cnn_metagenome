# S06: Long-Sequence Inference Audit and Voter Extraction - Checks

Profile: light
Plan: `.specs/features/s06-long-sequence-inference-audit/plan.md`

## Checks

### S1: Long-Sequence Slicing and Modular Voter Engine · 2 files · 18 KB · ~5k

**C1** - `LongSequenceVoter.slice_sequence` extracts 512 bp windows with stride 256 bp and normalizes characters to uppercase A, C, G, T (0, 1, 2, 3) and ambiguous/padding to token 4 (AC 1)
Proof: `.venv/bin/python3 -m unittest discover -s tests -k test_slice_sequence_window_stride_and_token_normalization`

**C2** - `LongSequenceVoter.slice_sequence` with `handle_short='skip'` returns 0 windows for sequences <512 bp (AC 2)
Proof: `.venv/bin/python3 -m unittest discover -s tests -k test_slice_sequence_handle_short_skip`

**C3** - `LongSequenceVoter.slice_sequence` with `handle_short='pad'` pads contigs <512 bp to 512 bp using token 4 and returns exactly 1 window (AC 3)
Proof: `.venv/bin/python3 -m unittest discover -s tests -k test_slice_sequence_handle_short_pad`

**C4** - `LongSequenceVoter.aggregate_window_logits` computes pre-softmax mean logits (`mean_logits.argmax()`) and hard window majority voting (`mode(window_winners)`) (AC 4)
Proof: `.venv/bin/python3 -m unittest discover -s tests -k test_aggregate_window_logits_mean_and_majority`

**C5** - `LongSequenceVoter.aggregate_window_logits` resolves majority voting ties deterministically by selecting the smallest class index (`tie_break='min_index'`) (AC 5)
Proof: `.venv/bin/python3 -m unittest discover -s tests -k test_aggregate_window_logits_deterministic_tie_break`

### S2: Logit Equivalence and Invariant Integrity Verification · 2 files · 20 KB · ~5k

**C6** - A single 512 bp window evaluated via `engine.predict_batch` and via `LongSequenceVoter` produces identical logits within absolute tolerance $< 10^{-5}$ (AC 6)
Proof: `.venv/bin/python3 -m unittest discover -s tests -k test_voter_window_logit_equivalence_with_engine`

**C7** - Predicted class indices [0, 13682] convert to 1-based taxonomic IDs via `tax_id = model_label + 1` matching AD-002 (AC 7)
Proof: `.venv/bin/python3 -m unittest discover -s tests -k test_voter_label_offset_one_based_tax_id`

**C8** - Ambiguous IUPAC characters (`N`, `R`, `Y`, `S`, `W`, `K`, `M`, `B`, `D`, `H`, `V`) are mapped to token 4 and produce null one-hot vectors without exception (AC 8)
Proof: `.venv/bin/python3 -m unittest discover -s tests -k test_voter_iupac_ambiguity_produces_null_one_hot`

**C9** - Model loading and Voter reject classification heads with output dimension $C \neq 13683$ for species rank with `ValueError` (AC 9)
Proof: `.venv/bin/python3 -m unittest discover -s tests -k test_voter_rejects_class_dimension_mismatch`

### S3: Training Genome Overlap, Error Stratification, and Audit Report · 3 files · 26 KB · ~7k

**C10** - `docs/long_sequence_inference_audit.md` documents causes of ConvNeXt collapse (read-length generation bias 500 vs 50, logit pulling, head lineage) and confirms ResNet50 72.69% reflects seen reference genomes rather than unseen generalization (AC 10)
Proof: `.venv/bin/python3 -m unittest discover -s tests -k test_audit_report_scientific_claims_and_evidence`

**C11** - `viral_cnn.inference.audit_long.quantify_training_overlap` computes exact sequence matching and $k$-mer overlap fraction between long test contigs and training references (AC 11)
Proof: `.venv/bin/python3 -m unittest discover -s tests -k test_quantify_training_overlap_kmers_and_exact_match`

**C12** - `viral_cnn.inference.audit_long.stratify_long_errors` stratifies accuracy across length bins and ambiguity proportions (AC 12)
Proof: `.venv/bin/python3 -m unittest discover -s tests -k test_stratify_long_errors_by_length_and_ambiguity`

**C13** - CLI `scripts/run_long_voter.py` executes long sequence inference, writes structured parquet predictions, and exits with code 0 (AC 13)
Proof: `.venv/bin/python3 -m unittest discover -s tests -k test_run_long_voter_cli_exit_zero_and_prediction_output`

## Coverage

| Set (size) | Member -> proof | Unproven |
| --- | --- | --- |
| LongSequenceVoter short contig handling modes (2) | `skip` C2 · `pad` C3 | - |
| Consensus prediction aggregation methods (2) | `pred_mean_logits` C4 · `pred_majority_vote` C4 | - |
| Sequence length stratification intervals (3) | `len_512_to_2000` C12 · `len_2000_to_5000` C12 · `len_5000_to_10000` C12 | - |
| Ambiguity proportion stratification intervals (3) | `ambig_0_pct` C12 · `ambig_0_to_5_pct` C12 · `ambig_gt_5_pct` C12 | - |
| CLI execution status codes (2) | `0` C13 · `2` C9 | - |

- Claims naming a status code or response shape: C9, C13 - each has a proof crossing the process boundary
- No other check claims more than the single case its proof exercises

## Swept

- validation: C9
- failure modes: C2
- idempotency: C6
- authorization: n/a - local scientific CLI tool with no network or authorization boundary
- concurrency: n/a - deterministic batch processing on single machine with no concurrent state mutation
- data lifecycle: C10
- dependency failure: C6
- state transitions: n/a - stateless window slicing and logit aggregation
- observability: C13

## Handoff

- S1 = 5k in voter engine and window slicing; S2 = 5k in logit equivalence and contract verification; S3 = 7k in overlap audit, stratification, CLI, and docs. Total = ~17k tokens, well under the 150k budget - one builder

- **Mechanism:** one builder

