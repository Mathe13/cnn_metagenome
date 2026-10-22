# S06: Long-Sequence Inference Audit and Voter Extraction

## Problem

The manuscript's Section 3.5 presents sliding-window taxonomic consensus on variable-length genomic sequences, reporting that standard ResNet50 achieved 72.69% species-level mean-logit accuracy while ConvNeXt variants suffered a severe collapse down to 3.49%–19.99% (Table 12).
Reviewer feedback (Note v2 on Section 3.5) specifically identified two critical vulnerabilities:
1. ConvNeXt performance on long sequences (3.49%–20.38%) drops precipitously compared to 512 bp fragments (58%–61%), and genus-transferred ConvNeXt (`ConvNeXt_T`) plummets to 3.49% while the scratch model (`ConvNeXt_Z`) obtains 12.01%.
2. Long sequences were downloaded from NCBI Nucleotide for species present in training and may coincide with the exact reference genomes from which training fragments were derived.

Currently, long-sequence inference is trapped in ad-hoc Jupyter notebooks (`long/cnn-long-voter.ipynb`, `long/cnn-long-voter-compare.ipynb`, and `long/prepare database.ipynb`).
The historical implementation (`text_to_tensor` in cell 15 of `cnn-long-voter.ipynb`) silently discarded sequences shorter than 512 bp (`if not windows: return None`), dropped up to 255 bp of trailing end-sequence per read (`seq_len - window_size + 1`), omitted uppercase normalization (`.upper()`), and relied on an ad-hoc remapping table (`mapping_species_final.csv`) with only 13,036 species instead of the canonical 13,683 classes.
Furthermore, the simulation script (`calculate_n_reads` in `prepare database.ipynb`) generated 500 reads for large viral genomes (>100 kb) versus only 50 reads for small genomes (<10 kb), introducing massive class-imbalance bias toward giant viruses (e.g., *Pandoravirus*, *Chlorella virus*, *Mollivirus*, *Orpheovirus*, and *Bracoviriform*) in error counts.
Without an auditable investigation, presenting 72.69% as generalization to unseen genomes or attributing the ConvNeXt collapse to inherent architectural failure lacks scientific validity.

When this ships, `docs/long_sequence_inference_audit.md` provides an exhaustive causal report distinguishing verifiable mechanisms from negative checks; `src/viral_cnn/inference/voter.py` implements a modular `LongSequenceVoter` preserving the historical protocol (window 512, stride 256, pre-softmax logit averaging, majority voting with mode tie-breaking) while introducing robust handling for contigs <512 bp; `src/viral_cnn/inference/audit_long.py` provides exact window logit equivalence proofs, training reference genome overlap quantification, and error stratification; `scripts/run_long_voter.py` provides an auditable CLI; and contract tests verify all invariants without requiring multi-gigabyte remote checkpoints.

## Flow

Reuses `src/viral_cnn/models/registry.py`, `src/viral_cnn/data/encoding.py`, and `src/viral_cnn/eval/metrics.py` while preserving existing model checkpoints and notebook artifacts in place.

1. variable-length genomic sequence string -> `viral_cnn.inference.voter:slice_sequence` (new, door 1) - slices sequences into 512 bp windows with stride 256 bp, supporting both historical skip mode (`handle_short='skip'`) and standard padded mode (`handle_short='pad'`)
2. batched window token tensors -> `viral_cnn.inference.engine:predict_batch` (exists) - executes model forward passes over all windows and collects unnormalized raw logits
3. window logits -> `viral_cnn.inference.voter:aggregate_window_logits` (new, door 2) - computes pre-softmax mean logits (`mean_logits.argmax()`) and hard majority voting (`mode(window_winners)` with deterministic tie-breaking)
4. sequence predictions, lengths, and ambiguity rates -> `viral_cnn.inference.audit_long:stratify_long_errors` (new, door 3) - stratifies classification error across length bins, ambiguous character fractions, and false-positive species counts
5. `scripts/run_long_voter.py` (new, no door - placement per conventions) - CLI executing long-sequence consensus inference and writing structured prediction parquets and audit summaries
6. out: auditable consensus predictions, stratified error reports, and scientific audit documentation in `docs/long_sequence_inference_audit.md`

## Impact

| Front | What changes |
| --- | --- |
| domain | new term: `LongSequenceVoter` - sliding-window consensus classifier aggregating fixed-window (512 bp) model predictions over variable-length contigs via pre-softmax mean logits and majority voting |
| domain | new term: `VoterExecutionMode` - execution contract distinguishing historical behavior (`mode='historical_skip'`, omitting sequences <512 bp and trailing residues) from robust standardized behavior (`mode='standard_pad'`, padding contigs <512 bp to 512 bp with null vectors) |
| domain | new term: `ReferenceGenomeOverlapFraction` - proportion of long test contig windows sharing exact sequence or high $k$-mer identity with training reference fragments, bounded between 0.0 and 1.0 |
| stored data | nothing to migrate: historical notebooks (`long/*.ipynb`) and parquets remain untouched; new inference outputs write to user-configured directories |

## Relations

None - no stored-data shape change

## Surface

None - nothing consumed outside

## Landing

| One-way door | Literal shape | Alternative rejected |
| --- | --- | --- |
| Voter consensus aggregation contract | `mean_logits = logits.mean(dim=0)` with `pred_mean = mean_logits.argmax().item()` and `pred_majority = torch.mode(logits.argmax(dim=1)).values.item()` | post-softmax probability averaging or soft voting with temperature - rejected because Table 12 specifically evaluated pre-softmax average pooling and hard window majority voting |
| Short contig (<512 bp) handling modes | parameter `handle_short: Literal['skip', 'pad'] = 'pad'` where `'skip'` raises or returns empty result and `'pad'` pads right with token 4 (null vector) up to 512 bp | silently dropping short sequences unconditionally - rejected because dropping sequences distorts evaluation support and fails on viral contigs <512 bp |
| Deterministic tie-breaking specification | `torch.mode` returns smallest class index among maximum-count ties; tie-breaking policy explicitly recorded as `tie_break='min_index'` | arbitrary random choice or non-deterministic hash ordering - rejected because scientific reproduction requires deterministic matching of notebook outputs |

- Nothing else in this change is hard to reverse

## Criteria

### S1: Long-Sequence Slicing and Modular Voter Engine (P1)

Implementation of the sliding-window slicing and aggregation voter supporting both historical and robust execution modes.

**Acceptance Criteria**

1. WHEN given a sequence of length $\ge 512$ bp THEN the system SHALL extract windows of 512 bp with a stride of 256 bp and map characters strictly to uppercase A, C, G, T tokens (0, 1, 2, 3) and ambiguous/padding tokens to 4.
2. IF a sequence has length $< 512$ bp AND `handle_short='skip'` THEN the system SHALL return no windows, matching historical notebook behavior.
3. IF a sequence has length $< 512$ bp AND `handle_short='pad'` THEN the system SHALL pad the sequence to 512 bp using token 4 (null vector in one-hot embedding) and return exactly 1 window.
4. WHEN aggregating logits across $N$ windows THEN the system SHALL compute `pred_mean_logits = logits.mean(dim=0).argmax().item()` and `pred_majority_vote = torch.mode(logits.argmax(dim=1)).values.item()`.
5. The system SHALL resolve majority voting ties deterministically by selecting the smallest class index among classes with the maximal window count.

**Independent test:** run unit tests in `tests/test_voter.py` asserting that window slicing with length 1024 bp produces exactly 3 windows (`[0:512]`, `[256:768]`, `[512:1024]`), contigs <512 bp behave according to `handle_short`, and aggregation matches known pre-softmax mean logits and majority mode.

### S2: Logit Equivalence and Invariant Integrity Verification (P1)

Verification that individual window forward passes match the short-read pipeline and validation of taxonomic mapping offsets.

**Acceptance Criteria**

6. WHEN a single 512 bp window is evaluated through `viral_cnn.inference.engine:predict_batch` and through `viral_cnn.inference.voter:LongSequenceVoter` THEN the system SHALL produce identical output logits with maximum absolute difference $< 10^{-5}$.
7. The system SHALL verify that predicted class indices $[0, N_{\text{classes}} - 1]$ convert to 1-based taxonomic IDs via `tax_id = model_label + 1`, matching AD-002.
8. The system SHALL verify that ambiguous IUPAC characters (`N`, `R`, `Y`, `S`, `W`, `K`, `M`, `B`, `D`, `H`, `V`) are converted to token 4 and produce null one-hot vectors $[0.0, 0.0, 0.0, 0.0]$ without raising runtime exceptions.
9. IF a model head has output dimension $C \neq 13683$ for species rank THEN the system SHALL reject execution with ValueError indicating class count mismatch.

**Independent test:** run unit tests in `tests/test_voter.py` comparing logits between `engine.py` and `voter.py` on identical sequences, verifying label offsets and IUPAC null vectors.

### S3: Training Genome Overlap, Error Stratification, and Audit Report (P1)

Quantification of sequence overlap between long contigs and training references, stratification of classification errors, and publication of the audit report.

**Acceptance Criteria**

10. The system SHALL publish a comprehensive audit report in `docs/long_sequence_inference_audit.md` documenting:
    - the technical and methodological causes of the ConvNeXt performance drop (including read-length generation bias favoring giant viruses with 500 reads vs 50 reads, logit magnitude/bias pulling in pre-softmax averaging, and checkpoint lineage of `ConvNeXt_T`);
    - the confirmation that the 72.69% accuracy of standard ResNet50 evaluated on NCBI sequences reflects reference genomes seen during training and SHALL NOT be reported as generalization to unseen genomes;
    - explicit negative checks ruling out label inversion, incorrect channel order, and missing softmax.
11. The system SHALL quantify the sequence overlap between evaluated long contigs and training data, calculating the fraction of test windows sharing 100% identity or high $k$-mer overlap with training reference sequences.
12. WHEN given model prediction outputs with sequence lengths and ambiguous character counts THEN the system SHALL stratify Top-1 accuracy across length bins ($[512, 2000)$, $[2000, 5000)$, $[5000, 10000]$ bp) and ambiguity fractions ($0\%$, $(0\%, 5\%]$, $>5\%$).
13. WHEN executing `scripts/run_long_voter.py` with an input fasta or csv and a model variant THEN the system SHALL return exit code 0, emit structured predictions containing `pred_majority_vote` and `pred_mean_logits`, and print summary accuracy.

**Independent test:** run `python3 scripts/run_long_voter.py --model-variant resnet_clean_species --input tests/fixtures/sample_genomes.csv --handle-short pad --output /tmp/test_voter_out.parquet` and assert exit code 0 and valid output schema.

## Out of scope

| Excluded | Why |
| --- | --- |
| Re-running multi-day NCBI nucleotide queries for all 13,683 species | NCBI scraping was already executed in historical `long/download_from_ncbi.ipynb`; local audit uses existing manifests and synthetic fixtures |
| Training new 1D CNN models from scratch | S06 audits existing checkpoints and voter inference; new architectures or training runs belong to future research extensions |
| Simulating Oxford Nanopore squiggle or homopolymer models | As established in AGENTS.md and S05, noise protocols evaluate historical point corruptions; sequencer-specific physics is out of scope |

## Assumptions

| Assumption | Chosen default | Rationale | Confirmed? |
| --- | --- | --- | --- |
| Window size and stride | 512 bp window, 256 bp stride | Exact configuration specified in manuscript Section 3.5 and `cnn-long-voter.ipynb` | y |
| Default short sequence policy for CLI | `pad` | Robust standard behavior preventing silent data loss while supporting `skip` via flag | y |
| Overlap audit k-mer length | $k=15$ | Standard viral taxonomy k-mer length matching Kraken2 minimizer sensitivity | y |

**Open questions:** none - all resolved or logged above.

## Observable

| Surface | Decision | Landing |
| --- | --- | --- |
| Python API `viral_cnn.inference.voter` | functions `slice_sequence`, `aggregate_window_logits`, and class `LongSequenceVoter` | AC 1, 2, 3, 4, 5 |
| Python API `viral_cnn.inference.audit_long` | functions `verify_window_logit_equivalence`, `quantify_training_overlap`, `stratify_long_errors` | AC 6, 11, 12 |
| CLI `scripts/run_long_voter.py` | CLI flags (`--input`, `--model-variant`, `--window-size`, `--stride`, `--handle-short`, `--output`) | AC 13 |
| CLI `scripts/run_long_voter.py` | error handling and exit codes (exit 0 on success, exit 2 on invalid arguments) | AC 9, 13 |
| screen | n/a - no user-facing graphical screen exists; interaction is programmatic and CLI-based | n/a - no GUI surface |

## Sources

- Manuscript Section 3.5: "Long-Sequence Classification via Sliding-Window Consensus", Table 12, and Note v2
- Historical notebooks: `long/cnn-long-voter.ipynb`, `long/cnn-long-voter-compare.ipynb`, and `long/prepare database.ipynb`
- Repository rules: `AGENTS.md` and Architectural Decisions AD-001 to AD-009 in `.specs/STATE.md`

