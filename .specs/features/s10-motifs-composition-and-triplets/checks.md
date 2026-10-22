# S10: Motifs, Compositional Controls, and Triplet Analysis - Checks

Profile: light
Plan: `.specs/features/s10-motifs-composition-and-triplets/plan.md`

## Checks

### S1: First-Layer Filter Consensus Extraction & Occurrence Scanning · 2 files · 22 KB · ~6k

**C1** - `viral_cnn.interpretability.motifs.extract_first_layer_motifs` extracts consensus 7-mers using `consensus_rule="max_positive"` over first-layer convolution weights (AC 1, AC 2)
Proof: `.venv/bin/python3 -m unittest discover -s tests -k test_extract_first_layer_motifs_max_positive`

**C2** - `viral_cnn.interpretability.motifs.extract_first_layer_motifs` extracts consensus 7-mers using `consensus_rule="max_abs"` over first-layer convolution weights (AC 3)
Proof: `.venv/bin/python3 -m unittest discover -s tests -k test_extract_first_layer_motifs_max_abs`

**C3** - `viral_cnn.interpretability.motifs.scan_motif_occurrences` constructs 1D spatial convolution kernels and returns exact 0-based match coordinates across 4-channel sequences (AC 4)
Proof: `.venv/bin/python3 -m unittest discover -s tests -k test_scan_motif_occurrences_exact_coordinates`

**C4** - `viral_cnn.interpretability.motifs.scan_motif_occurrences` excludes positions with padding or ambiguous bases encoded as null vectors or negative indices (AC 5)
Proof: `.venv/bin/python3 -m unittest discover -s tests -k test_scan_motif_occurrences_padding_exclusion`

### S2: Historical Matched-Pair Perturbation & FDR Validation · 2 files · 24 KB · ~6k

**C5** - `viral_cnn.interpretability.motifs.perturb_sequence_mask` substitutes target motif positions with different nucleotides using random offsets $\in \{1, 2, 3\}$ (AC 6)
Proof: `.venv/bin/python3 -m unittest discover -s tests -k test_perturb_sequence_mask_different_nucleotides`

**C6** - `viral_cnn.interpretability.motifs.sample_matched_control_region` with `control_policy="uniform_random"` samples non-overlapping 7-bp valid windows from the same sequence (AC 8)
Proof: `.venv/bin/python3 -m unittest discover -s tests -k test_sample_matched_control_uniform_random_non_overlapping`

**C7** - `viral_cnn.interpretability.motifs.evaluate_motif_perturbation` computes paired probability drops ($\Delta p_{\text{motif}}$, $\Delta p_{\text{random}}$, $\Delta p_{\text{specific}}$) across $N_{\text{repeats}} = 5$ repetitions with fixed seed 42 (AC 7, AC 9)
Proof: `.venv/bin/python3 -m unittest discover -s tests -k test_evaluate_motif_perturbation_paired_repeats`

**C8** - `viral_cnn.interpretability.motifs.summarize_motif_control` performs paired one-sided Wilcoxon signed-rank tests and Benjamini-Hochberg FDR adjustment at $\alpha = 0.05$ on motifs with $N_{\text{sequences}} \ge 5$ (AC 10, AC 11, AC 12)
Proof: `.venv/bin/python3 -m unittest discover -s tests -k test_summarize_motif_control_wilcoxon_and_fdr`

### S3: Compositional & Homopolymer Sensitivity Controls · 2 files · 25 KB · ~6k

**C9** - `viral_cnn.interpretability.motifs.sample_matched_control_region` with `control_policy="gc_matched"` samples non-overlapping control windows with identical GC count ($\Delta\text{GC}=0$, fallback $\le 1$) (AC 13, AC 14)
Proof: `.venv/bin/python3 -m unittest discover -s tests -k test_sample_matched_control_gc_matched`

**C10** - `viral_cnn.interpretability.motifs.sample_matched_control_region` with `control_policy="complexity_matched"` samples non-overlapping control windows matching maximum homopolymer run length (AC 15)
Proof: `.venv/bin/python3 -m unittest discover -s tests -k test_sample_matched_control_complexity_matched`

**C11** - `viral_cnn.interpretability.motifs.evaluate_motif_perturbation` under `control_policy="gc_matched"` computes GC-controlled specific drops ($\Delta p_{\text{specific\_GC}}$) and Wilcoxon FDR statistics (AC 16)
Proof: `.venv/bin/python3 -m unittest discover -s tests -k test_evaluate_motif_perturbation_gc_matched`

**C12** - `viral_cnn.interpretability.motifs.audit_motif_composition_robustness` compares historical uniform drops vs GC-matched drops and flags compositionally confounded homopolymers (AC 17, AC 18)
Proof: `.venv/bin/python3 -m unittest discover -s tests -k test_audit_motif_composition_flags_confounded_homopolymers`

### S4: Co-occurrence Symmetry, Orientation Audit & Null Model Testing · 2 files · 24 KB · ~6k

**C13** - `viral_cnn.interpretability.motifs.compute_jaccard_cooccurrence` constructs exact symmetric pairwise Jaccard co-occurrence matrices across test sequence presence (AC 19)
Proof: `.venv/bin/python3 -m unittest discover -s tests -k test_compute_jaccard_cooccurrence_symmetric_matrix`

**C14** - `viral_cnn.interpretability.motifs.analyze_cooccurrence_pairs` audits orientation dependency between canonical $(u, v)$ and reversed $(v, u)$ top-200 edges, computing $|\Delta r|$ (AC 20)
Proof: `.venv/bin/python3 -m unittest discover -s tests -k test_analyze_cooccurrence_pairs_orientation_dependency`

**C15** - `viral_cnn.interpretability.motifs.analyze_cooccurrence_pairs` computes symmetric intraclass correlation (ICC) and duplicate bidirectional pair correlation over all $2K$ ordered pairs $[(x_i, y_i), (y_i, x_i)]$ (AC 21)
Proof: `.venv/bin/python3 -m unittest discover -s tests -k test_analyze_cooccurrence_pairs_symmetric_intraclass_correlation`

**C16** - `viral_cnn.interpretability.motifs.analyze_cooccurrence_pairs` evaluates statistical significance via a node-permutation null model (1,000 permutations) and computes empirical $p_{\text{perm}}$ (AC 22, AC 23)
Proof: `.venv/bin/python3 -m unittest discover -s tests -k test_analyze_cooccurrence_pairs_node_permutation_null`

### S5: Exploratory Triplet Composition vs. Biological Codon Usage Audit · 2 files · 24 KB · ~6k

**C17** - `viral_cnn.interpretability.motifs.compute_exploratory_triplet_rscu` computes triplet RSCU across `frame0`, `forward_3frames`, and `six_frames` modes (AC 24, AC 25)
Proof: `.venv/bin/python3 -m unittest discover -s tests -k test_compute_exploratory_triplet_rscu_multiframe`

**C18** - `viral_cnn.interpretability.motifs.compute_exploratory_triplet_rscu` computes 0th-order multinomial base-composition null expectations $P(b_1 b_2 b_3) = P(b_1)P(b_2)P(b_3)$ (AC 26)
Proof: `.venv/bin/python3 -m unittest discover -s tests -k test_compute_exploratory_triplet_multinomial_null`

**C19** - `viral_cnn.interpretability.motifs.audit_translation_boundary_signals` tests ATG start and TAA/TAG/TGA stop frequencies in motifs against mononucleotide composition expectations (AC 27)
Proof: `.venv/bin/python3 -m unittest discover -s tests -k test_audit_translation_boundary_signals_against_null`

**C20** - `viral_cnn.interpretability.motifs.compute_exploratory_triplet_rscu` designates all metric outputs as `ExploratoryTripletComposition` and issues warnings when reading frame is unverified (AC 28)
Proof: `.venv/bin/python3 -m unittest discover -s tests -k test_compute_exploratory_triplet_demarcates_unverified_frames`

### S6: CLI Entrypoint, Multi-Panel Figures & Comprehensive Scientific Report · 3 files · 35 KB · ~9k

**C21** - CLI `scripts/run_motif_analysis.py` executes motif analysis pipeline and outputs `motif_summary.parquet`, `cooccurrence_synergy.json`, and `run_manifest.json` with exit code 0 (AC 29)
Proof: `.venv/bin/python3 -m unittest discover -s tests -k test_run_motif_analysis_cli_exit_zero_and_artifacts`

**C22** - CLI `scripts/run_motif_analysis.py` exits with code 2 on invalid arguments, missing input files, or unsupported control policies (AC 29)
Proof: `.venv/bin/python3 -m unittest discover -s tests -k test_run_motif_analysis_cli_invalid_args_exit_2`

**C23** - `viral_cnn.interpretability.motifs.generate_motif_audit_figures` renders multi-panel figures attributing each panel to verified model checkpoints (AC 30)
Proof: `.venv/bin/python3 -m unittest discover -s tests -k test_generate_motif_audit_figures_checkpoint_attribution`

**C24** - `docs/motifs_and_triplet_composition_audit.md` documents comparison of historical tables/figures vs GC/symmetry controls and text revision recommendations (AC 31, AC 32)
Proof: `.venv/bin/python3 -m unittest discover -s tests -k test_motif_audit_doc_documents_controls_and_recommendations`

**C25** - Motif analysis pipeline executes deterministically on synthetic fixtures without GPU hardware or external checkpoint downloads (AC 1, AC 6, AC 13)
Proof: `.venv/bin/python3 -m unittest discover -s tests -k test_motif_pipeline_synthetic_deterministic_execution`

## Coverage

| Set (size) | Member -> proof | Unproven |
| --- | --- | --- |
| Consensus extraction rules evaluated (2) | `max_positive` C1 · `max_abs` C2 | - |
| Paired control sampling policies audited (3) | `uniform_random` C6 · `gc_matched` C9 · `complexity_matched` C10 | - |
| Triplet frame analysis modes audited (3) | `frame0` C17 · `forward_3frames` C17 · `six_frames` C17 | - |
| Co-occurrence correlation metrics evaluated (3) | `canonical_upper_triangle` C14 · `reversed_order` C14 · `symmetric_intraclass` C15 | - |
| Translation boundary signal codons evaluated (4) | `start_atg` C19 · `stop_taa` C19 · `stop_tag` C19 · `stop_tga` C19 | - |
| CLI execution status codes (2) | `0` C21 · `2` C22 | - |

- Claims naming a status code or response shape: C21, C22 - each has a proof crossing the process boundary
- No other check claims more than the single case its proof exercises

## Swept

- validation: C22
- failure modes: C4
- idempotency: C25
- authorization: n/a - no authorization boundary on local research scripts
- concurrency: n/a - batch processing without shared concurrent write states
- data lifecycle: n/a - no persisted database lifecycle or TTL
- dependency failure: C4
- state transitions: C12
- observability: C21

## Handoff

S1-S6: 5 files, 90.5 KB (~23k tokens), all in interpretability and CLI evaluation, under the 150k budget - one builder

Mechanism: one builder (compaction accepted)

- **Boundary:** C1-C25 closed
- **Settled mid-build:** pure Python/NumPy implementations for Wilcoxon signed-rank test and Benjamini-Hochberg FDR without external scipy/statsmodels dependencies; pure Python standalone PNG generation for audit figures when matplotlib is absent.
- **Abandoned:** none

