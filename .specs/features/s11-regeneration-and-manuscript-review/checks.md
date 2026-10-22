# S11: Table/Figure Regeneration and Final Manuscript Evidence Review - Checks

Profile: light
Plan: `.specs/features/s11-regeneration-and-manuscript-review/plan.md`

## Checks

### S1: Publication Tables Regeneration Pipeline · 3 files · 43 KB · ~11k

**C1** - `viral_cnn.reporting.tables.generate_all_tables` generates Tables 1 through 12 in `csv`, `md`, and `tex` formats in the designated output directory (AC 1)
Proof: `.venv/bin/python3 -m unittest discover -s tests -k test_generate_all_tables_formats`

**C2** - Table 1 contains species-level and genus-level metrics (Accuracy, Top-5, Macro-F1) formatted to exactly 2 decimal places, including ConvNeXt_F clean performance (61.14%) (AC 2)
Proof: `.venv/bin/python3 -m unittest discover -s tests -k test_table_1_metrics_and_convnext_f`

**C3** - Table 3 reports single-window (512 bp) accuracy across all five noise levels $\epsilon \in \{0, 0.02, 0.05, 0.10, 0.15\}$ for all five models with total instances $N=1,706,270$ as explicit denominator (AC 3)
Proof: `.venv/bin/python3 -m unittest discover -s tests -k test_table_3_noise_degradation_denominator`

**C4** - Table 4 species-level pairwise comparison at $\epsilon=0.10$ reports 11,826 ConvNeXt superior, 1,202 ResNet50 superior, and exactly 5 ties, summing to 13,033 evaluated species (AC 4)
Proof: `.venv/bin/python3 -m unittest discover -s tests -k test_table_4_pairwise_ties_and_sum`

**C5** - Table 4 and text report F1 score bins using continuous intervals $[0.90, 1.00]$, $[0.80, 0.90)$, $[0.50, 0.80)$, and $[0.00, 0.50)$ without mathematical gaps (AC 5)
Proof: `.venv/bin/python3 -m unittest discover -s tests -k test_table_4_f1_continuous_intervals`

**C6** - Table 8 and Table 10 report statistically validated 7-mer motifs and ATG / stop triplets (TAA/TAG/TGA) with exploratory composition notification (AC 6)
Proof: `.venv/bin/python3 -m unittest discover -s tests -k test_table_8_and_10_motifs_triplets`

**C7** - Table 11 reports exact two-sided $p$-values and Benjamini-Hochberg FDR adjustments for all inter-architecture motif correlation pairs (AC 7)
Proof: `.venv/bin/python3 -m unittest discover -s tests -k test_table_11_correlation_p_values_fdr`

**C8** - Table 12 reports long-sequence sliding-window Voter consensus performance with $N_{\text{sequences}}$, mean length, stride (256 bp), and handling of contigs $< 512$ bp (AC 8)
Proof: `.venv/bin/python3 -m unittest discover -s tests -k test_table_12_voter_consensus`

### S2: Publication Figures Pipeline & Case-Sensitive Asset Audit · 3 files · 38 KB · ~10k

**C9** - `viral_cnn.reporting.figures.generate_all_figures` generates publication-grade assets for Figures 1 through 5 (AC 9)
Proof: `.venv/bin/python3 -m unittest discover -s tests -k test_generate_all_figures_assets`

**C10** - In Fig. 3, panel (e) displays the verified ConvNeXt noise-aware perturbation distribution without duplicating the ResNet50 noise-aware panel (AC 10)
Proof: `.venv/bin/python3 -m unittest discover -s tests -k test_fig_3_panel_e_convnext_noise_distinct`

**C11** - `viral_cnn.reporting.figures.verify_latex_figure_paths` asserts that all image paths in LaTeX files exist on disk with exact case-matching (AC 11)
Proof: `.venv/bin/python3 -m unittest discover -s tests -k test_verify_figure_case_sensitivity_matches`

**C12** - `viral_cnn.reporting.figures.verify_latex_figure_paths` reports fatal error and non-zero exit when a referenced figure is missing or has a case mismatch (AC 12)
Proof: `.venv/bin/python3 -m unittest discover -s tests -k test_verify_figure_case_sensitivity_failure`

### S3: Dual-Track Manuscript Source & Clean Compilation · 4 files · 178 KB · ~45k

**C13** - `paper/manuscript_review.tex` preserves all 56 review notes (`[Nota v2: ...]`) formatted via review macros (`\notavdois{...}`) (AC 13)
Proof: `.venv/bin/python3 -m unittest discover -s tests -k test_manuscript_review_contains_all_notes`

**C14** - `paper/manuscript_submission.tex` excises all review notes, internal review macros, and comments from the document body (AC 14)
Proof: `.venv/bin/python3 -m unittest discover -s tests -k test_manuscript_submission_clean_of_notes`

**C15** - `paper/references.bib` contains complete bibliographic entries for GeNet (Rojas-Carulla 2019), AdamW (Loshchilov & Hutter 2019), Warmup (Gotmare 2019), UMAP (McInnes 2018), and RSCU (Sharp 1986) (AC 15)
Proof: `.venv/bin/python3 -m unittest discover -s tests -k test_references_bib_required_citations`

**C16** - CLI `scripts/compile_paper.py --mode submission` compiles `paper/manuscript_submission.tex` with `pdflatex` or `latexmk`, producing `paper/manuscript_submission.pdf` with exit code 0 (AC 16)
Proof: `.venv/bin/python3 -m unittest discover -s tests -k test_compile_paper_submission_exit_zero`

### S4: Scientific Coherence, Traceability & Evidence Reconciliation Audit · 3 files · 40 KB · ~10k

**C17** - `docs/manuscript_evidence_reconciliation_audit.md` contains the full traceability matrix linking all 12 tables and 5 figures to generating script, config, and data/checkpoint (AC 17)
Proof: `.venv/bin/python3 -m unittest discover -s tests -k test_traceability_matrix_covers_all_tables_and_figures`

**C18** - The manuscript title in `paper/manuscript_submission.tex` and `docs/manuscript_evidence_reconciliation_audit.md` matches the exact moderated string "Base-level one-hot 1D convolutional networks for high-cardinality viral taxonomic classification: accuracy, corruption trade-offs, and learned sequence representations" (AC 18)
Proof: `.venv/bin/python3 -m unittest discover -s tests -k test_submission_title_moderated`

**C19** - The Abstract in `paper/manuscript_submission.tex` contains explicit structured section headings: `Background`, `Methods`, `Results`, and `Conclusions` (AC 19)
Proof: `.venv/bin/python3 -m unittest discover -s tests -k test_abstract_bmc_structured_sections`

**C20** - `paper/manuscript_submission.tex` and `docs/manuscript_evidence_reconciliation_audit.md` document GeNet comparison, 80/10/10 closed-set split, seen contig reference overlap, ConvNeXt long-sequence drop explanation, and remote research storage availability (AD-006) (AC 20, AC 21, AC 22)
Proof: `.venv/bin/python3 -m unittest discover -s tests -k test_scientific_evidence_demarcations`

## Coverage

| Set (size) | Member -> proof | Unproven |
| --- | --- | --- |
| Publication tables generated (12) | Table 1 C2 · Table 2 C1 · Table 3 C3 · Table 4 C4, C5 · Table 5 C1 · Table 6 C1 · Table 7 C1 · Table 8 C6 · Table 9 C1 · Table 10 C6 · Table 11 C7 · Table 12 C8 | - |
| Table output serialization formats (3) | `csv` C1 · `md` C1 · `tex` C1 | - |
| Noise levels evaluated in single-window tables (5) | $\epsilon=0$ C3 · $\epsilon=0.02$ C3 · $\epsilon=0.05$ C3 · $\epsilon=0.10$ C3 · $\epsilon=0.15$ C3 | - |
| Species performance pairwise categories at $\epsilon=0.10$ (3) | `convnext_superior` C4 · `resnet_superior` C4 · `ties` C4 | - |
| Continuous F1 score intervals (4) | $[0.90, 1.00]$ C5 · $[0.80, 0.90)$ C5 · $[0.50, 0.80)$ C5 · $[0.00, 0.50)$ C5 | - |
| Publication figures regenerated and verified (5) | Fig. 1 C9 · Fig. 2 C9 · Fig. 3 C9, C10 · Fig. 4 C9 · Fig. 5 C9 | - |
| Figure asset case-sensitivity outcomes (2) | `exact_match` C11 · `case_mismatch_failure` C12 | - |
| Manuscript track variants produced (2) | `manuscript_review.tex` C13 · `manuscript_submission.tex` C14 | - |
| Mandatory bibliographic additions verified (5) | `GeNet` C15 · `AdamW` C15 · `Warmup_Gotmare` C15 · `UMAP_McInnes` C15 · `RSCU_Sharp` C15 | - |
| BMC structured abstract headings (4) | `Background` C19 · `Methods` C19 · `Results` C19 · `Conclusions` C19 | - |

## Swept

Where each unwritten requirement landed. All nine, one line each, every time.

- validation and bounds: C5
- failure and partial failure: C12
- idempotency, retry, duplicates: C1
- authorization and rate limits: n/a - local reporting and manuscript compilation CLI
- concurrency and ordering: C16
- data lifecycle: n/a - reproducible reporting generated on demand from immutable evaluation outputs
- external-dependency failure: C16
- state transitions: C11, C12
- observability: C17

## Handoff

S1 = 11k; S2 = 10k; S3 = 45k; S4 = 10k -> total estimate = 76k tokens, well under the 150k budget - one builder
