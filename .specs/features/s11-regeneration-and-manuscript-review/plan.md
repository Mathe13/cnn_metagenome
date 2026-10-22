# S11: Table/Figure Regeneration and Final Manuscript Evidence Review

## Problem

The scientific manuscript for BioData Mining reports performance, robustness, latent representations, sequence motifs, and sliding-window contig inferences across 13,683 viral species, but the published draft and associated reporting exhibit critical discrepancies, incomplete traceability, and unmoderated claims:

1. **Unlinked Numbers and Missing Provenance**: The 12 tables and 5 figures in the manuscript were compiled from historical exploratory notebooks without a centralized, automated pipeline mapping every reported statistic to an auditable CLI script, parameter configuration, input dataset, and model checkpoint.
2. **Precision, Denominator, and Rounding Inconsistencies**: Genus-level ConvNeXt metrics in Table 1 appear rounded to whole or half percentiles (85.00%, 92.50%, 84.00%), ConvNeXt_F clean accuracy (61.14%) was omitted from Table 1 despite appearing in Table 3, and the species-level pairwise comparison at $\epsilon=0.10$ in Table 4 omitted 5 ties (summing to 13,028 instead of all 13,033 evaluated species). Furthermore, single-window metrics inconsistently mixed denominators between total evaluated test reads ($N=1,706,270$) and valid non-padding subsets.
3. **Asset Glitches and Platform Sensitivity**: In Fig. 3, panel (e) (ConvNeXt noise-aware) erroneously duplicated the image from ResNet50 noise-aware. In addition, graphic filenames in historical figures used inconsistent capitalization (`ConvNext` vs. `ConvNeXt`), causing LaTeX compilation failures on case-sensitive Linux operating systems.
4. **Overclaimed Generalization and Unmoderated Titles**: The manuscript title claims that 1D-CNN representations "enable robust" classification, despite species accuracy deteriorating to 0.30%–17.38% at $\epsilon=0.15$. The abstract highlights 72.69% accuracy on long sequences without qualifying that the NCBI contigs share genome accessions with the training references (seen genomes) and without explaining why ConvNeXt architectures suffer catastrophic drops to 3.49%–20.38%. Moreover, the manuscript lacks structured BMC headings (Background, Methods, Results, Conclusions), omits comparison to foundational predecessor GeNet, leaves ICTV MSL41 and NCBI GenBank accessions unversioned, and embeds 56 revision notes (`[Nota v2: ...]`) that cannot be published.

When this ships, `scripts/generate_paper_tables.py` re-generates all 12 publication tables with strict denominators, exact rounding, and tie resolution; `scripts/generate_paper_figures.py` verifies and re-generates all 5 figures with correct ConvNeXt panels and Linux case-safe filenames; `paper/manuscript_review.tex` preserves an auditable, annotatable version with review macros; `paper/manuscript_submission.tex` provides a clean, publication-ready BMC manuscript with structured abstract, GeNet positioning, moderated title/claims, and clean LaTeX compilation (`scripts/compile_paper.py`); and `docs/manuscript_evidence_reconciliation_audit.md` provides an exhaustive traceability matrix linking every result in the paper to its reproducing script and evidence.

## Flow

Reuses `src/viral_cnn/eval/calibration.py` (exists), `src/viral_cnn/eval/latent.py` (exists), `src/viral_cnn/interpretability/motifs.py` (exists), `src/viral_cnn/baselines/paired.py` (exists), and `src/viral_cnn/inference/voter.py` (exists) for metric evaluation and data structures.

1. prediction manifests and evaluation parquets -> `viral_cnn.reporting.tables:generate_all_tables` (new, door 1) - formats Tables 1 through 12 in Markdown, CSV, and LaTeX with strict denominators and rounding
2. evaluation artifacts and latent/motif outputs -> `viral_cnn.reporting.figures:generate_all_figures` (new, door 2) - produces Figs 1 through 5 with corrected panels and case-safe filenames
3. manuscript templates and bibtex -> `viral_cnn.reporting.manuscript:build_manuscript_variants` (new, door 3) - generates `manuscript_review.tex` and `manuscript_submission.tex` with updated metadata and moderated claims
4. latex source and figure assets -> `scripts/compile_paper.py` (new, door 4) - invokes pdflatex/latexmk on Linux and validates compilation with zero errors
5. out: `docs/manuscript_evidence_reconciliation_audit.md`, updated `paper/*.tex`, `paper/references.bib`, and regenerated publication tables and figures

## Impact

| Front | What changes |
| --- | --- |
| domain | new term: `EvidenceReconciliationMatrix` - comprehensive mapping table linking every reported number, table, and figure to its generating CLI command, config, and data checkpoint, lives in `docs/manuscript_evidence_reconciliation_audit.md` |
| domain | new term: `SubmissionManuscript` - submission-ready LaTeX document formatted according to BioData Mining / BMC requirements with structured abstract, updated citations, and all revision notes removed, lives in `paper/manuscript_submission.tex` |
| domain | new term: `ReviewManuscript` - dual-track LaTeX source preserving all co-author revision notes (`[Nota v2: ...]`) and revision macros for ongoing peer review, lives in `paper/manuscript_review.tex` |
| domain | existing term: `RobustTaxonomicClassification` historically asserted absolute noise resilience; now redesignated as `CorruptionTradeoff` documenting the empirical trade-off between clean-sequence accuracy and controlled resampling degradation |
| stored data | nothing to migrate: historical notebooks, parquets, and PDF remain intact; new generated tables land in `results_tables/`, figures in `paper/figures/`, and audit reports in `docs/` |

## Relations

None - no stored-data shape change

## Surface

None - nothing consumed outside

## Landing

| One-way door | Literal shape | Alternative rejected |
| --- | --- | --- |
| Table output format standardization | `generate_all_tables(output_dir: Path, formats: tuple[str, ...] = ("csv", "md", "tex"))` with explicit denominator column `denominator_n` and rounded `float_format="%.2f"` | generating only ad-hoc console printouts - rejected because submission requires LaTeX tables while audit requires machine-readable CSVs |
| Figure asset case-sensitivity standard | `paper/figures/fig_{1..5}_{subpanel}.png` with strictly lowercase or exact camelCase `ConvNeXt` matching LaTeX `\includegraphics` | arbitrary mixed case (`ConvNext` vs `ConvNeXt`) - rejected because Linux filesystems fail compilation on case mismatch |
| Dual-track manuscript source separation | `paper/manuscript_review.tex` (with `\newcommand{\notavdois}[1]{\textcolor{red}{[#1]}}` and active notes) alongside `paper/manuscript_submission.tex` (with clean text and zero notes) | single destructive overwrite of original manuscript - rejected because co-authors require traceable revision notes while publishers reject submission files with editorial notes |
| Title and abstract framing standard | Title: *"Base-level one-hot 1D convolutional networks for high-cardinality viral taxonomic classification: accuracy, robustness, and learned representations"*; Abstract: BMC structured sections `Background`, `Methods`, `Results`, `Conclusions` | retaining *"enable robust"* in title and unstructured abstract - rejected because scientific invariants in AGENTS.md forbid claiming unverified robustness and BMC guidelines mandate structured abstracts |
| Long sequence qualification standard | Text explicitly clarifies: (1) 72.69% ResNet50 accuracy reflects NCBI reference contigs sharing genome accession with training data (seen genomes), and (2) ConvNeXt degradation to 3.49%–20.38% stems from read length bias, logit pulling, and calibration loss | reporting 72.69% as generalization to unseen viral genomes - rejected because S06 and S02/S03 audit confirmed accession sharing with Kraken2 references |

- Nothing else in this change is hard to reverse

## Criteria

### S1: Publication Tables Regeneration Pipeline (P1)

Automated re-generation of Tables 1 through 12 from reproducible pipeline predictions, audit logs, and configuration manifests.

**Acceptance Criteria**

1. WHEN invoking the table generation pipeline THEN the system SHALL generate Tables 1 through 12 in CSV, Markdown, and LaTeX formats in the target output directory.
2. WHEN formatting Table 1 THEN the system SHALL report species-level and genus-level metrics (Accuracy, Top-5, Macro-F1) formatted to exactly 2 decimal places, including ConvNeXt_F clean performance (61.14%).
3. WHEN formatting Table 3 THEN the system SHALL report single-window (512 bp) accuracy across all five noise levels $\epsilon \in \{0, 0.02, 0.05, 0.10, 0.15\}$ for all five species-level models and genus models with total evaluated test instances $N=1,706,270$ explicitly documented as the denominator.
4. WHEN formatting Table 4 THEN the system SHALL partition species-level pairwise performance at $\epsilon=0.10$ into ConvNeXt superior, ResNet50 superior, and exactly 5 ties, accounting for all 13,033 evaluated species ($11,826 + 1,202 + 5 = 13,033$).
5. WHEN categorizing species-level performance in Table 4 and text THEN the system SHALL report F1 score bins using continuous intervals $[0.90, 1.00]$, $[0.80, 0.90)$, $[0.50, 0.80)$, and $[0.00, 0.50)$ without mathematical gaps.
6. WHEN reporting validated first-layer motifs in Table 8 and Table 10 THEN the system SHALL report 7-mer consensus sequences and ATG / stop-associated triplets (TAA/TAG/TGA) with explicit notification that values reflect exploratory composition rather than translation initiation/termination.
7. WHEN computing inter-architecture motif correlation pairs in Table 11 THEN the system SHALL include exact two-sided $p$-values and Benjamini-Hochberg FDR adjustments.
8. WHEN formatting Table 12 THEN the system SHALL report sliding-window Voter consensus performance for long sequences, explicitly documenting $N_{\text{sequences}}$, mean length, stride (256 bp), and handling of contigs $< 512$ bp.

**Independent test:** unit tests in `tests/test_reporting_tables.py` validating dataframes, column sums, decimal formatting, and mathematical consistency.

### S2: Publication Figures Pipeline & Case-Sensitive Asset Audit (P1)

Automated generation and verification of Figures 1 through 5 with correct subpanels and Linux case-safe file naming.

**Acceptance Criteria**

9. WHEN generating publication figures THEN the system SHALL generate vector (PDF) or high-resolution raster (PNG $\ge 300$ DPI) assets for Figures 1 through 5.
10. WHEN producing Fig. 3 THEN the system SHALL ensure panel (e) displays the verified ConvNeXt noise-aware perturbation distribution and does NOT duplicate the ResNet50 noise-aware panel.
11. The system SHALL verify that every image filename referenced in LaTeX `\includegraphics` matches the filesystem file path exactly with respect to character case (e.g., `ConvNeXt` vs `ConvNext`).
12. IF any referenced figure file does not exist on disk or has a case mismatch THEN the verification tool SHALL report a fatal error and exit non-zero.

**Independent test:** unit tests in `tests/test_reporting_figures.py` asserting file existence, case matches, and image dimensions/validity.

### S3: Dual-Track Manuscript Source & Clean Compilation (P1)

Generation and maintenance of separate review and submission LaTeX sources with updated citations and automated compilation checks.

**Acceptance Criteria**

13. The system SHALL maintain `paper/manuscript_review.tex` containing all 56 original review notes (`[Nota v2: ...]`) formatted via review macros (`\notavdois{...}`) alongside tracked responses.
14. The system SHALL generate `paper/manuscript_submission.tex` with all editorial review notes, macros, and internal annotations completely excised.
15. The system SHALL provide `paper/references.bib` including complete bibliographic entries for: GeNet (Rojas-Carulla et al., 2019), AdamW (Loshchilov & Hutter, 2019), Warmup & fine-tuning (Gotmare et al., 2019), UMAP (McInnes et al., 2018), and RSCU (Sharp et al., 1986).
16. WHEN compiling `paper/manuscript_submission.tex` using `pdflatex` or `latexmk` THEN the system SHALL execute with exit code 0, producing `paper/manuscript_submission.pdf` without unresolved fatal cross-references or missing graphics.

**Independent test:** command `python3 scripts/compile_paper.py --mode submission` executing pdflatex and exiting code 0.

### S4: Scientific Coherence, Traceability & Evidence Reconciliation Audit (P1)

Comprehensive audit document reconciling every claim, table, and figure in the paper with the reproducible experimental evidence from S01 through S10.

**Acceptance Criteria**

17. The system SHALL generate `docs/manuscript_evidence_reconciliation_audit.md` containing the complete mapping table: `Manuscript Item -> Generating Script -> Config -> Input Data/Checkpoint -> Scientific Status`.
18. WHEN setting the manuscript title in the submission document and audit report THEN the system SHALL replace the historical title with the exact string "Base-level one-hot 1D convolutional networks for high-cardinality viral taxonomic classification: accuracy, corruption trade-offs, and learned sequence representations".
19. WHEN formatting the submission manuscript Abstract THEN the system SHALL partition the text into four explicit BMC headings: `Background`, `Methods`, `Results`, and `Conclusions`.
20. The system SHALL explicitly document contributions beyond GeNet, DeepMicrobes, CHEER, and MetageNN in the Background and Discussion sections: high-cardinality closed-set viral taxonomy (13,683 species), systematic corruption trade-offs with corrected effective substitution rate $3\epsilon/4$, formal calibration (ECE/Brier), paired GC/complexity motif controls, and long-sequence voter mechanics.
21. The system SHALL declare the closed-set nature of the fragment split (80/10/10), acknowledge that long-sequence NCBI contigs share accessions with Kraken2 training references, and explain the ConvNeXt long-read drop as length-bias and logit-pulling rather than true lack of taxonomic knowledge.
22. WHEN declaring availability of data and materials THEN the system SHALL state that bulky checkpoints and Parquets reside on remote research storage (AD-006) while synthetic deterministic verification is available locally.

**Independent test:** verification script inspecting `docs/manuscript_evidence_reconciliation_audit.md` and `paper/manuscript_submission.tex` for required sections, headings, citations, and invariant compliance.

## Out of scope

| Excluded | Why |
| --- | --- |
| Full GPU re-training of 13,683 species models | Full training exceeds local sandbox compute budget and was completed historically on JupyterHub cluster (AD-006) |
| Changing underlying model architectures or class order | AGENTS.md invariant forbids altering model definitions or class index mapping |
| Generating artificial manuscript acceptance by journal | Editorial submission and reviewer response process is conducted out-of-band |

## Assumptions

| Assumption | Chosen default | Rationale | Confirmed? |
| --- | --- | --- | --- |
| Target journal format | BioData Mining (BMC Springer Nature) format with structured abstract and Background, Methods, Results, Discussion, Conclusions, Declarations | Follows published PDF template and journal submission instructions | y |
| Submitter affiliation details | Include author affiliations matching PDF and add optional research group in acknowledgments | Conforms to note 3 and institutional standards | y |

**Open questions:** none - all resolved or logged above.

## Observable

| Surface | Decision | Landing |
| --- | --- | --- |
| CLI `scripts/generate_paper_tables.py` | output format and verbosity | AC 1, AC 2, AC 3, AC 4, AC 5, AC 6, AC 7, AC 8 |
| CLI `scripts/generate_paper_figures.py` | resolution and case-sensitive paths | AC 9, AC 10, AC 11, AC 12 |
| CLI `scripts/compile_paper.py` | exit codes and compilation error reporting | AC 16 |
| Document `paper/manuscript_submission.tex` | structure, tone, and excised notes | AC 14, AC 15, AC 18, AC 19, AC 20, AC 21, AC 22 |
| Document `paper/manuscript_review.tex` | preserved review macros and notes | AC 13 |
| Document `docs/manuscript_evidence_reconciliation_audit.md` | full traceability matrix | AC 17 |

## Sources

- `ROADMAP_SPECS_ARTIGO.md` - defines S11 objectives, scope, dependencies, and acceptance criteria
- `paper/BioData_Minning___Matheus.pdf` - compiled manuscript with 56 embedded review notes (`[Nota v2: ...]`)
- `AGENTS.md` - permanent scientific invariants, naming conventions, and evidence reconciliation rules
