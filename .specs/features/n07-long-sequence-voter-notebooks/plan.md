# n07-long-sequence-voter-notebooks

## Problem

In the original manuscript and historical notebooks in `long/` (`download_from_ncbi.ipynb`, `prepare database.ipynb`, `cnn-long-voter.ipynb`, `cnn-long-voter-compare.ipynb`, and `long_loss_function.ipynb`), taxonomic classification was evaluated on variable-length genomic contigs via sliding-window consensus voting. However, several critical methodological and reproducibility limitations affected these experiments:

1. **Coupling to monolithic legacy scripts**: the original inference and preparation pipelines were implemented inside ad-hoc notebooks containing unmodularized loops, inline data loading, hardcoded paths, and ad-hoc remapping tables (`mapping_species_final.csv`) with only 13,036 species instead of the canonical 13,683 classes (AD-001, AD-004).
2. **Conflating training reference overlap with generalization to unseen genomes**: viral sequences downloaded from NCBI Nucleotide for long-sequence evaluation ($N=4,360$) shared significant sequence identity with the reference genomes in the Kraken2 training database (`quantify_training_overlap`). Consequently, the reported 72.69% accuracy for standard ResNet50 reflects performance on sequences derived from seen reference genomes rather than generalization to novel, unseen viral species or genomes (Notas 11–15, 44–47; AD-010).
3. **Sampling bias toward giant viruses**: the simulation procedure (`calculate_n_reads` in `prepare database.ipynb`) generated 500 reads for large viral genomes (>100 kb) versus only 50 reads for small viral genomes (<10 kb), introducing massive class-imbalance bias toward giant viruses (*Pandoravirus*, *Chlorella virus*, *Mollivirus*, *Orpheovirus*, and *Bracoviriform*) in error counts and accuracy aggregations (AD-010).
4. **Uninvestigated ConvNeXt collapse**: ConvNeXt performance on long sequences suffered a catastrophic collapse down to 3.49%–20.38% (Table 12), with genus-transferred `ConvNeXt_T` dropping to 3.49%. The historical analysis lacked error stratification by contig length, ambiguous IUPAC bases, pre-softmax logit pulling, and calibration degradation to explain this drop (AD-010).
5. **Silent truncation and lack of short contig handling**: the historical window slicing (`text_to_tensor`) silently discarded contigs shorter than 512 bp and dropped up to 255 bp of trailing end-sequence without documentation.
6. **Absence of clean reproduction notebooks**: there are currently no reproduction notebooks in `notebooks/reproduction/long/` executing the long-sequence evaluation chain with scientific rigor.

When this feature ships, researchers will have four clean, modular reproduction notebooks in `notebooks/reproduction/long/` (`01_ncbi_download_and_prepare_database.ipynb`, `02_cnn_long_voter_inference.ipynb`, `03_cnn_long_voter_compare.ipynb`, and `04_long_motifs_and_gc_analysis.ipynb`) that consume `viral_cnn.inference.*`, `viral_cnn.baselines.long_eval`, and `viral_cnn.interpretability.motifs`, run end-to-end on remote JupyterHub or locally under `SMOKE_TEST=1`, audit training reference overlap, verify window logit equivalence, reproduce Table 12, and formally document the ConvNeXt collapse mechanisms.

## Flow

This feature reuses sliding-window inference and consensus aggregation from `viral_cnn.inference.voter`, long-sequence audit and overlap engines from `viral_cnn.inference.audit_long`, baseline integration and stratification from `viral_cnn.baselines.long_eval`, motif extraction from `viral_cnn.interpretability.motifs`, and runtime context orchestration from `viral_cnn.notebook_runtime`.

1. Raw NCBI accessions and preparation metadata -> `notebooks/reproduction/long/01_ncbi_download_and_prepare_database.ipynb` (new, door 1) - consolidates download and preparation, audits genome overlap with training references via `viral_cnn.inference.audit_long:quantify_training_overlap`, and records the read sampling distribution.
2. Genomic contigs and model checkpoints -> `viral_cnn.inference.voter:LongSequenceVoter` (exists) - executes sliding-window inference (window 512 bp, stride 256 bp) with pre-softmax mean logits and hard majority voting, supporting both `handle_short='skip'` and `handle_short='pad'`.
3. Slicing and logit equivalence proof -> `viral_cnn.inference.audit_long:verify_window_logit_equivalence` (exists) - verifies exact floating-point logit equivalence between 512 bp short-fragment inference and long-sequence voter windows.
4. Voter inference orchestration -> `notebooks/reproduction/long/02_cnn_long_voter_inference.ipynb` (new, door 2) - runs voter inference across clean and noise levels $\epsilon \in \{0.0, 0.02, 0.05, 0.10, 0.15\}$, exporting predictions to Parquet under `results_reproduction/long_voter/`.
5. Error stratification and comparative diagnosis -> `viral_cnn.inference.audit_long:stratify_long_errors` and `viral_cnn.baselines.long_eval:evaluate_long_baselines` (exists) - stratifies errors across length bins, IUPAC ambiguity rates, and reference genome presence (`seen_genome` vs `unseen_genome`).
6. Comparison notebook orchestration -> `notebooks/reproduction/long/03_cnn_long_voter_compare.ipynb` (new, door 3) - reproduces Table 12, evaluates baselines, and generates causal diagnostics of the ConvNeXt collapse.
7. Sliding-window motif analysis -> `notebooks/reproduction/long/04_long_motifs_and_gc_analysis.ipynb` (new, door 4) - extracts 7-mer motif activations and GC profiles across long contig windows using `viral_cnn.interpretability.motifs`.

## Impact

| Front | What changes |
| --- | --- |
| domain | existing modules: `viral_cnn.inference.voter`, `viral_cnn.inference.audit_long`, `viral_cnn.baselines.long_eval`, `viral_cnn.interpretability.motifs` - directly consumed by reproduction notebooks in `notebooks/reproduction/long/` |
| stored data | nothing to migrate - remote NCBI parquet files and checkpoints remain untouched; new voter predictions, stratified metrics, and manifests are written to `results_reproduction/long_voter/` |

## Relations

None - no stored-data shape change

## Surface

None - nothing consumed outside

## Landing

| One-way door | Literal shape | Alternative rejected |
| --- | --- | --- |
| NCBI preparation and overlap reproduction notebook | `notebooks/reproduction/long/01_ncbi_download_and_prepare_database.ipynb` | Modifying historical notebooks `long/download_from_ncbi.ipynb` and `long/prepare database.ipynb`; rejected because historical notebooks must remain immutable evidence per AD-004. |
| Long sequence voter inference reproduction notebook | `notebooks/reproduction/long/02_cnn_long_voter_inference.ipynb` | Modifying historical notebook `long/cnn-long-voter.ipynb`; rejected because historical notebooks must remain immutable evidence per AD-004. |
| Long sequence comparison and error stratification reproduction notebook | `notebooks/reproduction/long/03_cnn_long_voter_compare.ipynb` | Modifying historical notebook `long/cnn-long-voter-compare.ipynb`; rejected because historical notebooks must remain immutable evidence per AD-004. |
| Long contig motifs and GC analysis reproduction notebook | `notebooks/reproduction/long/04_long_motifs_and_gc_analysis.ipynb` | Modifying historical notebook `long/long_loss_function.ipynb`; rejected because historical notebooks must remain immutable evidence per AD-004. |

- Nothing else in this change is hard to reverse

## Criteria

Grouped by slice - one observable outcome each, never a layer.

### S1: NCBI Download, Database Preparation & Overlap Audit Notebook (`01_ncbi_download_and_prepare_database.ipynb`) (P1)

Creates `notebooks/reproduction/long/01_ncbi_download_and_prepare_database.ipynb` reproducing sequence preparation, quantifying reference genome overlap with training data, and documenting sampling bias.

**Acceptance Criteria**

1. WHEN `01_ncbi_download_and_prepare_database.ipynb` executes under `SMOKE_TEST=1` THEN the system SHALL complete execution without exceptions, quantify training reference overlap via `viral_cnn.inference.audit_long:quantify_training_overlap`, and export `long_database_preparation_audit.csv` to `results_reproduction/long_voter/`.
2. The system SHALL ensure `01_ncbi_download_and_prepare_database.ipynb` imports all audit and preparation logic exclusively from `viral_cnn.*` and contains zero imports of legacy modules.
3. WHEN sequence preparation is audited in `01_ncbi_download_and_prepare_database.ipynb` THEN the system SHALL document the read count disparity between giant viral genomes (500 reads) and small viral genomes (50 reads).
4. The system SHALL preserve historical notebooks `long/download_from_ncbi.ipynb` and `long/prepare database.ipynb` strictly unmodified.

**Independent test:** Run execution of `01_ncbi_download_and_prepare_database.ipynb` under `SMOKE_TEST=1` and verify generation of `long_database_preparation_audit.csv`.

### S2: CNN Long Sequence Voter Inference & Noise Robustness Notebook (`02_cnn_long_voter_inference.ipynb`) (P1)

Creates `notebooks/reproduction/long/02_cnn_long_voter_inference.ipynb` running sliding-window inference with `LongSequenceVoter`, proving window logit equivalence, and comparing short contig handling.

**Acceptance Criteria**

5. WHEN `02_cnn_long_voter_inference.ipynb` executes under `SMOKE_TEST=1` THEN the system SHALL complete execution without exceptions, verify exact window logit equivalence via `verify_window_logit_equivalence`, and export voter predictions to Parquet under `results_reproduction/long_voter/`.
6. The system SHALL ensure `02_cnn_long_voter_inference.ipynb` imports all inference logic exclusively from `viral_cnn.inference.voter` and contains zero imports of legacy modules.
7. WHEN sliding-window consensus is computed in `02_cnn_long_voter_inference.ipynb` THEN the system SHALL evaluate both pre-softmax mean logits (`pred_mean_logits`) and hard majority voting (`pred_majority_vote`) across clean and noise levels $\epsilon \in \{0.0, 0.02, 0.05, 0.10, 0.15\}$.
8. WHEN contigs shorter than 512 bp are evaluated in `02_cnn_long_voter_inference.ipynb` THEN the system SHALL compare historical skip mode (`handle_short='skip'`) with standardized padded mode (`handle_short='pad'`).
9. The system SHALL preserve historical notebook `long/cnn-long-voter.ipynb` strictly unmodified.

**Independent test:** Run execution of `02_cnn_long_voter_inference.ipynb` under `SMOKE_TEST=1` and verify generation of `predictions_long_voter_*.parquet` and exact logit equivalence check.

### S3: Long Sequence Comparison, Error Stratification & ConvNeXt Collapse Diagnosis Notebook (`03_cnn_long_voter_compare.ipynb`) (P2)

Creates `notebooks/reproduction/long/03_cnn_long_voter_compare.ipynb` reproducing Table 12, stratifying errors by contig length, ambiguity, and reference presence, and diagnosing the ConvNeXt collapse.

**Acceptance Criteria**

10. WHEN `03_cnn_long_voter_compare.ipynb` executes under `SMOKE_TEST=1` THEN the system SHALL complete execution without exceptions, reproduce Table 12 metrics, and export `table_12_long_voter_comparison.csv` and `stratified_long_errors.csv` to `results_reproduction/long_voter/`.
11. The system SHALL ensure `03_cnn_long_voter_compare.ipynb` imports all stratification and comparison logic exclusively from `viral_cnn.inference.audit_long` and `viral_cnn.baselines.long_eval`.
12. WHEN error stratification is computed in `03_cnn_long_voter_compare.ipynb` THEN the system SHALL partition performance across length bins ($[512, 2000)$, $[2000, 5000)$, $\ge 5000$ bp), ambiguous character fractions, and reference presence (`seen_genome` vs `unseen_genome`).
13. WHEN the ConvNeXt collapse is analyzed in `03_cnn_long_voter_compare.ipynb` THEN the system SHALL document the three causal mechanisms: giant virus sampling bias, pre-softmax logit pulling across windows, and loss of calibration.
14. The system SHALL preserve historical notebook `long/cnn-long-voter-compare.ipynb` strictly unmodified.

**Independent test:** Run execution of `03_cnn_long_voter_compare.ipynb` under `SMOKE_TEST=1` and verify generation of `table_12_long_voter_comparison.csv`.

### S4: Long Contig Motifs and GC Profile Analysis Notebook (`04_long_motifs_and_gc_analysis.ipynb`) (P2)

Creates `notebooks/reproduction/long/04_long_motifs_and_gc_analysis.ipynb` evaluating sliding-window motif activations and GC profiles across long contigs.

**Acceptance Criteria**

15. WHEN `04_long_motifs_and_gc_analysis.ipynb` executes under `SMOKE_TEST=1` THEN the system SHALL complete execution without exceptions, compute sliding-window 7-mer motif activations, and export `long_contig_motif_summary.csv` and `figure_long_motifs_gc.png` to `results_reproduction/long_voter/`.
16. The system SHALL ensure `04_long_motifs_and_gc_analysis.ipynb` imports all motif analysis logic exclusively from `viral_cnn.interpretability.motifs` and contains zero legacy imports.
17. The system SHALL preserve historical notebook `long/long_loss_function.ipynb` strictly unmodified.

**Independent test:** Run execution of `04_long_motifs_and_gc_analysis.ipynb` under `SMOKE_TEST=1` and verify generation of `long_contig_motif_summary.csv` and `figure_long_motifs_gc.png`.

## Out of scope

| Excluded | Why |
| --- | --- |
| Downloading full gigabase NCBI fasta archives locally | Remote JupyterHub storage holds the full database; local environment uses pre-extracted partitions or synthetic test data per AD-005/AD-006 |
| Retraining CNN backbones on full genomes | Retraining is reserved for S12; feature N07 focuses strictly on reproducing long-sequence inference and audit |
| In-place modification of historical notebooks in `long/` | Historical notebooks are preserved as immutable evidence per AD-004 |

## Assumptions

| Assumption | Chosen default | Rationale | Confirmed? |
| --- | --- | --- | --- |
| Slicing window size | 512 base pairs | Matches first-layer model input shape and historical long voter window | y |
| Slicing stride | 256 base pairs | Matches 50% overlap historical parameterization in `cnn-long-voter.ipynb` | y |
| Contig length bins | [512, 2000), [2000, 5000), >= 5000 bp | Standard tripartite length partition in `stratify_long_errors` | y |
| Ambiguity bins | [0.0, 0.01), [0.01, 0.05), >= 0.05 | Standard IUPAC non-ACGT fraction bins in `stratify_long_errors` | y |
| Short contig default mode | pad | Standardized mode preserving contigs <512 bp with null vectors (AD-010) | y |

**Open questions:** none - all resolved or logged above.

## Observable

| Surface | Decision | Landing |
| --- | --- | --- |
| Reproduction notebook 01 | execution exit 0, outputs `long_database_preparation_audit.csv` | AC 1 |
| Reproduction notebook 02 | execution exit 0, outputs `predictions_long_voter_*.parquet` | AC 5 |
| Reproduction notebook 03 | execution exit 0, outputs `table_12_long_voter_comparison.csv` | AC 10 |
| Reproduction notebook 04 | execution exit 0, outputs `long_contig_motif_summary.csv`, figure | AC 15 |
| Voter manifest | schema with `methods`, `window_size`, `stride`, `overlap_metrics` | AC 5 |

## Sources

- Manuscript Section 3.5 and Table 12: Published sliding-window taxonomic consensus on variable-length genomic sequences.
- Historical notebooks in `long/`: `download_from_ncbi.ipynb`, `prepare database.ipynb`, `cnn-long-voter.ipynb`, `cnn-long-voter-compare.ipynb`, and `long_loss_function.ipynb`.
- Decision `AD-010`: Auditoria do Long Voter: desincorporação de generalização para genomas não vistos e causas da queda de ConvNeXt.
- Repository scientific invariant rule `AGENTS.md` and Roadmap `ROADMAP_REPRODUCAO_NOTEBOOKS.md` Section N07.

