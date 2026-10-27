# n05-short-motifs-and-triplets-notebooks

## Problem

In the original manuscript and early exploratory notebooks (`short/full_loss_function.ipynb` and `short/codon_usage.ipynb`), the first-layer $7 \times 4$ convolutional filters across 1D CNNs were evaluated as sequence motifs representing viral biological patterns, specifically codon usage bias (RSCU), translation initiation and termination signals, and cooperative motif synergy.
However, several methodological, statistical, and architectural discrepancies were identified in S10 and the manuscript notes:

1. **Coupling to monolithic notebooks and legacy scripts**: the original analyses were conducted inside exploratory notebooks containing unseeded operations, hardcoded local paths (`short/utils/*`), and ad-hoc inline plotting.
2. **Confounding motifs with extreme GC content and low-complexity homopolymers**: extracted consensus 7-mers across models are dominated by simple runs (e.g., `CCCCCCC`, `GGGGGGG`, `TCCCCCC`). The historical matched-control protocol perturbs a learned 7-mer against a random non-overlapping 7-bp window drawn uniformly from the same sequence. Because random regions have genome-average GC (~40–50%), perturbing a 100% GC homopolymer against a uniform random window inflates the reported sequence-specific drop ($\Delta p_{\text{specific}} = \Delta p_{\text{motif}} - \Delta p_{\text{random}}$) due to general GC sensitivity rather than sequence-specific recognition (Notas 26–28, 40).
3. **Figure 3 panel duplication and checkpoint misattribution**: in historical Figure 3, panel (e) erroneously duplicated panel (b) instead of evaluating the noise fine-tuned ConvNeXt (`convnext_noise_species`, Nota 39).
4. **Orientation artifact in top 200 co-occurrence pairs**: historical analysis extracted the top 200 edges by Jaccard score from an upper triangular matrix and arbitrarily mapped row indices to Motif A ($X$-axis) and column indices to Motif B ($Y$-axis), computing standard Pearson correlation on an arbitrarily oriented scatter plot without node-label permutation null testing (Nota 43).
5. **Overinterpretation of triplet composition as biological codon usage**: Relative Synonymous Codon Usage (RSCU) was computed across 512 bp test reads by fixing frame 0 without verified open reading frames (ORFs) or annotated coding sequences (CDS), conflating trinucleotide composition with translational selection (Notas 41–42).
6. **Absence of clean reproduction notebooks**: there are currently no clean, modular reproduction notebooks in `notebooks/reproduction/short/` executing this analysis with scientific invariants and producing Tables 8–11 and Figures 3–5.

When this feature ships, researchers will have two modular, fully reproducible reproduction notebooks (`07_first_layer_motifs_and_perturbation.ipynb` and `08_triplet_composition_and_cooccurrence.ipynb`) in `notebooks/reproduction/short/` that consume `viral_cnn.interpretability.motifs`, run end-to-end on remote JupyterHub or locally under `SMOKE_TEST=1`, enforce paired GC-matched controls, fix Figure 3 panel attribution, audit top-200 co-occurrence symmetry, and correctly demarcate exploratory triplet composition from biological codon usage.

## Flow

This feature reuses motif extraction, perturbation, and triplet analysis engines from `viral_cnn.interpretability.motifs`, model loading from `viral_cnn.models.registry`, spatial encoding from `viral_cnn.data.encoding`, and runtime context orchestration from `viral_cnn.notebook_runtime`.

1. Checkpoint loading -> `viral_cnn.models.registry:load_model` (exists) - loads species 1D CNN checkpoints (`resnet50_species`, `resnet50_noise`, `convnext_zero_species`, `convnext_transfer_species`, `convnext_noise_species`).
2. Motif consensus extraction -> `viral_cnn.interpretability.motifs:extract_first_layer_motifs` (exists) - derives consensus 7-mers from `conv1` (ResNet50) and `stem` (ConvNeXt) filters.
3. Paired perturbation -> `viral_cnn.interpretability.motifs:evaluate_motif_perturbation` and `summarize_motif_control` (exists) - computes probability drops under uniform random vs. GC-matched controls (`sample_matched_control_region`) with Wilcoxon signed-rank tests and Benjamini-Hochberg FDR correction.
4. Jaccard co-occurrence computation -> `viral_cnn.interpretability.motifs:compute_jaccard_cooccurrence` (exists) - scans sequences and computes pairwise motif co-occurrence matrices saved to Parquet.
5. First notebook orchestration -> `notebooks/reproduction/short/07_first_layer_motifs_and_perturbation.ipynb` (new, door 1) - executes S1, generating Table 8, Figure 3 panels (a-e), and Jaccard matrix parquet artifact.
6. Exploratory triplet analysis -> `viral_cnn.interpretability.motifs:compute_exploratory_triplet_rscu` (exists) - calculates triplet frequencies across frame 0, 3-frame, and 6-frame modes with FDR correction and exploratory warnings.
7. Translation boundary audit -> `viral_cnn.interpretability.motifs:audit_translation_boundary_signals` (exists) - audits start (`ATG`) and stop (`TAA`, `TAG`, `TGA`) triplet prevalence across learned motifs.
8. Symmetric pair co-occurrence audit -> `viral_cnn.interpretability.motifs:analyze_cooccurrence_pairs` (exists) - evaluates top-200 co-occurrence pairs with orientation symmetry testing, permutation null model, and inter-architecture correlations.
9. Second notebook orchestration -> `notebooks/reproduction/short/08_triplet_composition_and_cooccurrence.ipynb` (new, door 2) - executes S2, consuming Jaccard matrices from notebook 07 and generating Tables 9–11 and Figures 4–5.

## Impact

| Front | What changes |
| --- | --- |
| domain | existing module: `viral_cnn.interpretability.motifs` - directly consumed by reproduction notebooks 07 and 08 without code changes |
| stored data | nothing to migrate - parquet and model checkpoints remain untouched; new motif summaries, Jaccard matrices, and triplet audits are saved to `results_reproduction/motifs/` |

## Relations

None - no stored-data shape change

## Surface

None - nothing consumed outside

## Landing

| One-way door | Literal shape | Alternative rejected |
| --- | --- | --- |
| First-layer motifs and perturbation reproduction notebook | `notebooks/reproduction/short/07_first_layer_motifs_and_perturbation.ipynb` | Modifying historical notebook `short/full_loss_function.ipynb`; rejected because historical notebooks must remain immutable evidence per AD-004. |
| Triplet composition and co-occurrence reproduction notebook | `notebooks/reproduction/short/08_triplet_composition_and_cooccurrence.ipynb` | Modifying historical notebook `short/codon_usage.ipynb`; rejected because historical notebooks must remain immutable evidence per AD-004. |
| Jaccard matrix artifact chaining | `results_reproduction/motifs/motif_interaction_jaccard_{variant}.parquet` written by notebook 07 and read by notebook 08 | Recomputing 1D motif convolutions on the entire test set in notebook 08; rejected because decoupled intermediate artifacts enable modular execution and rapid auditing. |

- Nothing else in this change is hard to reverse

## Criteria

Grouped by slice - one observable outcome each, never a layer.

### S1: First-Layer Motifs, Paired Perturbation & Jaccard Co-occurrence Notebook (`07_first_layer_motifs_and_perturbation.ipynb`) (P1)

Creates `notebooks/reproduction/short/07_first_layer_motifs_and_perturbation.ipynb` reproducing first-layer motif extraction, paired perturbation under uniform vs GC-matched controls, Wilcoxon signed-rank tests with FDR correction, Jaccard co-occurrence matrix generation, Table 8, and Figure 3.

**Acceptance Criteria**

1. WHEN `07_first_layer_motifs_and_perturbation.ipynb` executes under `SMOKE_TEST=1` THEN the system SHALL complete execution without errors, output Table 8 summary metrics, generate Figure 3 panels, and save Jaccard co-occurrence matrices to `results_reproduction/motifs/`.
2. The system SHALL ensure `07_first_layer_motifs_and_perturbation.ipynb` imports all motif analysis logic exclusively from `viral_cnn.interpretability.motifs` and contains zero imports of `short/utils/*`.
3. WHEN consensus 7-mers are extracted in `07_first_layer_motifs_and_perturbation.ipynb` THEN the system SHALL link panel (e) of Figure 3 to checkpoint `convnext_noise_species` distinct from panel (b) `resnet_noise_species`.
4. WHEN paired motif perturbation is evaluated in `07_first_layer_motifs_and_perturbation.ipynb` THEN the system SHALL compute both uniform random control drops and GC-matched control drops (`|ΔGC| = 0`) on the same sequences without overlap.
5. WHEN motif perturbation results are summarized in `07_first_layer_motifs_and_perturbation.ipynb` THEN the system SHALL report paired Wilcoxon signed-rank test statistics and Benjamini-Hochberg FDR-adjusted $q$-values for both control policies.
6. WHEN pairwise motif co-occurrence is evaluated in `07_first_layer_motifs_and_perturbation.ipynb` THEN the system SHALL export the symmetric Jaccard co-occurrence matrix to Parquet under `results_reproduction/motifs/`.
7. The system SHALL preserve historical notebook `short/full_loss_function.ipynb` strictly unmodified.

**Independent test:** Run execution of `07_first_layer_motifs_and_perturbation.ipynb` under `SMOKE_TEST=1` and verify generation of motif summary parquet, Jaccard co-occurrence parquet, and Figure 3 image artifacts.

### S2: Exploratory Triplet Composition, Boundary Signals & Top-200 Co-occurrence Notebook (`08_triplet_composition_and_cooccurrence.ipynb`) (P2)

Creates `notebooks/reproduction/short/08_triplet_composition_and_cooccurrence.ipynb` reproducing exploratory triplet composition (RSCU), translation initiation/termination boundary signals, top-200 co-occurrence pair symmetry, and permutation null models, generating Tables 9–11 and Figures 4–5.

**Acceptance Criteria**

8. WHEN `08_triplet_composition_and_cooccurrence.ipynb` executes under `SMOKE_TEST=1` THEN the system SHALL complete execution without errors, generate Tables 9, 10, and 11, and output Figure 4 and Figure 5 artifacts.
9. The system SHALL ensure `08_triplet_composition_and_cooccurrence.ipynb` imports all triplet and co-occurrence logic exclusively from `viral_cnn.interpretability.motifs` and contains zero imports of `short/utils/*`.
10. WHEN exploratory triplet frequencies and RSCU are computed in `08_triplet_composition_and_cooccurrence.ipynb` THEN the system SHALL evaluate frame 0, forward 3-frames, and 6-frames modes and emit explicit warnings that calculations represent unannotated metagenomic triplet composition rather than proven biological codon usage.
11. WHEN translation boundary signals are audited in `08_triplet_composition_and_cooccurrence.ipynb` THEN the system SHALL quantify frequencies of `ATG` and `TAA`/`TAG`/`TGA` across learned motifs against genome background frequencies.
12. WHEN top-200 co-occurrence pairs are analyzed in `08_triplet_composition_and_cooccurrence.ipynb` THEN the system SHALL consume the Jaccard matrix generated in notebook 07, test orientation symmetry between $(A, B)$ and $(B, A)$, and evaluate edge significance against a node-label permutation null model (1,000 permutations).
13. WHEN inter-architecture co-occurrence correlations are computed in `08_triplet_composition_and_cooccurrence.ipynb` THEN the system SHALL report Pearson $r$ and Spearman $\rho$ with two-tailed exact $p$-values and Benjamini-Hochberg FDR adjustment.
14. The system SHALL preserve historical notebook `short/codon_usage.ipynb` strictly unmodified.

**Independent test:** Run execution of `08_triplet_composition_and_cooccurrence.ipynb` under `SMOKE_TEST=1` consuming notebook 07 Jaccard artifacts and verify generation of Tables 9–11 and Figures 4–5.

## Out of scope

| Excluded | Why |
| --- | --- |
| Biological open reading frame (ORF) annotation of all metagenomic reads | Metagenomic short reads (512 bp) lack universal coding sequence boundaries; predicting viral ORFs across 13,683 species is out of scope |
| In vitro or biological assay of transcription factor binding | Purely in silico computational analysis; wet-lab assays are impossible within the software codebase |
| In-place modification of historical notebooks `short/full_loss_function.ipynb` and `short/codon_usage.ipynb` | Historical notebooks are preserved as immutable evidence per AD-004 and AGENTS.md |
| Retraining CNN backbones with motif regularization | Retraining is out of scope for interpretability reproduction; covered in S12 |

## Assumptions

| Assumption | Chosen default | Rationale | Confirmed? |
| --- | --- | --- | --- |
| Kernel motif length | 7 base pairs | Matches first-layer convolutional kernel size across ResNet50 and ConvNeXt architectures | y |
| Random repeat count | 5 repeats with seed 42 | Matches exact parameterization in historical `short/full_loss_function.ipynb` | y |
| Minimum sequence support for FDR | 5 sequences | Matches filtering threshold in `short/full_loss_function.ipynb` cells 11 and 56 | y |
| Co-occurrence edge count | Top 200 edges by Jaccard score | Matches `short/codon_usage.ipynb` cell 16 and cell 20 | y |
| Null model permutation count | 1000 permutations | Provides stable empirical p-value resolution within computational limits | y |
| GC-matched tolerance | Exact match (delta_gc = 0) with delta_gc = 1 fallback | Guarantees exact nucleotide composition matching whenever feasible | y |

**Open questions:** none - all resolved or logged above.

## Observable

| Surface | Decision | Landing |
| --- | --- | --- |
| Reproduction notebook 07 | execution exit 0, outputs Table 8, Fig 3 | AC 1 |
| Reproduction notebook 08 | execution exit 0, outputs Tables 9-11, Figs 4-5 | AC 8 |
| Jaccard parquet artifacts | schema with columns `motif_i`, `motif_j`, `jaccard` | AC 6 |
| Triplet audit artifact | schema with `codon`, `amino_acid`, `rscu`, `p_value`, `q_value` | AC 10 |

## Sources

- Manuscript Section 2.9, Section 3.4, Tables 8–11, and Figures 3–5: Published motif extraction, perturbation controls, and codon usage synergy claims.
- Historical notebooks `short/full_loss_function.ipynb` and `short/codon_usage.ipynb`: Reference implementations for first-layer motif extraction, Wilcoxon signed-rank perturbation, and Jaccard co-occurrence matrices.
- Repository scientific invariant rule `AGENTS.md` and Roadmap `ROADMAP_REPRODUCAO_NOTEBOOKS.md` Section N05: Mandatory paired controls, separation of historical results from sensitivity analyses, symmetric null models, and exploratory triplet demarcation.

