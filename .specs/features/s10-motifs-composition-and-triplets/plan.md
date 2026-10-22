# S10: Motifs, Compositional Controls, and Triplet Analysis

## Problem

The manuscript and exploratory notebooks (specifically `short/full_loss_function.ipynb`, `short/codon_usage.ipynb`, and `short/analize_convnext.ipynb`) interpret the first-layer $7 \times 4$ convolutional filters as sequence motifs that capture biologically grounded viral features, such as codon usage bias (RSCU), translation initiation/termination signals, and cooperative motif synergy.
However, several major methodological and statistical vulnerabilities affect these findings:

1. **Confounding Motifs with Low-Complexity Homopolymers and GC Bias**: The primary motifs extracted from the first-layer filters across models are dominated by simple homopolymers and extreme GC runs (e.g., `CCCCCCC`, `GGGGGGG`, `GGGGGGA`, `TCCCCCC`). The historical matched-control protocol perturbs a learned 7-mer and contrasts it against a random non-overlapping 7-bp window drawn uniformly from the same sequence. Because random regions have genome-average GC (~40–50%) and standard sequence entropy, perturbing a 100% GC homopolymer against a random-composition window inflates the "specific drop" ($\Delta p_{\text{specific}} = \Delta p_{\text{motif}} - \Delta p_{\text{random}}$) purely due to general nucleotide composition sensitivity, rather than sequence-specific motif recognition.
2. **Absence of Compositional and Homopolymer Sensitivity Controls**: The codebase lacks paired controls matched for local GC content and sequence complexity. Without testing whether learned 7-mers maintain significant probability drops when compared against GC-matched and homopolymer-matched control windows of the same sequence, the claim of sequence-specific motif learning remains unverified.
3. **Artifactual Orientation Dependency in the Top 200 Co-occurrence Pairs**: The analysis of motif synergy extracts the top 200 edges by Jaccard co-occurrence from the upper triangular matrix (`np.triu(..., k=1)`), arbitrarily assigning the row index to Motif A ($X$-axis) and the column index to Motif B ($Y$-axis) based on matrix row ordering. Because Jaccard co-occurrence is an undirected metric ($J(u, v) = J(v, u)$), calculating a standard Pearson correlation $r$ on an arbitrarily oriented $(X, Y)$ scatter plot produces orientation-dependent artifacts. Furthermore, testing statistical significance requires an appropriate null distribution (such as a node-label permutation preserving graph topology) rather than treating undirected co-occurring pairs as independent bivariate Gaussian observations.
4. **Unjustified Biological Framing of RSCU on Unannotated Metagenomic Fragments**: The Relative Synonymous Codon Usage (RSCU) metric was historically computed across test reads by fixing frame 0 (`seq[i:i+3]`) on 512 bp fragments lacking annotated coding sequences (CDS), translation start sites, or known reading frames (+1, +2, +3, -1, -2, -3). Labeling frame-blind triplet distributions as "Codon Usage" and claiming translational synergy overinterprets an exploratory compositional normalization of trinucleotide frequencies.
5. **Overclaiming Functional Causality**: The historical narrative suggests that convolutional filters detect functional viral regulatory motifs or host tRNA adaptation. Per AGENTS.md scientific invariants, associative triplet composition and nucleotide preference must be strictly demarcated from causal translation mechanisms, functional transcription factor binding sites, or proven biological reading frames.

When this ships, `src/viral_cnn/interpretability/motifs.py` provides auditable algorithms for first-layer consensus extraction, exact matched-pair perturbation under seed-controlled repetitions with FDR correction, GC-matched and complexity-matched sensitivity controls, symmetric and null-model co-occurrence pair audits, and frame-aware exploratory triplet frequency analysis; `scripts/run_motif_analysis.py` provides an auditable CLI; `docs/motifs_and_triplet_composition_audit.md` provides an exhaustive scientific audit report contrasting historical findings against compositional controls; and contract tests verify all calculations deterministically.

## Flow

Reuses `src/viral_cnn/models/registry.py` for model loading and architecture inspection, and `src/viral_cnn/data/encoding.py` for spatial nucleotide channel mapping.

1. trained 1D-CNN model (`ResNet50` or `ConvNeXt`) -> `viral_cnn.interpretability.motifs:extract_first_layer_motifs` (new, door 1) - inspects the first convolutional layer (`conv1` or stem) and extracts consensus 7-mers using positive or absolute channel weights
2. consensus 7-mers and sequence batches -> `viral_cnn.interpretability.motifs:scan_motif_occurrences` (new, door 2) - performs 1D exact convolution matching across 4-channel spatial encodings to locate motif start coordinates
3. sequence, motif coordinates, and control mode -> `viral_cnn.interpretability.motifs:sample_matched_control_region` (new, door 3) - samples non-overlapping control windows matched under historical uniform random, GC-matched, or homopolymer-matched policies
4. baseline and perturbed sequences -> `viral_cnn.interpretability.motifs:evaluate_motif_perturbation` (new, door 4) - computes paired drops ($\Delta p_{\text{motif}}$, $\Delta p_{\text{control}}$, $\Delta p_{\text{specific}}$) across $N_{\text{repeats}}$ repetitions with paired Wilcoxon signed-rank test and Benjamini-Hochberg FDR
5. test presence matrix -> `viral_cnn.interpretability.motifs:compute_jaccard_cooccurrence` (new, no door - placement per conventions) - calculates exact pairwise Jaccard co-occurrence matrix across validated motifs
6. Jaccard matrix and motif triplet values -> `viral_cnn.interpretability.motifs:analyze_cooccurrence_pairs` (new, door 5) - performs symmetric orientation audit, intraclass correlation, and node-permutation null hypothesis significance testing on top $K=200$ edges
7. raw sequences or dataloader -> `viral_cnn.interpretability.motifs:compute_exploratory_triplet_rscu` (new, door 6) - computes triplet frequencies across frame 0, 3 forward frames, or 6 frames, auditing deviation between frame-blind triplet normalization and true biological codon usage
8. evaluation inputs and configuration -> `scripts/run_motif_analysis.py` (new, no door - placement per conventions) - CLI orchestrating motif extraction, historical vs. GC-controlled perturbation, co-occurrence graph analysis, and triplet auditing
9. out: motif perturbation summary parquet/CSV, co-occurrence synergy tables, publication-ready figures, and scientific audit in `docs/motifs_and_triplet_composition_audit.md`

## Impact

| Front | What changes |
| --- | --- |
| domain | new term: `ConsensusMotif` - nucleotide sequence of length $K=7$ derived by selecting the channel with maximal weight per position in a first-layer convolutional kernel, lives in `src/viral_cnn/interpretability/motifs.py` |
| domain | new term: `GCMatchedControl` - paired non-overlapping control window of length $K=7$ sampled from the same sequence with identical GC count ($\Delta\text{GC} = 0$) to isolate sequence-order specificity from base composition, lives in `src/viral_cnn/interpretability/motifs.py` |
| domain | new term: `HomopolymerComplexityControl` - paired non-overlapping control window sampled matching the longest homopolymer run length or Shannon entropy of the query motif, lives in `src/viral_cnn/interpretability/motifs.py` |
| domain | new term: `SymmetricCooccurrenceCorrelation` - orientation-invariant correlation for undirected graph edges, evaluated via intraclass correlation coefficient (ICC) and duplicate bidirectional pair symmetrization $[(u, v), (v, u)]$, lives in `src/viral_cnn/interpretability/motifs.py` |
| domain | new term: `NodePermutationNullModel` - permutation testing framework shuffling node triplet properties across graph vertices while holding network edge weights fixed, computing empirical $p$-values for co-occurrence synergy, lives in `src/viral_cnn/interpretability/motifs.py` |
| domain | existing term: `CodonUsageRSCU` historically referred to blind frame 0 triplet frequency; now explicitly redesignated as `ExploratoryTripletComposition` on unannotated fragments without verified open reading frames |
| stored data | nothing to migrate: historical notebooks and parquet files remain unmodified; outputs write to `results_motifs/` and `results_reports/` |

## Relations

None - no stored-data shape change

## Surface

None - nothing consumed outside

## Landing

| One-way door | Literal shape | Alternative rejected |
| --- | --- | --- |
| Consensus motif weight rule | `consensus_rule: Literal["max_positive", "max_abs"] = "max_positive"` over kernel weights of shape $(C_{\text{out}}, 4, K)$ with $A=0, C=1, G=2, T=3$ | unparameterized implicit argmax - rejected because positive weights represent excitatory filter responses while absolute weights confound inhibitory and excitatory channels |
| Paired control sampling policies | `control_policy: Literal["uniform_random", "gc_matched", "complexity_matched"]` with fallback tolerance $\delta_{\text{GC}} \in \{0, 1\}$ and mandatory non-overlap check | uniform random sampling only - rejected because homopolymers and extreme GC motifs pass uniform random controls purely through base composition bias |
| Multiple test correction standard | Benjamini-Hochberg FDR (`statsmodels.stats.multitest.multipletests(method="fdr_bh", alpha=0.05)`) over motifs with minimum sequence support $N_{\text{sequences}} \ge 5$ | uncorrected $p$-values or Bonferroni over-conservatism - rejected because uncorrected tests inflate false discoveries across 64 filters while Bonferroni discards true correlates |
| Undirected co-occurrence symmetry standard | `compute_symmetric_edge_correlation(df_top_edges, rscu_map, method="intraclass")` evaluating bidirectional pairs $[(x_i, y_i), (y_i, x_i)]$ alongside orientation swap testing | arbitrary upper-triangle matrix extraction without orientation checks - rejected because matrix ordering artificially determines $X$ and $Y$ coordinates on undirected graph edges |
| Triplet metric biological classification | `metric_label: Literal["exploratory_triplet_normalization", "codon_usage_rscu"] = "exploratory_triplet_normalization"` with explicit warning when reading frame is unverified | asserting bona fide biological codon usage on unannotated metagenomic fragments - rejected because scientific invariants in AGENTS.md forbid claiming functional reading frames without biological annotation |

- Nothing else in this change is hard to reverse

## Criteria

### S1: First-Layer Filter Consensus Extraction & Occurrence Scanning (P1)

Extraction of consensus $K$-mers from first-layer convolution weights across architectures and exact sequence scanning via 1D spatial convolutions.

**Acceptance Criteria**

1. The system SHALL extract consensus 7-mers from the first convolutional layer (`conv1` or stem) of ResNet50 and ConvNeXt models using the designated channel order $A=0, C=1, G=2, T=3$.
2. WHERE `consensus_rule="max_positive"` is specified THEN the system SHALL select the nucleotide corresponding to the maximum positive weight at each position $k \in \{0, \dots, K-1\}$.
3. WHERE `consensus_rule="max_abs"` is specified THEN the system SHALL select the nucleotide corresponding to the maximum absolute weight at each position $k \in \{0, \dots, K-1\}$.
4. WHEN scanning sequences for motif occurrences THEN the system SHALL construct one-hot convolution kernels for exact pattern matching and return all 0-based start indices within valid sequence positions.
5. IF a sequence position contains padding or ambiguous bases (encoded as null vectors or negative indices) THEN the system SHALL exclude that position from valid motif matching.

**Independent test:** unit tests in `tests/test_motifs.py` verifying consensus extraction on synthetic convolutional weights and exact coordinate matching against synthetic DNA sequences.

### S2: Historical Matched-Pair Perturbation & FDR Validation (P1)

Reproduction of the historical perturbation pipeline evaluating motif-specific probability drops against uniform random non-overlapping regions with fixed seeds, repeats, and FDR correction.

**Acceptance Criteria**

6. WHEN running motif perturbation THEN the system SHALL substitute each base in the target region with a different nucleotide $(x + \text{offset}) \bmod 4$ using random offset $\in \{1, 2, 3\}$.
7. The system SHALL evaluate paired probability drops across $N_{\text{repeats}} = 5$ repetitions with a fixed random seed (default 42).
8. WHEN sampling a historical uniform random control window THEN the system SHALL ensure the control window has length $K=7$, contains only valid $A/C/G/T$ nucleotides, resides in the same sequence, and does not overlap any occurrence of the target motif.
9. The system SHALL compute the motif-specific drop as $\Delta p_{\text{specific}} = \Delta p_{\text{motif}} - \Delta p_{\text{control}}$ for each sequence-motif pair.
10. The system SHALL perform a one-sided paired Wilcoxon signed-rank test (`alternative="greater"`) comparing motif drops against control drops for all motifs with sequence support $N_{\text{sequences}} \ge 5$.
11. The system SHALL adjust $p$-values across evaluated motifs using Benjamini-Hochberg FDR at significance threshold $\alpha = 0.05$.
12. The system SHALL classify a motif as statistically validated under historical criteria IF AND ONLY IF `significant_fdr == True`, `mean_specific_drop > 0`, and `n_sequences >= 5`.

**Independent test:** unit tests in `tests/test_motifs.py` asserting deterministic Wilcoxon signed-rank statistics, FDR adjustment behavior, and exact replication of cell logic from `short/full_loss_function.ipynb`.

### S3: Compositional & Homopolymer Sensitivity Controls (P1)

Sensitivity testing evaluating whether validated motifs withstand paired control windows matched for exact GC content and sequence complexity/homopolymer run lengths.

**Acceptance Criteria**

13. WHEN `control_policy="gc_matched"` is specified THEN the system SHALL sample a non-overlapping control window of length $K=7$ from the same sequence whose GC count exactly matches the GC count of the query motif ($|\Delta\text{GC}| \le \delta_{\text{GC}}$ with default $\delta_{\text{GC}}=0$).
14. IF no non-overlapping window in the sequence matches the exact GC count THEN the system SHALL relax tolerance to $\delta_{\text{GC}} = 1$ or report that no valid GC-matched control exists.
15. WHEN `control_policy="complexity_matched"` is specified THEN the system SHALL sample a non-overlapping control window matching the maximum homopolymer run length of the target motif within $\pm 1$ base.
16. The system SHALL compute GC-controlled specific drops ($\Delta p_{\text{specific\_GC}}$) and homopolymer-controlled specific drops ($\Delta p_{\text{specific\_comp}}$) and compute paired Wilcoxon $p$-values with FDR correction.
17. The system SHALL output a comparative validation summary recording which historical motifs lose statistical significance under GC-matched and complexity-matched controls.
18. IF a motif achieves historical significance under uniform random control but fails under GC-matched control THEN the system SHALL classify that motif as compositionally confounded.

**Independent test:** unit tests in `tests/test_motifs.py` confirming that synthetic homopolymers (e.g. `CCCCCCC`) tested against GC-matched controls yield $\Delta p_{\text{specific}} \approx 0$ while sequence-specific non-homopolymeric motifs retain significance.

### S4: Co-occurrence Symmetry, Orientation Audit & Null Model Testing (P2)

Rigorous auditing of top-200 Jaccard co-occurrence pairs, testing for orientation dependency and building a node-permutation null distribution.

**Acceptance Criteria**

19. WHEN constructing the motif co-occurrence network THEN the system SHALL compute the pairwise Jaccard index $J(u, v) = \frac{|S_u \cap S_v|}{|S_u \cup S_v|}$ across test sequences, ensuring strict symmetry $J(u, v) = J(v, u)$ with diagonal $J(u, u) = 1.0$.
20. WHEN extracting the top $K=200$ co-occurring edges THEN the system SHALL test orientation dependency by computing Pearson correlation $r$ under both canonical upper-triangle ordering $(u, v)$ and reversed ordering $(v, u)$, reporting the orientation difference $|\Delta r| = |r_{(u, v)} - r_{(v, u)}|$.
21. The system SHALL compute the symmetric intraclass correlation coefficient (ICC) and duplicate bidirectional pair correlation over all $2K$ ordered pairs $[(x_i, y_i), (y_i, x_i)]$, providing an orientation-invariant synergy metric.
22. WHEN evaluating statistical significance of co-occurrence synergy THEN the system SHALL construct a node-permutation null distribution (minimum 1,000 permutations) by randomly shuffling motif triplet values across graph vertices while holding the network topology and Jaccard edge weights constant.
23. The system SHALL compute empirical two-sided permutation $p$-values ($p_{\text{perm}} = \frac{1}{B} \sum_{b=1}^B \mathbf{1}_{|r_b| \ge |r_{\text{obs}}|}$) to determine whether observed synergy exceeds random vertex assignment.

**Independent test:** unit tests in `tests/test_motifs.py` evaluating symmetric edge correlation, asserting that swapping edge tuples $(u, v) \leftrightarrow (v, u)$ does not alter symmetric ICC, and verifying that the node-permutation null model produces uniformly distributed $p$-values on random graphs.

### S5: Exploratory Triplet Composition vs. Biological Codon Usage Audit (P2)

Reassessment of frame-blind triplet counting on unannotated fragments and strict demarcation from functional codon usage bias.

**Acceptance Criteria**

24. WHEN computing triplet frequencies on input fragments THEN the system SHALL support explicit frame extraction modes: `frame0` (historical single frame), `forward_3frames` (all 3 forward reading frames), and `six_frames` (forward and reverse-complement frames).
25. The system SHALL calculate Relative Synonymous Codon Usage (RSCU) across the standard genetic code as $RSCU_i = \frac{x_i}{\frac{1}{m_i} \sum_{j=1}^{m_i} x_j}$, where $m_i$ is the number of synonymous codons for the corresponding amino acid.
26. The system SHALL compute expected triplet frequencies under a 0th-order base-composition multinomial null model ($P(b_1 b_2 b_3) = P(b_1) P(b_2) P(b_3)$) to quantify the proportion of triplet frequency explained by simple mononucleotide GC bias.
27. The system SHALL evaluate the frequency of start (`ATG`) and stop (`TAA`, `TAG`, `TGA`) triplets in validated motifs against the mononucleotide background expectation, determining whether apparent translation boundary enrichment is statistically distinguishable from base composition.
28. The system SHALL label all triplet metrics on metagenomic fragments without verified coding annotations as `ExploratoryTripletComposition` in all programmatic outputs and scientific audit reports.

**Independent test:** unit tests in `tests/test_motifs.py` verifying triplet frequency calculations across frame modes, validating multinomial null model expectations, and checking proper metric labeling.

### S6: CLI Workflow, Audit Figures & Comprehensive Scientific Report (P2)

End-to-end command-line orchestration, publication-ready multi-panel figure generation, and publication of the comprehensive scientific audit document.

**Acceptance Criteria**

29. The system SHALL provide an auditable CLI in `scripts/run_motif_analysis.py` accepting `--model-type`, `--checkpoint-path`, `--data-dir`, `--output-dir`, `--control-policy`, `--n-repeats`, and `--seed`, exiting with code 0 upon successful execution.
30. The system SHALL generate structured audit figures where each panel is linked to its verified model checkpoint, showing: (a) motif extraction consensus, (b) paired drop distribution under historical vs. GC-matched controls, (c) symmetric co-occurrence synergy with permutation null intervals, and (d) exploratory triplet frequency profiles.
31. The system SHALL publish an exhaustive scientific audit report in `docs/motifs_and_triplet_composition_audit.md` comparing historical Table 8, Table 9, and Figures 3–5 against the compositional and symmetry controls.
32. IF the GC-matched and complexity-matched controls show that homopolymeric motifs lack sequence-specific predictive value THEN the system SHALL document the finding in `docs/motifs_and_triplet_composition_audit.md` and recommend moving extended codon usage claims to supplementary materials.

**Independent test:** run `python3 scripts/run_motif_analysis.py --help` exiting with code 0, execute deterministic synthetic pipeline generating figures and tables, and verify existence of `docs/motifs_and_triplet_composition_audit.md`.

## Out of scope

| Excluded | Why |
| --- | --- |
| Retraining 1D-CNN backbones with homopolymer penalty loss | Feature S10 audits representations of existing checkpoints; architectural retraining is reserved for S12 |
| Full biological open reading frame (ORF) annotation of all test fragments | Metagenomic fragments (512 bp) lack universal coding sequence coordinates; predicting bacterial/viral ORFs across 13,683 species is out of scope |
| In vitro or functional biological validation of transcription factor binding | Purely in silico computational analysis; wet-lab functional assays are impossible within the software codebase |
| In-place modification of historical notebooks `short/full_loss_function.ipynb` and `short/codon_usage.ipynb` | Historical notebooks are preserved as immutable evidence per decision AD-004 and AGENTS.md |

## Assumptions

| Assumption | Chosen default | Rationale | Confirmed? |
| --- | --- | --- | --- |
| Kernel motif length | $K=7$ bases | Matches first-layer filter kernel length across ResNet50 and ConvNeXt architectures | y |
| Historical repeat count | $N_{\text{repeats}} = 5$ with seed 42 | Matches exact parameterization in `short/full_loss_function.ipynb` Cell 1 | y |
| Minimum sequence support for FDR | $N_{\text{sequences}} \ge 5$ | Matches exact filtering threshold in `short/full_loss_function.ipynb` Cell 11 and Cell 56 | y |
| Co-occurrence edge count | Top 200 edges by Jaccard score | Matches `short/codon_usage.ipynb` Cell 16 and Cell 20 | y |
| Null model permutation count | 1,000 permutations | Provides stable empirical $p$-value resolution ($\ge 0.001$) within computational limits | y |
| GC-matched tolerance | $\delta_{\text{GC}} = 0$ with fallback to $\delta_{\text{GC}} = 1$ | Guarantees exact base composition matching whenever feasible in 512 bp sequences | y |

**Open questions:** none - all resolved or logged above.

## Observable

| Surface | Decision | Landing |
| --- | --- | --- |
| CLI `run_motif_analysis.py` | exit codes, flags, help message | AC 29 |
| CLI `run_motif_analysis.py` | error handling on invalid checkpoint or empty dataset | AC 5 |
| Document `docs/motifs_and_triplet_composition_audit.md` | structure, tone, depth, and manuscript revision recommendations | AC 31, AC 32 |
| Artifact tables `summary_*.parquet` and CSVs | column definitions, metric names, FDR flags | AC 11, AC 17, AC 28 |
| Artifact figures `panel_*.png` | multi-panel layout, checkpoint attribution, null distribution overlays | AC 30 |

## Sources

- Manuscript Section 2.9, Section 3.4, Table 8, Table 9, and Figures 3–5: Published motif extraction, perturbation controls, and codon usage synergy claims.
- Historical notebooks `short/full_loss_function.ipynb`, `short/analize_convnext.ipynb`, and `short/codon_usage.ipynb`: Reference implementations for first-layer motif extraction, Wilcoxon signed-rank perturbation, and Jaccard co-occurrence matrices.
- Repository scientific invariant rule `AGENTS.md` and Roadmap `ROADMAP_SPECS_ARTIGO.md` Section S10: Mandatory paired controls, separation of historical results from sensitivity analyses, symmetric null models, and exploratory triplet demarcation.

