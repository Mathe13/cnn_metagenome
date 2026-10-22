# S08: Latent Space Re-evaluation and Taxonomic Hierarchy Audit

## Problem

The manuscript evaluates learned feature representations of 1D convolutional networks (ResNet50 and ConvNeXt) via UMAP visualization (Figure 2), KNN accuracy and Silhouette Score across taxonomic ranks (Table 5), and hierarchical rank correlations (Table 6).
However, multiple methodological and interpretive conflations exist in the historical notebook (`short/ICTV_latent_space-bkp.ipynb`) and manuscript:
1. **Visualization vs. Quantitative Metric Conflation**: UMAP is a non-linear 2D projection parameterized by cosine distance, $n=15$, and random seed 42. It cannot be used to infer quantitative cluster separation, phylogenetic distance, or relative embedding quality.
2. **Transductive Test KNN vs. Linear Classifier Bottleneck**: In historical analysis, ConvNeXt achieved ~63% species-level 5-NN accuracy on test embeddings, exceeding the trained linear classifier's accuracy (58-61%), prompting the hypothesis of a "bottleneck in the classifier head". However, the 5-NN evaluation was transductive (performed on an 80/20 reference/query split of the test set itself, using test set labels for reference lookup). A non-parametric nearest-neighbor lookup on test embeddings does not demonstrate an architectural bottleneck in the classifier head; testing for a head capacity or training deficit requires training an independent linear probe exclusively on train/validation embeddings and evaluating it on test embeddings.
3. **Silhouette Subsampling and Class Drop Bias**: Computing exact Silhouette on 1.7M embeddings is computationally prohibitive, so up to 50,000 samples were drawn uniformly at random. In a high-cardinality label space with 13,683 species (~3.65 samples/species on average), random subsampling results in hundreds of species having $<2$ samples, which are silently excluded from Silhouette calculation without reporting the valid class rate or evaluating stratified sampling.
4. **Fragment Split vs. Genome-Disjoint Generalization**: The 80/20 test split for KNN was performed across randomly partitioned fragments. Because ~95.89% of test fragments share identical 512 bp sequences with the training set (S02 audit) and test fragments originate from the same genomes, nearest neighbors often retrieve fragments from the identical source genome. KNN accuracy under fragment splitting reflects local fragment identity rather than taxonomic generalization across independent genomes.
5. **Taxonomic Hierarchy Aggregation and Incomplete Correlation Reporting**: Table 6 reports Spearman's rank correlation ($\rho$) for child-parent metric aggregation, but omits Pearson's correlation ($r$), sample counts ($N$), and the exact list of excluded classes across Species $\to$ Genus and Genus $\to$ Order. Furthermore, hierarchical correlation measures geometric consistency across taxonomic ranks, not phylogenetic evolutionary distance.

When this ships, `src/viral_cnn/eval/latent.py` provides auditable, deterministic modules for pre-classifier embedding extraction, GPU/CPU batched cosine KNN ($k=5$, disjoint 80/20 split, zero self-neighbor), genome-disjoint KNN evaluation, high-dimensional Silhouette scoring with class-inclusion reporting under random and stratified sampling, hierarchical child-parent correlation (Pearson $r$ and Spearman $\rho$), and independent train/val linear probing; `scripts/run_latent_evaluation.py` provides an auditable CLI; `docs/latent_space_audit.md` provides an exhaustive scientific audit report correcting Table 5 and Table 6, establishing valid sample counts $N$, framing UMAP as strictly qualitative, and resolving the ConvNeXt "bottleneck" hypothesis with evidence-bounded conclusions; and contract tests verify all invariants deterministically.

## Flow

Reuses `src/viral_cnn/models/builder.py`, `src/viral_cnn/taxonomy/mapping.py`, and `src/viral_cnn/eval/metrics.py` while preserving existing notebook artifacts and datasets.

1. trained PyTorch model and sequence batch -> `viral_cnn.eval.latent:extract_latent_embeddings` (new, door 1) - registers forward pre-hook on `model.fc` and extracts high-dimensional pre-classifier embeddings
2. test embeddings and taxonomic labels -> `viral_cnn.eval.latent:evaluate_cosine_knn` (new, door 2) - performs stratified 80/20 reference/query split, $L_2$ normalization, batched cosine similarity, and majority vote with $k=5$ for global and per-taxon accuracy
3. test embeddings, taxonomic labels, and genome accessions -> `viral_cnn.eval.latent:evaluate_genome_disjoint_knn` (new, door 3) - performs genome-grouped disjoint reference/query split and computes out-of-genome nearest-neighbor accuracy
4. test embeddings and taxonomic labels -> `viral_cnn.eval.latent:evaluate_silhouette_with_audit` (new, door 4) - computes cosine Silhouette Score on subsample ($\le 50,000$), audits excluded classes with $<2$ samples, and compares random vs stratified sampling
5. child and parent per-taxon metrics -> `viral_cnn.eval.latent:calculate_taxonomic_correlations` (new, door 5) - aggregates child metrics by parent, maintaining separate valid subsets, and computes Pearson $r$ and Spearman $\rho$ with sample count $N$
6. train/val embeddings and test embeddings -> `viral_cnn.eval.latent:train_linear_probe` (new, door 6) - trains linear probe exclusively on train/val embeddings and evaluates generalization on test embeddings to test the head bottleneck hypothesis
7. `scripts/run_latent_evaluation.py` (new, no door - placement per conventions) - CLI executing latent space evaluation, exporting metrics CSVs, UMAP 2D projections with qualitative disclaimers, and JSON execution manifests
8. out: auditable metric tables, qualitative UMAP projections, and scientific audit documentation in `docs/latent_space_audit.md`

## Impact

| Front | What changes |
| --- | --- |
| domain | new term: `PreClassifierEmbedding` - high-dimensional feature vector immediately prior to the linear classification head (`model.fc`), lives in `src/viral_cnn/eval/latent.py` |
| domain | new term: `TransductiveTestKNN` - $k$-nearest neighbor classification of 20% test queries using an 80% reference partition from the test set itself |
| domain | new term: `GenomeDisjointKNN` - $k$-nearest neighbor classification where reference and query partitions share zero genome accessions |
| domain | new term: `IndependentLinearProbe` - linear classification head trained strictly on training or validation split embeddings and evaluated on test embeddings |
| domain | new term: `ValidClassInclusionRate` - proportion of taxonomic classes represented by $\ge 2$ samples within a subsampled Silhouette subset |
| stored data | nothing to migrate: historical notebooks and parquet files remain unmodified in place |

## Relations

None - no stored-data shape change

## Surface

None - nothing consumed outside

## Landing

| One-way door | Literal shape | Alternative rejected |
| --- | --- | --- |
| Pre-classifier feature hook | `model.fc.register_forward_pre_hook(hook_fn)` capturing input tensor `inputs[0]` before linear projection | extracting post-fc logits or pooling layers - rejected because latent space in manuscript is defined strictly as the pre-classifier input representation |
| PyTorch cosine KNN engine | `F.normalize(X_test, p=2) @ F.normalize(X_ref, p=2).T` with `torch.topk(k=5)` and `torch.mode` on GPU/CPU | external Faiss or scikit-learn dependency - rejected to preserve minimal dependencies and maintain exact numerical equivalence with notebook Cell 12 |
| Independent valid subsets for correlations | computing $N_{\text{parents}}$, Pearson $r$, and Spearman $\rho$ independently for KNN and Silhouette without joint intersection dropping | filtering to common parent taxa across both metrics - rejected because dropping valid KNN parents due to Silhouette sample size truncation distorts sample sizes |
| Strict train/val linear probe protocol | `LinearProbeClassifier` trained strictly on `(X_train, y_train)` or `(X_val, y_val)` and evaluated on `(X_test, y_test)` | using test-to-test 5-NN accuracy to claim head bottleneck - rejected because transductive neighbor lookup on test embeddings does not prove parametric linear head underfitting |
| Explicit Silhouette class accounting | reporting `valid_classes`, `excluded_classes`, `valid_sample_rate`, and contrasting random vs stratified subsampling | silently discarding classes with $<2$ samples without reporting denominator loss - rejected because high-cardinality subsampling drops hundreds of classes |

- Nothing else in this change is hard to reverse

## Criteria

### S1: Pre-Classifier Embedding Extraction and Cosine KNN (P1)

Extraction of high-dimensional pre-classifier embeddings and evaluation of $k$-nearest neighbor classification under cosine distance.

**Acceptance Criteria**

1. WHEN a model is evaluated for latent features THEN the system SHALL register a forward pre-hook on `model.fc` and capture the exact high-dimensional input tensor to the classification layer without dimensionality reduction.
2. WHEN extracting embeddings across batches THEN the system SHALL concatenate features and ground-truth labels into aligned arrays or tensors matching the dataset size.
3. WHEN evaluating KNN accuracy THEN the system SHALL partition evaluation embeddings into an 80% reference set and a 20% query set using a stratified split with fixed random seed 42.
4. The system SHALL enforce that reference and query sets are disjoint so that no query embedding can be its own nearest neighbor.
5. WHEN computing neighbor similarity THEN the system SHALL $L_2$-normalize reference and query embeddings and compute cosine similarity.
6. WHEN classifying query embeddings with $k=5$ THEN the system SHALL assign predicted labels via majority vote among the top 5 reference neighbors, breaking ties deterministically by lowest class label index.
7. The system SHALL evaluate KNN accuracy independently at Species, Genus, and Order levels, outputting global accuracy and per-taxon accuracy with sample counts.

**Independent test:** unit tests in `tests/test_latent_space.py` verifying that hooks extract exact pre-classifier dimensions for ResNet50 (2048) and ConvNeXt (768), and that 5-NN classification reproduces exact majority vote predictions with no self-neighbors.

### S2: Genome-Disjoint KNN Evaluation (P1)

Evaluation of nearest-neighbor consistency when reference and query sets originate from strictly independent viral genomes.

**Acceptance Criteria**

8. WHERE genome accession annotations are provided for evaluation fragments THEN the system SHALL partition reference and query sets by grouping on `genome_id`, ensuring zero accession overlap between reference and query.
9. IF a species possesses only a single reference genome THEN the system SHALL exclude that species from the genome-disjoint closed-set query evaluation and report the excluded species count.
10. WHEN computing genome-disjoint KNN accuracy with $k=5$ THEN the system SHALL retrieve nearest neighbors exclusively from reference genomes distinct from the query genome.
11. The system SHALL report both fragment-split KNN accuracy and genome-disjoint KNN accuracy, computing the accuracy difference in percentage points.

**Independent test:** unit tests in `tests/test_latent_space.py` confirming that genome accessions in reference and query are strictly disjoint and that fragments from the same genome never act as neighbors.

### S3: High-Dimensional Silhouette Score & Subsampling Audit (P1)

Computation of Silhouette Score in the original feature space and systematic auditing of subsampling class dropouts.

**Acceptance Criteria**

12. The system SHALL compute Silhouette Scores directly in the original high-dimensional embedding space using cosine distance.
13. WHEN subsampling up to 50,000 embeddings with random seed 42 THEN the system SHALL identify and record all taxonomic classes with fewer than 2 samples.
14. The system SHALL report the total number of evaluated classes, the number of valid classes ($\ge 2$ samples), the number of excluded classes ($<2$ samples), and the percentage of retained samples.
15. WHERE stratified subsampling is selected THEN the system SHALL sample at least 2 samples per eligible class up to the target sample size, maximizing the number of valid classes evaluated.
16. The system SHALL compute both the global Silhouette Score and per-taxon mean Silhouette values across Species, Genus, and Order levels.

**Independent test:** unit tests in `tests/test_latent_space.py` confirming that random subsampling correctly audits classes with $<2$ samples, that stratified subsampling retains all eligible classes, and that Silhouette values match exact scikit-learn cosine outputs.

### S4: Hierarchical Taxonomic Correspondence and Correlation Audit (P1)

Hierarchical aggregation of latent cluster metrics across taxonomic ranks and computation of dual Pearson/Spearman correlation coefficients.

**Acceptance Criteria**

17. WHEN mapping child taxa to parent taxa (Species $\to$ Genus or Genus $\to$ Order) THEN the system SHALL aggregate child metrics by computing the arithmetic mean of child values grouped by parent taxon.
18. WHEN computing parent-child correlations THEN the system SHALL compute both Pearson correlation coefficient ($r$, $p$-value) and Spearman rank correlation coefficient ($\rho$, $p$-value).
19. IF a parent taxon contains fewer than 3 child taxa with valid metrics THEN the system SHALL exclude that parent taxon from correlation calculations.
20. The system SHALL maintain independent valid sample subsets for KNN Accuracy ($N_{\text{parents\_KNN}}$) and Silhouette Score ($N_{\text{parents\_Sil}}$), without dropping valid KNN parents due to Silhouette sample size exclusions.

**Independent test:** unit tests in `tests/test_latent_space.py` verifying parent-child aggregation against synthetic taxonomic trees, confirming independent $N$ accounting for KNN and Silhouette, and verifying Pearson and Spearman outputs against `scipy.stats`.

### S5: ConvNeXt Head Bottleneck & Independent Linear Probe Audit (P1)

Empirical testing of the classifier head bottleneck hypothesis via independent linear probing on training/validation representations.

**Acceptance Criteria**

21. WHEN testing for a classifier head bottleneck THEN the system SHALL train a linear classification probe (`LinearProbeClassifier`) exclusively on training or validation split embeddings.
22. The system SHALL forbid the linear probe from accessing or evaluating on test set embeddings during training or optimization.
23. WHEN evaluating the trained linear probe on test set embeddings THEN the system SHALL compute Top-1 accuracy and Macro-F1 across target classes.
24. The system SHALL output a comparative bottleneck diagnostic table contrasting: (a) native model classifier accuracy, (b) trained linear probe accuracy, and (c) transductive test 5-NN accuracy.
25. The system SHALL record in the diagnostic output that transductive test 5-NN accuracy does not demonstrate undercapacity of the parametric classifier head.

**Independent test:** unit tests in `tests/test_latent_space.py` verifying that the linear probe trains on synthetic train embeddings without leakage, evaluates on test embeddings, and generates the comparative diagnostic summary.

### S6: CLI Runner, Qualitative UMAP Export, and Latent Space Audit Report (P2)

CLI workflow, qualitative 2D UMAP projection export, and publication of the comprehensive scientific latent space audit document.

**Acceptance Criteria**

26. The system SHALL provide an auditable CLI in `scripts/run_latent_evaluation.py` accepting `--model-type`, `--checkpoint-path`, `--data-dir`, `--taxonomy-file`, `--sample-size`, and `--output-dir`, exiting with code 0 on successful execution.
27. WHERE UMAP projection is requested THEN the system SHALL compute 2D coordinates using cosine distance ($n=15$, $\text{min\_dist}=0.1$, random seed 42) and export coordinates with an explicit header declaring that UMAP is qualitative only and excluded from quantitative metrics.
28. The system SHALL publish an exhaustive scientific audit report in `docs/latent_space_audit.md` documenting Table 5 (corrected with class counts and subsampling rates), Table 6 (with both Pearson $r$ and Spearman $\rho$, and explicit $N$), the genome-disjoint KNN findings, and the linear probe bottleneck resolution.

**Independent test:** run `python3 scripts/run_latent_evaluation.py --help` exiting with code 0, execute synthetic latent evaluation pipeline producing CSV outputs and JSON execution manifests, and verify that `docs/latent_space_audit.md` exists and contains required sections.

## Out of scope

| Excluded | Why |
| --- | --- |
| Full pairwise Silhouette computation across all 1,706,270 test embeddings | Quadratically prohibitive $O(N^2)$ memory and compute, requiring $>5.8$ TB of pairwise float32 distances |
| Retraining full 1D CNN backbones from scratch on genome-disjoint splits | Full backbone retraining is reserved for the S12 controlled experiment |
| Inferring evolutionary or phylogenetic distance from UMAP projections | Contrary to AGENTS.md scientific invariants: geometric embedding distance does not establish phylogenetic lineage |
| Modifying historical notebooks `short/ICTV_latent_space-bkp.ipynb` in place | Preserved as historical evidence under AGENTS.md and decision AD-004 |

## Assumptions

| Assumption | Chosen default | Rationale | Confirmed? |
| --- | --- | --- | --- |
| KNN neighbor count | $k=5$ | Matches manuscript Section 2.8.1 and notebook Cell 12 | y |
| KNN reference/query split | 80% reference, 20% query with seed 42 | Matches manuscript Section 2.8.1 and notebook Cell 12 | y |
| Silhouette sample size default | 50,000 samples with seed 42 | Matches manuscript Section 2.8.1 and notebook Cell 12 | y |
| Minimum children for correlation | $\ge 3$ child taxa per parent | Matches notebook Cell 20 `if len(knn_summary) >= 3` | y |
| UMAP hyperparameters | cosine metric, $n=15$, $\text{min\_dist}=0.1$, seed 42 | Matches notebook Cell 6 for qualitative 2D projection | y |
| Linear probe optimization | Logistic regression with $L_2$ regularization / AdamW | Standard non-leaking probing protocol to measure linear separability of frozen embeddings | y |

**Open questions:** none - all resolved or logged above.

## Observable

| Surface | Decision | Landing |
| --- | --- | --- |
| CLI `scripts/run_latent_evaluation.py` | exit codes and help output | AC 26 |
| CLI `scripts/run_latent_evaluation.py` | structured CSV outputs for global metrics, per-cluster metrics, and correlations | AC 26 |
| CLI `scripts/run_latent_evaluation.py` | JSON execution manifest with parameter provenance | AC 26 |
| UMAP 2D projection export | qualitative-only disclaimer header in output CSV/parquet | AC 27 |
| Audit documentation `docs/latent_space_audit.md` | publication of corrected Tables 5 and 6 and evidence boundaries | AC 28 |

## Sources

- `paper/BioData_Minning___Matheus.pdf` (Sections 2.8.1, 2.8.2, 3.4, Tables 5 and 6, Figure 2) - binds experimental definition of KNN, Silhouette, UMAP, and hierarchical correlations
- `short/ICTV_latent_space-bkp.ipynb` (Cells 5, 6, 12, 13, 14, 15, 20, 21, 22) - binds implementation details of feature hook, PyTorch GPU KNN, sample size, and correlation summaries
- `AGENTS.md` (Latent representations and scientific invariants) - binds pre-classifier embedding extraction, original high-dimensional metric computation, UMAP qualitative boundary, and avoidance of overreaching claims

