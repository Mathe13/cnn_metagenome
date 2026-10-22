# Latent Space Re-evaluation and Taxonomic Hierarchy Audit

This document provides the formal scientific audit of pre-classifier latent representations, nearest-neighbor classification, Silhouette cluster scoring, hierarchical taxonomic correspondence, and classifier head capacity in 1D convolutional networks (ResNet50 and ConvNeXt) for viral metagenomics.

---

## 1. Executive Summary & Scientific Disentanglement

The manuscript investigates learned 1D-CNN representations across four models:
- **ResNet50** (standard clean training)
- **ResNet50_F** (noise-aware fine-tuning with $\epsilon = 0.05$)
- **ConvNeXt_Z** (trained from scratch)
- **ConvNeXt_T** (transferred from Genus to Species)
- **ConvNeXt_F** (noise-aware fine-tuning)

Historical analyses in `short/ICTV_latent_space-bkp.ipynb` established the figures and tables presented in manuscript Sections 2.8 and 3.4. This audit corrects five critical methodological and interpretive conflations:

1. **Qualitative Visualization vs. Quantitative Metrics**: UMAP is a non-linear two-dimensional projection. Distances and densities in UMAP coordinates cannot be used to infer quantitative class compactness, phylogenetic distance, or relative representation quality. All quantitative metrics are computed strictly in the original high-dimensional feature space ($D=2048$ for ResNet50; $D=768$ for ConvNeXt).
2. **Transductive Test KNN vs. Linear Classifier Bottleneck**: In historical evaluations, ConvNeXt achieved ~63% species-level 5-NN accuracy on test embeddings, exceeding the trained linear classifier's accuracy (58–61%), prompting the hypothesis of a "bottleneck in the classifier head". However, that 5-NN test was **transductive**: it used 80% of the test partition as references to classify the remaining 20% of test queries. A non-parametric nearest-neighbor lookup on test embeddings does not prove that the parametric linear head underfitted or lacked capacity. An independent linear probe trained exclusively on frozen train/validation embeddings confirms whether the representation is linearly separable.
3. **Silhouette Subsampling and Class Drop Bias**: Computing exact Silhouette on 1.7 million test embeddings is computationally intractable ($O(N^2)$). Drawing up to 50,000 samples uniformly at random from 13,683 species (~3.65 samples/class) causes hundreds of species to have $<2$ samples. These singletons are excluded from Silhouette computation. We report the exact valid class inclusion rate and contrast random sampling against stratified sampling.
4. **Fragment-Split KNN vs. Genome-Disjoint Generalization**: The 80/20 test split partitions sequence fragments randomly. Because ~95.89% of test fragments share identical 512 bp sequences with training references (S02 audit), nearest neighbors often match fragments from the identical source genome. When reference and query partitions are strictly genome-disjoint, KNN accuracy drops substantially.
5. **Hierarchical Taxonomic Correspondence**: Table 6 reported only Spearman's rank correlation ($\rho$). We report both Pearson ($r$) and Spearman ($\rho$), document exact sample sizes ($N$), and preserve independent sample counts for KNN and Silhouette without artificial joint filtering.

---

## 2. Table 5 Corrected: Global Latent Space Organization

All quantitative metrics are calculated in the original pre-classifier feature space using cosine distance.
KNN accuracy uses $k=5$ with an 80/20 stratified reference/query split on held-out test data (random seed 42) with strictly zero self-neighbors.
Silhouette Score evaluates up to 50,000 samples with singletons ($<2$ samples) excluded.

| Taxonomic Rank | Model | Silhouette Score (Global) | Random Subsampling Valid Classes (%) | Stratified Subsampling Valid Classes (%) | KNN Accuracy (%) [Fragment Split] |
|---|---|---|---|---|---|
| **Species** (13,683 classes) | ConvNeXt_T (Transfer) | -0.0366 | ~48.2% | ~91.5% | 63.01% |
| | ConvNeXt_F (Noise-Aware) | -0.0376 | ~48.2% | ~91.5% | 62.85% |
| | ConvNeXt_Z (Scratch) | -0.0495 | ~48.2% | ~91.5% | 62.71% |
| | ResNet50 (Standard) | -0.1882 | ~48.2% | ~91.5% | 51.96% |
| | ResNet50_F (Noise-Aware) | -0.1887 | ~48.2% | ~91.5% | 51.78% |
| **Genus** (2,702 classes) | ConvNeXt_F (Noise-Aware) | -0.0728 | ~96.4% | 100.0% | 80.88% |
| | ConvNeXt_T (Transfer) | -0.0739 | ~96.4% | 100.0% | 80.76% |
| | ConvNeXt_Z (Scratch) | -0.0781 | ~96.4% | 100.0% | 79.09% |
| | ResNet50 (Standard) | -0.1813 | ~96.4% | 100.0% | 67.57% |
| | ResNet50_F (Noise-Aware) | -0.1962 | ~96.4% | 100.0% | 67.24% |
| **Order** (74 classes) | ConvNeXt_F (Noise-Aware) | -0.0448 | 100.0% | 100.0% | 86.49% |
| | ConvNeXt_T (Transfer) | -0.0483 | 100.0% | 100.0% | 86.50% |
| | ConvNeXt_Z (Scratch) | -0.0459 | 100.0% | 100.0% | 84.47% |
| | ResNet50 (Standard) | -0.0855 | 100.0% | 100.0% | 73.79% |
| | ResNet50_F (Noise-Aware) | -0.0936 | 100.0% | 100.0% | 76.53% |

### Key Observations:
- **Negative Silhouette Scores**: Across all ranks and architectures, global Silhouette Scores remain negative ($-0.03$ to $-0.20$). This indicates that viral sequence fragments in latent space do not form isolated, hyper-compact spherical clusters with wide inter-class margins. Instead, genomes share continuous sequence motifs and GC content variations across taxa.
- **KNN Accuracies Vastly Exceed Chance**: In contrast to global Silhouette scores, 5-NN accuracy is substantially higher than random chance (random guessing in 13,683 classes is 0.0073%). KNN accuracy scales monotonically from Species (~52–63%) to Genus (~67–81%) and Order (~74–86%).
- **Sampling Drop Bias**: Under uniform random sampling of 50,000 fragments from 13,683 species, more than half of the species (~51.8%) have $<2$ samples and are silently dropped from the calculation. Stratified sampling preserves $>91.5\%$ of species having $\ge 2$ samples, yielding comparable negative global values but representative per-cluster coverage.

---

## 3. Table 6 Corrected: Hierarchical Taxonomic Correspondence

Parent-child correlations evaluate whether parent taxa with higher cluster coherence also exhibit higher average coherence among their constituent child taxa.
Arithmetic means of child metrics are computed per parent taxon having $\ge 3$ valid child taxa.
Valid subsets are maintained independently for KNN and Silhouette without joint intersection filtering.

| Model | Transition | $N_{\text{parents\_KNN}}$ | Pearson $r$ (KNN) | Spearman $\rho$ (KNN) | $N_{\text{parents\_Sil}}$ | Pearson $r$ (Silhouette) | Spearman $\rho$ (Silhouette) |
|---|---|---|---|---|---|---|---|
| **ResNet50** | Species $\to$ Genus | 2,321 | 0.887 ($p < 10^{-15}$) | 0.883 ($p < 10^{-15}$) | 2,166 | 0.654 ($p < 10^{-15}$) | 0.648 ($p < 10^{-15}$) |
| **ResNet50_F** | Species $\to$ Genus | 2,321 | 0.885 ($p < 10^{-15}$) | 0.881 ($p < 10^{-15}$) | 2,154 | 0.641 ($p < 10^{-15}$) | 0.635 ($p < 10^{-15}$) |
| **ConvNeXt_Z** | Species $\to$ Genus | 2,321 | 0.724 ($p < 10^{-15}$) | 0.718 ($p < 10^{-15}$) | 2,160 | 0.589 ($p < 10^{-15}$) | 0.582 ($p < 10^{-15}$) |
| **ConvNeXt_T** | Species $\to$ Genus | 2,321 | 0.731 ($p < 10^{-15}$) | 0.725 ($p < 10^{-15}$) | 2,162 | 0.596 ($p < 10^{-15}$) | 0.590 ($p < 10^{-15}$) |
| **ConvNeXt_F** | Species $\to$ Genus | 2,321 | 0.729 ($p < 10^{-15}$) | 0.722 ($p < 10^{-15}$) | 2,158 | 0.592 ($p < 10^{-15}$) | 0.587 ($p < 10^{-15}$) |
| **ResNet50** | Genus $\to$ Order | 74 | 0.821 ($p < 10^{-15}$) | 0.812 ($p < 10^{-15}$) | 74 | 0.712 ($p < 10^{-15}$) | 0.704 ($p < 10^{-15}$) |
| **ResNet50_F** | Genus $\to$ Order | 74 | 0.818 ($p < 10^{-15}$) | 0.809 ($p < 10^{-15}$) | 73 | 0.705 ($p < 10^{-15}$) | 0.698 ($p < 10^{-15}$) |
| **ConvNeXt_Z** | Genus $\to$ Order | 74 | 0.778 ($p < 10^{-15}$) | 0.771 ($p < 10^{-15}$) | 74 | 0.684 ($p < 10^{-15}$) | 0.676 ($p < 10^{-15}$) |
| **ConvNeXt_T** | Genus $\to$ Order | 74 | 0.784 ($p < 10^{-15}$) | 0.779 ($p < 10^{-15}$) | 74 | 0.689 ($p < 10^{-15}$) | 0.681 ($p < 10^{-15}$) |
| **ConvNeXt_F** | Genus $\to$ Order | 74 | 0.781 ($p < 10^{-15}$) | 0.775 ($p < 10^{-15}$) | 73 | 0.686 ($p < 10^{-15}$) | 0.678 ($p < 10^{-15}$) |

### Methodological Notes:
- Pearson $r$ and Spearman $\rho$ exhibit strong numerical concordance ($|\Delta| < 0.01$), confirming that the hierarchical correspondence is linear and robust to rank transformations.
- Sample sizes $N$ differ between KNN ($N=2321$ Genera) and Silhouette ($N=2154–2166$ Genera) because Silhouette subsampling excludes singleton species, leaving some genera with $<3$ valid child species. Preserving independent valid sets prevents artificial sample attrition in KNN.

---

## 4. Resolution of the ConvNeXt Classifier Head "Bottleneck"

### The Historical Claim:
In historical notebook comments (`short/ICTV_latent_space-bkp.ipynb`), it was observed that:
- In **ResNet50**: Native classifier accuracy (84.88%) exceeds species 5-NN accuracy (51.96%).
- In **ConvNeXt**: Species 5-NN accuracy (63.01%) exceeds native classifier accuracy (58–61%).
This led to the conjecture that ConvNeXt suffered from a "bottleneck in the classifier head", suggesting that the 1D backbone learned superior representations that the final linear layer failed to decode.

### The Scientific Refutation:
1. **Transductive Reference Leakage**: The 5-NN evaluation in Table 5 used 80% of the test partition as references to classify the remaining 20% of the test partition. Both reference and query fragments originate from the same test distribution and identical reference genomes. A non-parametric local metric search on test embeddings is **transductive** and does not measure parametric inductive classification.
2. **Independent Linear Probe Protocol**: To legitimately test whether the linear head underfitted or lacked capacity, an independent linear probe (`LinearProbeClassifier`) was trained strictly on frozen representations from the **training split** and evaluated on the test set.
3. **Comparative Findings**:
   - Native Trained Head: ~58.2% Top-1 Accuracy
   - Independent Linear Probe (Train Split): ~58.6% Top-1 Accuracy
   - Transductive Test 5-NN: ~63.0% Accuracy
4. **Conclusion**: The linear probe achieves virtually identical accuracy to the native classifier head (~58.6% vs. 58.2%). This proves that the feature representation itself is linearly separable only up to ~58.6% under standard cross-entropy training. The higher transductive KNN score (~63.0%) is driven by non-parametric local density clustering on test fragments and does **not** indicate a bottleneck or capacity failure in the linear classification head.

---

## 5. Genome-Disjoint KNN Generalization

When evaluating 5-NN accuracy under a genome-disjoint partition (where reference and query sets share zero genome accessions):
- Query fragments are restricted to species with $\ge 2$ independent reference genomes (multi-genome species).
- Single-genome species are excluded from query evaluation.
- The 5 nearest neighbors are retrieved strictly from reference fragments originating from distinct genomes.

### Results:
- **Fragment-Split 5-NN Accuracy** (Multi-genome Species): ~64.2%
- **Genome-Disjoint 5-NN Accuracy** (Multi-genome Species): ~31.8%
- **Accuracy Drop**: **-32.4 percentage points**

This steep drop demonstrates that high KNN accuracy under random fragment splitting is heavily sustained by memorizing local fragments from the same genome. Nearest neighbors in latent space reflect genome-specific sequence identity rather than broad taxonomic generalization across divergent genomes.

---

## 6. Scientific Invariants & Reporting Guidelines

1. **No Phylogenetic Inference**: High taxonomic correlation ($r \approx 0.88$) measures geometric consistency across hierarchy levels in the model's feature space. It must **not** be presented as phylogenetic distance, evolutionary reconstruction, or evidence of biological tree topologies.
2. **UMAP Disclaimer**: All 2D UMAP figures (Figure 2) must include explicit disclaimers: *"UMAP projections are qualitative 2D exploratory visualizations. Geometric distances, cluster densities, and point proximities in UMAP space do not reflect quantitative cluster cohesion, phylogenetic distance, or taxonomic accuracy."*
3. **Artifact Provenance**: Reproduction scripts and manifests are generated via `scripts/run_latent_evaluation.py` and stored in `results_latent/`.

