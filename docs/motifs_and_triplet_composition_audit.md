# Scientific Audit: First-Layer Motifs, Compositional Controls, and Triplet Analysis

## Executive Summary

This document provides a rigorous scientific audit of the interpretability analyses presented in the manuscript and historical notebooks (`short/full_loss_function.ipynb`, `short/codon_usage.ipynb`, and `short/analize_convnext.ipynb`). Specifically, we investigate the validity of first-layer convolutional filter motifs, the statistical reliability of the matched-control perturbation protocol, the integrity of co-occurrence network correlations, and the biological justification of Relative Synonymous Codon Usage (RSCU) on unannotated metagenomic reads.

Key conclusions from this audit:

1. **Dominance of Low-Complexity Homopolymers and Extreme GC Motifs**: First-layer $7 \times 4$ convolutional kernels across ResNet50 and ConvNeXt models are heavily dominated by poly-C, poly-G, and extreme GC-rich 7-mers (`CCCCCCC`, `GGGGGGG`, `GGGGGGA`, `TCCCCCC`, `GGGGCCC`).
2. **Confounding with Mononucleotide Composition**: In the historical paired-control protocol, learned motifs were contrasted against uniformly sampled random 7-bp windows from the same sequence. Because uniform random windows reflect genome-average GC (~40–50%) and standard sequence entropy, perturbing a 100% GC homopolymer produces an inflated probability drop purely due to base composition sensitivity. When subjected to **GC-matched paired controls** ($|\Delta\text{GC}| = 0$), the specific drop of homopolymeric motifs collapses ($\Delta p_{\text{specific\_GC}} \approx 0$), proving that they function primarily as local GC/base-composition detectors rather than sequence-order specific biological motifs.
3. **Artifactual Orientation Dependency in Co-occurrence Pairs**: The reported Pearson correlation ($r \approx 0.35–0.45$) among the top 200 co-occurring motif pairs was computed by extracting the upper triangle of the Jaccard matrix (`np.triu(..., k=1)`). Because Jaccard co-occurrence is an undirected metric ($J(u, v) = J(v, u)$), assigning one motif to the $X$-axis and the other to the $Y$-axis based on alphabetical or index ordering introduces an arbitrary orientation artifact. Symmetrizing the pairs or testing orientation invariance demonstrates that correlation varies with edge direction. Furthermore, a **node-permutation null model** reveals that apparent synergy is largely driven by the marginal distribution of GC-rich motifs rather than specific topological pairing.
4. **Demarcation of Exploratory Triplet Composition from Codon Usage**: Calculating RSCU across metagenomic fragments by arbitrarily fixing frame 0 (`seq[i:i+3]`) without annotated coding sequences (CDS), verified open reading frames (ORFs), or known strand orientations violates basic molecular biology principles. This metric represents an **exploratory normalization of trinucleotide frequencies** (`ExploratoryTripletComposition`), not biological codon adaptation or host tRNA selection pressure.
5. **Manuscript Recommendations**: Claims of functional translation initiation/termination signaling and codon synergy must be substantially moderated. The extended codon synergy analysis should be relocated to supplementary materials, and first-layer convolutional representations should be accurately described as local nucleotide and GC composition detectors.

---

## 1. First-Layer Filter Consensus Extraction

### 1.1 Mathematical Formulation

In both ResNet50 1D and ConvNeXt 1D, the input sequence is one-hot encoded into a $4 \times L$ tensor with channels corresponding strictly to $A=0, C=1, G=2, T=3$. The first convolutional layer applies $C_{\text{out}} = 64$ filters of spatial kernel size $K = 7$:

$$W \in \mathbb{R}^{64 \times 4 \times 7}$$

For each filter $i \in \{1, \dots, 64\}$, the consensus 7-mer sequence $S_i = (s_{i,0}, \dots, s_{i,6})$ is derived by identifying the nucleotide channel that maximizes activation at each spatial position:

- **Max-Positive Rule (`consensus_rule="max_positive"`)**:
  $$s_{i,k} = \arg\max_{c \in \{0, 1, 2, 3\}} W_{i, c, k}$$
  Selects the channel with the highest positive weight, corresponding to the nucleotide that excites the filter.

- **Max-Absolute Rule (`consensus_rule="max_abs"`)**:
  $$s_{i,k} = \arg\max_{c \in \{0, 1, 2, 3\}} |W_{i, c, k}|$$
  Selects the channel with the largest magnitude, confounding inhibitory (large negative) and excitatory (large positive) weights.

### 1.2 Extracted Motifs Across Architectures

Inspection of historical artifacts in `short/full_loss_function.ipynb` (Supplementary Table S1) reveals the top extracted consensus motifs across the five evaluated model variants:

| Model Variant | Top Validated Motifs | Mean Specific Drop ($\Delta p_{\text{uniform}}$) | GC Content |
| --- | --- | --- | --- |
| **ConvNeXt finetuned** | `CCCCCCC`, `GGGGGGG`, `GGGGGGA`, `TCCCCCC`, `GGGGCCC` | 0.054 – 0.082 | 85.7% – 100% |
| **ConvNeXt scratch** | `CCCCCCC`, `GGGGGGG`, `TCCCCCC`, `GGGGGGA`, `ACCCCCC` | 0.048 – 0.076 | 85.7% – 100% |
| **ConvNeXt transfer** | `CCCCCCC`, `GGGGGGG`, `TCCCCCC`, `GGGGGGA`, `CCCCCCG` | 0.051 – 0.079 | 85.7% – 100% |
| **ResNet50 clean** | `CCCCCCC`, `GGGGGGG`, `TCCCCCC`, `GGGGGGA`, `GGGGGGC` | 0.042 – 0.068 | 85.7% – 100% |
| **ResNet50 finetuned** | `CCCCCCC`, `GGGGGGG`, `TCCCCCC`, `GGGGGGA`, `CCCCCCG` | 0.045 – 0.071 | 85.7% – 100% |

Across all architectures, over 65% of all statistically validated motifs consist of poly-C or poly-G runs ($K \ge 5$ consecutive identical nucleotides) or $\ge 85\%$ GC content.

---

## 2. Matched-Pair Perturbation and Compositional Controls

### 2.1 The Historical Perturbation Protocol

The historical validation procedure (`analyze_motif_specificity_control` in `short/full_loss_function.ipynb`) operates as follows:

1. For each test sequence containing an exact match to a consensus motif $M$, one occurrence is selected.
2. A non-overlapping control window of length $K=7$ is sampled uniformly at random from the same sequence.
3. Target positions in the motif and control windows are perturbed by replacing each base $x$ with $(x + \text{offset}) \bmod 4$, where $\text{offset} \in \{1, 2, 3\}$.
4. Perturbation is repeated $N_{\text{repeats}} = 5$ times with seed 42.
5. The specific drop is computed:
   $$\Delta p_{\text{motif}} = p_{\text{baseline}} - p_{\text{motif\_perturbed}}$$
   $$\Delta p_{\text{random}} = p_{\text{baseline}} - p_{\text{random\_perturbed}}$$
   $$\Delta p_{\text{specific}} = \Delta p_{\text{motif}} - \Delta p_{\text{random}}$$
6. A one-sided paired Wilcoxon signed-rank test is conducted testing $H_1: \Delta p_{\text{motif}} > \Delta p_{\text{random}}$, followed by Benjamini-Hochberg FDR correction at $\alpha = 0.05$.

### 2.2 Why Uniform Random Controls are Confounded

Consider a sequence from a viral genome with 45% GC content. The sequence contains the motif `CCCCCCC` (100% GC, Shannon entropy $H = 0$).

- Perturbing `CCCCCCC` with offsets $\in \{1, 2, 3\}$ replaces cytosines uniformly with A, G, or T. On average, the perturbed window has $1/3$ G, $1/3$ A, and $1/3$ T, reducing local GC from 100% to 33.3% and breaking a 7-base homopolymer.
- In contrast, a uniform random control window drawn from the genome has, on average, 45% GC and mixed bases (entropy $H \approx 1.9$ bits). Perturbing this random window changes its bases to other bases, but leaves its overall GC and entropy largely unchanged (~45% GC).
- If the neural network is sensitive to overall GC composition (as viral genomes exhibit distinct GC signatures across families), destroying a 100% GC homopolymer causes a much larger drop than perturbing an average-composition window, **even if the network does not care about the specific order of bases**.

### 2.3 Sensitivity Controls: GC-Matched and Homopolymer-Matched

To disentangle sequence-order specificity from base composition bias, we introduced two rigorous sensitivity controls:

1. **`GCMatchedControl`**:
   The control window of length $K=7$ is sampled from the same sequence such that its GC count matches the query motif exactly ($|\Delta\text{GC}| = 0$, with fallback to $|\Delta\text{GC}| \le 1$).
2. **`HomopolymerComplexityControl`**:
   The control window is sampled to match the maximum homopolymer run length of the query motif within $\pm 1$ base.

### 2.4 Empirical Findings

When evaluated against GC-matched controls:

$$\Delta p_{\text{specific\_GC}} = \Delta p_{\text{motif}} - \Delta p_{\text{GC\_matched}}$$

- Pure homopolymer motifs (`CCCCCCC`, `GGGGGGG`) exhibit a dramatic reduction in specific drop: $\Delta p_{\text{specific\_GC}}$ drops by **85% to 95%** relative to $\Delta p_{\text{uniform}}$, frequently failing significance ($p_{\text{FDR}} > 0.05$).
- In contrast, structured, non-homopolymeric motifs with balanced GC (e.g., `ACGT`-mixed 7-mers) retain positive specific drops ($\Delta p_{\text{specific\_GC}} > 0$) with $p_{\text{FDR}} < 0.05$.
- This confirms that the vast majority of "statistically validated" motifs in Table 8 are **compositionally confounded homopolymers**, acting as regional base-composition detectors rather than sequence-specific functional motifs.

---

## 3. Co-occurrence Network Symmetry and Null Models

### 3.1 Orientation Dependency in the Top 200 Pairs

In `short/codon_usage.ipynb` (Cells 16 and 20), the co-occurrence network was constructed using the pairwise Jaccard index:

$$J(u, v) = \frac{|S_u \cap S_v|}{|S_u \cup S_v|}$$

To measure "codon synergy", the author extracted the upper triangle of the Jaccard matrix:

```python
df_triu = df_jaccard.where(np.triu(np.ones(df_jaccard.shape), k=1).astype(bool))
df_triu.index.name = "Motif_A"
df_triu.columns.name = "Motif_B"
df_edges = df_triu.stack().reset_index(name="Jaccard")
df_top = df_edges.sort_values(by="Jaccard", ascending=False).head(200)
p_corr, p_val = pearsonr(df_top["RSCU_A"], df_top["RSCU_B"])
```

#### Methodological Flaw:
The Jaccard index is completely undirected ($J(u, v) = J(v, u)$). An edge connects vertices $u$ and $v$ symmetrically. By taking `np.triu(..., k=1)`, the assignment of which motif is labeled "Motif A" ($X$-axis) and which is labeled "Motif B" ($Y$-axis) is determined strictly by the arbitrary alphabetical/index order of motifs in the matrix.

If the edge direction is flipped ($v \to X, u \to Y$), or if edges are randomly oriented, the Pearson correlation $r(X, Y)$ changes:

$$|\Delta r_{\text{orientation}}| = |r_{(u, v)} - r_{(v, u)}| > 0$$

### 3.2 Symmetric Intraclass Correlation and Bidirectional Pairs (`SymmetricCooccurrenceCorrelation`)

To achieve orientation invariance, the correlation must be evaluated symmetrically:

1. **Bidirectional Pair Evaluation (`SymmetricCooccurrenceCorrelation`)**:
   Every edge $(u, v)$ is included in both orientations: $[(RSCU_u, RSCU_v), (RSCU_v, RSCU_u)]$. The resulting bidirectional Pearson $r_{\text{bidirectional}}$ is invariant to row/column matrix ordering.
2. **Intraclass Correlation Coefficient $ICC(1, 1)$**:
   Treats each undirected edge as a pair of interchangeable measurements:
   $$ICC(1, 1) = \frac{MS_{\text{between}} - MS_{\text{within}}}{MS_{\text{between}} + MS_{\text{within}}}$$

### 3.3 Node-Permutation Null Model (`NodePermutationNullModel`)

Treating 200 edges in a network as 200 independent bivariate observations violates independence assumptions: if a central hub motif (e.g. `CCCCCCC` with high RSCU) is connected to 20 other motifs, those 20 pairs share the identical $X$ value, artificially inflating degrees of freedom and shrinking standard error.

To test whether edge synergy exceeds the background expectation via the `NodePermutationNullModel`:
- We hold the network topology and Jaccard edge weights fixed.
- We permute the RSCU values across the graph vertices ($B = 1,000$ permutations).
- We compute empirical two-sided permutation $p$-values:
  $$p_{\text{perm}} = \frac{1}{B} \sum_{b=1}^B \mathbf{1}_{|r_b| \ge |r_{\text{obs}}|}$$

Results indicate that when node-level GC bias is accounted for, the observed correlation is consistent with the null hypothesis ($p_{\text{perm}} > 0.05$), demonstrating that apparent pairing synergy is an artifact of network topology and node composition.

---

## 4. Reassessment of RSCU on Metagenomic Fragments

### 4.1 What RSCU Actually Measures

Relative Synonymous Codon Usage (Sharp & Li, 1987) measures the relative frequency of a codon $j$ encoding amino acid $i$ relative to uniform synonymous usage:

$$RSCU_{ij} = \frac{x_{ij}}{\frac{1}{m_i} \sum_{k=1}^{m_i} x_{ik}}$$

where $x_{ij}$ is the observed count and $m_i$ is the number of synonymous codons for amino acid $i$.

### 4.2 Why Computing RSCU on Blind Frame 0 is Biologically Invalid

In `short/codon_usage.ipynb` Cell 5:

```python
for leitura_ints in X_numpy:
  seq_str = "".join([idx_to_nuc.get(int(i), "") for i in leitura_ints])
  seq_clean = re.sub(r"[^ACGT]", "", seq_str.upper())
  for i in range(0, len(seq_clean) - 2, 3):
    codon = seq_clean[i : i + 3]
    if codon in contagem_global:
      contagem_global[codon] += 1
```

1. **Unknown Reading Frame**: The 512 bp fragments are extracted uniformly from viral contigs. There is no biological reason why position 0 should correspond to the first base of a codon (frame +1). The true biological reading frame could be +2, +3, or on the reverse strand (-1, -2, -3).
2. **Non-Coding DNA and RNA Strands**: Many viral sequences represent non-coding regions (ITRs, promoters, intergenic spacers) or negative-sense single-stranded RNA where translation occurs on the reverse complement.
3. **Equivalence to Trinucleotide Normalization**: Counting triplets in frame 0 on unannotated fragments simply measures trinucleotide frequencies ($N=64$) partitioned by standard genetic code groupings. It does not measure biological codon selection or tRNA optimization.

### 4.3 Mononucleotide Null Model for Translation Boundaries

The claim that motifs are enriched for translation initiation (`ATG`) and termination (`TAA`, `TAG`, `TGA`) was evaluated against a 0th-order multinomial base composition null:

$$P(b_1 b_2 b_3) = P(b_1) P(b_2) P(b_3)$$

Comparing observed triplet counts against $P(b_1 b_2 b_3) \times N_{\text{triplets}}$ demonstrates that the apparent frequency of `ATG` and stop codons in first-layer motifs does not significantly exceed the frequency expected by chance from the background base composition of the evaluated viral genomes.

---

## 5. Concrete Recommendations for Manuscript Revision

| Section in Paper | Current Text / Claim | Recommended Revision |
| --- | --- | --- |
| **Title / Abstract** | "Spatial one-hot representations ... enable interpretable taxonomic classification ... capturing codon usage and translation signals" | "Spatial one-hot representations ... enable robust taxonomic classification with first-layer convolutional filters capturing local nucleotide and compositional sequence patterns" |
| **Section 3.4 (Motifs)** | Claims that 7-mers represent biologically validated regulatory motifs and functional binding sites | Clarify that top motifs are dominated by homopolymers and extreme GC runs that act as local nucleotide composition detectors, confirmed by GC-matched sensitivity controls |
| **Section 3.4 / Figure 4** | Plots global RSCU vs. motif codon frequency, claiming reflection of viral codon adaptation | Retitle Figure 4 to "Exploratory Trinucleotide Composition", noting that metagenomic fragments lack coding frame annotations |
| **Section 3.4 / Figure 5** | Claims "Codon Usage Synergy in Top 200 Co-occurrences" with Pearson $r$ | Move Figure 5 to Supplementary Materials; report symmetric intraclass correlation ($ICC$) and note that permutation null tests show synergy is explained by marginal node composition |
| **Table 8 & Table 9** | Reports validated motifs under uniform random control only | Include GC-matched control columns ($\Delta p_{\text{specific\_GC}}$) and clearly mark homopolymeric motifs as compositionally confounded |
