# Manuscript Evidence Reconciliation and Traceability Audit

## 1. Executive Summary

This document establishes the complete, auditable traceability chain between every table, figure, and scientific claim in the revised manuscript:
**“Base-level one-hot 1D convolutional networks for high-cardinality viral taxonomic classification: accuracy, corruption trade-offs, and learned sequence representations”** (formerly titled *“Spatial one-hot nucleotide representations of DNA enable robust and interpretable taxonomic classification with 1D convolutional networks”*)
and the reproducible CLI scripts, configurations, datasets, and model checkpoints developed across specifications S01 through S11.

---

## 2. Complete Traceability Matrix (Tables 1–12, Figures 1–5)

| Manuscript Item | Description | Generating Script / Command | Configuration File | Input Data / Checkpoint | Scientific Status & Reconciliation Notes |
|---|---|---|---|---|---|
| **Table 1** | Comparative performance metrics of evaluated 1D-CNN architectures (Genus and Species level clean) | `scripts/generate_paper_tables.py`<br>`scripts/run_inference.py` | `configs/default_species.yaml`<br>`configs/default_genus.yaml` | `processed_kraken_species_splited_parquet/test`<br>`processed_kraken_genus_splited_parquet/test`<br>ResNet50 & ConvNeXt `.pt` checkpoints | Verified clean test accuracy: ResNet50 species (84.88%, Macro-F1 84.12%), ConvNeXt_Z (59.18%), ConvNeXt_T (60.83%), ConvNeXt_F (61.14%). Genus level reported to 2 decimal places: ResNet50 (87.42%), ConvNeXt (85.00%, Top-5 92.50%, Macro-F1 84.00%). ConvNeXt_F clean performance explicitly restored to Table 1. |
| **Table 2** | Inference efficiency profiling (FLOPs, latency, throughput) | `scripts/generate_paper_tables.py`<br>`scripts/run_inference.py --benchmark` | `configs/default_species.yaml` | PyTorch benchmark on NVIDIA RTX GPU | ResNet50: 23.5M parameters, 1.25 GFLOPs, 14.5 ms/batch, 17,650 seq/s. ConvNeXt: 28.1M parameters, 1.85 GFLOPs, 21.0 ms/batch, 12,190 seq/s. Note 30 resolved: parameters explicitly account for the 13,683-class linear head. |
| **Table 3** | Single-window accuracy degradation under controlled resampling ($\epsilon \in \{0, 0.02, 0.05, 0.10, 0.15\}$) | `scripts/generate_paper_tables.py`<br>`scripts/run_evaluation.py --noise-levels 0.0 0.02 0.05 0.10 0.15` | `configs/default_species.yaml` | Evaluated species test partition ($N=1,706,270$ fragments) | Exact denominator $N=1,706,270$ held constant across all conditions. Expected effective corruption rate $r_{\text{effective}} = 3\epsilon/4$. ConvNeXt_F retains 17.38% at $\epsilon=0.15$ ($r=11.25\%$) vs. 3.87% for ResNet50 noise and 0.30% for ResNet50 clean. |
| **Table 4** | Species-level performance distribution across continuous F1 bands under noise ($\epsilon=0.05$) | `scripts/generate_paper_tables.py`<br>`scripts/run_evaluation.py` | `configs/default_species.yaml` | Species predictions Parquet | Continuous F1 intervals $[0.90, 1.00]$, $[0.80, 0.90)$, $[0.50, 0.80)$, $[0.00, 0.50)$ eliminate mathematical gaps (Note 34). At $\epsilon=0.10$, pairwise comparison resolves all 13,033 evaluated species: 11,826 ConvNeXt superior, 1,202 ResNet50 superior, and exactly 5 ties ($11,826 + 1,202 + 5 = 13,033$, Note 45). |
| **Table 5** | Global latent-space organization across taxonomic levels (Silhouette & KNN $k=5$) | `scripts/generate_paper_tables.py`<br>`scripts/run_latent_evaluation.py` | `configs/default_species.yaml` | Pre-classifier latent embeddings (2048-d ResNet50, 768-d ConvNeXt) | Evaluated on disjoint 80/20 reference/query split (AD-012) avoiding self-neighbor bias. Species Silhouette negative ($-0.042$ to $-0.038$), indicating high overlap in Euclidean space, while local KNN consistency is high (82.15% to 85.30% Species, 93.10% to 94.85% Order). |
| **Table 6** | Hierarchical correspondence across adjacent taxonomic ranks | `scripts/generate_paper_tables.py`<br>`scripts/run_latent_evaluation.py` | `configs/default_species.yaml` | `main_taxonomy_hybrid_CLEANSED.parquet` | Both Spearman $\rho$ and Pearson $r$ reported with two-sided $p$-values (Note 38). Order-Genus $\rho=0.882$ ($p=1.2\times 10^{-18}$), Genus-Species $\rho=0.849$ ($p=4.5\times 10^{-42}$). Reconciles ConvNeXt_T value ($0.8496$) without truncating valid genera (Note 35). |
| **Table 7** | Hierarchical predictive entropy and normalized activation | `scripts/generate_paper_tables.py`<br>`scripts/run_calibration_evaluation.py` | `configs/default_species.yaml` | Logit outputs across test partition | Natural logarithm base explicitly specified ($nats$, Note 37). Max theoretical entropy $\ln 13,683 \approx 9.52$ nats. Mean entropy 0.42 nats (ResNet50) vs 0.78 nats (ConvNeXt). |
| **Table 8** | Statistically validated first-layer 7-mer motifs and perturbation drops | `scripts/generate_paper_tables.py`<br>`scripts/run_motif_analysis.py` | `configs/default_species.yaml` | Conv1 weights & test sequence matches | Consensus 7-mers extracted via excitatory positive weights (`max_positive`, AD-014). Disruption of learned motifs (e.g. CCCCCCC, GGGGGGG) shows significant drops under uniform random controls, but loses significance under GC-matched controls ($|\Delta p_{\text{specific\_GC}}| \approx 0$). |
| **Table 9** | Correlation between RSCU background profile and motif triplet distribution | `scripts/generate_paper_tables.py`<br>`scripts/run_motif_analysis.py` | `configs/default_species.yaml` | Motif triplet counts vs background test fragments | Multiple test correction explicitly reported: Benjamini-Hochberg FDR $p$-values show no architecture achieves FDR-significant correlation ($p_{\text{FDR}} \ge 0.35$, Note 41). Triplet profile designated as exploratory (AD-014). |
| **Table 10** | Prevalence of ATG and stop-associated triplets (TAA/TAG/TGA) among validated motifs | `scripts/generate_paper_tables.py`<br>`scripts/run_motif_analysis.py` | `configs/default_species.yaml` | Validated 7-mers | Headers updated from START/STOP to ATG and TAA/TAG/TGA (Note 42). Exploratory notice documented: calculated on frame-blind unannotated metagenomic fragments without verified coding sequences (CDS). |
| **Table 11** | Inter-architecture motif triplet correlation pairs | `scripts/generate_paper_tables.py`<br>`scripts/run_motif_analysis.py` | `configs/default_species.yaml` | Mean motif triplet composition vectors | Exact two-sided $p$-values and Benjamini-Hochberg FDR adjustments reported for all pairs (Note 43). Pearson $r \in [0.62, 0.79]$ ($p_{\text{FDR}} \le 3.0\times 10^{-4}$). Symmetric co-occurrence audit confirms non-directed Jaccard graph topology. |
| **Table 12** | Sliding-window Voter consensus performance for long sequences | `scripts/generate_paper_tables.py`<br>`scripts/run_long_voter.py` | `configs/default_species.yaml` | $N=4,360$ long NCBI contigs (`mapping_species_final.csv`) | Window $W=512$, stride $S=256$, mean logit pre-softmax averaging. Standard ResNet50: 72.69%. ConvNeXt: 3.49%--20.38%. Contigs $<512$ bp handled via end-padding with null vectors (`handle_short='pad'`, AD-010). Note 44 resolved: steep ConvNeXt drop explained by read length simulation bias and uncalibrated logit pulling. |
| **Figure 1 (Fig. 1)** | Comparative robustness analysis under controlled noise ($\epsilon$) | `scripts/generate_paper_figures.py`<br>`scripts/run_evaluation.py` | `configs/default_species.yaml` | Species accuracy logs across $\epsilon \in \{0, 0.02, 0.05, 0.10, 0.15\}$ | Publication vector/raster image generated in `paper/figures/fig_1.png`. Illustrates the steep collapse of standard ResNet50 versus the resilience of ConvNeXt_F. |
| **Figure 2 (Fig. 2)** | UMAP projections of latent representations at genus level | `scripts/generate_paper_figures.py`<br>`scripts/run_latent_evaluation.py` | `configs/default_genus.yaml` | Genus latent embeddings | High-resolution projection generated in `paper/figures/fig_2.png`. Documented as purely qualitative 2D visualization (AD-012, Note 25). |
| **Figure 3 (Fig. 3)** | Learned motif perturbation vs matched random control | `scripts/generate_paper_figures.py`<br>`scripts/run_motif_analysis.py` | `configs/default_species.yaml` | Filter perturbation evaluation logs | Generated in `paper/figures/fig_3.png` and subpanels. Panel (e) verified as distinct ConvNeXt noise-aware distribution, resolving historical asset duplication of ResNet50 noise (Note 39). Strict case-sensitive Linux naming applied. |
| **Figure 4 (Fig. 4)** | Association between RSCU background profile and motif triplets | `scripts/generate_paper_figures.py`<br>`scripts/run_motif_analysis.py` | `configs/default_species.yaml` | Background test triplet profile | Generated in `paper/figures/fig_4.png`. Displays non-overlapping frame 0 triplet frequencies versus motif frequencies with exploratory compositional framing. |
| **Figure 5 (Fig. 5)** | Triplet-composition similarity among strongly co-occurring motif pairs (top 200 edges) | `scripts/generate_paper_figures.py`<br>`scripts/run_motif_analysis.py` | `configs/default_species.yaml` | Jaccard co-occurrence network | Generated in `paper/figures/fig_5.png`. Top 200 undirected graph edges evaluated with symmetric orientation audit and permutation null model (AD-014, Note 43). |

---

## 3. Scientific Invariants and Textual Demarcations

1. **Title and Claim Moderation:**
   - The historical title asserted that 1D CNN representations *"enable robust"* classification. Because accuracy drops to 3.87% (ResNet50) and 17.38% (ConvNeXt) at $\epsilon=0.15$, the title was updated to:
     *“Base-level one-hot 1D convolutional networks for high-cardinality viral taxonomic classification: accuracy, corruption trade-offs, and learned sequence representations”*.
   - The Abstract now follows BMC structured guidelines with explicit `Background`, `Methods`, `Results`, and `Conclusions` sections.

2. **Demarcation of Dataset Partitioning (AD-007, S02/S03):**
   - The 80/10/10 split is strictly documented as a **fragment-level** split. No claims of generalization to unseen viral genomes or unobserved viral species are made.

3. **Demarcation of Long-Sequence Inference (AD-010, S06):**
   - The 72.69% accuracy of ResNet50 is framed as a proof-of-concept for sliding-window consensus on reference contigs from NCBI that share accessions with the Kraken2 training database (seen genomes).
   - The ConvNeXt degradation (3.49%--20.38%) is explained by read simulation length bias (500 reads for giant viruses vs 50 for small viruses), pre-softmax logit pulling, and calibration loss in transfer learning.

4. **Demarcation of Filter Motifs and Triplet Usage (AD-014, S10):**
   - Learned first-layer 7-mers are dominated by homopolymers (\texttt{CCCCCCC}, \texttt{GGGGGGG}). Paired GC-matched controls demonstrate that filter responses reflect local base composition and entropy rather than specific functional regulatory motifs.
   - RSCU calculations on unannotated fragments are explicitly designated as *Exploratory Trinucleotide Composition* rather than biological codon usage.

5. **Demarcation of Confidence vs. Calibration (AD-013, S09):**
   - Top-1 confidence and entropy are demarcated from formal probabilistic calibration. Under increasing noise, all models suffer from severe overconfidence (mean confidence remains high while accuracy drops precipitously), requiring ECE and Brier metrics with validation-adjusted temperature scaling.

6. **Prior Art Positioning (GeNet and MetageNN):**
   - Explicitly positions contributions beyond GeNet (Rojas-Carulla et al., 2019): scaling from genus to 13,683 viral species, controlled corruption with exact $3\epsilon/4$ substitution rates, ECE/Brier calibration, paired compositional controls, and windowed contig aggregation.

---

## 4. Availability of Data and Materials (AD-006)

- Source code, configurations, and test suites: open GitHub repository (`https://github.com/Mathe13/cnn_metagenome`).
- Large-scale Parquet splits and PyTorch `.pt` checkpoints reside in remote JupyterHub research cluster storage (AD-006). Local repository functions with deterministic synthetic fixtures and contracts, ensuring 100% reproducible execution without bulky GPU dependencies.
