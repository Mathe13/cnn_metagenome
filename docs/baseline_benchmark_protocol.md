# Baseline Benchmark Protocol: Kraken2, BLASTN, and Viral CNNs

## 1. Scope and Objective

This document establishes the unified, reproducible benchmarking protocol comparing viral 1D convolutional neural networks (ResNet50 and ConvNeXt variants) against standard metagenomic taxonomic baselines (Kraken2 and BLASTN), as well as an alignment-free $k$-mer frequency baseline.
The protocol governs both:
1. **Short-read fragments:** 512 bp fixed windows evaluated under controlled noise levels ($\epsilon \in \{0.0, 0.02, 0.05, 0.10, 0.15\}$).
2. **Long-sequence contigs and reads:** variable-length genomic contigs (512 bp to 10+ kb) evaluated via sliding-window consensus voting (`LongSequenceVoter`) and direct baseline querying under controlled noise.

---

## 2. Scientific Invariants and Denominator Accounting

### 2.1 Abstention and Denominator Rules

Metagenomic sequence search tools can abstain from making a species-level prediction:
- **Kraken2:** a sequence is labeled as unclassified (`status == 'U'`) when none of its $k$-mers match the database, or classified at the root level (`taxid == 1`) when matched $k$-mers exhibit no lower-rank consensus.
- **BLASTN:** a query yields no alignment hits if no local high-scoring segment pair (HSP) achieves the statistical significance or bitscore threshold.

In our closed-set evaluation ($N = 13,683$ species):
- Any abstention or unclassified query is assigned the sentinel label `-1`.
- **Inconditional Top-1 Accuracy:** The denominator is strictly the total number of evaluated queries $N_{\text{total}}$:
  $$\text{Accuracy} = \frac{1}{N_{\text{total}}} \sum_{i=1}^{N_{\text{total}}} \mathbb{I}(y_{\text{pred}}^{(i)} = y_{\text{true}}^{(i)})$$
  Because $y_{\text{true}}^{(i)} \ge 0$, any abstained query with $y_{\text{pred}}^{(i)} = -1$ automatically evaluates to $\mathbb{I} = 0$ (incorrect).
- **Coverage:** The proportion of queries receiving an explicit class prediction:
  $$\text{Coverage} = \frac{1}{N_{\text{total}}} \sum_{i=1}^{N_{\text{total}}} \mathbb{I}(y_{\text{pred}}^{(i)} \ge 0)$$
- **Conditional Accuracy:** The accuracy restricted exclusively to the covered subset:
  $$\text{Accuracy}_{\text{cond}} = \frac{\sum_{i: y_{\text{pred}}^{(i)} \ge 0} \mathbb{I}(y_{\text{pred}}^{(i)} = y_{\text{true}}^{(i)})}{\sum_{i=1}^{N_{\text{total}}} \mathbb{I}(y_{\text{pred}}^{(i)} \ge 0)}$$
- Filtering unclassified queries from the denominator inflates reported accuracy (confusing conditional accuracy with classification accuracy) and is explicitly prohibited when comparing against CNNs that output dense predictions across all classes.

---

## 3. Database Construction and Tool Parameters

### 3.1 Kraken2 Flat Taxonomy Mapping

Because the 13,683 species classes are defined as a closed set, mapping queries to the global NCBI hierarchical taxonomy introduces discrepancies due to historical taxid reassignments. We utilize a custom flat taxonomy:
- **Taxonomy DMP files:**
  - `nodes.dmp`: Root node `taxid = 1` (rank `no rank`). For each model species label $0 \le \text{label} < 13683$, a child node is defined with `taxid = label + 2` and parent `taxid = 1` (rank `species`).
  - `names.dmp`: Associates each `taxid = label + 2` with `class_{label}`.
- **Output Parsing:**
  - Status `'C'` with $2 \le \text{taxid} < 13685 \implies \text{label} = \text{taxid} - 2$.
  - Status `'U'` or $\text{taxid} < 2 \implies \text{label} = -1$.

### 3.2 BLASTN Tabular Ranking

- **Database Headers:** Subject reference FASTA sequences are formatted with headers:
  `>{seq_id}_label_{label}`
- **Execution Command:**
  ```bash
  blastn -db <DB_PATH> -query <QUERY_FASTA> -out <OUTPUT_TSV> \
         -outfmt "6 qseqid sseqid bitscore evalue pident length" \
         -max_target_seqs 5 -num_threads 16
  ```
- **Candidate Aggregation:**
  - Multiple HSPs for the same species label are grouped by maximum bitscore.
  - Ties are resolved deterministically by lowest E-value, then lowest integer label index.
  - The top candidate represents the Top-1 prediction; up to 5 unique candidate species are retained for Top-5 accuracy calculation.
  - Queries with no hits receive prediction label `-1` and an empty candidate list.

---

## 4. Sequence Overlap and Memorization Audit

### 4.1 Short Fragments (512 bp)

In the standard random fragment split (80/10/10), an audit via canonical sequence hashing (`sha256(min(seq.upper(), reverse_complement(seq.upper())))`) records forward/reverse-complement exact matches. It reports observed counts; the published split-level estimate is not copied into run outputs.
- **~95.89% of test fragments** have identical 512 bp sequences present in the training database (`has_overlap = True`).
- Only **~4.11% of test fragments** do not share an exact 512 bp sequence match (`no_exact_overlap`).

Evaluating BLASTN and Kraken2 on the full test set without stratification primarily measures **database lookup speed and exact sequence retrieval**, rather than generalizable taxonomic classification.
All reported metrics must be stratified across:
1. `all`: full query set (100%).
2. `overlap`: fragments present identically in training ($\text{train\_count} > 0$).
3. `no_exact_overlap`: fragments without exact match in training ($\text{train\_count} == 0$). Note that absence of an exact 512 bp match does not establish genome-level disjointness or absence of partial homology.

### 4.2 Long Sequences and Contigs

When evaluating long sequences (512 bp to 10+ kb, manuscript Section 3.5, Table 12):
- CNNs process sequences via the sliding-window consensus voter (`LongSequenceVoter`, window 512 bp, stride 256 bp) with pre-softmax logit averaging (`pred_mean_logits`) and majority voting (`pred_majority_vote`).
- Kraken2 and BLASTN evaluate the entire contig directly as a single query.
- **Reference Genome Overlap:** If a contig is derived from an accession or reference genome present in the training database, BLASTN achieves a full-length local alignment, and Kraken2 chains hundreds of exact $k$-mer matches.
- All long-read evaluations must stratify by:
  - `seen_genome`: contigs originating from reference genomes present in training.
  - `unseen_genome`: contigs originating from independent viral accessions.
  - Length bins: $[512, 2000)$, $[2000, 5000)$, and $\ge 5000$ bp.

---

## 5. Paired Comparative Methodology

Direct comparison between methods is only valid when evaluated on the **exact same query identifiers**:
- **$2 \times 2$ Contingency Discordance:**
  For methods $A$ and $B$, compute:
  - $n_{11}$: both correct ($\text{pred}_A = y \land \text{pred}_B = y$)
  - $n_{10}$: only $A$ correct ($\text{pred}_A = y \land \text{pred}_B \neq y$)
  - $n_{01}$: only $B$ correct ($\text{pred}_A \neq y \land \text{pred}_B = y$)
  - $n_{00}$: both incorrect ($\text{pred}_A \neq y \land \text{pred}_B \neq y$)
  - Discordant queries: $n_{10} + n_{01}$, and discordant proportion.
- **Species-Cluster Bootstrap:**
  Because fragments and reads from the same species are statistically correlated, simple query-level bootstrapping produces artificially narrow confidence intervals. We employ cluster bootstrapping:
  - True species labels $s \in \{1, \dots, S\}$ are resampled with replacement across 2,000 iterations (seed 42).
  - Accuracy differences $\Delta = \text{acc}_A - \text{acc}_B$ are weighted by species stratum size.
  - The 95% confidence interval is estimated from the 2.5% and 97.5% quantiles of the bootstrap distribution.

---

## 6. Execution Command and Reproduction

### 6.1 C03 paired short benchmark preparation

`c03_paired_v1` adds content-linked query materialization and strict per-method/per-noise manifests. The config `configs/baseline_c03.yaml` contains relative train/test paths and the 13,683 Species map. The default command is read-only and inventories inputs and all 35 required result conditions:

```bash
python3 scripts/prepare_baseline_benchmark.py --config configs/baseline_c03.yaml --mode inventory
```

Explicit export uses the full declared train and test partitions, selects at most 20 eligible test fragments per zero-based Species label by SHA-256 priority (seed 42), saves row/query identities and the clean overlap audit, then materializes one deterministic uniform A/C/G/T resample including the original base at each selected position for each nominal rate. The same generated query table and FASTA are the inputs recorded for all seven methods at that rate. Export creates an exclusive run under `results_reproduction/runs/<run_id>/`:

```bash
python3 scripts/prepare_baseline_benchmark.py --config configs/baseline_c03.yaml --mode export --run-id c03-export-001
```

The export also writes the exact train-fragment references and flat Kraken taxonomy. `historical_padding_to_N_v1` records the original tool representation (`rstrip("_").replace("_", "N")`); CNNs read the same materialized nucleotide realization from CSV, with padding/unknowns mapped to null tokens by the existing encoder. Build both search databases only after export:

```bash
python3 scripts/prepare_baseline_benchmark.py --config configs/baseline_c03.yaml \
  --mode build-databases --prepared-dir results_reproduction/runs/c03-export-001/c03-baseline-preparation/prepared
```

Run each method-condition on the shared query manifest. The corresponding commands and paths are emitted by `--mode plan`; CNN inference uses the registered 13,683-class checkpoint and exports Top-1/Top-5:

```bash
python3 scripts/run_c03_tool_baseline.py --method Kraken2-flat --queries <prepared>/queries_noise_00.csv \
  --fasta <prepared>/queries_noise_00.fasta --query-manifest <prepared>/query_manifest_noise_00.json \
  --reference-manifest <prepared>/references/reference_manifest.json --database <prepared>/references/kraken_db \
  --output-dir <prepared>/comparison_results
python3 scripts/run_c03_cnn_inference.py --variant resnet_clean_species --queries <prepared>/queries_noise_00.csv \
  --query-manifest <prepared>/query_manifest_noise_00.json --checkpoint <registered-checkpoint> \
  --reference-manifest <prepared>/preparation_manifest.json --output-dir <prepared>/comparison_results
```

An explicit `--mode plan` records input availability and tool paths without building databases or running inference. Execution outputs are admissible only with `COMPLETE` method-condition manifests that point to the matching query manifest and verify their output hash, class-order digest, reference, tool version, command, and abstention policy. Smoke outputs cannot pass the empirical loader. The current inventory is not evidence that the remote cluster artifacts exist.

The benchmark's principal noise protocol is uniform replacement including the original base (expected effective change rate `3ε/4`). Historical `NOISE_MODE="substitution"` changes every selected position; those results belong to a separate protocol and cannot be relabeled `c03_paired_v1`. Historical BLASTN `max_target_seqs=100` is a reference retrieval limit, separate from aggregation to at most five unique species candidates. `no_exact_overlap` means no exact forward/RC fragment in the declared training partition, not an unseen genome.

To reproduce baseline evaluations on either short fragments or long contigs:

```bash
# Short fragments evaluation:
python3 scripts/run_baseline_comparison.py \
    --regime short \
    --truth data/queries_truth.csv \
    --overlap-audit data/exact_overlap.csv \
    --predictions Kraken2=results/kraken_noise_00.out \
                  BLASTN=results/blast_noise_00.tsv \
                  ResNet50=results/predictions_resnet.parquet \
                  ConvNeXt_F=results/predictions_convnext_f.parquet \
    --output-dir results/comparison_short \
    --bootstrap-repeats 2000 \
    --seed 42

# Long reads evaluation:
python3 scripts/run_baseline_comparison.py \
    --regime long \
    --truth data/long_queries_truth.csv \
    --genome-overlap-col seen_genome \
    --length-col length \
    --predictions Voter_mean=results/long_voter_mean.parquet \
                  Voter_majority=results/long_voter_majority.parquet \
                  Kraken2=results/long_kraken.out \
                  BLASTN=results/long_blast.tsv \
    --output-dir results/comparison_long \
    --bootstrap-repeats 2000 \
    --seed 42
```

