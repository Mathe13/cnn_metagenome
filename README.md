# Base-Level One-Hot 1D Convolutional Networks for High-Cardinality Viral Taxonomic Classification

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Reproducibility: Spec-Driven](https://img.shields.io/badge/reproducibility-tlc--spec--lean-brightgreen.svg)](#scientific-reproducibility--contracts)

Official research code and reproducibility framework for the manuscript:

> **"Base-level one-hot 1D convolutional networks for high-cardinality viral taxonomic classification: accuracy, corruption trade-offs, and learned sequence representations"**  
> *(Working title: "Spatial one-hot nucleotide representations of DNA enable robust and interpretable taxonomic classification with 1D convolutional networks")*  
> **Authors:** Matheus de Lima$^1$, Ulisses Nunes da Rocha$^2$, Márcio Dorn$^{1,3*}$  
> $^1$ Structural Bioinformatics and Computational Biology Lab, Institute of Informatics, Universidade Federal do Rio Grande do Sul (UFRGS), Brazil  
> $^2$ Department of Computational Biology and Chemistry, Helmholtz Centre for Environmental Research (UFZ), Leipzig, Germany  
> $^3$ Center for Biotechnology, Universidade Federal do Rio Grande do Sul (UFRGS), Brazil  
> $^*$ Corresponding author: `mdorn@inf.ufrgs.br`

---

## Table of Contents

- [Overview](#overview)
- [Scientific Invariants & Disclaimers](#scientific-invariants--disclaimers)
- [Repository Structure](#repository-structure)
- [Environment & Configuration](#environment--configuration)
  - [Dual-Mode Architecture (Remote Cluster vs. Local Smoke Test)](#dual-mode-architecture-ad-017)
  - [Installation & Environment Setup](#installation--environment-setup)
- [Quickstart & CLI Pipelines](#quickstart--cli-pipelines)
  - [Single Sequence Inference](#1-single-sequence-inference)
  - [Model Evaluation & Noise Robustness](#2-model-evaluation--noise-robustness)
  - [Calibration & Confidence Audit](#3-calibration--confidence-audit)
  - [Kraken2 and BLASTN Baseline Comparisons](#4-kraken2-and-blastn-baseline-comparisons)
  - [First-Layer Motif & Triplet Analysis](#5-first-layer-motif--triplet-analysis)
  - [Latent Space & Embedding Analysis](#6-latent-space--embedding-analysis)
  - [Long Sequence Windowed Inference](#7-long-sequence-windowed-inference)
  - [Regenerating Paper Tables & Figures](#8-regenerating-paper-tables--figures)
  - [Compiling Manuscript Documents](#9-compiling-manuscript-documents)
- [Reproduction Notebooks](#reproduction-notebooks)
- [Testing & Quality Assurance](#testing--quality-assurance)
- [Architecture Decision Records (ADRs)](#architecture-decision-records-adrs)
- [Citation](#citation)

---

## Overview

High-cardinality viral taxonomic classification presents major challenges due to extreme class cardinality, rapid mutation rates, and sequencing noise. This repository evaluates 1D adaptations of **ResNet50** and **ConvNeXt** architectures trained directly on 512-bp nucleotide windows with spatial one-hot encoding under a closed-set viral taxonomy:

- **Taxonomic Cardinality:** 13,683 viral species and 2,702 genera derived from curated NCBI, Kraken2, and ICTV reference databases.
- **Nucleotide Corruption:** Systematic robustness evaluation across nominal Bernoulli substitution noise rates $\epsilon \in \{0, 0.02, 0.05, 0.10, 0.15\}$, corresponding to an effective substitution rate of $3\epsilon/4$ (up to 11.25% effective nucleotide replacement).
- **Architecture Trade-offs:** Standard ResNet50 attains the highest accuracy on pristine 512-bp fragments (84.88% Top-1, 84.12% Macro-F1), while noise-aware ConvNeXt ($\text{ConvNeXt}_F$, fine-tuned at $\epsilon=0.05$) demonstrates superior robustness under extreme sequence degradation (17.38% Top-1 vs. 3.87% for noise-aware ResNet50 at $\epsilon=0.15$).
- **Interpretability & Sequence Patterns:** Characterization of first-layer convolutional filters, paired GC/complexity controls for 7-mer motif detection, and exploratory triplet usage analysis.
- **Long-Sequence Windowed Inference:** Sliding-window consensus inference (512-bp window, 256-bp stride) evaluated with pre-softmax logit aggregation and majority voting across long NCBI contigs and synthetic reads.
- **Baselines:** Rigorous comparison with standard alignment and $k$-mer tools (Kraken2 and BLASTN) with strict denominator tracking and exact training overlap stratification.

---

## Scientific Invariants & Disclaimers

In accordance with [`AGENTS.md`](file:///home/matheus/Documents/cnn_metagenome/AGENTS.md) and the formal project Architecture Decision Records ([`docs/adr/`](file:///home/matheus/Documents/cnn_metagenome/docs/adr)), all code and experiments in this repository adhere to the following non-negotiable contracts:

1. **Closed-Set Taxonomic Classification (AD-001):**  
   All evaluated species (13,683) and genera (2,702) are present in the training reference database. This framework performs closed-set classification and does not evaluate open-set novelty detection or novel taxon discovery.
2. **Fragment-Level Split vs. Genome Disjointness (AD-007):**  
   The primary evaluation split is stratified by 512-bp **sequence fragments** (80% train, 10% validation, 10% test; $N = 1,706,270$ test fragments), **not** partitioned by whole genomes. In viral metagenomics, many species contain only a single reference genome, resulting in ~95.89% exact sequence overlap between train and test fragments. Consequently, test accuracy metrics measure fragment classification within known viral reference sequences, **not generalization to unseen viral genomes**.
3. **Spatial One-Hot Encoding Contract (AD-003):**  
   Input sequences are strictly 512 bp encoded into 4 channels in order **A, C, G, T** (indices 0, 1, 2, 3). Ambiguous bases (e.g., `N`, `R`, `Y`) and zero-padding are mapped to index 4, which is converted to an all-zeros null vector by the `OneHotEmbedding` layer.
4. **Noise Model (AD-009):**  
   Synthetic substitution noise uniformly re-samples each mutated position across $\{A, C, G, T\}$ including the original base. Thus, nominal noise parameter $\epsilon$ produces an **expected effective mutation rate of $3\epsilon/4$**. This is an idealized uniform substitution benchmark and does not model ONT-specific indel profiles, context dependencies, or homopolymer run compression.
5. **Pre-Classifier Logit Aggregation (AD-010):**  
   For long sequences, consensus classification averages **unnormalized logits before softmax** (rather than averaging post-softmax probabilities) to prevent high-confidence erroneous windows from distorting the sequence-level prediction.
6. **Strict Denominator in Baselines (AD-011):**  
   Kraken2 and BLASTN comparisons maintain the exact total query count $N$ in the denominator; queries abstained as *unclassified* or lacking hits are recorded as misclassifications ($\text{label} = -1$) to guarantee valid statistical comparisons.

---

## Repository Structure

```text
cnn_metagenome/
├── AGENTS.md                  # Scientific invariants, guidelines, and reproduction rules
├── README.md                  # This entrypoint document
├── ROADMAP_SPECS_ARTIGO.md    # Specification roadmap for reproduction tasks
├── ROADMAP_REPRODUCAO_NOTEBOOKS.md # JupyterHub reproduction roadmap
├── env.example                # Template for environment configuration variables
├── configs/                   # Declarative YAML experiment configurations
│   ├── default_species.yaml   # Species-level hyperparameters (13,683 classes)
│   ├── default_genus.yaml     # Genus-level hyperparameters (2,702 classes)
│   ├── models.yaml            # Model registry and checkpoint signatures
│   └── controlled/            # Strict controlled architecture comparison configs
├── src/viral_cnn/             # Core modular Python package
│   ├── data/                  # Encoding (A/C/G/T), tokenization, Parquet data loaders
│   ├── models/                # 1D-ResNet50, 1D-ConvNeXt, OneHotEmbedding, ModelRegistry
│   ├── eval/                  # Metrics (Top-k, Macro-F1), noise injection, calibration
│   ├── inference/             # Forward runner, long sequence voter, sliding window
│   ├── interpretability/      # 1D-GradCAM, first-layer motifs, paired GC/complexity controls
│   ├── baselines/             # Kraken2/BLASTN evaluation, exact overlap auditing
│   ├── taxonomy/              # Taxonomic mappings (Species/Genus/Order) and index conversions
│   ├── reporting/             # Manuscript tables (1-12) and figures (1-5) generation
│   ├── training/              # Training loop, AMP, optimizer/scheduler parity
│   ├── utils/                 # Path logging ([LOG_PATH]), file caching, manifest helpers
│   └── notebook_runtime.py    # Remote server resolution and automated execution manifests
├── notebooks/
│   ├── reproduction/          # Modern, reproducible, self-contained reproduction notebooks
│   │   ├── 00_regenerate_all_paper_artifacts.ipynb # End-to-end pipeline consolidation
│   │   ├── short/             # 512-bp fragment notebooks (01 to 11)
│   │   └── long/              # Long contig and voter notebooks (01 to 04)
│   └── original/              # Historical notebooks (short/ and long/) preserved untouched
├── scripts/                   # Production CLI entrypoints
│   ├── run_inference.py       # Single-sequence and FASTA forward inference
│   ├── run_evaluation.py      # Batch evaluation under clean and noisy conditions
│   ├── run_calibration_evaluation.py # Calibration, ECE, Brier score, and entropy audit
│   ├── run_baseline_comparison.py    # Kraken2 & BLASTN overlap-stratified benchmark
│   ├── run_motif_analysis.py  # 1st-layer 7-mer motif extraction & triplet usage audit
│   ├── run_latent_evaluation.py      # Latent space embedding, KNN (k=5), Silhouette, UMAP
│   ├── run_long_voter.py      # Sliding-window consensus inference on long sequences
│   ├── generate_paper_tables.py      # Regenerate LaTeX/CSV tables 1-12
│   ├── generate_paper_figures.py     # Regenerate publication figures 1-5
│   └── compile_paper.py       # LaTeX manuscript compilation (submission and review)
├── tests/                     # Unit and scientific contract test suite
│   ├── test_encoding.py       # A/C/G/T encoding, null vector padding
│   ├── test_noise_and_evaluation.py  # 3ε/4 mutation rate, bincount F1 calculation
│   ├── test_model_registry.py # Model weight loading, class count validation
│   ├── test_baselines.py      # Overlap auditing, Kraken2/BLASTN parsing
│   └── fixtures/              # Deterministic synthetic fixtures for offline testing
├── docs/                      # Architectural audits, protocol specs, and data manifests
│   ├── adr/                   # Architecture Decision Records (AD-001 through AD-018)
│   ├── inventory.md           # Full artifact, dataset, and notebook inventory
│   ├── dataset_manifest.md    # Description of raw and processed Parquet splits
│   └── model_registry.md      # Checkpoint architecture and verification details
└── paper/                     # Manuscript LaTeX source, bibliography, and compiled PDF
```

---

## Environment & Configuration

### Dual-Mode Architecture (AD-017)

To decouple lightweight code development from multi-gigabyte genomic datasets and cluster checkpoints, the runtime operates in two mutually compatible modes:

| Mode | Environment Flag | Target Infrastructure | Data / Checkpoints | Execution Speed |
|---|---|---|---|---|
| **Remote Cluster Mode** | `REMOTE_SERVER=true` | JupyterHub / Slurm GPU Cluster | Multi-GB Parquet partitions & `.pt` weights | Full scientific runs (minutes to hours) |
| **Local Smoke-Test Mode** | `REMOTE_SERVER=False` or `SMOKE_TEST=1` | Local workstation / CI runner (CPU) | Deterministic synthetic in-memory fixtures | Instant verification (seconds) |

### Installation & Environment Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-org/cnn_metagenome.git
   cd cnn_metagenome
   ```

2. **Create and activate a Python virtual environment:**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install core dependencies:**
   ```bash
   pip install --upgrade pip
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu # or cu118/cu121 for GPU
   pip install pyarrow pandas numpy scipy scikit-learn matplotlib seaborn pyyaml tqdm
   ```

4. **Configure environment variables:**
   Copy the provided example environment template:
   ```bash
   cp env.example .env
   ```
   Edit `.env` to configure your data paths and execution mode:
   ```bash
   # Enable cluster execution (set to False for local synthetic testing)
   REMOTE_SERVER=False

   # Root directory containing Parquet splits and checkpoints (cluster mode)
   DATA_ROOT=/home/mlima/workspace

   # Device configuration ('cuda' or 'cpu')
   DEVICE=cpu

   # Traceability: log all artifact loads ([LOG_PATH]) to console
   LOG_PATH=true
   ```

---

## Quickstart & CLI Pipelines

All pipelines can be run via standalone scripts in the [`scripts/`](file:///home/matheus/Documents/cnn_metagenome/scripts) directory. In the absence of heavy checkpoints, all scripts automatically support the `--synthetic-test` flag or initialize random deterministic weights to allow end-to-end structural testing.

### 1. Single Sequence Inference

Run clean forward inference on a raw nucleotide sequence or FASTA file:

```bash
# Evaluate an inline 512-bp sequence
python3 scripts/run_inference.py \
  --sequence "ACGTACGTACGT..." \
  --model-variant resnet_clean_species \
  --device cpu

# Evaluate sequences from a FASTA file
python3 scripts/run_inference.py \
  --fasta sample.fasta \
  --model-variant convnext_transfer_species \
  --device cpu
```

### 2. Model Evaluation & Noise Robustness

Evaluate a model variant across nominal noise rates ($\epsilon \in \{0, 0.02, 0.05, 0.10, 0.15\}$):

```bash
python3 scripts/run_evaluation.py \
  --input dataset_split_parquet/test.parquet \
  --model-variant resnet_clean_species \
  --noise 0.05 \
  --batch-size 512 \
  --output-dir results_reproduction/eval_resnet_noise05
```

### 3. Calibration & Confidence Audit

Compute multiclass Expected Calibration Error (ECE), Brier score, Negative Log-Likelihood (NLL), and confidence distributions:

```bash
python3 scripts/run_calibration_evaluation.py \
  --model-variant resnet_clean_species \
  --data-dir dataset_split_parquet/ \
  --output-dir results_reproduction/calibration \
  --synthetic-test
```

### 4. Kraken2 and BLASTN Baseline Comparisons

Audit exact sequence overlap and compute paired metrics against standard taxonomic classification tools:

```bash
python3 scripts/run_baseline_comparison.py \
  --test-data dataset_split_parquet/test.parquet \
  --train-data dataset_split_parquet/train.parquet \
  --kraken-report kraken2_output.report \
  --blast-results blastn_output.tsv \
  --output-dir results_reproduction/baselines
```

### 5. First-Layer Motif & Triplet Analysis

Extract first-layer 7-mer convolutional filters and evaluate significance using paired GC-matched and complexity-matched background controls:

```bash
python3 scripts/run_motif_analysis.py \
  --model-variant resnet_clean_species \
  --control-policy gc_matched \
  --n-repeats 10 \
  --output-dir results_reproduction/motifs \
  --synthetic-test
```

### 6. Latent Space & Embedding Analysis

Extract pre-classifier representations, evaluate metric clustering with Cosine KNN ($k=5$, 80/20 train/test reference split) and Silhouette scores, and project via UMAP:

```bash
python3 scripts/run_latent_evaluation.py \
  --model-variant resnet_clean_species \
  --output-dir results_reproduction/latent \
  --synthetic-test
```

### 7. Long Sequence Windowed Inference

Run consensus inference over long genomic sequences (512-bp window, 256-bp stride) using pre-softmax logit averaging:

```bash
python3 scripts/run_long_voter.py \
  --fasta-file long_viral_contigs.fasta \
  --model-variant resnet_clean_species \
  --window-size 512 \
  --stride 256 \
  --aggregation mean_logits \
  --output-dir results_reproduction/long_voter
```

### 8. Regenerating Paper Tables & Figures

Regenerate all publication artifacts from reproduction results:

```bash
# Generate LaTeX and CSV tables (Tables 1 through 12)
python3 scripts/generate_paper_tables.py --output-dir paper/tables

# Generate publication figures (Figures 1 through 5)
python3 scripts/generate_paper_figures.py --output-dir paper/figures
```

### 9. Compiling Manuscript Documents

Compile both review (with reviewer macro callouts) and clean submission LaTeX documents:

```bash
python3 scripts/compile_paper.py --target both
```

---

## Reproduction Notebooks

All reproduction notebooks are stored in [`notebooks/reproduction/`](file:///home/matheus/Documents/cnn_metagenome/notebooks/reproduction). Each notebook begins with an autonomous path injection block that discovers the project root and adds `src/` to `sys.path`, enabling immediate execution without requiring local package installation:

```python
import sys
from pathlib import Path

current_dir = Path.cwd().resolve()
for r in [current_dir, current_dir.parent, current_dir.parent.parent, current_dir.parent.parent.parent]:
    if (r / "src" / "viral_cnn").exists():
        if str(r / "src") not in sys.path:
            sys.path.insert(0, str(r / "src"))
        break
```

### Short Fragment Notebooks (`notebooks/reproduction/short/`)

| Notebook | Scientific Scope | Manuscript Deliverables |
|---|---|---|
| `01_benchmark_clean_and_noise.ipynb` | Profiling, clean accuracy, and noise degradation ($\epsilon \in [0, 0.15]$) | **Tables 1, 2, 3, 4, Fig. 1** |
| `02_analyze_confidence_and_calibration.ipynb` | Overconfidence audit, multiclass ECE, entropy by rank | **Table 4, Table 7, Fig. 1** |
| `03_finetune_resnet50_species.ipynb` | Fine-tuning ResNet50 with $\epsilon=0.05$ substitution noise | Support for **Tables 1, 3, 4** |
| `04_finetune_convnext_species.ipynb` | Fine-tuning ConvNeXt with $\epsilon=0.05$ substitution noise | Support for **Tables 1, 3, 4** |
| `05_ictv_taxonomy_curation.ipynb` | ICTV Master Species List (MSL41) & NCBI cross-mapping | Taxonomic mappings |
| `06_ictv_latent_space_and_hierarchy.ipynb` | High-dimensional embeddings, Cosine KNN ($k=5$), Silhouette, UMAP | **Tables 5, 6, Fig. 2** |
| `07_first_layer_motifs_and_perturbation.ipynb` | 1st-layer 7-mers, paired GC-matched & random controls (Wilcoxon FDR) | **Table 8, Fig. 3** |
| `08_triplet_composition_and_cooccurrence.ipynb` | Exploratory triplet composition, relative synonymous codon usage, co-occurrence | **Tables 9, 10, 11, Figs. 4, 5** |
| `09_benchmark_cnns_kraken2_blast_overlap.ipynb` | Paired baseline benchmark, exact training overlap stratification | **Baseline benchmarks** |
| `10_decisao_retreinamento_por_genoma.ipynb` | Feasibility audit of genome-disjoint taxonomic splits | **AD-007 Audit** |
| `11_auditoria_taxonomia_ictv_split.ipynb` | Verification of ICTV/NCBI taxonomy reconciliation integrity | Taxonomic audit |

### Long Sequence Notebooks (`notebooks/reproduction/long/`)

| Notebook | Scientific Scope | Manuscript Deliverables |
|---|---|---|
| `01_ncbi_download_and_prepare_database.ipynb` | Downloading NCBI viral accessions and windowed chunking | Long sequence database |
| `02_cnn_long_voter_inference.ipynb` | Windowed inference (512 bp, 256 bp stride) on NCBI long reads/contigs | **Table 12** |
| `03_cnn_long_voter_compare.ipynb` | Consensus comparison, pre-softmax logit pulling vs. majority voting | **Table 12, contig figures** |
| `04_long_motifs_and_gc_analysis.ipynb` | Long-read GC distribution, motif consistency across full contigs | Long-read motif audit |

---

## Testing & Quality Assurance

Unit tests verify all core scientific contracts and algorithmic implementations offline without requiring external network access or GPU clusters.

Run the complete test suite:
```bash
python3 -m unittest discover tests
```

Run a specific test module:
```bash
python3 -m unittest tests/test_encoding.py
python3 -m unittest tests/test_noise_and_evaluation.py
python3 -m unittest tests/test_baselines.py
```

### Verified Scientific Contracts

- **Encoding Order & Ambiguity:** Verifies that A/C/G/T strictly maps to channels 0/1/2/3 and that ambiguous nucleotides (`N`, `R`, `Y`) and padding produce null vectors.
- **Expected Mutation Rate:** Proves empirically and analytically that uniform resampling over $\{A, C, G, T\}$ achieves an effective substitution rate of $3\epsilon/4$.
- **Disjoint Worker Partitioning:** Ensures that `ParquetBatchIterableDataset` splits Parquet file chunks disjointly across multiple DataLoader workers without sample duplication or omission.
- **Pre-Classifier Logits:** Verifies that consensus voters compute mean unnormalized logits prior to softmax transformation.
- **Exact Overlap Deduplication:** Validates that input sequences with duplicate column schemas are sanitized and audited correctly against training reference dictionaries.

---

## Architecture Decision Records (ADRs)

Key architectural and methodological decisions are formally documented in [`docs/adr/`](file:///home/matheus/Documents/cnn_metagenome/docs/adr):

- **AD-001:** Closed-set viral taxonomy with 13,683 Species and 2,702 Genera.
- **AD-002:** 1-based CSV ID to 0-based model label offset mapping (`model_label = id - 1`).
- **AD-003:** Strict 4-channel spatial one-hot encoding with null vector padding.
- **AD-004:** Preservation of historical notebooks in original locations (`short/` and `long/`).
- **AD-005:** Synthetic contract verification for offline testing without cluster weights.
- **AD-006:** Large Parquet partitions and `.pt` checkpoints reside on cluster storage.
- **AD-007:** Genome-disjoint split requires $\ge 2$ genomes per species; fragment split represents fragment classification.
- **AD-008:** Canonical model registry and checkpoint head dimension validation.
- **AD-009:** Controlled noise protocol ($3\epsilon/4$ effective rate) and vectorized F1 calculation.
- **AD-010:** Long voter audit: logit pulling and length bias analysis in ConvNeXt.
- **AD-011:** Baseline benchmark protocol for Kraken2 and BLASTN with exact overlap stratification.
- **AD-012:** Latent space evaluation: Cosine KNN ($k=5$) without self-neighbors and sample-constrained Silhouette score.
- **AD-013:** Calibration audit: multiclass ECE, Brier score, and overconfidence under noise.
- **AD-014:** GC-matched and complexity-matched paired controls for 7-mer motif discovery.
- **AD-015:** Unified table/figure regeneration pipeline and dual manuscript compilation.
- **AD-016:** Controlled architecture comparison protocol (ResNet50 vs. ConvNeXt).
- **AD-017:** Remote server topology resolution (`REMOTE_SERVER`, `DATA_ROOT`) and `SMOKE_TEST` fallback.
- **AD-018:** GPU pipeline optimization: non-blocking batch transfer, device-side noise injection, and atomic checkpointing.

---

## Citation

If you use this codebase, models, or methodology in your research, please cite:

```bibtex
@article{lima2026viralcnn,
  title={Base-level one-hot 1D convolutional networks for high-cardinality viral taxonomic classification: accuracy, corruption trade-offs, and learned sequence representations},
  author={de Lima, Matheus and da Rocha, Ulisses Nunes and Dorn, M{\'a}rcio},
  journal={BioData Mining},
  year={2026},
  publisher={BioMed Central}
}
```

---

## License

This project is licensed under the MIT License.
