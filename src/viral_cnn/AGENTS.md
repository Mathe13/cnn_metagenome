# Agent Instructions — Core Modules (`src/viral_cnn`)

## 1. Scope and Core Purpose

The `src/viral_cnn/` package is the decoupled, production-grade Python library powering all training, inference, evaluation, interpretability, baseline benchmarking, and artifact reporting in this repository.

When working within `src/viral_cnn/`, you are operating on the central scientific engine of the project. Any code written here must be reusable across both CLI scripts (`scripts/`) and reproduction notebooks (`notebooks/reproduction/`), adhering strictly to the scientific invariants established in the published manuscript and project Architecture Decision Records ([`docs/adr/`](file:///home/matheus/Documents/cnn_metagenome/docs/adr)).

---

## 2. Non-Negotiable Scientific Contracts

Every agent modifying or creating code in `src/viral_cnn/` MUST uphold the following contracts:

### 2.1 Spatial 4-Channel Encoding (`viral_cnn.data.encoding`) — AD-003
- **Order:** Channel 0 is **A**, Channel 1 is **C**, Channel 2 is **G**, Channel 3 is **T**.
- **Padding & Ambiguity:** Any ambiguous base (`N`, `R`, `Y`, `S`, `W`, `K`, `M`, `B`, `D`, `H`, `V`) and zero-padding must map to token index `4`.
- **Null Vector:** Token index `4` must produce an all-zeros null vector $\mathbf{0} \in \mathbb{R}^4$ in `OneHotEmbedding`. Never assign a fifth one-hot channel or non-zero embedding to ambiguous or padded positions.

### 2.2 Taxonomy & Label Offset (`viral_cnn.taxonomy.mapping`) — AD-001 & AD-002
- **Cardinality:** The closed-set taxonomy defines exactly **13,683 Species** (`species_name_ids.csv`) and **2,702 Genera** (`genus_name_ids.csv`).
- **Offset Formula:** CSV databases store 1-based IDs $[1, K]$. Neural network output logits are 0-based indices $[0, K-1]$. Always enforce:
  $$\text{model\_label} = \text{id} - 1 \quad \iff \quad \text{id} = \text{model\_label} + 1$$
- Never swap or re-index classes. Checkpoint weights, taxonomy lookup tables, and metric reports must share the exact same alphabetical class ordering.

### 2.3 Substitution Noise Protocol (`viral_cnn.eval.noise`) — AD-009
- When applying nominal substitution noise $\epsilon \in [0, 1]$, each selected nucleotide is re-sampled uniformly from $\{A, C, G, T\}$ *including the original nucleotide*.
- **Effective Mutation Rate:** The expected effective substitution rate is strictly $3\epsilon/4$ (e.g., $\epsilon=0.05 \implies 3.75\%$ effective mutations; $\epsilon=0.15 \implies 11.25\%$).
- Do not add insertions or deletions to this benchmark.

### 2.4 Pre-Softmax Logit Aggregation (`viral_cnn.inference.voter`) — AD-010
- When aggregating windowed predictions across long contigs or reads, average the **raw unnormalized logits prior to softmax** (`mean_logits`), rather than averaging post-softmax probabilities. This prevents single overconfident windows from distorting sequence-level classifications.

### 2.5 Strict Baseline Denominators (`viral_cnn.baselines`) — AD-011
- In comparisons against Kraken2 and BLASTN, all $N$ test queries must remain in the denominator. Unclassified queries or hits lacking valid taxonomy mappings must be marked with label `-1` (misclassification). Never discard unclassified queries when reporting Top-1/Top-5 accuracy.

### 2.6 Latent Space Distance & Exclusions (`viral_cnn.eval.latent`) — AD-012
- Features must be extracted from the penultimate pooling layer immediately prior to the linear classification head.
- KNN evaluation must use $k=5$, $L_2$-normalized embeddings, cosine distance, and a stratified 80/20 train/test reference split with **strict self-neighbor exclusion**.

---

## 3. Submodule Dependency Graph & Layering

To prevent circular dependencies and architectural degradation, modules in `src/viral_cnn/` must strictly respect the following Directed Acyclic Graph (DAG):

```text
Level 0 (Foundation):
  └── viral_cnn.utils (logging_utils)

Level 1 (Data & Domain Primitives):
  ├── viral_cnn.data.encoding
  └── viral_cnn.taxonomy.mapping, curation

Level 2 (Data Access & Model Architectures):
  ├── viral_cnn.data (parquet_dataset, grouped_split, audit)
  └── viral_cnn.models (resnet, convnext, builder, registry)

Level 3 (Evaluation, Inference, & Analysis):
  ├── viral_cnn.eval (metrics, noise, runner, calibration, latent, confusion)
  ├── viral_cnn.inference (engine, voter, audit_long)
  ├── viral_cnn.interpretability (motifs, gradcam)
  └── viral_cnn.baselines (overlap, paired, taxonomy, kmer_baseline, long_eval)

Level 4 (Workflow & Management):
  ├── viral_cnn.training (trainer, profiler, config)
  └── viral_cnn.notebook_runtime

Level 5 (Reporting & Presentation):
  └── viral_cnn.reporting (tables, figures, manuscript, controlled_audit)
```

### Dependency Rules:
1. **Never import higher-level modules into lower-level modules** (e.g., `viral_cnn.models` must never import from `viral_cnn.eval` or `viral_cnn.reporting`).
2. **Zero notebook dependencies:** Never import from `short/`, `long/`, or `notebooks/`.
3. **No wildcard imports:** `from module import *` is strictly forbidden.
4. **Isolated imports:** Use explicit function and class imports.

---

## 4. Performance & High-Throughput GPU Rules (AD-018)

When implementing or modifying training and evaluation pipelines:

1. **No Per-Batch CPU Synchronization:**
   - Never call `.item()`, `.tolist()`, or `.cpu().numpy()` inside batch iteration loops.
   - Accumulate loss, correct Top-1/Top-5 counts, and confusion matrices directly on the GPU using `torch.bincount` and tensor accumulators.
2. **Device-Side Noise Injection:**
   - Perform nucleotide token transfers using `tensor.to(device, non_blocking=True)`.
   - Apply Bernoulli masking and nucleotide mutation directly on the GPU tensor before the forward pass.
3. **FP32 Conversion for Entropy & Normalization:**
   - When using Automatic Mixed Precision (AMP), activations in `TaxonomyEntropyAnalyzer` must be cast to `torch.float32` before computing $L_1$ norms and Shannon entropy. Failure to do so causes FP16 numerical overflow and yields `NaN`.
4. **Atomic Checkpointing:**
   - When saving `.pt` weights, write to a temporary file first (`.pt.tmp`) and atomically rename it via `os.replace` to prevent corrupted checkpoints upon cluster job interruption.

---

## 5. Observability Contract (`LOG_PATH`)

Every function that loads an external or persistent artifact (model weights, Parquet partition, CSV taxonomy, or configuration file) MUST call `log_artifact_path` from `viral_cnn.utils.logging_utils`:

```python
from viral_cnn.utils.logging_utils import log_artifact_path

# Example pattern:
log_artifact_path("model checkpoint", checkpoint_path)
log_artifact_path("taxonomy mapping", csv_path, extra=f"rank={rank}")
```

This ensures full auditability across console logs and JupyterHub cell executions.

---

## 6. Offline Testability & Synthetic Contracts (AD-005, AD-017)

1. **No External Network Calls:** Modules in `src/viral_cnn/` must never make unmocked HTTP/FTP requests during instantiation or standard imports.
2. **Synthetic Fallbacks:** Components handling heavy data or checkpoints must support in-memory synthetic execution (e.g., `read_parquet_sample`, `synthetic_test=True`, or random deterministic weight initialization when `checkpoint_path` is omitted).
3. **Unit Tests:** Any new function or bug fix in `src/viral_cnn/` must be covered by a corresponding unit test in `tests/test_<module>.py` that runs in $<1$ second on CPU.

---

## 7. Agent Verification Checklist

Before completing any task modifying `src/viral_cnn/`:

- [ ] **Type Annotations & Docstrings:** All public functions, classes, and methods have full type hints and Google/NumPy-style docstrings.
- [ ] **Contract Integrity:** A/C/G/T order, 0-based label offset, and $3\epsilon/4$ noise rate are strictly preserved.
- [ ] **Observability:** `log_artifact_path` is invoked when loading files or weights.
- [ ] **Unit Tests:** Execute `.venv/bin/python3 -m unittest discover tests` (or the specific test file) and ensure all tests pass cleanly.
- [ ] **Import Cleanliness:** Verify no circular imports were introduced (`python3 -c "import viral_cnn"` succeeds instantly).

