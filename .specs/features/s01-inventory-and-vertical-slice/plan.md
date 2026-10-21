# S01: Inventory, Organization, and Vertical Slice Extraction

## Problem

The repository contains 15 unindexed notebooks split between `short/` and `long/` that mix exploratory data prep, model training, evaluation, and plotting.
Code relies on hardcoded paths (`/home/mlima/...`, `../cnn_kraken_species/...`), lacks an installable Python package structure, and currently has no model checkpoints (`.pt`) or preprocessed Parquet splits on disk.
Anyone trying to evaluate or reproduce a result must manually decipher undocumented cell dependencies across large notebooks without knowing which notebook corresponds to which manuscript table or figure.

When this ships, all historical notebooks and artifacts are systematically mapped to the paper's tables and figures, the core encoding and model definitions are extracted into a clean modular package (`src/viral_cnn/`), and a working CLI vertical slice (`scripts/run_inference.py`) executes 512 bp DNA encoding and clean forward inference with proof-backed deterministic contract tests.

## Flow

Reuses `short/utils/models.py:OneHotEmbedding`, `short/utils/models.py:resnet50_1d`, and `short/utils/models.py:convnext_tiny_1d` definitions while preserving the historical originals in place.

1. input DNA string or FASTA -> `viral_cnn.data.encoding` (new, no door - placement per conventions) - validates characters, applies 512 bp length padding/truncation, maps A/C/G/T to channels 0..3 and pad/ambiguous to null vector
2. `viral_cnn.models.builder` (new, no door - placement per conventions) - instantiates ResNet50-1D or ConvNeXt-1D with specified `num_classes`
3. `viral_cnn.inference.engine` (new, no door - placement per conventions) - computes logits, applies softmax, extracts Top-1 and Top-5 predictions
4. out: JSON report with `predicted_class_index`, `top5_class_indices`, and `probabilities` to stdout (door 4)

## Impact

| Front | What changes |
| --- | --- |
| domain | new term: `SpatialOneHot` - 4-channel tensor encoding of DNA where A=0, C=1, G=2, T=3 and any pad/ambiguous nucleotide produces an all-zero vector `[0.0, 0.0, 0.0, 0.0]`, lives in `src/viral_cnn/data` |
| domain | existing term: `model_label` meant integer class index in historical notebooks, now explicitly codified as 0-based index with `model_label = taxonomic_id - 1` |
| stored data | nothing to migrate: historical notebooks and CSV datasets in `dataset_kraken/` remain read-only; new code outputs JSON and structured metrics |

## Relations

None - no stored-data shape change

## Surface

None - nothing consumed outside

## Landing

| One-way door | Literal shape | Alternative rejected |
| --- | --- | --- |
| modular core package namespace | `src/viral_cnn/` containing `models/`, `data/`, `inference/`, and `taxonomy/` | in-place modification of historical `short/utils/` - rejected because altering files in-place violates reproducibility and obscures baseline differences |
| DNA 4-channel spatial encoding | 4 channels strictly mapped `A: 0, C: 1, G: 2, T: 3`, with padding and ambiguous characters mapped to `[0.0, 0.0, 0.0, 0.0]` null vector | 5-channel representation or `[0.25, 0.25, 0.25, 0.25]` uniform vector for N - rejected because historical `OneHotEmbedding` in `short/utils/models.py` explicitly drops the 5th channel |
| class index taxonomic offset | `model_label = id - 1` converting 1-based CSV ID to 0-based logit dimension | 1-based indexing with a dummy class at index 0 - rejected because the classifier heads are sized to `NUM_CLASSES = 13683` (Species) and `2702` (Genus) |
| CLI inference output format | stdout outputs JSON with keys `predicted_class_index` (int), `top5_class_indices` (list[int]), `confidence` (float) | free-form console logging - rejected because non-deterministic string output cannot be mechanically verified by scripts |

- Nothing else in this change is hard to reverse

## Criteria

### S1: Repository Inventory and Dependency Audit (P1)

Complete inventory mapping all notebooks to paper tables/figures and documenting missing checkpoints.

**Acceptance Criteria**

1. The system SHALL preserve all 15 historical notebooks in `short/` and `long/` without in-place modification.
2. The system SHALL document all 15 notebooks and their mappings to manuscript tables (Tables 1-12, Figures 1-5) in `docs/inventory.md`.
3. IF model weights (`.pt`) or parquet splits are absent on disk THEN the system SHALL record the missing artifacts as a blocking precondition in `docs/inventory.md` and validate inference through synthetic deterministic test fixtures.
4. The system SHALL map taxonomic CSV IDs to zero-based model class indices using the offset `model_label = id - 1`.

**Independent test:** run inventory verification script checking that all 15 notebooks and 6 CSV datasets match records in `docs/inventory.md`.

### S2: Modular Package and Clean Inference CLI (P1)

Extracted Python package and CLI entrypoint for 512 bp encoding and forward inference.

**Acceptance Criteria**

5. WHEN given an input DNA sequence of up to 512 bp THEN the system SHALL encode nucleotides into 4 channels in strict order A=0, C=1, G=2, T=3 with ambiguous bases and padding mapped to all-zero vectors `[0.0, 0.0, 0.0, 0.0]`.
6. WHEN executing `scripts/run_inference.py --sequence <seq> --model-type <resnet50_1d|convnext_tiny_1d> --num-classes <int>` THEN the system SHALL return exit code 0 and emit a JSON payload containing `input_length`, `channels`, `predicted_class_index`, and `top5_class_indices`.
7. IF an invalid nucleotide string or unknown model architecture is passed to `scripts/run_inference.py` THEN the system SHALL exit with code 2 and write an informative error message to standard error.
8. The system SHALL produce identical numerical output tensors between the extracted `viral_cnn.data.encoding` module and the historical `short/utils/models.py:OneHotEmbedding` across identical test sequences.

**Independent test:** run `python3 scripts/run_inference.py --sequence "ACGTACGTNNNN" --model-type resnet50_1d --num-classes 10` and assert valid JSON exit 0.

## Out of scope

| Excluded | Why |
| --- | --- |
| Full model retraining or fine-tuning | S01 is restricted to inventory, modular extraction, and a small-batch verifiable vertical slice; training belongs to future specs under explicit GPU/epoch budgets |
| Parquet split regeneration or download | S02 is dedicated to dataset provenance and split audit; S01 preserves raw CSVs and handles missing Parquets via synthetic contracts |
| Long sequence sliding-window voter | Voter audit and long-read evaluation belong to S06; S01 focuses on single-window (512 bp) vertical slice |

## Assumptions

| Assumption | Chosen default | Rationale | Confirmed? |
| --- | --- | --- | --- |
| Default sequence length for the vertical slice | 512 bp | Standard window size throughout the manuscript and historical models | y |
| Model architecture support for vertical slice | Both `resnet50_1d` and `convnext_tiny_1d` supported via model builder | Both architectures are central to the paper comparison | y |
| Execution mode when `.pt` checkpoints are absent on disk | Deterministic synthetic weights initialization | Real checkpoint files are not in the workspace; synthetic contracts verify pipeline mechanics without faking empirical accuracy | y |

**Open questions:** none - all resolved or logged above.

## Observable

| Surface | Decision | Landing |
| --- | --- | --- |
| CLI `scripts/run_inference.py` | output format and verbosity | AC 6 |
| CLI `scripts/run_inference.py` | error handling and exit codes | AC 7 |
| CLI `scripts/run_inference.py` | CLI flags and options (`--sequence`, `--model-type`, `--num-classes`, `--device`) | AC 6 |
| screen | n/a - this feature exposes no user-facing GUI; all interaction occurs via CLI and Python API |
| API | n/a - no network-facing web API or webhook is added in this feature |

## Sources

- `ROADMAP_SPECS_ARTIGO.md` - definition and acceptance criteria for S01
- `AGENTS.md` - repository invariant rules, directory organization guidelines, and scientific contracts
- `paper/BioData_Minning___Matheus.pdf` - manuscript definitions for 1D-CNN architectures, 512 bp windows, and taxonomic granularities

