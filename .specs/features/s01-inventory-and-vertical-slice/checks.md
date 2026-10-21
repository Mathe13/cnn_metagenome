# S01: Inventory, Organization, and Vertical Slice Extraction - Checks

Profile: light
Plan: `.specs/features/s01-inventory-and-vertical-slice/plan.md`

## Checks

### S1: Repository Inventory and Dependency Audit · 2 files · 18 KB · ~5k

**C1** - All 15 historical notebooks in `short/` and `long/` remain present and unmodified from baseline inventory (AC 1)
Proof: `.venv/bin/python3 -m unittest discover -s tests -k test_historical_notebooks_intact`

**C2** - `docs/inventory.md` documents all 15 notebooks and maps each to manuscript Tables 1-12 and Figures 1-5 (AC 2)
Proof: `.venv/bin/python3 -m unittest discover -s tests -k test_notebook_paper_mappings`

**C3** - Missing `.pt` checkpoints and Parquet splits are documented as remote blocking preconditions in `docs/inventory.md` (AC 3)
Proof: `.venv/bin/python3 -m unittest discover -s tests -k test_missing_checkpoints_documented`

**C4** - Taxonomic CSV IDs map to zero-based class indices via `model_label = id - 1` across species and genus (AC 4)
Proof: `.venv/bin/python3 -m unittest discover -s tests -k test_label_zero_based_offset`

### S2: Modular Package and Clean Inference CLI · 6 files · 35 KB · ~9k

**C5** - 512 bp spatial one-hot encoding strictly maps A=0, C=1, G=2, T=3 and produces null vector `[0.0, 0.0, 0.0, 0.0]` for padding and ambiguous bases (AC 5)
Proof: `.venv/bin/python3 -m unittest discover -s tests -k test_spatial_one_hot_strict_channels_and_null_pad`

**C6** - Extracted `viral_cnn.data.encoding` produces identical numerical output tensors to historical `short/utils/models.py:OneHotEmbedding` (AC 8)
Proof: `.venv/bin/python3 -m unittest discover -s tests -k test_parity_with_historical_one_hot_embedding`

**C7** - CLI `scripts/run_inference.py` executes forward pass on ResNet50-1D and ConvNeXt-1D returning exit code 0 and valid JSON output (AC 6)
Proof: `.venv/bin/python3 -m unittest discover -s tests -k test_clean_inference_exit_zero_and_json`

**C8** - CLI `scripts/run_inference.py` rejects invalid nucleotide inputs and unsupported model types with exit code 2 (AC 7)
Proof: `.venv/bin/python3 -m unittest discover -s tests -k test_invalid_inputs_exit_code_2`

## Coverage

| Set (size) | Member -> proof | Unproven |
| --- | --- | --- |
| Model architectures evaluated (2) | `resnet50_1d` C7 · `convnext_tiny_1d` C7 | - |
| Canonical nucleotide channels (4) | `A=0` C5 · `C=1` C5 · `G=2` C5 · `T=3` C5 | - |
| Non-canonical base and padding mappings (3) | `pad _` C5 · `ambiguous N` C5 · `gap -` C5 | - |
| CLI exit status codes (2) | `0` C7 · `2` C8 | - |
| Taxonomic granularity tables (2) | `species (13683)` C4 · `genus (2702)` C4 | - |

- Claims naming a status code or response shape: C7, C8 - each has a proof crossing the process boundary
- No other check claims more than the single case its proof exercises

## Swept

- validation: C8
- failure modes: C3
- idempotency: C5
- authorization: n/a - local scientific CLI tool with no network or authorization boundary
- concurrency: n/a - synchronous batch inference on local CPU/GPU with no concurrent shared state
- data lifecycle: C1
- dependency failure: C3
- state transitions: n/a - stateless forward inference without persisted state transitions
- observability: C7

## Handoff

- S1 = 5k in docs and inventory; S2 = 9k in src/viral_cnn, scripts, and tests. Total = 14k tokens, under the 150k budget - one builder
