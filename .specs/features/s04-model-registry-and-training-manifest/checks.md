# S04: Model Registry, Taxonomic Mappings, and Training Manifest - Checks

Profile: light
Plan: `.specs/features/s04-model-registry-and-training-manifest/plan.md`

## Checks

### S1: Model Registry and Training Manifest Documentation · 1 file · 14 KB · ~3k

**C1** - `docs/model_registry.md` documents all seven model variants (5 species + 2 genus), specifying architecture, rank, class count, and checkpoint path (AC 1)
Proof: `.venv/bin/python3 -m unittest discover -s tests -k test_manifest_documents_all_seven_model_variants`

**C2** - `docs/model_registry.md` documents 4-channel spatial encoding A/C/G/T (0, 1, 2, 3 with null vector for padding/ambiguous bases) and 1-based CSV to 0-based logit offset `model_label = id - 1` (AC 2)
Proof: `.venv/bin/python3 -m unittest discover -s tests -k test_manifest_documents_encoding_and_label_offset`

**C3** - `docs/model_registry.md` records verified hyperparameters (AdamW, CosineAnnealingLR, label smoothing, AMP) and explicitly tags undocumented parameters as unverified per AGENTS.md (AC 3)
Proof: `.venv/bin/python3 -m unittest discover -s tests -k test_manifest_records_verified_and_unverified_hyperparameters`

**C4** - `docs/model_registry.md` explicitly distinguishes between best validation checkpoint and final training epoch checkpoint across all variants (AC 4)
Proof: `.venv/bin/python3 -m unittest discover -s tests -k test_manifest_distinguishes_best_validation_from_last_checkpoint`

### S2: Unified Model Registry and Checkpoint Loader Engine · 2 files · 24 KB · ~7k

**C5** - `viral_cnn.models.registry.load_model_checkpoint` loads weights identically from either a bare state_dict or a nested dict with model_state_dict (AC 5)
Proof: `.venv/bin/python3 -m unittest discover -s tests -k test_checkpoint_loader_handles_bare_and_nested_dict`

**C6** - `viral_cnn.models.registry.load_model_checkpoint` strips `_orig_mod.` compile prefixes without key errors (AC 6)
Proof: `.venv/bin/python3 -m unittest discover -s tests -k test_checkpoint_loader_strips_orig_mod_compile_prefix`

**C7** - `viral_cnn.models.registry.load_model_checkpoint` raises ValueError on classification head class count mismatch between checkpoint and target model (AC 7)
Proof: `.venv/bin/python3 -m unittest discover -s tests -k test_checkpoint_loader_rejects_class_count_mismatch`

**C8** - `viral_cnn.models.registry` provides programmatic APIs get_model_spec, list_model_variants, and build_registered_model (AC 8)
Proof: `.venv/bin/python3 -m unittest discover -s tests -k test_model_registry_api_queries_and_instantiation`

### S3: Declarative Configuration and CLI Integration · 3 files · 18 KB · ~5k

**C9** - `configs/models.yaml` defines all 7 model variants with declarative paths, hyperparameters, and verification status without absolute or personal paths (AC 9)
Proof: `.venv/bin/python3 -m unittest discover -s tests -k test_models_yaml_declares_all_variants_without_personal_paths`

**C10** - CLI `scripts/run_inference.py --model-variant <variant>` instantiates the architecture, configures class count, and executes forward inference (AC 10)
Proof: `.venv/bin/python3 -m unittest discover -s tests -k test_run_inference_cli_with_model_variant_flag`

**C11** - Model registry and checkpoint loading routines execute deterministically on synthetic fixtures without GPU or massive external downloads (AC 11)
Proof: `.venv/bin/python3 -m unittest discover -s tests -k test_model_registry_deterministic_synthetic_execution`

## Coverage

| Set (size) | Member -> proof | Unproven |
| --- | --- | --- |
| Model variants registered (7) | `resnet_clean_species` C8 · `resnet_noise_species` C8 · `convnext_zero_species` C8 · `convnext_transfer_species` C8 · `convnext_noise_species` C8 · `resnet_genus` C8 · `convnext_genus` C8 | - |
| Checkpoint dictionary wrapper formats (2) | `bare_state_dict` C5 · `nested_model_state_dict` C5 | - |
| Checkpoint compile prefix states (2) | `uncompiled_keys` C5 · `compiled_orig_mod_prefix` C6 | - |
| Model head class verification outcomes (2) | `dimension_match_accepted` C5 · `dimension_mismatch_rejected` C7 | - |
| Hyperparameter verification flags (2) | `verified_evidenced` C3 · `unverified_notebook_gap` C3 | - |
| CLI execution status codes (2) | `0` C10 · `2` C10 | - |

- Claims naming a status code or response shape: C10 - has a proof crossing the process boundary
- No other check claims more than the single case its proof exercises

## Swept

- validation: C7
- failure modes: C7
- idempotency: C5
- authorization: n/a - local scientific CLI tool with no network or authorization boundary
- concurrency: n/a - deterministic batch processing on single machine with no concurrent state mutation
- data lifecycle: C1
- dependency failure: C11
- state transitions: n/a - stateless forward inference and checkpoint loading
- observability: C10

## Handoff

- S1 = 3k in docs; S2 = 7k in src/viral_cnn/models/registry.py and tests; S3 = 5k in configs, scripts, and fixtures. Total = ~15k tokens, well under the 150k budget - one builder

- **Mechanism:** one builder
