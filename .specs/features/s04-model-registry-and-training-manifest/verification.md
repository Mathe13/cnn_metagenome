# S04: Model Registry, Taxonomic Mappings, and Training Manifest - verification

**Feature**: s04-model-registry-and-training-manifest
**Verdict**: PASS
**Profile**: light
**Diff range**: ad556a1..HEAD
**Round**: 1 - full
**Verifier**: independent sub-agent (author != verifier)
**Date**: 2026-10-21
**Result**: PASS

## Checks

| Check | Claim | Proof run | Evidence | Result |
| --- | --- | --- | --- | --- |
| C1 | `docs/model_registry.md` documents all seven model variants (5 species + 2 genus), specifying architecture, rank, class count, and checkpoint path (AC 1) | `.venv/bin/python3 -m unittest discover -s tests -k test_manifest_documents_all_seven_model_variants` exit 0 | `tests/test_model_registry.py:48` - `self.assertIn(variant, content)` · `tests/test_model_registry.py:50` - `self.assertIn("13.683", content)` · `tests/test_model_registry.py:51` - `self.assertIn("2.702", content)` | PASS |
| C2 | `docs/model_registry.md` documents 4-channel spatial encoding A/C/G/T (0, 1, 2, 3 with null vector for padding/ambiguous bases) and 1-based CSV to 0-based logit offset `model_label = id - 1` (AC 2) | `.venv/bin/python3 -m unittest discover -s tests -k test_manifest_documents_encoding_and_label_offset` exit 0 | `tests/test_model_registry.py:58` - `self.assertIn("A/C/G/T", content)` · `tests/test_model_registry.py:59` - `self.assertIn("model_label", content)` · `tests/test_model_registry.py:60` - `self.assertIn("csv_id", content)` · `tests/test_model_registry.py:61` - `self.assertIn("vetor nulo", content.lower())` | PASS |
| C3 | `docs/model_registry.md` records verified hyperparameters (AdamW, CosineAnnealingLR, label smoothing, AMP) and explicitly tags undocumented parameters as unverified per AGENTS.md (AC 3) | `.venv/bin/python3 -m unittest discover -s tests -k test_manifest_records_verified_and_unverified_hyperparameters` exit 0 | `tests/test_model_registry.py:68` - `self.assertIn("AdamW", content)` · `tests/test_model_registry.py:69` - `self.assertIn("CosineAnnealingLR", content)` · `tests/test_model_registry.py:70` - `self.assertIn("label_smoothing", content)` · `tests/test_model_registry.py:71` - `self.assertIn("GradScaler", content)` · `tests/test_model_registry.py:72` - `self.assertIn("unverified", content.lower())` | PASS |
| C4 | `docs/model_registry.md` explicitly distinguishes between best validation checkpoint and final training epoch checkpoint across all variants (AC 4) | `.venv/bin/python3 -m unittest discover -s tests -k test_manifest_distinguishes_best_validation_from_last_checkpoint` exit 0 | `tests/test_model_registry.py:79` - `self.assertIn("finetune_noise_14.pt", content)` · `tests/test_model_registry.py:80` - `self.assertIn("best_resnet_classifier__fast_phase4.pt", content)` · `tests/test_model_registry.py:81` - `self.assertIn("última época", content.lower())` · `tests/test_model_registry.py:82` - `self.assertIn("melhor", content.lower())` | PASS |
| C5 | `viral_cnn.models.registry.load_model_checkpoint` loads weights identically from either a bare state_dict or a nested dict with model_state_dict (AC 5) | `.venv/bin/python3 -m unittest discover -s tests -k test_checkpoint_loader_handles_bare_and_nested_dict` exit 0 | `tests/test_model_registry.py:108` - `self.assertEqual(metadata.get("epoch"), 14)` · `tests/test_model_registry.py:109` - `self.assertEqual(metadata.get("metrics"), {"val_loss": 0.123})` · `tests/test_model_registry.py:113` - `self.assertTrue(torch.equal(model1.state_dict()[k], model2.state_dict()[k]))` | PASS |
| C6 | `viral_cnn.models.registry.load_model_checkpoint` strips `_orig_mod.` compile prefixes without key errors (AC 6) | `.venv/bin/python3 -m unittest discover -s tests -k test_checkpoint_loader_strips_orig_mod_compile_prefix` exit 0 | `tests/test_model_registry.py:131` - `load_model_checkpoint(model, compiled_state, expected_classes=10, strict=True)` · `tests/test_model_registry.py:134` - `self.assertTrue(torch.equal(model.state_dict()[k], clean_state[k]))` | PASS |
| C7 | `viral_cnn.models.registry.load_model_checkpoint` raises ValueError on classification head class count mismatch between checkpoint and target model (AC 7) | `.venv/bin/python3 -m unittest discover -s tests -k test_checkpoint_loader_rejects_class_count_mismatch` exit 0 | `tests/test_model_registry.py:144` - `with self.assertRaises(ValueError) as ctx:` · `tests/test_model_registry.py:148` - `self.assertIn("2702", err_msg)` · `tests/test_model_registry.py:149` - `self.assertIn("13683", err_msg)` · `tests/test_model_registry.py:150` - `self.assertIn("Classification head class count mismatch", err_msg)` | PASS |
| C8 | `viral_cnn.models.registry` provides programmatic APIs get_model_spec, list_model_variants, and build_registered_model (AC 8) | `.venv/bin/python3 -m unittest discover -s tests -k test_model_registry_api_queries_and_instantiation` exit 0 | `tests/test_model_registry.py:155` - `self.assertEqual(len(variants), 7)` · `tests/test_model_registry.py:159` - `self.assertIsInstance(spec, ModelVariantSpec)` · `tests/test_model_registry.py:169` - `self.assertIsInstance(model, torch.nn.Module)` · `tests/test_model_registry.py:170` - `self.assertEqual(model.num_classes, spec.num_classes)` | PASS |
| C9 | `configs/models.yaml` defines all 7 model variants with declarative paths, hyperparameters, and verification status without absolute or personal paths (AC 9) | `.venv/bin/python3 -m unittest discover -s tests -k test_models_yaml_declares_all_variants_without_personal_paths` exit 0 | `tests/test_model_registry.py:177` - `self.assertTrue(yaml_path.exists())` · `tests/test_model_registry.py:183` - `self.assertEqual(len(models_data), 7)` · `tests/test_model_registry.py:189` - `self.assertFalse(ckpt_path.startswith("/"))` · `tests/test_model_registry.py:190` - `self.assertNotIn("/home/", ckpt_path)` | PASS |
| C10 | CLI `scripts/run_inference.py --model-variant <variant>` instantiates the architecture, configures class count, and executes forward inference (AC 10) | `.venv/bin/python3 -m unittest discover -s tests -k test_run_inference_cli_with_model_variant_flag` exit 0 | `tests/test_model_registry.py:206` - `self.assertEqual(res.returncode, 0)` · `tests/test_model_registry.py:209` - `self.assertEqual(out_json["status"], "success")` · `tests/test_model_registry.py:210` - `self.assertEqual(out_json["model_type"], "resnet50_1d")` · `tests/test_model_registry.py:211` - `self.assertEqual(out_json["num_classes"], 13683)` · `tests/test_model_registry.py:224` - `self.assertEqual(bad_res.returncode, 2)` | PASS |
| C11 | Model registry and checkpoint loading routines execute deterministically on synthetic fixtures without GPU or massive external downloads (AC 11) | `.venv/bin/python3 -m unittest discover -s tests -k test_model_registry_deterministic_synthetic_execution` exit 0 | `tests/test_model_registry.py:240` - `metadata = load_model_checkpoint(loaded_model, ckpt_file, expected_classes=10)` · `tests/test_model_registry.py:241` - `self.assertEqual(metadata.get("epoch"), 5)` · `tests/test_model_registry.py:251` - `self.assertTrue(torch.allclose(out_base, out_loaded, atol=1e-6))` | PASS |

## Coverage

| Set (size) | Member -> proof | Unproven |
| --- | --- | --- |
| Model variants registered (7) | `resnet_clean_species` C8 · `resnet_noise_species` C8 · `convnext_zero_species` C8 · `convnext_transfer_species` C8 · `convnext_noise_species` C8 · `resnet_genus` C8 · `convnext_genus` C8 | - |
| Checkpoint dictionary wrapper formats (2) | `bare_state_dict` C5 · `nested_model_state_dict` C5 | - |
| Checkpoint compile prefix states (2) | `uncompiled_keys` C5 · `compiled_orig_mod_prefix` C6 | - |
| Model head class verification outcomes (2) | `dimension_match_accepted` C5 · `dimension_mismatch_rejected` C7 | - |
| Hyperparameter verification flags (2) | `verified_evidenced` C3 · `unverified_notebook_gap` C3 | - |
| CLI execution status codes (2) | `0` C10 · `2` C10 | - |

## Level and sampling gaps

- Boundary level checks: C10 asserts CLI execution crossing process boundary via `subprocess.run` verifying exit codes 0 and 2, argument validation, and structured JSON stdout schema.
- Programmatic units: C5–C8 assert checkpoint loading, key stripping, dimension mismatch exceptions, and factory model construction at unit level in Python.
- Documentation contract: C1–C4 assert documentation presence, variant inventory, encoding specifications, offset formulas, and verified/unverified labeling directly against `docs/model_registry.md`.
- Determinism and environment independence: C11 verifies deterministic forward execution and weight restoration using synthetic tensor fixtures without external network downloads or GPU dependencies.
- Sampling gaps: Zero gaps identified. All 7 canonical variants are tested for registration and instantiation.

## Swept existing

- **validation (C7)**: `src/viral_cnn/models/registry.py:272-279` inspects `clean_state_dict["fc.weight"].shape[0]` against expected classes and raises `ValueError`. Proven in `tests/test_model_registry.py:136-151`.
- **failure modes (C7)**: Mismatched checkpoint classes (e.g., loading genus checkpoint with 2,702 classes into species model with 13,683 classes) are intercepted and rejected before corrupting inference. Handled and verified.
- **idempotency (C5)**: `load_model_checkpoint` loads state dictionaries identically from bare or nested dict formats, yielding bitwise identical parameters (`torch.equal`). Proven in `tests/test_model_registry.py:112-116`.
- **authorization**: n/a — local scientific modeling toolkit with no network or authorization boundaries.
- **concurrency**: n/a — deterministic single-machine inference and batch evaluation.
- **data lifecycle (C1)**: Documentation and manifest track full lifecycle provenance from training lineage and checkpoint selection criteria to evaluation.
- **dependency failure (C11)**: Synthetic fixture tests verify loading and inference without external weight files or GPU dependencies.
- **state transitions**: n/a — stateless forward inference and pure functional weight loading.
- **observability (C10)**: CLI `scripts/run_inference.py` reports structured JSON with status, model metadata, class indices, and confidence scores, emitting exit code 2 on invalid parameters.

## Traceability

- **AC 1 (7 Model Variants)**: C1 · `docs/model_registry.md:15-23` · `configs/models.yaml:5-136` · `src/viral_cnn/models/registry.py:30-182`
- **AC 2 (4-Channel Spatial Encoding & Offset)**: C2 · `docs/model_registry.md:29-44` · `src/viral_cnn/data/encoding.py` · `scripts/run_inference.py:188`
- **AC 3 (Verified vs Unverified Hyperparameters)**: C3 · `docs/model_registry.md:65-86` · `configs/models.yaml` · `src/viral_cnn/models/registry.py:25-26`
- **AC 4 (Best vs Last Checkpoint Distinction)**: C4 · `docs/model_registry.md:49-60` · `configs/models.yaml` · `src/viral_cnn/models/registry.py:59,87`
- **AC 5 (Bare and Nested Checkpoint Formats)**: C5 · `src/viral_cnn/models/registry.py:255-260` · `tests/test_model_registry.py:86-117`
- **AC 6 (Compile Prefix `_orig_mod.` Stripping)**: C6 · `src/viral_cnn/models/registry.py:262-265` · `tests/test_model_registry.py:118-135`
- **AC 7 (Classification Head Shape Validation)**: C7 · `src/viral_cnn/models/registry.py:272-279` · `tests/test_model_registry.py:136-151`
- **AC 8 (Registry APIs `get_model_spec`, `list_model_variants`, `build_registered_model`)**: C8 · `src/viral_cnn/models/registry.py:201-217,284-312` · `tests/test_model_registry.py:152-171`
- **AC 9 (Declarative `configs/models.yaml` without Personal Paths)**: C9 · `configs/models.yaml` · `tests/test_model_registry.py:174-191`
- **AC 10 (CLI `--model-variant` Integration)**: C10 · `scripts/run_inference.py:54-59,93-107` · `tests/test_model_registry.py:192-226`
- **AC 11 (Deterministic Synthetic Execution)**: C11 · `tests/test_model_registry.py:227-252`

## Gate

`.venv/bin/python3 -m unittest discover -s tests` - 40 passed, 0 failed

## Verdict

PASS. All 11 checks proven with located `file:line` assertions, zero surviving defects, complete model variant coverage, and full conformance to scientific invariants in `AGENTS.md`.
