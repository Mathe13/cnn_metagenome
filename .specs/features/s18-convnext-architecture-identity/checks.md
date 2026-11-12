# S18 ConvNeXt Architecture Identity checks

Profile: light
Plan: `.specs/features/s18-convnext-architecture-identity/plan.md`

## Intent

Historically, the repository defined the 1D ConvNeXt variant as `ConvNeXt_Tiny_1D`, which can mistakenly imply a direct 1D conversion of canonical ConvNeXt-Tiny rather than the repository's hybrid architecture (ResNet stem, `[3, 4, 6, 3]` depths, ConvNeXt-inspired blocks, global max pooling, 135.98M backbone parameters). This feature establishes `HybridConvNeXt1D` as the canonical implementation class (with alias `ConvNeXtInspired1D`), preserves `ConvNeXt_Tiny_1D` for backward compatibility, exposes structured architecture metadata distinguishing family from implementation identity, and ensures full model builder and checkpoint parity.

5 checks in 1 slice · 2 one-way doors · 0 open, of which 0 block

## Checks

### S1 - ConvNeXt architecture identity and metadata contract · 3 files · ~20 KB · ~6k

**C1** - `HybridConvNeXt1D` exposes `ARCHITECTURE_METADATA` attribute and `get_architecture_metadata()` method matching required schema and values (AC 1)
Proof: `REMOTE_SERVER=False PYTHONPATH=src .venv/bin/python3 -m unittest tests.test_s18_convnext_architecture_identity.TestS18ConvNeXtArchitectureIdentity.test_hybrid_convnext_metadata_schema -k test_hybrid_convnext_metadata_schema`

**C2** - `ConvNeXt_Tiny_1D` and `ConvNeXtInspired1D` resolve directly to `HybridConvNeXt1D` preserving class identity and metadata (AC 2)
Proof: `REMOTE_SERVER=False PYTHONPATH=src .venv/bin/python3 -m unittest tests.test_s18_convnext_architecture_identity.TestS18ConvNeXtArchitectureIdentity.test_backward_compatible_aliases -k test_backward_compatible_aliases`

**C3** - `build_model` successfully instantiates `HybridConvNeXt1D` for `"hybrid_convnext_1d"`, `"convnext_inspired_1d"`, `"convnext_tiny_1d"`, and `"convnext"` (AC 3)
Proof: `REMOTE_SERVER=False PYTHONPATH=src .venv/bin/python3 -m unittest tests.test_s18_convnext_architecture_identity.TestS18ConvNeXtArchitectureIdentity.test_builder_resolves_all_aliases -k test_builder_resolves_all_aliases`

**C4** - State dict weights load symmetrically with `strict=True` between alias names and produce identical forward output logits (AC 4)
Proof: `REMOTE_SERVER=False PYTHONPATH=src .venv/bin/python3 -m unittest tests.test_s18_convnext_architecture_identity.TestS18ConvNeXtArchitectureIdentity.test_state_dict_and_forward_parity -k test_state_dict_and_forward_parity`

**C5** - `profile_inference_model` decomposes parameters on `HybridConvNeXt1D` matching exact expected backbone and head parameter counts (AC 1, AC 3)
Proof: `REMOTE_SERVER=False PYTHONPATH=src .venv/bin/python3 -m unittest tests.test_s18_convnext_architecture_identity.TestS18ConvNeXtArchitectureIdentity.test_parameter_profile_decomposition -k test_parameter_profile_decomposition`

## Coverage

| Set (size) | Member -> proof | Unproven |
| --- | --- | --- |
| architecture symbols (3) | `HybridConvNeXt1D` C1 · `ConvNeXtInspired1D` C2 · `ConvNeXt_Tiny_1D` C2 | - |
| builder identifiers (4) | `hybrid_convnext_1d` C3 · `convnext_inspired_1d` C3 · `convnext_tiny_1d` C3 · `convnext` C3 | - |
| serialization parity (1) | `state_dict_parity` C4 | - |
| parameter audit (1) | `profile_decomposition` C5 | - |

- No check claims more than the single case its proof exercises

## Swept

- validation : C1, C2, C3
- failure modes : C3, C4
- idempotency : C1, C4
- authorization : n/a - local neural network architecture definition
- concurrency : n/a - stateless model initialization and forward evaluation
- data lifecycle : n/a - ephemeral model memory without persistence changes
- dependency failure : n/a - pure PyTorch modules
- state transitions : n/a - stateless neural network forward passes
- observability : C1, C5

## Handoff

Files touched:
- `src/viral_cnn/models/convnext.py` (7 KB)
- `src/viral_cnn/models/__init__.py` (1 KB)
- `src/viral_cnn/models/builder.py` (2 KB)
- `tests/test_s18_convnext_architecture_identity.py` (6 KB)

Total estimate: ~16 KB · ~5k tokens.
Budget: 150k tokens. Estimate is well within budget: one builder.
