# S18 — ConvNeXt Architecture Identity and Metadata Contract

## Problem

Historically, the repository defined the 1D ConvNeXt variant as `ConvNeXt_Tiny_1D` (`src/viral_cnn/models/convnext.py`). This class name can mistakenly imply a direct 1D transcription of the canonical ConvNeXt-Tiny architecture (Liu et al., 2022). In reality, the repository implementation is intentionally hybrid:
- ResNet-style convolutional stem (`7x1 conv`, `BatchNorm1d`, `ReLU`, `MaxPool1d`), whereas canonical ConvNeXt uses a non-overlapping patchify stem (`4x1` strided conv + `LayerNorm`);
- `[3, 4, 6, 3]` stage depths (resembling ResNet50 stage repetitions rather than ConvNeXt-Tiny's `[3, 3, 9, 3]`);
- ConvNeXt-inspired depthwise separable blocks with `LayerNorm1d`, GELU, inverted bottleneck MLP (`4x` expansion), and Layer Scale;
- Custom channel progression (`[256, 512, 1024, 2048]`);
- Global max pooling (`F.max_pool1d`) into a classification head with `D=2048` input features, rather than canonical global average pooling with `D=768`.

Without explicit architecture identity metadata, scientific reporting and manuscript texts risk conflating the repository's custom 1D hybrid design with canonical ConvNeXt-Tiny, obscuring why the backbone parameter count is 135.98M instead of the canonical 28M.

When this ships, the canonical implementation class is named `HybridConvNeXt1D` (with alias `ConvNeXtInspired1D`), maintaining `ConvNeXt_Tiny_1D` as a backward-compatible alias. The model exposes structured architecture metadata through `get_architecture_metadata()` and `ARCHITECTURE_METADATA`, explicitly distinguishing the canonical architecture family (`convnext-inspired`) from the repository implementation identity (`HybridConvNeXt1D`), with full model builder and registry support.

## Flow

This change reuses `viral_cnn.models.convnext` (exists) and model construction mechanisms without breaking existing checkpoint loading.

1. Model definition and metadata enter `HybridConvNeXt1D` (door 1) - defines class attributes and instance method `get_architecture_metadata` returning canonical identity dictionary.
2. Backward-compatibility aliases `ConvNeXtInspired1D` and `ConvNeXt_Tiny_1D` point to `HybridConvNeXt1D` (door 2).
3. Model builders and registry (`viral_cnn.models.builder` (exists)) register `"hybrid_convnext_1d"`, `"convnext_inspired_1d"`, `"convnext_tiny_1d"`, and `"convnext"`.
4. Inspection utilities and profilers can query `get_architecture_metadata()` on classes or instances to distinguish `convnext-inspired` from canonical variants.

## Impact

| Front | What changes |
| --- | --- |
| domain | `src/viral_cnn/models/convnext.py` defines `HybridConvNeXt1D`, `ConvNeXtInspired1D`, `ConvNeXt_Tiny_1D`, and `get_architecture_metadata` |
| domain | `src/viral_cnn/models/__init__.py` and `src/viral_cnn/models/builder.py` export and register the hybrid identity symbols |
| stored data | no stored files or checkpoints are modified; state dict keys remain identical |

## Relations

None - no stored-data shape change

## Surface

None - nothing consumed outside

## Landing

| One-way door | Literal shape | Alternative rejected |
| --- | --- | --- |
| 1 — identidade primária HybridConvNeXt1D com metadados estruturados | `HybridConvNeXt1D` com `get_architecture_metadata()` retornando `{"architecture_family": "convnext-inspired", "implementation": "HybridConvNeXt1D", "stage_depths": [3, 4, 6, 3], "stem": "resnet-style", "pooling": "global-max"}` | Manter apenas o nome `ConvNeXt_Tiny_1D` sem metadados: induz leitores e auditores ao erro de assumir que o modelo replica estritamente o ConvNeXt-Tiny canônico. |
| 2 — retrocompatibilidade transparente via aliases | `ConvNeXtInspired1D = HybridConvNeXt1D` e `ConvNeXt_Tiny_1D = HybridConvNeXt1D` | Renomear a classe sem manter `ConvNeXt_Tiny_1D`: quebra desserialização de checkpoints históricos, scripts legados e testes existentes. |

- Nothing else in this change is hard to reverse

## Criteria

### S1: ConvNeXt architecture identity and metadata contract (P1)

A implementação de ConvNeXt 1D deve explicitar sua identidade híbrida (`HybridConvNeXt1D` / `ConvNeXtInspired1D`), expor metadados estruturados de arquitetura e preservar compatibilidade binária e programática com `ConvNeXt_Tiny_1D`.

**Acceptance Criteria**

1. WHEN `HybridConvNeXt1D` is instantiated THEN the model SHALL possess class and instance attribute `ARCHITECTURE_METADATA` and method `get_architecture_metadata()` returning a dictionary matching `architecture_family == "convnext-inspired"`, `implementation == "HybridConvNeXt1D"`, `stage_depths == [3, 4, 6, 3]`, `stem == "resnet-style"`, and `pooling == "global-max"`.
2. WHEN `ConvNeXt_Tiny_1D` or `ConvNeXtInspired1D` is referenced or instantiated THEN the system SHALL resolve to `HybridConvNeXt1D`, producing identical state dict keys, forward activations, and architecture metadata.
3. WHEN `build_model` is invoked with model names `"hybrid_convnext_1d"`, `"convnext_inspired_1d"`, `"convnext_tiny_1d"`, or `"convnext"` THEN the system SHALL return an instance of `HybridConvNeXt1D`.
4. WHEN historical checkpoint state dicts containing weights trained under `ConvNeXt_Tiny_1D` are loaded into `HybridConvNeXt1D` THEN `load_state_dict` SHALL succeed with `strict=True` and produce identical output logits.

**Independent test:** Execute `python3 -m unittest tests/test_s18_convnext_architecture_identity.py` verifying class identity, backward-compatible aliases, architecture metadata dictionary, model builder resolution, and state dict weight parity.

## Out of scope

| Excluded | Why |
| --- | --- |
| Alteração da arquitetura convolucional ou pesos treinados | S18 documenta e formaliza a identidade da arquitetura existente sem alterar tensores ou hiperparâmetros de treinamento. |
| Implementação de um ConvNeXt-Tiny canônico alternativo | O escopo do artigo e da reprodução avalia a variante híbrida efetivamente executada nos experimentos. |

## Assumptions

| Assumption | Chosen default | Rationale | Confirmed? |
| --- | --- | --- | --- |
| Nome canônico da classe | `HybridConvNeXt1D` | Descreve fielmente a combinação de stem ResNet, blocos ConvNeXt e pooling global máximo. | y |
| Aliases preservados | `ConvNeXt_Tiny_1D` e `ConvNeXtInspired1D` | Garante retrocompatibilidade com notebooks, checkpoints e scripts históricos. | y |

**Open questions:** none - all resolved or logged above.

## Observable

None - no user-facing surface

## Sources

- ROADMAP_IMPROVEMENTS.md (Section 12: S18 — ConvNeXt architecture identity) - define o contrato científico de metadados, o nome canônico `HybridConvNeXt1D`, os aliases e a terminologia do manuscrito.
- ROADMAP_IMPROVEMENTS.md (Section 20: Milestone C — Scientific scope) - alinha código, documentação e manuscrito na mesma terminologia científica.
- AGENTS.md (Invariantes científicos) - não mudar arquitetura, pesos ou pré-processamento sem identificador explícito.
