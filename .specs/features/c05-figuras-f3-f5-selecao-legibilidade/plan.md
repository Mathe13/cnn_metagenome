# C05 — Figuras F3 e F5, Selecao e Legibilidade

## Problem

A renderização e validação científica das figuras principais de publicação (especialmente F3 e F5) exigem verificação estrita de completude, coordenadas geométricas e legendas desmistificadas:

1. **Figura F3 (Painéis de Perturbação de Motivos em 5 Variantes):**
   - A Figura 3 representa o impacto da perturbação pareada de 7-mers nos filtros de 1ª camada através de cinco painéis ($a$ a $e$).
   - O contrato científico exige a presença obrigatória de exatamente todas as 5 variantes de espécies (`ResNet50_Clean`, `ConvNeXt_Z`, `ConvNeXt_T`, `ResNet50_F`, `ConvNeXt_F`). A ausência de qualquer variante (como a exclusão histórica de `ConvNeXt_F` em análises preliminares) deve interromper a validação com erro explicito, proibindo terminantemente preencher painéis ausentes com queda zero ou barras vazias.
   - Os controles devem manter pareamento estrito na mesma sequência com controle uniforme e controle de composição GC pareado ($|\Delta\text{GC}|=0$).

2. **Figura F5 (Grafo de Rede de Coocorrência de Motivos):**
   - A Figura 5 ilustra a rede de coocorrência de 7-mers gerada em `short/08`.
   - Historicamente, tentou-se renderizar gráficos de barras ou tabelas sem coordenadas geométricas para F5, falhando na visualização em rede.
   - O contrato exige o consumo de `edge_table.csv` com coordenadas geométricas 2D explícitas (`x1`, `y1`, `x2`, `y2`), nós de origem/destino e pesos de arestas (`weight` ou `jaccard`), rejeitando qualquer entrada sem coordenadas 2D.
   - A legenda e metadados devem explicitar que o grafo reflete análise exploratória de rede, sem sugerir significância biológica individual para arestas isoladas.

3. **Inspeção de Legibilidade e Checklist de Renderização (F1–F5):**
   - As cinco figuras de publicação (F1: robustez a ruído, F2: projeção 2D UMAP, F3: perturbação pareada de motivos, F4: perfil RSCU exploratório, F5: rede de coocorrência) devem atender a padrões de contraste, limites de dimensões, rótulos de eixos legíveis e declaração qualitativa explícita em F2.
   - Deve haver uma rotina automatizada de verificação e geração de checklist de inspeção de figuras (`verify_figure_publication_readiness`), validando dimensões de imagem, presença de arquivos e integridade de legendas.

## Flow

1. Metadados de seleção de figuras entram em `render_figures` (exists/updated) - validar parâmetros de seleção (`seed`, `ids`, `sampling`, `filters`), coordenadas 2D e completude de variantes (door 1, door 2).
2. Artefatos tabulares de F3 e F5 entram em seus respectivos transformadores e renderizadores (exists/updated) - conferir presença das 5 variantes em F3 e coordenadas `x1, y1, x2, y2` em F5, gerando saídas PNG com alta resolução (door 1, door 2).
3. Saídas das figuras geradas entram em `verify_figure_publication_readiness` (exists/new) - inspecionar existência, dimensões em pixels, integridade de cabeçalhos PNG e gerar manifesto de checklist de figuras (door 1, door 3).
4. Manifestos e figuras verificados entram em `validate_selection` (exists) - admitir com status `DERIVED_VERIFIED` sob modo estrito (door 1, door 2).

## Impact

| Front | What changes |
| --- | --- |
| domain | new term: `figure_readiness_checklist` - relatório estruturado de validação de qualidade visual, dimensional e científica das figuras F1–F5 |
| domain | updated term: `f3_five_variant_panel` - painel estrito de 5 variantes (`ResNet50_Clean`, `ConvNeXt_Z`, `ConvNeXt_T`, `ResNet50_F`, `ConvNeXt_F`) sem tolerância a variantes faltantes ou preenchimento zero |
| stored data | nada a migrar |

## Relations

`None - no stored-data shape change`

## Surface

| Route | In | Out | Status |
| --- | --- | --- | --- |
| `viral_cnn.reporting.publication:render_figures(config, output_dir, rows)` | `dict`, `Path`, `list[dict]` | `dict[str, Path]` | `200`, `400` |
| `viral_cnn.reporting.figures:verify_figure_publication_readiness(figure_outputs, metadata)` | `dict[str, Path]`, `dict` | `dict[str, Any]` (checklist) | `200`, `400` |
| `viral_cnn.reporting.transformations:transform_figure_3_paired_motif_perturbation(df, meta)` | `pd.DataFrame`, `dict` | `pd.DataFrame` (F3) | `200`, `400` |
| `viral_cnn.reporting.transformations:transform_figure_5_motif_cooccurrence(df, meta)` | `pd.DataFrame`, `dict` | `pd.DataFrame` (F5) | `200`, `400` |

## Landing

| One-way door | Literal shape | Alternative rejected |
| --- | --- | --- |
| 1 — Autenticidade estrita dos 5 modelos de espécies em F3 | Exigir exatamente as 5 variantes (`ResNet50_Clean`, `ConvNeXt_Z`, `ConvNeXt_T`, `ResNet50_F`, `ConvNeXt_F`) e painéis $a$–$e$ em F3, rejeitando peremptoriamente conjuntos incompletos com `PublicationValidationError`. | Omitir `ConvNeXt_F` ou desenhar painéis vazios com queda zero. |
| 2 — Exigência imperativa de coordenadas geométricas 2D em F5 | Exigir colunas `x1, y1, x2, y2` em `edge_table.csv` para traçado de arestas e nós do grafo de coocorrência, rejeitando dados tabulares unidimensionais. | Renderizar gráficos de barras ou listas de texto como substitutos de F5. |
| 3 — Checklist automatizado de legibilidade e integridade de figuras | Inspecionar formalmente integridade de PNGs (cabeçalho, largura/altura $>0$, razão de aspecto, contraste de nós e advertência qualitativa de F2) via `verify_figure_publication_readiness`. | Confiar apenas na geração silenciosa de arquivos sem checagem de integridade visual. |

## Criteria

### S1: Verificação estrita das 5 variantes e painéis de F3 (P1)

Garantir integridade arquitetural e ausência de omissões em F3.

**Acceptance Criteria**

1. WHEN `transform_figure_3_paired_motif_perturbation` receives data with all 5 species model variants THEN the system SHALL output a validated DataFrame for panels $a$–$e$.
2. IF any of the 5 required variants (`ResNet50_Clean`, `ConvNeXt_Z`, `ConvNeXt_T`, `ResNet50_F`, `ConvNeXt_F`) is missing from F3 data THEN the system SHALL raise `PublicationValidationError`.
3. The system SHALL verify that no missing variant in F3 is substituted with zero activation drop or blank bars.

**Independent test:** testar `transform_figure_3_paired_motif_perturbation` e `render_figures` com as 5 variantes completas e com variantes omitidas, conferindo rejeição.

### S2: Exigência de coordenadas geométricas 2D e layout de F5 (P1)

Garantir representação fidedigna em rede de coocorrência.

**Acceptance Criteria**

4. WHEN `transform_figure_5_motif_cooccurrence` executes THEN the system SHALL require columns `["x1", "y1", "x2", "y2"]` from `edge_table.csv`.
5. IF `x1, y1, x2, y2` coordinates are absent from F5 input THEN the system SHALL raise `PublicationValidationError`.
6. WHEN `render_figures` renders F5 THEN the system SHALL plot edge segments and node points from geometric coordinates, logging plot parameters.

**Independent test:** executar transformação e renderização de F5 com dados de arestas válidos e com dados de gráfico de barras sem coordenadas, verificando exceção.

### S3: Checklist de legibilidade, resolução e parâmetros de seleção (P1)

Inspeção de qualidade e integridade das figuras F1–F5.

**Acceptance Criteria**

7. WHEN `verify_figure_publication_readiness` executes on generated figure outputs THEN the system SHALL inspect image validity (PNG signature, width $> 0$, height $> 0$) and return a structured checklist.
8. IF any figure output file is missing or has zero bytes THEN `verify_figure_publication_readiness` SHALL report failure.
9. The system SHALL verify that F2 figure metadata contains the qualitative projection disclaimer.
10. The system SHALL verify that figure selection dictionaries for F1–F5 contain `seed`, `ids`, `sampling`, and `filters`.

**Independent test:** executar `verify_figure_publication_readiness` sobre saídas válidas e saídas corrompidas/ausentes, confirmando o diagnóstico.

## Out of scope

| Excluded | Why |
| --- | --- |
| Execução de inferência ou amostragem pesada de UMAP | Realizada na execução empírica do cluster |
| Alteração manual de cores sem especificação de contraste | Mantém a paleta publication-ready existente |

## Assumptions

| Assumption | Chosen default | Rationale | Confirmed? |
| --- | --- | --- | --- |
| Formato de F5 | `network_edges` com coordenadas `x1, y1, x2, y2` | Representa graficamente a topologia da rede de coocorrência calculada em `short/08` | y |
| Resolução de exportação de figuras | Mínimo de 180 DPI para painéis e 300 DPI para figuras finais | Garante nitidez para submissão editorial conforme padrões da BMC | y |

**Open questions:** none - all resolved or logged above.

## Observable

| Surface | Decision | Landing |
| --- | --- | --- |
| Module `viral_cnn.reporting.transformations:transform_figure_3_paired_motif_perturbation` | exigência estrita das 5 variantes de espécies | AC 1, AC 2, AC 3 |
| Module `viral_cnn.reporting.transformations:transform_figure_5_motif_cooccurrence` | exigência das coordenadas `x1, y1, x2, y2` | AC 4, AC 5 |
| Module `viral_cnn.reporting.publication:render_figures` | renderização geométrica de F3 e F5 | AC 1, AC 6 |
| Module `viral_cnn.reporting.figures:verify_figure_publication_readiness` | checklist e validação de resolução/dimensões | AC 7, AC 8, AC 9, AC 10 |

## Sources

- `PLANO_CORRECOES_CNN_METAGENOME_AGENTS_2026-11-15.md` - Planejamento de correções pré-empíricas e definição de escopo da spec C05
- `docs/publication_data_contracts.md` - Matriz executável de contratos de dados de publicação
- `.specs/STATE.md` - Decisões AD-032, AD-033, AD-035, AD-043, AD-044, AD-045, AD-046
