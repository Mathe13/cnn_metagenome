# C05 — Figuras F3 e F5, Selecao e Legibilidade checks

Profile: light
Plan: `.specs/features/c05-figuras-f3-f5-selecao-legibilidade/plan.md`

10 checks in 3 slices · 3 one-way doors · 0 open, of which 0 block

## Checks

Grouped by the plan's slices; numbering runs across the whole feature.

### S1 - Verificacao estrita das 5 variantes e paineis de F3 · 2 files · 40 KB · ~10k

**C1** - A funcao `transform_figure_3_paired_motif_perturbation` valida dados com as 5 variantes de espécies para os paineis a–e (AC 1)
Proof: `.venv/bin/python -m unittest -k test_f3_accepts_five_variants tests/test_c05_figure_readiness.py`

**C2** - A funcao `transform_figure_3_paired_motif_perturbation` rejeita dados com variantes ausentes com PublicationValidationError (AC 2)
Proof: `.venv/bin/python -m unittest -k test_f3_rejects_missing_variants tests/test_c05_figure_readiness.py`

**C3** - A funcao `render_figures` valida que nenhum painel de F3 contenha variantes ausentes ou preenchimento artificial (AC 3)
Proof: `.venv/bin/python -m unittest -k test_f3_rendering_rejects_incomplete_panels tests/test_c05_figure_readiness.py`

### S2 - Exigencia de coordenadas geometricas 2D e layout de F5 · 2 files · 40 KB · ~10k

**C4** - A funcao `transform_figure_5_motif_cooccurrence` exige a presenca das coordenadas 2D `x1, y1, x2, y2` de `edge_table.csv` (AC 4)
Proof: `.venv/bin/python -m unittest -k test_f5_accepts_valid_edge_coordinates tests/test_c05_figure_readiness.py`

**C5** - A funcao `transform_figure_5_motif_cooccurrence` rejeita dados tabulares sem coordenadas 2D com PublicationValidationError (AC 5)
Proof: `.venv/bin/python -m unittest -k test_f5_rejects_missing_coordinates tests/test_c05_figure_readiness.py`

**C6** - A funcao `render_figures` traca segmentos de arestas e nós a partir das coordenadas geometricas `x1, y1, x2, y2` de F5 (AC 6)
Proof: `.venv/bin/python -m unittest -k test_f5_renders_network_graph tests/test_c05_figure_readiness.py`

### S3 - Checklist de legibilidade, resolucao e parametros de selecao · 2 files · 40 KB · ~10k

**C7** - A funcao `verify_figure_publication_readiness` inspeciona a assinatura PNG e as dimensoes de cada figura retornando checklist estruturado (AC 7)
Proof: `.venv/bin/python -m unittest -k test_verify_figure_readiness_valid_png tests/test_c05_figure_readiness.py`

**C8** - A funcao `verify_figure_publication_readiness` reporta falha se algum arquivo de figura estiver ausente ou corrompido com zero bytes (AC 8)
Proof: `.venv/bin/python -m unittest -k test_verify_figure_readiness_detects_missing_or_corrupt tests/test_c05_figure_readiness.py`

**C9** - A funcao `verify_figure_publication_readiness` verifica a presenca de advertencia qualitativa nos metadados da projecao F2 (AC 9)
Proof: `.venv/bin/python -m unittest -k test_verify_figure_readiness_checks_f2_disclaimer tests/test_c05_figure_readiness.py`

**C10** - A funcao `verify_figure_publication_readiness` valida que a selecao de F1–F5 contenha `seed`, `ids`, `sampling` e `filters` (AC 10)
Proof: `.venv/bin/python -m unittest -k test_verify_figure_readiness_validates_selection_parameters tests/test_c05_figure_readiness.py`

## Coverage

| Set (size) | Member -> proof | Unproven |
| --- | --- | --- |
| Variantes F3 (3) | f3_cinco_variantes C1 · f3_rejeicao_faltante C2 · f3_paineis_render C3 | - |
| Coordenadas F5 (3) | f5_coordenadas_validas C4 · f5_rejeicao_faltante C5 · f5_grafo_render C6 | - |
| Checklist e qualidade (4) | checklist_png_dimensoes C7 · checklist_ausente_corrompido C8 · checklist_disclaimer_f2 C9 · checklist_parametros_selecao C10 | - |

## Swept

Where each unwritten requirement landed. All nine, one line each, every time.

- validation: C2, C5, C8, C10
- failure modes: C2, C5, C8
- idempotency: existing - inspecao e renderizacao deterministas
- authorization: n/a - processamento em disco local sem autenticacao de rede
- concurrency: n/a - execucao sequencial e monotarefa
- data lifecycle: n/a - figuras salvas no diretorio de saida sem purga temporal
- dependency failure: C2, C5, C8
- state transitions: existing - status derivado auditado no pacote anterior
- observability: existing - registro em log dos caminhos de figuras geradas

## Handoff

S1-S3 = ~30k tokens acumulados (figures, publication, transformations, tests), bem abaixo do orcamento de 150k - one builder.

- Boundary: C1-C10 fechados em conjunto por um unico builder
