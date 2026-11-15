Created At: 2026-11-15T13:17:00-03:00
Completed At: 2026-11-15T13:17:00-03:00
Verdict: PASS
Status: PASS
Profile: light
Round: 1 - full
Verifier: independent sub-agent (author != verifier)
Plan: `.specs/features/c05-figuras-f3-f5-selecao-legibilidade/plan.md`
Checks: `.specs/features/c05-figuras-f3-f5-selecao-legibilidade/checks.md`

# C05 — Figuras F3 e F5, Selecao e Legibilidade verification

## Scope Resolution

A verificação independente avaliou as alterações implementadas para a especificação C05 no repositório `cnn_metagenome`. O escopo cobre a verificação estrita de completude das 5 variantes de espécies na Figura 3, a exigência de coordenadas geométricas 2D para o grafo de coocorrência na Figura 5, e a rotina de checklist de legibilidade e prontidão para publicação das figuras F1–F5.

- **Arquivos modificados no escopo:**
  - `src/viral_cnn/reporting/transformations.py`: `transform_figure_3_paired_motif_perturbation` (exige exatamente as 5 variantes de espécies `ResNet50_Clean`, `ConvNeXt_Z`, `ConvNeXt_T`, `ResNet50_F`, `ConvNeXt_F`, rejeitando qualquer omissão com `PublicationValidationError`); `transform_figure_5_motif_cooccurrence` (exige colunas `x1, y1, x2, y2` de `edge_table.csv`, rejeitando dados tabulares unidimensionais).
  - `src/viral_cnn/reporting/figures.py`: `_inspect_png` (inspeção de assinatura de cabeçalho PNG e resolução largura/altura via IHDR) e `verify_figure_publication_readiness` (gera checklist estruturado inspecionando integridade de arquivos, dimensões, disclaimer qualitativo de F2 e dicionários de seleção de F1–F5).
  - `src/viral_cnn/reporting/__init__.py`: exportação da função `verify_figure_publication_readiness`.
  - `src/viral_cnn/reporting/publication.py`: `render_figures` com validação de integridade dos painéis $a$–$e$ de F3 e traçado geométrico de arestas e nós em F5.
  - `tests/test_c05_figure_readiness.py`: suíte com 10 testes unitários independentes para verificação das exigências.
- **Conformidade de fronteiras:** nenhuma alteração excedeu os limites estabelecidos no plano de especificação.

## Verification Results

Todos os 10 checks foram executados e validados de maneira autônoma com o interpretador do ambiente virtual `.venv/bin/python` e `MPLCONFIGDIR=/tmp/mpl`:

| Check | Description | Command Executed | Evidence | Proof | Result |
| --- | --- | --- | --- | --- | --- |
| C1 | `transform_figure_3_paired_motif_perturbation` valida dados com as 5 variantes de espécies para os painéis a–e (AC 1) | `MPLCONFIGDIR=/tmp/mpl .venv/bin/python -m unittest -k test_f3_accepts_five_variants tests/test_c05_figure_readiness.py` | `tests/test_c05_figure_readiness.py:32` | Ran 1 test in 0.002s (OK) | PASS |
| C2 | `transform_figure_3_paired_motif_perturbation` rejeita dados com variantes ausentes com PublicationValidationError (AC 2) | `MPLCONFIGDIR=/tmp/mpl .venv/bin/python -m unittest -k test_f3_rejects_missing_variants tests/test_c05_figure_readiness.py` | `tests/test_c05_figure_readiness.py:44` | Ran 1 test in 0.002s (OK) | PASS |
| C3 | `render_figures` valida que nenhum painel de F3 contenha variantes ausentes ou preenchimento artificial (AC 3) | `MPLCONFIGDIR=/tmp/mpl .venv/bin/python -m unittest -k test_f3_rendering_rejects_incomplete_panels tests/test_c05_figure_readiness.py` | `tests/test_c05_figure_readiness.py:57` | Ran 1 test in 0.409s (OK) | PASS |
| C4 | `transform_figure_5_motif_cooccurrence` exige a presença das coordenadas 2D `x1, y1, x2, y2` de `edge_table.csv` (AC 4) | `MPLCONFIGDIR=/tmp/mpl .venv/bin/python -m unittest -k test_f5_accepts_valid_edge_coordinates tests/test_c05_figure_readiness.py` | `tests/test_c05_figure_readiness.py:95` | Ran 1 test in 0.001s (OK) | PASS |
| C5 | `transform_figure_5_motif_cooccurrence` rejeita dados tabulares sem coordenadas 2D com PublicationValidationError (AC 5) | `MPLCONFIGDIR=/tmp/mpl .venv/bin/python -m unittest -k test_f5_rejects_missing_coordinates tests/test_c05_figure_readiness.py` | `tests/test_c05_figure_readiness.py:108` | Ran 1 test in 0.001s (OK) | PASS |
| C6 | `render_figures` traça segmentos de arestas e nós a partir das coordenadas geométricas `x1, y1, x2, y2` de F5 (AC 6) | `MPLCONFIGDIR=/tmp/mpl .venv/bin/python -m unittest -k test_f5_renders_network_graph tests/test_c05_figure_readiness.py` | `tests/test_c05_figure_readiness.py:118` | Ran 1 test in 0.821s (OK) | PASS |
| C7 | `verify_figure_publication_readiness` inspeciona a assinatura PNG e as dimensões de cada figura retornando checklist estruturado (AC 7) | `MPLCONFIGDIR=/tmp/mpl .venv/bin/python -m unittest -k test_verify_figure_readiness_valid_png tests/test_c05_figure_readiness.py` | `tests/test_c05_figure_readiness.py:155` | Ran 1 test in 0.031s (OK) | PASS |
| C8 | `verify_figure_publication_readiness` reporta falha se algum arquivo de figura estiver ausente ou corrompido com zero bytes (AC 8) | `MPLCONFIGDIR=/tmp/mpl .venv/bin/python -m unittest -k test_verify_figure_readiness_detects_missing_or_corrupt tests/test_c05_figure_readiness.py` | `tests/test_c05_figure_readiness.py:166` | Ran 1 test in 0.002s (OK) | PASS |
| C9 | `verify_figure_publication_readiness` verifica a presença de advertência qualitativa nos metadados da projeção F2 (AC 9) | `MPLCONFIGDIR=/tmp/mpl .venv/bin/python -m unittest -k test_verify_figure_readiness_checks_f2_disclaimer tests/test_c05_figure_readiness.py` | `tests/test_c05_figure_readiness.py:181` | Ran 1 test in 0.011s (OK) | PASS |
| C10 | `verify_figure_publication_readiness` valida que a seleção de F1–F5 contenha `seed`, `ids`, `sampling` e `filters` (AC 10) | `MPLCONFIGDIR=/tmp/mpl .venv/bin/python -m unittest -k test_verify_figure_readiness_validates_selection_parameters tests/test_c05_figure_readiness.py` | `tests/test_c05_figure_readiness.py:199` | Ran 1 test in 0.012s (OK) | PASS |

- **Execução consolidada da suíte C05:**
  - Comando: `MPLCONFIGDIR=/tmp/mpl .venv/bin/python -m unittest tests/test_c05_figure_readiness.py`
  - Resultado: 10 testes executados em 1.013s, 0 falhas, 0 erros.
- **Execução da suíte de regressão de reporting de publicação (C01–C05 + M07):**
  - Comando: `MPLCONFIGDIR=/tmp/mpl .venv/bin/python -m unittest tests/test_c05_figure_readiness.py tests/test_c04_t11_scientific_decision.py tests/test_c03_publication_adapters.py tests/test_c02_derived_manifests.py tests/test_c01_data_contracts.py tests/test_reporting_figures.py tests/test_m07_publication_contracts.py`
  - Resultado: 79 testes executados em 4.175s, 0 falhas, 0 erros.

## Boundary Conditions Checked

1. **One-Way Door 1 — Autenticidade estrita dos 5 modelos de espécies em F3:**
   - O contrato científico de F3 veta a exclusão ou preenchimento com zero de qualquer uma das 5 variantes de espécies (`ResNet50_Clean`, `ConvNeXt_Z`, `ConvNeXt_T`, `ResNet50_F`, `ConvNeXt_F`).
   - `transform_figure_3_paired_motif_perturbation` confere a presença integral do conjunto e levanta `PublicationValidationError` quando qualquer variante está ausente (comprovado em C1 e C2).
   - `render_figures` valida que os painéis configurados correspondam exatamente aos dados presentes no artefato, proibindo renderização parcial (comprovado em C3).

2. **One-Way Door 2 — Exigência imperativa de coordenadas geométricas 2D em F5:**
   - O contrato científico de F5 exige representação em grafo de rede com coordenadas bidimensionais (`x1, y1, x2, y2`) oriundas de `edge_table.csv` gerada em `short/08`.
   - Substitutos unidimensionais (como gráficos de barras) são rejeitados com `PublicationValidationError` (comprovado em C4 e C5).
   - O renderizador traça explicitamente segmentos de arestas e nós a partir das coordenadas geométricas informadas (comprovado em C6).

3. **One-Way Door 3 — Checklist automatizado de legibilidade e integridade de figuras:**
   - A rotina `verify_figure_publication_readiness` valida a integridade binária de arquivos PNG (assinatura `\x89PNG\r\n\x1a\n`, dimensões de largura e altura positivas via decodificação do chunk IHDR), reportando falhas em arquivos ausentes ou corrompidos de tamanho zero (comprovado em C7 e C8).
   - Para a Figura F2, a presença do disclaimer qualitativo ("qualitative") nos metadados é verificada formalmente (comprovado em C9).
   - Para as figuras F1–F5, os parâmetros de seleção declarados nos metadados devem conter as chaves `seed`, `ids`, `sampling` e `filters` (comprovado em C10).

## Non-Functional Properties Swept

- **Validation:** Cobertura em C2, C5, C8 e C10 assegurando que artefatos incompletos ou dados com esquemas discrepantes sejam interceptados antes da geração editorial.
- **Failure modes:** Comprovado em C2, C5 e C8 mediante emissão de `PublicationValidationError` estruturada e registro explícito das mensagens de falha.
- **Idempotency:** A verificação e renderização geram saídas determinísticas a partir de artefatos e sementes fixadas.
- **Authorization:** Não aplicável; a manipulação é restrita ao sistema de arquivos local do workspace.
- **Concurrency:** Não aplicável; execução sequencial em pipeline de publicação.
- **Data lifecycle:** Imagens e manifestos são gravados no diretório de destino designado sem exclusão temporal arbitrária.
- **Dependency failure:** Tratado em C2, C5 e C8 diante de arquivos ausentes, corrompidos ou colunas omitidas.
- **State transitions:** Status derivados de figuras (`DERIVED_VERIFIED`) são consistentes com os contratos do validador de seleção.
- **Observability:** As funções registram os caminhos absolutos dos artefatos processados (`[LOG_PATH]`) e retornam checklists estruturados para auditoria.

## Residual Risks & Recommendations

- **Ambiente de execução headless:** A renderização com matplotlib exige configuração adequada de backend não interativo (`Agg`) e variável `MPLCONFIGDIR` apontando para diretório com permissão de escrita em nós de computação sem home gravável.
- **Interpretação de projeções 2D (F2):** Reitera-se a recomendação editorial de que métricas quantitativas (KNN, Silhouette) sejam sempre calculadas e reportadas no espaço de alta dimensão, mantendo o UMAP restrito a inspeção qualitativa com disclaimer explícito.
- **Publicação final e metadados de proveniência:** Recomenda-se manter a geração de `figure_render_manifest.json` com os hashes SHA-256 dos artefatos fontes e parâmetros de plotagem para garantia de rastreabilidade integral.
