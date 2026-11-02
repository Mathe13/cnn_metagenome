# r08-dynamic-paper-tables-and-manuscript-reconciliation — Plan

## Problem

Na auditoria do módulo de relatórios `src/viral_cnn/reporting/tables.py`, do gerador do manuscrito `src/viral_cnn/reporting/manuscript.py`, dos fontes LaTeX em `paper/` e dos artefatos em `results_reproduction/`:
1. **Tabelas de publicação com valores estáticos hardcoded:** As funções `generate_table_1_metrics()` a `generate_table_12_voter_consensus()` em `tables.py` retornam dicionários literais estáticos em vez de ingerir os CSVs e Parquets empiricamente gerados pelas specs `R01` a `R07` em `results_reproduction/`. Além disso, `generate_all_tables` não possui verificação de integridade empírica (`strict_empirical`), permitindo que artefatos de smoke test gerem tabelas como se fossem resultados de produção.
2. **Contagem de parâmetros desatualizada nas Tabelas 1 e 2:** As Tabelas 1 e 2 do manuscrito (`paper/manuscript_submission.tex`, `paper/manuscript_review.tex`) e de `tables.py` informam `23.5 M` (ResNet50) e `28.1 M` (ConvNeXt) tanto para Species quanto para Genus — valores herdados das arquiteturas 2D originais do ImageNet. Nas redes 1D reais do repositório, com dimensão pré-classificador $D = 2.048$, os parâmetros exatos são:
   - **ResNet50 1D:** Backbone = $15{,}96\text{ M}$; Cabeça Genus = $5{,}54\text{ M}$ (Total = $21{,}49\text{ M}$); Cabeça Species = $28{,}04\text{ M}$ (Total = $43{,}99\text{ M}$).
   - **ConvNeXt 1D (`HybridConvNeXt`):** Backbone = $135{,}98\text{ M}$; Cabeça Genus = $5{,}54\text{ M}$ (Total = $141{,}51\text{ M}$); Cabeça Species = $28{,}04\text{ M}$ (Total = $164{,}01\text{ M}$).
3. **Ausência de relatório formal de reconciliação de evidências:** Não existe pipeline automatizado que compare formalmente cada célula das 12 tabelas calculadas a partir dos resultados de `results_reproduction/` contra os números impressos no texto do artigo, documentando discrepâncias e confirmando a rastreabilidade.

Quando esta especificação for concluída, `tables.py` fornecerá ingestão dinâmica de artefatos de reprodução com modo `strict_empirical=True` que recusa manifestos de smoke test (`RuntimeError`), as Tabelas 1 e 2 refletirão os parâmetros exatos de backbone e cabeça em todos os artefatos e arquivos `.tex`, o manuscrito documentará formalmente a arquitetura da `HybridConvNeXt`, e será gerado o relatório de reconciliação (`results_reproduction/reconciliation_report.csv` e `.md`).

## Flow

Reutiliza `generate_all_tables`, `_serialize_table`, `build_manuscript_variants`, `generate_manuscript_content` e `compile_manuscript` existentes em `src/viral_cnn/reporting/`.

1. `src/viral_cnn/reporting/tables.py` (existing) -> atualiza `generate_table_1_metrics` e `generate_table_2_profiling` com os parâmetros reais (door 1), implementa leitura dinâmica dos CSVs/Parquets de `results_dir`, validação com fail-fast sob `strict_empirical=True` caso algum manifesto possua `is_smoke_test: true` (door 2), e emissão de `reconciliation_report.csv` e `.md`.
2. `src/viral_cnn/reporting/manuscript.py` (existing) -> atualiza o template LaTeX do manuscrito nas Tabelas 1 e 2 com as contagens exatas de parâmetros e adiciona explicação metodológica sobre o backbone de $135{,}98\text{ M}$ da `HybridConvNeXt` (estágios de 256 a 2048 canais com expansão $4\times$).
3. `paper/manuscript_submission.tex` e `paper/manuscript_review.tex` (existing) -> regenerados via `build_manuscript_variants(Path("paper"))` e compilados via `compile_manuscript`.
4. `tests/test_r08_reconciliation_and_tables.py` (new) -> suite de testes de contrato validando a rejeição estrita de smoke test sob `strict_empirical=True`, os parâmetros das Tabelas 1 e 2, a integridade do relatório de reconciliação e a compilação limpa do LaTeX.
5. out: `paper/tables/` com Tabelas 1 a 12 (formatos `.csv`, `.md`, `.tex`), `results_reproduction/reconciliation_report.csv`, `results_reproduction/reconciliation_report.md`, `paper/manuscript_submission.tex` e `paper/manuscript_submission.pdf`.

## Impact

| Front | What changes |
| --- | --- |
| reporting tables generator | `src/viral_cnn/reporting/tables.py` ingere artefatos dinâmicos de `results_dir`, valida `strict_empirical`, atualiza parâmetros para 43.99 M e 164.01 M, e gera relatório de reconciliação |
| manuscript LaTeX templates | `src/viral_cnn/reporting/manuscript.py` atualiza Tabelas 1 e 2 e detalha a arquitetura da HybridConvNeXt em Methods |
| paper sources and compiled pdf | `paper/manuscript_submission.tex` e `manuscript_review.tex` regenerados e sincronizados com parâmetros reais |
| reconciliation tests | `tests/test_r08_reconciliation_and_tables.py` adiciona contratos para strict_empirical, parâmetros e compilação do artigo |

## Relations

None - no stored-data shape change

## Surface

None - nothing consumed outside

## Landing

| One-way door | Literal shape | Alternative rejected |
| --- | --- | --- |
| Door 1: Reconciliação dos parâmetros 1D nas Tabelas 1 e 2 | Atualizar Tabelas 1 e 2 com Backbone ($15{,}96\text{ M}$ / $135{,}98\text{ M}$) e Total ($21{,}49\text{ M}$ Genus / $43{,}99\text{ M}$ Species na ResNet50 e $141{,}51\text{ M}$ Genus / $164{,}01\text{ M}$ Species na ConvNeXt) | Manter os números 2D herdados do ImageNet (23.5 M e 28.1 M) que não correspondem às redes 1D implementadas |
| Door 2: Portão estrito de integridade empírica para publicação | `generate_all_tables(..., strict_empirical=True)` verificando todos os `manifest*.json` em `results_dir` e levantando `RuntimeError` caso algum manifesto contenha `"is_smoke_test": true` | Permitir exportação de tabelas oficiais do artigo baseadas em fixtures ou execuções de teste em CPU |

- Nothing else in this change is hard to reverse

## Criteria

### S1: Reconciliação exata de parâmetros e atualização do manuscrito LaTeX (P1)

Garante sincronização rigorosa entre as arquiteturas 1D reais e as tabelas/textos do manuscrito.

**Acceptance Criteria**

1. WHEN `generate_table_1_metrics` is called THEN the returned DataFrame SHALL report total parameters of 43.99 M for ResNet50 Species, 21.49 M for ResNet50 Genus, 164.01 M for ConvNeXt Species, and 141.51 M for ConvNeXt Genus.
2. WHEN `generate_table_2_profiling` is called THEN the returned DataFrame SHALL report total species parameters of 43.99 M for ResNet50 1D and 164.01 M for ConvNeXt 1D.
3. WHEN `paper/manuscript_submission.tex` and `paper/manuscript_review.tex` are regenerated via `build_manuscript_variants` THEN Tables 1 and 2 SHALL contain 43.99 and 164.01 M parameters and SHALL NOT contain legacy 23.5 or 28.1 M values.
4. WHEN compiled via `compile_manuscript` THEN `manuscript_submission.tex` SHALL compile cleanly to PDF with exit code 0.

**Independent test:** Executar testes unitários em `tests/test_r08_reconciliation_and_tables.py` validando os parâmetros e a compilação do documento.

### S2: Ingestão dinâmica de artefatos e portão estrito strict_empirical (P1)

Garante leitura direta dos artefatos de reprodução e bloqueio de artefatos de smoke test na emissão de tabelas finais.

**Acceptance Criteria**

1. WHEN `generate_all_tables(..., strict_empirical=True)` is called on a directory containing any `manifest*.json` with `"is_smoke_test": true` THEN the function SHALL raise `RuntimeError`.
2. WHEN `generate_all_tables` is called with a populated `results_dir` THEN it SHALL read available CSV and Parquet files dynamically and SHALL emit `reconciliation_report.csv` and `reconciliation_report.md`.
3. WHEN executed with `strict_empirical=False` THEN `generate_all_tables` SHALL successfully generate all 12 publication tables in CSV, Markdown, and LaTeX formats.

**Independent test:** Executar `PYTHONPATH=src .venv/bin/python3 -m unittest tests/test_r08_reconciliation_and_tables.py` validando o bloqueio de smoke test e a geração do relatório de reconciliação.

## Out of scope

| Excluded | Why |
| --- | --- |
| Retreinamento de redes neurais | O escopo restringe-se à consolidação de métricas, parâmetros e reconciliação documental |
| Alteração de formatação visual do PDF da BMC | O layout segue rigorosamente o template LaTeX da revista e as diretrizes editoriais |

## Assumptions

| Assumption | Chosen default | Rationale | Confirmed? |
| --- | --- | --- | --- |
| Contagem de parâmetros da HybridConvNeXt | Backbone = 135.98 M, cabeça fc Species = 28.04 M, total = 164.01 M | Verificado matematicamente e empiricamente a partir dos tensores em `models/convnext.py` | y |
| Formato do relatório de reconciliação | Tabela comparando métricas do manuscrito com métricas empíricas e indicador de status (MATCH / UPDATED) | Fornece auditoria transparente exigida pelo checklist do artigo | y |
| Formatos de saída de tabelas | CSV, Markdown (.md) e LaTeX (.tex) | Garante integração com os pipelines de compilação e inspeção rápida | y |

**Open questions:** none - all resolved or logged above.

## Observable

None - no user-facing surface

## Sources

- `ROADMAP_CORRECAO_AUDITORIA.md` (Seção 3, `R08 — r08-dynamic-paper-tables-and-manuscript-reconciliation`) - define a reconciliação dos parâmetros 1D, o portão strict_empirical e o relatório de reconciliação.
- `src/viral_cnn/reporting/tables.py` e `src/viral_cnn/reporting/manuscript.py` - código de geração de tabelas e textos do manuscrito.
- `paper/manuscript_submission.tex` e `paper/manuscript_review.tex` - fontes LaTeX da submissão.

