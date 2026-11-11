# M07 — Contratos de Publicação e Validação Local (antes do cluster)

## Problem

A cadeia de publicação e auditoria estrita C06 (`src/viral_cnn/reporting/publication.py` e `notebooks/reproduction/00_regenerate_all_paper_artifacts.ipynb`), embora implemente o protocolo `c06_strict_v1`, apresenta lacunas que impedem a validação e certificação local antes da execução completa no cluster:

1. **Ausência de transformações específicas para as fontes de dados realmente produzidas:** O arquivo `configs/publication_c06.yaml` referencia nomes de transformações como strings arbitrárias (ex.: `table_1_metrics`, `figure_3_paired_motif_perturbation`), mas não há funções de transformação registradas que convertam as saídas estruturadas dos notebooks de reprodução (`short/01–11`, `long/01–04`) nos formatos exatos esperados para as tabelas T1–T12 e figuras F1–F5.
2. **Falta de diagnósticos em lote com taxonomia clara de divergências:** Antes de disparar a montagem estrita ou processamentos pesados no cluster, o pesquisador precisa saber o estado de prontidão de cada item de publicação (T1–T12, F1–F5). Hoje, a validação apenas aponta ausência genérica (`MISSING`) ou erro genérico (`INVALID`), sem distinguir claramente entre artefatos não gerados (`FALTA`), divergência de esquema de colunas (`SCHEMA_MISMATCH`), inconsistência de linhagem/split/sementes (`IDENTITY_MISMATCH`), ou violação de contrato científico (`SCIENTIFIC_MISMATCH`).
3. **Ausência de manifestos derivados de publicação com proveniência explícita:** Quando um CSV de publicação é gerado a partir de um artefato predecessor (por exemplo, `table_8.csv` gerado de `motif_perturbation_by_variant.csv` de 07, ou `table_12.csv` gerado de `table_12_long_voter_comparison.csv` de long/03), deve ser emitido um manifesto derivado que vincule formalmente a entrada upstream, a função de transformação, o hash do código e o suporte amostral $N$, sem simular proveniência empírica inexistente.
4. **Necessidade de verificação de contratos negativos científicos rigorosos:**
   - O renderizador de publicação consome estritamente CSV/Parquet e não aceita PNGs de notebooks como fontes de tabelas ou figuras.
   - F3 exige rigorosamente os cinco modelos Species (`ResNet50_Clean`, `ConvNeXt_Z`, `ConvNeXt_T`, `ResNet50_F`, `ConvNeXt_F`), recusando artefatos que omitam ConvNeXt_F ou qualquer outra variante.
   - T11 deve rejeitar dados de coocorrência de motivos apresentados como correlação entre arquiteturas sem o devido controle nulo por permutação de nós e sem variância.
   - F5 deve exigir coordenadas geométricas explícitas de rede (`x1`, `y1`, `x2`, `y2`) quando configurada como grafo de rede, rejeitando dados de gráfico de barras.
   - O modo empírico estrito deve rejeitar qualquer dependência transitiva com `is_smoke_test: true`.

Esta entrega implementa a fase M07 do roadmap de 09/11/2026, estabelecendo o registro unificado de transformações, o diagnóstico em lote com 5 estados, a emissão de manifestos derivados e os contratos negativos rigorosos para garantir prontidão local total antes da execução no cluster.

## Flow

Reutiliza o runtime de publicação existente (`publication.py`, `tables.py`, `figures.py`), os manifestos atômicos e as decisões AD-024, AD-032, AD-033 e AD-034.

1. Configuração de publicação e artefatos de entrada entram em `transformations.py` (new) ou `publication.py` (exists/updated): selecionar a transformação específica registrada para cada item T1–T12 e F1–F5 (door 1).
2. Artefatos de saída dos estágios (`short/01..11`, `long/01..04`) entram em `execute_transformation`: aplicar a conversão de esquema e colunas, calcular $N$ e emitir o artefato de publicação intermediário (door 1).
3. Metadados upstream e parâmetros de transformação entram em `derive_publication_manifest`: gerar manifesto derivado vinculando a entrada predecessor, o SHA-256 da transformação, o protocolo e os suportes sem simular dados sintéticos como empíricos (door 3).
4. A lista completa de 17 itens entra em `diagnose_publication_readiness`: inspecionar artefato, manifesto e esquema, classificando cada item em `OK`, `FALTA`, `SCHEMA_MISMATCH`, `IDENTITY_MISMATCH` ou `SCIENTIFIC_MISMATCH`, emitindo tabela diagnóstica com fontes propostas e recomendações acionáveis (door 2).
5. Regras científicas entram nos validadores de contrato negativo: bloquear F3 sem 5 modelos, bloquear T11 com matrizes idênticas/sem teste de permutação, bloquear F5 em rede sem coordenadas e bloquear imagens PNG passadas como insumo tabular (door 4).
6. Configuração validada entra em `assemble_publication`: em modo estrito (`strict_empirical=True`), abortar se houver itens bloqueados; em modo diagnóstico (`diagnostic_only=True`), exportar a matriz diagnóstica sem falha (door 5).
7. Notebook `00_regenerate_all_paper_artifacts.ipynb` (exists/updated): executar a célula de diagnóstico em lote e a montagem isolada sob `results_reproduction/runs/<run_id>/publication/` (door 5).

## Impact

| Front | What changes |
| --- | --- |
| arquivos previstos | Novo módulo `src/viral_cnn/reporting/transformations.py`; nova configuração `configs/reproduction_m07.yaml`; atualizações em `src/viral_cnn/reporting/publication.py`, `src/viral_cnn/reporting/tables.py` e `src/viral_cnn/reporting/figures.py`; atualização do notebook `notebooks/reproduction/00_regenerate_all_paper_artifacts.ipynb`; testes em `tests/test_m07_publication_contracts.py`. |
| saídas | Pacote isolado sob `results_reproduction/runs/<run_id>/publication/` contendo `publication_diagnostic_report.csv`, `publication_diagnostic_report.md`, `evidence_matrix.csv`, `publication_manifest.json`, e diretórios `tables/` e `figures/` quando em modo completo. |
| números afetados | Tabelas T1–T12 e Figuras F1–F5 consom estritamente dados tabulares derivados das execuções reproduzíveis, eliminando qualquer dependência de constantes estáticas legadas. |
| cadeia científica | Estágios `short/01..11` e `long/01..04` → `transformations.py` → Manifestos derivados → Diagnóstico em lote C06 → Montagem estrita do artigo e reconciliação. |
| compatibilidade | O protocolo `c06_strict_v1` permanece 100% retrocompatível com a suíte de testes de `test_c06_publication_package.py`. |
| demarcação científica | PNGs de notebooks nunca são aceitos como fontes de dados; dados sem proveniência ou com premissas científicas violadas (ex.: r=1 espúrio em T11 ou F5 sem arestas) são classificados como `SCIENTIFIC_MISMATCH` e bloqueados no portão estrito. |

## Relations

`None - no stored-data shape change`

## Surface

None - nenhuma API HTTP ou serviço de rede externa. Interfaces locais Python e notebook:

- `src/viral_cnn/reporting/transformations.py`:
  - `register_transformation(name, func)`: registra uma transformação de publicação.
  - `get_transformation(name)`: recupera a função de transformação correspondente.
  - `transform_publication_item(item_id, source_df, metadata)`: executa a transformação e retorna o DataFrame transformado e o manifesto derivado.
- `src/viral_cnn/reporting/publication.py`:
  - `diagnose_publication_readiness(config_or_path, *, project_root=None)`: retorna DataFrame com status em `{"OK", "FALTA", "SCHEMA_MISMATCH", "IDENTITY_MISMATCH", "SCIENTIFIC_MISMATCH"}` e detalhes por item.
  - `assemble_publication(config_path, output_dir, strict_empirical=True, diagnostic_only=False, ...)`: montagem estrita em diretório exclusivo.
- `configs/reproduction_m07.yaml`: mapeamento canônico de fontes propostas, transformações e esquemas para os 17 itens T1–T12 e F1–F5.
- `notebooks/reproduction/00_regenerate_all_paper_artifacts.ipynb`: notebook atualizado com célula de diagnóstico em lote e montagem de pacote.
- Status do manifesto: `COMPLETE`, `FAILED` ou `DIAGNOSTIC`.

## Landing

| One-way door | Literal shape | Alternative rejected |
| --- | --- | --- |
| 1 — transformações auditadas para T1–T12 e F1–F5 | Mapeamento explícito `TRANSFORMATION_REGISTRY[item_id]` executando funções determinísticas que validam colunas de entrada e produzem o esquema canônico da publicação com cálculo de $N$. | Assumir que as saídas dos notebooks já coincidem exatamente com o esquema final do manuscrito ou copiar arquivos sem transformação auditada. |
| 2 — diagnóstico em lote com 5 estados granulares | Função `diagnose_publication_readiness` retornando `status` em `{"OK", "FALTA", "SCHEMA_MISMATCH", "IDENTITY_MISMATCH", "SCIENTIFIC_MISMATCH"}` acompanhado de `proposed_source` e `recommendation`. | Manter apenas erro binário ou exceção que interrompe no primeiro arquivo faltando, impedindo a visualização da matriz completa de prontidão. |
| 3 — emissão de manifestos derivados | Todo artefato gerado via transformação emite manifesto com `evidence_kind="derived"`, `parent_manifest_sha256`, hash do código da transformação e `is_smoke_test` herdado do progenitor. | Falsificar a proveniência gerando manifestos que fingem ser a evidência empírica original sem registrar a etapa de transformação. |
| 4 — contratos negativos científicos estritos | Rejeitar estritamente: (a) PNG de notebook como fonte; (b) F3 sem os 5 modelos de espécie; (c) T11 de correlação de arquiteturas alimentada com matriz de coocorrência sem teste de permutação; (d) F5 de rede alimentada com barras; (e) qualquer insumo com `is_smoke_test: true` sob `strict_empirical=True`. | Permitir aprovação de pacotes de publicação com premissas científicas corrompidas ou dados sintéticos disfarçados. |
| 5 — integração do notebook de montagem de pacote | Célula inicial de diagnóstico em lote em `00_regenerate_all_paper_artifacts.ipynb` exibindo a tabela com os 17 itens antes da tentativa de montagem do pacote estrito. | Deixar o notebook falhar com traceback opaco sem explicar quais itens estão pendentes antes da execução no cluster. |

## Criteria

### S1: Registro e execução de transformações específicas com manifestos derivados (P1)

**Acceptance Criteria**

1. The system SHALL provide a centralized registry of publication transformations mapping each item in T1–T12 and F1–F5 to an explicit transformation function.
2. WHEN a table transformation executes THEN the system SHALL validate the input columns, convert formats, compute support $N$, and output a DataFrame matching the publication column contract.
3. WHEN a transformation finishes THEN the system SHALL emit a derived manifest recording the parent manifest path and SHA-256, transformation name, code hash, and support $N$ with `evidence_kind="derived"`.
4. IF an input artifact is a PNG or non-tabular image file THEN the system SHALL reject it with a PublicationValidationError, requiring structured CSV or Parquet data.
5. WHILE executing in smoke test mode (`SMOKE_TEST=1`) THEN all 17 transformation functions SHALL run completely on synthetic fixtures within 15 seconds.

**Independent test:** testar registro e execução de transformações para T1–T12 e F1–F5 sobre fixtures; verificar emissão de manifestos derivados; testar rejeição de PNGs como fonte.

### S2: Diagnóstico em lote com taxonomia granular de prontidão (P1)

**Acceptance Criteria**

6. WHEN `diagnose_publication_readiness` executes THEN the system SHALL evaluate all 17 items (T1–T12, F1–F5) and return a diagnostic matrix with columns `item_id`, `status`, `diagnostic`, `proposed_source`, and `recommendation`.
7. IF an artifact or manifest path is unset or does not exist on disk THEN the system SHALL assign status `FALTA`.
8. IF an artifact exists but lacks required columns or plot coordinate parameters THEN the system SHALL assign status `SCHEMA_MISMATCH`.
9. IF an artifact exists but exhibits conflicting seed, split_id, or taxonomic class ordering THEN the system SHALL assign status `IDENTITY_MISMATCH`.
10. IF an artifact violates scientific invariants (such as F3 lacking 5 models, T11 claiming architecture correlation without permutation null test, or F5 lacking edge coordinates) THEN the system SHALL assign status `SCIENTIFIC_MISMATCH`.

**Independent test:** verificar classificação dos cinco estados (`OK`, `FALTA`, `SCHEMA_MISMATCH`, `IDENTITY_MISMATCH`, `SCIENTIFIC_MISMATCH`) com casos de teste sintéticos controlados.

### S3: Contratos negativos e integração de notebook (P1)

**Acceptance Criteria**

11. WHEN F3 is evaluated THEN the system SHALL require exactly five species model variants (`ResNet50_Clean`, `ConvNeXt_Z`, `ConvNeXt_T`, `ResNet50_F`, `ConvNeXt_F`), rejecting inputs missing ConvNeXt_F with status `SCIENTIFIC_MISMATCH`.
12. WHEN T11 is evaluated THEN the system SHALL verify that co-occurrence matrices without permutation null testing or without variance are rejected as architecture correlation tables with status `SCIENTIFIC_MISMATCH`.
13. WHEN F5 is evaluated as a network graph THEN the system SHALL verify that node and edge coordinates (`x1`, `y1`, `x2`, `y2`) are present, rejecting bar chart data with status `SCHEMA_MISMATCH` or `SCIENTIFIC_MISMATCH`.
14. WHEN `assemble_publication` runs with `strict_empirical=True` THEN the system SHALL reject any artifact or dependency with `is_smoke_test: true`, raising PublicationValidationError and marking gate `empirical="BLOCKED"`.
15. WHEN `00_regenerate_all_paper_artifacts.ipynb` executes THEN the system SHALL render the batch diagnostic table and invoke `assemble_publication` isolating outputs in `results_reproduction/runs/<run_id>/publication/`.
16. WHEN artifacts are inspected, loaded, or written across M07 workflows THEN the system SHALL log paths with prefix `[LOG_PATH]`, respecting `LOG_PATH=0/false`.

**Independent test:** testar os 5 contratos negativos (F3, T11, F5, PNG, synthetic); verificar execução do notebook em modo diagnóstico e isolamento de saídas sob `results_reproduction/runs/`.

## Out of scope

Product capabilities only. Process and harness rules live in AGENTS.md or as Observable `n/a`.

| Excluded | Why |
| --- | --- |
| Execução pesada de treinamento ou inferência de genomas no cluster | M07 estabelece os contratos de validação e transformação local pré-cluster; o processamento em massa ocorre na rodada empírica do cluster. |
| Alteração manual de números do manuscrito para coincidir com o PDF publicado | As tabelas e figuras devem ser derivadas estritamente dos artefatos produzidos pelos notebooks. |
| Substituição retrospectiva de saídas históricas por simulações sintéticas | Dados sintéticos são demarcados como tais e não são promovidos a evidência empírica. |

## Assumptions

Defaults that are not already a numbered criterion. Drop a row once it is.

| Assumption | Chosen default | Rationale | Confirmed? |
| --- | --- | --- | --- |
| Formato de serialização das tabelas de publicação | CSV, Markdown (.md) e LaTeX (.tex) | Garante legibilidade para inspeção e integração direta no manuscrito LaTeX. | y |
| Formato de renderização das figuras de publicação | PNG (180 DPI) e PDF vetorial | Atende aos requisitos de submissão de periódicos e visualização local. | y |
| Resolução padrão de fontes propostas no diagnóstico | Caminhos padrão de `results_reproduction/runs/<run_id>/` | Orienta o pesquisador sobre onde encontrar ou gerar cada artefato pendente. | y |

**Open questions:** none - all resolved or logged above.

## Observable

Worksheet, not the review. `n/a` needs its reason. One row may group the same decision across several routes.

| Surface | Decision | Landing |
| --- | --- | --- |
| `transform_publication_item` (T1–T12, F1–F5) | transformation execution and schema conversion | AC 1, AC 2 |
| `derive_publication_manifest` | derived manifest emission with lineage and support | AC 3 |
| artifact file format validation | rejection of PNG images as table/figure source | AC 4 |
| all M07 workflows under `SMOKE_TEST=1` | fast CPU execution under 15s | AC 5 |
| `diagnose_publication_readiness` | batch evaluation with 5 readiness states | AC 6, AC 7, AC 8, AC 9, AC 10 |
| Figure 3 validation | strict requirement of five species models | AC 11 |
| Table 11 validation | rejection of raw cooccurrence as architecture correlation | AC 12 |
| Figure 5 validation | network coordinates requirement (rejection of bar charts) | AC 13 |
| `assemble_publication` in strict mode | strict rejection of synthetic evidence | AC 14 |
| notebook `00_regenerate_all_paper_artifacts.ipynb` | batch diagnostic display and run directory isolation | AC 15 |
| all file operations | logging path traceability | AC 16 |

## Sources

- `ROADMAP_MELHORIAS_REPRODUCTION_2026-11-09.md` (M07 — Contratos de publicação e validação local) - define transformações de fontes, alinhamento de F3/F5, diagnóstico em lote com 5 estados e testes negativos.
- `AGENTS.md` e `AD-024` - fixam o protocolo `c06_strict_v1`, proveniência explícita, separação entre dados empíricos e sintéticos e integridade dos 17 itens.
- `AD-032`, `AD-033` e `AD-034` - fixam contratos de F3 com cinco modelos, T11/F5 com teste de permutação/rede de motivos, e T12 com população pareada de contigs.
- `configs/publication_c06.yaml` - define o esquema de configuração e os 17 itens da publicação.
