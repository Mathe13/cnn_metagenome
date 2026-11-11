# M06 — Sequências Longas e Análise de Estratos (long/01–04)

## Problem

A cadeia de execução de sequências longas (`notebooks/reproduction/long/{01,02,03,04}_*.ipynb`) contém acoplamentos a caminhos estáticos e lacunas metodológicas que impedem uma reprodução rastreável e imutável sob o ciclo de vida M00 (`m00_run_plan_v1`):

1. **Acoplamento a caminhos estáticos e run IDs hardcoded:** A primeira célula de `long/03_cnn_long_voter_compare.ipynb` força `os.environ["C02_LONG_RUN_ID"] = "c02-20261109T012615-2e5a72b2"`, impedindo que uma execução nova consuma dinamicamente a saída de `long/02`. O notebook `long/04_long_motifs_and_gc_analysis.ipynb` grava diretamente no diretório estático legado `results_reproduction/long_voter`. Além disso, em `notebook_runtime.py:NOTEBOOK_REQUIREMENTS`, `long/03` aponta para um manifesto fixo e `long/04` não declara dependência dos motivos aprendidos em `short/07`.
2. **Seleção arbitrária e motivos fixos em long/04:** O notebook `long/04` seleciona a primeira linha do Parquet (`df_long["sequence"].iloc[0]`) e monitora quatro 7-mers fixos (`["CCCCCCC", "GGGGGGG", "ATATATA", "TTTTTTT"]`), desconsiderando os motivos reais extraídos e validados na etapa `short/07` (M04). Não há critérios explícitos de amostragem de contigs exemplares por tamanho ou teor de GC.
3. **Ausência de figuras de estratos e comparação pareada de agregação:** A análise de estratos atual não gera as figuras exigidas pelo roadmap de reprodução: curva/barras de acurácia por faixa de comprimento e número de janelas com tamanho amostral $N$ visível, e a comparação pareada entre voto majoritário e média de logits pré-softmax por variante.
4. **Demarcação causal e terminologia científica (`AD-010`, `AD-020`):** Preparações históricas rotularam a queda de acurácia de ConvNeXt como deficiência arquitetural comprovada e referiram-se a contigs simulados como "long-reads". Conforme `AGENTS.md` e `AD-020`, os contigs do NCBI avaliados não possuem anotação de genoma independente (proveniência genômica `unknown`), a perturbação uniforme não simula perfis contextuais de indels de tecnologias de sequenciamento (ONT), e diagnósticos de estratos constituem observações descritivas ou hipóteses (`HYPOTHESIS` / `INCONCLUSIVE`), nunca prova causal de superioridade arquitetural sem retreinamento pareado controlado.

Esta entrega implementa a fase M06 do roadmap de 09/11/2026, isolando os quatro notebooks da cadeia sob o protocolo `m06_long_strata_audit_v1`, consumindo dinamicamente os artefatos de inferência e os motivos aprendidos em 07, implementando a amostragem determinística de contigs, gerando as figuras adicionais de estratos e agregadores, e assegurando total conformidade com `AGENTS.md` e `AD-020`.

## Flow

Reutiliza o runtime M00/M01, o manifesto atômico, o registro de modelos de espécie, o `LongSequenceVoter`, as funções de auditoria e estratificação de `viral_cnn.inference.audit_long` e `viral_cnn.baselines.long_eval`, e o extrator de motivos de `viral_cnn.interpretability.motifs`, mantendo estritamente os invariantes de janela de 512 bp, stride 256 bp e codificação 4 canais A/C/G/T com vetor nulo para bases desconhecidas/padding.

1. Ambiente, configuração e plano M00 entram em `m06_workflow` (new), que reutiliza `notebook_runtime` e `run_planning` (exists): resolver entradas explícitas, parâmetros de controle, semente 42 e reserva de diretórios exclusivos para `long/01`, `long/02`, `long/03` e `long/04` (door 1).
2. Entradas do NCBI longo e split de treino entram em `preflight_notebook("long/01", ...)` e `long/01` (exists/updated): quantificar sobreposição de k-mers ($k=15$) com o treino de espécies e emitir `manifest_01_ncbi_prep.json` e `long_database_preparation_audit.csv` (door 2).
3. Contigs preparados e os cinco checkpoints de espécie entram em `preflight_notebook("long/02", ...)` e `long/02` (exists/updated): executar inferência por janelas deslizantes (512 bp, stride 256 bp, `handle_short="pad"`) com `LongSequenceVoter` computando tanto `mean_logits` (média antes de softmax) quanto `majority_vote`, exportando predições pareadas e `manifest_02_voter_inference.json` (door 2).
4. Manifestos e predições pareadas de `long/02` entram em `preflight_notebook("long/03", ...)` e `long/03` (exists/updated): resolver dinamicamente o `run_id` de `long/02` (sem ID hardcoded), alinhar consultas estritamente nos cinco modelos e computar a Tabela 12 (`table_12_long_voter_comparison.csv`) e estratificação de erros (`stratified_long_errors.csv`) (door 3).
5. Estratos de comprimento, número de janelas e predições pareadas entram nos geradores de figura em `viral_cnn.reporting.figures` (new/updated): exportar `figure_long_accuracy_by_length_and_windows.png` e `figure_long_paired_vote_vs_mean_logits.png`, ambos com tamanhos amostrais $N$ explicitamente declarados (door 4).
6. Diagnósticos de sensibilidade a logits e tamanho de genoma entram em `build_diagnostic_finding` (exists): registrar achados com status `HYPOTHESIS` ou `INCONCLUSIVE`, demarcando ausência de prova causal e proveniência genômica `unknown` em `long_voter_diagnosis.json` e `manifest_03_voter_compare.json` (door 4).
7. Contigs longos e motivos validados de `short/07` (M04) entram em `preflight_notebook("long/04", ...)` e `long/04` (exists/updated): resolver dinamicamente motivos top de 07 ou fallback representativo, amostrar deterministicamente contigs exemplares por critérios predefinidos (semente 42, comprimento $\ge 2000$ bp), computar teor de GC e ativação de motivos em janelas deslizantes e exportar `long_contig_motif_summary.csv`, `figure_long_motifs_gc.png` e `manifest_04_long_motifs.json` (door 5).
8. Runtime grava manifestos atômicos `manifest.json` para cada etapa sob `results_reproduction/runs/<run_id>/long/<stage>/` com protocolo `m06_long_strata_audit_v1`, status `COMPLETE` ou `FAILED`, hashes das entradas/saídas e vínculo M00 (door 1).

## Impact

| Front | What changes |
| --- | --- |
| arquivos previstos | Notebooks `notebooks/reproduction/long/{01,02,03,04}_*.ipynb`; novos módulos `src/viral_cnn/m06_workflow.py` e configuração `configs/reproduction_m06.yaml`; atualizações em `notebook_runtime.py`, `reporting/figures.py` e `reporting/tables.py`; testes em `tests/test_m06_long_sequences_strata.py`. |
| saídas | Cada etapa grava em namespace exclusivo `results_reproduction/runs/<run_id>/long/<stage>/` (01, 02, 03, 04). O diretório estático legado `results_reproduction/long_voter` não é sobrescrito. |
| números afetados | Tabela 12 reflete população rigorosamente pareada entre os cinco modelos Species; estratos reportam suporte $N$ por faixa de comprimento (<512, 512–2000, 2000–5000, 5000–10000, >10000) e número de janelas; `long/04` reporta métricas de contigs amostrados deterministicamente. |
| cadeia científica | NCBI contigs + 5 modelos Species → `long/01` (overlap) → `long/02` (inferência pareada vote + mean logits) → `long/03` (T12 + estratos + figuras de acurácia/estratos e voto vs média de logits + diagnóstico hipótese) → `long/04` (motivos 07 + amostragem determinística + perfil GC/motivos). |
| compatibilidade | O validador de publicação C06 consome `table_12.csv` derivado de `table_12_long_voter_comparison.csv` via transformação auditada sem divergências de colunas. |
| demarcação científica | Diagnósticos de queda de ConvNeXt permanecem explicitamente como hipóteses observacionais (`HYPOTHESIS`); genomas sem accession independente são classificados como `unknown`; contigs simulados são referidos como *sequências longas / contigs* e não *long-reads* reais. |

## Relations

`None - no stored-data shape change`

## Surface

None - nenhuma API HTTP ou serviço de rede externa. Interfaces locais dos notebooks e CLI:

- `notebooks/reproduction/long/01_ncbi_download_and_prepare_database.ipynb`: consome `DATA_ROOT`, `SMOKE_TEST`, plano M00 opcional; produz `manifest_01_ncbi_prep.json`, `long_database_preparation_audit.csv` e `manifest.json`.
- `notebooks/reproduction/long/02_cnn_long_voter_inference.ipynb`: consome contigs preparados de 01 e os cinco checkpoints de espécie; produz predições pareadas `.parquet`, `manifest_02_voter_inference.json` e `manifest.json`.
- `notebooks/reproduction/long/03_cnn_long_voter_compare.ipynb`: consome predições de 02 (com `run_id` resolvido dinamicamente via ambiente, plano M00 ou diretório pai); produz `table_12_long_voter_comparison.csv`, `stratified_long_errors.csv`, `figure_long_accuracy_by_length_and_windows.png`, `figure_long_paired_vote_vs_mean_logits.png`, `long_voter_diagnosis.json`, `manifest_03_voter_compare.json` e `manifest.json`.
- `notebooks/reproduction/long/04_long_motifs_and_gc_analysis.ipynb`: consome contigs de NCBI/01 e os motivos validados de `short/07` (M04); produz `long_contig_motif_summary.csv`, `figure_long_motifs_gc.png`, `manifest_04_long_motifs.json` e `manifest.json`.
- `configs/reproduction_m06.yaml`: configuração YAML declarando protocolo `m06_long_strata_audit_v1`, faixas de estratificação, sementes e parâmetros de amostragem de contigs.
- Status de manifesto: `COMPLETE` ou `FAILED`.

Precedência de resolução: (1) plano M00 explícito quando fornecido; (2) variáveis de ambiente explícitas (`M06_LONG_RUN_ID`, `C02_LONG_RUN_ID`, `DATA_ROOT`); (3) arquivo de configuração `configs/reproduction_m06.yaml`; (4) topologia padrão do cluster / diretório de dados.

## Landing

| One-way door | Literal shape | Alternative rejected |
| --- | --- | --- |
| 1 — ciclo de vida e adoção M00 em long/01–04 | `protocol_id=m06_long_strata_audit_v1`, `notebook_id=long/<stage>`, `run_id`, `is_smoke_test`, semente 42, `git_commit`, hashes dos artefatos de entrada e produtos. Reserva atômica via `reserve_run_directory`. | Usar caminhos estáticos em `results_reproduction/long_voter`: sobrescreve execuções prévias e corrompe rastreabilidade de artefatos. |
| 2 — avaliação pareada estrita dos 5 modelos | Exigir predições sobre exatamente o mesmo conjunto de consultas e ordem de classes canônica para as 5 variantes (`resnet_clean_species`, `resnet_noise_species`, `convnext_zero_species`, `convnext_transfer_species`, `convnext_noise_species`), computando tanto `mean_logits` quanto `majority_vote`. | Avaliar apenas 1 variante ou permitir conjuntos de consultas díspares entre modelos: invalida comparação da Tabela 12. |
| 3 — resolução dinâmica de dependências em long/03 | Resolver o diretório e manifesto de 02 via `M06_LONG_RUN_ID`, `C02_LONG_RUN_ID`, plano M00 ou descoberta do run de 02 mais recente no namespace ativo, falhando com mensagem clara se ausente. | Forçar `os.environ["C02_LONG_RUN_ID"] = "c02-20261109T012615-2e5a72b2"` na célula 1 do notebook 03: quebra execuções autônomas de novos runs. |
| 4 — estratificação com suporte N e figuras científicas | Exportar `stratified_long_errors.csv` por faixa de comprimento (<512, 512–2000, 2000–5000, 5000–10000, >10000) e janelas (1, 2–10, 11–50, >50) com suporte $N$, gerando `figure_long_accuracy_by_length_and_windows.png` e `figure_long_paired_vote_vs_mean_logits.png`, classificando causas de queda como `HYPOTHESIS`. | Afirmar causalidade arquitetural intrínseca sem retreinamento ou omitir $N$: induz leitores a conclusões não sustentadas por evidência empírica. |
| 5 — integração dinâmica de motivos e amostragem em long/04 | Resolver os 7-mers validados de `short/07` (M04) e amostrar deterministicamente contigos exemplares por critérios predefinidos (semente 42, comprimento $\ge 2000$ bp, diversidade GC), salvando metadados em `long_contig_motif_summary.csv`. | Selecionar a primeira linha do Parquet (`head(1)`) e motivos fixos arbitrários `["CCCCCCC", ...]` sem rastreabilidade aos filtros do modelo. |

## Criteria

### S1: Isolamento de execução, adoção M00 e avaliação pareada dos 5 modelos (P1)

**Acceptance Criteria**

1. The system SHALL reserve collision-free, isolated directories under `results_reproduction/runs/<run_id>/long/<stage>/` for stages `long/01`, `long/02`, `long/03`, and `long/04` under protocol `m06_long_strata_audit_v1`.
2. WHEN `long/01` executes THEN the system SHALL audit input long sequences and calculate k-mer overlap against training fragments with $k=15$, saving `long_database_preparation_audit.csv` and `manifest_01_ncbi_prep.json`.
3. WHEN `long/02` executes in empirical mode THEN the system SHALL verify that all five Species checkpoints exist and evaluate predictions for all five variants on identical long queries, computing both `mean_logits` and `majority_vote` with `handle_short="pad"`.
4. IF any required input checkpoint or parquet is missing in empirical mode THEN the system SHALL raise FileNotFoundError and write an atomic failure manifest with status `FAILED`.
5. WHEN `long/02` finishes successfully THEN the system SHALL write paired prediction parquet files and emit `manifest_02_voter_inference.json` and `manifest.json` with status `COMPLETE`.

**Independent test:** verificar preflight e reserva de diretórios; testar falha atômica em modo empírico quando faltar checkpoint; testar persistência pareada dos cinco modelos em modo smoke.

### S2: Tabela 12, resolução dinâmica de 02 e estratos com suporte N (P1)

**Acceptance Criteria**

6. WHEN `long/03` executes THEN the system SHALL dynamically resolve the `run_id` of `long/02` without hardcoded run IDs and verify query identity and class ordering alignment across all five models before computing Table 12.
7. WHEN Table 12 is computed THEN the system SHALL export `table_12_long_voter_comparison.csv` containing overall accuracy and coverage for both `mean_logits` and `majority_vote` across all five model variants.
8. WHEN `long/03` performs stratification THEN the system SHALL compute prediction error and accuracy across five length bins (`[0, 512)`, `[512, 2000)`, `[2000, 5000)`, `[5000, 10001)`, `[10001, inf)`) and window count bins, exporting `stratified_long_errors.csv` with the sample count $N$ explicitly reported for each stratum.
9. WHEN `long/03` visualizes strata and aggregators THEN the system SHALL generate `figure_long_accuracy_by_length_and_windows.png` displaying accuracy across length and window strata with $N$ indicated, and `figure_long_paired_vote_vs_mean_logits.png` displaying paired delta accuracy and agreement between voting and mean logits.
10. WHEN `long/03` evaluates mechanisms for ConvNeXt accuracy drop THEN the system SHALL categorize all mechanism findings as `HYPOTHESIS` or `INCONCLUSIVE` in `long_voter_diagnosis.json`, explicitly disclaiming causal architectural claims without controlled multi-seed retraining.

**Independent test:** verificar resolução dinâmica do run ID sem variáveis estáticas em 03; checar alinhamento pareado dos cinco modelos na Tabela 12; validar presença de $N$ nos estratos e nas novas figuras de acurácia e agregação.

### S3: Integração de motivos de 07, amostragem e perfis de contigs em 04 (P1)

**Acceptance Criteria**

11. WHEN `long/04` executes THEN the system SHALL dynamically load top learned 7-mer motifs from stage `short/07` (M04) or a verified fallback, and sample contigs deterministically with seed 42 and length threshold $\ge 2000$ bp rather than taking the first file row.
12. WHEN `long/04` processes sampled contigs THEN the system SHALL slice sequences into 512 bp windows (stride 256 bp) and export `long_contig_motif_summary.csv` containing window positions, GC content percentage, and learned motif counts.
13. WHEN `long/04` finishes THEN the system SHALL render `figure_long_motifs_gc.png` showing the dual-panel profile of GC percentage and motif occurrences, and emit `manifest_04_long_motifs.json` and `manifest.json` with status `COMPLETE`.
14. WHILE executing in smoke test mode (`SMOKE_TEST=1`) THEN all long stages SHALL execute completely on CPU fixtures within 15 seconds, setting `is_smoke_test: true` in all generated manifests.
15. WHEN reporting and documenting long-sequence results THEN the system SHALL refer to evaluated sequences as *long-sequences* or *contigs*, reserving *long-reads* exclusively for authentic sequencer read data.
16. WHEN artifacts are loaded or persisted across all M06 workflows THEN the system SHALL emit console and log messages with prefix `[LOG_PATH]` indicating the absolute path, respecting `LOG_PATH=0/false`.

**Independent test:** testar amostragem determinística e consumo de motivos de 07 em 04; validar geração de `figure_long_motifs_gc.png`; validar demarcação de hipótese nos diagnósticos e rastreabilidade `[LOG_PATH]`.

## Out of scope

Product capabilities only. Process and harness rules live in AGENTS.md or as Observable `n/a`.

| Excluded | Why |
| --- | --- |
| Treinamento ou fine-tuning de novos modelos para sequências longas | M06 é estritamente uma etapa de inferência, auditoria de agregação e estratificação sobre checkpoints congelados. |
| Simulação de erros estocásticos de sequenciamento ONT com indels contextuais | O procedimento de avaliação avalia fatiamento de contigs em janelas; simulação avançada de sequenciadores requer ferramentas dedicadas (ex.: Badread) fora do escopo do artigo. |
| Atribuição causal definitiva da queda de ConvNeXt à arquitetura | Diferenças em contigs sem retreinamento controlado multi-semente permanecem hipóteses científicas e diagnósticos observacionais. |

## Assumptions

Defaults that are not already a numbered criterion. Drop a row once it is.

| Assumption | Chosen default | Rationale | Confirmed? |
| --- | --- | --- | --- |
| Limiar de comprimento para amostragem de contigs exemplares em 04 | Comprimento $\ge 2.000$ bp | Garante número mínimo de janelas deslizantes ($\ge 7$ janelas com stride 256 bp) para traçar perfis contínuos de GC e motivos. | y |
| Quantidade de contigs exemplares amostrados em 04 | 3 contigs representativos com variação de GC | Permite inspeção de perfis sob diferentes faixas de composição sem sobrecarregar artefatos visuais. | y |
| Faixas de número de janelas para estratificação em 03 | 1 janela, 2–10 janelas, 11–50 janelas, >50 janelas | Captura transições de contigs curtos a genomas virais completos. | y |

**Open questions:** none - all resolved or logged above.

## Observable

Worksheet, not the review. `n/a` needs its reason. One row may group the same decision across several routes.

| Surface | Decision | Landing |
| --- | --- | --- |
| notebook `long/01_ncbi_download_and_prepare_database.ipynb` | empty / missing NCBI inputs state | AC 2, AC 4 |
| notebook `long/01_ncbi_download_and_prepare_database.ipynb` | successful execution / manifest complete | AC 1, AC 2 |
| notebook `long/02_cnn_long_voter_inference.ipynb` | missing checkpoints / error state | AC 3, AC 4 |
| notebook `long/02_cnn_long_voter_inference.ipynb` | successful paired inference across 5 models | AC 3, AC 5 |
| notebook `long/03_cnn_long_voter_compare.ipynb` | dynamic resolution of 02 run ID without hardcoding | AC 6 |
| notebook `long/03_cnn_long_voter_compare.ipynb` | Table 12 export with paired models | AC 7 |
| notebook `long/03_cnn_long_voter_compare.ipynb` | strata calculation with sample size N | AC 8 |
| figures `figure_long_accuracy_by_length_and_windows.png` & `figure_long_paired_vote_vs_mean_logits.png` | visualization of strata and aggregators with N | AC 9 |
| artifact `long_voter_diagnosis.json` | causal disclaimer and hypothesis status | AC 10 |
| notebook `long/04_long_motifs_and_gc_analysis.ipynb` | dynamic loading of 07 motifs and deterministic contig sampling | AC 11 |
| artifact `long_contig_motif_summary.csv` | windowed GC and motif activation summary | AC 12 |
| figure `figure_long_motifs_gc.png` | dual-panel GC and motif occurrence rendering | AC 13 |
| all workflows in smoke test mode | deterministic CPU execution under 15s | AC 14 |
| scientific terminology in reports and notebooks | long-sequences/contigs terminology | AC 15 |
| all file read/write operations | logging path traceability | AC 16 |

## Sources

- `ROADMAP_MELHORIAS_REPRODUCTION_2026-11-09.md` (M06 — Sequências longas e análise de estratos) - define requisitos para long/01–04, eliminação de caminhos fixos, novas figuras de estratos e agregação pareada, amostragem determinística em 04 e demarcação de hipótese.
- `AGENTS.md`, `AD-010` e `AD-020` - fixam invariantes de inferência com `handle_short="pad"`, 512 bp/256 bp stride, proveniência genômica `unknown` na ausência de accession, demarcação não-causal de estratos e uso rigoroso do termo *sequências longas / contigs*.
- `configs/publication_c06.yaml` - define requisitos de esquema para a Tabela 12 (`table_12.csv`).
