# r06-real-baselines-and-full-overlap-integration — Plan

## Problem

Na auditoria do notebook de reprodução `notebooks/reproduction/short/09_benchmark_cnns_kraken2_blast_overlap.ipynb`, dos módulos em `src/viral_cnn/baselines/` e dos artefatos em `results_reproduction/baselines/`:
1. **Overlap de 0,00% por amostragem insuficiente no Notebook `09` (Célula 3):** Quando executado fora de smoke test, a Célula 3 amostra apenas `max_rows=1000` do teste e `max_rows=5000` do treino ($0{,}036\%$ dos $13{,}65\text{ M}$ de fragmentos de treino), resultando em `0 (0.00%)` de sobreposição exata — aparentando contradizer os `95,94%` do teste completo (notebook 10) e os `95,91%` ($262.470 / 273.660$) do notebook histórico `short/benchmark_cnns_kraken2_blast_overlap.ipynb`.
2. **Predições fabricadas no Notebook `09` (Célula 5):** Mesmo quando `Smoke Test Mode: False`, a Célula 5 ignora completamente os arquivos reais gerados no diretório do cluster `/home/mlima/workspace/cnn_kraken/benchmark_kraken_blast_20_per_species/` e fabrica as predições de todos os 7 métodos (`ResNet50`, `ResNet_F`, `ConvNeXt_Z`, `ConvNeXt_T`, `ConvNeXt_F`, `Kraken2-flat` e `BLASTN`) aplicando máscaras aleatórias `rng.uniform(0, 1, size=num_queries) > acc_target` sobre `y_true_all`.
3. **Ausência de carregador padronizado de benchmark de baselines em `src/viral_cnn/baselines/`:** Não existe função que valide e carregue os artefatos preparados de baselines (`exact_overlap.csv`, `comparison_results/predictions_*.parquet` e `runtime.jsonl`), exigindo alinhamento estrito de `query_id`, estratificação em `all`, `overlap` e `no_exact_overlap`, e tratamento uniforme de abstenções (`-1`) no denominador total $N$.

Quando esta especificação for concluída, `src/viral_cnn/baselines/overlap.py` fornecerá `load_prepared_baseline_benchmark`, o notebook `09` consumirá diretamente os artefatos empíricos do cluster com fail-fast estrito (`FileNotFoundError`) se ausentes fora de smoke mode, eliminando qualquer fabricação de predições com `acc_target`, e o teste automatizado verificará todos os contratos científicos.

## Flow

Reutiliza `audit_exact_overlap`, `evaluate_predictions`, `paired_discordance`, `bootstrap_species_difference` e `KmerCentroidClassifier` existentes em `src/viral_cnn/baselines/`.

1. `src/viral_cnn/baselines/overlap.py` (existing) -> implementa `load_prepared_baseline_benchmark` (door 1) para ler `exact_overlap.csv` (ou `truth.parquet`), validar a unicidade e o alinhamento de `query_id`, derivar máscaras booleanas para `all`, `overlap` e `no_exact_overlap`, e carregar as matrizes de predição e top-5 de `comparison_results/predictions_<method>_noise_<pct>.parquet` para todos os 7 métodos e 5 níveis de ruído.
2. `notebooks/reproduction/short/09_benchmark_cnns_kraken2_blast_overlap.ipynb` (existing) -> Célula 3 e Célula 5 utilizam `load_prepared_baseline_benchmark` quando `not ctx.is_smoke_test`, levantando `FileNotFoundError` imediatamente se o diretório de baselines estiver ausente (door 2), preservando a geração determinística de 40 consultas sob smoke test.
3. `tests/test_n06_kraken2_blastn_benchmark.py` (existing) -> expande testes de contrato para verificar alinhamento estrito de `query_id`, detecção de anomalias/incompatibilidades, fail-fast fora do smoke mode e execução íntegra do bootstrap agrupado por espécie.
4. out: `results_reproduction/baselines/` com `exact_overlap_audit.csv`, `benchmark_metrics_stratified.csv`, `paired_discordance.csv`, `species_cluster_bootstrap.csv`, `kmer_baseline_metrics.csv`, `figure_benchmark_overlap.png` e `manifest_09_baselines.json`.

## Impact

| Front | What changes |
| --- | --- |
| baseline benchmark loader | `src/viral_cnn/baselines/overlap.py` e `__init__.py` ganham `load_prepared_baseline_benchmark` com validação de alinhamento e suporte a aliases de métodos |
| baseline benchmark notebook | `09_benchmark_cnns_kraken2_blast_overlap.ipynb` elimina fabricação sintética por `acc_target` e amostragem de 1000 linhas fora de smoke mode, falhando com `FileNotFoundError` se arquivos reais estiverem ausentes |
| test contracts | `tests/test_n06_kraken2_blastn_benchmark.py` adiciona contratos para `load_prepared_baseline_benchmark` e fail-fast em ambiente de produção |

## Relations

None - no stored-data shape change

## Surface

None - nothing consumed outside

## Landing

| One-way door | Literal shape | Alternative rejected |
| --- | --- | --- |
| Door 1: Carregador padronizado de benchmark de baselines | `load_prepared_baseline_benchmark(benchmark_dir: Union[str, Path], methods: Optional[List[str]] = None, noise_levels: Optional[List[float]] = None) -> BenchmarkArtifacts` validando unicidade de `query_id`, alinhamento estrito e retornando `audit_df`, `strata_masks`, `predictions`, `top_candidates` e `runtime` | Carregar CSVs e Parquets ad-hoc dentro de cada célula do notebook sem validação de alinhamento |
| Door 2: Fail-fast estrito no Notebook 09 fora do smoke mode | Na Célula 3 e Célula 5 do Notebook 09, se `not ctx.is_smoke_test` e `kraken_blast_dir` não existir, levantar `FileNotFoundError` imediatamente, proibindo geração sintética ou uso de `acc_target` | Gerar predições aleatórias silenciosas com `rng.uniform` simulando acurácias arbitrárias fora do smoke mode |

- Nothing else in this change is hard to reverse

## Criteria

### S1: Carregamento robusto e validação de artefatos de baselines (`load_prepared_baseline_benchmark`) (P1)

Garante leitura confiável, validação de unicidade de consultas e alinhamento estrito de predições entre a verdade/overlap e as saídas das CNNs, Kraken2 e BLASTN.

**Acceptance Criteria**

1. WHEN `load_prepared_baseline_benchmark` is called with a directory containing `exact_overlap.csv` and `comparison_results/predictions_<method>_noise_<pct>.parquet` files THEN the function SHALL return a `BenchmarkArtifacts` object containing `audit_df` with boolean `has_overlap`, `strata_masks` for `all`, `overlap`, and `no_exact_overlap`, and aligned prediction arrays for all methods.
2. WHEN any prediction file in `load_prepared_baseline_benchmark` has duplicate `query_id` values or IDs mismatched with the truth table THEN the function SHALL raise `ValueError`.
3. WHEN `benchmark_dir` passed to `load_prepared_baseline_benchmark` does not exist THEN the function SHALL raise `FileNotFoundError`.

**Independent test:** Executar testes unitários em `tests/test_n06_kraken2_blastn_benchmark.py` validando `load_prepared_baseline_benchmark` sob diretórios válidos, corrompidos e inexistentes.

### S2: Eliminação de predições fabricadas e reprodução de overlap no Notebook 09 (P1)

Garante fail-fast em ambiente de produção sem dados reais e execução limpa com métricas estratificadas completas.

**Acceptance Criteria**

1. WHEN `is_smoke_test == False` in `notebooks/reproduction/short/09_benchmark_cnns_kraken2_blast_overlap.ipynb` THEN the notebook SHALL load real benchmark artifacts via `load_prepared_baseline_benchmark` from `ctx.topology.paths["short"]["kraken_blast_dir"]` and SHALL raise `FileNotFoundError` if the directory does not exist.
2. WHEN `is_smoke_test == False` in notebook 09 THEN Cell 5 SHALL NOT execute random prediction generation with `acc_target` or `rng.uniform`.
3. WHEN executed in smoke test mode (`is_smoke_test == True`) THEN notebook 09 SHALL execute end-to-end within 60 seconds producing 40 balanced queries with exactly 20 overlap and 20 non-overlap queries and exporting all required CSVs, figures, and manifests.

**Independent test:** Executar `PYTHONPATH=src .venv/bin/python3 -m unittest tests/test_n06_kraken2_blastn_benchmark.py` verificando a execução rápida em smoke mode e o fail-fast sob `is_smoke_test = False`.

## Out of scope

| Excluded | Why |
| --- | --- |
| Re-execução do BLASTN e Kraken2 no cluster | O escopo restringe-se à ingestão, validação de alinhamento e auditoria estatística dos artefatos gerados |
| Modificação de `short/benchmark_cnns_kraken2_blast_overlap.ipynb` | Preservação estrita dos notebooks legados conforme `AGENTS.md` e `AD-004` |
| Alteração de hiperparâmetros de inferência de CNN | As avaliações consomem os checkpoints congelados de 13.683 espécies de acordo com AD-001 e AD-008 |

## Assumptions

| Assumption | Chosen default | Rationale | Confirmed? |
| --- | --- | --- | --- |
| Estrutura de diretórios de benchmark | `benchmark_kraken_blast_20_per_species/` contém `exact_overlap.csv` e `comparison_results/predictions_*.parquet` | Formato documentado e produzido pelo notebook histórico `short/benchmark_cnns_kraken2_blast_overlap.ipynb` | y |
| Nomes e aliases dos métodos | Suporta nomes canônicos (`resnet50_species`, `convnext_noise_species`, etc.) e variantes históricas (`ResNet50`, `ConvNeXt_F`, etc.) | Garante compatibilidade transparente com artefatos históricos do cluster e com a taxonomia do artigo | y |
| Tratamento de abstenção em baselines | Rótulos não-classificados ou sem hit recebem `-1` e contam como erro no denominador total $N$ | Invariante metodológico estrito definido em AD-011 | y |

**Open questions:** none - all resolved or logged above.

## Observable

None - no user-facing surface

## Sources

- `ROADMAP_CORRECAO_AUDITORIA.md` (Seção 3, `R06 — r06-real-baselines-and-full-overlap-integration`) - define os requisitos para carregamento dos artefatos do cluster, eliminação de predições fabricadas e bootstrap pareado.
- `short/benchmark_cnns_kraken2_blast_overlap.ipynb` (Células 3, 6, 7, 15, 16, 18, 20, 30) - código fonte histórico de geração de `exact_overlap.csv`, inferência e salvamento de `comparison_results`.
- `src/viral_cnn/baselines/overlap.py` e `src/viral_cnn/baselines/paired.py` - módulos de métricas de sobreposição, Wilson CI e bootstrap por espécie.
