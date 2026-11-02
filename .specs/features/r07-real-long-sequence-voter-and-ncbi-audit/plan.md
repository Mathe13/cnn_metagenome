# r07-real-long-sequence-voter-and-ncbi-audit — Plan

## Problem

Na auditoria dos notebooks de sequências longas `notebooks/reproduction/long/01` a `04`, da topologia de runtime `src/viral_cnn/notebook_runtime.py` e dos módulos de inferência em `src/viral_cnn/inference/`:
1. **Resolução de caminhos restrita em `notebook_runtime.py` e fallback silencioso no Notebook `long/01`:** `ctx.topology.paths["long"]["ncbi_parquet_dir"]` procura apenas `/home/mlima/workspace/processed_ncbi_splited_parquet`, sem resolver nas subpastas típicas do workspace remoto (`long/`, `cnn_long/`, `cnn_kraken/`). Quando os dados não são encontrados diretamente na raiz, o notebook `01` cai silenciosamente em geração sintética de 15 sequências mesmo quando `Smoke Test Mode: False`.
2. **Uso incondicional de `MockVoterModel` no Notebook `long/02`:** A Célula 3 instancia incondicionalmente um modelo mock com 5 classes e 4 contigs artificiais (`read_001` a `read_004`), ignorando os checkpoints reais de espécies e os contigs de teste do NCBI quando executado fora de smoke test.
3. **Valores literais hardcoded na Tabela 12 e amostragem aleatória no Notebook `long/03`:** A Célula 3 transcreve a Tabela 12 em uma lista literal estática `table_12_data` desacoplada das predições avaliadas, e a Célula 5 sorteia 60 predições pseudoaleatórias com `rng.randint(0, 5, size=60)` para estratificar erros.
4. **Contig aleatório sem motivos no Notebook `long/04`:** A Célula 3 gera um contig com bases aleatórias uniformes (`rng.choice(bases, size=3000)`), resultando em zero ocorrências para os 4 motivos de 1ª camada rastreados.

Quando esta especificação for concluída, `notebook_runtime.py` fornecerá `resolve_long_artifact_path` com busca multi-candidato, os notebooks `long/01` a `04` falharão estritamente com `FileNotFoundError` fora de smoke mode quando os artefatos reais não existirem (proibindo `MockVoterModel`, `table_12_data` literal e contigs aleatórios fora de smoke mode), a Tabela 12 será derivada matematicamente a partir de DataFrames de predição, e o modo smoke continuará executando em <60s de forma determinística.

## Flow

Reutiliza `setup_notebook_runtime`, `LongSequenceVoter`, `verify_window_logit_equivalence`, `quantify_training_overlap`, `stratify_long_errors` e `scan_motif_occurrences` existentes em `src/viral_cnn/`.

1. `src/viral_cnn/notebook_runtime.py` (existing) -> implementa `resolve_long_artifact_path` (door 1) em `JupyterHubDataTopology` para resolver artefatos de sequências longas em `self.root`, `self.root / "long"`, `self.root / "cnn_long"`, `self.root / "cnn_kraken"` e `self.root.parent / "long"`.
2. `notebooks/reproduction/long/01_ncbi_download_and_prepare_database.ipynb` (existing) -> Célula 3 exige dados reais de `processed_ncbi_splited_parquet/test` fora de smoke mode com fail-fast estrito (`FileNotFoundError`) se ausentes, preservando fixtures sob smoke test.
3. `notebooks/reproduction/long/02_cnn_long_voter_inference.ipynb` (existing) -> Célula 3 e Célula 5 carregam modelos reais (`build_registered_model`) e o mapeamento de espécies fora de smoke mode, levantando `FileNotFoundError` se ausentes (door 2), reservando `MockVoterModel` estritamente para `is_smoke_test == True`.
4. `notebooks/reproduction/long/03_cnn_long_voter_compare.ipynb` (existing) -> Célula 3 e Célula 5 eliminam o dicionário hardcoded `table_12_data` e o `rng.randint`: derivam a Tabela 12 matematicamente a partir de DataFrames de predição e computam a estratificação de erros via `stratify_long_errors`.
5. `notebooks/reproduction/long/04_long_motifs_and_gc_analysis.ipynb` (existing) -> Célula 3 carrega contigs reais do teste fora de smoke mode com `FileNotFoundError` se ausentes, mantendo contig sintético determinístico em smoke mode.
6. out: `results_reproduction/long_voter/` com `long_database_preparation_audit.csv`, `predictions_long_voter_resnet50_species.csv`, `table_12_long_voter_comparison.csv`, `stratified_long_errors.csv`, `convnext_collapse_diagnosis.json`, `long_contig_motif_summary.csv`, `figure_long_motifs_gc.png` e manifestos 01–04.

## Impact

| Front | What changes |
| --- | --- |
| long artifact topology resolution | `src/viral_cnn/notebook_runtime.py` ganha `resolve_long_artifact_path` com múltiplos candidatos de diretório |
| ncbi prep notebook | `long/01_ncbi_download_and_prepare_database.ipynb` falha com `FileNotFoundError` fora de smoke mode quando o parquet não existe |
| voter inference notebook | `long/02_cnn_long_voter_inference.ipynb` proíbe `MockVoterModel` fora de smoke mode e exige checkpoints/dados reais |
| voter compare notebook | `long/03_cnn_long_voter_compare.ipynb` calcula a Tabela 12 e estratificação dinamicamente sem literais soltos ou `rng.randint` |
| motifs and gc notebook | `long/04_long_motifs_and_gc_analysis.ipynb` exige contigs reais fora de smoke mode com fail-fast se ausentes |
| test contracts | `tests/test_n07_long_sequence_voter.py` adiciona contratos para fail-fast fora do smoke mode e derivação dinâmica da Tabela 12 |

## Relations

None - no stored-data shape change

## Surface

None - nothing consumed outside

## Landing

| One-way door | Literal shape | Alternative rejected |
| --- | --- | --- |
| Door 1: Resolução multi-candidato de artefatos de sequências longas | `JupyterHubDataTopology.resolve_long_artifact_path(artifact_name: str) -> Optional[Path]` inspecionando `self.root`, subpastas `long/`, `cnn_long/`, `cnn_kraken/` e caminhos relativos ao projeto | Manter caminho rígido único `/home/mlima/workspace/processed_ncbi_splited_parquet` que falhava ao buscar subdiretórios remotos |
| Door 2: Fail-fast estrito nos notebooks long/01 a 04 fora de smoke test | Nos 4 notebooks, se `not ctx.is_smoke_test` e os arquivos reais não existirem, levantar `FileNotFoundError` imediatamente, proibindo `MockVoterModel`, `table_12_data` literal e dados aleatórios fora de smoke mode | Executar mocks silenciosos ou tabelas hardcoded estáticas fora de smoke mode |

- Nothing else in this change is hard to reverse

## Criteria

### S1: Resolução de topologia e fail-fast estrito nos notebooks long/01 a 04 (P1)

Garante resolução robusta de artefatos em cluster e impede fallbacks silenciosos ou mocks fora de smoke test.

**Acceptance Criteria**

1. WHEN `resolve_long_artifact_path` is called with an artifact name present in any of the topology candidate directories THEN the function SHALL return the resolved existing `Path`.
2. WHEN `is_smoke_test == False` in `notebooks/reproduction/long/01_ncbi_download_and_prepare_database.ipynb` and `ncbi_parquet_dir` does not exist on disk THEN Cell 3 SHALL raise `FileNotFoundError` without creating synthetic sequences.
3. WHEN `is_smoke_test == False` in `notebooks/reproduction/long/02_cnn_long_voter_inference.ipynb` and real checkpoints or test parquets are missing on disk THEN Cell 3 SHALL raise `FileNotFoundError` without instantiating `MockVoterModel`.
4. WHEN `is_smoke_test == False` in `notebooks/reproduction/long/04_long_motifs_and_gc_analysis.ipynb` and real contigs are missing on disk THEN Cell 3 SHALL raise `FileNotFoundError` without generating random nucleotide strings.

**Independent test:** Executar testes unitários em `tests/test_n07_long_sequence_voter.py` verificando a resolução de caminhos e o fail-fast sob `is_smoke_test = False`.

### S2: Derivação dinâmica da Tabela 12 e estratificação empírica de erros no Notebook long/03 (P1)

Garante eliminação de dicionários literais desacoplados e cálculo matemático exato das métricas de consenso.

**Acceptance Criteria**

1. WHEN `notebooks/reproduction/long/03_cnn_long_voter_compare.ipynb` generates Table 12 THEN the metrics (`acc_mean_logits`, `acc_majority_vote`) SHALL be computed mathematically from the prediction DataFrames.
2. WHEN `is_smoke_test == False` in `long/03_cnn_long_voter_compare.ipynb` and long-read prediction files are missing on disk THEN the notebook SHALL raise `FileNotFoundError` without executing `rng.randint` or hardcoded `table_12_data`.
3. WHEN executed in smoke test mode (`is_smoke_test == True`) THEN notebooks `long/01` a `long/04` SHALL execute end-to-end within 60 seconds each, reproducing Table 12, the error stratification, and all expected summary artifacts.

**Independent test:** Executar `PYTHONPATH=src .venv/bin/python3 -m unittest tests/test_n07_long_sequence_voter.py` verificando a derivação da Tabela 12 e a execução em smoke mode.

## Out of scope

| Excluded | Why |
| --- | --- |
| Retreinamento de backbones 1D em leituras longas | O escopo restringe-se à inferência deslizante (Voter) e auditoria de consenso sobre sequências longas |
| Modificação dos notebooks históricos em `long/` | Preservação intocada de `long/cnn-long-voter.ipynb`, `prepare database.ipynb`, etc. conforme AD-004 |
| Download direto de genomas via rede externa NCBI | Dados do NCBI já foram baixados e processados no cluster remoto conforme AD-006 |

## Assumptions

| Assumption | Chosen default | Rationale | Confirmed? |
| --- | --- | --- | --- |
| Parâmetros do LongSequenceVoter | Janela de 512 bp com stride de 256 bp e comparação `handle_short="pad"` vs `"skip"` | Protocolo documentado e validado no artigo e em `long/cnn-long-voter.ipynb` | y |
| Agregação pré-softmax vs pós-softmax | Consenso principal via média de logits antes de softmax (`mean_logits`) e voto majoritário | Invariante metodológico estrito do artigo e de AD-010 | y |
| Diagnóstico de colapso de ConvNeXt | 3 mecanismos auditados: viés de leitura de vírus gigantes, puxamento de logits pré-softmax e perda de calibração em ConvNeXt_T | Invariante científico formalizado em AD-010 e verificado no cluster | y |

**Open questions:** none - all resolved or logged above.

## Observable

None - no user-facing surface

## Sources

- `ROADMAP_CORRECAO_AUDITORIA.md` (Seção 3, `R07 — r07-real-long-sequence-voter-and-ncbi-audit`) - define os requisitos para resolução multi-candidato, eliminação de mocks e derivação da Tabela 12.
- `long/cnn-long-voter.ipynb` e `long/cnn-long-voter-compare.ipynb` - notebooks originais de inferência e comparação de consenso de sequências longas.
- `src/viral_cnn/inference/voter.py` e `src/viral_cnn/inference/audit_long.py` - módulos de inferência deslizante e estratificação de erros.
