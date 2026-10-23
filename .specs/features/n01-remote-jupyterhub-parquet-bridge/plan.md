# N01: Remote JupyterHub Parquet Bridge, Inspection Utilities, and Notebook Runtime

## Problem

Os notebooks originais do estudo em `short/` e `long/` dependem de módulos utilitários locais (`short/utils/dna_dataset_2.py`, `short/utils/gradcam.py` e `short/utils/analize_funcs.py`) e de células de configuração ad-hoc que violam contratos de reprodutibilidade científica e impedem o desacoplamento via `src/viral_cnn/`:

1. **Acoplamento e Embaralhamento Incondicional no Dataset Parquet (`short/utils/dna_dataset_2.py`):** As classes `ParquetCPUDataset` e `DaskCuDFIntegerDataset` invocam `random.shuffle(my_files)` incondicionalmente dentro de `__iter__()` usando o gerador global do módulo `random`. Na avaliação do conjunto de teste ($N=1.706.270$ fragmentos), esse embaralhamento altera a ordem das partições a cada época ou worker sem controle explícito de `shuffle=False`, dificultando comparações pareadas consulta a consulta, enquanto no treinamento multi-worker compartilha estado global mutável. Além disso, `src/viral_cnn/data/` ainda não expõe um `IterableDataset` de partições Parquet compatível com `DataLoader(..., batch_size=None)` para substituir `short/utils/dna_dataset_2.py` nos novos notebooks de reprodução.
2. **Acoplamento Arquitetural e Estado Global em Explicabilidade 1D (`short/utils/gradcam.py`):** A implementação histórica de `ScoreCAM1D._forward_from_one_hot` assume exclusivamente atributos internos da `ResNet50_1D` (`self.conv1`, `self.bn1`, `self.relu`, `self.maxpool`, `self.layer1..4`), falhando ou exigindo adaptação manual quando invocada sobre `ConvNeXt_Tiny_1D` ou modelos que já aceitam tensores 3D `(B, 4, L)` diretamente em `forward()`. Adicionalmente, as funções de visualização (`show_binary_data_as_image`, `visualize_heatmap`, `visualize_heat_map_examples_by_class`) reimportam bibliotecas no meio do arquivo, definem constantes globais (`CAM_TYPE = "GRAD"`) e forçam `plt.show()` interativo sem retornar objetos `Figure` testáveis.
3. **Inspeção de Confusão Desconectada do Mapeamento Canônico (`short/utils/analize_funcs.py`):** As rotinas `plot_confusion_matrix_insights`, `get_confused_class_samples`, `inspect_csv` e `get_class_label` reimplementam manualmente a conversão de índice 1-based (`id`) para 0-based (`model_label = id - 1`) via leitura repetida de CSV e acoplam coleta de predições com plotagem interativa, sem integração com `viral_cnn.taxonomy.mapping`.
4. **Ausência de Runtime Unificado para JupyterHub Remoto e Smoke-Test Determinístico:** Os notebooks originais espalham caminhos relativos inconsistentes (`../cnn_kraken_species/...`, `./processed_kraken_species_splited_parquet/...`, `./processed_ncbi_splited_parquet/...`) e configurações manuais de `CUDA`/`cuDF`/sementes sem emissão de manifesto rastreável (`NotebookRunManifest`) e sem fallback determinístico (`SMOKE_TEST=1`) quando executados fora do armazenamento do JupyterHub remoto (`AD-005`, `AD-006`).

Quando esta feature for entregue, `src/viral_cnn/` passará a fornecer `viral_cnn.data.parquet_dataset`, `viral_cnn.interpretability.gradcam`, `viral_cnn.eval.confusion` e `viral_cnn.notebook_runtime`, permitindo que todos os notebooks de reprodução (`N02`–`N08`) operem exclusivamente sobre `viral_cnn.*` conectados ao JupyterHub remoto ou em modo `SMOKE_TEST=1` local, mantendo `short/` e `long/` 100% intocados (`AD-004`).

## Flow

Reutiliza `src/viral_cnn/data/encoding.py` (exists) para o vocabulário canônico `A/C/G/T` (`0/1/2/3`, nulo/padding = `4`) e `OneHotEmbedding`, `src/viral_cnn/models/registry.py` (exists) para resolução de variantes e checkpoints (`REGISTERED_VARIANTS`), e `src/viral_cnn/taxonomy/mapping.py` (exists) para conversão estrita entre `taxonomic_id` (1-based) e `model_label` (0-based).

1. caminhos de partições Parquet (JupyterHub remoto ou sintéticas em `SMOKE_TEST`) -> `viral_cnn.data.parquet_dataset:ParquetBatchIterableDataset` (new, door 1) - contabiliza linhas via metadados PyArrow, particiona arquivos disjuntamente por worker e emite lotes pré-agrupados `(batch_size, 512)` em CPU (`pyarrow`/`pandas`) ou GPU (`cudf`)
2. modelo 1D registrado e lote de sequências tokenizadas -> `viral_cnn.eval.confusion` (new, door 2) - computa matriz de confusão normalizada, extrai os pares `(score, true_id, pred_id)` mais confundidos resolvidos via `viral_cnn.taxonomy.mapping` (exists) e coleta amostras reais por classe alvo
3. exemplos de classes confundidas e camada convolucional alvo -> `viral_cnn.interpretability.gradcam` (new, door 3) - calcula mapas de ativação 1D (`GradCAM1D` e `ScoreCAM1D`) compatíveis tanto com `ResNet50_1D` quanto com `ConvNeXt_Tiny_1D`, zerando posições de padding e gerando figuras comparativas
4. variáveis de ambiente (`DATA_ROOT`, `SMOKE_TEST`, `SEED`) e `configs/models.yaml` -> `viral_cnn.notebook_runtime` (new, door 4) - resolve a topologia de diretórios `short/` e `long/` no JupyterHub remoto, sintetiza partições determinísticas quando em smoke-test e persiste `NotebookRunManifest` em JSON/YAML

## Impact

| Front | What changes |
| --- | --- |
| domain | new term: `ParquetBatchIterableDataset` - dataset iterável de partições Parquet com suporte a `shuffle` explícito, RNG isolado por worker e modo `legacy_global_rng` para equivalência histórica, vive em `viral_cnn.data.parquet_dataset` |
| domain | new term: `NotebookRunManifest` - registro estruturado e auditável de execução de notebook (timestamp, git sha, versões de bibliotecas, GPU, semente, `DATA_ROOT`, modo `smoke_test`, checkpoints e hashes), vive em `viral_cnn.notebook_runtime` |
| domain | new term: `JupyterHubDataTopology` - mapeamento resolvido de caminhos de partições Parquet (`short/` e `long/`), CSVs taxonômicos e checkpoints `.pt` a partir de `DATA_ROOT`, vive em `viral_cnn.notebook_runtime` |
| stored data | nada a migrar: nenhum arquivo em `short/`, `long/` ou `dataset_kraken/` é alterado; artefatos de smoke-test e manifestos são gravados em diretórios isolados de saída ou temporários |

## Relations

None - no stored-data shape change

## Surface

None - nothing consumed outside

## Landing

| One-way door | Literal shape | Alternative rejected |
| --- | --- | --- |
| Contrato de dataset iterável Parquet pré-agrupado em lotes | `ParquetBatchIterableDataset(file_paths, padded_column_name="sequence_padded", label_column="label", maxlen=512, nucleotides=NUCLEOTIDES, batch_size=1024, backend="cpu", shuffle=False, seed=42, legacy_global_rng=False)` | `Dataset` indexado item a item com `DataLoader(batch_size=1024)` - rejeitado porque exige carregar todas as partições Parquet em RAM ou quebrar a equivalência lote a lote com `DataLoader(..., batch_size=None)` dos notebooks históricos |
| Política explícita de embaralhamento de partições Parquet | `shuffle: bool = False` por padrão para inferência/teste determinístico, `shuffle=True` com `random.Random(seed + worker_id)` para treino reprodutível, e `legacy_global_rng=True` para reprodução exata da sequência de lotes de `ParquetCPUDataset` | manter `random.shuffle(my_files)` incondicional no RNG global - rejeitado porque destrói a ordem determinística de consultas no conjunto de teste ($N=1.706.270$) e acopla o estado de sorteio ao processo global |
| Contrato de Score-CAM 1D agnóstico entre ResNet50 e ConvNeXt | `ScoreCAM1D(model, target_layer)` utilizando o tensor one-hot `(B, 4, L)` diretamente via `model(masked_input_B_C_L)` (suportado por `ResNet50_1D.forward` e `ConvNeXt_Tiny_1D.forward` quando `x.dim() == 3`) com fallback `_forward_from_one_hot` para instâncias legadas | manter `_forward_from_one_hot` chamando `model.bn1`/`model.relu` fixos - rejeitado porque pode falhar em arquiteturas cujo stem ou estágios diferem do layout clássico da ResNet |
| Topologia e manifesto de runtime de notebooks para JupyterHub remoto | `RuntimeContext = setup_notebook_runtime(data_root=None, seed=42, smoke_test=None)` retornando caminhos resolvidos (`short` e `long`), flag `is_smoke_test`, dispositivo e emitindo `NotebookRunManifest` serializável em JSON/YAML | caminhos relativos hardcoded (`../cnn_kraken_species/...`) dentro de cada notebook - rejeitado por `AGENTS.md` e `AD-006` porque quebra ao executar fora do diretório original ou em diferentes pontos de montagem do JupyterHub |

- Nothing else in this change is hard to reverse

## Criteria

### S1: Dataset Iterável de Partições Parquet com Equivalência Histórica e Divisão Disjunta (P1)

Migração de `ParquetCPUDataset` e `DaskCuDFIntegerDataset` para `src/viral_cnn/data/parquet_dataset.py` preservando o contrato de lotes pré-agrupados e adicionando determinismo estrito.

**Acceptance Criteria**

1. The system SHALL provide `ParquetBatchIterableDataset` in `src/viral_cnn/data/parquet_dataset.py` returning `(inputs_tensor, labels_tensor)` tuples of `dtype=torch.long` where `inputs_tensor` has shape `(batch_len, maxlen)` tokenized with `A=0, C=1, G=2, T=3` and `4` for padding or ambiguous bases.
2. WHEN `ParquetBatchIterableDataset` is initialized with valid Parquet file paths THEN the system SHALL compute `total_samples` by summing `pyarrow.parquet.read_metadata(f).num_rows` without loading full tables into memory and return `math.ceil(total_samples / batch_size)` from `__len__()`.
3. WHEN `ParquetBatchIterableDataset` is iterated with `shuffle=True` and `legacy_global_rng=True` under the same global Python `random.seed(42)` THEN the system SHALL yield batches numerically identical (`torch.equal`) in exact sequence to `short/utils/dna_dataset_2.py:ParquetCPUDataset`.
4. WHILE `shuffle=False` the system SHALL iterate over `file_paths` in their exact input order without invoking any random permutation, guaranteeing a deterministic query sequence across repeated runs.
5. WHEN `ParquetBatchIterableDataset` is consumed through a PyTorch `DataLoader` with `batch_size=None` and `num_workers > 1` THEN the system SHALL partition `file_paths` disjointly across workers via `get_worker_info()` such that the union of yielded samples across all workers equals `total_samples` with zero duplicated or omitted rows.
6. IF `backend="cudf"` is requested when `cudf` is not installed in the active Python environment THEN the system SHALL raise an `ImportError` (or fall back to `"cpu"` when `fallback_to_cpu=True`) with an explicit diagnostic message.

**Independent test:** `pytest tests/test_n01_parquet_bridge.py -k "parquet_dataset"` comparing `ParquetBatchIterableDataset` directly against `short/utils/dna_dataset_2.py:ParquetCPUDataset` on multi-file synthetic Parquet partitions in single-worker and multi-worker `DataLoader` modes.

### S2: Explicabilidade 1D (`GradCAM1D` e `ScoreCAM1D`) Desacoplada e Agnóstica de Arquitetura (P1)

Encapsulamento limpo de `GradCAM1D`, `ScoreCAM1D` e funções de plotagem de `short/utils/gradcam.py` em `src/viral_cnn/interpretability/gradcam.py`.

**Acceptance Criteria**

7. The system SHALL provide `GradCAM1D` and `ScoreCAM1D` in `src/viral_cnn/interpretability/gradcam.py` that register forward/backward hooks on a target 1D convolutional layer and return a 1D `np.ndarray` heatmap of shape `(seq_len,)` normalized to `[0.0, 1.0]`.
8. WHEN `GradCAM1D` or `ScoreCAM1D` is invoked on identical `ResNet50_1D` weights and input token tensor `(1, 512)` THEN the system SHALL produce a heatmap numerically matching (`np.allclose(..., atol=1e-5)`) the historical implementation in `short/utils/gradcam.py`.
9. WHEN `GradCAM1D` or `ScoreCAM1D` is invoked on `ConvNeXt_Tiny_1D` THEN the system SHALL execute the forward and masked one-hot passes without raising attribute errors and return a valid `(seq_len,)` heatmap in `[0.0, 1.0]`.
10. WHEN `visualize_heat_map_examples_by_class` processes input sequences containing padding tokens (`>= 4`, which produce all-zero columns in `OneHotEmbedding`) THEN the system SHALL zero out the heatmap attribution values at all padded positions (`is_padding`) and return a `matplotlib.figure.Figure` object without requiring interactive window display (`show=False` supported).
11. IF `visualize_heat_map_examples_by_class` is called with a `cam_type` other than `"GRAD"` or `"SCORE"` THEN the system SHALL raise a `ValueError`.

**Independent test:** `pytest tests/test_n01_parquet_bridge.py -k "gradcam"` verifying numerical equivalence with `short/utils/gradcam.py` on `ResNet50_1D`, compatibility with `ConvNeXt_Tiny_1D`, padding zero-masking, and headless figure generation.

### S3: Utilitários de Matriz de Confusão e Coleta de Pares Confundidos Integrados à Taxonomia (P1)

Migração de `plot_confusion_matrix_insights` e `get_confused_class_samples` de `short/utils/analize_funcs.py` para `src/viral_cnn/eval/confusion.py`.

**Acceptance Criteria**

12. The system SHALL provide `compute_and_plot_confusion_insights` (and alias `plot_confusion_matrix_insights`) in `src/viral_cnn/eval/confusion.py` that collects model predictions over a `DataLoader`, computes the row-normalized confusion matrix across `num_classes`, and returns the sorted list of Top-K off-diagonal confused tuples `(score, true_id, pred_id)`.
13. WHEN a taxonomy mapping CSV path is supplied to `plot_confusion_matrix_insights`, `get_class_label`, or `inspect_csv` THEN the system SHALL resolve 0-based `model_label` indices to 1-based CSV `id` taxon names using `viral_cnn.taxonomy.mapping`.
14. WHEN `get_confused_class_samples` is called with `confused_pairs_info` and `n_examples_per_class` THEN the system SHALL iterate through the `DataLoader`, collect up to `n_examples_per_class` CPU input tensors per unique `true_id`, stop early once all target quotas are met, and return a list of dictionaries matching the historical schema `{"score_principal_confusao": float, "class_true": {"id": int, "target_pred_id": int, "examples": list[Tensor]}}`.

**Independent test:** `pytest tests/test_n01_parquet_bridge.py -k "confusion"` testing Top-K off-diagonal confusion extraction, taxonomic label formatting, and sample collection against `short/utils/analize_funcs.py`.

### S4: Runtime de Notebooks para JupyterHub Remoto, Manifesto de Execução e Smoke-Test (P1)

Módulo `src/viral_cnn/notebook_runtime.py` para inicialização padronizada dos notebooks de reprodução, resolução de caminhos remotos e geração de dados sintéticos determinísticos em modo smoke-test.

**Acceptance Criteria**

15. The system SHALL provide `setup_notebook_runtime` in `src/viral_cnn/notebook_runtime.py` that seeds Python, NumPy, and PyTorch (`seed=42`), detects CPU/GPU and optional `cudf`/`pyarrow` versions, and resolves `DATA_ROOT` from explicit argument, `DATA_ROOT` environment variable, or configuration file without hardcoded personal paths.
16. The system SHALL resolve the complete JupyterHub directory and file topology for both `short/` (`processed_kraken_species_splited_parquet/{train,validation,test}`, `processed_kraken_genus_splited_parquet/{train,validation,test}`, `dataset_kraken/{species,genus}_name_ids.csv`, `benchmark_kraken_blast_20_per_species`, and all 7 registered checkpoint paths from `configs/models.yaml`) and `long/` (`processed_ncbi_splited_parquet/{train,validation,test}`, `mapping_species_final.csv`, `mapping_genus_final.csv`, `raw_database/final.csv`, `dataset_reads_sinteticos_CLEAN.parquet`, `taxonomia_recuperada.csv`).
17. WHEN `smoke_test=True` (or `SMOKE_TEST=1` in the environment, or remote Parquet directories are absent and `auto_smoke_fallback=True`) THEN `ensure_smoke_test_artifacts` SHALL generate deterministic temporary Parquet partitions for `short/` (`sequence_padded`, `label`) and `long/` (`species_id`, `genus_id`, `origin_len`, `read_len`, `sequence`) along with matching synthetic taxonomy CSVs so that reproduction notebooks can execute end-to-end in automated test gates.
18. WHEN `write_notebook_manifest` is called at the end of a notebook run THEN the system SHALL write a JSON/YAML manifest containing `notebook_id`, `timestamp_utc`, `git_commit`, `seed`, `data_root`, `is_smoke_test`, `device`, `library_versions`, `checkpoints`, `input_files`, and `metrics_summary`.
19. The system SHALL preserve all original files in `short/` and `long/` completely untouched (`git status --porcelain short/ long/` empty).

**Independent test:** `pytest tests/test_n01_parquet_bridge.py -k "runtime or preservation"` verifying topology resolution, smoke-test Parquet/CSV generation, manifest serialization, and zero modifications in `short/` and `long/`.

## Out of scope

Product capabilities only. Process and harness rules live in AGENTS.md or as Observable `n/a`.

| Excluded | Why |
| --- | --- |
| Criação dos notebooks de reprodução `notebooks/reproduction/short/01..09` e `long/01..04` | Escopo das features subsequentes `N02` a `N08` que consumirão a infraestrutura criada na `N01` |
| Modificação ou refatoração in-place de `short/utils/dna_dataset_2.py`, `short/utils/gradcam.py` ou `short/utils/analize_funcs.py` | Proibido por `AD-004` e pela Regra Invariante 1 do `ROADMAP_REPRODUCAO_NOTEBOOKS.md` para preservar a evidência histórica intacta |
| Gerenciamento de clusters Dask-CUDA multi-nó (`LocalCUDACluster` com protocolo UCX) dentro de `ParquetBatchIterableDataset` | `ParquetCPUDataset` (PyArrow/Pandas) e leitura direta `cudf.read_parquet` substituem o cliente distribuído Dask nos DataLoaders PyTorch evitando conflito de contexto CUDA em workers |

## Assumptions

Defaults that are not already a numbered criterion. Drop a row once it is.

| Assumption | Chosen default | Rationale | Confirmed? |
| --- | --- | --- | --- |
| Coluna padrão de sequência nos Parquets de fragmentos curtos | `"sequence_padded"` com fallback automático para `"padded_Representation"` ou `"sequence"` | Todos os notebooks históricos em `short/` definem `PADDED_COLUMN = 'sequence_padded'`, enquanto o texto do roadmap também menciona `padded_Representation` e `sequence`; suportar fallback automático garante compatibilidade total | y |
| Backend padrão de `ParquetBatchIterableDataset` | `"cpu"` (`pandas`/`pyarrow`) por padrão, com `"cudf"` selecionável explicitamente | Nos notebooks históricos mais recentes (`analize_convnext.ipynb`, `finetune_*.ipynb`, `benchmark_cnns_kraken2_blast_overlap.ipynb`), `ParquetCPUDataset` tornou-se o padrão para permitir `num_workers > 0` com `pin_memory=True` sem erro de inicialização CUDA | y |

**Open questions:** none - all resolved or logged above.

## Observable

Worksheet, not the review. `n/a` needs its reason. One row may group the same decision across several routes.

| Surface | Decision | Landing |
| --- | --- | --- |
| API `ParquetBatchIterableDataset` | formato de saída, tipos dos tensores e contagem de lotes | AC 1, AC 2 |
| API `ParquetBatchIterableDataset` | ordenação determinística, equivalência de shuffle e divisão multi-worker | AC 3, AC 4, AC 5 |
| API `ParquetBatchIterableDataset` | tratamento de erro quando `cudf` não está disponível | AC 6 |
| API `GradCAM1D` / `ScoreCAM1D` / `visualize_heat_map_examples_by_class` | shape do heatmap, suporte a ResNet/ConvNeXt, máscara de padding e validação de `cam_type` | AC 7, AC 8, AC 9, AC 10, AC 11 |
| API `plot_confusion_matrix_insights` / `get_confused_class_samples` | formato dos pares confundidos, resolução taxonômica e coleta limitada por classe | AC 12, AC 13, AC 14 |
| API `setup_notebook_runtime` / `write_notebook_manifest` | resolução de `DATA_ROOT`, fallback de smoke-test e schema do manifesto JSON/YAML | AC 15, AC 16, AC 17, AC 18 |
| Todos os módulos Python de `N01` | autenticação, rate limits e estados de tela | n/a - biblioteca científica local/Jupyter sem endpoints HTTP ou interface gráfica standalone |

## Sources

- `ROADMAP_REPRODUCAO_NOTEBOOKS.md` (Spec N01) - define o objetivo, escopo e critérios de aceitação para `n01-remote-jupyterhub-parquet-bridge`
- `short/utils/dna_dataset_2.py`, `short/utils/gradcam.py`, `short/utils/analize_funcs.py` - implementações históricas de referência cujo comportamento numérico deve ser preservado
- `AGENTS.md` e `.specs/STATE.md` (`AD-001` a `AD-006`) - invariantes científicos de codificação `A/C/G/T`, offset taxonômico, preservação de notebooks originais e armazenamento no JupyterHub remoto

