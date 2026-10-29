# r04-real-latent-embeddings-and-taxonomy-audit — Plan

## Problem

Na auditoria dos notebooks de reprodução `notebooks/reproduction/short/05_ictv_taxonomy_curation.ipynb` e `06_ictv_latent_space_and_hierarchy.ipynb`, bem como no módulo `src/viral_cnn/eval/latent.py` e nos artefatos em `results_reproduction/latent_space/`:
1. **Manifesto de curadoria poluído com registros sintéticos (Notebook `05`, Célula 4):** A Célula 4 cria 100 registros sintéticos inventados (`mock_records`) mesmo quando `Smoke Test Mode: False`, gravando `"genome_disjoint_test_classes": 25` no manifesto de produção (`results_reproduction/taxonomy/manifest.json`), sugerindo erroneamente que os fragmentos do teste possuem proveniência de genoma mapeada.
2. **Cópia espúria de `KNN_Top1` para `KNN_Top5` em `latent.py`:** A função `evaluate_cosine_knn` em `src/viral_cnn/eval/latent.py` calcula apenas o voto majoritário (Top-1) e não computa a fração de acertos no Top-5. Como consequência, o notebook `06` atribui `"KNN_Top5": knn_res["global_knn_accuracy"]`, duplicando identicamente o valor de Top-1 na Tabela 5.
3. **Representações latentes sintéticas e dimensão incorreta de ConvNeXt (Notebook `06`, Células 3–7):** O notebook não carrega checkpoints reais nem dados de teste, gerando tensores aleatórios com `np.random.randn` mesmo fora do modo smoke. Além disso, assume dimensão `dim = 768` para as três variantes de ConvNeXt, quando a arquitetura real `ConvNeXt_Tiny_1D` possui `in_features = 2048` em `fc` (idêntica à `ResNet50_1D`). Essa simplificação gera `N_taxa = 3` e valores `NaN` para correlações hierárquicas na Tabela 6.
4. **Violação da política Fail-Fast:** Quando `is_smoke_test == False`, o notebook `06` recai silenciosamente em dados aleatórios em vez de abortar com `FileNotFoundError` caso os checkpoints ou dados reais estejam ausentes.

Quando esta especificação for concluída, `evaluate_cosine_knn` calculará `KNN_Top1` e `KNN_Top5` genuínos, o notebook `05` manterá o manifesto empírico livre de registros sintéticos, e o notebook `06` usará a dimensão real de 2048 para todos os 5 modelos, extrairá embeddings reais via `extract_latent_embeddings` com suporte a cache Parquet e falhará imediatamente se executado fora do modo smoke sem os artefatos reais.

## Flow

Reutiliza `build_registered_model`, `load_model_weights`, `extract_latent_embeddings`, `evaluate_silhouette_with_audit`, `calculate_taxonomic_correlations`, `LinearProbeClassifier` e `export_umap_projection` existentes em `src/viral_cnn/`.

1. `src/viral_cnn/eval/latent.py` (existing) -> aprimora `evaluate_cosine_knn` (door 1) para computar de forma vetorizada `global_knn_top5_accuracy` conjuntamente com `global_knn_accuracy`.
2. `notebooks/reproduction/short/05_ictv_taxonomy_curation.ipynb` (existing) -> condiciona a geração de `mock_records` ao modo smoke (door 2), registrando no manifesto empírico `"genome_provenance_complete": False` e `"genome_disjoint_test_classes": None`.
3. `notebooks/reproduction/short/06_ictv_latent_space_and_hierarchy.ipynb` (existing) -> ajusta `dim = 2048` para os 5 modelos (door 3), implementa extração real de embeddings e fail-fast fora do modo smoke, preenche `KNN_Top1` e `KNN_Top5` com valores distintos na Tabela 5, computa correlações hierárquicas na Tabela 6 e executa a sonda linear e projeção UMAP.
4. out: `results_reproduction/latent_space/` com `table5_latent_clustering_metrics.csv`, `table6_taxonomic_rank_correlations.csv`, `umap_projection_convnext_f.csv` e `manifest.json`.

## Impact

| Front | What changes |
| --- | --- |
| latent evaluation | `src/viral_cnn/eval/latent.py` calcula `global_knn_top5_accuracy` em `evaluate_cosine_knn` |
| taxonomy curation | `05_ictv_taxonomy_curation.ipynb` isola `mock_records` no modo smoke e registra ausência de `genome_id` no manifesto empírico |
| latent analysis notebook | `06_ictv_latent_space_and_hierarchy.ipynb` usa `dim = 2048`, extrai embeddings reais com fail-fast fora do smoke, e calcula Tabelas 5 e 6 sem duplicação |

## Relations

None - no stored-data shape change

## Surface

None - nothing consumed outside

## Landing

| One-way door | Literal shape | Alternative rejected |
| --- | --- | --- |
| Door 1: Cálculo vetorizado de KNN Top-5 cosseno em `evaluate_cosine_knn` | `(topk_labels == y_query_batch.unsqueeze(1)).any(dim=1)` no loop em batches de `evaluate_cosine_knn`, retornando `"global_knn_top5_accuracy": float(top5_acc)` com garantia `global_knn_top5_accuracy >= global_knn_accuracy` | Copiar o valor de `KNN_Top1` para `KNN_Top5` ou utilizar busca exaustiva não-vetorizada em Python |
| Door 2: Isolamento de dados sintéticos de genoma no manifesto de curadoria | Se `ctx.is_smoke_test`, rodar `mock_records`; se `not ctx.is_smoke_test`, ler `results_reproduction/retraining_decision/decision.json` (ou registrar `"genome_provenance_complete": False`, `"genome_disjoint_test_classes": None`) | Gravar 25 espécies fictícias no manifesto de produção do cluster |
| Door 3: Dimensão pré-classificador 2048 e extração empírica de embeddings no Notebook `06` | Configurar `dim = 2048` para ResNet50 e todas as 3 variantes de ConvNeXt; em `not ctx.is_smoke_test`, instanciar modelos via `build_registered_model`, carregar pesos e chamar `extract_latent_embeddings` com fail-fast via `FileNotFoundError` | Manter `dim = 768` para ConvNeXt e síntese aleatória incondicional `np.random.randn` |

- Nothing else in this change is hard to reverse

## Criteria

### S1: Cálculo vetorizado de KNN Top-1 e Top-5 cosseno (P1)

Garante cálculo exato e simultâneo de Top-1 (voto majoritário) e Top-5 (presença do rótulo correto entre os 5 vizinhos mais próximos).

**Acceptance Criteria**

1. WHEN `evaluate_cosine_knn` is executed with query and reference representations THEN the function SHALL return a dictionary containing both `"global_knn_accuracy"` and `"global_knn_top5_accuracy"` with float values in $[0.0, 1.0]$.
2. WHEN `evaluate_cosine_knn` computes accuracies on any input dataset THEN the function SHALL guarantee that `"global_knn_top5_accuracy"` is greater than or equal to `"global_knn_accuracy"`.
3. WHEN `models_to_evaluate` is defined for latent space evaluation THEN all 5 evaluated species models (ResNet50, ResNet50_F, ConvNeXt_Z, ConvNeXt_T, ConvNeXt_F) SHALL specify pre-classifier embedding dimension of 2048.

**Independent test:** Executar testes unitários em `tests/test_latent_space.py` e `tests/test_n04_taxonomy_and_latent_space.py` validando o retorno de `global_knn_top5_accuracy >= global_knn_accuracy` e ausência de self-neighbors.

### S2: Extração de embeddings e conformidade empírica no Notebook `06` (P1)

Garante extração real de embeddings, conformidade das Tabelas 5 e 6 e política fail-fast fora do modo smoke.

**Acceptance Criteria**

1. WHEN `is_smoke_test == False` in `notebooks/reproduction/short/06_ictv_latent_space_and_hierarchy.ipynb` THEN the notebook SHALL extract latent representations via `extract_latent_embeddings` from registered models and test DataLoaders without invoking `np.random.randn`.
2. WHEN `is_smoke_test == False` and any required checkpoint or test partition does not exist THEN the notebook SHALL raise `FileNotFoundError` immediately.
3. WHEN Table 5 is generated by notebook `06` THEN the output DataFrame SHALL contain distinct `"KNN_Top1"` and `"KNN_Top5"` columns and SHALL save `table5_latent_clustering_metrics.csv`.
4. WHEN Table 6 is generated by notebook `06` THEN the output DataFrame SHALL report hierarchical correlations with valid parent taxon counts and SHALL save `table6_taxonomic_rank_correlations.csv`.

**Independent test:** Executar testes automatizados em `tests/test_n04_taxonomy_and_latent_space.py` validando execução rápida em smoke mode, disparo de `FileNotFoundError` sob `SMOKE_TEST=0` sem artefatos e integridade das colunas das Tabelas 5 e 6.

### S3: Saneamento de registros de curadoria no Notebook `05` (P1)

Garante que o manifesto de curadoria reflita a proveniência real dos dados sem registros sintéticos fora do modo smoke.

**Acceptance Criteria**

1. WHEN `is_smoke_test == False` in `notebooks/reproduction/short/05_ictv_taxonomy_curation.ipynb` THEN the notebook SHALL record `"genome_provenance_complete": False` and `"genome_disjoint_test_classes": None` in `manifest.json`.
2. WHEN executed in smoke test mode (`is_smoke_test == True`) THEN notebook `05` and notebook `06` SHALL complete execution end-to-end within 60 seconds without errors.

**Independent test:** Executar testes em `tests/test_n04_taxonomy_and_latent_space.py` verificando a integridade do manifesto e dos artefatos Parquet gerados.

## Out of scope

| Excluded | Why |
| --- | --- |
| Execução completa dos 1,7M fragmentos de teste em GPU | Restrita à execução no cluster via `REMOTE_SERVER=true` conforme AD-006 e AD-017 |
| Modificação de notebooks originais em `short/` | Preservação estrita dos notebooks históricos conforme `AGENTS.md` e `AD-004` |
| Retreinamento de modelos de classificação | O objetivo desta especificação é a análise de representações latentes e taxonomia, não treinamento |

## Assumptions

| Assumption | Chosen default | Rationale | Confirmed? |
| --- | --- | --- | --- |
| Dimensão latente pré-classificador do ConvNeXt | 2048 para `ConvNeXt_Tiny_1D` e `ResNet50_1D` | O estágio 4 do ConvNeXt 1D possui 2048 canais alimentando `self.fc` conforme AD-012 | y |
| Cache de representações latentes pré-extraídas | Suportar carregamento de `results_reproduction/latent_space/features_<model>.parquet` se existente | Evita re-extração computacionalmente cara no cluster se os tensores já tiverem sido gerados | y |
| Amostragem para cálculo de Silhouette em larga escala | Amostrar até 5000 instâncias estratificadas quando executando em dados reais | O cálculo de Silhouette tem complexidade quadrática $O(N^2)$ em distância e memória | y |

**Open questions:** none - all resolved or logged above.

## Observable

None - no user-facing surface

## Sources

- `ROADMAP_CORRECAO_AUDITORIA.md` (Seção 3, `R04 — r04-real-latent-embeddings-and-taxonomy-audit`) - define as correções para `KNN_Top5`, dimensões de 2048 e saneamento de manifestos.
- `src/viral_cnn/eval/latent.py` - implementação de KNN cosseno, Silhouette e correlações taxonômicas.
- `notebooks/reproduction/short/05_ictv_taxonomy_curation.ipynb` e `06_ictv_latent_space_and_hierarchy.ipynb` - pipelines de curadoria e análise de espaço latente.

