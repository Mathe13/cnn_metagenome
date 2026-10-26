# Roadmap de Specs (`tlc-spec-lean`) — Reprodução dos Notebooks com `src/viral_cnn` em JupyterHub Remoto

Este roteiro define a sequência de features sob o fluxo `tlc-spec-lean` (`PLAN → CHECKS → BUILD → VERIFY`) para **reproduzir todos os notebooks originais do estudo usando os novos módulos de `src/viral_cnn/`**, conectados aos dados Parquet e checkpoints `.pt` armazenados no **JupyterHub remoto** (`AD-006`).

---

## Regras Invariantes deste Roadmap

1. **Preservação absoluta dos notebooks originais (`AD-004` e `AGENTS.md`):**
   - Nenhum arquivo em [`short/`](file:///home/matheus/Documents/cnn_metagenome/short) ou [`long/`](file:///home/matheus/Documents/cnn_metagenome/long) será apagado, movido ou modificado.
   - Todos os novos notebooks serão criados exclusivamente em `notebooks/reproduction/short/` e `notebooks/reproduction/long/`.
2. **Desacoplamento via `src/viral_cnn/` (`AGENTS.md`):**
   - Os novos notebooks não devem importar `short/utils/*` com `import *` nem redefinir arquiteturas (`ResNet50_1D`, `HybridConvNeXt`), funções de ruído ou métricas inline. Toda lógica de modelo, codificação, ruído, inferência, métricas, espaço latente, motivos e geração de tabelas/figuras deve ser importada de `viral_cnn.*`.
3. **Execução conectada ao JupyterHub Remoto + Smoke-Test Local (`AD-005` e `AD-006`):**
   - Os caminhos para as partições Parquet (`cnn_kraken/processed_kraken_*_splited_parquet/...`, `processed_ncbi_splited_parquet/...`) e para os checkpoints `.pt` (`cnn_kraken/...`, `cnn_kraken_species/...`, `cnn_kraken_genus/...`) são resolvidos via `DATA_ROOT` e `configs/models.yaml` (`configs/default_species.yaml`, `configs/default_genus.yaml`), sem caminhos pessoais hardcoded.
   - Cada notebook terá uma célula inicial padronizada de configuração de ambiente (`DATA_ROOT`, `DEVICE`, `SEED = 42`, verificação de GPU/cuDF/PyArrow e validação de existência dos artefatos no JupyterHub remoto), injeção autônoma de `src/` em `sys.path` antes de importar `viral_cnn` (garantindo execução imediata via JupyterHub remoto sem depender de `pip install -e .` ou variáveis prévias de ambiente), e suportará um modo `SMOKE_TEST=1` (lote determinístico pequeno) para que `pytest` / `nbformat` / `nbclient` consigam provar a execução das células nos gates do `tlc-spec-lean` antes e durante a execução pesada no cluster remoto.
4. **Rastreabilidade Científica:**
   - Cada execução no JupyterHub remoto salva um manifesto JSON/YAML junto às saídas Parquet/CSV contendo: identificador do notebook, commit Git, `DATA_ROOT`, checkpoints carregados (e seus hashes/metadados), split utilizado, semente (`42`), versões de Python/PyTorch/CUDA e comparação entre o valor obtido e o valor histórico do notebook original / artigo.
5. **Rastreabilidade e Log de Caminhos (`LOG_PATH`):**
   - Toda vez que dataloaders, datasets (ex: Parquet, CSV), modelos ou checkpoints forem instanciados ou carregados, o caminho absoluto ou relativo de origem deve ser registrado explicitamente no console e no log (ex: `[LOG_PATH] Loading ... from: <path>`).
   - Esse comportamento é ativado por padrão (`LOG_PATH=true` ou ausente) e pode ser desativado definindo `LOG_PATH=0` ou `LOG_PATH=false`.

---

## Mapa Completo: Notebooks Originais → Novos Notebooks → Módulos `src/viral_cnn`

| Spec | Notebook / Módulo Original (Intocado) | Novo Notebook em `notebooks/reproduction/` | Módulos `src/viral_cnn` Utilizados | Tabelas / Figuras / Notas do Artigo |
|---|---|---|---|---|
| **N01** | `short/utils/dna_dataset_2.py`<br>`short/utils/gradcam.py`<br>`short/utils/analize_funcs.py` | *(Infraestrutura compartilhada de dados Parquet, análise de confusão/GradCAM e setup remoto)* | `viral_cnn.data` (novo `parquet_dataset.py`, `encoding.py`), `viral_cnn.models.registry`, `viral_cnn.interpretability` (novo `gradcam.py`), `viral_cnn.eval` (novo `confusion.py`), `viral_cnn.notebook_runtime` | Base para todas as execuções no JupyterHub (`short/` e `long/`) |
| **N02** | `short/benchmark.ipynb` *(profiling)*<br>`short/finetune_*_species.ipynb` *(células 5–7, 12: `evaluate_robustness`)*<br>`long/cnn-long-voter-compare.ipynb` *(células 14–33: `comparar_especialistas`)*<br>`short/analize_convnext.ipynb`<br>`short/full_loss_function.ipynb` *(células 37–49: `TaxonomyEntropyAnalyzer`)* | `notebooks/reproduction/short/01_benchmark_clean_and_noise.ipynb`<br>`notebooks/reproduction/short/02_analyze_confidence_and_calibration.ipynb` | `viral_cnn.models.registry`, `viral_cnn.eval.runner`, `viral_cnn.eval.metrics`, `viral_cnn.eval.noise`, `viral_cnn.eval.calibration` (incluindo `TaxonomyEntropyAnalyzer`), `viral_cnn.eval.confusion`, `viral_cnn.interpretability.gradcam`, `viral_cnn.taxonomy.mapping` | **Tabelas 1, 2, 3, 4, 7, Fig. 1** (Notas 19, 20, 29–37, 45; S05 e S09) |
| **N03** | `short/finetune_resnet50_species.ipynb` *(células 8–11: `train_noise_aware`)*<br>`short/finetune_convnext_species.ipynb` *(células 8–11: `train_noise_aware`)* | `notebooks/reproduction/short/03_finetune_resnet50_species.ipynb`<br>`notebooks/reproduction/short/04_finetune_convnext_species.ipynb` | `viral_cnn.training.trainer`, `viral_cnn.training.config`, `viral_cnn.training.profiler`, `viral_cnn.models.registry`, `viral_cnn.eval.noise` | Checkpoints e suporte às **Tabelas 1, 3, 4** e **S12** (Notas 21–23; S04 e S12) |
| **N04** | `short/ICTV_improve_kraken_database.ipynb`<br>`short/improve_kraken_database-Copy1.ipynb`<br>`short/ICTV_latent_space-bkp.ipynb` | `notebooks/reproduction/short/05_ictv_taxonomy_curation.ipynb`<br>`notebooks/reproduction/short/06_ictv_latent_space_and_hierarchy.ipynb` | `viral_cnn.taxonomy.mapping`, `viral_cnn.data.audit`, `viral_cnn.data.grouped_split`, `viral_cnn.eval.latent` | **Tabelas 5, 6, Fig. 2** (Notas 7–10, 24, 25, 35, 38; S02, S03 e S08) |
| **N05** | `short/full_loss_function.ipynb` *(células 1–36, 50–64: motivos 7-mer, perturbação e matriz de Jaccard)*<br>`short/codon_usage.ipynb` | `notebooks/reproduction/short/07_first_layer_motifs_and_perturbation.ipynb`<br>`notebooks/reproduction/short/08_triplet_composition_and_cooccurrence.ipynb` | `viral_cnn.interpretability.motifs`, `viral_cnn.models.registry`, `viral_cnn.data.encoding` | **Tabelas 8, 9, 10, 11, Figs. 3, 4, 5** (Notas 26–28, 39–43; S10) |
| **N06** | `short/benchmark_cnns_kraken2_blast_overlap.ipynb` | `notebooks/reproduction/short/09_benchmark_cnns_kraken2_blast_overlap.ipynb` | `viral_cnn.baselines.overlap`, `viral_cnn.baselines.paired`, `viral_cnn.baselines.taxonomy`, `viral_cnn.baselines.kmer_baseline` | **Baselines Kraken2 / BLASTN** (Notas 4–6; S07) |
| **N07** | `long/download_from_ncbi.ipynb`<br>`long/prepare database.ipynb`<br>`long/cnn-long-voter.ipynb`<br>`long/cnn-long-voter-compare.ipynb`<br>`long/long_loss_function.ipynb` | `notebooks/reproduction/long/01_ncbi_download_and_prepare_database.ipynb`<br>`notebooks/reproduction/long/02_cnn_long_voter_inference.ipynb`<br>`notebooks/reproduction/long/03_cnn_long_voter_compare.ipynb`<br>`notebooks/reproduction/long/04_long_motifs_and_gc_analysis.ipynb` | `viral_cnn.inference.voter`, `viral_cnn.inference.audit_long`, `viral_cnn.baselines.long_eval`, `viral_cnn.interpretability.motifs` | **Tabela 12** e figuras de sequências longas (Notas 11–15, 44–47; S06) |
| **N08** | *(Consolidação dinâmica das saídas Parquet/CSV/JSON dos notebooks N02–N07)* | `notebooks/reproduction/00_regenerate_all_paper_artifacts.ipynb` | `viral_cnn.reporting.tables`, `viral_cnn.reporting.figures`, `viral_cnn.reporting.manuscript`, `viral_cnn.reporting.controlled_audit` | **Tabelas 1–12, Figuras 1–5** e relatório automático de reconciliação (S11) |

---

## Sequência Detalhada de Specs (`tlc-spec-lean`)

### N01 — Ponte de Dados Parquet para JupyterHub Remoto, Utilitários de Inspeção e Runtime de Notebooks (`n01-remote-jupyterhub-parquet-bridge`)

**Objetivo:** Migrar para `src/viral_cnn/` a leitura iterável de partições Parquet em larga escala (`ParquetCPUDataset` / `DaskCuDFIntegerDataset` de `short/utils/dna_dataset_2.py`), a explicabilidade 1D (`GradCAM1D` / `ScoreCAM1D` de `short/utils/gradcam.py`), as rotinas de inspeção de pares confundidos (`plot_confusion_matrix_insights` e `get_confused_class_samples` de `short/utils/analize_funcs.py`) e o gerenciador de contexto/configuração para execução em JupyterHub remoto.

**Escopo:**
- Criar `src/viral_cnn/data/parquet_dataset.py` com `ParquetBatchIterableDataset`:
  - Suporta backend `pyarrow`/`pandas` em CPU e `cudf` opcional em GPU, retornando tensores pré-agrupados em lotes `(batch_size, 512)` compatíveis com `DataLoader(..., batch_size=None)` (contrato idêntico ao histórico).
  - Divisão disjunta de arquivos por worker via `get_worker_info()`.
  - Controle explícito `shuffle: bool = False` (no teste/avaliação, `shuffle=False` garante ordem determinística estrita das consultas para comparações pareadas e denominador constante $N=1.706.270$; no treino, `shuffle=True` usa gerador isolado `random.Random(seed + worker_id)`, além de modo `legacy_global_rng=True` para provar equivalência lote a lote com `ParquetCPUDataset`).
  - Contagem rápida via metadados PyArrow `pq.read_metadata` e tokenização compatível com `OneHotEmbedding` (`A/C/G/T = 0/1/2/3` e padding/ambíguas = `4`).
- Criar `src/viral_cnn/interpretability/gradcam.py` encapsulando `GradCAM1D`, `ScoreCAM1D` e visualização 1D de `short/utils/gradcam.py` sem estado global nem `import *`.
- Criar `src/viral_cnn/eval/confusion.py` migrando `plot_confusion_matrix_insights` e `get_confused_class_samples` de `short/utils/analize_funcs.py` integrados a `viral_cnn.taxonomy.mapping` (`load_taxonomy_mapping`, `model_label_to_taxonomic_id`).
- Criar `src/viral_cnn/notebook_runtime.py` com funções utilitárias para os notebooks de reprodução:
  - Resolução unificada de `DATA_ROOT` (via variável de ambiente `DATA_ROOT` no JupyterHub ou `.env` / YAML) para os artefatos de `short/` (`cnn_kraken/`, `cnn_kraken_species/`, `cnn_kraken_genus/`, `dataset_kraken/`) e de `long/` (`processed_ncbi_splited_parquet/`, `mapping_species_final.csv`, `mapping_genus_final.csv`, `raw_database/final.csv`, `dataset_reads_sinteticos_CLEAN.parquet`, `taxonomia_recuperada.csv`).
  - Verificação automática de checkpoints de `configs/models.yaml` no filesystem do JupyterHub.
  - Emissão automática de manifesto de execução (`NotebookRunManifest`: timestamp, git sha, versões de bibliotecas, GPU detectada, sementes, arquivos lidos e hashes).
  - Suporte transparente a `SMOKE_TEST` (quando `SMOKE_TEST=1` ou os Parquets remotos não estão montados localmente, instancia partições sintéticas temporárias determinísticas com o mesmo esquema de colunas `padded_Representation` / `sequence` e `label` para validação estrutural das células).

**Critério de Aceitação (`checks.md`):**
- Equivalência numérica exata lote a lote entre `short/utils/dna_dataset_2.py:ParquetCPUDataset` e `viral_cnn.data.parquet_dataset.ParquetBatchIterableDataset` para as mesmas partições Parquet e semente.
- Ordem determinística invariável quando `shuffle=False` e divisão disjunta provada sob `num_workers > 1` (nenhuma linha duplicada ou omitida entre workers).
- `notebook_runtime` resolve toda a topologia do JupyterHub (`short/` e `long/`) sem caminhos pessoais embutidos.

---

### N02 — Notebooks de Avaliação Limpa, Corrupção por Ruído, Entropia Hierárquica e Calibração (`n02-short-benchmark-and-calibration-notebooks`)

**Depende de:** N01.

**Objetivo:** Reproduzir e consolidar a avaliação de fragmentos curtos distribuída entre `short/benchmark.ipynb` (profiling), `short/finetune_*_species.ipynb` (células 5–7, 12: `evaluate_robustness`), `long/cnn-long-voter-compare.ipynb` (células 14–33: `comparar_especialistas`), `short/analize_convnext.ipynb` e `short/full_loss_function.ipynb` (células 37–49: `TaxonomyEntropyAnalyzer`) em `notebooks/reproduction/short/` usando `viral_cnn.models.registry`, `viral_cnn.eval.*`, `viral_cnn.interpretability.gradcam` e `viral_cnn.taxonomy.mapping`.

**Escopo:**
- Unificar em `src/viral_cnn/eval/metrics.py` (`compute_species_f1_bands`) o suporte parametrizado aos limiares da Tabela 4 da Nota 34 (`thresholds=(0.0, 0.50, 0.80, 0.90, 1.0)` → `[0.90, 1.00]`, `[0.80, 0.90)`, `[0.50, 0.80)`, `[0.00, 0.50)`) e migrar `TaxonomyEntropyAnalyzer` (das células 37–49 de `short/full_loss_function.ipynb`: entropia em *nats* $\ln 13.683 \approx 9{,}52$, agregação Species → Genus via `scatter_add_`, $\Delta H = H_{\text{species}} - H_{\text{genus}}$ e entropia normalizada de ativação pré-classificadora $H_{\text{latent}} / \ln D$, Nota 37) para `src/viral_cnn/eval/calibration.py`.
- Criar `notebooks/reproduction/short/01_benchmark_clean_and_noise.ipynb`:
  - Carrega os 5 modelos de espécie (`resnet_clean_species`, `resnet_noise_species`, `convnext_zero_species`, `convnext_transfer_species`, `convnext_noise_species`) e os 2 modelos de gênero (`resnet_genus`, `convnext_genus`) via `build_registered_model` + `DATA_ROOT`.
  - Avalia o split de teste (`processed_kraken_species_splited_parquet/test` com denominador estrito $N=1.706.270$ e `processed_kraken_genus_splited_parquet/test`) para $\epsilon \in \{0.0, 0.02, 0.05, 0.10, 0.15\}$ usando `inject_noise` / `verify_effective_noise_rate` (taxa efetiva $3\epsilon/4$, semente 42, Notas 19–20) e `evaluate_model` / `compute_classification_metrics`.
  - Calcula Top-1, Top-5, Loss, Macro-F1, Weighted-F1 (incluindo `ConvNeXt_F` limpo = 61,14% e duas casas decimais em Gênero, Notas 29–33), faixas contínuas de F1 por espécie (`compute_species_f1_bands` em $\epsilon=0{,}05$, Nota 34), comparação pareada por espécie (`compute_paired_comparison` e resolução explícita dos 5 empates em $\epsilon=0{,}10$: $11.826 + 1.202 + 5 = 13.033$, Nota 45) e perfil de eficiência computacional (`ResourceProfiler` — parâmetros com cabeça de 13.683 classes, latência e throughput, Nota 30).
  - Salva predições e métricas rastreáveis em `results_reproduction/` para alimentar as **Tabelas 1, 2, 3, 4 e Fig. 1**.
- Criar `notebooks/reproduction/short/02_analyze_confidence_and_calibration.ipynb`:
  - Reproduz as análises de `short/analize_convnext.ipynb` usando `viral_cnn.eval.confusion` (`plot_confusion_matrix_insights`, `get_confused_class_samples`), `viral_cnn.taxonomy.mapping` (`load_taxonomy_mapping`, `model_label_to_taxonomic_id`), inspeção de identidade de sequência entre pares confundidos e atribuição 1D (`GradCAM1D` / `ScoreCAM1D` de `viral_cnn.interpretability.gradcam`).
  - Executa `TaxonomyEntropyAnalyzer` (reproduzindo as células 37–49 de `short/full_loss_function.ipynb` para a **Tabela 7**, Nota 37).
  - Adiciona lado a lado as métricas formais de calibração (`viral_cnn.eval.calibration`: `compute_calibration_metrics`, `compute_reliability_diagram`, `TemperatureScaler` e `audit_model_calibration_by_noise` — ECE com 15 bins, MCE, Brier score multiclasse, NLL e Temperature Scaling ajustado exclusivamente na validação) para sustentar a **Tabela 4, Tabela 7** e a auditoria S09 (`AD-013`).

**Critério de Aceitação (`checks.md`):**
- Os dois novos notebooks executam do início ao fim (`nbclient` / `pytest`) sem erros, sem importar `short/utils/*` e mantendo todos os notebooks originais intocados.
- Todas as saídas salvam artefatos estruturados em `results_reproduction/` que alimentam diretamente as Tabelas 1, 2, 3, 4, 7 e Fig. 1.

---

### N03 — Notebooks de Treinamento e Fine-Tuning com Ruído (`n03-short-finetune-training-notebooks`)

**Depende de:** N01.

**Objetivo:** Reproduzir a rotina de treinamento com ruído (`train_noise_aware`, células 8–11) de `short/finetune_resnet50_species.ipynb` e `short/finetune_convnext_species.ipynb` usando `viral_cnn.training.*` e `viral_cnn.models.registry`.

**Escopo:**
- Estender `src/viral_cnn/training/trainer.py` (`ControlledTrainer`) e `src/viral_cnn/training/config.py` para suportar:
  - Injeção controlada de ruído (`noise_rate: float = 0.0`, usando `inject_noise` com $\epsilon=0{,}05$) tanto no passo de treino quanto na validação (`AD-009`, Notas 21–23).
  - Congelamento opcional de camadas e BatchNorm em modo `eval` (`freeze_layers` / `set_bn_eval_for_freezed_layers`) sem quebrar os testes existentes da S12.
- Criar `notebooks/reproduction/short/03_finetune_resnet50_species.ipynb` e `notebooks/reproduction/short/04_finetune_convnext_species.ipynb`:
  - Substituir o loop manual `train_noise_aware` pelo motor `viral_cnn.training.trainer` (`ControlledTrainer` com AMP, `GradScaler`, `AdamW`, `CosineAnnealingLR`, `label_smoothing=0.1` e corrupção $\epsilon=0.05$ em treino e validação).
  - Preservar dois modos explícitos configuráveis na primeira célula:
    1. `MODE = "historical_reproduction"`: usa exatamente os hiperparâmetros históricos registrados em `configs/models.yaml` (`dropout=0.5` para ResNet50 com seleção `final_training_epoch_14` → `finetune_noise_14.pt`; `dropout=0.7` para ConvNeXt com seleção `best_validation_epoch_7` → `finetune_noise_7.pt`, Notas 21–23).
    2. `MODE = "controlled_parity"`: usa o protocolo pareado de `configs/controlled/baseline_pair.yaml` (S12, `AD-016`) com telemetria de recursos (`ResourceProfiler`).
  - Modo de segurança por padrão (`DRY_RUN_EPOCHS` / `MAX_BATCHES` configurável) para evitar sobrescrever checkpoints históricos no JupyterHub remoto, salvando novos checkpoints em diretório versionado próprio (`results_reproduction/checkpoints/`).

**Critério de Aceitação (`checks.md`):**
- Notebooks executam o fluxo completo de carregamento de checkpoint base, treinamento/validação com $\epsilon=0.05$, logging de métricas e salvamento seguro de checkpoint com manifesto, sem alterar checkpoints originais.

---

### N04 — Notebooks de Curadoria Taxonômica ICTV e Espaço Latente (`n04-short-taxonomy-and-latent-space-notebooks`)

**Depende de:** N01, N02.

**Objetivo:** Reproduzir `short/ICTV_improve_kraken_database.ipynb` (consolidando `improve_kraken_database-Copy1.ipynb`) e `short/ICTV_latent_space-bkp.ipynb` usando `viral_cnn.taxonomy.*`, `viral_cnn.data.audit`, `viral_cnn.data.grouped_split` e `viral_cnn.eval.latent`.

**Escopo:**
- Criar `notebooks/reproduction/short/05_ictv_taxonomy_curation.ipynb`:
  - Reproduz a integração entre `dataset_kraken/species_name_ids.csv`, taxonomia NCBI (2025) e planilha ICTV (`ICTV_Master_Species_List_2025_MSL41.v1.xlsx`, Notas 7–8), gerando `main_taxonomy_hybrid_CLEANSED.parquet` e executando também a auditoria de proveniência do dataset e de viabilidade do split por genoma (`audit_partition_duplicates`, `audit_cross_partition_overlap` em `viral_cnn.data.audit` e `build_genome_disjoint_split` em `viral_cnn.data.grouped_split`, Notas 9–10, S02/S03).
- Criar `notebooks/reproduction/short/06_ictv_latent_space_and_hierarchy.ipynb`:
  - Extrai embeddings antes da camada classificadora (`extract_latent_embeddings` em `viral_cnn.eval.latent`) para ResNet50 (2048-d) e ConvNeXt (768-d).
  - Calcula KNN (`evaluate_cosine_knn` e `evaluate_genome_disjoint_knn`: $k=5$, distância cosseno, normalização $L_2$, split 80/20 estratificado sem auto-vizinhança, semente 42) e Silhouette no espaço original de alta dimensão (`evaluate_silhouette_with_audit`, registrando suporte de classes válidas e comparando amostragem aleatória vs estratificada, Notas 24–25).
  - Calcula agregações hierárquicas Species → Genus → Order (`calculate_taxonomic_correlations`) mantendo conjuntos válidos separados para KNN e Silhouette, reportando correlações de Pearson ($r$) e Spearman ($\rho$) com $p$-valores bilaterais e $N$ explícito (conciliando o valor de `ConvNeXt_T` sem truncar gêneros válidos, Notas 35 e 38).
  - Inclui a sonda linear treinada exclusivamente em treino/validação (`LinearProbeClassifier` / `audit_classifier_bottleneck`) para testar a hipótese de gargalo na cabeça classificadora sem vazamento do conjunto de teste (`AD-012`).
  - Gera a projeção UMAP 2D (`export_umap_projection`) identificada estritamente como visualização qualitativa (**Tabelas 5, 6 e Fig. 2**).

**Critério de Aceitação (`checks.md`):**
- Notebooks executam sem erros, preservam os invariantes de S02/S03/S08 (ausência de auto-vizinho no KNN, Silhouette em alta dimensão antes do UMAP, sonda linear disjunta do teste) e geram os artefatos das Tabelas 5, 6 e Fig. 2.

---

### N05 — Notebooks de Motivos de 1ª Camada, Coocorrência de Jaccard e Composição de Tripletos (`n05-short-motifs-and-triplets-notebooks`)

**Depende de:** N01, N02.

**Objetivo:** Reproduzir `short/full_loss_function.ipynb` (células 1–36 e 50–64) e `short/codon_usage.ipynb` usando `viral_cnn.interpretability.motifs`.

**Escopo:**
- Criar `notebooks/reproduction/short/07_first_layer_motifs_and_perturbation.ipynb`:
  - Extrai os motivos de consenso de 7-mers dos filtros `conv1` / `stem` das variantes de espécie (`extract_first_layer_motifs`), garantindo que cada painel da Fig. 3 aponte para o checkpoint exato e corrigindo a duplicação histórica do painel `(e)` (`convnext_noise_species` distinto de `resnet_noise_species`, Nota 39).
  - Executa o experimento de perturbação pareada na mesma sequência sem sobreposição (`evaluate_motif_perturbation`, `summarize_motif_control`, `audit_motif_composition_robustness`), reportando lado a lado:
    1. O controle aleatório uniforme original do artigo;
    2. O controle composicional pareado por GC (`|ΔGC| = 0`, `sample_matched_control_region`) e complexidade local (Notas 26–28, 40; S10), com teste de Wilcoxon pareado (`compute_wilcoxon_signed_rank`) e correção Benjamini-Hochberg (`multipletests_fdr_bh`).
  - Computa e salva a matriz de coocorrência de Jaccard dos motivos validados (`compute_jaccard_cooccurrence` → `results_reproduction/motifs/motif_interaction_jaccard_*.parquet`, correspondente às células 32–36 de `short/full_loss_function.ipynb`) e gera os dados e gráficos da **Tabela 8 e Fig. 3**.
- Criar `notebooks/reproduction/short/08_triplet_composition_and_cooccurrence.ipynb`:
  - Reproduz a análise de composição de tripletos e RSCU exploratório (`compute_exploratory_triplet_rscu`) com correção de múltiplos testes Benjamini-Hochberg (FDR, Nota 41 — **Tabela 9** e **Fig. 4**).
  - Audita a prevalência de tripletos `ATG` e `TAA/TAG/TGA` nos motivos validados (`audit_translation_boundary_signals`) com demarcação explícita de composição exploratória sem CDS anotado (Nota 42 — **Tabela 10**).
  - Consome a matriz de Jaccard gerada no notebook `07` e executa a análise dos 200 pares de maior coocorrência (`analyze_cooccurrence_pairs`) com teste de simetria/orientação, modelo nulo de permutação e correlações inter-arquitetura com $p$-valores bilaterais exatos e ajuste FDR (Nota 43 — **Tabela 11** e **Fig. 5**).

**Critério de Aceitação (`checks.md`):**
- Notebooks executam com reprodutibilidade determinística (semente 42), vinculam cada painel de motivo ao checkpoint correto, preservam o encadeamento da matriz de Jaccard entre `07` e `08` e produzem as saídas para Tabelas 8–11 e Figs. 3–5.

---

### N06 — Notebook de Benchmark Comparativo CNNs vs Kraken2 e BLASTN (`n06-short-kraken2-blastn-benchmark-notebook`)

**Depende de:** N01, N02.

**Objetivo:** Reproduzir `short/benchmark_cnns_kraken2_blast_overlap.ipynb` usando `viral_cnn.baselines.*`.

**Escopo:**
- Criar `notebooks/reproduction/short/09_benchmark_cnns_kraken2_blast_overlap.ipynb`:
  - Carrega ou reconstrói o mapa de sobreposição exata entre treino e teste (`audit_exact_overlap` em `viral_cnn.baselines.overlap`: estratos `all`, `overlap`, `no_exact_overlap`).
  - Integra as saídas do Kraken2 (`parse_kraken_output`) e do BLASTN (`parse_blast_output`) do diretório remoto `cnn_kraken/benchmark_kraken_blast_20_per_species/` com as predições das CNNs nas **mesmas consultas** e nos mesmos níveis de ruído $\epsilon \in \{0.0, 0.02, 0.05, 0.10, 0.15\}$.
  - Aplica a política estrita de denominador único $N$ (abstenções = `-1` contam como erro na acurácia global, reportando cobertura e acurácia condicional separadamente via `evaluate_predictions` e `paired_discordance` em `viral_cnn.baselines.paired`).
  - Calcula intervalos de confiança de 95% para as diferenças pareadas via cluster bootstrap por espécie (`bootstrap_species_difference`, 2.000 repetições, semente 42) e inclui o baseline opcional de frequência de $k$-mers (`KmerCentroidClassifier` em `viral_cnn.baselines.kmer_baseline`).

**Critério de Aceitação (`checks.md`):**
- Notebook executa e produz a tabela comparativa pareada completa por estrato de sobreposição e nível de ruído, com denominador único auditado.

---

### N07 — Notebooks de Sequências Longas (`LongSequenceVoter`, Comparação e Motivos) (`n07-long-sequence-voter-notebooks`)

**Depende de:** N01, N02, N05.

**Objetivo:** Reproduzir os 5 notebooks da pasta `long/` em `notebooks/reproduction/long/` usando `viral_cnn.inference.voter`, `viral_cnn.inference.audit_long`, `viral_cnn.baselines.long_eval` e `viral_cnn.interpretability.motifs`.

**Escopo:**
- Criar `notebooks/reproduction/long/01_ncbi_download_and_prepare_database.ipynb`:
  - Consolida de forma limpa e parametrizada `long/download_from_ncbi.ipynb` e `long/prepare database.ipynb`, auditando a interseção de accessions/genomas entre as sequências longas do NCBI ($N=4.360$) e a base de treinamento Kraken2 (`quantify_training_overlap` em `viral_cnn.inference.audit_long`, Notas 11–15, 44–47).
- Criar `notebooks/reproduction/long/02_cnn_long_voter_inference.ipynb`:
  - Reproduz `long/cnn-long-voter.ipynb` (incluindo avaliação limpa e sob ruído das células 40–44) utilizando `LongSequenceVoter` (`viral_cnn.inference.voter`, janela 512, stride 256, agregação por média de logits pré-softmax e voto majoritário, comparando `handle_short='skip'` histórico vs `handle_short='pad'` padronizado, `AD-010`).
  - Verifica a equivalência exata de logits de uma janela entre o caminho de fragmentos curtos e o `LongSequenceVoter` (`verify_window_logit_equivalence`).
- Criar `notebooks/reproduction/long/03_cnn_long_voter_compare.ipynb`:
  - Reproduz `long/cnn-long-voter-compare.ipynb` e a **Tabela 12**, executando `stratify_long_errors` (`viral_cnn.inference.audit_long`) e `integrate_long_voter_predictions` / `evaluate_long_baselines` (`viral_cnn.baselines.long_eval`) para estratificar erros por espécie, comprimento do contig, proporção de bases IUPAC ambíguas, presença do genoma no treino (`seen_genome` vs `unseen_genome`) e diagnosticar a queda do ConvNeXt (3,49%–20,38% vs 72,69% da ResNet50: viés de amostragem 500 vs 50 reads em vírus gigantes, *logit pulling* pré-softmax e perda de calibração).
- Criar `notebooks/reproduction/long/04_long_motifs_and_gc_analysis.ipynb`:
  - Reproduz `long/long_loss_function.ipynb` usando `viral_cnn.interpretability.motifs` sobre as janelas das sequências longas.

**Critério de Aceitação (`checks.md`):**
- Os 4 novos notebooks em `notebooks/reproduction/long/` executam de ponta a ponta sem modificar `long/*.ipynb`, reproduzem a Tabela 12 e emitem o diagnóstico completo da auditoria S06.

---

### N08 — Notebook Mestre de Regeneração Dinâmica de Tabelas/Figuras e Reconciliação (`n08-master-paper-regeneration-notebook`)

**Depende de:** N02–N07.

**Objetivo:** Atualizar `viral_cnn.reporting.tables` e `viral_cnn.reporting.figures` para ingerirem dinamicamente os artefatos reais gerados pelos notebooks `N02–N07` em `results_reproduction/` (mantendo fallback de referência para smoke-test), disponibilizar o notebook mestre `notebooks/reproduction/00_regenerate_all_paper_artifacts.ipynb` e documentar o guia completo no `README.md`.

**Escopo:**
- Atualizar `src/viral_cnn/reporting/tables.py` e `src/viral_cnn/reporting/figures.py`:
  - Aceitar `artifacts_dir: Path | None = None` para construir as **Tabelas 1 a 12** e **Figuras 1 a 5** diretamente a partir dos arquivos Parquet/CSV/JSON produzidos por `N02–N07` no JupyterHub remoto, comparando cada célula calculada com os valores publicados no manuscrito e emitindo uma tabela de reconciliação (`exact_match` vs `delta` explicado).
- Criar `notebooks/reproduction/00_regenerate_all_paper_artifacts.ipynb`:
  - Verifica todos os manifestos de execução (`NotebookRunManifest`) gerados pelos notebooks `N02–N07` no JupyterHub remoto.
  - Invoca `generate_all_tables` e `generate_all_figures` sobre `results_reproduction/` para regenerar as **Tabelas 1 a 12** (LaTeX/CSV/Markdown) e as **Figuras 1 a 5** (PNG/PDF com verificação case-sensitive no Linux).
  - Produz o relatório automático de reconciliação entre os valores obtidos no JupyterHub e os valores publicados (`viral_cnn.reporting.manuscript`).
- Atualizar `README.md` com:
  - Topologia de diretórios do JupyterHub remoto (`DATA_ROOT` para `short/` e `long/`), variáveis de ambiente e requisitos de pacotes.
  - Ordem exata de execução dos notebooks em `notebooks/reproduction/` conectados ao kernel remoto.

**Critério de Aceitação (`checks.md`):**
- Notebook mestre executa sem erros, valida a presença e integridade dos manifestos de `N02–N07`, gera todas as tabelas e figuras e atualiza `README.md`.

---

## Portões de Verificação (`tlc-spec-lean`)

Para cada spec `N01` a `N08`, o ciclo seguirá estritamente:
1. **PLAN (`.specs/features/<nXX-...>/plan.md`):** Definição do problema, fluxo (`Flow`), impacto (`Impact`), relações (`Relations`), superfície (`Surface`), decisões irreversíveis (`Landing`) e critérios EARS (` SHALL `), validado com `validate_plan.py`.
2. **CHECKS (`.specs/features/<nXX-...>/checks.md`):** Derivação das obrigações observáveis com prova automatizada (`pytest` exercitando os módulos e executando os novos notebooks via `nbclient` em modo smoke-test determinístico + checagem estática de que nenhum notebook original em `short/` ou `long/` foi tocado via `git status`/`sha256`), validado com `validate_checks.py`.
3. **BUILD:** Implementação dos módulos de ponte necessários e criação dos novos notebooks em `notebooks/reproduction/`, mantendo todos os testes existentes + novos testes verdes.
4. **VERIFY (`.specs/features/<nXX-...>/verification.md`):** Despacho obrigatório de um subagente **Verifier** independente sobre `<base>..HEAD` e validação final via `validate_verification.py`.

