# Inventário Geral e Mapeamento de Artefatos — Artigo de Classificação Taxonômica Viral

## 1. Visão Geral do Repositório

Este documento estabelece o inventário completo dos notebooks, módulos de utilidade, conjuntos de dados, checkpoints e saídas do repositório, conciliando as evidências de código com o artigo:
**“Spatial one-hot nucleotide representations of DNA enable robust and interpretable taxonomic classification with 1D convolutional networks”** (BioData Mining / submetido).

---

## 2. Inventário de Notebooks

O repositório possui 15 notebooks divididos entre `short/` (classificação de fragmentos de 512 bp, espaço latente, motivos e benchmarks) e `long/` (inferência em sequências longas/contigs via janela deslizante, curadoria do NCBI e preparação de dados).

| Notebook | Células (Code/MD) | Classificação | Entradas Principais | Saídas / Artefatos Gerados | Tabelas / Figuras do Artigo |
|---|---|---|---|---|---|
| `short/benchmark.ipynb` | 12 (12 / 0) | **Histórico / Executável** | Checkpoints ResNet50 e ConvNeXt (`.pt`), Parquet `processed_kraken_species_splited_parquet/test` | Métricas de avaliação limpa e sob ruído ($\epsilon \in \{0, 0.02, 0.05, 0.10, 0.15\}$), profiling de inferência | **Tabela 1, Tabela 2, Tabela 3, Tabela 4, Fig. 1** |
| `short/finetune_resnet50_species.ipynb` | 14 (14 / 0) | **Histórico / Treinamento** | Checkpoint limpo `best_resnet_classifier__fast_phase4.pt`, Parquet split species | Checkpoint com fine-tuning de ruído: `checkpoint_finetune_resnet/finetune_noise_*.pt` | Suporte a **Tabela 1, 3, 4** |
| `short/finetune_convnext_species.ipynb` | 14 (14 / 0) | **Histórico / Treinamento** | Checkpoint transferido `best_convnext_classifier_fast_phase4_2_restarts.pt`, Parquet split species | Checkpoint com fine-tuning de ruído: `checkpoint_transfer_species_conv/finetune_noise_*.pt` | Suporte a **Tabela 1, 3, 4** |
| `short/analize_convnext.ipynb` | 40 (34 / 6) | **Exploratório / Análise** | Checkpoints Species/Genus, `species_name_ids.csv`, `genus_name_ids.csv`, Parquet test | Distribuição de confiança, matrizes de confusão top-k, entropia por nível | **Tabela 4, Tabela 7** |
| `short/ICTV_latent_space-bkp.ipynb` | 27 (27 / 0) | **Histórico / Análise** | Checkpoints, `main_taxonomy_hybrid_CLEANSED.parquet`, Parquet test | UMAP embeddings Parquet, métricas KNN ($k=5$) e Silhouette em `results_reports/` | **Tabela 5, Tabela 6, Fig. 2** |
| `short/full_loss_function.ipynb` | 66 (66 / 0) | **Histórico / Interpretabilidade** | Checkpoints de espécies, `main_df.parquet`, Parquet test | Extração de motivos 7-mers, testes pareados Wilcoxon (FDR), saídas em Parquet | **Tabela 8, Fig. 3** |
| `short/codon_usage.ipynb` | 27 (27 / 0) | **Histórico / Interpretabilidade** | Motivos validados (`full_loss_function`), Parquet test | Perfil RSCU de fundo, frequências de tripletos em motivos, ATG / stop códons, sinergia de pares | **Tabela 9, Tabela 10, Tabela 11, Fig. 4, Fig. 5** |
| `short/ICTV_improve_kraken_database.ipynb` | 24 (24 / 0) | **Curadoria / Preparação** | `dataset_kraken/species_name_ids.csv`, planilha `ICTV_Master_Species_List_2025_MSL41.v1.xlsx` | `main_taxonomy_hybrid_ictv_ncbi.parquet`, `main_taxonomy_hybrid_CLEANSED.parquet` | Base taxonômica de **Tabela 5, Tabela 6** |
| `short/improve_kraken_database-Copy1.ipynb` | 23 (23 / 0) | **Exploratório / Rascunho** | `species_name_ids.csv`, fontes taxonômicas | Parquets intermediários de taxonomia em `data_taxonomy_parquet/` | Pré-etapa de curadoria |
| `short/benchmark_cnns_kraken2_blast_overlap.ipynb` | 32 (21 / 11) | **Histórico / Benchmark** | Parquet train/test species, executáveis externos Kraken2 e BLASTN | Arquivos de sobreposição exata (`exact_overlap.csv`), relatórios comparativos pareados | Baseline do manuscrito / **S07** |
| `long/cnn-long-voter.ipynb` | 48 (44 / 4) | **Histórico / Executável** | Checkpoints Species/Genus, `processed_ncbi_splited_parquet`, `mapping_species_final.csv` | `resultados_inferencia_long_reads.parquet`, `relatorio_erros_biologicos.csv` | **Tabela 12** |
| `long/cnn-long-voter-compare.ipynb` | 37 (33 / 4) | **Histórico / Análise** | Resultados do voter (`.parquet`), mapeamentos taxonômicos | Heatmap de performance, boxplots de erros, gráficos de calibração | **Tabela 12, Figuras complementares** |
| `long/long_loss_function.ipynb` | 40 (40 / 0) | **Exploratório / Análise** | Checkpoints, `processed_ncbi_splited_parquet`, motivos pré-calculados | Impacto de motivos e GC em sequências longas, Parquet em `results_motifs/` | Análise exploratória long-reads |
| `long/download_from_ncbi.ipynb` | 14 (14 / 0) | **Curadoria / Download** | Acessos do NCBI, IDs Kraken2 | `raw_database/final.csv`, `raw_database/genus_new_labels.csv` | Construção da base bruta de sequências |
| `long/prepare database.ipynb` | 34 (34 / 0) | **Curadoria / Preparação** | `raw_database/final.csv` | Partições Parquet (`processed_ncbi_splited_parquet/train`, `test`, `validation`) | Split e fragmentação de sequências longas |

---

## 3. Mapeamento dos Módulos em `short/utils/`

| Arquivo | Classes / Funções Principais | Papel | Dependências Relevantes |
|---|---|---|---|
| `short/utils/models.py` | `OneHotEmbedding`<br>`ConvNeXtBlock1D`<br>`LayerNorm1d`<br>`BottleneckBlock1D`<br>`ResNet50_1D`<br>`HybridConvNeXt`<br>`resnet50_1d()`<br>`convnext_tiny_1d()` | Definição das arquiteturas 1D e camada de codificação one-hot A/C/G/T (512 bp, 4 canais). | `torch`, `torch.nn`, `torch.nn.functional` |
| `short/utils/dna_dataset_2.py` | `DaskCuDFIntegerDataset`<br>`ParquetCPUDataset` | Leitura iterável em lotes de sequências e rótulos a partir de arquivos Parquet, tokenização e padding. | `torch`, `pyarrow.parquet`, `pandas`, `cudf` (opcional/GPU) |
| `short/utils/train_test.py` | `train_validate_model`<br>`test_model`<br>`topk_accuracy`<br>`print_loss_info`<br>`salvar_lote_invalido`<br>`plot_metrics` | Rotinas de treinamento com AMP (`GradScaler`, `autocast`), avaliação limpa/ruidosa, cálculo de acurácia Top-1/Top-5 e F1. | `torch`, `sklearn.metrics`, `pandas`, `numpy` |
| `short/utils/analize_funcs.py` | `plot_confusion_matrix_insights`<br>`inspect_csv`<br>`get_class_label`<br>`get_confused_class_samples`<br>`tensor_to_img` | Mapeamento de rótulos 0-based para 1-based no CSV de taxonomia, inspeção de predições e confusão. | `torch`, `pandas`, `sklearn.metrics`, `matplotlib` |
| `short/utils/gradcam.py` | `GradCAM1D`<br>`ScoreCAM1D`<br>`visualize_heatmap`<br>`visualize_heat_map_examples_by_class` | Métodos de explicabilidade e atribuição 1D para convoluções. | `torch`, `numpy`, `matplotlib` |

---

## 4. Inventário de Dados e Checkpoints

### 4.1 Dados Presentes no Repositório
- `dataset_kraken/species_name_ids.csv` (13.684 linhas: cabeçalho + 13.683 mapeamentos `species,id`)
- `dataset_kraken/genus_name_ids.csv` (2.703 linhas: cabeçalho + 2.702 mapeamentos `genus,id`)
- `dataset_kraken/family_name_ids.csv` (247 linhas: cabeçalho + mapeamentos `family,id`)
- `dataset_kraken/species_seqs.csv` (5,0 GB: `sequence,label`)
- `dataset_kraken/genus_seqs.csv` (2,4 GB: `sequence,label`)
- `dataset_kraken/family_seqs.csv` (401 MB: `sequence,label`)
- `paper/BioData_Minning___Matheus.pdf` (Manuscrito completo compilado)

### 4.2 Artefatos Externos / Checkpoints Ausentes no Repositório Atual
Os seguintes caminhos são referenciados nos notebooks históricos, mas **não se encontram no disco local**:
1. **Checkpoints de Espécie:**
   - `../cnn_kraken_species/checkpoints/best_resnet_classifier__fast_phase4.pt` (ResNet50 limpo)
   - `./checkpoint_finetune_resnet/finetune_noise_14.pt` (ResNet50 com fine-tuning de ruído)
   - `./checkpoint_species_conv_zero/best_convnext_phase3_restarts_v2.pt` (ConvNeXt_Z do zero)
   - `./checkpoint_transfer_species_conv/best_convnext_classifier_fast_phase4_2_restarts.pt` (ConvNeXt_T transferido)
   - `./checkpoint_transfer_species_conv/finetune_noise_7.pt` (ConvNeXt_F com fine-tuning de ruído)
2. **Checkpoints de Gênero:**
   - `../cnn_kraken_genus/best_resnet_classifier__fast_phase4.pt` (ResNet50 Gênero)
   - `../cnn_kraken/checkpoint_genus_conv/best_convnext_classifier_fast_phase4_2.pt` (ConvNeXt Gênero)
3. **Partições Pré-processadas em Parquet:**
   - `./processed_kraken_species_splited_parquet/` (`train/`, `validation/`, `test/`)
   - `./processed_kraken_genus_splited_parquet/`
   - `./processed_ncbi_splited_parquet/`
4. **Bases Taxonômicas Intermediárias:**
   - `main_taxonomy_hybrid_CLEANSED.parquet`
   - `data_taxonomy_parquet/main_df.parquet`
   - `ICTV_Master_Species_List_2025_MSL41.v1.xlsx`

> **Registro de Bloqueio Conforme AGENTS.md:**
> Como os arquivos `.pt` e as partições Parquet divididas não estão presentes no ambiente atual, a reprodução de inferência real sobre dados de teste fica bloqueada até a provisão desses artefatos. A validação do fluxo extraído na S01 deve operar em conformidade com o critério de aceitação de AGENTS.md: validação de contratos sintéticos, verificação estrita de invariantes (ordem A/C/G/T, padding nulo, formas de entrada/saída) e execução de lote pequeno com pesos determinísticos/sintéticos.

---

## 5. Proposta de Estrutura de Diretórios para Migração Gradual

Conforme preconizado em `AGENTS.md` e no roteiro `ROADMAP_SPECS_ARTIGO.md`:

```text
cnn_metagenome/
├── README.md                      # Documentação de entrada e ambiente
├── AGENTS.md                      # Invariantes e regras permanentes
├── ROADMAP_SPECS_ARTIGO.md        # Sequência de specs (S01 a S12)
├── .specs/                        # Memória, decisões e planos da metodologia tlc-spec-lean
│   ├── STATE.md
│   └── features/s01-inventory-and-vertical-slice/
├── configs/                       # Configurações declarativas (sem caminhos pessoais)
│   ├── default_species.yaml
│   └── default_genus.yaml
├── src/                           # Código extraído e desacoplado em pacote Python
│   ├── viral_cnn/
│   │   ├── __init__.py
│   │   ├── models/                # ResNet50-1D, ConvNeXt-1D, OneHotEmbedding
│   │   ├── data/                  # Codificação A/C/G/T, padding, datasets sintéticos/reais
│   │   ├── inference/             # Rotinas de inferência limpa e ruidosa
│   │   └── taxonomy/              # Mapeamentos e conversão 1-based <-> 0-based
├── notebooks/                     # Notebooks organizados por papel
│   ├── original/                  # Notebooks históricos preservados sem alteração
│   │   ├── short/
│   │   └── long/
│   └── reproduction/              # Notebooks limpos de reprodução (nas specs futuras)
├── scripts/                       # Pontos de entrada CLI
│   └── run_inference.py           # CLI para o fluxo vertical extraído
├── tests/                         # Testes automatizados de contratos científicos
│   ├── test_encoding_invariants.py
│   ├── test_model_forward.py
│   └── test_cli_inference.py
└── docs/                          # Documentação e inventário de proveniência
    └── inventory.md               # Este documento
```

---

## 6. Fluxo Vertical Selecionado para Extração na S01

Para viabilizar a primeira fatia vertical testável e autônoma sem depender de pesos externos não versionados:
1. **Entrada de DNA:** Sequências de DNA brutas (strings ou lotes com comprimento arbitrário $\le 512$ bp).
2. **Codificação / Pré-processamento:**
   - Truncamento ou padding para 512 posições.
   - Ordem estrita dos canais: `0: A`, `1: C`, `2: G`, `3: T`.
   - Caracteres ambíguos (`N`, `R`, `Y`, `-`, `_`, etc.) mapeados para o canal de padding (gerando vetor nulo `[0, 0, 0, 0]`).
   - Tensor de saída: `(batch_size, 4, 512)` no `OneHotEmbedding`.
3. **Modelos:**
   - `ResNet50_1D` e `ConvNeXt_Tiny_1D` com número parametrizável de classes (`13683` para espécies, `2702` para gênero, ou valor reduzido para testes sintéticos).
4. **Ponto de Entrada CLI (`scripts/run_inference.py`):**
   - Aceita parâmetros explícitos: `--sequence`, `--fasta`, `--model-type`, `--num-classes`, `--device`.
   - Produz saída estruturada com predição Top-1 / Top-5 e probabilidades sem caminhos pessoais embutidos.
5. **Demonstração de Equivalência:**
   - Teste unitário comparando a codificação matemática da classe histórica `OneHotEmbedding` com o módulo extraído.
   - Teste do forward pass com lote sintético garantindo exata coincidência numérica.

