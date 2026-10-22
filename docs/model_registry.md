# Registro de Modelos, Mapeamentos Taxonômicos e Treinamento

Este documento registra a proveniência, arquiteturas, mapeamentos taxonômicos, pesos de checkpoints e hiperparâmetros de treinamento para os modelos neurais 1D avaliados no artigo **“Spatial one-hot nucleotide representations of DNA enable robust and interpretable taxonomic classification with 1D convolutional networks”**.

Conforme os invariantes científicos estabelecidos em `AGENTS.md`, todos os hiperparâmetros evidenciados no código dos notebooks históricos são documentados como **verificados**, enquanto parâmetros não documentados ou ausentes dos registros de treino são formalmente rotulados como **não verificados** (`unverified`), jamais preenchidos por suposição.

---

## 1. Visão Geral das Variantes Registradas

O artigo avalia **7 modelos neurais convolucionais 1D** distribuídos em dois níveis taxonômicos:
- **5 modelos no nível de Espécie (Species):** 13.683 classes fechadas (*closed-set*).
- **2 modelos no nível de Gênero (Genus):** 2.702 classes fechadas (*closed-set*).

| Identificador Canônico | Arquitetura | Nível Taxonômico | Classes | Origem / Linhagem | Caminho do Checkpoint Histórico | Dropout |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `resnet_clean_species` | `resnet50_1d` | Species | 13.683 | Treinado do zero com dados limpos | `cnn_kraken_species/checkpoints/best_resnet_classifier__fast_phase4.pt` | 0.5 (avaliação) / 0.7 (treino) |
| `resnet_noise_species` | `resnet50_1d` | Species | 13.683 | Fine-tuning com corrupção de $\epsilon=0,05$ | `cnn_kraken/checkpoint_finetune_resnet/finetune_noise_14.pt` | 0.5 / 0.7 |
| `convnext_zero_species` | `convnext_tiny_1d` | Species | 13.683 | Treinado do zero em espécies (`ConvNeXt_Z`) | `cnn_kraken/checkpoint_species_conv_zero/best_convnext_phase3_restarts_v2.pt` | 0.7 |
| `convnext_transfer_species` | `convnext_tiny_1d` | Species | 13.683 | Transferido de Gênero para Espécie (`ConvNeXt_T`) | `cnn_kraken/checkpoint_transfer_species_conv/best_convnext_classifier_fast_phase4_2_restarts.pt` | 0.7 |
| `convnext_noise_species` | `convnext_tiny_1d` | Species | 13.683 | Transferido de Gênero e fine-tuned com $\epsilon=0,05$ (`ConvNeXt_F`) | `cnn_kraken/checkpoint_transfer_species_conv/finetune_noise_7.pt` | 0.7 |
| `resnet_genus` | `resnet50_1d` | Genus | 2.702 | Treinado do zero em gênero | `cnn_kraken_genus/best_resnet_classifier__fast_phase4.pt` | 0.5 |
| `convnext_genus` | `convnext_tiny_1d` | Genus | 2.702 | Treinado do zero em gênero | `cnn_kraken/checkpoint_genus_conv/best_convnext_classifier_fast_phase4_2.pt` | 0.7 |

---

## 2. Invariantes de Entrada e Mapeamento Taxonômico

### 2.1 Representação Espacial One-Hot (4 Canais)
- **Tamanho da janela de entrada:** 512 pares de base (bp).
- **Stride:** 256 bp (50% de sobreposição durante fragmentação).
- **Canais:** 4 canais ordenados rigorosamente como `A`, `C`, `G`, `T`:
  - `A` = índice 0 $\to [1, 0, 0, 0]$
  - `C` = índice 1 $\to [0, 1, 0, 0]$
  - `G` = índice 2 $\to [0, 0, 1, 0]$
  - `T` = índice 3 $\to [0, 0, 0, 1]$
- **Padding e bases ambíguas:** Qualquer caractere desconhecido ou ambíguo (`N`, `R`, `Y`, `S`, `W`, `K`, `M`, `B`, `D`, `H`, `V`, `-`, `_`) recebe índice `-1` e projeta-se em **vetor nulo de 4 canais** $[0, 0, 0, 0]$.

### 2.2 Offset de Rótulos de Saída (1-based para 0-based)
- Os arquivos de taxonomia do Kraken (`species_name_ids.csv` e `genus_name_ids.csv`) usam identificadores inteiros 1-based no intervalo $[1, N_{\text{classes}}]$.
- A camada classificadora dos modelos emite logits no espaço $[0, N_{\text{classes}} - 1]$.
- A relação contratual estabelecida é `model_label = csv_id - 1` (ou equivalentemente `csv_id = model_label + 1`).
- A ordem das classes na camada linear de saída corresponde estritamente à ordenação crescente dos IDs no mapeamento de taxonomia.

---

## 3. Critérios de Seleção de Checkpoint: Melhor vs Último

Um princípio fundamental do `AGENTS.md` é nunca rebatizar o último checkpoint como "melhor" sem evidência documentada.

1. **Checkpoints de Pré-treino e Transferência (`best_*.pt`):**
   - Checkpoints com nome prefixado por `best_` (ex: `best_resnet_classifier__fast_phase4.pt`, `best_convnext_phase3_restarts_v2.pt`, `best_convnext_classifier_fast_phase4_2_restarts.pt`) representam checkpoints selecionados pelo **melhor desempenho na validação** durante os ciclos de treino em fases com *Cosine Annealing Restarts*.
   - Critério verificado: Menor CrossEntropy Loss ou maior F1-score no conjunto de validação durante o término de um ciclo de restart.

2. **Checkpoints de Fine-Tuning com Ruído (`finetune_noise_*.pt`):**
   - No fine-tuning sob ruído artificial de $\epsilon=0,05$ (`short/finetune_resnet50_species.ipynb` e `short/finetune_convnext_species.ipynb`), o pipeline salva `finetune_noise_{epoch}.pt` e `finetune_noise_last.pt` a cada época.
   - Para a ResNet50: o checkpoint utilizado na avaliação final do artigo é `finetune_noise_14.pt`, correspondente ao estado após a **última época concluída** (época 14 de 15, indexada em 0).
   - Para o ConvNeXt: o checkpoint utilizado é `finetune_noise_7.pt`, correspondente à **época 7** (onde a acurácia de validação estabilizou antes de overfitting).
   - Esta distinção fica expressamente documentada no registro: `finetune_noise_14` é o checkpoint da última época de adaptação da ResNet, enquanto `best_*` reflete o ponto ótimo de validação.

---

## 4. Hiperparâmetros de Treinamento: Verificados vs Não Verificados

### 4.1 Hiperparâmetros Verificados (Evidenciados no Código)

| Parâmetro | Valor Evidenciado no Código | Notebook / Fonte de Evidência |
| :--- | :--- | :--- |
| **Otimizador** | `torch.optim.AdamW` | `finetune_resnet50_species.ipynb`, `finetune_convnext_species.ipynb` |
| **Learning Rate Máxima (Fine-Tuning)** | `5e-5` | `finetune_resnet50_species.ipynb` (Cell 9), `finetune_convnext_species.ipynb` (Cell 9) |
| **Weight Decay** | `1e-4` | `finetune_resnet50_species.ipynb`, `finetune_convnext_species.ipynb` |
| **Scheduler** | `torch.optim.lr_scheduler.CosineAnnealingLR` com $\eta_{\min}=10^{-7}$ | `finetune_resnet50_species.ipynb`, `finetune_convnext_species.ipynb` |
| **Função de Perda** | `torch.nn.CrossEntropyLoss(label_smoothing=0.1)` | `short/full_loss_function.ipynb`, `finetune_*.ipynb` |
| **Precisão Mista (AMP)** | `torch.amp.GradScaler('cuda')` | `finetune_resnet50_species.ipynb`, `finetune_convnext_species.ipynb` |
| **Matmul Precision** | `torch.set_float32_matmul_precision('high')` | Notebooks raiz e `short/*.ipynb` |
| **Batch Size (Avaliação / Dask)** | 8.192 (ou 1.024 em análise de CPU) | `short/benchmark.ipynb` (Cell 2), `short/analize_convnext.ipynb` (Cell 7) |
| **Taxa de Ruído em Fine-Tuning** | $\epsilon = 0,05$ (5% de corrupção A/C/G/T) | `finetune_resnet50_species.ipynb` (`NOISE_RATE_TRAIN = 0.05`) |
| **Épocas de Fine-Tuning** | 15 épocas planejadas | `NUM_EPOCHS_FINETUNE = 15` |

### 4.2 Lacunas e Hiperparâmetros Não Verificados (`unverified`)

Em conformidade com a regra de integridade científica de `AGENTS.md`, os seguintes parâmetros não constam explicitamente nos códigos ou logs preservados e são registrados como não verificados:
- **Paciência de Early Stopping das Fases Iniciais:** Não há registro explícito de paciência ou threshold de parada antecipada nas fases 1 a 3 do treino do zero (`unverified - logs de treino das fases iniciais não preservados`).
- **Semente Aleatória das Fases de Pré-Treino de ConvNeXt:** Enquanto o split e testes usam semente 42, a semente exata utilizada na inicialização dos pesos da fase 1 do ConvNeXt treinado do zero não foi gravada no notebook (`unverified - seed de inicialização ausente`).
- **LR Máxima nas Fases 1 a 3:** A taxa de aprendizado exata nas fases iniciais de aquecimento não está preservada nos notebooks curtos (`unverified - notebook registra apenas fase 4 e fine-tuning`).

---

## 5. Normalização do Formato de Checkpoint e Prefixo `_orig_mod.`

Durante o ciclo de vida dos modelos, dois padrões de salvamento de pesos ocorreram:
1. **Dicionário Completo de Treinamento:**
   Checkpoints contendo chaves: `epoch`, `model_state_dict`, `optimizer_state_dict`, `scheduler_state_dict`, `scaler_state_dict`, `metrics`.
2. **State Dict Bruto:**
   Checkpoints contendo diretamente o `OrderedDict` mapeando nomes de tensores para parâmetros.

Além disso, modelos treinados sob `torch.compile(model)` inserem o prefixo `_orig_mod.` em todas as chaves (ex: `_orig_mod.conv1.weight`). O carregador unificado implementado em `src/viral_cnn/models/registry.py` normaliza transparentemente ambas as situações:
```python
state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
clean_state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
```

Adicionalmente, o carregador valida que o número de classes na camada linear final corresponde exatamente ao número esperado pelo modelo (`fc.weight.shape[0] == num_classes`), impedindo o carregamento acidental de checkpoints de Gênero em modelos de Espécie.
