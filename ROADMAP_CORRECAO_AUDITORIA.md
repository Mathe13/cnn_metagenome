# Roadmap de Correção e Reprodução Científica (`tlc-spec-lean`) — Pós-Auditoria de Execução

Este roadmap define a sequência criteriosa de especificações sob a metodologia **`tlc-spec-lean`** (`PLAN → CHECKS → BUILD → VERIFY`) para corrigir todas as pendências identificadas na auditoria dos notebooks executados no cluster (`commits 6566831` e `2c0c84d`), dos artefatos publicados em `results_reproduction/` e das tabelas do manuscrito (`paper/manuscript_submission.tex`).

---

## 1. Princípios Inegociáveis e Guardas Científicas (`AGENTS.md`)

1. **Proibição de Fallback Sintético Silencioso (`Fail-Fast` quando `SMOKE_TEST=0`):**
   - Nenhum notebook em `notebooks/reproduction/` ou módulo chamado por ele pode recair silenciosamente em dados sintéticos (`np.random.randn`, `rng.randint`, strings `"ACGT" * 128`, `MockVoterModel`, `mock_records` ou `torch.zeros`) quando `ctx.is_smoke_test == False`.
   - Se `ctx.is_smoke_test == False` e qualquer arquivo real obrigatório (partição Parquet, planilha/ Parquet taxonômico, saída do Kraken2/BLASTN ou checkpoint `.pt`) não existir, o código **DEVE (`SHALL`)** interromper a execução imediatamente com `FileNotFoundError` ou `RuntimeError`, nos mesmos moldes já aplicados nos notebooks `10` e `11`.
2. **Preservação Intocada dos Notebooks Históricos (`AD-004`):**
   - Nenhum arquivo em `short/` ou `long/` pode ser modificado, movido ou removido.
3. **Separação Estrita entre Contrato de Código (`SMOKE_TEST=1`) e Reprodução Científica (`SMOKE_TEST=0`):**
   - Testes automatizados locais (`REMOTE_SERVER=False` / `SMOKE_TEST=1`) provam contratos matemáticos, integridade de pipeline e o disparo da guarda `fail-fast` quando um artefato real está ausente fora do modo smoke.
   - A reprodução científica empírica só é atestada após a execução no cluster (`REMOTE_SERVER=true`, `SMOKE_TEST=0`) com manifesto registrando `"is_smoke_test": false`, contagens reais de amostras ($N = 1.706.270$ no teste de Species, $N = 816.756$ em Genus, $N = 273.660$ nos baselines pareados, $N = 4.360$ em sequências longas) e hashes/caminhos dos checkpoints.
4. **Visibilidade Dupla (Output do `.ipynb` + CSV/JSON Versionado no Git):**
   - Toda tabela gerada por um notebook de reprodução deve ser simultaneamente **impressa no output da célula** (para auditoria direta do `.ipynb` no GitHub) e **salva em `results_reproduction/`** com rastreamento pelo Git para arquivos leves (`.csv`, `.json`, `.yaml`, `.png`), mantendo ignorados apenas binários volumosos (`*.pt`, `*.parquet`, `*.sqlite`).

---

## 2. Visão Geral das Specs (`R01` a `R08`)

| Spec | Slug em `.specs/features/` | Alvo Principal | Problema Auditado que Resolve | Depende de |
| :--- | :--- | :--- | :--- | :--- |
| **R01** | `r01-parameter-reconciliation-and-benchmark-tracking` | `short/01_benchmark_clean_and_noise.ipynb`, `profiler.py`, `.gitignore` | Contagem de parâmetros (`43,99 M` e `164,01 M` vs. `23,5 M` e `28,1 M`); CSVs de `n02_benchmark` presos no smoke test de 16 amostras; tabelas não impressas no output da Célula 4 do notebook `01`. | — |
| **R02** | `r02-finetune-lr-schedule-and-trainer-parity` | `short/03_finetune_resnet50_species.ipynb`, `short/04_finetune_convnext_species.ipynb`, `trainer.py` | LR inicial $20\times$ menor ($2{,}5\times 10^{-6}$ em vez de $5\times 10^{-5}$) em `ControlledTrainer`; `scheduler.step()` por batch vs. por época; `UserWarning` antes de `optimizer.step()` sob AMP; ausência de `resume` e avaliação pré-treino (época 0). | R01 |
| **R03** | `r03-real-calibration-and-hierarchical-entropy` | `short/02_analyze_confidence_and_calibration.ipynb`, `calibration.py` | `species_to_genus = torch.zeros(13683)` zerando `H_genus`; calibração calculada sobre 16 strings `"ACGT" * 128`; identidade de pares confundidos calculada em strings fixas; `H_act: nan` na ResNet50. | R01 |
| **R04** | `r04-real-latent-embeddings-and-taxonomy-audit` | `short/05_ictv_taxonomy_curation.ipynb`, `short/06_ictv_latent_space_and_hierarchy.ipynb`, `latent.py` | `05` grava 100 registros inventados (`25 espécies avaliáveis`) no manifesto; `06` usa `np.random.randn` sem carregar checkpoints/Parquets, assume `dim=768` (real é `2048`), copia `KNN_Top1` em `KNN_Top5` e produz `N_taxa=3` / `NaN` na Tabela 6. | R01 |
| **R05** | `r05-real-motif-jaccard-and-triplet-analysis` | `short/07_first_layer_motifs_and_perturbation.ipynb`, `short/08_triplet_composition_and_cooccurrence.ipynb`, `motifs.py` | Célula 9 do `07` fabrica matriz de presença fixa ($40\times 15$), gerando `r=0.0` / `p=1.0` na Tabela 11 e `r=1.0` entre arquiteturas no `08`; Célula 5 do `08` usa lista fixa de 7 motivos para a Tabela 10. | R01 |
| **R06** | `r06-real-baselines-and-full-overlap-integration` | `short/09_benchmark_cnns_kraken2_blast_overlap.ipynb`, `baselines/*` | Overlap de `0.00%` por amostrar $1.000$ vs. $5.000$ fragmentos; fabricação incondicional de predições para CNNs, Kraken2 e BLASTN na Célula 5 fora do modo smoke. | R01 |
| **R07** | `r07-real-long-sequence-voter-and-ncbi-audit` | `long/01` a `long/04`, `notebook_runtime.py`, `voter.py`, `audit_long.py` | `long/01` cai em fallback sintético por caminho não resolvido; `long/02` usa `MockVoterModel`; `long/03` transcreve a Tabela 12 literal e sorteia 60 erros aleatórios; `long/04` analisa contig aleatório de $3.000\text{ bp}$. | R01, R05 |
| **R08** | `r08-dynamic-paper-tables-and-manuscript-reconciliation` | `reporting/tables.py`, `reporting/figures.py`, `00_regenerate_all_paper_artifacts.ipynb`, `paper/*.tex` | `tables.py` retorna dicionários hardcoded; reconciliação das Tabelas 1–12 (incluindo parâmetros de backbone vs. cabeça `fc` em Species/Genus e resultados empíricos verificados no cluster) no LaTeX do manuscrito e da dissertação. | R01–R07 |

---

## 3. Detalhamento Criterioso de Cada Spec (`tlc-spec-lean`)

### R01 — Reconciliação de Parâmetros, Visibilidade de Saídas no Notebook `01` e Rastreamento Git (`r01-parameter-reconciliation-and-benchmark-tracking`)

#### Problema e Evidência
1. **Contagem de parâmetros (Tabelas 1 e 2):**
   - O manuscrito (`paper/manuscript_submission.tex` linhas 137–157) e `src/viral_cnn/reporting/tables.py` informam `23,5 M` (ResNet50) e `28,1 M` (ConvNeXt) tanto para Species quanto para Genus — números copiados das versões 2D do ImageNet-1K.
   - Nas arquiteturas 1D reais (`src/viral_cnn/models/resnet.py` e `src/viral_cnn/models/convnext.py`, idênticas a `short/utils/models.py`), a dimensão pré-classificador é $D = 2.048$ em ambas as redes (pois `ConvNeXt_Tiny_1D` usa canais base `[64, 128, 256, 512]` com `expansion = 4`, resultando em estágios de `[256, 512, 1024, 2048]` canais e MLPs invertidos $4\times$ que no estágio 4 projetam $2048 \to 8192 \to 2048$):
     - **`ResNet50_1D`:** Backbone = **$15.955.584$ ($15{,}96\text{ M}$)** | Cabeça `fc` Genus ($2.702$) = **$5.536.398$ ($5{,}54\text{ M}$)** $\Rightarrow$ Total Genus = **$21.491.982$ ($21{,}49\text{ M}$)** | Cabeça `fc` Species ($13.683$) = **$28.036.467$ ($28{,}04\text{ M}$)** $\Rightarrow$ Total Species = **$43.992.051$ ($43{,}99\text{ M}$)**.
     - **`ConvNeXt_Tiny_1D` (`HybridConvNeXt`):** Backbone = **$135.977.344$ ($135{,}98\text{ M}$)** | Cabeça `fc` Genus ($2.702$) = **$5.536.398$ ($5{,}54\text{ M}$)** $\Rightarrow$ Total Genus = **$141.513.742$ ($141{,}51\text{ M}$)** | Cabeça `fc` Species ($13.683$) = **$28.036.467$ ($28{,}04\text{ M}$)** $\Rightarrow$ Total Species = **$164.013.811$ ($164{,}01\text{ M}$)**.
2. **Saídas ocultas e CSVs desatualizados no Notebook `01`:**
   - A Célula 4 de `01_benchmark_clean_and_noise.ipynb` executou a avaliação real no cluster, mas salvou `benchmark_clean_noise_summary.csv` sem imprimir o `DataFrame` no output da célula.
   - Como o `.gitignore` continha `results_reproduction/` na linha 12, o `git commit` no cluster não incluiu os CSVs regenerados, deixando `results_reproduction/n02_benchmark/*` com os dados do smoke test antigo em CPU (`16` amostras, `is_smoke_test: true`).

#### Mudanças Obrigatórias (`Surface` & `Landing`)
- **`src/viral_cnn/training/profiler.py` (`profile_inference_model`):** Retornar explicitamente `backbone_params`, `head_params`, `total_params`, `trainable_params`, `size_mb`, `gflops`, `avg_latency_ms`, `std_latency_ms` e `throughput_seq_sec`.
- **`notebooks/reproduction/short/01_benchmark_clean_and_noise.ipynb`:**
  - Quando `not ctx.is_smoke_test`, validar que todos os 7 checkpoints existem no disco (sem fallback para pesos aleatórios) e que `n_samples == 1_706_270` (Species) e `n_samples == 816_756` (Genus).
  - Imprimir na saída das células as tabelas completas `df_prof` (Tabela 2), `df_summary` (Tabelas 1 e 3), `df_f1_bands` (Tabela 4) e a comparação especialista por espécie tanto em $\epsilon = 0{,}05$ quanto em $\epsilon = 0{,}10$ (Nota 45).
  - Registrar no `manifest.json` os hashes SHA-256 ou tamanhos/caminhos exatos dos 7 checkpoints avaliados.
- **`.gitignore`:** Substituir a regra genérica `results_reproduction/` por regras seletivas que ignoram apenas artefatos binários pesados (`results_reproduction/**/*.pt`, `results_reproduction/**/*.parquet`, `results_reproduction/**/*.sqlite*`), permitindo que `git status` e `git commit` capturem todos os `.csv`, `.json`, `.yaml` e `.png` gerados no cluster.

#### Critérios de Verificação (`checks.md`)
- `SHALL` provar que `profile_inference_model` retorna exatamente `backbone_params = 15_955_584` e `head_params = 28_036_467` para `ResNet50_1D(13683)`, e `backbone_params = 135_977_344` e `head_params = 28_036_467` para `ConvNeXt_Tiny_1D(13683)`.
- `SHALL` provar que `git check-ignore` não bloqueia `results_reproduction/n02_benchmark/benchmark_clean_noise_summary.csv`, mas bloqueia `results_reproduction/checkpoints/resnet50_noise/finetune_noise_last.pt`.
- `SHALL` provar que o notebook `01` imprime `df_prof`, `df_summary` e `df_f1_bands` no stdout das células e falha com `FileNotFoundError` se `is_smoke_test=False` e um checkpoint estiver ausente.

---

### R02 — Correção do Fine-Tuning com Ruído (`ControlledTrainer` e Notebooks `03` e `04`) (`r02-finetune-lr-schedule-and-trainer-parity`)

#### Problema e Evidência
1. **LR inicial $20\times$ menor:** Em `src/viral_cnn/training/trainer.py` (linha 55), `ControlledTrainer.__init__` define `initial_lr = self._compute_scheduled_lr(0)`. Como `ControlledExperimentConfig.warmup_steps` vale `20` por padrão, `initial_lr` torna-se $5\times 10^{-5} / 20 = 2{,}5\times 10^{-6}$. Quando os notebooks `03` e `04` criam `CosineAnnealingLR(trainer.optimizer, ...)`, o scheduler assume `base_lrs = [2.5e-6]`, impedindo a adaptação do modelo sob ruído (no `03`, a acurácia de validação estagnou em `5,40%`).
2. **Escalonamento por batch vs. por época e `GradScaler`:** Nos notebooks históricos (`short/finetune_resnet50_species.ipynb` e `short/finetune_convnext_species.ipynb`), `lr_scheduler.step()` era chamado **uma vez ao final de cada época** e não havia `clip_grad_norm_`. Já `ControlledTrainer.train_epoch` chama `scheduler.step()` a cada batch — inclusive quando `GradScaler.step(optimizer)` pula `optimizer.step()` por gradientes `inf`/`NaN` no primeiro batch em FP16, causando o `UserWarning` registrado no notebook `04`.
3. **Interrupção na Época 3/15 do Notebook `04`:** Sem lógica de retomada (`resume_from_checkpoint`) ativa nos notebooks `03` e `04`, uma queda de sessão no JupyterHub exige reiniciar as 15 épocas do zero (~10 horas de GPU).

#### Mudanças Obrigatórias (`Surface` & `Landing`)
- **`src/viral_cnn/training/config.py` e `src/viral_cnn/training/trainer.py`:**
  - Quando `MODE == "historical_reproduction"` (ou quando um scheduler PyTorch externo é acoplado), inicializar `AdamW` diretamente com `lr = config.learning_rate` ($5\times 10^{-5}$), sem dividir por `warmup_steps`.
  - Suportar `scheduler_step_unit: Literal["epoch", "batch"] = "epoch"` em `train_epoch`:
    - No modo `"epoch"` (paridade com o treino histórico), `scheduler.step()` é chamado uma vez ao final da época (após `optimizer.step()` ter ocorrido).
    - No modo `"batch"`, verificar se `self.scaler.get_scale()` não diminuiu no passo antes de chamar `scheduler.step()`, eliminando o `UserWarning` de chamada antes de `optimizer.step()`.
  - Adicionar `load_resume_checkpoint(resume_path)` em `ControlledTrainer` restaurando `model_state_dict`, `optimizer_state_dict`, `scheduler_state_dict`, `scaler_state_dict`, `epoch` e `training_history`.
- **`notebooks/reproduction/short/03_finetune_resnet50_species.ipynb` e `04_finetune_convnext_species.ipynb`:**
  - Adicionar na Célula 3 uma **Avaliação Pré-Treino (Época 0)** em um subconjunto de validação ($\epsilon = 0{,}0$ e $\epsilon = 0{,}05$) e imprimir a acurácia inicial para comprovar que o checkpoint base (`best_resnet_classifier__fast_phase4.pt` ou `best_convnext_classifier_fast_phase4_2_restarts.pt`) foi carregado corretamente antes de iniciar o fine-tuning.
  - Ativar retomada automática a partir de `finetune_noise_last.pt` caso o treinamento seja interrompido e retomado no JupyterHub.
  - Imprimir `history_df` completo e `learning_rate` real ($5\times 10^{-5}$ na Época 1) na saída das células.

#### Critérios de Verificação (`checks.md`)
- `SHALL` provar em teste unitário que `trainer.optimizer.param_groups[0]["lr"] == 5e-5` no início da Época 1 quando configurado com `learning_rate=5e-5` e `CosineAnnealingLR`.
- `SHALL` provar que nenhum `UserWarning` de `lr_scheduler.step() before optimizer.step()` é emitido mesmo quando `GradScaler` encontra gradientes `inf` no primeiro passo.
- `SHALL` provar que `load_resume_checkpoint` retoma exatamente da época `k + 1` preservando o histórico de épocas `1..k`.

---

### R03 — Auditoria Real de Confiança, Entropia Hierárquica e Calibração (`r03-real-calibration-and-hierarchical-entropy`)

#### Problema e Evidência
No notebook `notebooks/reproduction/short/02_analyze_confidence_and_calibration.ipynb`:
1. **Célula 4:** Ignora os exemplos reais coletados em `confused_samples` e passa `seq1 = "ACGT" * 128` e `seq2 = "ACGA" * 128` para `compare_confused_species_identity`.
2. **Célula 6:** Define `species_to_genus = torch.zeros(13683, dtype=torch.long, device=ctx.device)` quando `not ctx.is_smoke_test`, colapsando toda a probabilidade predita em um único gênero fictício (`H_genus = 0.0000`, `Gap = H_species`). Além disso, o output salvo mostrava `H_act: nan` para a ResNet50.
3. **Célula 7:** Executa `audit_model_calibration_by_noise` sobre 16 strings fixas `"ACGT" * 128` com rótulos `i % 5`, mesmo com `Smoke Test Mode: False`.

#### Mudanças Obrigatórias (`Surface` & `Landing`)
- **`src/viral_cnn/taxonomy/mapping.py`:**
  - Adicionar função `build_species_to_genus_tensor(taxonomy_parquet_path, num_species=13683, device="cpu") -> torch.Tensor` que carrega o mapeamento real `model_label (0..13682) -> genus_idx (0..N_genera-1)` a partir de `results_reproduction/taxonomy/main_taxonomy_hybrid_CLEANSED.parquet` ou `cnn_kraken/data_taxonomy_parquet/main_taxonomy_hybrid_ictv_ncbi.parquet`, levantando `ValueError` se todos os índices mapearem para uma única classe fora do modo smoke.
- **`src/viral_cnn/eval/calibration.py`:**
  - Adicionar `audit_dataloader_calibration_by_noise(models_dict, test_loader, val_loader, noise_rates, device, n_bins=15, max_batches=None)` para calcular ECE, MCE, Brier score multiclasse, NLL, confiança média Top-1, acurácia e temperatura ótima ($T^*$ ajustada exclusivamente em `val_loader`) sobre lotes reais de `test_loader` sem estourar memória RAM/VRAM (usando acumulação em bins e expansão quadrática incremental para o Brier score).
- **`notebooks/reproduction/short/02_analyze_confidence_and_calibration.ipynb`:**
  - **Célula 4:** Extrair as sequências nucleotídicas reais decodificadas (`token_to_seq`) de `confused_samples[true_lbl]` e `confused_samples[pred_lbl]` para calcular a identidade real entre as espécies confundidas, imprimindo `df_identity`.
  - **Célula 6:** Carregar `species_to_genus` real via `build_species_to_genus_tensor`, avaliar as 5 variantes de espécie e imprimir `df_entropy` completo (verificando que `H_genus > 0` e `not np.isnan(H_act)`).
  - **Célula 7:** Executar `audit_dataloader_calibration_by_noise` sobre `test_loader` e `val_loader` reais para $\epsilon \in \{0.0, 0.02, 0.05, 0.10, 0.15\}$ e imprimir `df_cal` completo no output da célula.

#### Critérios de Verificação (`checks.md`)
- `SHALL` provar que `build_species_to_genus_tensor` produz um tensor de comprimento $13.683$ com $\ge 2.300$ gêneros distintos sobre o Parquet taxonômico real e falha se receber um mapeamento degenerado de classe única.
- `SHALL` provar que `TaxonomyEntropyAnalyzer` retorna `H_genus > 0` e `H_act` finito (`not math.isnan`) em precisão mista (FP16/FP32).
- `SHALL` provar que o notebook `02` não contém strings `"ACGT" * 128` nem `torch.zeros(13683)` no caminho `is_smoke_test == False`.

---

### R04 — Extração Real de Espaço Latente, Correção de `KNN_Top5` e Saneamento da Curadoria (`r04-real-latent-embeddings-and-taxonomy-audit`)

#### Problema e Evidência
1. **Notebook `05` (`05_ictv_taxonomy_curation.ipynb`, Célula 4):** Cria 100 registros inventados (`mock_records`) mesmo com `Smoke Test Mode: False` e grava `"genome_disjoint_test_classes": 25` no `manifest.json` de produção.
2. **Notebook `06` (`06_ictv_latent_space_and_hierarchy.ipynb`, Células 3–7):**
   - Não carrega nenhum modelo nem Parquet de teste; gera `raw_feats = np.random.randn(1000, m["dim"])` somando `y_species[i] * 0.1` nas 10 primeiras colunas.
   - Define `"dim": 768` para os modelos ConvNeXt (quando o backbone real `ConvNeXt_Tiny_1D` tem dimensão `2048`).
   - Preenche `"KNN_Top1": knn_res["global_knn_accuracy"]` e `"KNN_Top5": knn_res["global_knn_accuracy"]` com o mesmo valor.
   - Produz `N_taxa = 3` (Species $\to$ Genus) e `NaN` / `N_taxa = 1` (Genus $\to$ Order) na Tabela 6.

#### Mudanças Obrigatórias (`Surface` & `Landing`)
- **`notebooks/reproduction/short/05_ictv_taxonomy_curation.ipynb` (Célula 4):**
  - Quando `not ctx.is_smoke_test`, ler o relatório real da auditoria de split (`results_reproduction/retraining_decision/decision.json` gerado pelo notebook `10`) e registrar no manifesto `"genome_provenance_complete": False` e `"genome_disjoint_test_classes": None` (com nota explícita de que os Parquets de fragmentos não possuem `genome_id`). Manter a demonstração algorítmica de `build_genome_disjoint_split` exclusivamente sob `if ctx.is_smoke_test:` ou claramente isolada do manifesto empírico.
- **`src/viral_cnn/eval/latent.py` (`evaluate_cosine_knn`):**
  - Além de `global_knn_accuracy` (Top-1 por voto majoritário dos $k=5$ vizinhos), calcular e retornar `global_knn_top5_accuracy` (fração de consultas cujo rótulo verdadeiro aparece entre os $k=5$ vizinhos mais próximos no conjunto de referência disjunto 80/20).
- **`notebooks/reproduction/short/06_ictv_latent_space_and_hierarchy.ipynb`:**
  - Quando `not ctx.is_smoke_test`, proibir `np.random.randn`: carregar cada um dos 5 modelos de espécie (`build_registered_model`, todos com dimensão pré-`fc` real verificada de `model.fc.in_features == 2048`) e extrair os embeddings reais via `extract_latent_embeddings` sobre o `test_loader` real (ou carregar o cache real `results_latent/features_*.parquet` se existir e for validado, salvando o cache em `results_reproduction/latent_space/`).
  - Mapear os rótulos reais `y_species` para `y_genus` e `y_order` usando `main_taxonomy_hybrid_CLEANSED.parquet` completo ($13.683$ espécies $\to 2.321$ gêneros $\to 69$ ordens).
  - Calcular a Tabela 5 com `KNN_Top1` e `KNN_Top5` distintos e Silhouette amostral real (registrando número de amostras e taxa de classes válidas com $\ge 2$ exemplos), a Tabela 6 (`Species -> Genus` e `Genus -> Order` com $N$ real de táxons pais e $p$-valores bilaterais) e a Sonda Linear (`LinearProbeClassifier`) treinada em embeddings extraídos do split `train` e avaliada em `test`.

#### Critérios de Verificação (`checks.md`)
- `SHALL` provar que `evaluate_cosine_knn` retorna chaves `global_knn_accuracy` (Top-1) e `global_knn_top5_accuracy` (Top-5) com `global_knn_top5_accuracy >= global_knn_accuracy`.
- `SHALL` provar que o notebook `06` extrai embeddings reais através de `extract_latent_embeddings(model, ...)` e lança `FileNotFoundError` se `is_smoke_test=False` e os Parquets/checkpoints estiverem ausentes.

---

### R05 — Coocorrência Real de Jaccard e Sinais de Fronteira em Motivos de 1ª Camada (`r05-real-motif-jaccard-and-triplet-analysis`)

#### Problema e Evidência
1. **Notebook `07` (`07_first_layer_motifs_and_perturbation.ipynb`, Célula 9):** Embora as Células 3, 5 e 7 usem os 5 checkpoints reais e avaliem perturbação em $2.048$ fragmentos reais, a Célula 9 constrói uma matriz sintética fixa `presence = np.zeros((40, len(motifs_list)))` com `presence[:, 0] = 1.0`, `presence[:20, 1] = 1.0`, `presence[:15, 2] = 1.0` para todas as 5 arquiteturas.
2. **Notebook `08` (`08_triplet_composition_and_cooccurrence.ipynb`, Células 5, 7 e 9):**
   - Na Célula 5 (Tabela 10), avalia uma lista hardcoded de 7 strings (`sample_motifs = ["ATGAAAA", "TAACCCC", ...]`) em vez dos motivos extraídos/validados no notebook `07`.
   - Nas Células 7 e 9 (Tabela 11 e correlações inter-arquitetura), lê os Parquets de Jaccard degenerados salvos pela Célula 9 do notebook `07`, resultando em `r_canonical = 0.0`, `p_permutation_null = 1.0` e correlação `r = 1.0` entre modelos distintos.

#### Mudanças Obrigatórias (`Surface` & `Landing`)
- **`src/viral_cnn/interpretability/motifs.py`:**
  - Adicionar `compute_motif_presence_matrix_from_loader(loader, motifs_list, max_sequences=10000)` que escaneia os fragmentos reais do `eval_loader` (decodificando tokens `0,1,2,3 -> A,C,G,T`) e preenche `presence[seq_idx, motif_idx] = 1.0` sempre que o 7-mer ocorrer na sequência (equivalente à função histórica `fast_motif_cooccurrence` da Célula 32 de `short/full_loss_function.ipynb`).
- **`notebooks/reproduction/short/07_first_layer_motifs_and_perturbation.ipynb`:**
  - Avaliar a perturbação de motivos (Célula 7) para todas as 5 variantes de espécie (não apenas `resnet50_species`), salvando todos os motivos auditados em `table_08_motif_perturbation.csv` e `motif_summary.parquet`.
  - Na Célula 9, calcular a matriz de presença real via `compute_motif_presence_matrix_from_loader` sobre as sequências reais do Parquet de teste para cada variante, gerando matrizes de Jaccard empiricamente genuínas.
- **`notebooks/reproduction/short/08_triplet_composition_and_cooccurrence.ipynb`:**
  - Na Célula 3, amostrar um conjunto representativo maior de sequências reais do teste (ex.: $50.000$ fragmentos quando `not ctx.is_smoke_test`) e calcular também a correlação de Spearman ($\rho$, $p$-valor e FDR) entre o perfil RSCU de fundo e a frequência de tripletos nos motivos de cada uma das 5 variantes (Tabela 9 do manuscrito).
  - Na Célula 5 (Tabela 10), carregar os motivos reais de `results_reproduction/motifs/motif_summary.parquet` (gerado pelo notebook `07`) para auditar `ATG` e `TAA/TAG/TGA` por arquitetura.
  - Nas Células 7 e 9 (Tabela 11), reexecutar a análise de simetria, permutação nula e correlação inter-arquitetura sobre as matrizes de Jaccard reais geradas pelo notebook `07`.

#### Critérios de Verificação (`checks.md`)
- `SHALL` provar que `compute_motif_presence_matrix_from_loader` detecta exatamente a presença de 7-mers reais nas sequências do `DataLoader`, sem linhas constantes fabricadas.
- `SHALL` provar que o notebook `08` carrega os motivos validados de `motif_summary.parquet` quando `is_smoke_test == False` e falha com `FileNotFoundError` se o artefato do notebook `07` estiver ausente.

---

### R06 — Benchmark Real de Baselines (CNNs, Kraken2 e BLASTN) e Estratificação de Overlap Completa (`r06-real-baselines-and-full-overlap-integration`)

#### Problema e Evidência
No notebook `notebooks/reproduction/short/09_benchmark_cnns_kraken2_blast_overlap.ipynb`:
1. **Célula 3 (0% de overlap):** Amostra apenas `max_rows=1000` do teste e `max_rows=5000` do treino ($0{,}036\%$ dos $13{,}65\text{ M}$ de fragmentos de treino), resultando em `0 (0.00%)` de sobreposição exata — aparentando contradizer os `95,94%` do notebook `10` (teste completo) e os `95,91%` do notebook histórico `short/benchmark_cnns_kraken2_blast_overlap.ipynb` ($273.660$ consultas balanceadas, 20 por espécie).
2. **Célula 5 (Predições fabricadas):** Mesmo com `Smoke Test Mode: False`, ignora os arquivos reais em `/home/mlima/workspace/cnn_kraken/benchmark_kraken_blast_20_per_species/` e fabrica as predições de todos os 7 métodos aplicando máscaras aleatórias `rng.uniform(0, 1, size=num_queries) > acc_target` sobre `y_true_all`.

#### Mudanças Obrigatórias (`Surface` & `Landing`)
- **`src/viral_cnn/baselines/` (`overlap.py` e `paired.py`):**
  - Adicionar carregador `load_prepared_baseline_benchmark(benchmark_dir: Path)` que lê os artefatos reais gerados no diretório `cnn_kraken/benchmark_kraken_blast_20_per_species/` do cluster (conforme `short/benchmark_cnns_kraken2_blast_overlap.ipynb` Células 22–31):
    1. Tabela de verdade e auditoria de overlap contra o treino completo (`exact_overlap.csv` / `truth.parquet`, $N = 273.660$ consultas, $262.470$ `overlap` = $95{,}91\%$ e $11.190$ `no_exact_overlap` = $4{,}09\%$, ou o índice completo de `results_reproduction/retraining_decision`).
    2. Predições reais por consulta e nível de ruído $\epsilon \in \{0.0, 0.02, 0.05, 0.10, 0.15\}$ em `comparison_results/predictions_<method>_noise_<pct>.parquet` (ou reconstrução direta via `parse_kraken_output`, `parse_blast_output` e inferência dos checkpoints caso apenas os arquivos brutos `.out`/`.tsv` estejam presentes).
    3. Medições reais de tempo de execução (`comparison_results/runtime.jsonl`).
- **`notebooks/reproduction/short/09_benchmark_cnns_kraken2_blast_overlap.ipynb`:**
  - Quando `not ctx.is_smoke_test`, exigir a presença dos artefatos reais em `ctx.topology.paths["short"]["kraken_blast_dir"]` (proibindo qualquer uso de `acc_target` ou perturbação sintética de `y_true_all`).
  - Auditar e imprimir o resumo real de overlap ($N = 273.660$, ~95,91% overlap, 5.992 espécies sem nenhuma query livre de overlap exato), as métricas estratificadas reais (`all`, `overlap`, `no_exact_overlap`), as discordâncias pareadas e o cluster bootstrap por espécie (2.000 repetições, semente 42).

#### Critérios de Verificação (`checks.md`)
- `SHALL` provar que `load_prepared_baseline_benchmark` valida a unicidade e o alinhamento estrito de `query_id` entre a tabela de overlap e todos os arquivos `predictions_<method>_noise_<pct>.parquet`.
- `SHALL` provar que o notebook `09` lança `FileNotFoundError` quando `is_smoke_test=False` e o diretório `benchmark_kraken_blast_20_per_species` não existe, jamais executando geração sintética fora do modo smoke.

---

### R07 — Reprodução Empírica de Sequências Longas (`LongSequenceVoter`, Tabela 12 e Motivos) (`r07-real-long-sequence-voter-and-ncbi-audit`)

#### Problema e Evidência
Nos notebooks `notebooks/reproduction/long/01` a `04`:
1. **`long/01`:** `ctx.topology.paths["long"]["ncbi_parquet_dir"]` procurava apenas `/home/mlima/workspace/processed_ncbi_splited_parquet`, sem buscar nas subpastas do workspace remoto (como `/home/mlima/workspace/long/`, `/home/mlima/workspace/cnn_long/` ou `/home/mlima/workspace/cnn_kraken/`), caindo silenciosamente no bloco `if ... not Path(long_parquet).exists():` que criava 15 sequências sintéticas mesmo com `Smoke Test Mode: False`.
2. **`long/02`:** Instanciava `MockVoterModel(num_classes=5)` e 4 contigs aleatórios (`read_001` a `read_004`) de forma incondicional.
3. **`long/03`:** Transcrevia manualmente os números da Tabela 12 em uma lista literal `table_12_data` e gerava 60 linhas aleatórias (`rng.randint(0, 5, size=60)`) para calcular a estratificação de erros.
4. **`long/04`:** Gerava um único contig aleatório de $3.000\text{ bp}$ (`rng.choice(bases, size=3000)`) com zero ocorrências dos motivos rastreados.

#### Mudanças Obrigatórias (`Surface` & `Landing`)
- **`src/viral_cnn/notebook_runtime.py` (`JupyterHubDataTopology`):**
  - Implementar resolução multi-candidato (`resolve_long_artifact_path`) para os artefatos de sequências longas (`processed_ncbi_splited_parquet`, `mapping_species_final.csv`, `mapping_genus_final.csv`, `raw_database/final.csv`, `dataset_reads_sinteticos_CLEAN.parquet`, `taxonomia_recuperada.csv` e `resultados_inferencia_long_reads_*.parquet`), buscando em `self.root`, `self.root / "long"`, `self.root / "cnn_long"`, `self.root / "cnn_kraken"` e `project_root / "long"`.
- **`notebooks/reproduction/long/01_ncbi_download_and_prepare_database.ipynb`:**
  - Quando `not ctx.is_smoke_test`, exigir os arquivos reais de `processed_ncbi_splited_parquet/test` (e `raw_database/final.csv` / `dataset_reads_sinteticos_CLEAN.parquet` se presentes), calculando o overlap real com o treino e a distribuição empírica de leituras por categoria de comprimento genômico.
- **`notebooks/reproduction/long/02_cnn_long_voter_inference.ipynb`:**
  - Quando `not ctx.is_smoke_test`, carregar os 5 checkpoints reais de espécie (`build_registered_model`) e o mapeamento `mapping_species_final.csv`.
  - Verificar a equivalência de logits (`verify_window_logit_equivalence`) no modelo real `resnet_clean_species`.
  - Executar o `LongSequenceVoter` (janela 512, stride 256, `mean_logits` pré-softmax e `majority_vote`, comparando `handle_short="pad"` vs `"skip"`) sobre o conjunto de teste real `processed_ncbi_splited_parquet/test` ($N = 4.360$, ou carregar/validar contra os Parquets `resultados_inferencia_long_reads_*.parquet` salvos no cluster mais lote verificável ao vivo), salvando as predições completas em `results_reproduction/long_voter/`.
- **`notebooks/reproduction/long/03_cnn_long_voter_compare.ipynb`:**
  - Quando `not ctx.is_smoke_test`, remover o dicionário hardcoded `table_12_data` e o `rng.randint`: computar a **Tabela 12** (`acc_mean_logits`, `acc_majority_vote`, `n_sequences`) e a estratificação de erros (`stratify_long_errors` por faixa de comprimento, proporção de bases IUPAC ambíguas e espécie/gênero via `taxonomia_recuperada.csv`) diretamente a partir dos Parquets reais de predição das 5 variantes.
- **`notebooks/reproduction/long/04_long_motifs_and_gc_analysis.ipynb`:**
  - Quando `not ctx.is_smoke_test`, carregar contigs reais de `processed_ncbi_splited_parquet/test` e os modelos reais de espécie para analisar o perfil de GC e o impacto/contagem de motivos de 1ª camada ao longo das janelas de $512\text{ bp}$ (reproduzindo `long/long_loss_function.ipynb`).

#### Critérios de Verificação (`checks.md`)
- `SHALL` provar que `long/01`, `long/02`, `long/03` e `long/04` lançam `FileNotFoundError` quando `is_smoke_test=False` e os artefatos reais de `long` não estão presentes, sem jamás executar `MockVoterModel`, `table_12_data` literal ou `rng.randint` fora do modo smoke.
- `SHALL` provar que `long/03` deriva os valores da Tabela 12 matematicamente a partir do DataFrame de predições carregado.

---

### R08 — Regeneração Dinâmica de Tabelas/Figuras e Reconciliação Final do Manuscrito (`r08-dynamic-paper-tables-and-manuscript-reconciliation`)

#### Problema e Evidência
1. **`src/viral_cnn/reporting/tables.py`:** As funções `generate_table_1_metrics()` a `generate_table_12_voter_consensus()` retornam listas literais fixas em vez de ler os CSVs/Parquets gerados em `results_reproduction/`.
2. **Manuscrito (`paper/manuscript_submission.tex` e `paper/manuscript_review.tex`):** As Tabelas 1 e 2 ainda exibem `23.5` e `28.1` milhões de parâmetros, e os demais números precisam ser confrontados célula a célula contra os CSVs reais gerados no cluster pelas specs `R01–R07`.

#### Mudanças Obrigatórias (`Surface` & `Landing`)
- **`src/viral_cnn/reporting/tables.py` e `src/viral_cnn/reporting/figures.py`:**
  - Modificar `generate_all_tables(output_dir, results_dir=Path("results_reproduction"), strict_empirical=False)` para ler diretamente os CSVs/Parquets gerados por `R01–R07`:
    - Verificar em cada `manifest*.json` de `results_reproduction/` se `"is_smoke_test": false` quando `strict_empirical=True` (recusando gerar tabelas de artigo a partir de artefatos de smoke test).
    - Emitir um relatório automático de reconciliação (`results_reproduction/reconciliation_report.csv` e `.md`) comparando cada número calculado na execução real com o número anteriormente impresso no manuscrito.
- **`paper/manuscript_submission.tex`, `paper/manuscript_review.tex` e `src/viral_cnn/reporting/manuscript.py`:**
  - Atualizar as Tabelas 1 e 2 para explicitar os parâmetros reais de **Backbone 1D** ($15{,}96\text{ M}$ na ResNet50 vs. $135{,}98\text{ M}$ na `ConvNeXt_Tiny_1D` / `HybridConvNeXt`) e **Total com Cabeça `fc`** ($21{,}49\text{ M}$ em Genus e $43{,}99\text{ M}$ em Species para ResNet50; $141{,}51\text{ M}$ em Genus e $164{,}01\text{ M}$ em Species para ConvNeXt), documentando no texto de *Methods* por que a adaptação `HybridConvNeXt` (`expansion=4` nos estágios `[256, 512, 1024, 2048]` + MLP invertido $4\times$) possui $135{,}98\text{ M}$ de parâmetros no backbone.
  - Sincronizar todas as tabelas e métricas do texto com os resultados empíricos verificados em `results_reproduction/`.

#### Critérios de Verificação (`checks.md`)
- `SHALL` provar que `generate_all_tables(..., strict_empirical=True)` falha com `RuntimeError` se qualquer manifesto em `results_reproduction/` tiver `"is_smoke_test": true`.
- `SHALL` provar que as Tabelas 1 e 2 geradas em `paper/tables/` e no `.tex` refletem os parâmetros exatos ($43{,}99\text{ M}$ / $21{,}49\text{ M}$ para ResNet50 e $164{,}01\text{ M}$ / $141{,}51\text{ M}$ para ConvNeXt).

---

## 4. Ordem de Execução Recomendada

Para cada spec `R01` a `R08`, o ciclo seguirá estritamente os 4 portões do `tlc-spec-lean`:

1. **`PLAN`:** Escrever `.specs/features/<slug>/plan.md` e validar com `python3 .agent/skills/tlc-spec-lean/scripts/validate_plan.py <slug>`.
2. **`CHECKS`:** Derivar `.specs/features/<slug>/checks.md` e validar com `python3 .agent/skills/tlc-spec-lean/scripts/validate_checks.py <slug>`.
3. **`BUILD`:** Implementar os testes e o código em `src/viral_cnn/` e `notebooks/reproduction/`, validar localmente com `REMOTE_SERVER=False` e commitar com `check_commit.py`.
4. **`VERIFY`:** Despachar um subagente **Verifier** independente sobre `<base>..HEAD` e validar `.specs/features/<slug>/verification.md` com `python3 .agent/skills/tlc-spec-lean/scripts/validate_verification.py <slug>`.
5. **Execução no Cluster (JupyterHub):** Após o `VERIFY` de cada spec (ou do bloco `R01–R07`), rodar os notebooks correspondentes no JupyterHub (`REMOTE_SERVER=true`, `SMOKE_TEST=0`) e commitar tanto os `.ipynb` com outputs completos quanto os `.csv`/`.json`/`.png` em `results_reproduction/` antes de fechar a reconciliação final na **`R08`**.

