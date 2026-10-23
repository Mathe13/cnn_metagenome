# N02: Short-Read Clean Benchmark, Noise Robustness, Hierarchical Entropy, and Calibration Notebooks

## Problem

A avaliação de fragmentos curtos (512 bp) que sustenta as Tabelas 1, 2, 3, 4 e 7 e a Figura 1 do manuscrito encontra-se fragmentada em cinco notebooks históricos distintos (`short/benchmark.ipynb`, `short/finetune_resnet50_species.ipynb`, `short/finetune_convnext_species.ipynb`, `short/analize_convnext.ipynb`, `short/full_loss_function.ipynb` e células auxiliares em `long/cnn-long-voter-compare.ipynb`), apresentando quatro lacunas de reprodutibilidade e padronização:

1. **Fragmentação e Limite Parcial de Lotes na Avaliação de Ruído (`short/finetune_*_species.ipynb` e `short/benchmark.ipynb`):** A função `evaluate_robustness` nas células 5–7 e 12 de `short/finetune_*_species.ipynb` truncava a avaliação em `limit_batches = 100`, usava o RNG global sem semente fixa por nível de ruído $\epsilon \in \{0.0, 0.02, 0.05, 0.10, 0.15\}$, não verificava a taxa efetiva de substituição ($3\epsilon/4$) sobre posições válidas e estava separada do profiling computacional (`profile_model` em `short/benchmark.ipynb`).
2. **Parametrização de Limiares da Tabela 4 e Resolução de Empates por Espécie (`src/viral_cnn/eval/metrics.py` e `long/cnn-long-voter-compare.ipynb`):** `compute_species_f1_bands` em `src/viral_cnn/eval/metrics.py` suporta apenas os cortes fixos `(0.0, 0.50, 0.80, 0.95, 1.0)`, enquanto a Tabela 4 do artigo (Nota 34) requer também a distribuição contínua nos limiares `(0.0, 0.50, 0.80, 0.90, 1.0)` (`[0.90, 1.00]`, `[0.80, 0.90)`, `[0.50, 0.80)`, `[0.00, 0.50)`). Além disso, a rotina `comparar_especialistas` / `gerar_relatorio_confiabilidade` (células 14–33 de `long/cnn-long-voter-compare.ipynb`) que contabiliza vitórias por espécie e empates ($11.826 + 1.202 + 5 = 13.033$ em $\epsilon=0{,}10$, Nota 45) ainda não está exposta em `viral_cnn.eval.metrics`.
3. **Acoplamento de `TaxonomyEntropyAnalyzer` ao Notebook Histórico (`short/full_loss_function.ipynb`):** A classe `TaxonomyEntropyAnalyzer` (células 37–49 de `short/full_loss_function.ipynb`), responsável pelo cálculo da entropia preditiva em *nats* ($\ln 13.683 \approx 9{,}52$), agregação probabilística Species $\to$ Genus via `scatter_add_`, gap hierárquico $\Delta H = H_{\text{species}} - H_{\text{genus}}$ e entropia de ativação pré-classificadora normalizada $H_{\text{latent}} / \ln D$ (Tabela 7, Nota 37), está definida apenas inline no notebook e ausente de `src/viral_cnn/eval/calibration.py`.
4. **Falta dos Notebooks Reprodutíveis Desacoplados e Auditados para Calibração Formal (`notebooks/reproduction/short/01` e `02`):** `short/analize_convnext.ipynb` importava `utils.analize_funcs` e `utils.gradcam` com `import *`, dependia de chamadas de rede ao NCBI Entrez com e-mail pessoal hardcoded para alinhamento de pares confundidos e não integrava lado a lado as métricas formais de calibração probabilística (`compute_calibration_metrics`, `compute_reliability_diagram`, `TemperatureScaler` ajustado exclusivamente na validação e `audit_model_calibration_by_noise` para ECE, MCE, Brier score e NLL, `AD-013`).

Quando esta feature for entregue, `src/viral_cnn/` proverá as extensões em `metrics.py`, `calibration.py`, `runner.py`, `confusion.py`, `profiler.py` e `notebook_runtime.py`, e os novos notebooks `notebooks/reproduction/short/01_benchmark_clean_and_noise.ipynb` e `notebooks/reproduction/short/02_analyze_confidence_and_calibration.ipynb` executarão de ponta a ponta via `viral_cnn.*` (conectados ao JupyterHub remoto ou em modo `SMOKE_TEST=1`), salvando artefatos estruturados em `results_reproduction/` para alimentar as Tabelas 1, 2, 3, 4 e 7 e a Figura 1, mantendo `short/` e `long/` intocados.

## Flow

Reutiliza `viral_cnn.notebook_runtime` (exists), `viral_cnn.data.parquet_dataset:ParquetBatchIterableDataset` (exists), `viral_cnn.models.registry` (exists), `viral_cnn.eval.noise` (exists), `viral_cnn.eval.confusion` (exists), `viral_cnn.interpretability.gradcam` (exists) e `viral_cnn.taxonomy.mapping` (exists).

1. `RuntimeContext` de `viral_cnn.notebook_runtime` (exists) + 7 variantes registradas em `viral_cnn.models.registry` (exists) -> `viral_cnn.training.profiler:profile_inference_model` (exists, estendido) e `viral_cnn.eval.runner:evaluate_dataloader` (exists, estendido) - perfila parâmetros, memória, GFLOPs, latência e throughput e executa inferência determinística em lotes sobre Species (13.683 classes) e Genus (2.702 classes) para $\epsilon \in \{0.0, 0.02, 0.05, 0.10, 0.15\}$
2. predições e probabilidades por consulta/espécie -> `viral_cnn.eval.metrics` (exists, door 1) - calcula Top-1, Top-5, Loss, Macro-F1, Weighted-F1, faixas contínuas de F1 parametrizadas (`compute_species_f1_bands` com suporte a `(0.0, 0.50, 0.80, 0.90, 1.0)` e `(0.0, 0.50, 0.80, 0.95, 1.0)`) e comparação especialista por espécie (`compute_species_specialist_comparison` resolvendo vitórias e empates)
3. modelo 1D + tensor de mapeamento `species_to_genus_map` + lotes de entrada -> `viral_cnn.eval.calibration:TaxonomyEntropyAnalyzer` (new, door 2) - captura ativações pré-`fc` via forward pre-hook, agrega probabilidades Species $\to$ Genus com `scatter_add_` e computa $H_{\text{species}}$, $H_{\text{genus}}$, $\Delta H$ e $H_{\text{activation}}$ normalizada em $[0, 1]$
4. pipelines orquestrados -> `notebooks/reproduction/short/01_benchmark_clean_and_noise.ipynb` (new, door 3) e `notebooks/reproduction/short/02_analyze_confidence_and_calibration.ipynb` (new, door 3) - persistem tabelas, curvas de calibração, matrizes de confusão, mapas GradCAM/ScoreCAM e manifestos `NotebookRunManifest` em `results_reproduction/`

## Impact

| Front | What changes |
| --- | --- |
| domain | new term: `TaxonomyEntropyAnalyzer` - analisador de entropia preditiva hierárquica em *nats* (Species e Genus via `scatter_add_`) e entropia de ativação pré-classificadora normalizada por $\ln D$, vive em `viral_cnn.eval.calibration` |
| domain | new term: `compute_species_specialist_comparison` - agregador por espécie de amostras, acertos, confiança média, F1-score e classificação de vitórias/empates entre dois modelos (`ResNet50` vs `ConvNeXt`), vive em `viral_cnn.eval.metrics` |
| domain | existing term: `compute_species_f1_bands` - mantinha limiares fixos `(0.0, 0.50, 0.80, 0.95, 1.0)`; passa a aceitar `thresholds: Tuple[float, ...] = (0.0, 0.50, 0.80, 0.95, 1.0)` preservando compatibilidade total com `tests/test_noise_and_evaluation.py` e adicionando suporte a `(0.0, 0.50, 0.80, 0.90, 1.0)` da Tabela 4 |
| stored data | nada a migrar: notebooks originais em `short/` e `long/` permanecem intocados; novos notebooks são criados em `notebooks/reproduction/short/` e gravam saídas em `results_reproduction/` |

## Relations

None - no stored-data shape change

## Surface

None - nothing consumed outside

## Landing

| One-way door | Literal shape | Alternative rejected |
| --- | --- | --- |
| Parametrização retrocompatível de faixas de F1 e comparação especialista por espécie | `compute_species_f1_bands(y_true, y_pred, num_classes, thresholds=(0.0, 0.50, 0.80, 0.95, 1.0), confidences=None)` e `compute_species_specialist_comparison(y_true, preds_resnet, preds_convnext, confs_resnet=None, confs_convnext=None, species_names=None)` em `src/viral_cnn/eval/metrics.py` | substituir os cortes `(0.0, 0.50, 0.80, 0.95, 1.0)` in-place sem parâmetro default - rejeitado porque quebraria os contratos existentes da S05 (`tests/test_noise_and_evaluation.py`) |
| Contrato de `TaxonomyEntropyAnalyzer` em `src/viral_cnn/eval/calibration.py` | `TaxonomyEntropyAnalyzer(model, species_to_genus_map)` com `register_forward_pre_hook` em `target_model.fc`, `get_genus_probs` via `scatter_add_`, `calculate_entropy` em *nats*, `calculate_activation_entropy(..., normalize=True)` dividido por $\ln D$ e `analyze_batch(inputs)` retornando `{"h_species", "h_genus", "h_activation", "hierarchical_entropy_gap"}` | reimplementar cálculo de entropia apenas sobre logits pós-classificador sem hook pré-`fc` - rejeitado porque não reproduz a coluna de entropia de ativação latente normalizada da Tabela 7 (`short/full_loss_function.ipynb` células 37–49) |
| Estrutura e contrato de execução dos notebooks `01` e `02` em `notebooks/reproduction/short/` | `01_benchmark_clean_and_noise.ipynb` e `02_analyze_confidence_and_calibration.ipynb` inicializados via `setup_notebook_runtime`, importando exclusivamente `viral_cnn.*`, suportando `SMOKE_TEST=1` sem acesso externo de rede e emitindo artefatos + `NotebookRunManifest` em `results_reproduction/` | manter chamadas obrigatórias ao NCBI Entrez (`Bio.Entrez`) ou imports `from utils.* import *` durante a execução padrão - rejeitado porque viola o isolamento de sandbox/CI e `AGENTS.md` |

- Nothing else in this change is hard to reverse

## Criteria

### S1: Faixas Parametrizadas de F1 por Espécie, Comparação Especialista e Profiling de Inferência (P1)

Extensão de `src/viral_cnn/eval/metrics.py`, `src/viral_cnn/eval/runner.py` e `src/viral_cnn/training/profiler.py` para cobrir integralmente os cálculos das Tabelas 1, 2, 3 e 4.

**Acceptance Criteria**

1. WHEN `compute_species_f1_bands` in `src/viral_cnn/eval/metrics.py` is called with the default `thresholds=(0.0, 0.50, 0.80, 0.95, 1.0)` THEN the system SHALL return the keys `band_0_to_50`, `band_50_to_80`, `band_80_to_95`, and `band_95_to_100` whose `count` sum strictly equals `num_classes`.
2. WHEN `compute_species_f1_bands` is called with Table 4 thresholds `thresholds=(0.0, 0.50, 0.80, 0.90, 1.0)` (and optional per-sample `confidences`) THEN the system SHALL partition all `num_classes` into four continuous intervals `[0.00, 0.50)`, `[0.50, 0.80)`, `[0.80, 0.90)`, and `[0.90, 1.00]` (`band_0_to_50`, `band_50_to_80`, `band_80_to_90`, `band_90_to_100`), computing class count, percentage, and mean confidence per band with exact class conservation (`sum(count) == num_classes`).
3. IF `compute_species_f1_bands` is called with `thresholds` that are not strictly increasing or do not span `[0.0, 1.0]` THEN the system SHALL raise a `ValueError`.
4. The system SHALL provide `compute_species_specialist_comparison` in `src/viral_cnn/eval/metrics.py` that computes per-species sample count, correct predictions, accuracy, F1-score, and mean confidence for two compared models (`resnet` vs `convnext`), classifying each evaluated species into `convnext_superior`, `resnet_superior`, `ties` (both with equal accuracy/correct count $> 0$), or `both_zero`, such that `convnext_superior + resnet_superior + ties + both_zero` equals total evaluated species.
5. The system SHALL provide `profile_inference_model` in `src/viral_cnn/training/profiler.py` and `evaluate_dataloader` in `src/viral_cnn/eval/runner.py` to measure parameter count (`total_params`, `trainable_params`, `size_mb`), GFLOPs, latency (`avg_latency_ms`, `std_latency_ms`), and throughput (`throughput_seq_sec`), and to evaluate a PyTorch `DataLoader` across noise rates $\epsilon \in \{0.0, 0.02, 0.05, 0.10, 0.15\}$ with deterministic `inject_noise` (`seed=42`) and empirical substitution rate verification (`3 * epsilon / 4`).

**Independent test:** `pytest tests/test_n02_benchmark_and_calibration.py -k "f1_bands or specialist or dataloader or profile"`

### S2: Analisador de Entropia Hierárquica e Ativação Pré-Classificadora (`TaxonomyEntropyAnalyzer`) (P1)

Migração de `TaxonomyEntropyAnalyzer` das células 37–49 de `short/full_loss_function.ipynb` para `src/viral_cnn/eval/calibration.py`.

**Acceptance Criteria**

6. The system SHALL provide `TaxonomyEntropyAnalyzer` in `src/viral_cnn/eval/calibration.py` that registers a forward pre-hook on the model's `fc` layer (handling `_orig_mod` wrappers if present) to capture pre-classifier latent representations of shape `(B, D)` and removes the hook cleanly via `remove_hook()` or context manager exit.
7. IF `TaxonomyEntropyAnalyzer` is initialized with a model lacking an `fc` classification layer or with a `species_to_genus_map` containing negative indices THEN the system SHALL raise an `AttributeError` or `ValueError` respectively.
8. WHEN `TaxonomyEntropyAnalyzer.analyze_batch(inputs)` is executed on a batch of tokenized sequences `(B, 512)` THEN the system SHALL compute species probabilities via `softmax`, aggregate genus probabilities via `scatter_add_` over `species_to_genus_map`, and return a dictionary containing 1D NumPy arrays `h_species` (in nats, bounded by $\ln N_{\text{species}}$), `h_genus` (in nats, bounded by $\ln N_{\text{genus}}$), `hierarchical_entropy_gap` ($\Delta H = H_{\text{species}} - H_{\text{genus}} \ge -10^{-5}$), and `h_activation` (normalized by $\ln D$ to `[0.0, 1.0]`).
9. WHEN `TaxonomyEntropyAnalyzer.analyze_dataloader(dataloader)` is executed THEN the system SHALL aggregate batch-level and sample-weighted mean values for `h_species`, `h_genus`, `h_activation`, and `hierarchical_entropy_gap` to populate Table 7 summaries.

**Independent test:** `pytest tests/test_n02_benchmark_and_calibration.py -k "entropy"`

### S3: Notebook `01_benchmark_clean_and_noise.ipynb` (Avaliação Limpa, Ruído e Profiling) (P1)

Criação e validação executável de `notebooks/reproduction/short/01_benchmark_clean_and_noise.ipynb`.

**Acceptance Criteria**

10. The system SHALL provide `notebooks/reproduction/short/01_benchmark_clean_and_noise.ipynb` using `viral_cnn.notebook_runtime`, `viral_cnn.models.registry`, `viral_cnn.data.parquet_dataset`, `viral_cnn.eval.*`, and `viral_cnn.training.profiler` without importing `short/utils/*` or defining model architectures inline.
11. WHEN `01_benchmark_clean_and_noise.ipynb` is executed (including under `SMOKE_TEST=1`) THEN the notebook SHALL evaluate the 5 species variants (`resnet_clean_species`, `resnet_noise_species`, `convnext_zero_species`, `convnext_transfer_species`, `convnext_noise_species`) across $\epsilon \in \{0.0, 0.02, 0.05, 0.10, 0.15\}$ and the 2 genus variants (`resnet_genus`, `convnext_genus`), profile computational efficiency (`profile_inference_model`), compute Table 4 F1 bands (`thresholds=(0.0, 0.50, 0.80, 0.90, 1.0)`) and pairwise species specialist comparisons, and save structured CSV/JSON artifacts plus `NotebookRunManifest` under `results_reproduction/n02_benchmark/`.

**Independent test:** `pytest tests/test_n02_benchmark_and_calibration.py -k "notebook_01"`

### S4: Notebook `02_analyze_confidence_and_calibration.ipynb` (Confusão, Identidade, GradCAM, Entropia e Calibração) (P1)

Criação e validação executável de `notebooks/reproduction/short/02_analyze_confidence_and_calibration.ipynb` e utilitário de identidade de sequência entre pares confundidos.

**Acceptance Criteria**

12. The system SHALL provide `compare_confused_species_identity` in `src/viral_cnn/eval/confusion.py` that computes global pairwise sequence identity and alignment score between representative sequences of confused class pairs directly from local dataset samples (with optional NCBI Entrez lookup disabled by default), avoiding mandatory external network calls.
13. The system SHALL provide `notebooks/reproduction/short/02_analyze_confidence_and_calibration.ipynb` using `viral_cnn.eval.confusion`, `viral_cnn.interpretability.gradcam`, `viral_cnn.eval.calibration`, and `viral_cnn.taxonomy.mapping` without importing `short/utils/*`.
14. WHEN `02_analyze_confidence_and_calibration.ipynb` is executed (including under `SMOKE_TEST=1`) THEN the notebook SHALL run confusion matrix insights and sequence identity inspection on confused pairs, compute 1D `GradCAM1D` / `ScoreCAM1D` attributions with padding zeroed, execute `TaxonomyEntropyAnalyzer` (for Table 7), run formal calibration auditing (`compute_calibration_metrics`, `compute_reliability_diagram`, `TemperatureScaler` fitted strictly on `"validation"`, and `audit_model_calibration_by_noise` across $\epsilon \in \{0.0, 0.02, 0.05, 0.10, 0.15\}$), and save structured CSV/JSON artifacts plus `NotebookRunManifest` under `results_reproduction/n02_calibration/`.
15. The system SHALL preserve all original files in `short/` and `long/` completely untouched (`git status --porcelain short/ long/` empty).

**Independent test:** `pytest tests/test_n02_benchmark_and_calibration.py -k "notebook_02 or identity or preservation"`

## Out of scope

Product capabilities only. Process and harness rules live in AGENTS.md or as Observable `n/a`.

| Excluded | Why |
| --- | --- |
| Treinamento e fine-tuning com ruído (`03_finetune_resnet50_species.ipynb` e `04_finetune_convnext_species.ipynb`) | Escopo exclusivo da Spec `N03` no `ROADMAP_REPRODUCAO_NOTEBOOKS.md` |
| Extração de motivos de 7-mers da 1ª camada, perturbação pareada por GC e matriz de Jaccard (`07` e `08`) | Escopo exclusivo da Spec `N05` no `ROADMAP_REPRODUCAO_NOTEBOOKS.md` |
| Avaliação de sequências longas via `LongSequenceVoter` (`notebooks/reproduction/long/01..04`) | Escopo exclusivo da Spec `N07` no `ROADMAP_REPRODUCAO_NOTEBOOKS.md` |

## Assumptions

Defaults that are not already a numbered criterion. Drop a row once it is.

| Assumption | Chosen default | Rationale | Confirmed? |
| --- | --- | --- | --- |
| Mapeamento `species_to_genus_map` quando o Parquet híbrido ICTV (`N04`) não está presente no modo smoke-test | Construir tensor determinístico `species_to_genus_map` em `ensure_smoke_test_artifacts` / helper de `calibration.py` agrupando as espécies sintéticas em gêneros válidos `[0 .. num_genus - 1]` | Permite que `TaxonomyEntropyAnalyzer` seja exercitado de ponta a ponta em `02_analyze_confidence_and_calibration.ipynb` antes da execução de `N04`, usando o Parquet real no JupyterHub remoto quando disponível | y |
| Comparação de identidade de sequência entre pares confundidos em `02_analyze_confidence_and_calibration.ipynb` | Alinhamento global pareado sobre as sequências representativas coletadas do próprio dataset de teste (`use_ncbi_online=False` por padrão) | Evita dependência de rede externa, rate-limit e credenciais pessoais do NCBI (`Entrez.email`) durante a reprodução padrão e nos testes em sandbox, mantendo opção explícita para consulta online se desejado | y |

**Open questions:** none - all resolved or logged above.

## Observable

Worksheet, not the review. `n/a` needs its reason. One row may group the same decision across several routes.

| Surface | Decision | Landing |
| --- | --- | --- |
| API `compute_species_f1_bands` / `compute_species_specialist_comparison` | suporte retrocompatível a limiares parametrizados `(0.0, 0.50, 0.80, 0.90, 1.0)`, validação de limites e resolução de vitórias/empates por espécie | AC 1, AC 2, AC 3, AC 4 |
| API `profile_inference_model` / `evaluate_dataloader` | métricas de eficiência computacional e avaliação em lote sob $\epsilon \in \{0.0, 0.02, 0.05, 0.10, 0.15\}$ com verificação de taxa efetiva $3\epsilon/4$ | AC 5 |
| API `TaxonomyEntropyAnalyzer` | registro/remoção de hook em `fc`, validação de `species_to_genus_map`, cálculo de $H_{\text{species}}$, $H_{\text{genus}}$, $\Delta H$ e $H_{\text{activation}}$ normalizada | AC 6, AC 7, AC 8, AC 9 |
| Notebook `01_benchmark_clean_and_noise.ipynb` | formato das saídas CSV/JSON, cobertura dos 7 modelos registrados, modo `SMOKE_TEST=1` e manifesto de execução | AC 10, AC 11 |
| Notebook `02_analyze_confidence_and_calibration.ipynb` e `compare_confused_species_identity` | alinhamento offline-safe de pares confundidos, atribuição GradCAM/ScoreCAM sem padding, entropia hierárquica e calibração formal (ECE/MCE/Brier/NLL/TemperatureScaler) | AC 12, AC 13, AC 14, AC 15 |
| Todos os módulos e notebooks de `N02` | autenticação HTTP, rate limits e estados de tela web | n/a - pipeline científico local/JupyterHub sem servidor web ou endpoints HTTP expostos |

## Sources

- `ROADMAP_REPRODUCAO_NOTEBOOKS.md` (Spec N02, linhas 64–85) - especifica o escopo exato dos módulos e dos notebooks `01_benchmark_clean_and_noise.ipynb` e `02_analyze_confidence_and_calibration.ipynb`
- `short/benchmark.ipynb`, `short/finetune_resnet50_species.ipynb`, `short/finetune_convnext_species.ipynb`, `short/analize_convnext.ipynb`, `short/full_loss_function.ipynb` (células 37–49) e `long/cnn-long-voter-compare.ipynb` (células 14–33) - implementações históricas conciliadas
- `AGENTS.md` e `.specs/STATE.md` (`AD-004`, `AD-005`, `AD-006`, `AD-008`, `AD-009`, `AD-013`, `AD-015`) - invariantes científicos de ruído $3\epsilon/4$, faixas contínuas de F1, calibração probabilística formal e preservação dos notebooks originais

