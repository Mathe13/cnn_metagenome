# r03-real-calibration-and-hierarchical-entropy — Plan

## Problem

Na auditoria do notebook de reprodução `notebooks/reproduction/short/02_analyze_confidence_and_calibration.ipynb` e dos artefatos publicados em `results_reproduction/n02_benchmark/` e `results_reproduction/calibration/`:
1. **Identidade de pares confundidos calculada sobre sequências sintéticas fixas (Célula 4):** A Célula 4 ignora as amostras biológicas coletadas em `confused_samples` e passa `seq1 = "ACGT" * 128` e `seq2 = "ACGA" * 128` fixas para `compare_confused_species_identity`, gerando uma identidade fictícia desvinculada dos fragmentos taxonômicos sob análise.
2. **Colapso da entropia preditiva de gênero e NaN na ativação pré-classificadora (Célula 6):** A Célula 6 define `species_to_genus = torch.zeros(13683, dtype=torch.long, device=ctx.device)` fora do modo smoke, colapsando todas as 13.683 classes de espécie em um único gênero fictício (`H_genus = 0.0000`, `Gap = H_species`), além de avaliar apenas 2 modelos em vez das 5 variantes de espécie e produzir `H_act: nan` em execuções sob AMP sem cast explícito em FP32.
3. **Auditoria de calibração executada sobre 16 strings sintéticas fixas (Célula 7):** A Célula 7 executa `audit_model_calibration_by_noise` sobre 16 sequências repetidas `"ACGT" * 128` com rótulos artificiais `i % 5`, em vez de processar lotes reais de `test_loader` e calibrar pós-hoc na validação (`val_loader`).

Quando esta especificação for concluída, `src/viral_cnn/taxonomy/mapping.py` fornecerá `build_species_to_genus_tensor` para mapear 13.683 espécies aos respectivos gêneros taxonômicos sem degeneração, `src/viral_cnn/eval/calibration.py` fornecerá `audit_dataloader_calibration_by_noise` com streaming em lotes e expansão quadrática de memória constante para o Brier score, e o notebook `02` utilizará sequências reais em pares confundidos, avaliará as 5 variantes de espécies com `H_genus > 0` e calculará calibração estrita sobre os DataLoaders reais sem recair em dados sintéticos quando `is_smoke_test == False`.

## Flow

Reutiliza `build_registered_model`, `TaxonomyEntropyAnalyzer`, `compare_confused_species_identity`, `get_confused_class_samples` e `write_notebook_manifest` existentes em `src/viral_cnn/`.

1. `src/viral_cnn/taxonomy/mapping.py` (existing) -> implementa `build_species_to_genus_tensor` (door 1) para carregar o mapeamento `model_label (0..num_species-1) -> genus_idx (0..N_genera-1)` a partir de Parquet taxonômico, validando dimensionalidade e rejeitando mapeamento degenerado.
2. `src/viral_cnn/eval/calibration.py` (existing) -> implementa `audit_dataloader_calibration_by_noise` (door 2) para avaliar ECE, MCE, NLL, Brier score via expansão quadrática, e temperatura ótima sobre `test_loader` e `val_loader`.
3. `src/viral_cnn/eval/confusion.py` (existing) -> assegura que `get_confused_class_samples` (door 3) colete instâncias tanto de `true_id` quanto de `pred_id` para os pares da Célula 4.
4. `notebooks/reproduction/short/02_analyze_confidence_and_calibration.ipynb` (existing) -> Célula 4 calcula identidade de sequências reais decodificadas de `confused_samples`, Célula 6 carrega `species_to_genus` real e avalia as 5 variantes com `H_genus > 0`, e Célula 7 executa calibração via `audit_dataloader_calibration_by_noise` imprimindo tabelas completas.
5. out: `results_reproduction/` com `confused_pairs_identity.csv`, `table7_hierarchical_entropy.csv` e `calibration_audit_noise.csv` contendo dados empíricos válidos e auditáveis.

## Impact

| Front | What changes |
| --- | --- |
| domain | `mapping.py` ganha `build_species_to_genus_tensor`; `calibration.py` ganha `audit_dataloader_calibration_by_noise` para streaming de lotes reais |
| confusion evaluation | `confusion.py` garante coleta pareada de sequências reais para classes preditas e verdadeiras em `get_confused_class_samples` |
| notebook runtime | `02_analyze_confidence_and_calibration.ipynb` elimina strings fixas `"ACGT" * 128` e `torch.zeros(13683)` fora do modo smoke, imprimindo `df_identity`, `df_entropy` e `df_cal` completos |

## Relations

None - no stored-data shape change

## Surface

None - nothing consumed outside

## Landing

| One-way door | Literal shape | Alternative rejected |
| --- | --- | --- |
| Door 1: Extração e validação do tensor species-to-genus a partir de Parquet taxonômico | `build_species_to_genus_tensor(taxonomy_parquet_path: str \| Path, num_species: int = 13683, device: str \| torch.device = "cpu") -> torch.Tensor` gerando mapeamento $0 \dots N_{\text{genera}}-1$ e disparando `ValueError` se `len(tensor) != num_species` ou se todos os índices colapsarem em uma única classe | Vetor estático `torch.zeros(13683)` no notebook ou dicionário serializado hardcoded que perde sincronia com a taxonomia real |
| Door 2: Streaming de calibração multiclasse e ruído sobre DataLoader com expansão quadrática | `audit_dataloader_calibration_by_noise(models_dict: Dict[str, torch.nn.Module], test_loader: DataLoader, val_loader: Optional[DataLoader] = None, noise_rates: List[float] = [0.0, 0.02, 0.05, 0.10, 0.15], device: str \| torch.device = "cpu", n_bins: int = 15, max_batches: Optional[int] = None, seed: int = 42) -> Dict[str, Any]` | Coletar matrizes de probabilidades $(N, C)$ em RAM para $1{,}7\times 10^6$ amostras (estouro de memória) ou usar 16 sequências sintéticas |
| Door 3: Extração de sequências nucleotídicas reais para pares confundidos | Coletar tensores em `confused_samples` para classes verdadeiras e preditas, decodificar via `_decode_to_dna_string` e alinhar via `compare_confused_species_identity` | Injetar strings hardcoded `"ACGT" * 128` e `"ACGA" * 128` |

- Nothing else in this change is hard to reverse

## Criteria

### S1: Mapeamento taxonômico hierárquico e validação de degeneração (P1)

Garante construção correta do tensor `species_to_genus` sem colapso de classes e integridade da entropia hierárquica.

**Acceptance Criteria**

1. WHEN `build_species_to_genus_tensor` is invoked with a valid taxonomy parquet containing species and genus annotations THEN the function SHALL return a `torch.LongTensor` of length equal to `num_species` with distinct genus indices in $[0, N_{\text{genera}}-1]$ where unique genus count is $> 1$.
2. WHEN `build_species_to_genus_tensor` receives a degraded mapping where all species map to a single genus index THEN the function SHALL raise `ValueError`.
3. WHEN `TaxonomyEntropyAnalyzer` is executed on batch predictions with a non-degenerate `species_to_genus` mapping THEN the analyzer SHALL produce $H_{\text{genus}} > 0$, $H_{\text{species}} \ge H_{\text{genus}}$, and finite $H_{\text{act}}$ values with zero NaN entries under FP16 and FP32.

**Independent test:** Executar testes unitários em `tests/test_n02_benchmark_and_calibration.py` validando `build_species_to_genus_tensor`, detecção de mapeamento degenerado e cálculo de entropia com tensores multiespécie.

### S2: Auditoria de calibração por ruído sobre DataLoaders reais (P1)

Garante cálculo de ECE, MCE, NLL e Brier score streaming em lotes reais sem recair em strings sintéticas.

**Acceptance Criteria**

1. WHEN `audit_dataloader_calibration_by_noise` is executed on a `DataLoader` across noise rates $\epsilon \in \{0.0, 0.05\}$ THEN the function SHALL compute Top-1 accuracy, Top-1 confidence, ECE, MCE, NLL, and Brier score directly from streamed batches without instantiating full $(N, C)$ probability arrays in memory.
2. WHEN `val_loader` is supplied to `audit_dataloader_calibration_by_noise` THEN the function SHALL fit optimal temperature scaling $T^*$ exclusively on the validation loader and report calibrated metrics.
3. WHEN `is_smoke_test == False` in `notebooks/reproduction/short/02_analyze_confidence_and_calibration.ipynb` THEN Cells 4, 6, and 7 SHALL not use `"ACGT" * 128` or `torch.zeros(13683)` and SHALL print full dataframes `df_identity`, `df_entropy`, and `df_cal` in cell output.

**Independent test:** Executar testes unitários e estáticos em `tests/test_n02_benchmark_and_calibration.py` validando streaming de calibração, supressão de strings sintéticas e impressão de tabelas.

## Out of scope

| Excluded | Why |
| --- | --- |
| Execução em GPU do dataset completo de 1,7M amostras | Restrito ao ambiente do cluster sob `REMOTE_SERVER=true` conforme AD-006 e AD-017 |
| Modificação de notebooks legados em `short/` | Preservação intocada de notebooks históricos conforme `AGENTS.md` e `AD-004` |
| Re-treinamento de checkpoints de classificação | O escopo desta especificação é de avaliação diagnóstica e calibração, não treino de modelos |

## Assumptions

| Assumption | Chosen default | Rationale | Confirmed? |
| --- | --- | --- | --- |
| Resolução da coluna de gênero no Parquet | Suportar colunas `Genus`, `genus`, `genus_id`, ou `genus_idx` convertendo para códigos categóricos densos | O Parquet taxonômico produzido por diferentes pipelines usa `Genus` ou `genus_id` | y |
| Tratamento de espécies sem gênero no Parquet taxonômico | Agrupar espécies sem anotação de gênero (`NaN` / vazio / `None`) em uma categoria dedicada (ex: código 0 ou último código) | Garante conservação das 13.683 classes de espécie sem descarte de amostras | y |
| Sequências para comparação de espécies confundidas | Extrair os tensores reais decodificados via `_decode_to_dna_string` a partir de `confused_samples` | Permite comparar sequências biológicas reais observadas nos lotes de teste | y |

**Open questions:** none - all resolved or logged above.

## Observable

None - no user-facing surface

## Sources

- `ROADMAP_CORRECAO_AUDITORIA.md` (Seção 3, `R03 — r03-real-calibration-and-hierarchical-entropy`) - define as regras de calibração real, mapeamento de espécies para gêneros e eliminação de strings fixas.
- `src/viral_cnn/eval/calibration.py` e `src/viral_cnn/taxonomy/mapping.py` - implementação de métricas de calibração, entropia preditiva e mapeamentos taxonômicos.
- `notebooks/reproduction/short/02_analyze_confidence_and_calibration.ipynb` - pipeline de confusão de pares, GradCAM, entropia hierárquica e calibração.
