# r05-real-motif-jaccard-and-triplet-analysis — Plan

## Problem

Na auditoria dos notebooks de reprodução `notebooks/reproduction/short/07_first_layer_motifs_and_perturbation.ipynb` e `08_triplet_composition_and_cooccurrence.ipynb`, no módulo `src/viral_cnn/interpretability/motifs.py` e nos artefatos em `results_reproduction/motifs/`:
1. **Matriz de presença de motivos hardcoded no Notebook `07` (Célula 9):** A Célula 9 constrói uma matriz sintética fixa `presence = np.zeros((40, len(motifs_list)))` com colunas manuais idênticas (`presence[:, 0] = 1.0`, `presence[:20, 1] = 1.0`, `presence[:15, 2] = 1.0`) para todas as 5 arquiteturas, em vez de escanear os fragmentos de sequências biológicas reais fornecidos pelo DataLoader.
2. **Matrizes de Jaccard degeneradas e correlações espúrias (Notebooks `07` e `08`):** Devido à matriz sintética idêntica fabricada na Célula 9 do notebook `07`, as matrizes salvas em `motif_interaction_jaccard_*.parquet` tornaram-se degeneradas, produzindo `r_canonical = 0.0`, `p_permutation_null = 1.0` na Tabela 11 e correlação inter-arquitetura espúria perfeita de $r = 1.0$ entre modelos distintos na Célula 9 do notebook `08`.
3. **Lista hardcoded de 7 motivos na auditoria de fronteiras de tradução no Notebook `08` (Célula 5):** A Célula 5 avalia uma lista estática manual de 7 strings (`sample_motifs = ["ATGAAAA", "TAACCCC", ...]`), desvinculada dos motivos de 1ª camada realmente aprendidos e validados por perturbação no notebook `07`.
4. **Perturbação restrita no Notebook `07` (Célula 7):** A avaliação de perturbação com controle por GC é executada apenas para `resnet50_species`, sem abranger as 5 variantes de espécies avaliadas no artigo.

Quando esta especificação for concluída, `motifs.py` fornecerá `compute_motif_presence_matrix_from_loader`, o notebook `07` gerará matrizes de presença e de Jaccard empiricamente genuínas a partir de sequências de teste reais para todas as 5 variantes de espécies (salvando `motif_summary.parquet`), e o notebook `08` consumirá `motif_summary.parquet` na Tabela 10 e calculará correlações inter-arquitetura não-degeneradas na Tabela 11.

## Flow

Reutiliza `extract_first_layer_motifs`, `evaluate_motif_perturbation_with_gc_control`, `compute_jaccard_cooccurrence`, `analyze_cooccurrence_pairs`, `calculate_exploratory_rscu` e `audit_translation_boundary_signals` existentes em `src/viral_cnn/interpretability/motifs.py`.

1. `src/viral_cnn/interpretability/motifs.py` (existing) -> implementa `compute_motif_presence_matrix_from_loader` (door 1) para escanear lotes do `DataLoader`, decodificar tokens nucleotídicos em strings A/C/G/T e computar a presença binária de cada 7-mer da lista de motivos.
2. `notebooks/reproduction/short/07_first_layer_motifs_and_perturbation.ipynb` (existing) -> Célula 7 avalia perturbação para todas as 5 variantes de espécie salvando `table_08_motif_perturbation.csv` e `motif_summary.parquet` (door 2), e Célula 9 calcula a matriz de presença real via `compute_motif_presence_matrix_from_loader`, exportando matrizes de Jaccard empiricamente válidas.
3. `notebooks/reproduction/short/08_triplet_composition_and_cooccurrence.ipynb` (existing) -> Célula 3 computa correlações de RSCU (Tabela 9), Célula 5 carrega os motivos reais de `motif_summary.parquet` com fail-fast fora do smoke (door 3) para auditar sinais de início/término (Tabela 10), e Células 7 e 9 reexecutam a simetria de coocorrência (Tabela 11) e correlações inter-arquitetura.
4. out: `results_reproduction/motifs/` com `table_08_motif_perturbation.csv`, `motif_summary.parquet`, `motif_interaction_jaccard_*.parquet`, `table_09_exploratory_rscu.csv`, `table_10_translation_boundary_signals.csv`, `table_11_cooccurrence_synergy.csv` e Figuras 3, 4 e 5.

## Impact

| Front | What changes |
| --- | --- |
| motif presence extraction | `src/viral_cnn/interpretability/motifs.py` ganha `compute_motif_presence_matrix_from_loader` |
| first layer motifs notebook | `07_first_layer_motifs_and_perturbation.ipynb` elimina presença sintética fixa, avalia as 5 variantes e gera matrizes de Jaccard reais |
| triplet and cooccurrence notebook | `08_triplet_composition_and_cooccurrence.ipynb` consome `motif_summary.parquet` real na Tabela 10 e analisa correlações inter-modelo sobre dados genuínos |

## Relations

None - no stored-data shape change

## Surface

None - nothing consumed outside

## Landing

| One-way door | Literal shape | Alternative rejected |
| --- | --- | --- |
| Door 1: Extração da matriz de presença de motivos a partir de DataLoader | `compute_motif_presence_matrix_from_loader(loader: Any, motifs_list: List[str], max_sequences: int = 10000, device: str \| torch.device = "cpu") -> np.ndarray` retornando matriz binária $(N, M)$ baseada na ocorrência real dos 7-mers | Matriz estática fabricada `presence[:, 0] = 1.0` ou carregar genomas inteiros desestruturados em RAM |
| Door 2: Avaliação de perturbação e extração de motivos para as 5 variantes de espécies no Notebook 07 | Avaliar perturbação em loop sobre as 5 variantes (`resnet50_species`, `resnet_noise_species`, `convnext_zero_species`, `convnext_transfer_species`, `convnext_noise_species`), gerando `table_08_motif_perturbation.csv` e `motif_summary.parquet` | Avaliar apenas `resnet50_species` e ignorar as 4 variantes restantes |
| Door 3: Encadeamento de motivos reais do Notebook 07 no Notebook 08 | Na Célula 5 do Notebook 08, ler `results_reproduction/motifs/motif_summary.parquet`; se `not ctx.is_smoke_test` e o arquivo não existir, levantar `FileNotFoundError` imediatamente | Hardcodar lista de 7 strings estáticas ou usar fallback silencioso desvinculado dos filtros aprendidos |

- Nothing else in this change is hard to reverse

## Criteria

### S1: Extração de presença de motivos e cálculo genuíno de Jaccard (P1)

Garante escaneamento exato de 7-mers nas sequências do DataLoader e produção de matrizes de Jaccard não-degeneradas.

**Acceptance Criteria**

1. WHEN `compute_motif_presence_matrix_from_loader` is executed with a sequence DataLoader and list of $M$ motifs THEN the function SHALL scan up to `max_sequences` sequences and return a binary NumPy matrix of shape $(N, M)$ where `matrix[i, j] == 1.0` if and only if motif $j$ occurs as a substring in sequence $i$.
2. WHEN `compute_jaccard_cooccurrence` is executed on empirical presence matrices THEN the function SHALL return a symmetric Jaccard DataFrame with 1.0 on the diagonal and non-constant co-occurrence values in $[0.0, 1.0]$.
3. WHEN `is_smoke_test == False` in `notebooks/reproduction/short/07_first_layer_motifs_and_perturbation.ipynb` THEN the notebook SHALL evaluate motif perturbation across all 5 species model variants and SHALL save `table_08_motif_perturbation.csv` and `motif_summary.parquet`.

**Independent test:** Executar testes unitários em `tests/test_motifs.py` e `tests/test_n05_motifs_and_triplets.py` validando `compute_motif_presence_matrix_from_loader` e simetria de Jaccard com dados empíricos.

### S2: Auditoria de fronteiras de tradução e correlações inter-arquitetura no Notebook 08 (P1)

Garante encadeamento dos motivos aprendidos reais na Tabela 10 e cálculo de correlações inter-modelo genuínas.

**Acceptance Criteria**

1. WHEN `is_smoke_test == False` in `notebooks/reproduction/short/08_triplet_composition_and_cooccurrence.ipynb` THEN Cell 5 SHALL load learned motifs from `results_reproduction/motifs/motif_summary.parquet` without using hardcoded `sample_motifs` and SHALL raise `FileNotFoundError` if the artifact is missing on disk.
2. WHEN Table 10 is generated by notebook `08` THEN the output DataFrame SHALL report initiation (`start_atg`) and termination (`stop_taa`, `stop_tag`, `stop_tga`) boundary signals computed on the real extracted motifs.
3. WHEN Cell 9 of notebook `08` evaluates inter-architecture co-occurrence correlations THEN the correlations SHALL be computed across distinct pairs of species variants without degenerate $r = 1.0$ identity across distinct networks.
4. WHEN executed in smoke test mode (`is_smoke_test == True`) THEN notebook `07` and notebook `08` SHALL execute end-to-end within 60 seconds without errors.

**Independent test:** Executar testes automatizados em `tests/test_n05_motifs_and_triplets.py` verificando carregamento de motivos, geração da Tabela 10 e execução rápida em smoke test.

## Out of scope

| Excluded | Why |
| --- | --- |
| Retreinamento de modelos de classificação | O escopo restringe-se à interpretação e análise pós-hoc de motivos e tripletos dos checkpoints existentes |
| Modificação de notebooks históricos em `short/` | Preservação estrita dos notebooks legados conforme `AGENTS.md` e `AD-004` |
| Anotação biológica experimental de CDS | A análise de tripletos é puramente exploratória sem verificação de fase biológica conforme AD-014 |

## Assumptions

| Assumption | Chosen default | Rationale | Confirmed? |
| --- | --- | --- | --- |
| Decodificação de tensores nucleotídicos | Mapeamento `0 -> A, 1 -> C, 2 -> G, 3 -> T` | Padrão unificado de codificação espacial de 4 canais estabelecido em AD-003 | y |
| Limite de sequências para escaneamento de motivos | Processar até 10.000 fragmentos fora do smoke mode (40 sob smoke mode) | Fornece representatividade estatística robusta sem gerar lentidão excessiva de I/O | y |
| Formato de `motif_summary.parquet` | Armazenar tabela com colunas `model_name`, `filter_idx`, `motif` e métricas de perturbação | Permite consumo direto pela análise de tripletos e fronteiras de tradução do notebook 08 | y |

**Open questions:** none - all resolved or logged above.

## Observable

None - no user-facing surface

## Sources

- `ROADMAP_CORRECAO_AUDITORIA.md` (Seção 3, `R05 — r05-real-motif-jaccard-and-triplet-analysis`) - define as correções para a matriz de presença, encadeamento de `motif_summary.parquet` e correlações de coocorrência.
- `src/viral_cnn/interpretability/motifs.py` - implementação de extração de motivos, perturbação e coocorrência de Jaccard.
- `notebooks/reproduction/short/07_first_layer_motifs_and_perturbation.ipynb` e `08_triplet_composition_and_cooccurrence.ipynb` - pipelines de motivos de 1ª camada e composição de tripletos.

