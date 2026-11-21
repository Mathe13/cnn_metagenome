# S19 — Partial Taxonomy Derivation, Conditional Genus Entropy, and Rank Transition Coverage checks

Profile: light
Plan: `.specs/features/s19-partial-taxonomy-and-conditional-entropy/plan.md`

12 checks in 3 slices · 3 one-way doors · 0 open, of which 0 block

## Checks

Grouped by the plan's slices; numbering runs across the whole feature.

### S1 - M01 Derive Partial Existing Mode · 2 files · 25 KB · ~8k

**C1** - `run_m01` with `notebook_id="short/05"` and `mode="derive_partial_existing"` loads `original_mapping` and `taxonomy_path` preserving both files without reading `ictv_master` (AC 1)
Proof: `.venv/bin/python3 -m unittest -k test_c1_m01_derive_partial_loads_inputs tests/test_s19_partial_taxonomy_and_conditional_entropy.py`

**C2** - `run_m01` with `mode="derive_partial_existing"` raises `ValueError` with `"IDs taxonômicos inválidos ou duplicados"` when IDs in `taxonomy_path` are invalid, duplicated, or non-positive (AC 2)
Proof: `.venv/bin/python3 -m unittest -k test_c2_m01_derive_partial_validates_ids tests/test_s19_partial_taxonomy_and_conditional_entropy.py`

**C3** - `run_m01` with `mode="derive_partial_existing"` sets `model_label = id - 1` when absent and raises `ValueError` with `"model_label existente difere de id - 1"` when existing `model_label` diverges (AC 3, AC 4)
Proof: `.venv/bin/python3 -m unittest -k test_c3_m01_derive_partial_model_label tests/test_s19_partial_taxonomy_and_conditional_entropy.py`

**C4** - `run_m01` with `mode="derive_partial_existing"` masks whitespace, empty, and Unassigned/Unclassified strings in Genus and Order to `pd.NA` (AC 5)
Proof: `.venv/bin/python3 -m unittest -k test_c4_m01_derive_partial_masks_missing_ranks tests/test_s19_partial_taxonomy_and_conditional_entropy.py`

**C5** - `run_m01` with `mode="derive_partial_existing"` saves `main_taxonomy_hybrid_PARTIAL.parquet` and emits `manifest.json` with `status="COMPLETE"`, `hierarchy_ready=False`, and protocol `m01_existing_partial_derive_v1` (AC 6, AC 7)
Proof: `.venv/bin/python3 -m unittest -k test_c5_m01_derive_partial_manifest_and_parquet tests/test_s19_partial_taxonomy_and_conditional_entropy.py`

**C6** - `preflight_notebook("short/05", ...)` under `M01_MODE="derive_partial_existing"` requires `short/hybrid_taxonomy_parquet` and `short/species_csv`, and does not require `short/ictv_master` (AC 8)
Proof: `.venv/bin/python3 -m unittest -k test_c6_preflight_short_05_derive_mode tests/test_s19_partial_taxonomy_and_conditional_entropy.py`

### S2 - Notebook 02 Conditional Genus Entropy · 2 files · 30 KB · ~10k

**C7** - `build_partial_species_to_genus_tensor` produces a LongTensor where known genera map to contiguous indices `0 .. N_known_genera - 1` and missing genera map to `-1`, never creating `"Unknown_Genus"` (AC 9, AC 10)
Proof: `.venv/bin/python3 -m unittest -k test_c7_build_partial_species_to_genus_tensor tests/test_s19_partial_taxonomy_and_conditional_entropy.py`

**C8** - `TaxonomyEntropyAnalyzer` accepts a mapping tensor containing `-1` when `allow_partial=True` without raising `ValueError` (AC 11)
Proof: `.venv/bin/python3 -m unittest -k test_c8_analyzer_accepts_partial_mapping tests/test_s19_partial_taxonomy_and_conditional_entropy.py`

**C9** - `TaxonomyEntropyAnalyzer.analyze_batch` computes `h_species`, `h_species_conditional`, `h_genus_conditional`, `gap_conditional >= -1e-5`, and `excluded_prob_mass` (AC 12, AC 13, AC 15)
Proof: `.venv/bin/python3 -m unittest -k test_c9_analyzer_batch_conditional_entropy tests/test_s19_partial_taxonomy_and_conditional_entropy.py`

**C10** - `TaxonomyEntropyAnalyzer.analyze_dataloader` tallies `n_total`, `n_truth_with_genus`, `n_truth_without_genus` based on batch target labels, averages conditional metrics strictly over fragments with known genus, and omits unnormalized `mean_h_genus` and `mean_hierarchical_gap` under partial mapping (AC 14, AC 16, AC 17)
Proof: `.venv/bin/python3 -m unittest -k test_c10_analyzer_dataloader_truth_counts tests/test_s19_partial_taxonomy_and_conditional_entropy.py`

### S3 - Notebook 06 Table 6 Transition Coverage · 2 files · 20 KB · ~7k

**C11** - Notebook 06 Table 6 exports `table6_taxonomic_rank_correlations.csv` preserving existing columns, requiring both Genus and Order for Genus -> Order, and using model-specific labels for fragment counts (AC 18, AC 19, AC 20)
Proof: `.venv/bin/python3 -m unittest -k test_c11_table6_rank_transition_coverage tests/test_s19_partial_taxonomy_and_conditional_entropy.py`

**C12** - `transform_table_6_hierarchy_correlations` preserves the enriched Table 6 columns without `PublicationValidationError` (AC 21)
Proof: `.venv/bin/python3 -m unittest -k test_c12_transform_table_6_preserves_enriched_columns tests/test_s19_partial_taxonomy_and_conditional_entropy.py`

## Coverage

| Set (size) | Member -> proof | Unproven |
| --- | --- | --- |
| M01 derive partial existing execution (6) | inputs_preservados C1 · ids_validados C2 · model_label_ajustado C3 · ranks_mascarados C4 · manifesto_parquet_emitidos C5 · preflight_atualizado C6 | - |
| Entropia condicional e mapeamento parcial (4) | tensor_parcial_sem_unknown C7 · analyzer_aceita_parcial C8 · batch_entropia_condicional C9 · dataloader_contagem_verdadeiros C10 | - |
| Cobertura de transicoes da Tabela 6 (2) | tabela6_colunas_enriquecidas C11 · transformacao_preserva_colunas C12 | - |
| Surface route short/05 (3) | status_200 C5 · status_400 C2 · status_500 C1 | - |

## Swept

Where each unwritten requirement landed. All nine, one line each, every time.

- validation: C2, C3, C4, C7, C8, C12
- failure modes: C2, C3, C8, C12
- idempotency: existing - execuções idempotentes com run_id exclusivo sem sobrescrever fontes
- authorization: n/a - execução em sistema de arquivos local
- concurrency: n/a - execução sequencial monotarefa
- data lifecycle: existing - saídas isoladas em diretório exclusivo de run
- dependency failure: C1, C5, C6
- state transitions: C5 - transição para `status="COMPLETE"` e `hierarchy_ready=False`
- observability: existing - registro em log dos artefatos e manifestos emitidos

## Handoff

S1-S3 = ~25k tokens acumulados (`m01_workflow.py`, `mapping.py`, `calibration.py`, `notebook_runtime.py`, notebooks 02, 05, 06 e suite de testes), bem abaixo do orçamento de 150k - one builder.

- Boundary: C1-C12 implementados e validados em conjunto por um único builder.
