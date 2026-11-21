# Plan: Partial Taxonomy Derivation, Conditional Genus Entropy, and Rank Transition Coverage

## Problem

In the existing codebase, three methodological risks affect taxonomic hierarchy processing and reporting:

1. In `m01_workflow.py`, `mode: curate` requires the raw ICTV Excel workbook (`MSL41.v1.xlsx`) and raises `FileNotFoundError` if absent, blocking environments where an existing curated Parquet is present. `mode: audit_existing` audits the Parquet read-only but cannot materialize a cleansed partial taxonomy copy with verified `model_label = id - 1` and normalized null ranks.
2. In `build_species_to_genus_tensor()`, missing Genus values are converted to `"Unknown_Genus"`. This assigns the 5,937 species without genus assignment to a single artificial category in Table 7, distorting genus entropy $H_{\text{genus}}$ and causing mismatched probabilistic supports between unconditional species entropy (13,683 classes) and genus-aggregated distributions.
3. In Notebook 06, Table 6 (`table6_taxonomic_rank_correlations.csv`) reports correlations and $N_{\text{taxa}}$, but lacks explicit counts of eligible and excluded species classes (7,746 eligible for Genus, 4,583 eligible for Order) and evaluated test fragments, obscuring the empirical support of each transition.

## Flow

Reuses existing `m01_workflow.py` runner, `TaxonomyEntropyAnalyzer` hook lifecycle, and Notebook 06 correlation pipelines.

1. `config` with `mode: derive_partial_existing` -> `viral_cnn.m01_workflow:_curate` (exists) - reads `taxonomy_path` and `original_mapping`, validates IDs and `model_label = id - 1`, masks placeholders in Genus and Order to `NA`, validates identity, and saves `main_taxonomy_hybrid_PARTIAL.parquet` with `hierarchy_ready=False` (door 1)
2. `main_taxonomy_hybrid_PARTIAL.parquet` -> `viral_cnn.taxonomy.mapping:build_partial_species_to_genus_tensor` (new, door 2) - produces a tensor where species with known genus map to contiguous `[0 .. N_known_genera - 1]` and unassigned species map to `-1`
3. `model` + partial mapping tensor + DataLoader -> `viral_cnn.eval.calibration:TaxonomyEntropyAnalyzer` (exists) - computes unconditional $H_{\text{species}}$, probability mass of excluded classes, and conditioned metrics $H_{\text{species\_conditional}}$, $H_{\text{genus\_conditional}}$, $\Delta H_{\text{conditional}}$ using `softmax(logits[:, known_mask])`, tracking `N_total`, `N_truth_with_genus`, `N_truth_without_genus`
4. test predictions + taxonomy -> `notebooks/reproduction/short/06_ictv_latent_space_and_hierarchy.ipynb` (exists) - exports Table 6 enriched with eligible/excluded class and fragment counts (door 3)

## Impact

| Front | What changes |
| --- | --- |
| domain | new mode: `derive_partial_existing` - derives and validates existing hybrid Parquet into `main_taxonomy_hybrid_PARTIAL.parquet` without ICTV workbook, lives in `viral_cnn.m01_workflow` |
| domain | new term: `build_partial_species_to_genus_tensor` - maps known genera to contiguous indices and unassigned genera to `-1`, lives in `viral_cnn.taxonomy.mapping` |
| domain | new term: conditional entropy gap - $\Delta H_{\text{conditional}} = H_{\text{species\_conditional}} - H_{\text{genus\_conditional}}$ over identical supports, lives in `viral_cnn.eval.calibration` |
| stored data | nothing to migrate; outputs written to isolated run namespaces with new `run_id` |

## Relations

`None - no stored-data shape change`

## Surface

| Route | In | Out | Status |
| --- | --- | --- | --- |
| `run_m01(notebook_id="short/05", ...)` with `mode: derive_partial_existing` | `original_mapping`, `taxonomy_path` | `manifest.json`, `main_taxonomy_hybrid_PARTIAL.parquet`, `curation_audit.json` | 200, 400, 500 |
| `build_partial_species_to_genus_tensor(...)` | `taxonomy_parquet_path`, `num_species=13683`, `device="cpu"` | `torch.Tensor` of shape `(num_species,)` with values in `[-1, N_known_genera - 1]` | 200, 400, 404 |
| `TaxonomyEntropyAnalyzer.analyze_batch(inputs, labels=None)` | `inputs` tensor `(B, 4, 512)`, optional `labels` tensor `(B,)` | dict with `h_species`, `h_species_conditional`, `h_genus_conditional`, `gap_conditional`, `excluded_prob_mass`, `h_activation` | 200, 400, 500 |
| `TaxonomyEntropyAnalyzer.analyze_dataloader(dataloader, ...)` | `dataloader` yielding `(inputs, labels, ...)` | dict with mean and max metrics, `n_total`, `n_truth_with_genus`, `n_truth_without_genus` | 200, 400, 500 |

## Landing

| One-way door | Literal shape | Alternative rejected |
| --- | --- | --- |
| Mode `derive_partial_existing` in M01 | `mode: "derive_partial_existing"`, protocol `m01_existing_partial_derive_v1`, output `main_taxonomy_hybrid_PARTIAL.parquet`, manifest `hierarchy_ready=False` | Altering `curate` mode to ignore missing ICTV workbook, which would conflate MSL curation with existing Parquet derivation |
| Partial genus mapping with `-1` | `build_partial_species_to_genus_tensor` returns `torch.long` with `-1` for unknown ranks and contiguous indices `[0, N_known - 1]` for known | Mutating existing `build_species_to_genus_tensor` to assign `-1`, which would break existing strict invariants expecting valid non-negative indices |
| Conditional entropy evaluation | Renormalized conditional probabilities `F.softmax(logits[:, known_mask], dim=-1)` over species with known genus | Subtracting $H_{\text{genus\_conditional}}$ from unconditional $H_{\text{species}}$, which would compare mismatched probability spaces |
| Table 6 transition counts | Columns `Eligible_Species`, `Excluded_Species`, `Eligible_Fragments`, `Excluded_Fragments` added to `table6_taxonomic_rank_correlations.csv` | Overwriting or renaming `N_taxa`, which would break publication adapters registered in `viral_cnn.reporting` |

- Nothing else in this change is hard to reverse

## Criteria

### S1: M01 Derive Partial Existing Mode (P1)

Derives and validates a partial taxonomy Parquet from an existing taxonomy source without ICTV Excel workbook.

**Acceptance Criteria**

1. WHEN `run_m01` executes with `notebook_id="short/05"` and `mode="derive_partial_existing"` THEN the system SHALL load `original_mapping` and `taxonomy_path` without reading `ictv_master`.
2. IF `id` values in `taxonomy_path` are invalid, duplicated, or less than 1 THEN the system SHALL raise `ValueError` with `"IDs taxonômicos inválidos ou duplicados"`.
3. WHEN `model_label` is absent in `taxonomy_path` THEN the system SHALL add `model_label = id - 1` to the memory copy.
4. IF `model_label` is present in `taxonomy_path` but differs from `id - 1` THEN the system SHALL raise `ValueError` with `"model_label existente difere de id - 1"`.
5. The system SHALL mask empty, whitespace, `"Unassigned"`, and `"Unclassified"` values in `"Genus"` and `"Order"` columns to `pd.NA`.
6. WHEN `validate_taxonomy_identity` reports `mapping_integrity_ok=True` THEN the system SHALL export `main_taxonomy_hybrid_PARTIAL.parquet` to the reserved run directory.
7. The system SHALL emit `manifest.json` with `status="COMPLETE"`, `hierarchy_ready=False`, `protocol_id="m01_existing_partial_derive_v1"`, and SHA-256 hashes of both input files and derived outputs.
8. WHEN `preflight_notebook("short/05", ...)` is invoked under `M01_MODE="derive_partial_existing"` THEN the preflight SHALL require `short/hybrid_taxonomy_parquet` and `short/species_csv`, and SHALL NOT require `short/ictv_master`.

**Independent test:** Execute `run_m01` with synthetic CSV mapping and partial taxonomy Parquet containing missing Genus/Order, verifying generated Parquet and manifest.

### S2: Notebook 02 Conditional Genus Entropy (P1)

Computes conditional predictive entropy on known genera without grouping unassigned species into an artificial category.

**Acceptance Criteria**

9. WHEN `build_partial_species_to_genus_tensor` is called on a taxonomy Parquet with unassigned genera THEN the system SHALL assign contiguous indices `0 .. N_known_genera - 1` to known genera and `-1` to unassigned species.
10. The system SHALL NOT create an `"Unknown_Genus"` category in `build_partial_species_to_genus_tensor`.
11. WHEN `TaxonomyEntropyAnalyzer` receives a mapping tensor containing `-1` with `allow_partial=True` THEN the system SHALL accept the mapping without raising `ValueError`.
12. WHEN `analyze_batch` processes inputs under partial mapping THEN the system SHALL compute unconditional `h_species`, conditional `h_species_conditional = calculate_entropy(softmax(logits[:, known_mask]))`, conditional `h_genus_conditional`, and `gap_conditional = h_species_conditional - h_genus_conditional`.
13. WHEN `analyze_batch` processes inputs under partial mapping THEN the system SHALL compute `excluded_prob_mass` equal to the sum of softmax probabilities assigned to species with mapping `-1`.
14. WHEN `analyze_dataloader` processes batches containing target labels THEN the system SHALL record `n_total`, `n_truth_with_genus`, and `n_truth_without_genus` based on whether true species labels map to a known genus.
15. The system SHALL ensure that `gap_conditional >= -1e-5` for all batches under partial mapping.
16. WHEN `analyze_dataloader` aggregates conditional entropy metrics (`mean_h_species_conditional`, `mean_h_genus_conditional`, `mean_gap_conditional`, `mean_excluded_prob_mass`) with target labels present THEN the system SHALL average these metrics strictly over fragments whose true species has a known genus (`truth_with_genus`).
17. WHEN partial mapping is active THEN the system SHALL omit unnormalized `mean_h_genus` and `mean_hierarchical_gap` from the exported summary, preventing misinterpretation of unnormalized probability distributions.

**Independent test:** Run `TaxonomyEntropyAnalyzer` with a known model and synthetic partial mapping tensor, asserting conditional metrics and excluded probability mass.

### S3: Notebook 06 Table 6 Transition Coverage (P2)

Enriches Table 6 with explicit counts of eligible and excluded species classes and evaluated test fragments.

**Acceptance Criteria**

18. WHEN Notebook 06 generates `table6_taxonomic_rank_correlations.csv` THEN the system SHALL preserve all existing columns (`Model`, `Transition`, `Pearson_r`, `Spearman_rho`, `N_taxa`).
19. The system SHALL include `Eligible_Species` and `Excluded_Species` in `table6_taxonomic_rank_correlations.csv` reporting the exact number of species classes eligible and excluded for each transition (7,746 eligible for Species -> Genus, 4,583 eligible for Genus -> Order on standard reference). For the transition `Genus -> Order`, the system SHALL require both `Genus` AND `Order` to be known for a species to be eligible, excluding species where `Order` is known but `Genus` is missing.
20. The system SHALL include `Eligible_Fragments` and `Excluded_Fragments` in `table6_taxonomic_rank_correlations.csv` computed from the test labels evaluated for that specific model (`labels_cache[m_name]`), requiring both ranks for `Genus -> Order`.
21. WHEN `transform_table_6_hierarchy_correlations` processes the enriched Table 6 THEN the transformation SHALL preserve the new columns without raising `PublicationValidationError`.

**Independent test:** Execute Table 6 generation logic in Notebook 06 smoke mode and assert column presence and non-negative counts.

## Out of scope

| Excluded | Why |
| --- | --- |
| Modifying strict `curate` mode | `curate` represents raw reconstruction from MSL workbook and must retain its validation gate |
| Modifying `build_species_to_genus_tensor` | Strict existing consumers rely on non-negative mappings; partial logic is isolated in `build_partial_species_to_genus_tensor` |
| Overwriting or modifying historical input files | Preserves strict scientific provenance and immutability of inputs |
| Full cluster retraining of models | Local execution and validation verifies pipeline mechanics and contracts without faking cluster training |

## Assumptions

| Assumption | Chosen default | Rationale | Confirmed? |
| --- | --- | --- | --- |
| Protocol ID for partial derivation | `m01_existing_partial_derive_v1` | Follows repository naming conventions (`m01_existing_taxonomy_audit_v1`, `m01_ictv_curate_v1`) | y |
| Output Parquet filename for partial derivation | `main_taxonomy_hybrid_PARTIAL.parquet` | Distinguishes partial derivation from `main_taxonomy_hybrid_CLEANSED.parquet` | y |
| Behavior of `TaxonomyEntropyAnalyzer` when labels are absent in batch | `n_truth_with_genus` and `n_truth_without_genus` set to `None` | Backward compatible with dataloaders providing only inputs | y |
| Renormalization method for conditional species distribution | `F.softmax(logits[:, known_mask], dim=-1)` | Properly defines probability distribution over the known support without distortion | y |

**Open questions:** none - all resolved or logged above.

## Observable

| Surface | Decision | Landing |
| --- | --- | --- |
| CLI / Workflow `run_m01(..., mode="derive_partial_existing")` | invalid input rejection and error message | AC 2, AC 4 |
| Workflow output files | Parquet and manifest emission | AC 6, AC 7 |
| Runtime preflight `preflight_notebook("short/05")` | required prerequisite resolution | AC 8 |
| Function `build_partial_species_to_genus_tensor` | missing genus represented as `-1` | AC 9, AC 10 |
| Analyzer `TaxonomyEntropyAnalyzer` | conditional entropy calculation | AC 11, AC 12, AC 13, AC 14, AC 15 |
| Tabular output `table6_taxonomic_rank_correlations.csv` | enriched column schema | AC 16, AC 17, AC 18, AC 19 |

## Sources

- User request: `derive_partial_existing`, `build_species_to_genus_tensor()` partial fix, Notebook 06 Table 6 transition counts
- Repository invariants: `AGENTS.md` (scientific invariants, 13,683 Species, 0-based `model_label = id - 1`)
- Decisions: `AD-001`, `AD-002`, `AD-018`, `AD-029`
