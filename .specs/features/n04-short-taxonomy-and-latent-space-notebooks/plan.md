# n04-short-taxonomy-and-latent-space-notebooks

## Problem

In the original study, viral taxonomy harmonization and latent space representation evaluation were conducted in monolithic notebooks (`short/ICTV_improve_kraken_database.ipynb`, `short/improve_kraken_database-Copy1.ipynb`, and `short/ICTV_latent_space-bkp.ipynb`). This original setup presents several methodological and reproducibility limitations:

1. Coupling to legacy paths and scripts: the historical notebooks rely on relative paths (`./main_taxonomy_hybrid_CLEANSED.parquet`, `../dataset_kraken/`), direct interactive network calls to NCBI Entrez via personal credentials without caching or synthetic offline fallbacks, and coupled legacy modules (`short/utils/dna_dataset_2.py`, `short/utils/models.py`).
2. Absence of modular taxonomy curation: the logic for building ICTV rename dictionaries, performing exact and fuzzy taxonomic string matching, resolving lineage ranks (Realm, Kingdom, Phylum, Class, Order, Family, Genus), and backfilling missing entries lives strictly inside unstructured notebook cells rather than reusable, tested functions in `src/viral_cnn/taxonomy/`.
3. Incomplete integration of provenance audits: while S02 and S03 established canonical hashing and genome-disjoint split feasibility (`viral_cnn.data.audit` and `viral_cnn.data.grouped_split`), these audits were never integrated into a unified taxonomic curation and provenance notebook.
4. Conflation in latent space interpretation: historical analysis in `short/ICTV_latent_space-bkp.ipynb` evaluated 5-NN accuracy transductively on test embeddings (80/20 test split with test labels for reference), conflated 2D UMAP qualitative projections with quantitative separation, evaluated sample-limited Silhouette scores without class-inclusion reporting, omitted Pearson correlation coefficients and sample counts $N$ in Table 6, and postulated a "classifier head bottleneck" in ConvNeXt without training an independent linear probe on train/val representations (Notas 24–25, 35, 38; AD-012).
5. Lack of decoupled reproduction notebooks: there are currently no clean, modular reproduction notebooks in `notebooks/reproduction/short/` reproducing the ICTV curation pipeline and the latent space / hierarchy evaluation.

When this feature ships, researchers will have a modular taxonomy curation engine in `src/viral_cnn/taxonomy/curation.py`, an updated runtime topology supporting ICTV artifacts, and two fully reproducible notebooks (`05_ictv_taxonomy_curation.ipynb` and `06_ictv_latent_space_and_hierarchy.ipynb`) that execute end-to-end under remote JupyterHub environments or locally under `SMOKE_TEST=1` with synthetic data, directly reproducing Tables 5 and 6 and Figure 2 with full scientific rigor.

## Flow

This feature reuses taxonomic mapping helpers from `viral_cnn.taxonomy.mapping`, dataset provenance and split audit engines from `viral_cnn.data.audit` and `viral_cnn.data.grouped_split`, latent space metrics, linear probing, and UMAP export from `viral_cnn.eval.latent`, model registry from `viral_cnn.models.registry`, and environment management from `viral_cnn.notebook_runtime`.

1. Raw taxonomic data & ICTV sheets -> `viral_cnn.taxonomy.curation:build_hybrid_taxonomy` (new, door 1) - merges species names, parses ICTV rename rules, executes exact and fuzzy matching, resolves ranks, and backfills lineage.
2. Curation notebook -> `05_ictv_taxonomy_curation.ipynb` (new, door 2) - executes hybrid curation, integrates dataset duplicate audit (`audit_partition_duplicates`), and evaluates genome-disjoint split feasibility (`build_genome_disjoint_split`).
3. Embeddings extraction -> `viral_cnn.eval.latent:extract_latent_embeddings` (exists) - captures pre-classifier features (2048-d ResNet50, 768-d ConvNeXt) across species models.
4. Latent clustering & KNN -> `viral_cnn.eval.latent:evaluate_cosine_knn` and `evaluate_silhouette_with_audit` (exists) - runs stratified 80/20 cosine KNN ($k=5$, zero self-neighbor) and high-dimensional Silhouette with class-inclusion reporting.
5. Hierarchical aggregation -> `viral_cnn.eval.latent:calculate_taxonomic_correlations` (exists) - computes Species -> Genus -> Order correlations (Pearson $r$ and Spearman $\rho$, two-tailed $p$-values, explicit sample count $N$).
6. Head bottleneck audit -> `viral_cnn.eval.latent:audit_classifier_bottleneck` (exists) - trains an independent linear probe on train/val embeddings and evaluates on test embeddings to test the bottleneck hypothesis without test leakage.
7. Latent notebook -> `06_ictv_latent_space_and_hierarchy.ipynb` (new, door 3) - coordinates latent space extraction, metrics calculation, qualitative UMAP 2D projection (`export_umap_projection`), and manifests export for Tables 5, 6, and Figure 2.

## Impact

| Front | What changes |
| --- | --- |
| domain | new term: `HybridTaxonomyTable` - consolidated, cleansed taxonomic DataFrame aligning 13,683 Species classes with ICTV MSL41 and NCBI lineages, lives in `viral_cnn.taxonomy.curation` |
| domain | existing term: `calculate_taxonomic_correlations` in `viral_cnn.eval.latent` - now directly consumed by reproduction notebook `06` to generate Table 6 reconciling ConvNeXt_T without arbitrary genus truncation |
| stored data | nothing to migrate - historical parquet and excel files remain untouched in remote storage; new cleansed taxonomy and latent metric reports are saved to `results_reproduction/taxonomy/` and `results_reproduction/latent_space/` |

## Relations

None - no stored-data shape change

## Surface

None - nothing consumed outside (internal Python modules and reproduction notebooks only)

## Landing

| One-way door | Literal shape | Alternative rejected |
| --- | --- | --- |
| Taxonomy curation module location | `src/viral_cnn/taxonomy/curation.py` with exports `build_ictv_rename_map`, `match_ictv_taxonomy`, `resolve_local_ictv_taxonomy`, `build_hybrid_taxonomy`, `sanitize_hybrid_taxonomy`, and `generate_synthetic_taxonomy` | Implementing curation logic as inline functions inside notebook cells; rejected because inline logic prevents unit testing, automated smoke testing, and reuse by downstream modules. |
| Offline / smoke-test curation fallback | `generate_synthetic_taxonomy(num_classes: int = 13683) -> pd.DataFrame` providing deterministic mock taxonomy matching ICTV column schema when `ICTV_Master_Species_List_2025_MSL41.v1.xlsx` or Entrez API is unavailable | Failing execution when remote Excel or NCBI API is absent; rejected because local tests and CI must execute offline deterministically in seconds per AD-005 and AD-017. |
| Decoupled reproduction notebooks | `notebooks/reproduction/short/05_ictv_taxonomy_curation.ipynb` and `notebooks/reproduction/short/06_ictv_latent_space_and_hierarchy.ipynb` | Modifying historical notebooks `short/ICTV_improve_kraken_database.ipynb` or `short/ICTV_latent_space-bkp.ipynb`; rejected because historical notebooks must remain untouched for auditability per AD-004. |

- Nothing else in this change is hard to reverse

## Criteria

Grouped by slice - one observable outcome each, never a layer.

### S1: Modular Taxonomy Curation and Provenance in `viral_cnn.taxonomy.curation` (P1)

Implements the modular taxonomy curation engine supporting ICTV rename mapping, exact and fuzzy matching, lineage resolution, sanitization, and synthetic taxonomy generation.

**Acceptance Criteria**

1. WHEN `build_ictv_rename_map` is passed an ICTV rename DataFrame THEN the system SHALL return a dictionary mapping lowercase old species names to lowercase updated species names for all rows marked with change type `Renamed`.
2. WHEN `resolve_local_ictv_taxonomy` matches species names against the ICTV master list THEN the system SHALL populate lineage ranks (`Realm`, `Kingdom`, `Phylum`, `Class`, `Order`, `Family`, `Genus`) and label the source as `ICTV Master`.
3. WHEN `sanitize_hybrid_taxonomy` processes a raw hybrid taxonomy DataFrame THEN the system SHALL ensure 0-based `model_label` and 1-based `id` are consistent (`model_label = id - 1`), backfill missing ranks with placeholder strings, and return a clean DataFrame.
4. WHEN `generate_synthetic_taxonomy` is invoked with `num_classes` THEN the system SHALL return a deterministic synthetic taxonomy DataFrame containing `id`, `species`, `Genus`, `Family`, `Order`, `Class`, `Phylum`, `Kingdom`, `Realm`, and `model_label` without requiring external files or network access.
5. WHEN `JupyterHubDataTopology` is initialized THEN the system SHALL include canonical path entries for `ictv_master` and `hybrid_taxonomy_parquet`.

**Independent test:** Run unit tests exercising `build_ictv_rename_map`, `resolve_local_ictv_taxonomy`, `sanitize_hybrid_taxonomy`, and `generate_synthetic_taxonomy` on synthetic fixtures, verifying column types, label conversions, and rank backfilling.

### S2: ICTV Taxonomy Curation and Provenance Reproduction Notebook (`05_ictv_taxonomy_curation.ipynb`) (P2)

Creates `notebooks/reproduction/short/05_ictv_taxonomy_curation.ipynb` reproducing ICTV and NCBI taxonomy integration, dataset duplicate audit, and genome-disjoint split feasibility analysis.

**Acceptance Criteria**

6. WHEN `05_ictv_taxonomy_curation.ipynb` executes under `SMOKE_TEST=1` THEN the system SHALL complete without errors, generate a sanitized taxonomy parquet file, compute dataset partition audit metrics, and save execution manifest and outputs to `results_reproduction/taxonomy/`.
7. The system SHALL ensure `05_ictv_taxonomy_curation.ipynb` imports all curation, audit, and split logic exclusively from `viral_cnn.*` and contains zero imports of `short/utils/*`.
8. WHEN `05_ictv_taxonomy_curation.ipynb` executes THEN the system SHALL evaluate partition duplicate rates using `viral_cnn.data.audit` and evaluate genome-disjoint split feasibility using `viral_cnn.data.grouped_split`.
9. The system SHALL preserve historical notebooks `short/ICTV_improve_kraken_database.ipynb` and `short/improve_kraken_database-Copy1.ipynb` strictly unmodified.

**Independent test:** Execute `05_ictv_taxonomy_curation.ipynb` under `SMOKE_TEST=1` via cell execution runner, verifying output parquet creation, manifest generation, and absence of legacy imports.

### S3: ICTV Latent Space and Taxonomic Hierarchy Reproduction Notebook (`06_ictv_latent_space_and_hierarchy.ipynb`) (P3)

Creates `notebooks/reproduction/short/06_ictv_latent_space_and_hierarchy.ipynb` reproducing pre-classifier embedding extraction, cosine KNN, high-dimensional Silhouette scoring, hierarchical rank correlations, independent linear probing, and qualitative UMAP export.

**Acceptance Criteria**

10. WHEN `06_ictv_latent_space_and_hierarchy.ipynb` executes under `SMOKE_TEST=1` THEN the system SHALL complete without errors, evaluate 5-NN cosine accuracy across Species, Genus, and Order, compute high-dimensional Silhouette scores with valid class inclusion reporting, compute hierarchical rank correlations (Pearson $r$ and Spearman $\rho$ with sample count $N$), evaluate the linear probe bottleneck audit, and save all outputs to `results_reproduction/latent_space/`.
11. The system SHALL ensure `06_ictv_latent_space_and_hierarchy.ipynb` imports all latent evaluation, model loading, and metrics logic exclusively from `viral_cnn.*` and contains zero imports of `short/utils/*`.
12. WHEN `06_ictv_latent_space_and_hierarchy.ipynb` exports UMAP projections THEN the system SHALL include explicit disclaimers identifying 2D UMAP as strictly qualitative visualization and not phylogenetic or quantitative cluster separation.
13. WHEN `06_ictv_latent_space_and_hierarchy.ipynb` computes hierarchical correlations between Species, Genus, and Order THEN the system SHALL report sample count $N$ alongside Pearson $r$ and Spearman $\rho$ for each model without truncating valid genera.
14. The system SHALL preserve historical notebook `short/ICTV_latent_space-bkp.ipynb` strictly unmodified.
15. WHILE `SMOKE_TEST=1` is active THEN both reproduction notebooks SHALL complete execution in under 60 seconds on CPU.

**Independent test:** Execute `06_ictv_latent_space_and_hierarchy.ipynb` under `SMOKE_TEST=1` via cell execution runner, verifying output CSV/JSON creation for Tables 5 and 6 and Figure 2, qualitative UMAP disclaimers, and absence of legacy imports.

## Out of scope

Product capabilities only. Process and harness rules live in AGENTS.md or as Observable `n/a`.

| Excluded | Why |
| --- | --- |
| Live interactive queries to NCBI Entrez without local cache | NCBI Entrez requires network and API keys; automated execution and CI must operate offline deterministically per AD-005. |
| Full 1.7M sequence latent feature extraction on CPU | Extracting 2048-d and 768-d features for 1.7M sequences across 5 models requires GPU cluster compute; local testing is verified via synthetic smoke tests per AD-005 and AD-006. |
| In-place modifications to historical notebooks in `short/` | Historical notebooks must remain preserved for provenance per AD-004. |

## Assumptions

Defaults that are not already a numbered criterion. Drop a row once it is.

| Assumption | Chosen default | Rationale | Confirmed? |
| --- | --- | --- | --- |
| ICTV MSL41 taxonomy rank levels | Realm, Kingdom, Phylum, Class, Order, Family, Genus, Species | Standard 8-level ICTV hierarchical rank scheme used in MSL41 and historical notebook cell 12. | y |
| Latent embedding extraction hook point | Forward pre-hook on `model.fc` capturing input tensor | Extracted features represent the exact high-dimensional representations (2048-d ResNet50, 768-d ConvNeXt) before linear classification. | y |
| Cosine KNN parameters | $k=5$, stratified 80/20 reference/query split, $L_2$ normalized cosine distance | Matches historical evaluation in `short/ICTV_latent_space-bkp.ipynb` cell 12 and S08 specification. | y |

**Open questions:** none - all resolved or logged above.

## Observable

Worksheet, not the review. `n/a` needs its reason. One row may group the same decision across several routes.

| Surface | Decision | Landing |
| --- | --- | --- |
| all notebooks `05_ictv_*.ipynb` and `06_ictv_*.ipynb` | execution under SMOKE_TEST=1 without remote GPU cluster | AC 6, AC 10, AC 15 |
| all notebooks `05_ictv_*.ipynb` and `06_ictv_*.ipynb` | error state on missing dataset partitions or missing base weights | existing - handled via `viral_cnn.notebook_runtime:setup_notebook_environment` |
| all notebooks `05_ictv_*.ipynb` and `06_ictv_*.ipynb` | non-destructive output persistence under `results_reproduction/` | AC 6, AC 10 |
| taxonomy curation CLI / scripts | exit codes and logging | n/a - notebook and module runtime only |
| HTTP APIs / webhooks | response/error status | n/a - no network HTTP endpoints |

## Sources

- `short/ICTV_improve_kraken_database.ipynb` - historical ICTV and NCBI taxonomy integration and cleansing pipeline.
- `short/improve_kraken_database-Copy1.ipynb` - historical taxonomy backfill and frequency analysis.
- `short/ICTV_latent_space-bkp.ipynb` - historical pre-classifier embedding extraction, KNN, Silhouette, UMAP, and hierarchical correlation analysis.
- `ROADMAP_REPRODUCAO_NOTEBOOKS.md` - section N04 specifying scope, inputs, outputs, and invariants.
- `docs/latent_space_audit.md` - S08 scientific audit establishing evidence-bounded interpretation of latent space metrics, Table 5, Table 6, and Figure 2.
- `AGENTS.md` - scientific invariants for taxonomy mappings, closed-set classification, and preservation of historical artifacts.

