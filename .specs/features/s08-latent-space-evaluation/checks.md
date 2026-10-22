# S08: Latent Space Re-evaluation and Taxonomic Hierarchy Audit - Checks

Profile: light
Plan: `.specs/features/s08-latent-space-evaluation/plan.md`

## Checks

### S1: Pre-Classifier Embedding Extraction and Cosine KNN · 2 files · 25 KB · ~7k

**C1** - `viral_cnn.eval.latent.extract_latent_embeddings` registers a forward pre-hook on `model.fc` and extracts high-dimensional representations before the linear classifier (AC 1)
Proof: `.venv/bin/python3 -m unittest discover -s tests -k test_extract_latent_embeddings_hook_preserves_dimensions`

**C2** - `viral_cnn.eval.latent.extract_latent_embeddings` batches inference and returns aligned feature array and label array matching dataset count (AC 2)
Proof: `.venv/bin/python3 -m unittest discover -s tests -k test_extract_latent_embeddings_batches_aligned_arrays`

**C3** - `viral_cnn.eval.latent.evaluate_cosine_knn` partitions embeddings into an 80% reference set and a 20% query set using a stratified split with fixed seed 42 (AC 3)
Proof: `.venv/bin/python3 -m unittest discover -s tests -k test_evaluate_cosine_knn_stratified_80_20_split`

**C4** - `viral_cnn.eval.latent.evaluate_cosine_knn` enforces strictly disjoint reference and query partitions with zero self-neighbor presence (AC 4)
Proof: `.venv/bin/python3 -m unittest discover -s tests -k test_evaluate_cosine_knn_zero_self_neighbors`

**C5** - `viral_cnn.eval.latent.evaluate_cosine_knn` $L_2$-normalizes feature vectors and computes cosine similarity (AC 5)
Proof: `.venv/bin/python3 -m unittest discover -s tests -k test_evaluate_cosine_knn_l2_normalized_cosine_similarity`

**C6** - `viral_cnn.eval.latent.evaluate_cosine_knn` classifies query samples via majority vote among 5 nearest reference neighbors with deterministic tie-breaking (AC 6)
Proof: `.venv/bin/python3 -m unittest discover -s tests -k test_evaluate_cosine_knn_majority_vote_top5`

**C7** - `viral_cnn.eval.latent.evaluate_cosine_knn` evaluates global accuracy and per-cluster accuracy across Species, Genus, and Order levels (AC 7)
Proof: `.venv/bin/python3 -m unittest discover -s tests -k test_evaluate_cosine_knn_across_taxonomic_ranks`

### S2: Genome-Disjoint KNN Evaluation · 2 files · 22 KB · ~6k

**C8** - `viral_cnn.eval.latent.evaluate_genome_disjoint_knn` partitions reference and query sets by genome accessions ensuring zero accession overlap (AC 8)
Proof: `.venv/bin/python3 -m unittest discover -s tests -k test_genome_disjoint_knn_zero_accession_overlap`

**C9** - `viral_cnn.eval.latent.evaluate_genome_disjoint_knn` identifies and excludes single-genome species from closed-set genome-disjoint testing, reporting excluded count (AC 9)
Proof: `.venv/bin/python3 -m unittest discover -s tests -k test_genome_disjoint_knn_excludes_single_genome_species`

**C10** - `viral_cnn.eval.latent.evaluate_genome_disjoint_knn` retrieves 5 nearest neighbors strictly from reference genomes distinct from the query genome (AC 10)
Proof: `.venv/bin/python3 -m unittest discover -s tests -k test_genome_disjoint_knn_retrieves_out_of_genome_neighbors`

**C11** - `viral_cnn.eval.latent.evaluate_genome_disjoint_knn` outputs comparative accuracy and computes percentage point difference relative to fragment-split KNN (AC 11)
Proof: `.venv/bin/python3 -m unittest discover -s tests -k test_genome_disjoint_knn_reports_drop_vs_fragment_split`

### S3: High-Dimensional Silhouette Score & Subsampling Audit · 2 files · 25 KB · ~7k

**C12** - `viral_cnn.eval.latent.evaluate_silhouette_with_audit` computes Silhouette Score directly in the original high-dimensional feature space using cosine distance (AC 12)
Proof: `.venv/bin/python3 -m unittest discover -s tests -k test_evaluate_silhouette_original_high_dimensional_cosine`

**C13** - `viral_cnn.eval.latent.evaluate_silhouette_with_audit` audits random subsampling of up to 50,000 samples, recording classes with $<2$ samples (AC 13)
Proof: `.venv/bin/python3 -m unittest discover -s tests -k test_evaluate_silhouette_random_subsampling_audits_dropped_classes`

**C14** - `viral_cnn.eval.latent.evaluate_silhouette_with_audit` reports total classes, valid classes ($\ge 2$ samples), excluded classes ($<2$ samples), and retained sample rate (AC 14)
Proof: `.venv/bin/python3 -m unittest discover -s tests -k test_evaluate_silhouette_reports_class_inclusion_rates`

**C15** - `viral_cnn.eval.latent.evaluate_silhouette_with_audit` supports stratified subsampling guaranteeing representation of all eligible classes with $\ge 2$ samples (AC 15)
Proof: `.venv/bin/python3 -m unittest discover -s tests -k test_evaluate_silhouette_stratified_subsampling_preserves_classes`

**C16** - `viral_cnn.eval.latent.evaluate_silhouette_with_audit` computes global Silhouette Score and per-taxon mean values across Species, Genus, and Order (AC 16)
Proof: `.venv/bin/python3 -m unittest discover -s tests -k test_evaluate_silhouette_global_and_per_taxon_metrics`

### S4: Hierarchical Taxonomic Correspondence and Correlation Audit · 2 files · 22 KB · ~6k

**C17** - `viral_cnn.eval.latent.calculate_taxonomic_correlations` aggregates child taxon metrics to parent taxa via arithmetic mean for Species $\to$ Genus and Genus $\to$ Order (AC 17)
Proof: `.venv/bin/python3 -m unittest discover -s tests -k test_calculate_taxonomic_correlations_child_aggregation_mean`

**C18** - `viral_cnn.eval.latent.calculate_taxonomic_correlations` computes both Pearson $r$ (and $p$-value) and Spearman $\rho$ (and $p$-value) for parent-child pairs (AC 18)
Proof: `.venv/bin/python3 -m unittest discover -s tests -k test_calculate_taxonomic_correlations_pearson_and_spearman`

**C19** - `viral_cnn.eval.latent.calculate_taxonomic_correlations` excludes parent taxa with fewer than 3 child taxa from correlation computation (AC 19)
Proof: `.venv/bin/python3 -m unittest discover -s tests -k test_calculate_taxonomic_correlations_excludes_parents_under_3_children`

**C20** - `viral_cnn.eval.latent.calculate_taxonomic_correlations` preserves independent sample counts $N_{\text{parents\_KNN}}$ and $N_{\text{parents\_Sil}}$ without joint intersection dropping (AC 20)
Proof: `.venv/bin/python3 -m unittest discover -s tests -k test_calculate_taxonomic_correlations_independent_valid_sets`

### S5: ConvNeXt Head Bottleneck & Independent Linear Probe Audit · 2 files · 24 KB · ~6k

**C21** - `viral_cnn.eval.latent.LinearProbeClassifier` trains a linear classification head strictly on training or validation split embeddings (AC 21)
Proof: `.venv/bin/python3 -m unittest discover -s tests -k test_linear_probe_trains_on_train_split_embeddings`

**C22** - `viral_cnn.eval.latent.LinearProbeClassifier` raises `ValueError` if trained directly on test set embeddings without explicit override (AC 22)
Proof: `.venv/bin/python3 -m unittest discover -s tests -k test_linear_probe_forbids_training_on_test_embeddings`

**C23** - `viral_cnn.eval.latent.LinearProbeClassifier` evaluates trained probe on test embeddings, reporting Top-1 accuracy and Macro-F1 (AC 23)
Proof: `.venv/bin/python3 -m unittest discover -s tests -k test_linear_probe_evaluates_on_test_embeddings`

**C24** - `viral_cnn.eval.latent.audit_classifier_bottleneck` outputs comparative diagnostic table contrasting native head accuracy, linear probe accuracy, and transductive test 5-NN accuracy (AC 24)
Proof: `.venv/bin/python3 -m unittest discover -s tests -k test_audit_classifier_bottleneck_comparative_table`

**C25** - `viral_cnn.eval.latent.audit_classifier_bottleneck` records evidence-bounded diagnostic text stating transductive test 5-NN does not prove head bottleneck (AC 25)
Proof: `.venv/bin/python3 -m unittest discover -s tests -k test_audit_classifier_bottleneck_evidence_bounded_statement`

### S6: CLI Runner, Qualitative UMAP Export, and Latent Space Audit Report · 3 files · 28 KB · ~8k

**C26** - CLI `scripts/run_latent_evaluation.py` executes end-to-end latent space evaluation on input models and exports structured CSV reports (AC 26)
Proof: `.venv/bin/python3 -m unittest discover -s tests -k test_run_latent_evaluation_cli_execution_and_reports`

**C27** - `viral_cnn.eval.latent.export_umap_projection` computes 2D UMAP coordinates and outputs file with explicit qualitative disclaimer header (AC 27)
Proof: `.venv/bin/python3 -m unittest discover -s tests -k test_export_umap_projection_qualitative_disclaimer`

**C28** - `docs/latent_space_audit.md` documents corrected Table 5, Table 6 with dual correlations and $N$, genome-disjoint results, and bottleneck audit (AC 28)
Proof: `.venv/bin/python3 -m unittest discover -s tests -k test_latent_space_audit_documentation_sections`

## Coverage

| Set (size) | Member -> proof | Unproven |
| --- | --- | --- |
| Taxonomic evaluation ranks (3) | `species` C7 · `genus` C7 · `order` C7 | - |
| KNN evaluation protocols (2) | `fragment_split` C6 · `genome_disjoint` C10 | - |
| Silhouette sampling strategies (2) | `random` C13 · `stratified` C15 | - |
| Correlation coefficients (2) | `pearson` C18 · `spearman` C18 | - |
| Correlation parent levels (2) | `species_to_genus` C17 · `genus_to_order` C17 | - |
| Head bottleneck diagnostic models (3) | `native_classifier` C24 · `linear_probe` C24 · `transductive_knn` C24 | - |
| CLI execution status codes (2) | `0` C26 · `2` C26 | - |

- Claims naming a status code or response shape: C22, C26 - each has a proof crossing the process boundary
- No other check claims more than the single case its proof exercises

## Swept

- validation: C3, C8
- failure modes: C9, C19
- idempotency: C5, C18
- authorization: existing - local filesystem permissions and script invocation
- concurrency: n/a - batch processing is sequential across evaluation batches
- data lifecycle: n/a - embeddings generated in-memory or persisted to user-specified output directory
- dependency failure: C1
- state transitions: C21, C22
- observability: C24, C26

## Handoff

- S1-S6: 4 files (`src/viral_cnn/eval/latent.py`, `scripts/run_latent_evaluation.py`, `tests/test_latent_space.py`, `docs/latent_space_audit.md`), ~53 KB total (~13.2k tokens), well under 150k budget - one builder
- Mechanism: one builder

- **Boundary:** C1-C28 closed in implementation commit
- **Settled mid-build:** pure NumPy and mpmath implementation of cosine silhouette, Pearson r, and Spearman rho eliminates external C-dependencies (scipy/scikit-learn) and computes Silhouette Score in O(N*K) via cluster centroids
- **Abandoned:** none
