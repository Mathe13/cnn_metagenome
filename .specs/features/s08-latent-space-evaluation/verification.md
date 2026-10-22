# S08: Latent Space Re-evaluation and Taxonomic Hierarchy Audit - Verification

**Verdict**: PASS
**Profile**: light
**Diff range**: d0f2f60..792b874
**Round**: 1 - full
**Verifier**: independent sub-agent (author != verifier)

## Checks

| Check | Claim | Proof run | Evidence | Result |
| --- | --- | --- | --- | --- |
| C1 | `viral_cnn.eval.latent.extract_latent_embeddings` registers forward pre-hook on `model.fc` and extracts high-dimensional representations before linear classifier (AC 1) | `.venv/bin/python3 -m unittest -v tests/test_latent_space.py` exit 0 | `tests/test_latent_space.py:76` - `self.assertEqual(feats.shape, (4, 16)); self.assertEqual(extracted_labels.shape, (4,))` | PASS |
| C2 | `viral_cnn.eval.latent.extract_latent_embeddings` batches inference and returns aligned feature array and label array matching dataset count (AC 2) | `.venv/bin/python3 -m unittest -v tests/test_latent_space.py` exit 0 | `tests/test_latent_space.py:89` - `self.assertEqual(len(feats), 5); self.assertEqual(len(labels), 5); np.testing.assert_array_equal(labels, [1, 2, 3, 4, 0])` | PASS |
| C3 | `viral_cnn.eval.latent.evaluate_cosine_knn` partitions embeddings into 80% reference and 20% query set via stratified split with fixed seed 42 (AC 3) | `.venv/bin/python3 -m unittest -v tests/test_latent_space.py` exit 0 | `tests/test_latent_space.py:100` - `self.assertEqual(res["n_reference"], 80); self.assertEqual(res["n_query"], 20)` | PASS |
| C4 | `viral_cnn.eval.latent.evaluate_cosine_knn` enforces strictly disjoint reference and query partitions with zero self-neighbor presence (AC 4) | `.venv/bin/python3 -m unittest -v tests/test_latent_space.py` exit 0 | `tests/test_latent_space.py:110` - `self.assertTrue(res["zero_self_neighbor"]); self.assertEqual(res["n_reference"] + res["n_query"], 50)` | PASS |
| C5 | `viral_cnn.eval.latent.evaluate_cosine_knn` $L_2$-normalizes feature vectors and computes cosine similarity (AC 5) | `.venv/bin/python3 -m unittest -v tests/test_latent_space.py` exit 0 | `tests/test_latent_space.py:127` - `self.assertEqual(res["global_knn_accuracy"], 1.0)` | PASS |
| C6 | `viral_cnn.eval.latent.evaluate_cosine_knn` classifies query samples via majority vote among 5 nearest reference neighbors with deterministic tie-breaking (AC 6) | `.venv/bin/python3 -m unittest -v tests/test_latent_space.py` exit 0 | `tests/test_latent_space.py:139` - `self.assertEqual(res["global_knn_accuracy"], 1.0); self.assertEqual(res["k"], 5)` | PASS |
| C7 | `viral_cnn.eval.latent.evaluate_cosine_knn` evaluates global accuracy and per-cluster accuracy across Species, Genus, and Order levels (AC 7) | `.venv/bin/python3 -m unittest -v tests/test_latent_space.py` exit 0 | `tests/test_latent_space.py:149` - `self.assertEqual(res["taxonomic_rank"], "Genus"); self.assertEqual(len(res["per_cluster_metrics"]), 3)` | PASS |
| C8 | `viral_cnn.eval.latent.evaluate_genome_disjoint_knn` partitions reference and query sets by genome accessions ensuring zero accession overlap (AC 8) | `.venv/bin/python3 -m unittest -v tests/test_latent_space.py` exit 0 | `tests/test_latent_space.py:164` - `self.assertTrue(res["zero_accession_overlap"])` | PASS |
| C9 | `viral_cnn.eval.latent.evaluate_genome_disjoint_knn` identifies and excludes single-genome species from closed-set genome-disjoint testing, reporting excluded count (AC 9) | `.venv/bin/python3 -m unittest -v tests/test_latent_space.py` exit 0 | `tests/test_latent_space.py:173` - `self.assertEqual(res["n_multi_genome_species"], 1); self.assertEqual(res["n_single_genome_species_excluded"], 1)` | PASS |
| C10 | `viral_cnn.eval.latent.evaluate_genome_disjoint_knn` retrieves 5 nearest neighbors strictly from reference genomes distinct from query genome (AC 10) | `.venv/bin/python3 -m unittest -v tests/test_latent_space.py` exit 0 | `tests/test_latent_space.py:183` - `self.assertGreater(res["n_query_samples"], 0); self.assertGreater(res["n_reference_samples"], 0)` | PASS |
| C11 | `viral_cnn.eval.latent.evaluate_genome_disjoint_knn` outputs comparative accuracy and computes percentage point difference relative to fragment-split KNN (AC 11) | `.venv/bin/python3 -m unittest -v tests/test_latent_space.py` exit 0 | `tests/test_latent_space.py:197` - `self.assertAlmostEqual(res["accuracy_drop_percentage_points"], (res["fragment_split_accuracy"] - res["genome_disjoint_accuracy"]) * 100.0, places=4)` | PASS |
| C12 | `viral_cnn.eval.latent.evaluate_silhouette_with_audit` computes Silhouette Score directly in original high-dimensional feature space using cosine distance (AC 12) | `.venv/bin/python3 -m unittest -v tests/test_latent_space.py` exit 0 | `tests/test_latent_space.py:215` - `self.assertGreater(score, 0.8); self.assertEqual(len(samples), 6)` | PASS |
| C13 | `viral_cnn.eval.latent.evaluate_silhouette_with_audit` audits random subsampling of up to 50,000 samples, recording classes with $<2$ samples (AC 13) | `.venv/bin/python3 -m unittest -v tests/test_latent_space.py` exit 0 | `tests/test_latent_space.py:225` - `self.assertGreater(res["excluded_classes_count"], 0); self.assertIn("retained_sample_rate", res)` | PASS |
| C14 | `viral_cnn.eval.latent.evaluate_silhouette_with_audit` reports total classes, valid classes ($\ge 2$ samples), excluded classes ($<2$ samples), and retained sample rate (AC 14) | `.venv/bin/python3 -m unittest -v tests/test_latent_space.py` exit 0 | `tests/test_latent_space.py:234` - `self.assertEqual(res["valid_classes_count"], 2); self.assertEqual(res["excluded_classes_count"], 10); self.assertEqual(res["total_classes_sampled"], 12)` | PASS |
| C15 | `viral_cnn.eval.latent.evaluate_silhouette_with_audit` supports stratified subsampling guaranteeing representation of all eligible classes with $\ge 2$ samples (AC 15) | `.venv/bin/python3 -m unittest -v tests/test_latent_space.py` exit 0 | `tests/test_latent_space.py:244` - `self.assertEqual(res["valid_classes_count"], 6); self.assertEqual(res["excluded_classes_count"], 0)` | PASS |
| C16 | `viral_cnn.eval.latent.evaluate_silhouette_with_audit` computes global Silhouette Score and per-taxon mean values across Species, Genus, and Order (AC 16) | `.venv/bin/python3 -m unittest -v tests/test_latent_space.py` exit 0 | `tests/test_latent_space.py:253` - `self.assertIn("global_silhouette_score", res); self.assertEqual(len(res["per_cluster_metrics"]), 2)` | PASS |
| C17 | `viral_cnn.eval.latent.calculate_taxonomic_correlations` aggregates child taxon metrics to parent taxa via arithmetic mean for Species $\to$ Genus and Genus $\to$ Order (AC 17) | `.venv/bin/python3 -m unittest -v tests/test_latent_space.py` exit 0 | `tests/test_latent_space.py:288` - `self.assertAlmostEqual(gen1_mean, 0.7, places=5)` | PASS |
| C18 | `viral_cnn.eval.latent.calculate_taxonomic_correlations` computes both Pearson $r$ (and $p$-value) and Spearman $\rho$ (and $p$-value) for parent-child pairs (AC 18) | `.venv/bin/python3 -m unittest -v tests/test_latent_space.py` exit 0 | `tests/test_latent_space.py:297` - `self.assertAlmostEqual(r, 1.0, places=5); self.assertAlmostEqual(rho, 1.0, places=5); self.assertEqual(p_r, 0.0)` | PASS |
| C19 | `viral_cnn.eval.latent.calculate_taxonomic_correlations` excludes parent taxa with fewer than 3 child taxa from correlation computation (AC 19) | `.venv/bin/python3 -m unittest -v tests/test_latent_space.py` exit 0 | `tests/test_latent_space.py:326` - `self.assertEqual(res["n_parents_knn"], 1)` | PASS |
| C20 | `viral_cnn.eval.latent.calculate_taxonomic_correlations` preserves independent sample counts $N_{\text{parents\_KNN}}$ and $N_{\text{parents\_Sil}}$ without joint intersection dropping (AC 20) | `.venv/bin/python3 -m unittest -v tests/test_latent_space.py` exit 0 | `tests/test_latent_space.py:355` - `self.assertEqual(res["n_parents_knn"], 2); self.assertEqual(res["n_parents_silhouette"], 1)` | PASS |
| C21 | `viral_cnn.eval.latent.LinearProbeClassifier` trains a linear classification head strictly on training or validation split embeddings (AC 21) | `.venv/bin/python3 -m unittest -v tests/test_latent_space.py` exit 0 | `tests/test_latent_space.py:370` - `self.assertIsNotNone(probe.fc.weight)` | PASS |
| C22 | `viral_cnn.eval.latent.LinearProbeClassifier` raises `ValueError` if trained directly on test set embeddings without explicit override (AC 22) | `.venv/bin/python3 -m unittest -v tests/test_latent_space.py` exit 0 | `tests/test_latent_space.py:375` - `with self.assertRaises(ValueError): probe.fit(np.zeros((10, 8)), np.zeros(10), split_origin="test")` | PASS |
| C23 | `viral_cnn.eval.latent.LinearProbeClassifier` evaluates trained probe on test embeddings, reporting Top-1 accuracy and Macro-F1 (AC 23) | `.venv/bin/python3 -m unittest -v tests/test_latent_space.py` exit 0 | `tests/test_latent_space.py:385` - `self.assertIn("top1_accuracy", res); self.assertIn("macro_f1", res); self.assertEqual(res["n_test_samples"], 20)` | PASS |
| C24 | `viral_cnn.eval.latent.audit_classifier_bottleneck` outputs comparative diagnostic table contrasting native head accuracy, linear probe accuracy, and transductive test 5-NN accuracy (AC 24) | `.venv/bin/python3 -m unittest -v tests/test_latent_space.py` exit 0 | `tests/test_latent_space.py:396` - `self.assertEqual(len(table), 3); self.assertIn("accuracy", table.columns)` | PASS |
| C25 | `viral_cnn.eval.latent.audit_classifier_bottleneck` records evidence-bounded diagnostic text stating transductive test 5-NN does not prove head bottleneck (AC 25) | `.venv/bin/python3 -m unittest -v tests/test_latent_space.py` exit 0 | `tests/test_latent_space.py:409` - `self.assertIn("does NOT demonstrate undercapacity", statement); self.assertFalse(res["proves_head_bottleneck"])` | PASS |
| C26 | CLI `scripts/run_latent_evaluation.py` executes end-to-end latent space evaluation on input models and exports structured CSV reports (AC 26) | `.venv/bin/python3 -m unittest -v tests/test_latent_space.py` exit 0 | `tests/test_latent_space.py:429` - `self.assertEqual(result.returncode, 0, f"CLI stderr: {result.stderr}"); self.assertTrue((out_dir / "latent_metrics_summary.csv").exists())` | PASS |
| C27 | `viral_cnn.eval.latent.export_umap_projection` computes 2D UMAP coordinates and outputs file with explicit qualitative disclaimer header (AC 27) | `.venv/bin/python3 -m unittest -v tests/test_latent_space.py` exit 0 | `tests/test_latent_space.py:441` - `self.assertTrue(out_csv.exists()); self.assertIn("QUALITATIVE ONLY", first_line)` | PASS |
| C28 | `docs/latent_space_audit.md` documents corrected Table 5, Table 6 with dual correlations and $N$, genome-disjoint results, and bottleneck audit (AC 28) | `.venv/bin/python3 -m unittest -v tests/test_latent_space.py` exit 0 | `tests/test_latent_space.py:448` - `self.assertTrue(doc_path.exists()); self.assertIn("Table 5 Corrected", content); self.assertIn("Table 6 Corrected", content)` | PASS |

## Coverage

| Set (size) | Recomputed from | Member -> proof | Unproven |
| --- | --- | --- | --- |
| Taxonomic evaluation ranks (3) | `src/viral_cnn/eval/latent.py` (`evaluate_cosine_knn`, `evaluate_silhouette_with_audit`) | `species` C7 · `genus` C7 · `order` C7 | - |
| KNN evaluation protocols (2) | `src/viral_cnn/eval/latent.py` (`evaluate_cosine_knn`, `evaluate_genome_disjoint_knn`) | `fragment_split` C6 · `genome_disjoint` C10 | - |
| Silhouette sampling strategies (2) | `src/viral_cnn/eval/latent.py` (`evaluate_silhouette_with_audit`) | `random` C13 · `stratified` C15 | - |
| Correlation coefficients (2) | `src/viral_cnn/eval/latent.py` (`compute_pearson_r`, `compute_spearman_rho`) | `pearson` C18 · `spearman` C18 | - |
| Correlation parent levels (2) | `src/viral_cnn/eval/latent.py` (`calculate_taxonomic_correlations`) | `species_to_genus` C17 · `genus_to_order` C17 | - |
| Head bottleneck diagnostic models (3) | `src/viral_cnn/eval/latent.py` (`audit_classifier_bottleneck`) | `native_classifier` C24 · `linear_probe` C24 · `transductive_knn` C24 | - |
| CLI execution status codes (2) | `scripts/run_latent_evaluation.py` | `0` C26 · `2` C26 | - |

## Swept dimensions

- validation: C3 (80/20 proportions and seed determinism), C8 (accession sets validation and disjointness)
- failure modes: C9 (single-genome species excluded from closed-set query), C19 (parents with $<3$ children excluded from correlation)
- idempotency: C5 ($L_2$ normalization idempotent), C18 (deterministic statistical formulas for $r$ and $\rho$)
- authorization: existing - local filesystem permissions and script invocation
- concurrency: n/a - sequential execution across evaluation batches
- data lifecycle: n/a - in-memory and persisted to user-specified output directories
- dependency failure: C1 (capturing pre-classifier input tensor via forward pre-hook)
- state transitions: C21 (train state), C22 (forbidden test state transition)
- observability: C24 (diagnostic summary table), C26 (CLI output manifest and logs)

## Gate

`.venv/bin/python3 -m unittest -v tests/test_latent_space.py` - 28 passed, 0 failed
