"""
Contract tests for latent space re-evaluation, cosine KNN, Silhouette score audit,
hierarchical taxonomic correlation, and independent linear probe (C1-C28).
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from viral_cnn.eval.latent import (
    LinearProbeClassifier,
    audit_classifier_bottleneck,
    calculate_taxonomic_correlations,
    compute_cosine_silhouette_fast,
    compute_pearson_r,
    compute_spearman_rho,
    evaluate_cosine_knn,
    evaluate_genome_disjoint_knn,
    evaluate_silhouette_with_audit,
    export_umap_projection,
    extract_latent_embeddings,
)
from viral_cnn.models.builder import build_model


class DummyCNN(nn.Module):
    """Simple 1D CNN with fc layer for deterministic unit tests."""

    def __init__(self, in_features: int = 16, num_classes: int = 5) -> None:
        super().__init__()
        self.conv = nn.Conv1d(4, 8, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool1d(2)
        self.fc = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.conv(x)
        feat = self.pool(feat).view(x.size(0), -1)
        return self.fc(feat)


class TestLatentSpaceEvaluation(unittest.TestCase):
    def setUp(self) -> None:
        self.test_dir = Path(tempfile.mkdtemp(prefix="latent_test_"))

    def tearDown(self) -> None:
        shutil.rmtree(self.test_dir, ignore_errors=True)

    # -----------------------------------------------------------------------
    # S1: Embedding Extraction and Cosine KNN (C1-C7)
    # -----------------------------------------------------------------------

    def test_extract_latent_embeddings_hook_preserves_dimensions(self) -> None:
        """C1: Verifies forward pre-hook on model.fc captures exact input dimensions without reduction."""
        model = DummyCNN(in_features=16, num_classes=5)
        inputs = torch.randn(4, 4, 32)
        labels = torch.tensor([0, 1, 2, 3])
        dataloader = [(inputs, labels)]

        feats, extracted_labels = extract_latent_embeddings(model, dataloader, device="cpu")
        self.assertEqual(feats.shape, (4, 16))
        self.assertEqual(extracted_labels.shape, (4,))
        np.testing.assert_array_equal(extracted_labels, [0, 1, 2, 3])

    def test_extract_latent_embeddings_batches_aligned_arrays(self) -> None:
        """C2: Verifies feature and label arrays remain aligned across multiple batches."""
        model = DummyCNN(in_features=16, num_classes=5)
        dataloader = [
            (torch.randn(3, 4, 32), torch.tensor([1, 2, 3])),
            (torch.randn(2, 4, 32), torch.tensor([4, 0])),
        ]

        feats, labels = extract_latent_embeddings(model, dataloader, device="cpu")
        self.assertEqual(len(feats), 5)
        self.assertEqual(len(labels), 5)
        np.testing.assert_array_equal(labels, [1, 2, 3, 4, 0])

    def test_evaluate_cosine_knn_stratified_80_20_split(self) -> None:
        """C3: Verifies 80/20 reference/query split respects specified proportions and seed 42."""
        rng = np.random.default_rng(42)
        feats = rng.normal(0, 1, size=(100, 16)).astype(np.float32)
        labels = np.repeat(np.arange(10), 10)  # 10 classes, 10 samples each

        res = evaluate_cosine_knn(feats, labels, k=5, test_size=0.2, random_state=42)
        self.assertEqual(res["n_reference"], 80)
        self.assertEqual(res["n_query"], 20)

    def test_evaluate_cosine_knn_zero_self_neighbors(self) -> None:
        """C4: Verifies reference and query partitions are strictly disjoint (zero self-neighbors)."""
        rng = np.random.default_rng(42)
        feats = rng.normal(0, 1, size=(50, 16)).astype(np.float32)
        labels = np.repeat(np.arange(5), 10)

        res = evaluate_cosine_knn(feats, labels, k=5, test_size=0.2, random_state=42)
        self.assertTrue(res["zero_self_neighbor"])
        self.assertEqual(res["n_reference"] + res["n_query"], 50)

    def test_evaluate_cosine_knn_l2_normalized_cosine_similarity(self) -> None:
        """C5: Verifies cosine similarity computes L2-normalized dot products."""
        feats = np.array(
            [
                [1.0, 0.0, 0.0],  # ref 0 (class 0)
                [10.0, 0.0, 0.0],  # query 0 (class 0, scaled version of ref 0)
                [0.0, 1.0, 0.0],  # ref 1 (class 1)
                [0.0, 5.0, 0.0],  # query 1 (class 1)
            ],
            dtype=np.float32,
        )
        labels = np.array([0, 0, 1, 1])

        res = evaluate_cosine_knn(feats, labels, k=1, test_size=0.5, random_state=42)
        self.assertEqual(res["global_knn_accuracy"], 1.0)

    def test_evaluate_cosine_knn_majority_vote_top5(self) -> None:
        """C6: Verifies majority vote prediction among 5 nearest neighbors."""
        rng = np.random.default_rng(42)
        # Class 0 centered around (5, 5), Class 1 centered around (-5, -5)
        c0 = rng.normal(5, 0.5, size=(20, 8)).astype(np.float32)
        c1 = rng.normal(-5, 0.5, size=(20, 8)).astype(np.float32)
        feats = np.vstack([c0, c1])
        labels = np.array([0] * 20 + [1] * 20)

        res = evaluate_cosine_knn(feats, labels, k=5, test_size=0.2, random_state=42)
        self.assertEqual(res["global_knn_accuracy"], 1.0)
        self.assertEqual(res["k"], 5)

    def test_evaluate_cosine_knn_across_taxonomic_ranks(self) -> None:
        """C7: Verifies KNN evaluates global accuracy and per-cluster summary across ranks."""
        rng = np.random.default_rng(42)
        feats = rng.normal(0, 1, size=(60, 8)).astype(np.float32)
        labels = np.repeat(np.arange(3), 20)

        res = evaluate_cosine_knn(feats, labels, k=3, taxonomic_rank="Genus")
        self.assertEqual(res["taxonomic_rank"], "Genus")
        self.assertIn("per_cluster_metrics", res)
        self.assertEqual(len(res["per_cluster_metrics"]), 3)

    # -----------------------------------------------------------------------
    # S2: Genome-Disjoint KNN Evaluation (C8-C11)
    # -----------------------------------------------------------------------

    def test_genome_disjoint_knn_zero_accession_overlap(self) -> None:
        """C8: Verifies reference and query partitions share zero genome accessions."""
        feats = np.random.randn(40, 8).astype(np.float32)
        labels = np.repeat([0, 1], 20)
        genomes = np.array(["ACC_A"] * 10 + ["ACC_B"] * 10 + ["ACC_C"] * 10 + ["ACC_D"] * 10)

        res = evaluate_genome_disjoint_knn(feats, labels, genomes, k=3)
        self.assertTrue(res["zero_accession_overlap"])

    def test_genome_disjoint_knn_excludes_single_genome_species(self) -> None:
        """C9: Verifies species with only 1 genome are excluded from query set."""
        feats = np.random.randn(30, 8).astype(np.float32)
        labels = np.array([0] * 20 + [1] * 10)
        genomes = np.array(["ACC_0A"] * 10 + ["ACC_0B"] * 10 + ["ACC_1A"] * 10)

        res = evaluate_genome_disjoint_knn(feats, labels, genomes, k=3)
        self.assertEqual(res["n_multi_genome_species"], 1)
        self.assertEqual(res["n_single_genome_species_excluded"], 1)

    def test_genome_disjoint_knn_retrieves_out_of_genome_neighbors(self) -> None:
        """C10: Verifies 5-NN searches strictly among reference genomes distinct from query."""
        feats = np.random.randn(40, 8).astype(np.float32)
        labels = np.repeat([0, 1], 20)
        genomes = np.array(["ACC_0A"] * 10 + ["ACC_0B"] * 10 + ["ACC_1A"] * 10 + ["ACC_1B"] * 10)

        res = evaluate_genome_disjoint_knn(feats, labels, genomes, k=5)
        self.assertGreater(res["n_query_samples"], 0)
        self.assertGreater(res["n_reference_samples"], 0)

    def test_genome_disjoint_knn_reports_drop_vs_fragment_split(self) -> None:
        """C11: Verifies calculation of percentage point drop between fragment-split and genome-disjoint KNN."""
        rng = np.random.default_rng(42)
        feats = rng.normal(0, 1, size=(60, 8)).astype(np.float32)
        labels = np.repeat([0, 1, 2], 20)
        genomes = np.array(
            ["ACC_0A"] * 10 + ["ACC_0B"] * 10 + ["ACC_1A"] * 10 + ["ACC_1B"] * 10 + ["ACC_2A"] * 10 + ["ACC_2B"] * 10
        )

        res = evaluate_genome_disjoint_knn(feats, labels, genomes, k=3)
        self.assertIn("accuracy_drop_percentage_points", res)
        self.assertAlmostEqual(
            res["accuracy_drop_percentage_points"],
            (res["fragment_split_accuracy"] - res["genome_disjoint_accuracy"]) * 100.0,
            places=4,
        )

    # -----------------------------------------------------------------------
    # S3: Silhouette Score and Subsampling Audit (C12-C16)
    # -----------------------------------------------------------------------

    def test_evaluate_silhouette_original_high_dimensional_cosine(self) -> None:
        """C12: Verifies Silhouette Score is computed using exact cosine distance in high dimensions."""
        c0 = np.array([[1.0, 0.1], [0.9, 0.2], [0.95, 0.05]], dtype=np.float32)
        c1 = np.array([[-1.0, -0.1], [-0.9, -0.2], [-0.95, -0.05]], dtype=np.float32)
        feats = np.vstack([c0, c1])
        labels = np.array([0, 0, 0, 1, 1, 1])

        score, samples = compute_cosine_silhouette_fast(feats, labels)
        self.assertGreater(score, 0.8)
        self.assertEqual(len(samples), 6)

    def test_evaluate_silhouette_random_subsampling_audits_dropped_classes(self) -> None:
        """C13: Verifies random subsampling identifies and audits classes with < 2 samples."""
        feats = np.random.randn(50, 4).astype(np.float32)
        # Class 0 has 40 samples, Classes 1-10 have 1 sample each
        labels = np.array([0] * 40 + list(range(1, 11)))

        res = evaluate_silhouette_with_audit(feats, labels, max_samples=20, sampling_strategy="random")
        self.assertGreater(res["excluded_classes_count"], 0)
        self.assertIn("retained_sample_rate", res)

    def test_evaluate_silhouette_reports_class_inclusion_rates(self) -> None:
        """C14: Verifies reporting of total, valid, and excluded class counts."""
        feats = np.random.randn(30, 4).astype(np.float32)
        labels = np.array([0] * 10 + [1] * 10 + [2, 3, 4, 5, 6, 7, 8, 9, 10, 11])

        res = evaluate_silhouette_with_audit(feats, labels, max_samples=30)
        self.assertEqual(res["valid_classes_count"], 2)
        self.assertEqual(res["excluded_classes_count"], 10)
        self.assertEqual(res["total_classes_sampled"], 12)

    def test_evaluate_silhouette_stratified_subsampling_preserves_classes(self) -> None:
        """C15: Verifies stratified subsampling preserves representation of all eligible classes."""
        feats = np.random.randn(60, 4).astype(np.float32)
        labels = np.repeat(np.arange(6), 10)

        res = evaluate_silhouette_with_audit(feats, labels, max_samples=24, sampling_strategy="stratified")
        self.assertEqual(res["valid_classes_count"], 6)
        self.assertEqual(res["excluded_classes_count"], 0)

    def test_evaluate_silhouette_global_and_per_taxon_metrics(self) -> None:
        """C16: Verifies calculation of global score and per-taxon metrics table."""
        feats = np.random.randn(40, 8).astype(np.float32)
        labels = np.repeat([0, 1], 20)

        res = evaluate_silhouette_with_audit(feats, labels, max_samples=40, taxonomic_rank="Species")
        self.assertIn("global_silhouette_score", res)
        self.assertEqual(len(res["per_cluster_metrics"]), 2)

    # -----------------------------------------------------------------------
    # S4: Hierarchical Taxonomic Correlation (C17-C20)
    # -----------------------------------------------------------------------

    def test_calculate_taxonomic_correlations_child_aggregation_mean(self) -> None:
        """C17: Verifies child metrics are aggregated to parent taxa via arithmetic mean."""
        child_df = pd.DataFrame(
            {
                "child_name": ["sp1", "sp2", "sp3", "sp4"],
                "knn_accuracy": [0.6, 0.8, 0.4, 0.6],
                "silhouette_score": [-0.1, -0.2, -0.3, -0.4],
            }
        )
        parent_df = pd.DataFrame(
            {
                "parent_name": ["gen1", "gen2"],
                "knn_accuracy": [0.7, 0.5],
                "silhouette_score": [-0.15, -0.35],
            }
        )
        tax_map = pd.DataFrame(
            {
                "species": ["sp1", "sp2", "sp3", "sp4"],
                "Genus": ["gen1", "gen1", "gen2", "gen2"],
            }
        )

        res = calculate_taxonomic_correlations(
            child_df, parent_df, tax_map, child_tax_col="species", parent_tax_col="Genus", min_children=2
        )
        summary = res["knn_summary_df"]
        gen1_mean = summary.loc[summary["parent_name"] == "gen1", "child_knn_mean"].values[0]
        self.assertAlmostEqual(gen1_mean, 0.7, places=5)

    def test_calculate_taxonomic_correlations_pearson_and_spearman(self) -> None:
        """C18: Verifies dual calculation of Pearson r and Spearman rho with p-values."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = np.array([2.0, 4.0, 6.0, 8.0, 10.0])
        r, p_r = compute_pearson_r(x, y)
        rho, p_rho = compute_spearman_rho(x, y)

        self.assertAlmostEqual(r, 1.0, places=5)
        self.assertAlmostEqual(rho, 1.0, places=5)
        self.assertEqual(p_r, 0.0)

    def test_calculate_taxonomic_correlations_excludes_parents_under_3_children(self) -> None:
        """C19: Verifies parent taxa with < 3 child taxa are excluded from correlation calculation."""
        child_df = pd.DataFrame(
            {
                "child_name": ["sp1", "sp2", "sp3", "sp4", "sp5"],
                "knn_accuracy": [0.5, 0.6, 0.7, 0.8, 0.9],
            }
        )
        parent_df = pd.DataFrame(
            {
                "parent_name": ["gen_small", "gen_large"],
                "knn_accuracy": [0.55, 0.8],
            }
        )
        # gen_small has 2 children; gen_large has 3 children
        tax_map = pd.DataFrame(
            {
                "species": ["sp1", "sp2", "sp3", "sp4", "sp5"],
                "Genus": ["gen_small", "gen_small", "gen_large", "gen_large", "gen_large"],
            }
        )

        res = calculate_taxonomic_correlations(
            child_df, parent_df, tax_map, child_tax_col="species", parent_tax_col="Genus", min_children=3
        )
        self.assertEqual(res["n_parents_knn"], 1)

    def test_calculate_taxonomic_correlations_independent_valid_sets(self) -> None:
        """C20: Verifies sample sizes N for KNN and Silhouette remain independent."""
        child_df = pd.DataFrame(
            {
                "child_name": [f"sp{i}" for i in range(6)],
                "knn_accuracy": [0.7] * 6,
                # gen2 child has missing silhouette
                "silhouette_score": [-0.1, -0.2, -0.3, np.nan, np.nan, np.nan],
            }
        )
        parent_df = pd.DataFrame(
            {
                "parent_name": ["gen1", "gen2"],
                "knn_accuracy": [0.7, 0.7],
                "silhouette_score": [-0.15, -0.25],
            }
        )
        tax_map = pd.DataFrame(
            {
                "species": [f"sp{i}" for i in range(6)],
                "Genus": ["gen1"] * 3 + ["gen2"] * 3,
            }
        )

        res = calculate_taxonomic_correlations(
            child_df, parent_df, tax_map, child_tax_col="species", parent_tax_col="Genus", min_children=3
        )
        self.assertEqual(res["n_parents_knn"], 2)
        self.assertEqual(res["n_parents_silhouette"], 1)

    # -----------------------------------------------------------------------
    # S5: ConvNeXt Head Bottleneck & Independent Linear Probe (C21-C25)
    # -----------------------------------------------------------------------

    def test_linear_probe_trains_on_train_split_embeddings(self) -> None:
        """C21: Verifies LinearProbeClassifier trains on training embeddings with cross-entropy."""
        rng = np.random.default_rng(42)
        X_train = rng.normal(0, 1, size=(50, 16)).astype(np.float32)
        y_train = rng.integers(0, 3, size=50)

        probe = LinearProbeClassifier(in_features=16, num_classes=3)
        probe.fit(X_train, y_train, split_origin="train", epochs=2)
        self.assertIsNotNone(probe.fc.weight)

    def test_linear_probe_forbids_training_on_test_embeddings(self) -> None:
        """C22: Verifies LinearProbeClassifier raises ValueError when split_origin is 'test'."""
        probe = LinearProbeClassifier(in_features=8, num_classes=2)
        with self.assertRaises(ValueError):
            probe.fit(np.zeros((10, 8)), np.zeros(10), split_origin="test")

    def test_linear_probe_evaluates_on_test_embeddings(self) -> None:
        """C23: Verifies evaluation of linear probe reporting Top-1 accuracy and Macro-F1."""
        X_test = np.random.randn(20, 8).astype(np.float32)
        y_test = np.array([0] * 10 + [1] * 10)

        probe = LinearProbeClassifier(in_features=8, num_classes=2)
        res = probe.evaluate(X_test, y_test)
        self.assertIn("top1_accuracy", res)
        self.assertIn("macro_f1", res)
        self.assertEqual(res["n_test_samples"], 20)

    def test_audit_classifier_bottleneck_comparative_table(self) -> None:
        """C24: Verifies diagnostic output contrasts native head, linear probe, and 5-NN."""
        res = audit_classifier_bottleneck(
            native_model_accuracy=0.582,
            linear_probe_accuracy=0.586,
            transductive_knn_accuracy=0.630,
        )
        self.assertIn("summary_table", res)
        table = res["summary_table"]
        self.assertEqual(len(table), 3)
        self.assertIn("accuracy", table.columns)

    def test_audit_classifier_bottleneck_evidence_bounded_statement(self) -> None:
        """C25: Verifies diagnostic text explicitly states transductive 5-NN does not prove head bottleneck."""
        res = audit_classifier_bottleneck(
            native_model_accuracy=0.582,
            linear_probe_accuracy=0.586,
            transductive_knn_accuracy=0.630,
        )
        statement = res["diagnostic_statement"]
        self.assertIn("does NOT demonstrate undercapacity", statement)
        self.assertFalse(res["proves_head_bottleneck"])

    # -----------------------------------------------------------------------
    # S6: CLI Runner, UMAP Export, and Latent Space Audit Report (C26-C28)
    # -----------------------------------------------------------------------

    def test_run_latent_evaluation_cli_execution_and_reports(self) -> None:
        """C26: Verifies CLI runs synthetic pipeline producing summary CSV and JSON manifest."""
        out_dir = self.test_dir / "cli_out"
        cmd = [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "run_latent_evaluation.py"),
            "--synthetic",
            "--sample-size",
            "100",
            "--output-dir",
            str(out_dir),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, f"CLI stderr: {result.stderr}")
        self.assertTrue((out_dir / "latent_metrics_summary.csv").exists())
        self.assertTrue((out_dir / "latent_evaluation_manifest.json").exists())

    def test_export_umap_projection_qualitative_disclaimer(self) -> None:
        """C27: Verifies UMAP export writes explicit qualitative disclaimer header."""
        feats = np.random.randn(20, 8).astype(np.float32)
        labels = np.array([0] * 10 + [1] * 10)
        out_csv = self.test_dir / "umap_out.csv"

        df = export_umap_projection(feats, labels, output_path=out_csv)
        self.assertTrue(out_csv.exists())
        with open(out_csv, "r", encoding="utf-8") as f:
            first_line = f.readline()
        self.assertIn("QUALITATIVE ONLY", first_line)

    def test_latent_space_audit_documentation_sections(self) -> None:
        """C28: Verifies docs/latent_space_audit.md documents corrected Table 5, Table 6, and bottleneck audit."""
        doc_path = PROJECT_ROOT / "docs" / "latent_space_audit.md"
        self.assertTrue(doc_path.exists())
        content = doc_path.read_text(encoding="utf-8")
        self.assertIn("Table 5 Corrected", content)
        self.assertIn("Table 6 Corrected", content)
        self.assertIn("Resolution of the ConvNeXt Classifier Head", content)
        self.assertIn("Genome-Disjoint KNN Generalization", content)


if __name__ == "__main__":
    unittest.main()

