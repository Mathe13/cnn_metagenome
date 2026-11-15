"""Tests for C03: Table and Figure Publication Adapters."""

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from viral_cnn.reporting.publication import PublicationValidationError
from viral_cnn.reporting.transformations import (
    execute_publication_transformation,
    get_transformation,
    transform_figure_1_noise_robustness,
    transform_figure_2_umap_visualization,
    transform_figure_4_rscu_profile,
    transform_table_1_metrics,
    transform_table_2_profiling,
    transform_table_3_noise_degradation,
    transform_table_4_species_f1_and_pairwise,
    transform_table_5_latent_metrics,
    transform_table_6_hierarchy_correlations,
    transform_table_7_hierarchical_entropy,
    transform_table_8_paired_motif_perturbation,
    transform_table_9_rscu_exploratory,
    transform_table_10_triplet_composition,
    transform_table_12_long_sequence_consensus,
)


class TestC03PublicationAdapters(unittest.TestCase):
    """Test suite verifying C03 publication adapters contracts."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name).resolve()

    def tearDown(self):
        self.temp_dir.cleanup()

    # C1: T1 produces complete columns without zero-fill
    def test_t1_produces_complete_columns_without_zero_fill(self):
        """C1: transform_table_1_metrics extracts complete published columns from measured data."""
        raw_df = pd.DataFrame({
            "model_name": ["ResNet50", "ConvNeXt_Z"],
            "rank": ["species", "species"],
            "backbone_m": [23.5, 27.8],
            "fc_params_m": [2.0, 3.5],
            "parameters_m": [25.5, 31.3],
            "top1_accuracy": [0.945, 0.932],
            "top5_accuracy": [0.988, 0.981],
            "macro_f1": [0.923, 0.910],
            "latency_ms": [1.25, 2.10],
        })
        out_df = transform_table_1_metrics(raw_df, {})
        expected_cols = [
            "model", "taxonomic_level", "backbone_m", "fc_params_m",
            "parameters_m", "accuracy", "top_5", "macro_f1", "latency_ms"
        ]
        self.assertEqual(list(out_df.columns), expected_cols)
        self.assertEqual(len(out_df), 2)
        self.assertEqual(out_df["model"].tolist(), ["ResNet50", "ConvNeXt_Z"])
        self.assertEqual(out_df["taxonomic_level"].tolist(), ["species", "species"])

    # C2: T1 rejects missing or null measurements
    def test_t1_rejects_missing_or_null_measurements(self):
        """C2: transform_table_1_metrics raises PublicationValidationError on missing or NaN columns."""
        # Missing column (latency_ms)
        missing_df = pd.DataFrame({
            "model": ["ResNet50"], "taxonomic_level": ["species"],
            "backbone_m": [23.5], "fc_params_m": [2.0], "parameters_m": [25.5],
            "accuracy": [0.95], "top_5": [0.99], "macro_f1": [0.94],
        })
        with self.assertRaises(PublicationValidationError) as ctx:
            transform_table_1_metrics(missing_df, {})
        self.assertIn("missing required measurement columns", str(ctx.exception))

        # Null measurement (NaN in latency_ms)
        null_df = missing_df.copy()
        null_df["latency_ms"] = [None]
        with self.assertRaises(PublicationValidationError) as ctx:
            transform_table_1_metrics(null_df, {})
        self.assertIn("contains null measurements", str(ctx.exception))

    # C3: T1 validates metric bounds [0, 1] or [0, 100]
    def test_t1_validates_metric_bounds(self):
        """C3: transform_table_1_metrics rejects accuracy, top_5, or macro_f1 outside [0, 1] / [0, 100]."""
        base_df = pd.DataFrame({
            "model": ["ResNet50"], "taxonomic_level": ["species"],
            "backbone_m": [23.5], "fc_params_m": [2.0], "parameters_m": [25.5],
            "accuracy": [0.95], "top_5": [0.99], "macro_f1": [0.94],
            "latency_ms": [1.2],
        })
        # Invalid negative accuracy
        bad_negative = base_df.copy()
        bad_negative["accuracy"] = [-0.05]
        with self.assertRaises(PublicationValidationError) as ctx:
            transform_table_1_metrics(bad_negative, {})
        self.assertIn("outside valid bounds", str(ctx.exception))

        # Invalid accuracy > 100
        bad_high = base_df.copy()
        bad_high["top_5"] = [105.0]
        with self.assertRaises(PublicationValidationError) as ctx:
            transform_table_1_metrics(bad_high, {})
        self.assertIn("outside valid bounds", str(ctx.exception))

        # Percentage format (0-100) is accepted
        valid_percentage = base_df.copy()
        valid_percentage["accuracy"] = [95.0]
        valid_percentage["top_5"] = [99.0]
        valid_percentage["macro_f1"] = [94.0]
        out_pct = transform_table_1_metrics(valid_percentage, {})
        self.assertEqual(len(out_pct), 1)

    # C4: T2 maps profiler columns strictly and rejects missing measurements
    def test_t2_maps_profiler_columns_strictly(self):
        """C4: transform_table_2_profiling maps profiler columns without zero-filling."""
        profiler_df = pd.DataFrame({
            "model_name": ["ConvNeXt_F"],
            "backbone_params": [135980000],
            "head_params": [28000000],
            "total_params": [163980000],
            "size_mb": [1450.0],
            "avg_latency_ms": [3.45],
            "throughput_seq_sec": [295.0],
            "gpu_hours": [12.5],
        })
        out_df = transform_table_2_profiling(profiler_df, {})
        self.assertEqual(out_df["model"].iloc[0], "ConvNeXt_F")
        self.assertEqual(out_df["backbone_m"].iloc[0], 135.98)
        self.assertEqual(out_df["fc_params_m"].iloc[0], 28.0)
        self.assertEqual(out_df["parameters_m"].iloc[0], 163.98)
        self.assertEqual(out_df["peak_vram_mb"].iloc[0], 1450.0)
        self.assertEqual(out_df["step_time_ms"].iloc[0], 3.45)
        self.assertEqual(out_df["throughput_seq_s"].iloc[0], 295.0)
        self.assertEqual(out_df["gpu_hours"].iloc[0], 12.5)

        # Missing required throughput
        missing_df = profiler_df.drop(columns=["throughput_seq_sec"])
        with self.assertRaises(PublicationValidationError):
            transform_table_2_profiling(missing_df, {})

    # C5: T3 and F1 noise_rate to epsilon conversion
    def test_t3_f1_noise_rate_conversion(self):
        """C5: transform_table_3_noise_degradation and transform_figure_1_noise_robustness map noise_rate to epsilon."""
        raw_t3 = pd.DataFrame({
            "model": ["ResNet50", "ResNet50"],
            "noise_rate": [0.0, 0.05],
            "accuracy": [0.945, 0.880],
            "top_5": [0.988, 0.950],
            "macro_f1": [0.920, 0.850],
        })
        out_t3 = transform_table_3_noise_degradation(raw_t3, {})
        self.assertIn("epsilon", out_t3.columns)
        self.assertEqual(out_t3["epsilon"].tolist(), [0.0, 0.05])

        out_f1 = transform_figure_1_noise_robustness(raw_t3, {})
        self.assertIn("epsilon", out_f1.columns)
        self.assertEqual(out_f1["epsilon"].tolist(), [0.0, 0.05])

    # C6: T6 Transition to level_pair mapping
    def test_t6_transition_to_level_pair(self):
        """C6: transform_table_6_hierarchy_correlations maps Transition to level_pair preserving correlations."""
        raw_t6 = pd.DataFrame({
            "Transition": ["Species -> Genus", "Genus -> Order"],
            "Model": ["ResNet50", "ResNet50"],
            "Pearson_r": [0.82, 0.65],
            "Spearman_rho": [0.80, 0.61],
            "N_taxa": [2702, 120],
        })
        out_t6 = transform_table_6_hierarchy_correlations(raw_t6, {})
        self.assertIn("level_pair", out_t6.columns)
        self.assertEqual(out_t6["level_pair"].tolist(), ["Species -> Genus", "Genus -> Order"])
        self.assertEqual(out_t6["pearson_r"].tolist(), [0.82, 0.65])
        self.assertEqual(out_t6["spearman_rho"].tolist(), [0.80, 0.61])
        self.assertEqual(out_t6["n_taxa"].tolist(), [2702, 120])

    # C7: T12 pivots variant aggregations
    def test_t12_pivots_variant_aggregations(self):
        """C7: transform_table_12_long_sequence_consensus pivots majority_vote and mean_logits per variant."""
        unpivoted_df = pd.DataFrame({
            "variant": ["ResNet50", "ResNet50", "ConvNeXt_F", "ConvNeXt_F"],
            "aggregation": ["majority_vote", "mean_logits", "majority_vote", "mean_logits"],
            "accuracy": [0.725, 0.742, 0.615, 0.638],
            "coverage": [1.0, 1.0, 1.0, 1.0],
            "N": [4514, 4514, 4514, 4514],
        })
        out_t12 = transform_table_12_long_sequence_consensus(unpivoted_df, {})
        self.assertEqual(len(out_t12), 2)
        self.assertIn("majority_vote_acc", out_t12.columns)
        self.assertIn("mean_logits_acc", out_t12.columns)
        self.assertEqual(out_t12["model"].tolist(), ["ConvNeXt_F", "ResNet50"])
        self.assertEqual(out_t12["majority_vote_acc"].tolist(), [0.615, 0.725])
        self.assertEqual(out_t12["mean_logits_acc"].tolist(), [0.638, 0.742])

    # C8: T12 rejects missing or duplicated aggregations
    def test_t12_rejects_missing_or_duplicated_aggregations(self):
        """C8: transform_table_12_long_sequence_consensus raises PublicationValidationError if an aggregation is missing or duplicated."""
        # Missing mean_logits
        missing_agg = pd.DataFrame({
            "variant": ["ResNet50"],
            "aggregation": ["majority_vote"],
            "accuracy": [0.725],
        })
        with self.assertRaises(PublicationValidationError) as ctx:
            transform_table_12_long_sequence_consensus(missing_agg, {})
        self.assertIn("missing required aggregation", str(ctx.exception))

        # Duplicated majority_vote
        duplicated_agg = pd.DataFrame({
            "variant": ["ResNet50", "ResNet50", "ResNet50"],
            "aggregation": ["majority_vote", "majority_vote", "mean_logits"],
            "accuracy": [0.725, 0.730, 0.740],
        })
        with self.assertRaises(PublicationValidationError) as ctx:
            transform_table_12_long_sequence_consensus(duplicated_agg, {})
        self.assertIn("duplicated variant-aggregation pairs", str(ctx.exception))

    # C9: T9 and F4 triplet mapping and reading frames
    def test_t9_f4_triplet_and_frames(self):
        """C9: transform_table_9_rscu_exploratory and transform_figure_4_rscu_profile map codon to triplet."""
        raw_rscu = pd.DataFrame({
            "codon": ["AAA", "AAC", "AAG"],
            "rscu_frame0": [1.12, 0.85, 1.05],
            "rscu_3frames": [1.08, 0.90, 1.02],
            "rscu_6frames": [1.01, 0.95, 0.99],
        })
        out_t9 = transform_table_9_rscu_exploratory(raw_rscu, {})
        self.assertIn("triplet", out_t9.columns)
        self.assertEqual(out_t9["triplet"].tolist(), ["AAA", "AAC", "AAG"])
        self.assertIn("rscu_frame0", out_t9.columns)
        self.assertIn("rscu_3frames", out_t9.columns)
        self.assertIn("rscu_6frames", out_t9.columns)

        out_f4 = transform_figure_4_rscu_profile(raw_rscu, {})
        self.assertIn("triplet", out_f4.columns)
        self.assertIn("rscu", out_f4.columns)
        self.assertEqual(out_f4["rscu"].tolist(), [1.12, 0.85, 1.05])

    # C10: T10 signal and codon preservation
    def test_t10_signal_codon_preservation(self):
        """C10: transform_table_10_triplet_composition preserves signal, codon, and observed counts."""
        raw_t10 = pd.DataFrame({
            "signal": ["ATG_start", "TAA_stop", "TAG_stop"],
            "codon": ["ATG", "TAA", "TAG"],
            "observed_count": [450, 120, 95],
            "expected_count": [300.0, 100.0, 100.0],
            "fold_enrichment": [1.5, 1.2, 0.95],
        })
        out_t10 = transform_table_10_triplet_composition(raw_t10, {})
        self.assertIn("signal", out_t10.columns)
        self.assertIn("codon", out_t10.columns)
        self.assertEqual(out_t10["signal"].tolist(), ["ATG_start", "TAA_stop", "TAG_stop"])
        self.assertEqual(out_t10["fold_enrichment"].tolist(), [1.5, 1.2, 0.95])

    # C11: F2 coordinates and comments handling
    def test_f2_coordinates_and_comments(self):
        """C11: transform_figure_2_umap_visualization processes 2D coordinates and handles header comments."""
        commented_csv = self.root / "umap_sample.csv"
        commented_csv.write_text(
            "# UMAP 2D projection for ConvNeXt_F\n"
            "# Generated by short/06b\n"
            "umap_1,umap_2,model,genus\n"
            "-1.25,3.40,ConvNeXt_F,Betacoronavirus\n"
            "0.80,-2.15,ConvNeXt_F,Flavivirus\n",
            encoding="utf-8",
        )
        df_read = pd.read_csv(commented_csv, comment="#")
        out_f2 = transform_figure_2_umap_visualization(df_read, {})
        self.assertIn("umap_1", out_f2.columns)
        self.assertIn("umap_2", out_f2.columns)
        self.assertEqual(len(out_f2), 2)
        self.assertEqual(out_f2["model"].tolist(), ["ConvNeXt_F", "ConvNeXt_F"])

        # Rejects missing coordinates
        missing_coord = pd.DataFrame({"x": [1.0], "y": [2.0]})
        with self.assertRaises(PublicationValidationError):
            transform_figure_2_umap_visualization(missing_coord, {})

    # C12: T4, T5, T7, T8 conformity with C01 data contracts
    def test_t4_t5_t7_t8_conformity(self):
        """C12: transform_table_{4,5,7,8} conform to C01 contract schemas."""
        # T4
        t4_df = pd.DataFrame({"f1_band": ["[0.9, 1.0]"], "num_species": [10500], "pct_species": [76.7]})
        out_t4 = transform_table_4_species_f1_and_pairwise(t4_df, {})
        self.assertIn("f1_bin", out_t4.columns)

        # T5
        t5_df = pd.DataFrame({"Model": ["ResNet50"], "KNN_Accuracy": [0.93], "Silhouette": [0.42]})
        out_t5 = transform_table_5_latent_metrics(t5_df, {})
        self.assertIn("model", out_t5.columns)
        self.assertIn("knn_accuracy", out_t5.columns)

        # T7
        t7_df = pd.DataFrame({"model_name": ["ResNet50"], "species_entropy": [1.25], "genus_entropy": [0.85]})
        out_t7 = transform_table_7_hierarchical_entropy(t7_df, {})
        self.assertIn("model", out_t7.columns)

        # T8
        t8_df = pd.DataFrame({"motif": ["CCCCCCC"], "activation_drop_uniform": [0.45], "activation_drop_gc": [0.02]})
        out_t8 = transform_table_8_paired_motif_perturbation(t8_df, {})
        self.assertIn("motif", out_t8.columns)


if __name__ == "__main__":
    unittest.main()
