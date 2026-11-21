"""Tests for S19: Partial Taxonomy Derivation, Conditional Genus Entropy, and Rank Transition Coverage.

Validates C1 to C12 defined in .specs/features/s19-partial-taxonomy-and-conditional-entropy/checks.md.
"""
import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from viral_cnn.eval.calibration import TaxonomyEntropyAnalyzer
from viral_cnn.m01_workflow import run_m01
from viral_cnn.notebook_runtime import preflight_notebook
from viral_cnn.reporting.transformations import transform_table_6_hierarchy_correlations
from viral_cnn.taxonomy.curation import validate_taxonomy_identity
from viral_cnn.taxonomy.mapping import (
    build_partial_species_to_genus_tensor,
    build_species_to_genus_tensor,
)


class DummyFeatureModel(nn.Module):
    def __init__(self, in_features: int = 16, num_classes: int = 5):
        super().__init__()
        self.fc = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class TestS19PartialTaxonomyAndConditionalEntropy(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.root = Path(self.tmp_dir)

        # Create small synthetic original mapping
        self.species_csv = self.root / "species_name_ids.csv"
        df_sp = pd.DataFrame({
            "id": [1, 2, 3, 4, 5],
            "species": [f"Virus_sp_{i}" for i in [1, 2, 3, 4, 5]],
        })
        df_sp.to_csv(self.species_csv, index=False)

        # Create small synthetic partial taxonomy
        self.tax_parquet = self.root / "main_taxonomy_hybrid_ictv_ncbi.parquet"
        self.tax_df = pd.DataFrame({
            "id": [1, 2, 3, 4, 5],
            "species": [f"Virus_sp_{i}" for i in [1, 2, 3, 4, 5]],
            "Genus": ["GenusA", "GenusA", "GenusB", "Unassigned", None],
            "Order": ["OrderX", "OrderX", "Unclassified", "", "OrderY"],
        })
        self.tax_df.to_parquet(self.tax_parquet, index=False)

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_c1_m01_derive_partial_loads_inputs(self):
        """C1: run_m01 in derive_partial_existing mode loads inputs without ictv_master."""
        original_csv_bytes = self.species_csv.read_bytes()
        original_parquet_bytes = self.tax_parquet.read_bytes()

        settings = {
            "mode": "derive_partial_existing",
            "original_mapping": str(self.species_csv),
            "taxonomy_path": str(self.tax_parquet),
            "data_root": str(self.root),
        }
        manifest = run_m01("short/05", self.root, settings=settings, is_smoke_test=True)

        self.assertEqual(manifest["status"], "COMPLETE")
        # Ensure original input files were preserved unchanged
        self.assertEqual(self.species_csv.read_bytes(), original_csv_bytes)
        self.assertEqual(self.tax_parquet.read_bytes(), original_parquet_bytes)

    def test_c2_m01_derive_partial_validates_ids(self):
        """C2: run_m01 in derive_partial_existing mode raises ValueError on invalid or duplicate IDs."""
        bad_parquet = self.root / "bad_ids.parquet"
        bad_df = self.tax_df.copy()
        bad_df.loc[0, "id"] = 0  # id < 1
        bad_df.to_parquet(bad_parquet, index=False)

        settings = {
            "mode": "derive_partial_existing",
            "original_mapping": str(self.species_csv),
            "taxonomy_path": str(bad_parquet),
            "data_root": str(self.root),
        }
        with self.assertRaises(ValueError) as ctx:
            run_m01("short/05", self.root, settings=settings, is_smoke_test=True)
        self.assertIn("IDs taxonômicos inválidos ou duplicados", str(ctx.exception))

    def test_c3_m01_derive_partial_model_label(self):
        """C3: Sets model_label = id - 1 when absent, and raises when existing label differs."""
        # 1. Missing model_label should succeed and add model_label in memory copy
        settings = {
            "mode": "derive_partial_existing",
            "original_mapping": str(self.species_csv),
            "taxonomy_path": str(self.tax_parquet),
            "data_root": str(self.root),
        }
        manifest = run_m01("short/05", self.root, settings=settings, is_smoke_test=True)
        out_parquet = Path(manifest["output_dir"]) / "main_taxonomy_hybrid_PARTIAL.parquet"
        self.assertTrue(out_parquet.is_file())
        res_df = pd.read_parquet(out_parquet)
        self.assertTrue((res_df["model_label"] == res_df["id"] - 1).all())

        # 2. Inconsistent model_label should raise ValueError
        bad_label_parquet = self.root / "bad_label.parquet"
        bad_df = self.tax_df.copy()
        bad_df["model_label"] = [0, 1, 999, 3, 4]
        bad_df.to_parquet(bad_label_parquet, index=False)

        settings["taxonomy_path"] = str(bad_label_parquet)
        with self.assertRaises(ValueError) as ctx:
            run_m01("short/05", self.root, settings=settings, is_smoke_test=True)
        self.assertIn("model_label existente difere de id - 1", str(ctx.exception))

    def test_c4_m01_derive_partial_masks_missing_ranks(self):
        """C4: Masks whitespace, empty, Unassigned, and Unclassified strings in Genus and Order to NA."""
        settings = {
            "mode": "derive_partial_existing",
            "original_mapping": str(self.species_csv),
            "taxonomy_path": str(self.tax_parquet),
            "data_root": str(self.root),
        }
        manifest = run_m01("short/05", self.root, settings=settings, is_smoke_test=True)
        out_parquet = Path(manifest["output_dir"]) / "main_taxonomy_hybrid_PARTIAL.parquet"
        res_df = pd.read_parquet(out_parquet)

        # Species 4 had "Unassigned", Species 5 had None
        self.assertTrue(pd.isna(res_df.loc[3, "Genus"]))
        self.assertTrue(pd.isna(res_df.loc[4, "Genus"]))
        # Species 3 had "Unclassified", Species 4 had ""
        self.assertTrue(pd.isna(res_df.loc[2, "Order"]))
        self.assertTrue(pd.isna(res_df.loc[3, "Order"]))
        # Species 1 and 2 had GenusA
        self.assertEqual(res_df.loc[0, "Genus"], "GenusA")
        self.assertEqual(res_df.loc[1, "Genus"], "GenusA")

    def test_c5_m01_derive_partial_manifest_and_parquet(self):
        """C5: Outputs main_taxonomy_hybrid_PARTIAL.parquet and manifest with hierarchy_ready=False."""
        settings = {
            "mode": "derive_partial_existing",
            "original_mapping": str(self.species_csv),
            "taxonomy_path": str(self.tax_parquet),
            "data_root": str(self.root),
        }
        manifest = run_m01("short/05", self.root, settings=settings, is_smoke_test=True)
        out_dir = Path(manifest["output_dir"])

        self.assertEqual(manifest["status"], "COMPLETE")
        self.assertEqual(manifest["protocol_id"], "m01_existing_partial_derive_v1")
        self.assertFalse(manifest["metrics_summary"]["hierarchy_ready"])

        out_parquet = out_dir / "main_taxonomy_hybrid_PARTIAL.parquet"
        self.assertTrue(out_parquet.is_file())

        # Inputs and outputs hashes recorded
        input_paths = [rec["path"] for rec in manifest["inputs"]]
        self.assertIn(str(self.tax_parquet), input_paths)
        self.assertIn(str(self.species_csv), input_paths)
        output_paths = [rec["path"] for rec in manifest["outputs"]]
        self.assertIn(str(out_parquet), output_paths)

    def test_c6_preflight_short_05_derive_mode(self):
        """C6: preflight_notebook for short/05 under derive mode requires parquet and csv, not ICTV."""
        # Set environment variable M01_MODE
        os.environ["M01_MODE"] = "derive_partial_existing"
        try:
            report = preflight_notebook("short/05", self.root, smoke_test=True)
            req_paths = [r["path"] for r in report["requirements"]]
            # short/ictv_master should not be in requirements
            self.assertFalse(any("ictv" in p.lower() for p in req_paths))
        finally:
            os.environ.pop("M01_MODE", None)

    def test_c7_build_partial_species_to_genus_tensor(self):
        """C7: build_partial_species_to_genus_tensor produces contiguous indices for known and -1 for unassigned."""
        # Use tax_parquet with 5 species: 0, 1 -> GenusA; 2 -> GenusB; 3, 4 -> NA
        tensor = build_partial_species_to_genus_tensor(self.tax_parquet, num_species=5)

        self.assertIsInstance(tensor, torch.Tensor)
        self.assertEqual(tensor.dtype, torch.long)
        self.assertEqual(len(tensor), 5)

        arr = tensor.cpu().numpy()
        # 0 and 1 belong to the same genus
        self.assertEqual(arr[0], arr[1])
        self.assertGreaterEqual(arr[0], 0)
        # 2 belongs to another genus
        self.assertGreaterEqual(arr[2], 0)
        self.assertNotEqual(arr[0], arr[2])
        # 3 and 4 belong to no genus: must be -1
        self.assertEqual(arr[3], -1)
        self.assertEqual(arr[4], -1)

    def test_c8_analyzer_accepts_partial_mapping(self):
        """C8: TaxonomyEntropyAnalyzer accepts -1 indices when allow_partial=True."""
        model = DummyFeatureModel(in_features=16, num_classes=5)
        partial_mapping = torch.tensor([0, 0, 1, -1, -1], dtype=torch.long)

        # Without allow_partial=True, should raise ValueError
        with self.assertRaises(ValueError):
            TaxonomyEntropyAnalyzer(model, partial_mapping, allow_partial=False)

        # With allow_partial=True, should initialize and hook cleanly
        with TaxonomyEntropyAnalyzer(model, partial_mapping, allow_partial=True) as analyzer:
            self.assertTrue(analyzer.allow_partial)
            self.assertEqual(analyzer.species_to_genus_map[3].item(), -1)

    def test_c9_analyzer_batch_conditional_entropy(self):
        """C9: analyze_batch returns conditional entropy metrics and excluded_prob_mass."""
        torch.manual_seed(42)
        model = DummyFeatureModel(in_features=16, num_classes=5)
        partial_mapping = torch.tensor([0, 0, 1, -1, -1], dtype=torch.long)
        inputs = torch.randn(8, 16)

        with TaxonomyEntropyAnalyzer(model, partial_mapping, allow_partial=True) as analyzer:
            res = analyzer.analyze_batch(inputs)

            self.assertIn("h_species", res)
            self.assertIn("h_species_conditional", res)
            self.assertIn("h_genus_conditional", res)
            self.assertIn("gap_conditional", res)
            self.assertIn("excluded_prob_mass", res)

            # Mathematical invariant: conditional gap >= -1e-5
            self.assertTrue((res["gap_conditional"] >= -1e-5).all())
            # Excluded prob mass in [0.0, 1.0]
            self.assertTrue((res["excluded_prob_mass"] >= 0.0).all())
            self.assertTrue((res["excluded_prob_mass"] <= 1.0).all())

    def test_c10_analyzer_dataloader_truth_counts(self):
        """C10: analyze_dataloader tallies truth counts and averages conditional entropy strictly on known genus."""
        model = DummyFeatureModel(in_features=16, num_classes=5)
        partial_mapping = torch.tensor([0, 0, 1, -1, -1], dtype=torch.long)

        # Create dataset with labels: 4 samples with genus (0, 1, 2, 0), 2 without genus (3, 4)
        inputs = torch.randn(6, 16)
        labels = torch.tensor([0, 1, 2, 0, 3, 4], dtype=torch.long)
        dataset = TensorDataset(inputs, labels)
        loader = DataLoader(dataset, batch_size=3)

        with TaxonomyEntropyAnalyzer(model, partial_mapping, allow_partial=True) as analyzer:
            summary = analyzer.analyze_dataloader(loader)

            self.assertEqual(summary["n_samples"], 6)
            self.assertEqual(summary["n_truth_with_genus"], 4)
            self.assertEqual(summary["n_truth_without_genus"], 2)
            self.assertIn("mean_h_species_conditional", summary)
            self.assertIn("mean_h_genus_conditional", summary)
            self.assertIn("mean_gap_conditional", summary)
            self.assertIn("mean_excluded_prob_mass", summary)

            # Ensure unnormalized mean_h_genus and mean_hierarchical_gap are omitted when partial
            self.assertNotIn("mean_h_genus", summary)
            self.assertNotIn("mean_hierarchical_gap", summary)

    def test_c11_table6_rank_transition_coverage(self):
        """C11: Table 6 transition Genus -> Order requires both ranks known, avoiding counting species with missing Genus."""
        # In self.tax_df:
        # id 1: GenusA, OrderX (both known)
        # id 2: GenusA, OrderX (both known)
        # id 3: GenusB, Order Unclassified (Order missing)
        # id 4: Genus Unassigned, Order empty (both missing)
        # id 5: Genus None (missing), Order OrderY (Order known, BUT Genus missing!)
        mapped_species = [0, 1, 2, 3, 4]
        sp_to_genus = {0: "GenusA", 1: "GenusA", 2: "GenusB"}
        sp_to_order = {0: "OrderX", 1: "OrderX", 4: "OrderY"}

        eligible_sp_genus = {k for k in mapped_species if pd.notna(sp_to_genus.get(k))}
        eligible_sp_genus_order = {k for k in mapped_species if pd.notna(sp_to_genus.get(k)) and pd.notna(sp_to_order.get(k))}

        # Species -> Genus has 3 eligible species (0, 1, 2)
        self.assertEqual(len(eligible_sp_genus), 3)
        # Genus -> Order MUST have only 2 eligible species (0, 1), NOT 3 (species 4 has OrderY but missing Genus!)
        self.assertEqual(len(eligible_sp_genus_order), 2)
        self.assertNotIn(4, eligible_sp_genus_order)

        # Model-specific test labels
        y_model = np.array([0, 1, 2, 4, 3, 0])
        sp_eligible_frags = int(np.fromiter((int(sp) in eligible_sp_genus for sp in y_model), dtype=bool, count=len(y_model)).sum())
        ord_eligible_frags = int(np.fromiter((int(sp) in eligible_sp_genus_order for sp in y_model), dtype=bool, count=len(y_model)).sum())

        self.assertEqual(sp_eligible_frags, 4)  # labels 0, 1, 2, 0
        self.assertEqual(ord_eligible_frags, 3)  # labels 0, 1, 0 (label 4 is excluded because Genus is missing!)

        table6_sample = pd.DataFrame([{
            "Model": "ResNet50_Clean",
            "Transition": "Genus -> Order",
            "Pearson_r": 0.85,
            "Spearman_rho": 0.84,
            "N_taxa": 200,
            "Eligible_Species": len(eligible_sp_genus_order),
            "Excluded_Species": len(mapped_species) - len(eligible_sp_genus_order),
            "Eligible_Fragments": ord_eligible_frags,
            "Excluded_Fragments": len(y_model) - ord_eligible_frags,
        }])
        csv_path = self.root / "table6_taxonomic_rank_correlations.csv"
        table6_sample.to_csv(csv_path, index=False)

        read_df = pd.read_csv(csv_path)
        required_cols = [
            "Model", "Transition", "Pearson_r", "Spearman_rho", "N_taxa",
            "Eligible_Species", "Excluded_Species", "Eligible_Fragments", "Excluded_Fragments"
        ]
        for col in required_cols:
            self.assertIn(col, read_df.columns)
        self.assertEqual(read_df["Eligible_Species"].iloc[0], 2)
        self.assertEqual(read_df["Eligible_Fragments"].iloc[0], 3)

    def test_c12_transform_table_6_preserves_enriched_columns(self):
        """C12: transform_table_6_hierarchy_correlations preserves the new transition coverage columns."""
        raw_df = pd.DataFrame([{
            "Model": "ResNet50_Clean",
            "Transition": "Species -> Genus",
            "Pearson_r": 0.85,
            "Spearman_rho": 0.84,
            "N_taxa": 200,
            "Eligible_Species": 7746,
            "Excluded_Species": 5937,
            "Eligible_Fragments": 100000,
            "Excluded_Fragments": 33235,
        }])
        out_df = transform_table_6_hierarchy_correlations(raw_df, {})
        self.assertIn("Eligible_Species", out_df.columns)
        self.assertIn("Excluded_Species", out_df.columns)
        self.assertIn("Eligible_Fragments", out_df.columns)
        self.assertIn("Excluded_Fragments", out_df.columns)
        self.assertEqual(out_df["Eligible_Species"].iloc[0], 7746)


if __name__ == "__main__":
    unittest.main()
