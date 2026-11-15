"""Unit tests for C01 — Publication Data Contracts and Inventory.

Covers checks C1 through C16 from .specs/features/c01-inventario-contratos-dados/checks.md.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd
import yaml

from viral_cnn.paired_umap import load_projection_coordinates
from viral_cnn.reporting.publication import (
    ITEM_IDS,
    PublicationValidationError,
    diagnose_publication_readiness,
    load_publication_config,
    sha256_file,
)
from viral_cnn.reporting.transformations import (
    execute_publication_transformation,
    get_transformation,
)


class TestC01DataContracts(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        (self.root / ".git").mkdir()
        (self.root / "configs").mkdir()
        (self.root / "evidence").mkdir()
        (self.root / "output").mkdir()

        self.source_txt = self.root / "evidence" / "source.txt"
        self.source_txt.write_text("deterministic source\n", encoding="utf-8")

        self.parent_manifest_path = self.root / "evidence" / "parent.manifest.json"
        self.parent_manifest_data = {
            "protocol": "test_stage_v1",
            "evidence_kind": "empirical",
            "is_smoke_test": False,
            "seed": 42,
            "split_id": "test",
            "scientific_identity": {
                "dataset_sha256": "abc",
                "seed": 42,
                "split_id": "test",
            },
            "inputs": [{"path": str(self.source_txt), "sha256": sha256_file(self.source_txt)}],
        }
        self.parent_manifest_path.write_text(json.dumps(self.parent_manifest_data), encoding="utf-8")

    def tearDown(self):
        self.tmp.cleanup()

    # C1: Matriz completa documentada em docs/publication_data_contracts.md
    def test_contract_matrix_completeness(self):
        matrix_path = Path("docs/publication_data_contracts.md")
        self.assertTrue(matrix_path.is_file(), "docs/publication_data_contracts.md must exist")
        content = matrix_path.read_text(encoding="utf-8")
        for iid in ITEM_IDS:
            self.assertIn(f"**{iid}**", content, f"Item {iid} must be present in publication contracts matrix")
        self.assertIn("short/01/table2_profiling.csv", content)
        self.assertIn("short/02/table7_hierarchical_entropy.csv", content)
        self.assertIn("short/06b/umap_projection_convnext_f.csv", content)
        self.assertIn("long/03/table_12_long_voter_comparison.csv", content)

    # C2: Fonte proposta para T2 em configs/reproduction_m07.yaml
    def test_t2_proposed_source_path(self):
        cfg = load_publication_config(Path("configs/reproduction_m07.yaml"))
        item_t2 = next(it for it in cfg["items"] if it["item_id"] == "T2")
        expected_substr = "short/01/table2_profiling.csv"
        self.assertIn(expected_substr, item_t2["proposed_source"])

    # C3: Fonte proposta para T7 em configs/reproduction_m07.yaml
    def test_t7_proposed_source_path(self):
        cfg = load_publication_config(Path("configs/reproduction_m07.yaml"))
        item_t7 = next(it for it in cfg["items"] if it["item_id"] == "T7")
        expected_substr = "short/02/table7_hierarchical_entropy.csv"
        self.assertIn(expected_substr, item_t7["proposed_source"])

    # C4: Fonte proposta para F2 em configs/reproduction_m07.yaml
    def test_f2_proposed_source_path(self):
        cfg = load_publication_config(Path("configs/reproduction_m07.yaml"))
        item_f2 = next(it for it in cfg["items"] if it["item_id"] == "F2")
        expected_substr = "short/06b/umap_projection_convnext_f.csv"
        self.assertIn(expected_substr, item_f2["proposed_source"])

    # C5: Colunas esperadas para T11 não vazias em configs/reproduction_m07.yaml
    def test_t11_expected_columns_non_empty(self):
        cfg = load_publication_config(Path("configs/reproduction_m07.yaml"))
        item_t11 = next(it for it in cfg["items"] if it["item_id"] == "T11")
        cols = item_t11.get("expected", {}).get("columns", [])
        self.assertGreater(len(cols), 0, "T11 expected columns must not be empty")
        for req in ("status", "p_permutation_null", "r_canonical"):
            self.assertIn(req, cols, f"T11 expected columns must include {req}")

    # C6: T1 rejeita esquemas sem medições obrigatórias sem preencher com 0.0
    def test_t1_rejects_missing_without_zero_fill(self):
        func = get_transformation("T1")
        # Missing parameters and latency
        df_incomplete = pd.DataFrame({
            "model": ["ResNet50"],
            "taxonomic_level": ["Species"],
            "accuracy": [85.0],
            "top_5": [96.0],
            "macro_f1": [84.0],
        })
        with self.assertRaises(PublicationValidationError) as ctx:
            func(df_incomplete, {})
        self.assertIn("Zero-filling is forbidden", str(ctx.exception))

        # Complete DataFrame passes and does not modify values
        df_complete = pd.DataFrame({
            "model": ["ResNet50"],
            "taxonomic_level": ["Species"],
            "backbone_m": [15.2],
            "fc_params_m": [28.0],
            "parameters_m": [43.2],
            "accuracy": [85.0],
            "top_5": [96.0],
            "macro_f1": [84.0],
            "latency_ms": [1.45],
        })
        res = func(df_complete, {})
        self.assertEqual(res.loc[0, "latency_ms"], 1.45)

    # C7: T2 mapeia colunas reais do profiler sem preenchimento artificial por zero
    def test_t2_maps_profiling_columns_without_zero_fill(self):
        func = get_transformation("T2")
        # Real profiling dataframe output from short/01
        real_profiling_df = pd.DataFrame({
            "model_name": ["resnet_clean_species"],
            "total_params": [43200000],
            "backbone_params": [15200000],
            "head_params": [28000000],
            "trainable_params": [43200000],
            "size_mb": [172.5],
            "gflops": [2.5],
            "avg_latency_ms": [1.45],
            "std_latency_ms": [0.05],
            "throughput_seq_sec": [22000.0],
        })
        res = func(real_profiling_df, {})
        self.assertEqual(res.loc[0, "model"], "resnet_clean_species")
        self.assertEqual(res.loc[0, "backbone_m"], 15.2)
        self.assertEqual(res.loc[0, "fc_params_m"], 28.0)
        self.assertEqual(res.loc[0, "parameters_m"], 43.2)
        self.assertEqual(res.loc[0, "step_time_ms"], 1.45)
        self.assertEqual(res.loc[0, "throughput_seq_s"], 22000.0)

        # Incomplete profiling dataframe raises PublicationValidationError
        df_incomplete = pd.DataFrame({
            "model": ["ResNet50"],
            "backbone_m": [15.0],
        })
        with self.assertRaises(PublicationValidationError) as ctx:
            func(df_incomplete, {})
        self.assertIn("Zero-filling is forbidden", str(ctx.exception))

    # C8: T3 e F1 convertem noise_rate para epsilon
    def test_t3_f1_converts_noise_rate_to_epsilon(self):
        func_t3 = get_transformation("T3")
        func_f1 = get_transformation("F1")
        raw_df = pd.DataFrame({
            "method": ["ResNet50", "ConvNeXt"],
            "noise_rate": [0.0, 0.05],
            "stratum": ["all", "all"],
            "accuracy": [85.0, 80.0],
            "n": [1706270, 1706270],
        })
        res_t3 = func_t3(raw_df, {})
        self.assertIn("epsilon", res_t3.columns)
        self.assertEqual(res_t3.loc[1, "epsilon"], 0.05)

        res_f1 = func_f1(raw_df, {})
        self.assertIn("epsilon", res_f1.columns)
        self.assertEqual(res_f1.loc[1, "epsilon"], 0.05)

    # C9: T6 converte Transition para level_pair
    def test_t6_converts_transition_to_level_pair(self):
        func_t6 = get_transformation("T6")
        raw_t6_df = pd.DataFrame({
            "Model": ["ResNet50"],
            "Transition": ["Species -> Genus"],
            "Pearson_r": [0.82],
            "Spearman_rho": [0.79],
            "N_taxa": [450],
        })
        res = func_t6(raw_t6_df, {})
        self.assertIn("level_pair", res.columns)
        self.assertEqual(res.loc[0, "level_pair"], "Species -> Genus")
        self.assertEqual(res.loc[0, "pearson_r"], 0.82)
        self.assertEqual(res.loc[0, "model"], "ResNet50")

    # C10: T9 e F4 mapeiam codon para triplet
    def test_t9_maps_codon_to_triplet_and_frames(self):
        func_t9 = get_transformation("T9")
        func_f4 = get_transformation("F4")
        raw_df = pd.DataFrame({
            "codon": ["AAA", "AAC"],
            "rscu_frame0": [1.1, 0.9],
            "rscu_forward_3frames": [1.05, 0.95],
            "rscu_six_frames": [1.02, 0.98],
            "multinomial_null": [0.015, 0.015],
        })
        res_t9 = func_t9(raw_df, {})
        self.assertIn("triplet", res_t9.columns)
        self.assertEqual(res_t9.loc[0, "triplet"], "AAA")
        self.assertEqual(res_t9.loc[0, "rscu_frame0"], 1.1)

        res_f4 = func_f4(raw_df, {})
        self.assertIn("triplet", res_f4.columns)
        self.assertIn("rscu", res_f4.columns)
        self.assertEqual(res_f4.loc[0, "rscu"], 1.1)

    # C11: T10 preserva signal e codon
    def test_t10_preserves_signal_and_codon(self):
        func_t10 = get_transformation("T10")
        raw_t10_df = pd.DataFrame({
            "signal": ["start_codon", "stop_codon"],
            "codon": ["ATG", "TAA"],
            "observed_count": [120, 95],
            "expected_count": [50.0, 48.0],
            "fold_enrichment": [2.4, 1.98],
            "null_probability": [0.0001, 0.0002],
        })
        res = func_t10(raw_t10_df, {})
        self.assertIn("signal", res.columns)
        self.assertIn("codon", res.columns)
        self.assertEqual(res.loc[0, "signal"], "start_codon")
        self.assertEqual(res.loc[0, "observed_count"], 120)

    # C12: T12 pivota variant x aggregation
    def test_t12_pivots_variant_aggregation_rows(self):
        func_t12 = get_transformation("T12")
        # Output from build_c02_table12
        raw_t12_df = pd.DataFrame({
            "variant": ["resnet_clean_species", "resnet_clean_species", "convnext_zero_species", "convnext_zero_species"],
            "aggregation": ["mean_logits", "majority_vote", "mean_logits", "majority_vote"],
            "N": [4514, 4514, 4514, 4514],
            "accuracy": [76.5, 75.8, 62.1, 60.4],
            "coverage": [1.0, 1.0, 1.0, 1.0],
        })
        res = func_t12(raw_t12_df, {})
        self.assertEqual(len(res), 2, "Pivoted Table 12 must have 1 row per model")
        self.assertIn("majority_vote_acc", res.columns)
        self.assertIn("mean_logits_acc", res.columns)
        row_resnet = res[res["model"] == "resnet_clean_species"].iloc[0]
        self.assertEqual(row_resnet["mean_logits_acc"], 76.5)
        self.assertEqual(row_resnet["majority_vote_acc"], 75.8)

    # C13: F2 lê CSV com comentários # no cabeçalho
    def test_f2_parses_commented_projection_csv(self):
        f2_path = self.root / "evidence" / "commented_f2.csv"
        content = (
            "# QUALITATIVE ONLY: independently fitted 2D projections; axes are not aligned.\n"
            "# Do not infer quantitative cluster separation or phylogenetic distance.\n"
            "umap_1,umap_2,label,row_id,Genus,projection_method,is_qualitative_only\n"
            "0.123,0.456,1,row_001,Flavivirus,UMAP,True\n"
            "0.789,0.012,2,row_002,Betacoronavirus,UMAP,True\n"
        )
        f2_path.write_text(content, encoding="utf-8")

        df = load_projection_coordinates(f2_path)
        self.assertEqual(len(df), 2)
        self.assertIn("umap_1", df.columns)
        self.assertIn("umap_2", df.columns)
        self.assertEqual(df.loc[0, "Genus"], "Flavivirus")

        # Also test via execute_publication_transformation
        out_csv, out_manifest = execute_publication_transformation(
            "F2",
            f2_path,
            self.root / "output",
            self.parent_manifest_path,
        )
        self.assertTrue(out_csv.is_file())
        transformed = pd.read_csv(out_csv)
        self.assertEqual(len(transformed), 2)

    # C14: F2 rejeita tabelas sem coordenadas com PublicationValidationError
    def test_f2_rejects_missing_coordinates(self):
        f2_bad_path = self.root / "evidence" / "bad_f2.csv"
        f2_bad_path.write_text("genus,count\nFlavivirus,100\n", encoding="utf-8")
        with self.assertRaises(PublicationValidationError):
            load_projection_coordinates(f2_bad_path)

        func_f2 = get_transformation("F2")
        with self.assertRaises(PublicationValidationError):
            func_f2(pd.DataFrame({"x": [1.0], "y": [2.0]}), {})

    # C15: diagnose_publication_readiness classifica colunas incompatíveis como SCHEMA_MISMATCH
    def test_diagnose_readiness_schema_mismatch(self):
        # Create an artifact with wrong columns
        art_path = self.root / "evidence" / "mismatched_t1.csv"
        pd.DataFrame({"wrong_column": [1, 2]}).to_csv(art_path, index=False)

        manifest_path = self.root / "evidence" / "mismatched_t1.manifest.json"
        manifest_data = {
            "protocol": "test_stage_v1",
            "evidence_kind": "empirical",
            "is_smoke_test": False,
            "seed": 42,
            "split_id": "test",
            "scientific_identity": {"dataset_sha256": "abc", "seed": 42, "split_id": "test"},
            "artifact_sha256": sha256_file(art_path),
            "inputs": [{"path": str(self.source_txt), "sha256": sha256_file(self.source_txt)}],
        }
        manifest_path.write_text(json.dumps(manifest_data), encoding="utf-8")

        cfg = {
            "publication_protocol": "c06_strict_v1",
            "schema_version": 1,
            "run_id": "test-diag",
            "items": [
                {
                    "item_id": "T1",
                    "artifact": str(art_path),
                    "manifest": str(manifest_path),
                    "sha256": sha256_file(art_path),
                    "expected": {
                        "columns": ["model", "taxonomic_level", "accuracy"],
                        "scientific_identity": {"dataset_sha256": "abc", "seed": 42, "split_id": "test"},
                    },
                }
            ],
        }
        report = diagnose_publication_readiness(cfg, project_root=self.root)
        self.assertEqual(report.loc[0, "status"], "SCHEMA_MISMATCH")
        self.assertIn("Missing expected table columns", report.loc[0, "diagnostic"])

    # C16: diagnose_publication_readiness classifica arquivo ausente como FALTA
    def test_diagnose_readiness_falta(self):
        cfg = {
            "publication_protocol": "c06_strict_v1",
            "schema_version": 1,
            "run_id": "test-diag",
            "items": [
                {
                    "item_id": "T2",
                    "artifact": str(self.root / "nonexistent" / "t2.csv"),
                    "manifest": str(self.root / "nonexistent" / "t2.manifest.json"),
                }
            ],
        }
        report = diagnose_publication_readiness(cfg, project_root=self.root)
        self.assertEqual(report.loc[0, "status"], "FALTA")
        self.assertIn("artifact missing", report.loc[0, "diagnostic"])


if __name__ == "__main__":
    unittest.main()
