"""Unit and integration tests for M07 publication contracts and local validation.

Maps directly to checks C1 through C16 in .specs/features/m07-publication-contracts-local-validation/checks.md.
"""

from __future__ import annotations

import io
import json
import os
import shutil
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import yaml

from viral_cnn.reporting.publication import (
    ITEM_IDS,
    PublicationValidationError,
    assemble_publication,
    diagnose_publication_readiness,
    load_publication_config,
    sha256_file,
)
from viral_cnn.reporting.transformations import (
    FIVE_VARIANTS_SPECIES,
    TRANSFORMATION_REGISTRY,
    emit_derived_manifest,
    execute_publication_transformation,
    get_transformation,
    register_transformation,
    validate_artifact_is_tabular,
)


class M07Tests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        (self.root / ".git").mkdir()
        (self.root / "configs").mkdir()
        (self.root / "evidence").mkdir()
        (self.root / "runs").mkdir(parents=True)

        self.source_txt = self.root / "evidence" / "source.txt"
        self.source_txt.write_text("dummy source\n", encoding="utf-8")

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

    def _create_minimal_valid_config(self, *, make_files: bool = True) -> tuple[Path, dict]:
        items = []
        for iid in ITEM_IDS:
            artifact_path = self.root / "evidence" / f"{iid}.csv"
            manifest_path = self.root / "evidence" / f"{iid}.manifest.json"

            if make_files:
                if iid == "F3":
                    df = pd.DataFrame([
                        {"panel": p, "variant": v, "motif": "ACGTACG", "activation_drop": 0.5}
                        for p, v in zip(["a", "b", "c", "d", "e"], FIVE_VARIANTS_SPECIES)
                    ])
                elif iid == "F5":
                    df = pd.DataFrame({"x1": [0.0], "y1": [0.0], "x2": [1.0], "y2": [1.0], "jaccard": [0.5]})
                elif iid == "T1":
                    df = pd.DataFrame({
                        "model": ["ResNet50"], "taxonomic_level": ["Species"], "accuracy": [85.0],
                        "top_5": [96.0], "macro_f1": [84.0], "backbone_m": [15.0], "fc_params_m": [28.0],
                        "parameters_m": [43.0], "latency_ms": [1.5],
                    })
                elif iid == "T11":
                    df = pd.DataFrame({
                        "model_pair": ["M1_M2"], "jaccard_overlap": [0.4], "edge_count": [10],
                        "permutation_p_value": [0.01], "is_permutation_tested": [True],
                    })
                elif iid == "T12":
                    df = pd.DataFrame({
                        "model": ["ResNet50"], "majority_vote_acc": [75.0], "mean_logits_acc": [76.0],
                        "coverage": [100.0],
                    })
                else:
                    df = pd.DataFrame({"x": [1, 2], "y": [3, 4], "epsilon": [0.0, 0.05], "accuracy": [80.0, 75.0],
                                       "f1_bin": ["0.9-1.0", "0.8-0.9"], "knn_accuracy": [80.0, 80.0],
                                       "level_pair": ["sp_gn", "gn_ord"], "model": ["ResNet50", "ConvNeXt"],
                                       "motif": ["MOTIF1", "MOTIF2"], "triplet": ["AAA", "AAC"],
                                       "rscu": [1.1, 0.9], "umap_1": [0.1, 0.2], "umap_2": [0.3, 0.4]})
                df.to_csv(artifact_path, index=False)

                manifest = {
                    "protocol": "c06_fixture_v1",
                    "status": "COMPLETE",
                    "evidence_kind": "empirical",
                    "is_smoke_test": False,
                    "seed": 42,
                    "split_id": "test",
                    "artifact_sha256": sha256_file(artifact_path),
                    "inputs": [{"path": str(self.source_txt), "sha256": sha256_file(self.source_txt)}],
                    "scientific_identity": {
                        "dataset_sha256": "abc",
                        "seed": 42,
                        "split_id": "test",
                    },
                }
                manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            plot = {"x": "epsilon" if iid == "F1" else "umap_1", "y": "accuracy" if iid == "F1" else "umap_2", "title": iid}
            if iid == "F3":
                plot = {"kind": "panel_grid", "x": "motif", "y": "activation_drop", "title": iid}
            elif iid == "F5":
                plot = {"kind": "network_edges", "x1": "x1", "y1": "y1", "x2": "x2", "y2": "y2", "title": iid}

            items.append({
                "item_id": iid,
                "artifact": str(artifact_path) if make_files else "",
                "manifest": str(manifest_path) if make_files else "",
                "sha256": sha256_file(artifact_path) if make_files else "",
                "protocol": "c06_fixture_v1",
                "transformation": f"transform_{iid.lower()}",
                "expected": {
                    "scientific_identity": {
                        "dataset_sha256": "abc",
                        "seed": 42,
                        "split_id": "test",
                    },
                },
                "output": f"{iid}.csv",
                "plot": plot,
            })

        cfg_data = {
            "publication_protocol": "c06_strict_v1",
            "schema_version": 1,
            "run_id": "test-run",
            "historical_source": {},
            "items": items,
        }
        cfg_path = self.root / "configs" / "publication_test.yaml"
        cfg_path.write_text(yaml.safe_dump(cfg_data), encoding="utf-8")
        return cfg_path, cfg_data

    # C1: Centralized registry of transformations
    def test_transformation_registry(self):
        for item_id in ITEM_IDS:
            self.assertIn(item_id, TRANSFORMATION_REGISTRY, f"Item {item_id} must have a registered transformation")
            func = get_transformation(item_id)
            self.assertTrue(callable(func), f"Transformation for {item_id} must be callable")

    # C2: Table transformations execution
    def test_table_transformations_execution(self):
        for i in range(1, 13):
            item_id = f"T{i}"
            func = get_transformation(item_id)
            # Create synthetic input df
            if item_id == "T1":
                df_in = pd.DataFrame({"model": ["M1"], "taxonomic_level": ["Species"], "accuracy": [90.0], "top_5": [98.0], "macro_f1": [89.0]})
            elif item_id == "T2":
                df_in = pd.DataFrame({"model": ["M1"], "backbone_m": [15.0], "fc_params_m": [28.0], "parameters_m": [43.0]})
            elif item_id == "T3":
                df_in = pd.DataFrame({"epsilon": [0.0, 0.05], "acc": [90.0, 85.0]})
            elif item_id == "T4":
                df_in = pd.DataFrame({"f1_bin": ["0.9-1.0"], "count": [100]})
            elif item_id == "T5":
                df_in = pd.DataFrame({"model": ["M1"], "knn_accuracy": [85.0]})
            elif item_id == "T6":
                df_in = pd.DataFrame({"level_pair": ["Species-Genus"], "r": [0.8]})
            elif item_id == "T7":
                df_in = pd.DataFrame({"model": ["M1"], "h_act": [1.2]})
            elif item_id == "T8":
                df_in = pd.DataFrame({"motif": ["AAAAAAA"], "drop": [0.3]})
            elif item_id == "T9":
                df_in = pd.DataFrame({"triplet": ["AAA"], "rscu": [1.0]})
            elif item_id == "T10":
                df_in = pd.DataFrame({"triplet": ["AAA"], "observed": [0.01]})
            elif item_id == "T11":
                df_in = pd.DataFrame({"model_pair": ["M1-M2"], "jaccard": [0.5], "is_permutation_tested": [True]})
            elif item_id == "T12":
                df_in = pd.DataFrame({"model": ["M1"], "majority_vote_acc": [70.0], "mean_logits_acc": [71.0]})

            transformed = func(df_in, {})
            self.assertIsInstance(transformed, pd.DataFrame)
            self.assertGreaterEqual(len(transformed), 1)

    # C3: Derived manifest emission
    def test_derived_manifest_emission(self):
        source_df = pd.DataFrame({"model": ["ResNet50"], "taxonomic_level": ["Species"], "accuracy": [85.0], "top_5": [95.0], "macro_f1": [84.0]})
        src_path = self.root / "evidence" / "raw_t1.csv"
        source_df.to_csv(src_path, index=False)

        out_csv, out_manifest = execute_publication_transformation(
            "T1",
            src_path,
            self.root / "output",
            self.parent_manifest_path,
        )
        self.assertTrue(out_csv.is_file(), "Transformed CSV must exist")
        self.assertTrue(out_manifest.is_file(), "Derived manifest must exist")

        data = json.loads(out_manifest.read_text(encoding="utf-8"))
        self.assertEqual(data["evidence_kind"], "derived", "Manifest evidence_kind must be 'derived'")
        self.assertEqual(data["transformation"], "transform_table_1_metrics")
        self.assertEqual(data["n"], 1)
        self.assertEqual(data["support"], 1)
        self.assertEqual(data["parent_manifest_sha256"], sha256_file(self.parent_manifest_path))

    # C4: Rejection of PNG source
    def test_reject_png_source(self):
        png_path = self.root / "evidence" / "test_image.png"
        png_path.write_bytes(b"\x89PNG\r\n\x1a\nfake-png-bytes")

        with self.assertRaises(PublicationValidationError) as ctx:
            validate_artifact_is_tabular(png_path)
        self.assertIn("Image file (.png) cannot be used as a publication source", str(ctx.exception))

        with self.assertRaises(PublicationValidationError) as ctx2:
            execute_publication_transformation("T1", png_path, self.root / "out", self.parent_manifest_path)
        self.assertIn("Image file", str(ctx2.exception))

    # C5: Smoke mode transformations speed (<15s)
    def test_smoke_mode_transformations_speed(self):
        start = time.perf_counter()
        os.environ["SMOKE_TEST"] = "1"
        try:
            for item_id in ITEM_IDS:
                func = get_transformation(item_id)
                if item_id == "F3":
                    df = pd.DataFrame([{"panel": p, "variant": v, "motif": "ACGTACG", "activation_drop": 0.5}
                                       for p, v in zip(["a", "b", "c", "d", "e"], FIVE_VARIANTS_SPECIES)])
                elif item_id == "F5":
                    df = pd.DataFrame({"x1": [0.0], "y1": [0.0], "x2": [1.0], "y2": [1.0]})
                elif item_id == "T1":
                    df = pd.DataFrame({"model": ["M"], "taxonomic_level": ["Species"], "accuracy": [90.0], "top_5": [98.0], "macro_f1": [89.0]})
                elif item_id == "T2":
                    df = pd.DataFrame({"model": ["M"], "backbone_m": [15.0], "fc_params_m": [28.0], "parameters_m": [43.0]})
                elif item_id == "T12":
                    df = pd.DataFrame({"model": ["M"], "majority_vote_acc": [70.0], "mean_logits_acc": [71.0]})
                elif item_id == "T11":
                    df = pd.DataFrame({"model_pair": ["M"], "is_permutation_tested": [True]})
                else:
                    df = pd.DataFrame({"x": [1], "y": [2], "epsilon": [0.0], "accuracy": [80.0], "f1_bin": ["0.9-1.0"],
                                       "knn_accuracy": [80.0], "level_pair": ["sp_gn"], "model": ["M"],
                                       "motif": ["MOTIF1"], "triplet": ["AAA"], "rscu": [1.0], "umap_1": [0.1], "umap_2": [0.2]})
                res = func(df, {})
                self.assertIsNotNone(res)
        finally:
            os.environ.pop("SMOKE_TEST", None)

        elapsed = time.perf_counter() - start
        self.assertLess(elapsed, 15.0, f"Transformations in smoke mode must take under 15s (took {elapsed:.2f}s)")

    # C6: Batch diagnosis matrix
    def test_batch_diagnosis_matrix(self):
        cfg_path, _ = self._create_minimal_valid_config(make_files=True)
        diag_df = diagnose_publication_readiness(cfg_path, project_root=self.root, strict_empirical=False)

        self.assertIsInstance(diag_df, pd.DataFrame)
        self.assertEqual(len(diag_df), 17, "Diagnostic matrix must contain all 17 items")
        expected_cols = {"item_id", "status", "diagnostic", "proposed_source", "recommendation"}
        self.assertTrue(expected_cols.issubset(diag_df.columns), f"Columns must include {expected_cols}")
        for status in diag_df["status"].tolist():
            self.assertEqual(status, "OK")

    # C7: Diagnosis status FALTA
    def test_diagnosis_status_falta(self):
        cfg_path, _ = self._create_minimal_valid_config(make_files=False)
        diag_df = diagnose_publication_readiness(cfg_path, project_root=self.root, strict_empirical=False)

        self.assertEqual(len(diag_df), 17)
        for _, row in diag_df.iterrows():
            self.assertEqual(row["status"], "FALTA")
            self.assertIn("missing", row["diagnostic"].lower())
            self.assertTrue(bool(row["proposed_source"]))

    # C8: Diagnosis status SCHEMA_MISMATCH
    def test_diagnosis_status_schema_mismatch(self):
        cfg_path, cfg_data = self._create_minimal_valid_config(make_files=True)
        # Modify T1 artifact to lack required columns
        t1_artifact = self.root / "evidence" / "T1.csv"
        pd.DataFrame({"unrelated_col": [1, 2, 3]}).to_csv(t1_artifact, index=False)
        # Update hash in config
        cfg_data["items"][0]["sha256"] = sha256_file(t1_artifact)
        cfg_data["items"][0]["expected"]["columns"] = ["accuracy", "top_5"]
        cfg_path.write_text(yaml.safe_dump(cfg_data), encoding="utf-8")

        diag_df = diagnose_publication_readiness(cfg_path, project_root=self.root, strict_empirical=False)
        t1_row = diag_df[diag_df["item_id"] == "T1"].iloc[0]
        self.assertEqual(t1_row["status"], "SCHEMA_MISMATCH")
        self.assertIn("Missing expected table columns", t1_row["diagnostic"])

    # C9: Diagnosis status IDENTITY_MISMATCH
    def test_diagnosis_status_identity_mismatch(self):
        cfg_path, cfg_data = self._create_minimal_valid_config(make_files=True)
        # Modify manifest for T2 to have seed 999 instead of 42
        t2_manifest = self.root / "evidence" / "T2.manifest.json"
        mdata = json.loads(t2_manifest.read_text(encoding="utf-8"))
        mdata["scientific_identity"]["seed"] = 999
        t2_manifest.write_text(json.dumps(mdata), encoding="utf-8")
        cfg_data["items"][1]["expected"]["scientific_identity"]["seed"] = 42
        cfg_path.write_text(yaml.safe_dump(cfg_data), encoding="utf-8")

        diag_df = diagnose_publication_readiness(cfg_path, project_root=self.root, strict_empirical=False)
        t2_row = diag_df[diag_df["item_id"] == "T2"].iloc[0]
        self.assertEqual(t2_row["status"], "IDENTITY_MISMATCH")
        self.assertIn("seed mismatch", t2_row["diagnostic"])

    # C10: Diagnosis status SCIENTIFIC_MISMATCH
    def test_diagnosis_status_scientific_mismatch(self):
        cfg_path, cfg_data = self._create_minimal_valid_config(make_files=True)
        # Feed an image file into T3
        img_file = self.root / "evidence" / "T3_chart.png"
        img_file.write_bytes(b"\x89PNG\r\n\x1a\nfake-chart")
        cfg_data["items"][2]["artifact"] = str(img_file)
        cfg_data["items"][2]["sha256"] = sha256_file(img_file)
        cfg_path.write_text(yaml.safe_dump(cfg_data), encoding="utf-8")

        diag_df = diagnose_publication_readiness(cfg_path, project_root=self.root, strict_empirical=False)
        t3_row = diag_df[diag_df["item_id"] == "T3"].iloc[0]
        self.assertEqual(t3_row["status"], "SCIENTIFIC_MISMATCH")
        self.assertIn("Image file (.png) passed as publication source", t3_row["diagnostic"])

    # C11: F3 requires exactly 5 models (including ConvNeXt_F)
    def test_f3_requires_five_models(self):
        # Missing ConvNeXt_F
        df_incomplete = pd.DataFrame([
            {"panel": "a", "variant": "ResNet50_Clean", "motif": "AAA", "activation_drop": 0.5},
            {"panel": "b", "variant": "ConvNeXt_Z", "motif": "AAA", "activation_drop": 0.4},
            {"panel": "c", "variant": "ConvNeXt_T", "motif": "AAA", "activation_drop": 0.3},
            {"panel": "d", "variant": "ResNet50_F", "motif": "AAA", "activation_drop": 0.2},
        ])
        f3_func = get_transformation("F3")
        with self.assertRaises(PublicationValidationError) as ctx:
            f3_func(df_incomplete, {})
        self.assertIn("missing species model variants", str(ctx.exception))
        self.assertIn("ConvNeXt_F", str(ctx.exception))

    # C12: T11 rejects raw co-occurrence without permutation null test
    def test_t11_rejects_raw_cooccurrence_without_permutation(self):
        df_no_perm = pd.DataFrame({
            "model_pair": ["ResNet50-ConvNeXt"],
            "correlation_r": [0.8],
            "is_permutation_tested": [False],
        })
        t11_func = get_transformation("T11")
        with self.assertRaises(PublicationValidationError) as ctx:
            t11_func(df_no_perm, {})
        self.assertIn("without permutation null testing cannot be published as architecture correlation", str(ctx.exception))

        # Spurious r=1 with no variance
        df_spurious = pd.DataFrame({
            "model_pair": ["ResNet50-ConvNeXt"],
            "correlation_r": [1.0],
            "p_value": [0.0],
            "has_variance": [False],
        })
        with self.assertRaises(PublicationValidationError) as ctx2:
            t11_func(df_spurious, {})
        self.assertIn("spurious r=1 correlation", str(ctx2.exception))

    # C13: F5 requires network coordinates (rejects bar charts)
    def test_f5_requires_network_coordinates(self):
        df_bars = pd.DataFrame({"motif": ["AAA", "AAC"], "count": [10, 20]})
        f5_func = get_transformation("F5")
        with self.assertRaises(PublicationValidationError) as ctx:
            f5_func(df_bars, {})
        self.assertIn("missing network edge coordinates", str(ctx.exception))
        self.assertIn("Bar chart data cannot be rendered as a network graph", str(ctx.exception))

    # C14: Strict rejection of synthetic evidence
    def test_strict_rejection_of_synthetic(self):
        cfg_path, cfg_data = self._create_minimal_valid_config(make_files=True)
        # Mark T1 manifest as synthetic
        t1_manifest = self.root / "evidence" / "T1.manifest.json"
        mdata = json.loads(t1_manifest.read_text(encoding="utf-8"))
        mdata["is_smoke_test"] = True
        t1_manifest.write_text(json.dumps(mdata), encoding="utf-8")

        out_dir = self.root / "runs" / "strict_run" / "publication"
        result = assemble_publication(cfg_path, out_dir, strict_empirical=True, diagnostic_only=False)
        self.assertEqual(result["status"], "BLOCKED")
        self.assertEqual(result["gates"]["empirical"], "BLOCKED")

    # C15: 00_regenerate notebook diagnostic & isolation
    def test_regenerate_notebook_diagnostic_and_isolation(self):
        cfg_path, _ = self._create_minimal_valid_config(make_files=True)
        out_dir = self.root / "runs" / "isolated_run" / "publication"

        result = assemble_publication(cfg_path, out_dir, strict_empirical=False, diagnostic_only=True)
        self.assertIn("matrix", result)
        self.assertTrue(result["matrix"].is_file())
        self.assertTrue((out_dir / "publication_diagnostic_report.csv").is_file())
        self.assertTrue((out_dir / "publication_manifest.json").is_file())

    # C16: Logging path traceability
    def test_logging_path_traceability(self):
        test_file = self.root / "evidence" / "traceable.txt"
        test_file.write_text("traceable content", encoding="utf-8")

        with patch("sys.stderr", new_callable=io.StringIO) as mock_stderr:
            validate_artifact_is_tabular(self.root / "evidence" / "T1.csv")
            sha256_file(test_file)
            cfg_path, _ = self._create_minimal_valid_config(make_files=True)
            diagnose_publication_readiness(cfg_path, project_root=self.root, strict_empirical=False)
            output = mock_stderr.getvalue()

        self.assertIn("[LOG_PATH]", output)
        self.assertIn(str(self.root / "evidence"), output)


if __name__ == "__main__":
    unittest.main()
