"""Tests for C02: Derived manifests, hashes, and experimental support validation in C06."""

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from viral_cnn.reporting.publication import (
    PublicationValidationError,
    sha256_file,
    validate_selection,
)
from viral_cnn.reporting.transformations import (
    emit_derived_manifest,
    execute_publication_transformation,
)


class TestC02DerivedManifests(unittest.TestCase):
    """Test suite verifying C02 executable contracts for derived evidence."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name).resolve()
        self.evidence_dir = self.root / "evidence"
        self.evidence_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir = self.root / "output"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Baseline scientific identity
        self.sci_identity = {
            "dataset_sha256": "dummy_dataset_sha",
            "seed": 42,
            "split_id": "test",
            "taxonomy_version": "ncbi_2024_01",
            "class_order_sha256": "dummy_class_order_sha",
            "checkpoints": [
                {"variant": "ResNet50", "path": "evidence/resnet.pt", "sha256": "dummy_resnet_sha"}
            ],
            "checkpoint_sha256s": {"ResNet50": "dummy_resnet_sha"},
            "model_variants": ["ResNet50"],
            "noise_conditions": [0.0],
            "metric_definition": "closed_set_macro",
            "population_id": "test_pop",
        }

        # Create upstream source CSV
        self.source_csv = self.evidence_dir / "upstream_table.csv"
        pd.DataFrame({
            "model": ["ResNet50"],
            "taxonomic_level": ["species"],
            "backbone_m": [23.5],
            "fc_params_m": [2.0],
            "parameters_m": [25.5],
            "accuracy": [0.95],
            "top_5": [0.99],
            "macro_f1": [0.94],
            "latency_ms": [1.2],
        }).to_csv(self.source_csv, index=False)
        self.source_sha = sha256_file(self.source_csv)

        # Create parent manifest
        self.parent_manifest_path = self.evidence_dir / "upstream.manifest.json"
        self.parent_manifest_data = {
            "item_id": "T1",
            "protocol": "c06_strict_v1",
            "evidence_kind": "empirical",
            "status": "COMPLETE",
            "is_smoke_test": False,
            "synthetic": False,
            "artifact_path": str(self.source_csv),
            "artifact_sha256": self.source_sha,
            "git_commit": "abc1234",
            "seed": 42,
            "split_id": "test",
            "n": 1706270,
            "support": 1706270,
            "denominator_n": 1706270,
            "scientific_identity": self.sci_identity,
            "inputs": [
                {"path": str(self.source_csv), "sha256": self.source_sha}
            ],
        }
        self.parent_manifest_path.write_text(
            json.dumps(self.parent_manifest_data, indent=2), encoding="utf-8"
        )
        self.parent_sha = sha256_file(self.parent_manifest_path)

        # Minimal config for validate_selection
        self.config = {
            "protocol": "c06_strict_v1",
            "_project_root": str(self.root),
            "items": [
                {
                    "item_id": "T1",
                    "protocol": "c06_strict_v1",
                    "transformation": "transform_table_1_metrics",
                    "output": "tables/table_1_metrics.csv",
                    "manifest": str(self.output_dir / "t1_derived.manifest.json"),
                    "artifact": str(self.output_dir / "t1_derived.csv"),
                    "expected": {
                        "columns": ["model", "taxonomic_level", "accuracy", "top_5", "macro_f1"],
                        "scientific_identity": self.sci_identity,
                    },
                }
            ],
        }

    def tearDown(self):
        self.temp_dir.cleanup()

    # C1: validate_selection accepts derived manifests with status DERIVED_VERIFIED
    def test_validate_selection_accepts_derived(self):
        """C1: validate_selection assigns status DERIVED_VERIFIED when derivation and hashes match."""
        derived_csv = self.output_dir / "t1_derived.csv"
        pd.DataFrame({
            "model": ["ResNet50"],
            "taxonomic_level": ["species"],
            "backbone_m": [23.5],
            "fc_params_m": [2.0],
            "parameters_m": [25.5],
            "accuracy": [0.95],
            "top_5": [0.99],
            "macro_f1": [0.94],
            "latency_ms": [1.2],
        }).to_csv(derived_csv, index=False)
        self.config["items"][0]["sha256"] = sha256_file(derived_csv)

        manifest_path = emit_derived_manifest(
            item_id="T1",
            output_artifact_path=derived_csv,
            parent_manifest_path=self.parent_manifest_path,
            source_artifact_path=self.source_csv,
            transformation_name="transform_table_1_metrics",
            output_rows=1,
        )
        self.assertTrue(manifest_path.is_file())

        rows = validate_selection(self.config, strict_empirical=True)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["item_id"], "T1")
        self.assertEqual(rows[0]["status"], "DERIVED_VERIFIED")

    # C2: validate_selection checks derivation block completeness
    def test_validate_selection_checks_derivation_block(self):
        """C2: validate_selection requires source_artifact, parent_manifest, and transformation in derivation."""
        derived_csv = self.output_dir / "t1_derived.csv"
        pd.DataFrame({
            "model": ["ResNet50"], "taxonomic_level": ["species"], "backbone_m": [23.5],
            "fc_params_m": [2.0], "parameters_m": [25.5], "accuracy": [0.95],
            "top_5": [0.99], "macro_f1": [0.94], "latency_ms": [1.2],
        }).to_csv(derived_csv, index=False)
        self.config["items"][0]["sha256"] = sha256_file(derived_csv)

        manifest_path = emit_derived_manifest(
            item_id="T1",
            output_artifact_path=derived_csv,
            parent_manifest_path=self.parent_manifest_path,
            source_artifact_path=self.source_csv,
            transformation_name="transform_table_1_metrics",
        )
        # Corrupt derivation block by removing parent_manifest
        manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))
        del manifest_data["derivation"]["parent_manifest"]
        manifest_path.write_text(json.dumps(manifest_data), encoding="utf-8")

        rows = validate_selection(self.config, strict_empirical=False)
        self.assertEqual(rows[0]["status"], "INVALID")
        self.assertIn("missing derivation block 'parent_manifest'", rows[0]["diagnostic"])

    # C3: validate_selection rejects tampered or missing source artifact/parent manifest
    def test_validate_selection_rejects_tampered_or_missing_source(self):
        """C3: validate_selection marks item INVALID if source artifact or parent manifest is missing or altered."""
        derived_csv = self.output_dir / "t1_derived.csv"
        pd.DataFrame({
            "model": ["ResNet50"], "taxonomic_level": ["species"], "backbone_m": [23.5],
            "fc_params_m": [2.0], "parameters_m": [25.5], "accuracy": [0.95],
            "top_5": [0.99], "macro_f1": [0.94], "latency_ms": [1.2],
        }).to_csv(derived_csv, index=False)
        self.config["items"][0]["sha256"] = sha256_file(derived_csv)

        emit_derived_manifest(
            item_id="T1",
            output_artifact_path=derived_csv,
            parent_manifest_path=self.parent_manifest_path,
            source_artifact_path=self.source_csv,
            transformation_name="transform_table_1_metrics",
        )

        # Alter source CSV content to create hash mismatch
        with open(self.source_csv, "a", encoding="utf-8") as f:
            f.write("\n# tampered line")

        rows = validate_selection(self.config, strict_empirical=False)
        self.assertEqual(rows[0]["status"], "INVALID")
        self.assertTrue("mismatch" in rows[0]["diagnostic"].lower())

    # C4: validate_selection rejects synthetic parent under strict empirical mode
    def test_validate_selection_rejects_synthetic_parent_under_strict(self):
        """C4: validate_selection marks item SYNTHETIC and fails closed if parent manifest has is_smoke_test or synthetic."""
        # Mark parent as smoke test
        self.parent_manifest_data["is_smoke_test"] = True
        self.parent_manifest_path.write_text(
            json.dumps(self.parent_manifest_data, indent=2), encoding="utf-8"
        )
        self.parent_sha = sha256_file(self.parent_manifest_path)

        derived_csv = self.output_dir / "t1_derived.csv"
        pd.DataFrame({
            "model": ["ResNet50"], "taxonomic_level": ["species"], "backbone_m": [23.5],
            "fc_params_m": [2.0], "parameters_m": [25.5], "accuracy": [0.95],
            "top_5": [0.99], "macro_f1": [0.94], "latency_ms": [1.2],
        }).to_csv(derived_csv, index=False)
        self.config["items"][0]["sha256"] = sha256_file(derived_csv)

        emit_derived_manifest(
            item_id="T1",
            output_artifact_path=derived_csv,
            parent_manifest_path=self.parent_manifest_path,
            source_artifact_path=self.source_csv,
            transformation_name="transform_table_1_metrics",
        )

        rows = validate_selection(self.config, strict_empirical=False)
        self.assertEqual(rows[0]["status"], "SYNTHETIC")

        with self.assertRaises(PublicationValidationError):
            validate_selection(self.config, strict_empirical=True)

    # C5: emit_derived_manifest records output_rows
    def test_emit_derived_manifest_records_output_rows(self):
        """C5: emit_derived_manifest records output_rows matching transformed row count."""
        derived_csv = self.output_dir / "t1_derived.csv"
        df_out = pd.DataFrame({
            "model": [f"model_{i}" for i in range(7)],
            "taxonomic_level": ["species"] * 7,
            "accuracy": [0.9] * 7,
        })
        df_out.to_csv(derived_csv, index=False)

        manifest_path = emit_derived_manifest(
            item_id="T1",
            output_artifact_path=derived_csv,
            parent_manifest_path=self.parent_manifest_path,
            source_artifact_path=self.source_csv,
            transformation_name="transform_table_1_metrics",
            output_rows=7,
        )
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(data["output_rows"], 7)

    # C6: emit_derived_manifest propagates parent support without overriding with output_rows
    def test_emit_derived_manifest_propagates_parent_support(self):
        """C6: emit_derived_manifest preserves parent n, support, and denominator_n."""
        derived_csv = self.output_dir / "t1_derived.csv"
        pd.DataFrame({"a": [1, 2, 3]}).to_csv(derived_csv, index=False)

        manifest_path = emit_derived_manifest(
            item_id="T1",
            output_artifact_path=derived_csv,
            parent_manifest_path=self.parent_manifest_path,
            source_artifact_path=self.source_csv,
            transformation_name="transform_table_1_metrics",
            output_rows=3,
        )
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(data["output_rows"], 3)
        self.assertEqual(data["n"], 1706270)
        self.assertEqual(data["support"], 1706270)
        self.assertEqual(data["denominator_n"], 1706270)

    # C7: emit_derived_manifest requires reason when support is not applicable
    def test_emit_derived_manifest_requires_reason_when_not_applicable(self):
        """C7: emit_derived_manifest raises error when support is missing and no reason is given."""
        # Create a parent without support
        no_support_parent = self.evidence_dir / "no_support.manifest.json"
        no_support_parent.write_text(json.dumps({
            "item_id": "T1",
            "protocol": "c06_strict_v1",
            "status": "COMPLETE",
            "is_smoke_test": False,
        }), encoding="utf-8")

        derived_csv = self.output_dir / "t1_derived.csv"
        pd.DataFrame({"x": [1]}).to_csv(derived_csv, index=False)

        with self.assertRaises(PublicationValidationError):
            emit_derived_manifest(
                item_id="T1",
                output_artifact_path=derived_csv,
                parent_manifest_path=no_support_parent,
                source_artifact_path=self.source_csv,
                transformation_name="transform_table_1_metrics",
            )

        # Providing explicit reason succeeds
        manifest_path = emit_derived_manifest(
            item_id="T1",
            output_artifact_path=derived_csv,
            parent_manifest_path=no_support_parent,
            source_artifact_path=self.source_csv,
            transformation_name="transform_table_1_metrics",
            extra_metadata={"support_not_applicable_reason": "qualitative summary matrix"},
        )
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(data["support_not_applicable_reason"], "qualitative summary matrix")

    # C8: T1 derived manifest preserves large denominator N=1706270
    def test_t1_derived_manifest_preserves_large_denominator(self):
        """C8: T1 derived manifest preserves N=1.706.270 with 5 output rows."""
        derived_csv = self.output_dir / "t1_derived.csv"
        pd.DataFrame({
            "model": ["ResNet50", "ConvNeXt_Z", "ConvNeXt_T", "ConvNeXt_F", "ResNet50_F"],
            "taxonomic_level": ["species"] * 5,
            "accuracy": [0.95, 0.94, 0.96, 0.96, 0.95],
        }).to_csv(derived_csv, index=False)

        manifest_path = emit_derived_manifest(
            item_id="T1",
            output_artifact_path=derived_csv,
            parent_manifest_path=self.parent_manifest_path,
            source_artifact_path=self.source_csv,
            transformation_name="transform_table_1_metrics",
            output_rows=5,
        )
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(data["output_rows"], 5)
        self.assertEqual(data["n"], 1706270)
        self.assertEqual(data["denominator_n"], 1706270)

    # C9: emit_derived_manifest records lineage hashes
    def test_emit_derived_manifest_records_lineage_hashes(self):
        """C9: emit_derived_manifest records source sha, parent manifest sha, and transformation code sha."""
        derived_csv = self.output_dir / "t1_derived.csv"
        pd.DataFrame({"x": [1]}).to_csv(derived_csv, index=False)

        manifest_path = emit_derived_manifest(
            item_id="T1",
            output_artifact_path=derived_csv,
            parent_manifest_path=self.parent_manifest_path,
            source_artifact_path=self.source_csv,
            transformation_name="transform_table_1_metrics",
        )
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        derivation = data["derivation"]
        self.assertEqual(derivation["source_artifact"]["path"], str(self.source_csv))
        self.assertEqual(derivation["source_artifact"]["sha256"], self.source_sha)
        self.assertEqual(derivation["parent_manifest"]["path"], str(self.parent_manifest_path))
        self.assertEqual(derivation["parent_manifest"]["sha256"], self.parent_sha)
        self.assertTrue(len(derivation["transformation"]["code_sha256"]) == 64)

    # C10: emit_derived_manifest inherits scientific identity
    def test_emit_derived_manifest_inherits_scientific_identity(self):
        """C10: emit_derived_manifest inherits scientific_identity dict from parent."""
        derived_csv = self.output_dir / "t1_derived.csv"
        pd.DataFrame({"x": [1]}).to_csv(derived_csv, index=False)

        manifest_path = emit_derived_manifest(
            item_id="T1",
            output_artifact_path=derived_csv,
            parent_manifest_path=self.parent_manifest_path,
            source_artifact_path=self.source_csv,
            transformation_name="transform_table_1_metrics",
        )
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        sci_id = data["scientific_identity"]
        self.assertEqual(sci_id["dataset_sha256"], "dummy_dataset_sha")
        self.assertEqual(sci_id["seed"], 42)
        self.assertEqual(sci_id["split_id"], "test")
        self.assertEqual(sci_id["taxonomy_version"], "ncbi_2024_01")
        self.assertEqual(sci_id["class_order_sha256"], "dummy_class_order_sha")

    # C11: emit_derived_manifest records figure selection parameters
    def test_emit_derived_manifest_records_figure_selection(self):
        """C11: figure derivations record selection parameters (seed, ids, sampling, filters)."""
        derived_csv = self.output_dir / "f2_derived.csv"
        pd.DataFrame({"umap_1": [0.1], "umap_2": [0.2], "taxid": [123]}).to_csv(derived_csv, index=False)

        manifest_path = emit_derived_manifest(
            item_id="F2",
            output_artifact_path=derived_csv,
            parent_manifest_path=self.parent_manifest_path,
            source_artifact_path=self.source_csv,
            transformation_name="transform_figure_2_umap_visualization",
            extra_metadata={
                "selection": {
                    "seed": 42,
                    "ids": [10239],
                    "levels": ["species"],
                    "sampling": "stratified",
                    "filters": ["species_count>=50"],
                }
            },
        )
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.assertIn("selection", data)
        sel = data["selection"]
        self.assertEqual(sel["seed"], 42)
        self.assertEqual(sel["ids"], [10239])
        self.assertEqual(sel["sampling"], "stratified")
        self.assertEqual(sel["filters"], ["species_count>=50"])


if __name__ == "__main__":
    unittest.main()
