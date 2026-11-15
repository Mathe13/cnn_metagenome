"""Tests for C04: Scientific Decision and Executable Contract for Table 11."""

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd
import yaml

from viral_cnn.reporting.publication import (
    PublicationValidationError,
    diagnose_publication_readiness,
    sha256_file,
    validate_selection,
)
from viral_cnn.reporting.transformations import (
    emit_derived_manifest,
    transform_table_11_architecture_correlations,
)


class TestC04T11ScientificDecision(unittest.TestCase):
    """Test suite verifying C04 scientific decisions and executable contracts for T11."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name).resolve()
        self.evidence_dir = self.root / "evidence"
        self.evidence_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir = self.root / "output"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.expected_cols = [
            "status",
            "motif_score",
            "n_valid_edges",
            "n_valid_permutations",
            "r_canonical",
            "r_reversed",
            "delta_r_orientation",
            "icc_symmetric",
            "r_bidirectional",
            "p_permutation_null",
        ]

        self.valid_t11_data = {
            "status": ["COMPLETE"],
            "motif_score": ["mean_exploratory_triplet_rscu_in_7mer"],
            "n_valid_edges": [185],
            "n_valid_permutations": [1000],
            "r_canonical": [0.342],
            "r_reversed": [0.315],
            "delta_r_orientation": [0.027],
            "icc_symmetric": [0.328],
            "r_bidirectional": [0.335],
            "p_permutation_null": [0.002],
        }

    def tearDown(self):
        self.temp_dir.cleanup()

    # C1: expected.columns canonical list for T11 in config
    def test_t11_expected_columns_canonical(self):
        """C1: reproduction_m07.yaml defines canonical cooccurrence columns for T11."""
        config_path = Path("configs/reproduction_m07.yaml")
        self.assertTrue(config_path.is_file())
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        t11_item = next(it for it in cfg["items"] if it["item_id"] == "T11")
        self.assertEqual(t11_item["expected"]["columns"], self.expected_cols)

    # C2: transform_table_11_architecture_correlations produces complete columns
    def test_t11_transforms_valid_cooccurrence_data(self):
        """C2: transform_table_11_architecture_correlations preserves all co-occurrence synergy columns."""
        df_raw = pd.DataFrame(self.valid_t11_data)
        df_out = transform_table_11_architecture_correlations(df_raw, {})
        self.assertEqual(list(df_out.columns), self.expected_cols)
        self.assertEqual(df_out["p_permutation_null"].iloc[0], 0.002)

        # Mapping p_perm -> p_permutation_null
        df_alt = df_raw.rename(columns={"p_permutation_null": "p_perm"})
        df_alt_out = transform_table_11_architecture_correlations(df_alt, {})
        self.assertIn("p_permutation_null", df_alt_out.columns)
        self.assertEqual(df_alt_out["p_permutation_null"].iloc[0], 0.002)

    # C3: Documentation and scientific contract in docs/publication_data_contracts.md
    def test_t11_documentation_decision_contract(self):
        """C3: publication_data_contracts.md documents node-permutation synergy and rejects inter-model claims."""
        doc_path = Path("docs/publication_data_contracts.md")
        self.assertTrue(doc_path.is_file())
        content = doc_path.read_text(encoding="utf-8")
        self.assertIn("Decisão C04", content)
        self.assertIn("p_permutation_null", content)
        self.assertIn("table_11_cooccurrence_synergy.csv", content)
        self.assertIn("table_11b_jaccard_variant_diagnostics.csv", content)

    # C4: Rejection when permutation test is missing
    def test_t11_rejects_missing_permutation_test(self):
        """C4: transform_table_11_architecture_correlations raises error if permutation null test is absent."""
        df_no_perm = pd.DataFrame({
            "status": ["COMPLETE"],
            "motif_score": ["rscu"],
            "correlation_r": [0.45],
        })
        with self.assertRaises(PublicationValidationError) as ctx:
            transform_table_11_architecture_correlations(df_no_perm, {})
        self.assertIn("permutation null testing evidence missing", str(ctx.exception))

    # C5: Rejection of spurious r=1 with zero variance
    def test_t11_rejects_spurious_r1_zero_variance(self):
        """C5: transform_table_11_architecture_correlations rejects r=1 with zero variance or identical matrices."""
        df_spurious = pd.DataFrame({
            "correlation_r": [1.0, 1.0],
            "p_value": [0.0, 0.0],
            "has_variance": [False, False],
            "p_permutation_null": [0.0, 0.0],
            "n_valid_permutations": [1000, 1000],
        })
        with self.assertRaises(PublicationValidationError) as ctx:
            transform_table_11_architecture_correlations(df_spurious, {})
        self.assertIn("spurious r=1 correlation", str(ctx.exception))

    # C6: Rejection when is_permutation_tested contains False
    def test_t11_rejects_false_permutation_flag(self):
        """C6: transform_table_11_architecture_correlations rejects is_permutation_tested == False."""
        df_false_perm = pd.DataFrame({
            "status": ["INCOMPLETE"],
            "is_permutation_tested": [False],
        })
        with self.assertRaises(PublicationValidationError) as ctx:
            transform_table_11_architecture_correlations(df_false_perm, {})
        self.assertIn("without permutation null testing cannot be published", str(ctx.exception))

    # C7: Rejection when valid permutations < 100
    def test_t11_rejects_insufficient_permutations(self):
        """C7: transform_table_11_architecture_correlations rejects n_valid_permutations < 100."""
        df_low_perm = pd.DataFrame(self.valid_t11_data)
        df_low_perm["n_valid_permutations"] = [50]
        with self.assertRaises(PublicationValidationError) as ctx:
            transform_table_11_architecture_correlations(df_low_perm, {})
        self.assertIn("insufficient valid permutations", str(ctx.exception))

    # C8: diagnose_publication_readiness reports SCIENTIFIC_MISMATCH if permutation test missing
    def test_diagnose_readiness_t11_missing_permutation(self):
        """C8: diagnose_publication_readiness classifies T11 without permutation null test as SCIENTIFIC_MISMATCH."""
        t11_file = self.evidence_dir / "t11_no_perm.csv"
        pd.DataFrame({
            "correlation_r": [0.75],
            "p_value": [0.01],
            "is_permutation_tested": [False],
        }).to_csv(t11_file, index=False)

        manifest_file = self.evidence_dir / "t11_no_perm.manifest.json"
        manifest_file.write_text(json.dumps({
            "item_id": "T11",
            "status": "COMPLETE",
            "is_smoke_test": False,
            "evidence_kind": "empirical",
            "n": 100,
            "support": 100,
        }), encoding="utf-8")

        config = {
            "publication_protocol": "c06_strict_v1",
            "schema_version": 1,
            "_project_root": str(self.root),
            "items": [
                {
                    "item_id": "T11",
                    "artifact": str(t11_file),
                    "manifest": str(manifest_file),
                    "expected": {"columns": self.expected_cols},
                }
            ],
        }

        diag_df = diagnose_publication_readiness(config, project_root=self.root)
        t11_diag = diag_df[diag_df["item_id"] == "T11"].iloc[0]
        self.assertEqual(t11_diag["status"], "SCIENTIFIC_MISMATCH")
        self.assertIn("permutation null test", t11_diag["diagnostic"])

    # C9: diagnose_publication_readiness reports OK for valid T11 cooccurrence artifact
    def test_diagnose_readiness_t11_valid_cooccurrence(self):
        """C9: diagnose_publication_readiness classifies valid T11 cooccurrence artifact as OK."""
        t11_file = self.evidence_dir / "t11_valid.csv"
        pd.DataFrame(self.valid_t11_data).to_csv(t11_file, index=False)

        manifest_file = self.evidence_dir / "t11_valid.manifest.json"
        manifest_file.write_text(json.dumps({
            "item_id": "T11",
            "status": "COMPLETE",
            "is_smoke_test": False,
            "evidence_kind": "empirical",
            "n": 100,
            "support": 100,
        }), encoding="utf-8")

        config = {
            "publication_protocol": "c06_strict_v1",
            "schema_version": 1,
            "_project_root": str(self.root),
            "items": [
                {
                    "item_id": "T11",
                    "artifact": str(t11_file),
                    "manifest": str(manifest_file),
                    "expected": {"columns": self.expected_cols},
                }
            ],
        }

        diag_df = diagnose_publication_readiness(config, project_root=self.root)
        t11_diag = diag_df[diag_df["item_id"] == "T11"].iloc[0]
        self.assertEqual(t11_diag["status"], "OK")

    # C10: validate_selection accepts T11 as DERIVED_VERIFIED
    def test_validate_selection_t11_derived_verified(self):
        """C10: validate_selection assigns status DERIVED_VERIFIED to T11 under strict empirical mode."""
        # Source CSV
        source_csv = self.evidence_dir / "source_t11.csv"
        pd.DataFrame(self.valid_t11_data).to_csv(source_csv, index=False)
        source_sha = sha256_file(source_csv)

        # Parent manifest
        parent_manifest = self.evidence_dir / "parent_t11.manifest.json"
        sci_identity = {
            "dataset_sha256": "dummy_dataset_sha",
            "seed": 42,
            "split_id": "test",
            "taxonomy_version": "ncbi_2024_01",
            "class_order_sha256": "dummy_class_order_sha",
            "checkpoint_sha256s": {"ResNet50": "dummy_resnet_sha"},
            "model_variants": ["ResNet50"],
            "noise_conditions": [0.0],
            "metric_definition": "closed_set_macro",
            "population_id": "test_pop",
        }
        parent_data = {
            "item_id": "T11",
            "protocol": "c06_strict_v1",
            "evidence_kind": "empirical",
            "status": "COMPLETE",
            "is_smoke_test": False,
            "synthetic": False,
            "artifact_path": str(source_csv),
            "artifact_sha256": source_sha,
            "git_commit": "def5678",
            "seed": 42,
            "split_id": "test",
            "n": 185,
            "support": 185,
            "denominator_n": 185,
            "scientific_identity": sci_identity,
            "inputs": [{"path": str(source_csv), "sha256": source_sha}],
        }
        parent_manifest.write_text(json.dumps(parent_data, indent=2), encoding="utf-8")

        # Derived artifact
        derived_csv = self.output_dir / "t11_derived.csv"
        pd.DataFrame(self.valid_t11_data).to_csv(derived_csv, index=False)

        manifest_path = emit_derived_manifest(
            item_id="T11",
            output_artifact_path=derived_csv,
            parent_manifest_path=parent_manifest,
            source_artifact_path=source_csv,
            transformation_name="transform_table_11_architecture_correlations",
            output_rows=1,
        )

        config = {
            "protocol": "c06_strict_v1",
            "_project_root": str(self.root),
            "items": [
                {
                    "item_id": "T11",
                    "protocol": "c06_strict_v1",
                    "transformation": "transform_table_11_architecture_correlations",
                    "output": "tables/table_11.csv",
                    "manifest": str(manifest_path),
                    "artifact": str(derived_csv),
                    "sha256": sha256_file(derived_csv),
                    "expected": {
                        "columns": self.expected_cols,
                        "scientific_identity": sci_identity,
                    },
                }
            ],
        }

        rows = validate_selection(config, strict_empirical=True)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["item_id"], "T11")
        self.assertEqual(rows[0]["status"], "DERIVED_VERIFIED")


if __name__ == "__main__":
    unittest.main()
