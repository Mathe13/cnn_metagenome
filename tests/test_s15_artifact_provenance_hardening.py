"""Contract tests for S15: Empirical artifact provenance hardening.

Verifies C1 through C9 from .specs/features/s15-artifact-provenance-hardening/checks.md.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd
import yaml

from viral_cnn.reporting.publication import (
    PublicationValidationError,
    sha256_file,
    validate_artifact_provenance,
    validate_selection,
)
from viral_cnn.reporting.transformations import emit_derived_manifest


class TestS15ArtifactProvenanceHardening(unittest.TestCase):
    """Test suite verifying fail-closed empirical artifact provenance contracts."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        (self.root / ".git").mkdir()
        (self.root / "evidence").mkdir()

        # Create dummy source inputs
        self.source_txt = self.root / "evidence" / "source.txt"
        self.source_txt.write_text("empirical sequence input data\n", encoding="utf-8")
        self.source_sha = sha256_file(self.source_txt)

        self.dataset_csv = self.root / "evidence" / "dataset.csv"
        self.dataset_csv.write_text("accession,label\nNC_001,0\n", encoding="utf-8")
        self.dataset_sha = sha256_file(self.dataset_csv)

        self.valid_manifest_data = {
            "protocol": "c03_paired_v1",
            "evidence_kind": "empirical",
            "is_smoke_test": False,
            "git_commit": "abcdef1234567890abcdef1234567890abcdef12",
            "created_utc": "2026-11-11T20:00:00+00:00",
            "library_versions": {
                "python": "3.10.12",
                "torch": "2.1.2",
                "numpy": "1.26.4",
                "pandas": "2.2.0",
                "pyarrow": "15.0.0",
            },
            "seed": 42,
            "split_id": "test",
            "scientific_identity": {
                "dataset_sha256": self.dataset_sha,
                "model_variants": ["ResNet50_Clean", "ConvNeXt_Z"],
                "noise_conditions": [0.0, 0.05],
                "checkpoint_sha256s": {"ResNet50_Clean": "ckpt1_sha", "ConvNeXt_Z": "ckpt2_sha"},
            },
            "configuration_sha256": "cfg_sha_123456",
            "inputs": [
                {"path": str(self.source_txt), "sha256": self.source_sha},
                {"path": str(self.dataset_csv), "sha256": self.dataset_sha},
            ],
        }

    def tearDown(self):
        self.tmp.cleanup()

    # C1: Valid full provenance passes
    def test_valid_full_provenance_passes(self):
        """C1: validate_artifact_provenance returns verified record when all 12 attributes exist."""
        result = validate_artifact_provenance(self.valid_manifest_data, self.root)
        self.assertIsInstance(result, dict)
        self.assertEqual(result["git_commit"], "abcdef1234567890abcdef1234567890abcdef12")
        self.assertFalse(result["is_smoke_test"])
        self.assertEqual(result["seed"], 42)
        self.assertEqual(result["split"], "test")
        self.assertEqual(len(result["verified_parents"]), 2)

    # C2: Smoke test manifest fails provenance
    def test_smoke_test_manifest_fails_provenance(self):
        """C2: validate_artifact_provenance raises PublicationValidationError if is_smoke_test is True or missing."""
        # Case A: is_smoke_test is True
        data_smoke = dict(self.valid_manifest_data, is_smoke_test=True)
        with self.assertRaises(PublicationValidationError) as ctx:
            validate_artifact_provenance(data_smoke, self.root)
        self.assertIn("empirical publication requires is_smoke_test=false", str(ctx.exception))

        # Case B: is_smoke_test is missing
        data_no_smoke = dict(self.valid_manifest_data)
        del data_no_smoke["is_smoke_test"]
        with self.assertRaises(PublicationValidationError) as ctx2:
            validate_artifact_provenance(data_no_smoke, self.root)
        self.assertIn("lacks explicit is_smoke_test flag", str(ctx2.exception))

    # C3: Missing git commit fails provenance
    def test_missing_git_commit_fails_provenance(self):
        """C3: validate_artifact_provenance raises PublicationValidationError if git_commit is missing or empty."""
        data_no_commit = dict(self.valid_manifest_data)
        del data_no_commit["git_commit"]
        with self.assertRaises(PublicationValidationError) as ctx:
            validate_artifact_provenance(data_no_commit, self.root)
        self.assertIn("lacks required non-empty git_commit", str(ctx.exception))

        data_empty_commit = dict(self.valid_manifest_data, git_commit="   ")
        with self.assertRaises(PublicationValidationError) as ctx2:
            validate_artifact_provenance(data_empty_commit, self.root)
        self.assertIn("lacks required non-empty git_commit", str(ctx2.exception))

    # C4: Missing timestamp fails provenance
    def test_missing_timestamp_fails_provenance(self):
        """C4: validate_artifact_provenance raises PublicationValidationError if timestamp is missing."""
        data_no_ts = dict(self.valid_manifest_data)
        del data_no_ts["created_utc"]
        with self.assertRaises(PublicationValidationError) as ctx:
            validate_artifact_provenance(data_no_ts, self.root)
        self.assertIn("lacks required execution timestamp", str(ctx.exception))

        # timestamp_utc also satisfies timestamp requirement
        data_alt_ts = dict(data_no_ts, timestamp_utc="2026-11-11T21:00:00Z")
        res = validate_artifact_provenance(data_alt_ts, self.root)
        self.assertEqual(res["execution_timestamp"], "2026-11-11T21:00:00Z")

    # C5: Missing software versions fails provenance
    def test_missing_software_versions_fails_provenance(self):
        """C5: validate_artifact_provenance raises PublicationValidationError if software versions missing."""
        data_no_ver = dict(self.valid_manifest_data)
        del data_no_ver["library_versions"]
        with self.assertRaises(PublicationValidationError) as ctx:
            validate_artifact_provenance(data_no_ver, self.root)
        self.assertIn("lacks required software environment metadata", str(ctx.exception))

        # versions field also satisfies software versions requirement
        data_alt_ver = dict(data_no_ver, versions={"python": "3.10.12", "torch": "2.1.2"})
        res = validate_artifact_provenance(data_alt_ver, self.root)
        self.assertIn("python", res["versions"])

    # C6: Broken parent artifact fails provenance
    def test_broken_parent_artifact_fails_provenance(self):
        """C6: validate_artifact_provenance raises PublicationValidationError if parent missing or hash mismatch."""
        # Case A: Missing parent file
        missing_parent = self.root / "evidence" / "nonexistent.txt"
        data_missing = dict(self.valid_manifest_data, inputs=[{"path": str(missing_parent), "sha256": "abc"}])
        with self.assertRaises(PublicationValidationError) as ctx:
            validate_artifact_provenance(data_missing, self.root)
        self.assertIn("Parent artifact missing on disk", str(ctx.exception))

        # Case B: Hash mismatch
        data_mismatch = dict(
            self.valid_manifest_data,
            inputs=[{"path": str(self.source_txt), "sha256": "0000000000000000000000000000000000000000000000000000000000000000"}],
        )
        with self.assertRaises(PublicationValidationError) as ctx2:
            validate_artifact_provenance(data_mismatch, self.root)
        self.assertIn("Parent artifact SHA-256 mismatch", str(ctx2.exception))

    # C7: Transitive smoke parent fails provenance
    def test_transitive_smoke_parent_fails_provenance(self):
        """C7: validate_artifact_provenance raises PublicationValidationError if upstream parent is smoke test."""
        parent_manifest_path = self.root / "evidence" / "parent_stage.manifest.json"
        parent_manifest_data = {
            "protocol": "c02_benchmark_v1",
            "is_smoke_test": True,
            "inputs": [{"path": str(self.source_txt), "sha256": self.source_sha}],
        }
        parent_manifest_path.write_text(json.dumps(parent_manifest_data), encoding="utf-8")
        parent_manifest_sha = sha256_file(parent_manifest_path)

        data_with_smoke_parent = dict(
            self.valid_manifest_data,
            inputs=[{"path": str(parent_manifest_path), "sha256": parent_manifest_sha}],
        )
        with self.assertRaises(PublicationValidationError) as ctx:
            validate_artifact_provenance(data_with_smoke_parent, self.root)
        self.assertIn("Parent manifest", str(ctx.exception))
        self.assertIn("declares synthetic or smoke test evidence", str(ctx.exception))

    # C8: emit_derived_manifest propagates full lineage
    def test_emit_derived_manifest_propagates_full_lineage(self):
        """C8: emit_derived_manifest emits derived manifest with git commit, versions, timestamp, and lineage."""
        parent_manifest_path = self.root / "evidence" / "upstream.manifest.json"
        parent_manifest_data = dict(self.valid_manifest_data)
        parent_manifest_path.write_text(json.dumps(parent_manifest_data), encoding="utf-8")

        output_csv = self.root / "evidence" / "derived_table.csv"
        pd.DataFrame({"x": [1, 2], "y": [3, 4]}).to_csv(output_csv, index=False)

        manifest_path = emit_derived_manifest(
            item_id="T1",
            output_artifact_path=output_csv,
            parent_manifest_path=parent_manifest_path,
            transformation_name="transform_table_1_metrics",
            support_n=100,
        )

        self.assertTrue(manifest_path.is_file())
        derived = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(derived["evidence_kind"], "derived")
        self.assertEqual(derived["git_commit"], self.valid_manifest_data["git_commit"])
        self.assertFalse(derived["is_smoke_test"])
        self.assertIn("created_utc", derived)
        self.assertIn("library_versions", derived)
        self.assertEqual(derived["n"], 100)
        self.assertEqual(derived["support"], 100)
        self.assertEqual(derived["parent_manifest_path"], str(parent_manifest_path.resolve()))
        self.assertEqual(derived["parent_manifest_sha256"], sha256_file(parent_manifest_path))

    # C9: validate_selection enforces artifact provenance
    def test_validate_selection_enforces_artifact_provenance(self):
        """C9: validate_selection under strict_provenance fails closed when empirical manifest lacks provenance."""
        artifact_path = self.root / "evidence" / "t1_data.csv"
        pd.DataFrame({"accuracy": [85.0], "top_5": [95.0]}).to_csv(artifact_path, index=False)

        # Manifest missing git_commit and versions
        manifest_path = self.root / "evidence" / "t1_incomplete.manifest.json"
        incomplete_manifest = {
            "protocol": "c06_test_v1",
            "evidence_kind": "empirical",
            "is_smoke_test": False,
            "status": "COMPLETE",
            "seed": 42,
            "split_id": "test",
            "artifact_sha256": sha256_file(artifact_path),
            "inputs": [{"path": str(self.source_txt), "sha256": self.source_sha}],
        }
        manifest_path.write_text(json.dumps(incomplete_manifest), encoding="utf-8")

        config = {
            "publication_protocol": "c06_strict_v1",
            "schema_version": 1,
            "_project_root": str(self.root),
            "items": [
                {
                    "item_id": "T1",
                    "artifact": str(artifact_path),
                    "manifest": str(manifest_path),
                    "sha256": sha256_file(artifact_path),
                    "protocol": "c06_test_v1",
                    "transformation": "transform_t1",
                    "expected": {"columns": ["accuracy", "top_5"]},
                }
            ],
        }

        # With strict_provenance=True, validation raises PublicationValidationError
        with self.assertRaises(PublicationValidationError) as ctx:
            validate_selection(config, strict_empirical=True, strict_provenance=True)
        self.assertIn("strict empirical gate blocked", str(ctx.exception))
        self.assertIn("git_commit", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
