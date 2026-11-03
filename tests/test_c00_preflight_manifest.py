import json
import tempfile
import unittest
from pathlib import Path

from viral_cnn.notebook_runtime import (
    NOTEBOOK_REQUIREMENTS,
    empirical_manifest_gate,
    preflight_notebook,
    reserve_run_directory,
)


class TestC00Contracts(unittest.TestCase):
    def test_registry_covers_all_reproduction_notebooks(self):
        self.assertEqual(len(NOTEBOOK_REQUIREMENTS), 15)

    def test_missing_inputs_block_preflight_without_products(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = preflight_notebook("short/09", tmp)
            self.assertEqual(report["status"], "BLOCKED")
            self.assertGreaterEqual(len(report["requirements"]), 3)
            self.assertFalse(any(Path(tmp).rglob("*.csv")))

    def test_reservation_fails_closed_preserving_existing_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            reserved = reserve_run_directory(tmp, "run-a", "short/01")
            sentinel = reserved / "sentinel.txt"
            sentinel.write_text("keep", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                reserve_run_directory(tmp, "run-a", "short/01")
            self.assertEqual(sentinel.read_text(encoding="utf-8"), "keep")
            self.assertNotEqual(reserved, reserve_run_directory(tmp, "run-a", "short/02"))

    def test_empirical_gate_rejects_synthetic_and_legacy_manifests(self):
        self.assertFalse(empirical_manifest_gate({"notebook_id": "old"})["accepted"])
        report = empirical_manifest_gate({"schema_version": 1, "run_id": "r", "notebook_id": "n", "git_commit": "abc", "is_smoke_test": True, "inputs": [], "outputs": [], "evidence": {"status": "synthetic"}})
        self.assertFalse(report["accepted"])
        self.assertIn("is_smoke_test=true", report["rejected_fields"])


if __name__ == "__main__":
    unittest.main()
