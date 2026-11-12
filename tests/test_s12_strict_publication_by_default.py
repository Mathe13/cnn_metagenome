"""Unit and integration tests for S12 strict publication by default (C1-C9)."""
from __future__ import annotations

import inspect
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from viral_cnn.reporting.tables import (
    generate_all_tables,
    generate_publication_tables,
    generate_historical_snapshot_tables,
)
import viral_cnn.reporting.tables as reporting_tables_module


class TestS12StrictPublicationByDefault(unittest.TestCase):
    """Test suite ensuring publication reporting fails closed and isolates historical constants."""

    def test_default_strict_empirical_is_true(self):
        """C1: Calling generate_all_tables without strict_empirical defaults to True."""
        sig = inspect.signature(generate_all_tables)
        self.assertIn("strict_empirical", sig.parameters)
        self.assertTrue(sig.parameters["strict_empirical"].default is True)

        with tempfile.TemporaryDirectory() as tmp_dir:
            out_dir = Path(tmp_dir) / "tables"
            # Since strict_empirical=True by default and publication_config is None, it must raise RuntimeError
            with self.assertRaises(RuntimeError):
                generate_all_tables(out_dir)

    def test_missing_publication_config_raises_runtime_error(self):
        """C2: Strict empirical without publication_config raises specific error."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_dir = Path(tmp_dir) / "tables"
            with self.assertRaises(RuntimeError) as ctx:
                generate_all_tables(out_dir, strict_empirical=True, publication_config=None)
            self.assertIn("Publication generation requires an explicit admissible empirical artifact selection.", str(ctx.exception))

    def test_generate_publication_tables_validates_config(self):
        """C3: generate_publication_tables fails when publication_config is missing or invalid."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_dir = Path(tmp_dir) / "tables"
            non_existent = Path(tmp_dir) / "missing_config.yaml"
            with self.assertRaises((FileNotFoundError, RuntimeError)):
                generate_publication_tables(out_dir, publication_config=non_existent)

    def test_generate_publication_tables_renders_dataframe_dict(self):
        """C4: generate_publication_tables with admissible selection renders dict of DataFrames."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_dir = Path(tmp_dir) / "tables"
            mock_config_path = Path(tmp_dir) / "config.yaml"
            mock_config_path.write_text("items: []\n", encoding="utf-8")

            with patch("viral_cnn.reporting.publication.load_publication_config", return_value={"items": [{"item_id": "T1"}]}), \
                 patch("viral_cnn.reporting.publication.validate_selection", return_value=[]), \
                 patch("viral_cnn.reporting.publication.render_tables", side_effect=lambda cfg, d, r: (d / "table_1.csv").write_text("a,b\n1,2\n", encoding="utf-8")):
                res = generate_publication_tables(out_dir, publication_config=mock_config_path)
                self.assertIsInstance(res, dict)
                self.assertIn(1, res)
                self.assertIsInstance(res[1], pd.DataFrame)

    def test_historical_snapshot_tables_emits_metadata(self):
        """C5: generate_historical_snapshot_tables returns 12 tables and explicit non-admissible metadata."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_dir = Path(tmp_dir) / "tables"
            tables, metadata = generate_historical_snapshot_tables(out_dir, formats=("csv",))
            self.assertEqual(len(tables), 12)
            self.assertIsInstance(metadata, dict)
            self.assertEqual(metadata.get("evidence_class"), "historical_snapshot")
            self.assertEqual(metadata.get("publication_admissible"), False)
            for i in range(1, 13):
                self.assertTrue((out_dir / f"table_{i}.csv").exists())

    def test_generate_all_tables_legacy_opt_in(self):
        """C6: generate_all_tables with explicit strict_empirical=False emits legacy snapshot tables."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_dir = Path(tmp_dir) / "tables"
            tables = generate_all_tables(out_dir, strict_empirical=False, formats=("csv",))
            self.assertEqual(len(tables), 12)
            self.assertTrue((out_dir / "table_1.csv").exists())

    def test_dead_strict_branch_is_removed(self):
        """C7: Dead strict code scanning results_dir.rglob is completely removed."""
        src = inspect.getsource(reporting_tables_module)
        self.assertNotIn('rglob("manifest*.json")', src)
        self.assertNotIn("results_dir.rglob", src)

    def test_cli_strict_mode_fails_closed_when_blocked(self):
        """C8: CLI under --mode strict exits with code 2 when artifact selection is blocked."""
        cmd = [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "generate_paper_tables.py"),
            "--mode", "strict",
            "--config", str(PROJECT_ROOT / "configs" / "publication_c06.yaml"),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        # Under local repo conditions without full cluster reproduction artifacts, it must block (exit 2)
        self.assertEqual(proc.returncode, 2)
        self.assertIn("BLOCKED", proc.stdout)

    def test_cli_legacy_mode_generates_snapshot(self):
        """C9: CLI under --mode legacy outputs snapshot tables and exits with code 0."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_dir = Path(tmp_dir) / "tables"
            cmd = [
                sys.executable,
                str(PROJECT_ROOT / "scripts" / "generate_paper_tables.py"),
                "--mode", "legacy",
                "--output-dir", str(out_dir),
                "--formats", "csv",
            ]
            proc = subprocess.run(cmd, capture_output=True, text=True)
            self.assertEqual(proc.returncode, 0)
            self.assertTrue((out_dir / "table_1.csv").exists())


if __name__ == "__main__":
    unittest.main()
