"""Contract tests for R08: dynamic publication tables, parameter reconciliation, and manuscript audit."""

import json
import shutil
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from viral_cnn.reporting.manuscript import (
    build_manuscript_variants,
    compile_manuscript,
    generate_manuscript_content,
)
from viral_cnn.reporting.tables import (
    generate_all_tables,
    generate_table_1_metrics,
    generate_table_2_profiling,
)


class TestR08ReconciliationAndTables(unittest.TestCase):
    """Test suite verifying C1-C6 contracts for R08."""

    def test_r08_table_1_parameters_reconciliation(self):
        """C1: generate_table_1_metrics reports exact 1D backbone, head, and total parameters."""
        df = generate_table_1_metrics()
        self.assertIsInstance(df, pd.DataFrame)

        # ResNet50 Species
        r_sp = df[df["model"].str.contains("ResNet50") & (df["taxonomic_level"] == "Species")].iloc[0]
        self.assertAlmostEqual(r_sp["backbone_m"], 15.96, places=2)
        self.assertAlmostEqual(r_sp["fc_params_m"], 28.04, places=2)
        self.assertAlmostEqual(r_sp["parameters_m"], 43.99, places=2)

        # ResNet50 Genus
        r_gen = df[df["model"].str.contains("ResNet50") & (df["taxonomic_level"] == "Genus")].iloc[0]
        self.assertAlmostEqual(r_gen["backbone_m"], 15.96, places=2)
        self.assertAlmostEqual(r_gen["fc_params_m"], 5.54, places=2)
        self.assertAlmostEqual(r_gen["parameters_m"], 21.49, places=2)

        # ConvNeXt Species
        c_sp = df[df["model"].str.contains("ConvNeXt") & (df["taxonomic_level"] == "Species")].iloc[0]
        self.assertAlmostEqual(c_sp["backbone_m"], 135.98, places=2)
        self.assertAlmostEqual(c_sp["fc_params_m"], 28.04, places=2)
        self.assertAlmostEqual(c_sp["parameters_m"], 164.01, places=2)

        # ConvNeXt Genus
        c_gen = df[df["model"].str.contains("ConvNeXt") & (df["taxonomic_level"] == "Genus")].iloc[0]
        self.assertAlmostEqual(c_gen["backbone_m"], 135.98, places=2)
        self.assertAlmostEqual(c_gen["fc_params_m"], 5.54, places=2)
        self.assertAlmostEqual(c_gen["parameters_m"], 141.51, places=2)

        # Legacy 2D parameters must not be present
        self.assertNotIn(23.5, df["parameters_m"].values)
        self.assertNotIn(28.1, df["parameters_m"].values)

    def test_r08_table_2_parameters_reconciliation(self):
        """C2: generate_table_2_profiling reports exact species parameters with backbone breakdown."""
        df = generate_table_2_profiling()
        self.assertIsInstance(df, pd.DataFrame)

        r50 = df[df["model"] == "ResNet50 1D"].iloc[0]
        self.assertAlmostEqual(r50["backbone_m"], 15.96, places=2)
        self.assertAlmostEqual(r50["fc_species_m"], 28.04, places=2)
        self.assertAlmostEqual(r50["parameters_m"], 43.99, places=2)

        cnx = df[df["model"] == "ConvNeXt 1D"].iloc[0]
        self.assertAlmostEqual(cnx["backbone_m"], 135.98, places=2)
        self.assertAlmostEqual(cnx["fc_species_m"], 28.04, places=2)
        self.assertAlmostEqual(cnx["parameters_m"], 164.01, places=2)

        self.assertNotIn(23.5, df["parameters_m"].values)
        self.assertNotIn(28.1, df["parameters_m"].values)

    def test_r08_manuscript_latex_parameters_and_text(self):
        """C3: manuscript submission and review sources contain exact parameters and architecture text."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_paper = Path(tmp_dir) / "paper"
            review_tex, submission_tex = build_manuscript_variants(tmp_paper)

            for tex_path in (review_tex, submission_tex):
                content = tex_path.read_text(encoding="utf-8")

                # Must contain reconciled parameters
                self.assertIn("43.99", content)
                self.assertIn("164.01", content)
                self.assertIn("21.49", content)
                self.assertIn("141.51", content)

                # Must not contain legacy parameters in tables
                self.assertNotIn("& 23.5 &", content)
                self.assertNotIn("& 28.1 &", content)

                # Must document HybridConvNeXt architecture
                self.assertIn("HybridConvNeXt", content)
                self.assertIn("135.98", content)
                self.assertIn("8192", content)

    def test_r08_manuscript_submission_pdf_compilation(self):
        """C4: compile_manuscript compiles paper/manuscript_submission.tex cleanly with exit code 0."""
        paper_dir = Path("paper")
        build_manuscript_variants(paper_dir)
        submission_tex = paper_dir / "manuscript_submission.tex"
        self.assertTrue(submission_tex.exists())

        retcode = compile_manuscript(submission_tex)
        self.assertEqual(retcode, 0)
        self.assertTrue((paper_dir / "manuscript_submission.pdf").exists())

    def test_r08_strict_empirical_rejects_smoke_test(self):
        """C5: generate_all_tables(..., strict_empirical=True) raises RuntimeError on smoke test manifests."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            out_dir = tmp_path / "tables"
            mock_results = tmp_path / "results"
            mock_results.mkdir(parents=True)

            # 1. Non-existent results dir raises RuntimeError
            with self.assertRaises(RuntimeError):
                generate_all_tables(out_dir, results_dir=tmp_path / "non_existent", strict_empirical=True)

            # 2. Empty results dir (no manifests) raises RuntimeError
            with self.assertRaises(RuntimeError):
                generate_all_tables(out_dir, results_dir=mock_results, strict_empirical=True)

            # 3. Smoke test manifest raises RuntimeError
            smoke_manifest = mock_results / "manifest_test.json"
            smoke_manifest.write_text(json.dumps({"is_smoke_test": True, "dataset": "synthetic"}), encoding="utf-8")
            with self.assertRaises(RuntimeError):
                generate_all_tables(out_dir, results_dir=mock_results, strict_empirical=True)

            # 4. Existing results_reproduction has smoke test manifests and must raise
            with self.assertRaises(RuntimeError):
                generate_all_tables(out_dir, results_dir=Path("results_reproduction"), strict_empirical=True)

    def test_r08_generate_all_tables_and_reconciliation_report(self):
        """C6: generate_all_tables loads available artifacts, emits 12 tables and reconciliation report."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_dir = Path(tmp_dir) / "tables"
            tables = generate_all_tables(
                output_dir=out_dir,
                results_dir=Path("results_reproduction"),
                strict_empirical=False,
                formats=("csv", "md", "tex"),
            )

            self.assertEqual(len(tables), 12)

            # All 12 tables serialized in all 3 formats
            for i in range(1, 13):
                self.assertTrue((out_dir / f"table_{i}.csv").exists(), f"table_{i}.csv missing")
                self.assertTrue((out_dir / f"table_{i}.md").exists(), f"table_{i}.md missing")
                self.assertTrue((out_dir / f"table_{i}.tex").exists(), f"table_{i}.tex missing")

            # Reconciliation report files
            csv_rep = out_dir / "reconciliation_report.csv"
            md_rep = out_dir / "reconciliation_report.md"
            self.assertTrue(csv_rep.exists())
            self.assertTrue(md_rep.exists())

            df_rep = pd.read_csv(csv_rep)
            self.assertEqual(len(df_rep), 12)
            self.assertIn("table_id", df_rep.columns)
            self.assertIn("empirical_status", df_rep.columns)
            self.assertIn("artifact_source", df_rep.columns)

            # Tables 1 and 2 report reconciled parameters
            status_map = dict(zip(df_rep["table_id"], df_rep["empirical_status"]))
            self.assertEqual(status_map[1], "RECONCILED_PARAMETERS")
            self.assertEqual(status_map[2], "RECONCILED_PARAMETERS")

            # Dynamic artifacts loaded from results_reproduction
            self.assertEqual(status_map[5], "DYNAMIC_ARTIFACT_LOADED")
            self.assertEqual(status_map[12], "DYNAMIC_ARTIFACT_LOADED")


if __name__ == "__main__":
    unittest.main()

