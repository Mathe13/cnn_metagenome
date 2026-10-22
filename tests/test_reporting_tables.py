import os
import sys
import unittest
from pathlib import Path
import tempfile
import pandas as pd

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from viral_cnn.reporting.tables import (
    generate_all_tables,
    generate_table_1_metrics,
    generate_table_3_noise_degradation,
    generate_table_4_pairwise_comparison,
    generate_table_8_motifs_summary,
    generate_table_10_boundary_triplets,
    generate_table_11_architecture_correlations,
    generate_table_12_voter_consensus,
)


class TestReportingTables(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.output_dir = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_generate_all_tables_formats(self):
        """C1: generate_all_tables produces Tables 1-12 in csv, md, and tex."""
        tables = generate_all_tables(self.output_dir, formats=("csv", "md", "tex"))
        self.assertEqual(len(tables), 12)
        for i in range(1, 13):
            csv_file = self.output_dir / f"table_{i}.csv"
            md_file = self.output_dir / f"table_{i}.md"
            tex_file = self.output_dir / f"table_{i}.tex"
            self.assertTrue(csv_file.exists(), f"Table {i} CSV missing")
            self.assertTrue(md_file.exists(), f"Table {i} MD missing")
            self.assertTrue(tex_file.exists(), f"Table {i} TeX missing")

    def test_table_1_metrics_and_convnext_f(self):
        """C2: Table 1 contains species/genus metrics to 2 decimal places and ConvNeXt_F clean."""
        df = generate_table_1_metrics()
        self.assertIn("accuracy", df.columns)
        self.assertIn("top_5", df.columns)
        self.assertIn("macro_f1", df.columns)
        # Check presence of ConvNeXt_F
        models = df["model"].tolist()
        self.assertTrue(any("convnext_f" in m.lower() or "fine-tuning" in m.lower() for m in models))
        convnext_f_row = df[df["model"].str.contains("convnext_f|fine-tuning", case=False, regex=True)]
        self.assertFalse(convnext_f_row.empty)
        # Accuracy is 61.14%
        self.assertAlmostEqual(float(convnext_f_row["accuracy"].iloc[0]), 61.14, places=2)

    def test_table_3_noise_degradation_denominator(self):
        """C3: Table 3 reports single-window accuracy across eps in {0, 0.02, 0.05, 0.10, 0.15} with N=1,706,270."""
        df = generate_table_3_noise_degradation()
        for eps in ["0.00", "0.02", "0.05", "0.10", "0.15"]:
            self.assertTrue(any(eps in c for c in df.columns), f"Epsilon {eps} missing from columns")
        self.assertIn("denominator_n", df.columns)
        for d in df["denominator_n"]:
            self.assertEqual(int(d), 1706270)

    def test_table_4_pairwise_ties_and_sum(self):
        """C4: Table 4 reports 11,826 ConvNeXt superior, 1,202 ResNet50 superior, 5 ties, sum=13,033."""
        pairwise = generate_table_4_pairwise_comparison()
        self.assertEqual(pairwise["convnext_superior"], 11826)
        self.assertEqual(pairwise["resnet_superior"], 1202)
        self.assertEqual(pairwise["ties"], 5)
        self.assertEqual(pairwise["total_species"], 13033)
        self.assertEqual(
            pairwise["convnext_superior"] + pairwise["resnet_superior"] + pairwise["ties"],
            pairwise["total_species"],
        )

    def test_table_4_f1_continuous_intervals(self):
        """C5: Table 4 reports F1 score bins using continuous intervals without mathematical gaps."""
        df = generate_table_4_pairwise_comparison()["f1_bands"]
        intervals = df["interval"].tolist()
        expected_intervals = [
            "[0.90, 1.00]",
            "[0.80, 0.90)",
            "[0.50, 0.80)",
            "[0.00, 0.50)",
        ]
        self.assertEqual(intervals, expected_intervals)

    def test_table_8_and_10_motifs_triplets(self):
        """C6: Table 8 and 10 report validated 7-mers, ATG, and stop triplets with exploratory note."""
        t8 = generate_table_8_motifs_summary()
        self.assertIn("consensus_7mer", t8.columns)
        self.assertIn("fdr_significant", t8.columns)
        self.assertIn("mean_specific_drop", t8.columns)

        t10 = generate_table_10_boundary_triplets()
        self.assertIn("atg_count", t10.columns)
        self.assertIn("stop_count", t10.columns)
        self.assertIn("exploratory_notice", t10.attrs)
        self.assertTrue(t10.attrs["exploratory_notice"])

    def test_table_11_correlation_p_values_fdr(self):
        """C7: Table 11 reports exact two-sided p-values and Benjamini-Hochberg FDR adjustments."""
        df = generate_table_11_architecture_correlations()
        self.assertIn("p_value", df.columns)
        self.assertIn("p_fdr_bh", df.columns)
        for p in df["p_value"]:
            self.assertTrue(0.0 <= p <= 1.0)
        for p in df["p_fdr_bh"]:
            self.assertTrue(0.0 <= p <= 1.0)

    def test_table_12_voter_consensus(self):
        """C8: Table 12 reports long-sequence sliding-window Voter consensus performance."""
        df = generate_table_12_voter_consensus()
        self.assertIn("n_sequences", df.columns)
        self.assertIn("stride_bp", df.columns)
        self.assertIn("handle_short", df.columns)
        self.assertIn("accuracy", df.columns)
        self.assertTrue((df["stride_bp"] == 256).all())


if __name__ == "__main__":
    unittest.main()

