import os
import sys
import unittest
from pathlib import Path
import tempfile

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from viral_cnn.reporting.figures import (
    generate_all_figures,
    verify_latex_figure_paths,
)


class TestReportingFigures(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.output_dir = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_generate_all_figures_assets(self):
        """C9: generate_all_figures generates publication-grade assets for Figures 1-5."""
        fig_paths = generate_all_figures(self.output_dir)
        self.assertEqual(len(fig_paths), 5)
        for i in range(1, 6):
            expected = self.output_dir / f"fig_{i}.png"
            self.assertTrue(expected.exists(), f"Figure {i} missing from {self.output_dir}")
            self.assertGreater(expected.stat().st_size, 1000)

    def test_fig_3_panel_e_convnext_noise_distinct(self):
        """C10: In Fig. 3, panel (e) displays verified ConvNeXt noise-aware without duplicating ResNet50."""
        fig_paths = generate_all_figures(self.output_dir)
        fig_3_panels = self.output_dir / "fig_3_panels"
        self.assertTrue(fig_3_panels.exists())
        panel_d = fig_3_panels / "fig_3_panel_d_resnet50_noise.png"
        panel_e = fig_3_panels / "fig_3_panel_e_convnext_noise.png"
        self.assertTrue(panel_d.exists())
        self.assertTrue(panel_e.exists())
        # They must not be binary identical
        self.assertNotEqual(panel_d.read_bytes(), panel_e.read_bytes())

    def test_verify_figure_case_sensitivity_matches(self):
        """C11: verify_latex_figure_paths asserts all image paths exist with exact case matching."""
        # Create a sample tex file with exact paths
        tex_path = self.output_dir / "sample.tex"
        img_path = self.output_dir / "ConvNeXt_model.png"
        img_path.write_bytes(b"PNGDATA")
        tex_content = r"\includegraphics{ConvNeXt_model.png}"
        tex_path.write_text(tex_content, encoding="utf-8")

        matches, mismatches = verify_latex_figure_paths(tex_path, self.output_dir)
        self.assertEqual(len(matches), 1)
        self.assertEqual(len(mismatches), 0)

    def test_verify_figure_case_sensitivity_failure(self):
        """C12: verify_latex_figure_paths reports fatal error on missing or case-mismatched figure."""
        tex_path = self.output_dir / "sample_fail.tex"
        img_path = self.output_dir / "ConvNeXt_model.png"
        img_path.write_bytes(b"PNGDATA")
        # In tex, refer to ConvNext_model.png (lowercase x)
        tex_content = r"\includegraphics{ConvNext_model.png}"
        tex_path.write_text(tex_content, encoding="utf-8")

        matches, mismatches = verify_latex_figure_paths(tex_path, self.output_dir)
        self.assertEqual(len(mismatches), 1)
        self.assertEqual(mismatches[0]["status"], "case_mismatch")


if __name__ == "__main__":
    unittest.main()
