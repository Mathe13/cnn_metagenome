"""Tests for C05: Publication Figures F3 and F5, Selection and Readiness Checklist."""

from __future__ import annotations

import unittest
from pathlib import Path
import tempfile
import pandas as pd

from viral_cnn.reporting.figures import (
    _create_pure_python_png,
    _inspect_png,
    verify_figure_publication_readiness,
)
from viral_cnn.reporting.publication import PublicationValidationError, render_figures
from viral_cnn.reporting.transformations import (
    FIVE_VARIANTS_SPECIES,
    transform_figure_3_paired_motif_perturbation,
    transform_figure_5_motif_cooccurrence,
)


class TestC05FigureReadiness(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    # S1 - F3 5-panel validation and rendering
    def test_f3_accepts_five_variants(self):
        """C1: transform_figure_3 accepts data containing exactly the 5 species model variants."""
        df = pd.DataFrame({
            "panel": ["a", "b", "c", "d", "e"],
            "variant": FIVE_VARIANTS_SPECIES,
            "motif": ["TATAA", "GCGCG", "CGCTA", "ATATA", "CCGGA"],
            "activation_drop": [0.15, 0.25, 0.05, 0.40, 0.30],
        })
        transformed = transform_figure_3_paired_motif_perturbation(df, {})
        self.assertEqual(len(transformed), 5)
        self.assertEqual(set(transformed["variant"]), set(FIVE_VARIANTS_SPECIES))

    def test_f3_rejects_missing_variants(self):
        """C2: transform_figure_3 rejects datasets omitting required variants (e.g. ConvNeXt_F)."""
        df_incomplete = pd.DataFrame({
            "panel": ["a", "b", "c", "d"],
            "variant": ["ResNet50_Clean", "ConvNeXt_Z", "ConvNeXt_T", "ResNet50_F"],
            "motif": ["TATAA", "GCGCG", "CGCTA", "ATATA"],
            "activation_drop": [0.15, 0.25, 0.05, 0.40],
        })
        with self.assertRaises(PublicationValidationError) as ctx:
            transform_figure_3_paired_motif_perturbation(df_incomplete, {})
        self.assertIn("missing species model variants", str(ctx.exception))
        self.assertIn("ConvNeXt_F", str(ctx.exception))

    def test_f3_rendering_rejects_incomplete_panels(self):
        """C3: render_figures fails if F3 panel set does not match configured panel-variant mapping."""
        csv_path = self.tmp_path / "f3_missing_panel.csv"
        df = pd.DataFrame({
            "panel": ["a", "b", "c", "d"],
            "variant": ["ResNet50_Clean", "ConvNeXt_Z", "ConvNeXt_T", "ResNet50_F"],
            "motif": ["TATAA", "GCGCG", "CGCTA", "ATATA"],
            "activation_drop": [0.15, 0.25, 0.05, 0.40],
        })
        df.to_csv(csv_path, index=False)

        config = {
            "items": [{
                "item_id": "F3",
                "plot": {"kind": "panel_grid", "x": "motif", "y": "activation_drop"},
                "expected": {
                    "panels": ["a", "b", "c", "d", "e"],
                    "panel_variants": {
                        "a": "ResNet50_Clean",
                        "b": "ConvNeXt_Z",
                        "c": "ConvNeXt_T",
                        "d": "ResNet50_F",
                        "e": "ConvNeXt_F",
                    },
                },
            }]
        }
        rows = [{
            "item_id": "F3",
            "status": "DERIVED_VERIFIED",
            "artifact": str(csv_path),
        }]
        out_dir = self.tmp_path / "figures"
        with self.assertRaises(PublicationValidationError) as ctx:
            render_figures(config, out_dir, rows)
        self.assertIn("artifact panel set differs from configured panel set", str(ctx.exception))

    # S2 - F5 2D geometric coordinates
    def test_f5_accepts_valid_edge_coordinates(self):
        """C4: transform_figure_5 accepts valid network edge coordinates (x1, y1, x2, y2)."""
        df = pd.DataFrame({
            "x1": [0.1, 0.5, 0.8],
            "y1": [0.2, 0.4, 0.9],
            "x2": [0.3, 0.7, 0.2],
            "y2": [0.4, 0.6, 0.1],
            "weight": [0.85, 0.60, 0.92],
        })
        transformed = transform_figure_5_motif_cooccurrence(df, {})
        self.assertEqual(len(transformed), 3)
        self.assertTrue({"x1", "y1", "x2", "y2"}.issubset(transformed.columns))

    def test_f5_rejects_missing_coordinates(self):
        """C5: transform_figure_5 rejects tabular data without 2D network edge coordinates."""
        df_bars = pd.DataFrame({
            "motif_pair": ["TATAA-GCGCG", "CGCTA-ATATA"],
            "synergy": [0.75, 0.42],
        })
        with self.assertRaises(PublicationValidationError) as ctx:
            transform_figure_5_motif_cooccurrence(df_bars, {})
        self.assertIn("missing network edge coordinates", str(ctx.exception))

    def test_f5_renders_network_graph(self):
        """C6: render_figures draws edges and nodes from x1, y1, x2, y2 coordinates."""
        csv_path = self.tmp_path / "f5_edges.csv"
        df = pd.DataFrame({
            "x1": [0.1, 0.5],
            "y1": [0.2, 0.4],
            "x2": [0.3, 0.7],
            "y2": [0.4, 0.6],
            "weight": [0.85, 0.60],
        })
        df.to_csv(csv_path, index=False)

        config = {
            "items": [{
                "item_id": "F5",
                "plot": {
                    "kind": "network_edges",
                    "x1": "x1",
                    "y1": "y1",
                    "x2": "x2",
                    "y2": "y2",
                    "title": "Figure 5 Network",
                },
            }]
        }
        rows = [{
            "item_id": "F5",
            "status": "DERIVED_VERIFIED",
            "artifact": str(csv_path),
        }]
        out_dir = self.tmp_path / "figures"
        outputs = render_figures(config, out_dir, rows)
        self.assertIn("F5", outputs)
        self.assertTrue(outputs["F5"].is_file())
        self.assertTrue(outputs["F5"].stat().st_size > 0)

    # S3 - Figure publication readiness checklist
    def test_verify_figure_readiness_valid_png(self):
        """C7: verify_figure_publication_readiness verifies valid PNG headers and dimensions."""
        png_path = self.tmp_path / "fig_1.png"
        png_path.write_bytes(_create_pure_python_png(400, 300, (10, 20, 30)))

        report = verify_figure_publication_readiness({"F1": png_path}, {})
        self.assertEqual(report["status"], "PASS")
        self.assertTrue(report["figures"]["F1"]["valid"])
        self.assertEqual(report["figures"]["F1"]["width"], 400)
        self.assertEqual(report["figures"]["F1"]["height"], 300)

    def test_verify_figure_readiness_detects_missing_or_corrupt(self):
        """C8: verify_figure_publication_readiness detects missing or 0-byte corrupt image files."""
        corrupt_path = self.tmp_path / "corrupt.png"
        corrupt_path.write_bytes(b"")
        missing_path = self.tmp_path / "nonexistent.png"

        report = verify_figure_publication_readiness({
            "F1": corrupt_path,
            "F2": missing_path,
        }, {})
        self.assertEqual(report["status"], "FAIL")
        self.assertFalse(report["figures"]["F1"]["valid"])
        self.assertFalse(report["figures"]["F2"]["valid"])
        self.assertEqual(len(report["failures"]), 2)

    def test_verify_figure_readiness_checks_f2_disclaimer(self):
        """C9: verify_figure_publication_readiness checks for qualitative projection disclaimer in F2 metadata."""
        png_path = self.tmp_path / "fig_2.png"
        png_path.write_bytes(_create_pure_python_png(200, 200, (50, 50, 50)))

        # Missing disclaimer fails
        report_fail = verify_figure_publication_readiness({"F2": png_path}, {"F2": {"disclaimer": ""}})
        self.assertEqual(report_fail["status"], "FAIL")
        self.assertFalse(report_fail["disclaimers"]["F2"]["has_qualitative_disclaimer"])

        # Valid qualitative disclaimer passes
        report_pass = verify_figure_publication_readiness(
            {"F2": png_path},
            {"F2": {"disclaimer": "Qualitative visualization only; metric analysis performed in high-dimensional space."}},
        )
        self.assertEqual(report_pass["status"], "PASS")
        self.assertTrue(report_pass["disclaimers"]["F2"]["has_qualitative_disclaimer"])

    def test_verify_figure_readiness_validates_selection_parameters(self):
        """C10: verify_figure_publication_readiness validates presence of seed, ids, sampling, and filters."""
        png_path = self.tmp_path / "fig_1.png"
        png_path.write_bytes(_create_pure_python_png(200, 200, (50, 50, 50)))

        # Incomplete selection fails
        meta_incomplete = {
            "F1": {
                "selection": {"seed": 42, "sampling": "stratified"},
            }
        }
        report_fail = verify_figure_publication_readiness({"F1": png_path}, meta_incomplete)
        self.assertEqual(report_fail["status"], "FAIL")
        self.assertFalse(report_fail["selection_verified"])

        # Complete selection passes
        meta_complete = {
            "F1": {
                "selection": {
                    "seed": 42,
                    "ids": ["ResNet50_Clean", "ConvNeXt_Z"],
                    "sampling": "deterministic",
                    "filters": {"noise_levels": [0.0, 0.02, 0.05, 0.10, 0.15]},
                }
            }
        }
        report_pass = verify_figure_publication_readiness({"F1": png_path}, meta_complete)
        self.assertEqual(report_pass["status"], "PASS")
        self.assertTrue(report_pass["selection_verified"])


if __name__ == "__main__":
    unittest.main()
