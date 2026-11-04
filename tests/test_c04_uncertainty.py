"""Scientific contracts for C04 uncertainty and sealed calibration."""
import sys
from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from viral_cnn.eval.uncertainty import bootstrap_classification_intervals
from viral_cnn.eval.calibration import TemperatureScaler, compute_reliability_diagram, save_reliability_diagram
from viral_cnn.eval.runner import EvaluationResult, save_c04_query_evidence
from viral_cnn.eval.metrics import ClassificationMetrics


class TestC04Uncertainty(unittest.TestCase):
    def test_bootstrap_matches_point_estimates_and_has_top5(self):
        y = np.array([0, 0, 1, 1, 2, 2])
        p = np.array([0, 1, 1, 0, 2, 2])
        result = bootstrap_classification_intervals(y, p, 4, top5_hit=[1, 0, 1, 0, 1, 1], repetitions=80)
        self.assertAlmostEqual(result["estimates"]["accuracy"]["estimate"], 4 / 6)
        self.assertEqual(result["estimates"]["macro_f1"]["declared_classes"], 4)
        self.assertEqual(result["estimates"]["macro_f1"]["present_classes"], 3)
        self.assertLessEqual(result["estimates"]["accuracy"]["ci_lower"], 4 / 6)

    def test_temperature_requires_provenance_match_and_seal(self):
        scaler = TemperatureScaler()
        with self.assertRaises(ValueError):
            scaler.fit(torch.tensor([[1., 0.]]), [0], validation_provenance={"split": "validation", "identity": "same"}, test_provenance={"identity": "same"})
        scaler.fit(torch.tensor([[1., 0.], [0., 1.]]), [0, 1], validation_provenance={"split": "validation", "identity": "val-hash"})
        scaler.seal()
        with self.assertRaises(RuntimeError):
            scaler.fit(torch.tensor([[1., 0.]]), [0])
        out = scaler.transform(torch.tensor([[1., 0.]]))
        self.assertEqual(out["temperature_state"], "SEALED")

    def test_empty_bins_are_not_plotted_as_observations_and_plot_failure_has_no_png_text(self):
        bins = compute_reliability_diagram(np.array([[.9,.1],[.8,.2]]), [0,0], n_bins=15)
        self.assertTrue(all(b.accuracy == 0 for b in bins if b.count == 0))
        with tempfile.TemporaryDirectory() as d:
            out = save_reliability_diagram(bins, d)
            if out["plot"]:
                self.assertTrue(Path(out["plot"]).read_bytes().startswith(b"\x89PNG"))
            else:
                self.assertFalse(list(Path(d).glob("*.png")))

    def test_compact_query_evidence_manifest_hashes_rows(self):
        result = EvaluationResult(
            model_name="fixture", noise_rate=0.0, effective_noise_rate=0.0, seed=42,
            metrics=ClassificationMetrics(100.0, 100.0, 1.0, 1.0, 0.0, 2, 2),
            f1_bands={}, predictions=[], y_true_arr=np.array([0, 1]), y_pred_arr=np.array([0, 1]),
            confs_arr=np.array([.8, .9]), top5_hit_arr=np.array([True, True]),
            query_ids_arr=np.array(["partition-a:0", "partition-a:1"]),
        )
        with tempfile.TemporaryDirectory() as d:
            out = save_c04_query_evidence(result, Path(d) / "evidence.parquet", checkpoint="ckpt", split="test", rank="species", num_classes=2)
            manifest = __import__("json").loads(Path(out["manifest"]).read_text())
            self.assertEqual(manifest["n"], 2)
            self.assertEqual(manifest["evidence_sha256"], out["sha256"])
            self.assertTrue(Path(out["evidence"]).is_file())


if __name__ == "__main__":
    unittest.main()
