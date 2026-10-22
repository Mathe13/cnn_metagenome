"""
Unit and contract tests for S09: Calibration and Confidence by Noise Level.
"""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from viral_cnn.eval.calibration import (
    CalibrationMetrics,
    ReliabilityDiagramBin,
    compute_calibration_metrics,
    compute_reliability_diagram,
    save_reliability_diagram,
    TemperatureScaler,
    audit_model_calibration_by_noise,
)


class TestCalibration(unittest.TestCase):

    # --- S1: Formal Probabilistic Calibration Metrics (C1 - C7) ---

    def test_compute_calibration_metrics_ece_equal_width(self):
        """C1: compute_calibration_metrics computes Top-1 ECE using declared equal-width binning (M=15)."""
        # Controlled synthetic probs: 10 samples, 3 classes
        probs = np.array([
            [0.9, 0.05, 0.05],
            [0.85, 0.1, 0.05],
            [0.8, 0.1, 0.1],
            [0.75, 0.2, 0.05],
            [0.6, 0.3, 0.1],
            [0.55, 0.4, 0.05],
            [0.4, 0.5, 0.1],
            [0.3, 0.6, 0.1],
            [0.2, 0.7, 0.1],
            [0.1, 0.8, 0.1],
        ])
        y_true = np.array([0, 0, 0, 1, 0, 1, 1, 1, 1, 1])

        metrics = compute_calibration_metrics(
            probs=probs,
            y_true=y_true,
            n_bins=15,
            binning_strategy="equal_width",
        )
        self.assertIsInstance(metrics, CalibrationMetrics)
        self.assertGreaterEqual(metrics.ece, 0.0)
        self.assertLessEqual(metrics.ece, 1.0)
        self.assertEqual(metrics.n_bins, 15)
        self.assertEqual(metrics.binning_strategy, "equal_width")
        self.assertAlmostEqual(metrics.sum_bin_weights, 1.0, places=5)

    def test_compute_calibration_metrics_ece_equal_mass(self):
        """C2: compute_calibration_metrics computes Top-1 ECE using declared equal-mass (adaptive) binning."""
        np.random.seed(42)
        n_samples = 100
        n_classes = 5
        logits = np.random.randn(n_samples, n_classes)
        probs = np.exp(logits) / np.sum(np.exp(logits), axis=1, keepdims=True)
        y_true = np.random.randint(0, n_classes, size=n_samples)

        metrics = compute_calibration_metrics(
            probs=probs,
            y_true=y_true,
            n_bins=10,
            binning_strategy="equal_mass",
        )
        self.assertIsInstance(metrics, CalibrationMetrics)
        self.assertGreaterEqual(metrics.ece, 0.0)
        self.assertLessEqual(metrics.ece, 1.0)
        self.assertEqual(metrics.binning_strategy, "equal_mass")
        self.assertAlmostEqual(metrics.sum_bin_weights, 1.0, places=5)

    def test_compute_calibration_metrics_mce(self):
        """C3: compute_calibration_metrics computes Maximum Calibration Error (MCE) across non-empty bins."""
        probs = np.array([
            [0.95, 0.05],
            [0.95, 0.05],
            [0.10, 0.90],
            [0.10, 0.90],
        ])
        # In bin [0.9, 1.0]: samples 0 and 1 have conf 0.95, true label is 0 (correct), acc=1.0 -> gap=|1.0-0.95|=0.05
        # and samples 2 and 3 predict class 1 with conf 0.90, true label is 0 (wrong) -> acc=0.0 -> gap=|0.0-0.90|=0.90
        y_true = np.array([0, 0, 0, 0])

        metrics = compute_calibration_metrics(
            probs=probs,
            y_true=y_true,
            n_bins=10,
            binning_strategy="equal_width",
        )
        self.assertGreater(metrics.mce, 0.0)
        self.assertGreaterEqual(metrics.mce, metrics.ece)

    def test_compute_calibration_metrics_nll_clamped(self):
        """C4: compute_calibration_metrics computes Negative Log-Likelihood (NLL) with epsilon clamping."""
        # 1 sample with 0 predicted probability for ground truth
        probs = np.array([
            [1.0, 0.0],
        ])
        y_true = np.array([1]) # true class is 1, but prob is 0.0

        metrics = compute_calibration_metrics(
            probs=probs,
            y_true=y_true,
            n_bins=5,
        )
        # NLL should not be inf or nan
        self.assertFalse(np.isinf(metrics.nll))
        self.assertFalse(np.isnan(metrics.nll))
        self.assertGreater(metrics.nll, 20.0) # clamped to ~ -log(1e-15) ~= 34.5

    def test_compute_calibration_metrics_brier_memory_efficient(self):
        """C5: compute_calibration_metrics calculates exact multiclass Brier score using memory-efficient identity."""
        np.random.seed(42)
        n_samples = 200
        n_classes = 100 # test across multiple classes
        logits = np.random.randn(n_samples, n_classes)
        probs = np.exp(logits) / np.sum(np.exp(logits), axis=1, keepdims=True)
        y_true = np.random.randint(0, n_classes, size=n_samples)

        # Reference calculation with dense one-hot
        one_hot = np.zeros((n_samples, n_classes))
        one_hot[np.arange(n_samples), y_true] = 1.0
        expected_brier = float(np.mean(np.sum((probs - one_hot) ** 2, axis=1)))

        metrics = compute_calibration_metrics(
            probs=probs,
            y_true=y_true,
            n_bins=10,
        )
        self.assertAlmostEqual(metrics.brier_score, expected_brier, places=5)

    def test_compute_calibration_metrics_perfect_predictions_zero_error(self):
        """C6: compute_calibration_metrics returns 0.0 ECE, 0.0 MCE, and 0.0 Brier score when all probs=1.0 on correct."""
        n_samples = 50
        n_classes = 10
        probs = np.zeros((n_samples, n_classes))
        y_true = np.array([i % n_classes for i in range(n_samples)])
        probs[np.arange(n_samples), y_true] = 1.0

        metrics = compute_calibration_metrics(
            probs=probs,
            y_true=y_true,
            n_bins=15,
        )
        self.assertAlmostEqual(metrics.ece, 0.0, places=6)
        self.assertAlmostEqual(metrics.mce, 0.0, places=6)
        self.assertAlmostEqual(metrics.brier_score, 0.0, places=6)
        self.assertAlmostEqual(metrics.nll, 0.0, places=4)
        self.assertAlmostEqual(metrics.top1_accuracy, 100.0, places=4)
        self.assertAlmostEqual(metrics.top1_confidence, 1.0, places=4)

    def test_compute_calibration_metrics_overconfidence_partition(self):
        """C7: compute_calibration_metrics computes mean confidence and accuracy separately for correct and incorrect."""
        probs = np.array([
            [0.9, 0.1], # correct, conf=0.9
            [0.8, 0.2], # correct, conf=0.8
            [0.7, 0.3], # incorrect (true=1), conf=0.7
            [0.6, 0.4], # incorrect (true=1), conf=0.6
        ])
        y_true = np.array([0, 0, 1, 1])

        metrics = compute_calibration_metrics(
            probs=probs,
            y_true=y_true,
            n_bins=5,
        )
        self.assertAlmostEqual(metrics.conf_on_correct, 0.85, places=4)
        self.assertAlmostEqual(metrics.conf_on_incorrect, 0.65, places=4)
        self.assertGreater(metrics.overconfidence_gap, 0.0)

    # --- S2: Reliability Diagrams and Bin Analysis (C8 - C10) ---

    def test_compute_reliability_diagram_structured_table(self):
        """C8: compute_reliability_diagram returns bin edges, count, mean confidence, accuracy, and gap."""
        np.random.seed(123)
        probs = np.random.dirichlet(np.ones(5), size=100)
        y_true = np.random.randint(0, 5, size=100)

        bins = compute_reliability_diagram(
            probs=probs,
            y_true=y_true,
            n_bins=10,
            binning_strategy="equal_width",
        )
        self.assertIsInstance(bins, list)
        self.assertEqual(len(bins), 10)
        for b in bins:
            self.assertIsInstance(b, ReliabilityDiagramBin)
            self.assertGreaterEqual(b.bin_upper, b.bin_lower)
            self.assertGreaterEqual(b.calibration_gap, 0.0)

    def test_compute_reliability_diagram_sample_conservation(self):
        """C9: compute_reliability_diagram allocates every sample to exactly one bin so sum of counts equals N."""
        n_samples = 377
        np.random.seed(999)
        probs = np.random.dirichlet(np.ones(4), size=n_samples)
        y_true = np.random.randint(0, 4, size=n_samples)

        bins = compute_reliability_diagram(probs=probs, y_true=y_true, n_bins=15)
        total_binned = sum(b.count for b in bins)
        self.assertEqual(total_binned, n_samples)

    def test_save_reliability_diagram_plot_and_table(self):
        """C10: save_reliability_diagram exports structured JSON/CSV data and renders a reliability diagram plot."""
        np.random.seed(55)
        probs = np.random.dirichlet(np.ones(3), size=60)
        y_true = np.random.randint(0, 3, size=60)
        bins = compute_reliability_diagram(probs=probs, y_true=y_true, n_bins=10)

        with tempfile.TemporaryDirectory() as tmp_dir:
            out_path = Path(tmp_dir)
            files = save_reliability_diagram(
                bins=bins,
                output_dir=out_path,
                title="Test Reliability Diagram",
            )
            self.assertTrue(Path(files["csv"]).exists())
            self.assertTrue(Path(files["json"]).exists())
            self.assertTrue(Path(files["plot"]).exists())

    # --- S3: Validation-Only Temperature Scaling (C11 - C14) ---

    def test_temperature_scaler_fit_minimizes_validation_nll(self):
        """C11: TemperatureScaler.fit optimizes scalar temperature T > 0 strictly minimizing NLL on validation logits."""
        torch.manual_seed(42)
        n_val = 500
        n_classes = 10
        # Create deliberately overconfident validation logits (scale factor 3.0)
        val_logits = torch.randn(n_val, n_classes) * 3.0
        val_labels = torch.randint(0, n_classes, (n_val,))

        scaler = TemperatureScaler()
        scaler.fit(val_logits, val_labels, split_name="validation")
        self.assertGreater(scaler.temperature, 0.0)
        # For overly sharp/confident logits with random labels, optimal T should be > 1.0 to soften logits
        self.assertGreater(scaler.temperature, 1.0)

    def test_temperature_scaler_rejects_test_fitting(self):
        """C12: TemperatureScaler.fit rejects fitting on test data without explicit validation split, raising ValueError."""
        logits = torch.randn(50, 5)
        labels = torch.randint(0, 5, (50,))

        scaler = TemperatureScaler()
        with self.assertRaises(ValueError):
            scaler.fit(logits, labels, split_name="test")

        with self.assertRaises(ValueError):
            scaler.fit(logits, labels, split_name=None)

    def test_temperature_scaler_transform_reduces_test_ece(self):
        """C13: TemperatureScaler.transform applies z/T scaling to test logits and computes post-calibration metrics."""
        torch.manual_seed(101)
        # Create validation and test sets from same overconfident distribution
        val_logits = torch.randn(300, 5) * 2.5
        val_labels = torch.randint(0, 5, (300,))
        test_logits = torch.randn(300, 5) * 2.5
        test_labels = torch.randint(0, 5, (300,))

        scaler = TemperatureScaler()
        scaler.fit(val_logits, val_labels, split_name="validation")

        cal_res = scaler.transform(test_logits, test_labels)
        self.assertIn("uncalibrated_metrics", cal_res)
        self.assertIn("calibrated_metrics", cal_res)
        self.assertIn("temperature", cal_res)
        self.assertGreater(cal_res["temperature"], 0.0)
        # Calibrated NLL should be lower or equal to uncalibrated NLL
        self.assertLessEqual(cal_res["calibrated_metrics"]["nll"], cal_res["uncalibrated_metrics"]["nll"])

    def test_temperature_scaler_identity_at_t1(self):
        """C14: TemperatureScaler.transform leaves probabilities and predictions unchanged when T=1.0."""
        scaler = TemperatureScaler()
        scaler.temperature = 1.0
        test_logits = torch.randn(50, 4)
        test_labels = torch.randint(0, 4, (50,))

        cal_res = scaler.transform(test_logits, test_labels)
        self.assertAlmostEqual(
            cal_res["uncalibrated_metrics"]["ece"],
            cal_res["calibrated_metrics"]["ece"],
            places=5,
        )
        self.assertAlmostEqual(
            cal_res["uncalibrated_metrics"]["nll"],
            cal_res["calibrated_metrics"]["nll"],
            places=5,
        )

    # --- S4: Controlled Noise Calibration Audit and Scientific Moderation (C15 - C19) ---

    def test_audit_model_calibration_by_noise_multi_rate(self):
        """C15: audit_model_calibration_by_noise evaluates calibration across models and nominal noise levels."""
        # Synthetic mock model returning random logits
        class MockModel(torch.nn.Module):
            def __init__(self, num_classes=5):
                super().__init__()
                self.num_classes = num_classes
            def forward(self, x):
                batch_size = x.shape[0]
                return torch.randn(batch_size, self.num_classes)

        mock_models = {
            "resnet_clean_species": MockModel(num_classes=5),
            "convnext_zero_species": MockModel(num_classes=5),
        }
        sequences = ["ACGTACGT" * 64] * 20
        labels = [i % 5 for i in range(20)]
        noise_rates = [0.0, 0.05, 0.10]

        report = audit_model_calibration_by_noise(
            models_dict=mock_models,
            sequences=sequences,
            labels=labels,
            noise_rates=noise_rates,
            num_classes=5,
            seed=42,
        )
        self.assertIn("models", report)
        self.assertIn("resnet_clean_species", report["models"])
        self.assertIn("convnext_zero_species", report["models"])
        for m_name in report["models"]:
            for r_str in ["noise_0.0", "noise_0.05", "noise_0.1"]:
                self.assertIn(r_str, report["models"][m_name])
                entry = report["models"][m_name][r_str]
                self.assertIn("ece", entry)
                self.assertIn("nll", entry)
                self.assertIn("brier_score", entry)
                self.assertIn("top1_confidence", entry)
                self.assertIn("top1_accuracy", entry)

    def test_calibration_audit_doc_distinguishes_confidence_and_calibration(self):
        """C16: docs/calibration_and_confidence_audit.md documents calibration table, distinguishing confidence from ECE."""
        doc_path = PROJECT_ROOT / "docs" / "calibration_and_confidence_audit.md"
        self.assertTrue(doc_path.exists(), "docs/calibration_and_confidence_audit.md must exist")
        content = doc_path.read_text(encoding="utf-8")

        self.assertIn("Expected Calibration Error", content)
        self.assertIn("ECE", content)
        self.assertIn("Confianca_Media", content)
        self.assertIn("Brier", content)
        self.assertIn("NLL", content)
        self.assertIn("sobreconfiança", content.lower())

    def test_calibration_audit_doc_isolates_label_smoothing_confounds(self):
        """C17: docs/calibration_and_confidence_audit.md analytically isolates label_smoothing=0.1 and dropout confounds."""
        doc_path = PROJECT_ROOT / "docs" / "calibration_and_confidence_audit.md"
        content = doc_path.read_text(encoding="utf-8")

        self.assertIn("label_smoothing", content)
        self.assertIn("0.1", content)
        self.assertIn("dropout", content.lower())
        self.assertIn("Müller", content) # Muller et al. 2019 citation

    def test_calibration_audit_doc_provides_text_moderation_recommendations(self):
        """C18: docs/calibration_and_confidence_audit.md provides explicit recommendations for text revisions in manuscript."""
        doc_path = PROJECT_ROOT / "docs" / "calibration_and_confidence_audit.md"
        content = doc_path.read_text(encoding="utf-8")

        self.assertIn("Recomendações", content)
        self.assertIn("Manuscrito", content)
        self.assertIn("Tabela", content)
        self.assertIn("moderação", content.lower())

    def test_audit_model_calibration_flags_overconfidence_regime(self):
        """C19: audit_model_calibration_by_noise flags overconfidence regimes when conf > acc + 10 pp."""
        # Create an extremely overconfident case: conf = 95%, acc = 20%
        probs = np.zeros((100, 5))
        probs[:, 0] = 0.95
        probs[:, 1:] = 0.05 / 4
        y_true = np.random.randint(1, 5, size=100) # true labels are 1..4, prediction is always 0 (0% acc)

        metrics = compute_calibration_metrics(probs=probs, y_true=y_true, n_bins=10)
        self.assertTrue(metrics.is_overconfident_regime)
        self.assertGreater(metrics.top1_confidence - (metrics.top1_accuracy / 100.0), 0.10)

    # --- S5: CLI Entrypoint and Reproducible Manifests (C20 - C22) ---

    def test_run_calibration_cli_exit_zero_and_artifacts(self):
        """C20: CLI scripts/run_calibration_evaluation.py outputs calibration_metrics.json, csv, and manifest with exit 0."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_path = Path(tmp_dir)
            data_file = out_path / "test_data.csv"
            # Write small dataset
            import pandas as pd
            df = pd.DataFrame({
                "sequence": ["ACGTACGT" * 64] * 10,
                "label": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
                "query_id": [f"q_{i}" for i in range(10)],
            })
            df.to_csv(data_file, index=False)

            cmd = [
                sys.executable,
                str(PROJECT_ROOT / "scripts" / "run_calibration_evaluation.py"),
                "--input", str(data_file),
                "--model", "resnet_clean_species",
                "--noise-rates", "0.0", "0.05",
                "--output-dir", str(out_path / "run_out"),
            ]
            res = subprocess.run(cmd, capture_output=True, text=True)
            self.assertEqual(res.returncode, 0, f"CLI stderr: {res.stderr}")
            self.assertTrue((out_path / "run_out" / "calibration_summary.json").exists())
            self.assertTrue((out_path / "run_out" / "run_manifest.json").exists())

    def test_run_calibration_cli_invalid_args_exit_2(self):
        """C21: CLI scripts/run_calibration_evaluation.py exits with code 2 on invalid arguments or missing files."""
        cmd = [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "run_calibration_evaluation.py"),
            "--input", "non_existent_file.csv",
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        self.assertEqual(res.returncode, 2)

    def test_calibration_pipeline_synthetic_deterministic_execution(self):
        """C22: Calibration evaluation pipeline executes deterministically on synthetic fixtures without GPU."""
        np.random.seed(42)
        n = 100
        probs = np.random.dirichlet(np.ones(10), size=n)
        y_true = np.random.randint(0, 10, size=n)

        m1 = compute_calibration_metrics(probs, y_true, n_bins=15)
        m2 = compute_calibration_metrics(probs, y_true, n_bins=15)
        self.assertEqual(m1.ece, m2.ece)
        self.assertEqual(m1.mce, m2.mce)
        self.assertEqual(m1.brier_score, m2.brier_score)
        self.assertEqual(m1.nll, m2.nll)


if __name__ == "__main__":
    unittest.main()
