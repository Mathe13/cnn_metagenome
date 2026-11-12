"""
Tests for S16: Calibration label fail-fast validation and policy enforcement.
Verifies that closed-set evaluation fails fast when ground-truth labels fall outside
[0, n_classes) under the default policy, and supports explicit opt-in exclusion.
"""

import unittest
import numpy as np

from viral_cnn.eval.calibration import (
    compute_calibration_metrics,
    compute_reliability_diagram,
    CalibrationMetrics,
)


class TestS16CalibrationLabelValidation(unittest.TestCase):
    """Unit tests for calibration label validation and fail-fast policies (S16)."""

    def setUp(self):
        np.random.seed(42)
        self.n_samples = 100
        self.n_classes = 5
        # Generate valid synthetic probabilities
        raw_logits = np.random.randn(self.n_samples, self.n_classes)
        exp_logits = np.exp(raw_logits - np.max(raw_logits, axis=1, keepdims=True))
        self.probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        self.valid_labels = np.random.randint(0, self.n_classes, size=self.n_samples)

    def test_valid_labels_compute_metrics_default_policy(self):
        """C1: compute_calibration_metrics with valid labels under default policy returns metrics with matching n_samples."""
        metrics = compute_calibration_metrics(self.probs, self.valid_labels)
        self.assertIsInstance(metrics, CalibrationMetrics)
        self.assertEqual(metrics.n_samples, self.n_samples)
        self.assertGreaterEqual(metrics.ece, 0.0)
        self.assertLessEqual(metrics.ece, 1.0)
        self.assertGreaterEqual(metrics.nll, 0.0)
        self.assertGreaterEqual(metrics.brier_score, 0.0)

    def test_invalid_labels_fail_fast_default_policy(self):
        """C2: compute_calibration_metrics with labels outside [0, n_classes) under default policy raises ValueError."""
        # 1. Negative label
        invalid_negative = self.valid_labels.copy()
        invalid_negative[10] = -1
        with self.assertRaises(ValueError) as ctx:
            compute_calibration_metrics(self.probs, invalid_negative)
        self.assertIn("Found 1 invalid labels outside [0, 5)", str(ctx.exception))
        self.assertIn("[-1]", str(ctx.exception))

        # 2. Out of bounds label (>= n_classes)
        invalid_oob = self.valid_labels.copy()
        invalid_oob[5] = 5
        invalid_oob[20] = 99
        with self.assertRaises(ValueError) as ctx:
            compute_calibration_metrics(self.probs, invalid_oob)
        self.assertIn("Found 2 invalid labels outside [0, 5)", str(ctx.exception))
        self.assertIn("5", str(ctx.exception))
        self.assertIn("99", str(ctx.exception))

    def test_exclude_policy_filters_invalid_samples(self):
        """C3: compute_calibration_metrics with invalid_label_policy='exclude' filters invalid samples and updates n_samples."""
        corrupted_labels = self.valid_labels.copy()
        corrupted_labels[2] = -1
        corrupted_labels[7] = 5
        corrupted_labels[12] = 100

        metrics = compute_calibration_metrics(
            self.probs,
            corrupted_labels,
            invalid_label_policy="exclude",
        )
        expected_n = self.n_samples - 3
        self.assertEqual(metrics.n_samples, expected_n)
        self.assertGreaterEqual(metrics.ece, 0.0)
        self.assertLessEqual(metrics.ece, 1.0)

        # Confirm exact match with manually filtered valid inputs
        valid_mask = (corrupted_labels >= 0) & (corrupted_labels < self.n_classes)
        filtered_metrics = compute_calibration_metrics(
            self.probs[valid_mask],
            corrupted_labels[valid_mask],
            invalid_label_policy="raise",
        )
        self.assertEqual(metrics.n_samples, filtered_metrics.n_samples)
        self.assertAlmostEqual(metrics.ece, filtered_metrics.ece, places=6)
        self.assertAlmostEqual(metrics.nll, filtered_metrics.nll, places=6)
        self.assertAlmostEqual(metrics.brier_score, filtered_metrics.brier_score, places=6)

    def test_exclude_policy_all_invalid_raises_value_error(self):
        """C4: compute_calibration_metrics with invalid_label_policy='exclude' and 0 valid samples raises ValueError."""
        all_invalid_labels = np.full(self.n_samples, -1, dtype=np.int64)
        with self.assertRaises(ValueError) as ctx:
            compute_calibration_metrics(
                self.probs,
                all_invalid_labels,
                invalid_label_policy="exclude",
            )
        self.assertIn("All samples were excluded due to invalid labels.", str(ctx.exception))

    def test_unrecognized_policy_raises_value_error(self):
        """C5: compute_calibration_metrics with unrecognized policy raises ValueError."""
        with self.assertRaises(ValueError) as ctx:
            compute_calibration_metrics(
                self.probs,
                self.valid_labels,
                invalid_label_policy="clamp",  # type: ignore
            )
        self.assertIn("Invalid invalid_label_policy 'clamp'", str(ctx.exception))
        self.assertIn("Must be 'raise' or 'exclude'", str(ctx.exception))

    def test_reliability_diagram_raises_on_invalid_labels(self):
        """C6: compute_reliability_diagram with invalid labels under invalid_label_policy='raise' raises ValueError."""
        invalid_labels = self.valid_labels.copy()
        invalid_labels[0] = -1
        with self.assertRaises(ValueError) as ctx:
            compute_reliability_diagram(self.probs, invalid_labels, invalid_label_policy="raise")
        self.assertIn("Found 1 invalid labels outside [0, 5)", str(ctx.exception))

    def test_reliability_diagram_exclude_policy_conserves_samples(self):
        """C7: compute_reliability_diagram with invalid_label_policy='exclude' conserves filtered sample count."""
        corrupted_labels = self.valid_labels.copy()
        corrupted_labels[0] = -1
        corrupted_labels[1] = 10
        bins = compute_reliability_diagram(
            self.probs,
            corrupted_labels,
            invalid_label_policy="exclude",
        )
        total_binned = sum(b.count for b in bins)
        self.assertEqual(total_binned, self.n_samples - 2)


if __name__ == "__main__":
    unittest.main()
