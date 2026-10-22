"""
Unit and contract tests for S05: Clean Evaluation, Controlled Noise Perturbation, and Metrics.
"""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import numpy as np
import torch

from viral_cnn.eval.noise import inject_noise, verify_effective_noise_rate
from viral_cnn.eval.metrics import (
    compute_f1_scores,
    compute_classification_metrics,
    compute_species_f1_bands,
    compute_paired_comparison,
)
from viral_cnn.eval.runner import evaluate_model, save_evaluation_artifacts
from viral_cnn.models.resnet import ResNet50_1D

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class TestNoiseAndEvaluation(unittest.TestCase):

    # --- S1: Controlled Noise Injection Engine (C1 - C4) ---

    def test_noise_protocol_documentation_and_proof(self):
        """C1: docs/noise_evaluation_protocol.md documents rates, uniform resampling, and 3*epsilon/4 proof."""
        doc_path = PROJECT_ROOT / "docs" / "noise_evaluation_protocol.md"
        self.assertTrue(doc_path.exists(), "docs/noise_evaluation_protocol.md must exist")
        content = doc_path.read_text(encoding="utf-8")

        for rate in ["0.0", "0.02", "0.05", "0.10", "0.15"]:
            self.assertIn(rate, content, f"Rate {rate} must be documented")

        self.assertIn("3\\epsilon/4", content, "Expected substitution rate formula must be documented")
        self.assertIn("Bernoulli", content)
        self.assertIn("padding", content.lower())

    def test_inject_noise_bernoulli_and_uniform_resampling(self):
        """C2: inject_noise applies Bernoulli selection and uniform 4-base resampling."""
        seqs = torch.tensor([[0, 1, 2, 3, 0, 1, 2, 3]], dtype=torch.long)

        # Zero noise leaves sequence unchanged
        clean = inject_noise(seqs, noise_rate=0.0)
        self.assertTrue(torch.equal(seqs, clean))

        # Noise with seed mutates positions only to valid bases in {0, 1, 2, 3}
        gen = torch.Generator().manual_seed(42)
        noisy = inject_noise(seqs, noise_rate=0.5, generator=gen)
        self.assertEqual(noisy.shape, seqs.shape)
        self.assertTrue(torch.all((noisy >= 0) & (noisy < 4)))

    def test_inject_noise_preserves_padding_and_ambiguous_tokens(self):
        """C3: inject_noise preserves padding and ambiguous tokens (<0) without perturbation."""
        # Sequence with valid bases and padding/ambiguous tokens (-1)
        seqs = torch.tensor([[-1, -1, 0, 1, 2, 3, -1, -1]], dtype=torch.long)

        # Even at 100% nominal noise rate, padding positions must remain -1
        gen = torch.Generator().manual_seed(999)
        noisy = inject_noise(seqs, noise_rate=1.0, generator=gen)

        padding_mask = (seqs == -1)
        self.assertTrue(torch.all(noisy[padding_mask] == -1), "Padding tokens must never be perturbed")
        valid_mask = (seqs >= 0)
        self.assertTrue(torch.all((noisy[valid_mask] >= 0) & (noisy[valid_mask] < 4)))

    def test_empirical_noise_rate_converges_to_three_quarters_epsilon(self):
        """C4: verify_effective_noise_rate confirms empirical substitution rate converges to 3*epsilon/4."""
        # Generate large synthetic sequence: 100,000 bases
        torch.manual_seed(101)
        large_seq = torch.randint(0, 4, (1, 100000), dtype=torch.long)

        rates_to_test = [0.02, 0.05, 0.10, 0.15]
        for eps in rates_to_test:
            gen = torch.Generator().manual_seed(42)
            noisy_seq = inject_noise(large_seq, noise_rate=eps, generator=gen)
            actual_rate = verify_effective_noise_rate(large_seq, noisy_seq, epsilon=eps, tolerance=0.005)
            expected = 0.75 * eps
            self.assertAlmostEqual(
                actual_rate,
                expected,
                delta=0.005,
                msg=f"Rate {actual_rate} did not match expected {expected} for epsilon={eps}",
            )

    # --- S2: Comprehensive Metrics and Paired Comparisons (C5 - C8) ---

    def test_compute_classification_metrics_accuracy_and_f1(self):
        """C5: compute_classification_metrics computes Top-1, Top-5, Macro-F1, and Weighted-F1."""
        y_true = np.array([0, 1, 2, 0, 1, 2])
        y_pred = np.array([0, 1, 1, 0, 1, 2])
        y_top5 = np.array([
            [0, 1, 2, 3, 4],
            [1, 0, 2, 3, 4],
            [1, 2, 0, 3, 4], # sample 2 has true label 2 in top 5
            [0, 2, 3, 4, 1],
            [1, 3, 4, 2, 0],
            [2, 1, 0, 3, 4],
        ])

        metrics = compute_classification_metrics(
            y_true=y_true,
            y_pred=y_pred,
            y_top5=y_top5,
            num_classes=5,
        )

        # 5 out of 6 correct Top-1: 83.3333%
        self.assertAlmostEqual(metrics.top1_accuracy, 83.3333, places=3)
        # All 6 correct Top-5: 100.0%
        self.assertEqual(metrics.top5_accuracy, 100.0)
        self.assertGreater(metrics.macro_f1, 0.0)
        self.assertGreater(metrics.weighted_f1, 0.0)
        self.assertEqual(metrics.n_samples, 6)
        self.assertEqual(metrics.n_classes, 5)

    def test_species_f1_bands_continuous_and_exhaustive(self):
        """C6: compute_species_f1_bands partitions into 4 continuous gapless bands conserving total classes."""
        num_classes = 10
        # True labels across 10 classes
        y_true = np.arange(num_classes)
        # Predict accurately for some, poorly for others
        y_pred = np.array([0, 1, 2, 3, 0, 0, 0, 0, 0, 0])

        bands_result = compute_species_f1_bands(y_true, y_pred, num_classes=num_classes)
        bands = bands_result["bands"]

        self.assertIn("band_0_to_50", bands)
        self.assertIn("band_50_to_80", bands)
        self.assertIn("band_80_to_95", bands)
        self.assertIn("band_95_to_100", bands)

        # Exact conservation of total class count
        total_counted = sum(b["count"] for b in bands.values())
        self.assertEqual(total_counted, num_classes)
        total_pct = sum(b["percent"] for b in bands.values())
        self.assertAlmostEqual(total_pct, 100.0, delta=0.1)

    def test_paired_comparison_percentage_points_and_contingency(self):
        """C7: compute_paired_comparison calculates percentage points difference and contingency table."""
        y_true = np.array([0, 1, 2, 3, 4, 5, 6, 7])
        preds_a = np.array([0, 1, 2, 3, 9, 9, 9, 9]) # 4 / 8 = 50.0%
        preds_b = np.array([0, 1, 9, 9, 9, 9, 9, 9]) # 2 / 8 = 25.0%

        comparison = compute_paired_comparison(preds_a, preds_b, y_true)
        self.assertEqual(comparison["accuracy_a"], 50.0)
        self.assertEqual(comparison["accuracy_b"], 25.0)
        self.assertEqual(comparison["accuracy_difference_pp"], 25.0)

        contingency = comparison["contingency"]
        self.assertEqual(contingency["both_correct"], 2) # items 0, 1
        self.assertEqual(contingency["only_a_correct"], 2) # items 2, 3
        self.assertEqual(contingency["only_b_correct"], 0)
        self.assertEqual(contingency["both_wrong"], 4) # items 4, 5, 6, 7

    def test_noise_protocol_fragment_bootstrap_interpretation(self):
        """C8: docs/noise_evaluation_protocol.md records that fragment bootstrap does not reflect genome variance."""
        doc_path = PROJECT_ROOT / "docs" / "noise_evaluation_protocol.md"
        content = doc_path.read_text(encoding="utf-8")

        self.assertIn("bootstrap", content.lower())
        self.assertIn("genomas", content.lower())
        self.assertIn("amostral", content.lower())

    # --- S3: Evaluation Runner and CLI Interface (C9 - C11) ---

    def test_run_evaluation_cli_exit_zero_and_prediction_manifest(self):
        """C9: scripts/run_evaluation.py executes evaluation, emits predictions and JSON summary with exit code 0."""
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir) / "eval_run"
            cli_script = PROJECT_ROOT / "scripts" / "run_evaluation.py"
            input_file = PROJECT_ROOT / "tests" / "fixtures" / "sample_genomes.csv"

            cmd = [
                sys.executable,
                str(cli_script),
                "--input",
                str(input_file),
                "--model-variant",
                "resnet_clean_species",
                "--noise",
                "0.05",
                "--output-dir",
                str(out_dir),
                "--seq-col",
                "sequence",
                "--label-col",
                "species_id",
                "--id-col",
                "fragment_id",
                "--seed",
                "42",
            ]
            res = subprocess.run(cmd, capture_output=True, text=True, cwd=str(PROJECT_ROOT))
            self.assertEqual(res.returncode, 0, f"CLI exited with error: {res.stderr}")

            # Check output JSON on stdout
            summary = json.loads(res.stdout)
            self.assertEqual(summary["model_name"], "resnet_clean_species")
            self.assertEqual(summary["nominal_noise_rate"], 0.05)
            self.assertEqual(summary["effective_noise_rate"], 0.0375)

            # Check written files on disk
            self.assertTrue((out_dir / "metrics_summary.json").exists())
            self.assertTrue((out_dir / "predictions.csv").exists())
            self.assertTrue((out_dir / "run_manifest.json").exists())

    def test_run_evaluation_cli_invalid_arguments_exit_code_2(self):
        """C10: scripts/run_evaluation.py exits with code 2 on unsupported noise or invalid inputs."""
        cli_script = PROJECT_ROOT / "scripts" / "run_evaluation.py"
        input_file = PROJECT_ROOT / "tests" / "fixtures" / "sample_genomes.csv"

        # 1. Unsupported noise rate (0.07)
        cmd_bad_noise = [
            sys.executable,
            str(cli_script),
            "--input",
            str(input_file),
            "--model-variant",
            "resnet_clean_species",
            "--noise",
            "0.07",
            "--output-dir",
            "/tmp/bad_noise_out",
        ]
        res = subprocess.run(cmd_bad_noise, capture_output=True, text=True, cwd=str(PROJECT_ROOT))
        self.assertEqual(res.returncode, 2)
        self.assertIn("Unsupported noise rate", res.stderr)

        # 2. Missing input file
        cmd_missing_file = [
            sys.executable,
            str(cli_script),
            "--input",
            "non_existent_file.csv",
            "--model-variant",
            "resnet_clean_species",
            "--output-dir",
            "/tmp/missing_file_out",
        ]
        res = subprocess.run(cmd_missing_file, capture_output=True, text=True, cwd=str(PROJECT_ROOT))
        self.assertEqual(res.returncode, 2)
        self.assertIn("Input file not found", res.stderr)

    def test_evaluation_pipeline_deterministic_synthetic_execution(self):
        """C11: Evaluation routines execute deterministically on synthetic fixtures without GPU/downloads."""
        torch.manual_seed(777)
        model = ResNet50_1D(num_classes=5)

        sequences = ["ACGTACGTACGT", "TGCATGCATGCA", "GGGGCCCCAAAA"]
        labels = [0, 1, 2]

        res1 = evaluate_model(
            model=model,
            sequences=sequences,
            labels=labels,
            noise_rate=0.05,
            num_classes=5,
            seed=42,
        )
        res2 = evaluate_model(
            model=model,
            sequences=sequences,
            labels=labels,
            noise_rate=0.05,
            num_classes=5,
            seed=42,
        )

        self.assertEqual(res1.metrics.top1_accuracy, res2.metrics.top1_accuracy)
        self.assertEqual(res1.metrics.macro_f1, res2.metrics.macro_f1)
        for p1, p2 in zip(res1.predictions, res2.predictions):
            self.assertEqual(p1.predicted_label, p2.predicted_label)
            self.assertEqual(p1.confidence, p2.confidence)


if __name__ == "__main__":
    unittest.main()
