"""
Contract tests for CLI inference entrypoint scripts/run_inference.py (C7, C8).
"""

import json
import subprocess
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PYTHON_EXE = PROJECT_ROOT / ".venv" / "bin" / "python3"
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "run_inference.py"


class TestCLIInference(unittest.TestCase):
    def test_clean_inference_exit_zero_and_json(self):
        """C7: Verifies CLI runs forward pass on ResNet50-1D and ConvNeXt-1D with exit 0 and valid JSON."""
        test_seq = "ACGTACGTACGTNNNN" * 4  # 64 bp

        for arch in ["resnet50_1d", "convnext_tiny_1d"]:
            cmd = [
                str(PYTHON_EXE),
                str(SCRIPT_PATH),
                "--sequence",
                test_seq,
                "--model-type",
                arch,
                "--num-classes",
                "10",
                "--device",
                "cpu",
            ]
            res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            self.assertEqual(
                res.returncode,
                0,
                f"CLI failed for architecture {arch}:\nSTDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}",
            )

            # Validate structured JSON output
            data = json.loads(res.stdout)
            self.assertEqual(data["model_type"], arch)
            self.assertEqual(data["num_classes"], 10)
            self.assertEqual(data["channels"], 4)
            self.assertEqual(data["input_length"], len(test_seq))
            self.assertIsInstance(data["predicted_class_index"], int)
            self.assertTrue(0 <= data["predicted_class_index"] < 10)
            self.assertEqual(len(data["top5_class_indices"]), 5)
            self.assertEqual(len(data["top5_confidences"]), 5)

    def test_invalid_inputs_exit_code_2(self):
        """C8: Verifies CLI exits with code 2 on invalid characters, empty input, or bad model type."""
        # 1. Invalid sequence characters
        cmd_invalid_chars = [
            str(PYTHON_EXE),
            str(SCRIPT_PATH),
            "--sequence",
            "ACGT123!@#",
            "--model-type",
            "resnet50_1d",
            "--num-classes",
            "10",
        ]
        res1 = subprocess.run(cmd_invalid_chars, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        self.assertEqual(res1.returncode, 2)
        self.assertIn("invalid", res1.stderr.lower())

        # 2. Unsupported model type
        cmd_invalid_model = [
            str(PYTHON_EXE),
            str(SCRIPT_PATH),
            "--sequence",
            "ACGTACGT",
            "--model-type",
            "unsupported_model_arch",
            "--num-classes",
            "10",
        ]
        res2 = subprocess.run(cmd_invalid_model, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        self.assertEqual(res2.returncode, 2)
        self.assertIn("unknown model-type", res2.stderr.lower())

        # 3. Empty sequence
        cmd_empty_seq = [
            str(PYTHON_EXE),
            str(SCRIPT_PATH),
            "--sequence",
            "",
            "--model-type",
            "resnet50_1d",
            "--num-classes",
            "10",
        ]
        res3 = subprocess.run(cmd_empty_seq, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        self.assertEqual(res3.returncode, 2)


if __name__ == "__main__":
    unittest.main()

