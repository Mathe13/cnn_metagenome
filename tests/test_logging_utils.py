"""
Unit tests verifying artifact and path logging behavior (LOG_PATH rule).
"""

from io import StringIO
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd
import torch

from viral_cnn.utils.logging_utils import is_path_logging_enabled, log_artifact_path
from viral_cnn.data.parquet_dataset import ParquetBatchIterableDataset
from viral_cnn.taxonomy.mapping import load_taxonomy_mapping
from viral_cnn.models.registry import build_registered_model, load_model_checkpoint
from viral_cnn.models.resnet import ResNet50_1D


class TestArtifactPathLogging(unittest.TestCase):
    """Verifies that datasets, dataloaders, and models log their source paths, configurable via LOG_PATH."""

    def test_default_is_enabled(self):
        """LOG_PATH defaults to True when unset."""
        with patch.dict(os.environ, {}, clear=True):
            self.assertTrue(is_path_logging_enabled())

    def test_disable_via_env_var(self):
        """LOG_PATH can be disabled via 0, false, no, off."""
        for val in ["0", "false", "False", "no", "off", "disabled"]:
            with patch.dict(os.environ, {"LOG_PATH": val}):
                self.assertFalse(is_path_logging_enabled(), f"Failed for LOG_PATH={val}")

    def test_explicit_enable_via_env_var(self):
        """LOG_PATH can be explicitly enabled."""
        for val in ["1", "true", "True", "yes"]:
            with patch.dict(os.environ, {"LOG_PATH": val}):
                self.assertTrue(is_path_logging_enabled(), f"Failed for LOG_PATH={val}")

    def test_log_artifact_path_output_when_enabled(self):
        """log_artifact_path prints to stdout with standard [LOG_PATH] prefix."""
        with patch.dict(os.environ, {"LOG_PATH": "1"}):
            with patch("sys.stderr", new=StringIO()) as fake_out:
                log_artifact_path("checkpoint", "/path/to/model.pt", extra="test")
                output = fake_out.getvalue()
                self.assertIn("[LOG_PATH] Loading checkpoint from: /path/to/model.pt", output)
                self.assertIn("test", output)

    def test_log_artifact_path_suppressed_when_disabled(self):
        """log_artifact_path produces no stdout when disabled."""
        with patch.dict(os.environ, {"LOG_PATH": "0"}):
            with patch("sys.stderr", new=StringIO()) as fake_out:
                log_artifact_path("checkpoint", "/path/to/model.pt")
                self.assertEqual(fake_out.getvalue(), "")

    def test_parquet_dataset_logs_path_on_init(self):
        """ParquetBatchIterableDataset logs its source directory or file paths upon initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pq_path = Path(tmpdir) / "sample.parquet"
            pd.DataFrame({
                "sequence_padded": ["ACGT" * 128],
                "label": [0]
            }).to_parquet(pq_path, index=False)

            with patch.dict(os.environ, {"LOG_PATH": "1"}):
                with patch("sys.stderr", new=StringIO()) as fake_out:
                    _ = ParquetBatchIterableDataset([str(pq_path)], batch_size=1)
                    output = fake_out.getvalue()
                    self.assertIn("[LOG_PATH] Loading DataLoader/ParquetDataset from:", output)
                    self.assertIn(str(pq_path), output)

    def test_load_taxonomy_mapping_logs_path(self):
        """load_taxonomy_mapping logs the CSV path upon loading."""
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "taxa.csv"
            csv_path.write_text("id,species\n1,Virus A\n")

            with patch.dict(os.environ, {"LOG_PATH": "1"}):
                with patch("sys.stderr", new=StringIO()) as fake_out:
                    _ = load_taxonomy_mapping(csv_path)
                    output = fake_out.getvalue()
                    self.assertIn("[LOG_PATH] Loading taxonomy mapping from:", output)
                    self.assertIn(str(csv_path), output)

    def test_load_model_checkpoint_logs_path(self):
        """load_model_checkpoint logs the checkpoint file path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = Path(tmpdir) / "test.pt"
            model = ResNet50_1D(num_classes=5)
            torch.save(model.state_dict(), ckpt_path)

            with patch.dict(os.environ, {"LOG_PATH": "1"}):
                with patch("sys.stderr", new=StringIO()) as fake_out:
                    eval_model = ResNet50_1D(num_classes=5)
                    load_model_checkpoint(eval_model, ckpt_path)
                    output = fake_out.getvalue()
                    self.assertIn("[LOG_PATH] Loading checkpoint from:", output)
                    self.assertIn(str(ckpt_path), output)


if __name__ == "__main__":
    unittest.main()

