"""
Unit tests for notebook runtime environment resolution and remote server topology (AD-017).
"""

import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from viral_cnn.notebook_runtime import (
    is_remote_server,
    setup_notebook_runtime,
    JupyterHubDataTopology,
)


class TestNotebookRuntimeEnvironment(unittest.TestCase):
    """Verifies environment resolution for remote server and local test modes."""

    def test_is_remote_server_default_is_true(self):
        """When REMOTE_SERVER is not defined or true, is_remote_server returns True."""
        with patch.dict(os.environ, {}, clear=True):
            self.assertTrue(is_remote_server())
        with patch.dict(os.environ, {"REMOTE_SERVER": "true"}, clear=True):
            self.assertTrue(is_remote_server())
        with patch.dict(os.environ, {"remote_server": "1"}, clear=True):
            self.assertTrue(is_remote_server())

    def test_is_remote_server_disabled(self):
        """When REMOTE_SERVER is False or 0, is_remote_server returns False."""
        for val in ["False", "false", "0", "no", "local"]:
            with patch.dict(os.environ, {"REMOTE_SERVER": val}, clear=True):
                self.assertFalse(is_remote_server(), f"Failed for REMOTE_SERVER={val}")
            with patch.dict(os.environ, {"remote_server": val}, clear=True):
                self.assertFalse(is_remote_server(), f"Failed for remote_server={val}")
                self.assertFalse(is_remote_server(), f"Failed for remote_server={val}")

    def test_setup_notebook_runtime_local_mode(self):
        """When REMOTE_SERVER=False, defaults to project root and smoke_test=True."""
        with patch.dict(os.environ, {"REMOTE_SERVER": "False", "SMOKE_TEST": "1"}, clear=True):
            ctx = setup_notebook_runtime()
            self.assertTrue(ctx.is_smoke_test)
            self.assertTrue(ctx.data_root.exists())

    def test_setup_notebook_runtime_explicit_data_root(self):
        """Explicit data_root overrides both remote and local auto-detection."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ctx = setup_notebook_runtime(data_root=tmpdir, smoke_test=True)
            self.assertEqual(ctx.data_root, Path(tmpdir).resolve())

    def test_topology_resolve_checkpoint_path(self):
        """JupyterHubDataTopology resolves existing checkpoint file or falls back cleanly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            top = JupyterHubDataTopology(root)
            # When file does not exist, returns canonical expected path
            p = top.resolve_checkpoint_path("resnet_clean_species")
            self.assertIsNotNone(p)
            self.assertEqual(p, root / "cnn_kraken_species/checkpoints/best_resnet_classifier__fast_phase4.pt")

            # Create checkpoint and verify it resolves
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text("fake_weights")
            resolved = top.resolve_checkpoint_path("resnet_clean_species")
            self.assertEqual(resolved, p)
            self.assertTrue(resolved.exists())


if __name__ == "__main__":
    unittest.main()
