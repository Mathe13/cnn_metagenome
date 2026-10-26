"""
Pytest configuration ensuring local test environment isolation.
"""

import os
import pytest


@pytest.fixture(autouse=True, scope="session")
def configure_local_test_env():
    """Ensures test runs always use local environment mode (REMOTE_SERVER=False)."""
    os.environ["REMOTE_SERVER"] = "False"
    os.environ["remote_server"] = "False"
    os.environ.setdefault("SMOKE_TEST", "1")

