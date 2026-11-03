"""
Global test suite configuration and environment isolation.

Contract (AD-017, AGENTS.md):
- Local tests must execute in offline, lightweight sandbox mode with deterministic fixtures.
- Sets REMOTE_SERVER=False and SMOKE_TEST=1 by default to prevent tests from resolving remote server paths.
"""

import os
import sys
from pathlib import Path

_src_dir = str(Path(__file__).resolve().parent.parent / "src")
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

os.environ["REMOTE_SERVER"] = "False"
os.environ["remote_server"] = "False"
os.environ.setdefault("SMOKE_TEST", "1")

