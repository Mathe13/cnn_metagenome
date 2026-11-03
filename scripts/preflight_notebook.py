#!/usr/bin/env python3
"""Inspect reproduction notebook inputs without loading dataset rows or models."""
import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from viral_cnn.notebook_runtime import NOTEBOOK_REQUIREMENTS, preflight_notebook


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("notebook_id", choices=sorted(NOTEBOOK_REQUIREMENTS))
    parser.add_argument("--data-root", default=os.getenv("DATA_ROOT", str(ROOT)))
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()
    report = preflight_notebook(args.notebook_id, args.data_root, smoke_test=args.smoke_test)
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "READY" else 2


if __name__ == "__main__":
    raise SystemExit(main())
