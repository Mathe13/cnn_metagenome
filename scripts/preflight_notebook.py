#!/usr/bin/env python3
"""Inspect inputs; short/09 also validates C03 tables and hashes without model inference."""
import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from viral_cnn.notebook_runtime import NOTEBOOK_REQUIREMENTS, preflight_notebook
from viral_cnn.run_planning import inspect_run_stage, load_run_plan


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("notebook_id", choices=sorted(set(NOTEBOOK_REQUIREMENTS) | {"short/06b"}))
    parser.add_argument("--data-root", default=os.getenv("DATA_ROOT", str(ROOT)))
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--run-plan", help="Explicit M00 run_plan.json; its inputs and mode take precedence")
    args = parser.parse_args()
    try:
        report = (inspect_run_stage(load_run_plan(args.run_plan), args.notebook_id) if args.run_plan
                  else preflight_notebook(args.notebook_id, args.data_root, smoke_test=args.smoke_test))
    except (OSError, ValueError, KeyError, TypeError) as exc:
        report = {"status": "BLOCKED", "error": str(exc)}
    print(json.dumps(report, indent=2))
    return 0 if report["status"] in ("READY", "SKIPPED") else 2


if __name__ == "__main__":
    raise SystemExit(main())
