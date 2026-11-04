#!/usr/bin/env python3
"""Build or diagnose an isolated C06 publication package."""

from __future__ import annotations

import argparse
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from viral_cnn.notebook_runtime import reserve_run_directory
from viral_cnn.reporting.publication import assemble_publication


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "publication_c06.yaml")
    parser.add_argument("--run-id", help="Unique namespace identifier; generated when omitted")
    parser.add_argument("--mode", choices=("strict", "diagnostic"), default="strict")
    parser.add_argument("--interface", choices=("cli", "notebook"), default="cli")
    args = parser.parse_args()
    run_id = args.run_id or f"c06-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}-{uuid.uuid4().hex[:8]}"
    output_dir = reserve_run_directory(PROJECT_ROOT, run_id, "publication")
    result = assemble_publication(
        args.config, output_dir, strict_empirical=args.mode == "strict",
        command="assemble_publication.py " + " ".join(sys.argv[1:]),
        interface=args.interface, diagnostic_only=args.mode == "diagnostic",
    )
    print(f"C06 status: {result['status']}; matrix: {result['matrix']}; manifest: {result['manifest']}")
    for row in result["rows"]:
        if row["status"] not in {"EMPIRICAL_VERIFIED", "DERIVED_VERIFIED"}:
            print(f"[{row['status']}] {row['item_id']}: {row['diagnostic']}")
    return 2 if result["status"] == "BLOCKED" else 0


if __name__ == "__main__":
    raise SystemExit(main())
