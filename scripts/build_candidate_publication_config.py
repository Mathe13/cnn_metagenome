#!/usr/bin/env python3
"""Build a candidate C06 publication configuration from an explicit M00 run_plan.json."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from viral_cnn.reporting.publication import build_candidate_publication_config


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-plan", required=True, type=Path, help="Path to run_plan.json")
    parser.add_argument(
        "--base-config",
        type=Path,
        default=ROOT / "configs" / "reproduction_m07.yaml",
        help="Base publication YAML configuration template",
    )
    parser.add_argument("--output", type=Path, default=None, help="Destination path for candidate YAML")
    parser.add_argument("--project-root", type=Path, default=ROOT, help="Project root directory")
    parser.add_argument("--json", action="store_true", help="Print diff report as JSON")
    args = parser.parse_args()

    try:
        result = build_candidate_publication_config(
            run_plan_path=args.run_plan,
            base_config_path=args.base_config,
            output_path=args.output,
            project_root=args.project_root,
        )
        if args.json:
            print(json.dumps(result, indent=2))
        else:
            print(f"Candidate C06 selection status: {result['status']} (run_id: {result['run_id']})")
            print(f"Summary: total={result['counts']['total']}, proposed={result['counts']['proposed']}, "
                  f"unchanged={result['counts']['unchanged']}, missing={result['counts']['missing']}")
            if result.get("output_path"):
                print(f"Candidate YAML saved to: {result['output_path']}")
            print(f"\n[NOTE] {result['human_review_directive']}")
            print("\nItem proposal details:")
            for item in result["diff"]:
                print(f"  [{item['status']}] {item['item_id']}: {item['candidate_artifact'] or 'NONE'}")
        return 0
    except Exception as exc:
        print(f"Error building candidate publication config: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
