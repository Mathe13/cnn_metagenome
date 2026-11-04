#!/usr/bin/env python3
"""CLI script to generate and verify publication figures for viral CNN paper."""

import argparse
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from viral_cnn.reporting.figures import generate_all_figures, verify_latex_figure_paths
from viral_cnn.notebook_runtime import reserve_run_directory
from viral_cnn.reporting.publication import assemble_publication


def main() -> int:
    parser = argparse.ArgumentParser(description="Assemble strict C06 publication figures; legacy output is opt-in.")
    parser.add_argument("--mode", choices=("strict", "diagnostic", "legacy"), default="strict")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "publication_c06.yaml")
    parser.add_argument("--run-id", default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "paper" / "figures",
        help="Legacy mode only. C06 modes always use the reserved run namespace.",
    )
    parser.add_argument(
        "--verify-tex",
        type=Path,
        default=None,
        help="Legacy mode only. C06 validates generated manuscript paths before compilation.",
    )
    args = parser.parse_args()

    if args.mode != "legacy":
        run_id = args.run_id or f"c06-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}-{uuid.uuid4().hex[:8]}"
        output_dir = reserve_run_directory(PROJECT_ROOT, run_id, "publication")
        result = assemble_publication(
            args.config, output_dir, strict_empirical=args.mode == "strict",
            diagnostic_only=args.mode == "diagnostic", interface="cli",
            command="generate_paper_figures.py " + " ".join(sys.argv[1:]),
        )
        print(f"C06 {result['status']}: figures and evidence outputs in {output_dir}")
        return 2 if result["status"] == "BLOCKED" else 0

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Generating publication figures in: {output_dir}")

    fig_paths = generate_all_figures(output_dir)
    print(f"Successfully generated {len(fig_paths)} figure assets.")

    if args.verify_tex:
        print(f"Verifying LaTeX image references in: {args.verify_tex}")
        matches, mismatches = verify_latex_figure_paths(args.verify_tex, output_dir)
        if mismatches:
            print(f"ERROR: Found {len(mismatches)} figure reference errors:")
            for m in mismatches:
                print(f"  - {m['ref']}: {m['status']}")
            return 1
        print(f"All {len(matches)} figure references match filesystem paths exactly.")

    return 0


if __name__ == "__main__":
    sys.exit(main())

