#!/usr/bin/env python3
"""CLI script to generate and verify publication figures for viral CNN paper."""

import argparse
import sys
from pathlib import Path

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from viral_cnn.reporting.figures import generate_all_figures, verify_latex_figure_paths


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate and verify publication figures.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "paper" / "figures",
        help="Directory to save generated figures.",
    )
    parser.add_argument(
        "--verify-tex",
        type=Path,
        default=None,
        help="Optional path to a .tex file to verify figure paths against assets.",
    )
    args = parser.parse_args()

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

