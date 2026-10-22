#!/usr/bin/env python3
"""CLI script to generate publication tables in CSV, Markdown, and LaTeX formats."""

import argparse
import sys
from pathlib import Path

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from viral_cnn.reporting.tables import generate_all_tables


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate publication tables for viral CNN paper.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "results_tables",
        help="Directory to save generated tables.",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        default=["csv", "md", "tex"],
        choices=["csv", "md", "tex"],
        help="Output formats for tables.",
    )
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Generating publication tables in: {output_dir}")

    tables = generate_all_tables(output_dir, formats=tuple(args.formats))
    print(f"Successfully generated {len(tables)} tables in {args.formats} formats.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

