#!/usr/bin/env python3
"""CLI script to compile manuscript LaTeX documents to PDF."""

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PAPER_DIR = PROJECT_ROOT / "paper"


def compile_manuscript(tex_path: Path, output_dir: Path | None = None) -> subprocess.CompletedProcess:
    """Compile LaTeX document to PDF using pdflatex."""
    tex_path = Path(tex_path).resolve()
    if output_dir is None:
        output_dir = tex_path.parent
    else:
        output_dir = Path(output_dir).resolve()

    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "pdflatex",
        "-interaction=nonstopmode",
        f"-output-directory={output_dir}",
        str(tex_path),
    ]

    # First pass
    proc = subprocess.run(cmd, cwd=tex_path.parent, capture_output=True, text=True, errors="replace")
    if proc.returncode != 0:
        return proc

    # Second pass for cross-references
    proc = subprocess.run(cmd, cwd=tex_path.parent, capture_output=True, text=True, errors="replace")
    return proc


def main() -> int:
    parser = argparse.ArgumentParser(description="Compile paper LaTeX files.")
    parser.add_argument(
        "--mode",
        choices=["submission", "review"],
        default="submission",
        help="Manuscript variant to compile.",
    )
    parser.add_argument(
        "--tex-file",
        type=Path,
        default=None,
        help="Custom .tex file to compile.",
    )
    args = parser.parse_args()

    if args.tex_file:
        tex_path = args.tex_file
    elif args.mode == "submission":
        tex_path = PAPER_DIR / "manuscript_submission.tex"
    else:
        tex_path = PAPER_DIR / "manuscript_review.tex"

    if not tex_path.exists():
        print(f"ERROR: LaTeX source file not found: {tex_path}", file=sys.stderr)
        return 1

    print(f"Compiling {tex_path.name} to PDF...")
    res = compile_manuscript(tex_path, output_dir=PAPER_DIR)
    if res.returncode != 0:
        print(f"Compilation FAILED with code {res.returncode}:", file=sys.stderr)
        print(res.stdout[-2000:], file=sys.stderr)
        return res.returncode

    pdf_name = tex_path.stem + ".pdf"
    print(f"SUCCESS: Compiled {PAPER_DIR / pdf_name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
