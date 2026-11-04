#!/usr/bin/env python3
"""CLI script to compile manuscript LaTeX documents to PDF."""

import argparse
import subprocess
import sys
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PAPER_DIR = PROJECT_ROOT / "paper"
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from viral_cnn.notebook_runtime import reserve_run_directory
from viral_cnn.reporting.publication import assemble_publication


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
        choices=["strict", "diagnostic", "legacy"],
        default="strict",
        help="C06 strict assembly (default), diagnostics, or explicitly historical compilation.",
    )
    parser.add_argument("--variant", choices=["submission", "review"], default="submission")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "publication_c06.yaml")
    parser.add_argument("--run-id", default=None)
    parser.add_argument(
        "--tex-file",
        type=Path,
        default=None,
        help="Legacy mode only. C06 compiles both package variants from its selected evidence.",
    )
    args = parser.parse_args()

    if args.mode != "legacy":
        run_id = args.run_id or f"c06-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}-{uuid.uuid4().hex[:8]}"
        output_dir = reserve_run_directory(PROJECT_ROOT, run_id, "publication")
        result = assemble_publication(
            args.config, output_dir, strict_empirical=args.mode == "strict",
            diagnostic_only=args.mode == "diagnostic", interface="cli",
            command="compile_paper.py " + " ".join(sys.argv[1:]),
        )
        print(f"C06 {result['status']}: package and compile outputs in {output_dir}")
        return 2 if result["status"] == "BLOCKED" else (1 if result["status"] == "FAILED" else 0)

    if args.tex_file:
        tex_path = args.tex_file
    elif args.variant == "submission":
        tex_path = PAPER_DIR / "manuscript_submission.tex"
    else:
        tex_path = PAPER_DIR / "manuscript_review.tex"

    if not tex_path.exists():
        print(f"ERROR: LaTeX source file not found: {tex_path}", file=sys.stderr)
        return 1

    print(f"Compiling {tex_path.name} to PDF...")
    # Never overwrite a versioned manuscript PDF or its auxiliary files.
    with tempfile.TemporaryDirectory(prefix="cnn-metagenome-paper-") as tmpdir:
        res = compile_manuscript(tex_path, output_dir=Path(tmpdir))
    if res.returncode != 0:
        print(f"Compilation FAILED with code {res.returncode}:", file=sys.stderr)
        print(res.stdout[-2000:], file=sys.stderr)
        return res.returncode

    pdf_name = tex_path.stem + ".pdf"
    print(f"SUCCESS: Compiled {pdf_name} in a temporary directory (not retained)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
