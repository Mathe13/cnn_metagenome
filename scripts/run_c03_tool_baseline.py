#!/usr/bin/env python3
"""Run Kraken2-flat or BLASTN over one identified C03 query condition."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from viral_cnn.baselines.preparation import sha256_file, write_method_condition_manifest
from viral_cnn.baselines.taxonomy import parse_blast_output, parse_kraken_output
from viral_cnn.utils.logging_utils import log_artifact_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("Kraken2-flat", "BLASTN"), required=True)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--fasta", type=Path, required=True)
    parser.add_argument("--query-manifest", type=Path, required=True)
    parser.add_argument("--reference-manifest", type=Path, required=True)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--confidence", type=float, default=0.0)
    parser.add_argument("--evalue", default="1e-5")
    parser.add_argument("--max-target-seqs", type=int, default=100)
    parser.add_argument("--max-hsps", type=int, default=1)
    args = parser.parse_args()
    for path in (args.queries, args.fasta, args.query_manifest, args.reference_manifest):
        if not path.is_file():
            raise FileNotFoundError(path)
    qmeta = json.loads(args.query_manifest.read_text(encoding="utf-8"))
    if qmeta.get("baseline_protocol") != "c03_paired_v1" or qmeta.get("noise_protocol") != "resample_including_original_v1":
        raise ValueError("Input query manifest is not an identified C03 query condition")
    if sha256_file(args.queries) != qmeta.get("query_content_sha256") or sha256_file(args.fasta) != qmeta.get("fasta_sha256"):
        raise ValueError("Query CSV/FASTA content does not match the shared query manifest")
    queries = pd.read_csv(args.queries, dtype={"query_id": str})
    qids = queries["query_id"].tolist()
    if not qids or len(qids) != len(set(qids)):
        raise ValueError("Query table must have a nonempty, unique query_id column")
    log_artifact_path("C03 query FASTA", args.fasta)
    log_artifact_path("C03 reference manifest", args.reference_manifest)
    log_artifact_path("C03 baseline reference database", args.database)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rate = float(qmeta["noise_rate"])
    tag = f"noise_{round(rate * 100):02d}"
    tool = "kraken2" if args.method == "Kraken2-flat" else "blastn"
    raw_path = args.output_dir / f"raw_{tool}_{tag}.txt"
    report_path = args.output_dir / f"report_kraken2_{tag}.txt"
    executable = "kraken2" if args.method == "Kraken2-flat" else "blastn"
    version_cmd = [executable, "--version"] if executable == "kraken2" else [executable, "-version"]
    version_proc = subprocess.run(version_cmd, text=True, capture_output=True, check=False)
    version = (version_proc.stdout or version_proc.stderr).strip().splitlines()[0] if version_proc.returncode == 0 else "unknown"
    if args.method == "Kraken2-flat":
        command = [executable, "--db", str(args.database), "--threads", str(args.threads),
                   "--confidence", str(args.confidence), "--report", str(report_path), "--output", str(raw_path), str(args.fasta)]
    else:
        command = [executable, "-task", "blastn", "-db", str(args.database), "-query", str(args.fasta),
                   "-out", str(raw_path), "-outfmt", "6 qseqid sseqid bitscore evalue pident length",
                   "-evalue", args.evalue, "-dust", "yes", "-max_target_seqs", str(args.max_target_seqs),
                   "-max_hsps", str(args.max_hsps), "-num_threads", str(args.threads)]
    ref_meta = json.loads(args.reference_manifest.read_text(encoding="utf-8"))
    database_manifest = args.reference_manifest.parent / "database_build_manifest.json"
    if not database_manifest.is_file():
        raise FileNotFoundError(f"Database build manifest not found: {database_manifest}")
    db_manifest_data = json.loads(database_manifest.read_text(encoding="utf-8"))
    if db_manifest_data.get("state") != "COMPLETE":
        raise ValueError(f"Database build is not COMPLETE: {database_manifest}")
    if bool(db_manifest_data.get("is_smoke_test")) != bool(qmeta.get("is_smoke_test")):
        raise ValueError("Query and database smoke/empirical states differ")
    reference = {"path": str(args.reference_manifest.resolve()), "sha256": sha256_file(args.reference_manifest),
                 "database_path": str(args.database.resolve()), "source_reference_sha256": ref_meta.get("train_sha256", "unknown"),
                 "database_build_manifest": {"path": str(database_manifest.resolve()), "sha256": sha256_file(database_manifest)}}
    manifest_path = args.output_dir / f"manifest_{args.method}_{tag}.json"
    normalized_path = args.output_dir / f"predictions_{args.method}_{tag}.csv"
    if normalized_path.exists() or manifest_path.exists() or raw_path.exists():
        raise FileExistsError(f"Refusing to overwrite C03 condition output under {args.output_dir}; use a fresh run namespace")
    write_method_condition_manifest(
        manifest_path, method=args.method, noise_rate=rate, query_manifest=args.query_manifest,
        class_order_sha256=qmeta["class_order_sha256"], reference=reference, tool_version=version,
        command=command, abstention_policy="raw tool output required before abstention normalization",
        completion="RUNNING", is_smoke_test=bool(qmeta["is_smoke_test"]),
    )
    try:
        proc = subprocess.run(command, text=True, capture_output=True, check=False)
        if proc.returncode != 0 or not raw_path.is_file():
            raise RuntimeError(f"{tool} failed (returncode={proc.returncode}): {proc.stderr[-4000:]}")
        if args.method == "Kraken2-flat":
            pred_map = parse_kraken_output(raw_path, valid_query_ids=qids)
            frame = queries[["query_id", "label"]].copy()
            frame["prediction"] = frame["query_id"].map(pred_map).astype(int)
            frame["abstention_reason"] = frame["prediction"].map(lambda value: "unclassified_or_root" if value == -1 else "")
        else:
            ranked = parse_blast_output(raw_path, valid_query_ids=qids, max_targets=5)
            frame = queries[["query_id", "label"]].copy()
            frame["prediction"] = frame["query_id"].map(lambda q: ranked[q][0] if ranked[q] else -1)
            frame["abstention_reason"] = frame["query_id"].map(lambda q: "no_hit" if not ranked[q] else "")
            for k in range(5):
                frame[f"top{k + 1}"] = frame["query_id"].map(lambda q: ranked[q][k] if len(ranked[q]) > k else -1)
        frame.to_csv(normalized_path, index=False)
        write_method_condition_manifest(
            manifest_path, method=args.method, noise_rate=rate, query_manifest=args.query_manifest,
            class_order_sha256=qmeta["class_order_sha256"], reference=reference, tool_version=version,
            command=command, abstention_policy="Kraken U/root => -1" if args.method == "Kraken2-flat" else "completed BLASTN with no hit => -1",
            completion="COMPLETE", is_smoke_test=bool(qmeta["is_smoke_test"]), output_path=normalized_path,
            execution_details={"raw_output_path": str(raw_path.resolve()), "raw_output_sha256": sha256_file(raw_path),
                               "report_path": str(report_path.resolve()) if args.method == "Kraken2-flat" else None,
                               "report_sha256": sha256_file(report_path) if args.method == "Kraken2-flat" and report_path.is_file() else None,
                               "query_csv_path": str(args.queries.resolve()), "query_csv_sha256": sha256_file(args.queries),
                               "query_fasta_path": str(args.fasta.resolve()), "query_fasta_sha256": sha256_file(args.fasta),
                               "threads": args.threads, "blast_max_target_seqs": args.max_target_seqs if args.method == "BLASTN" else None,
                               "blast_max_hsps": args.max_hsps if args.method == "BLASTN" else None},
        )
    except Exception as exc:
        write_method_condition_manifest(
            manifest_path, method=args.method, noise_rate=rate, query_manifest=args.query_manifest,
            class_order_sha256=qmeta["class_order_sha256"], reference=reference, tool_version=version,
            command=command, abstention_policy="raw run required before abstention normalization",
            completion="FAILED", is_smoke_test=bool(qmeta["is_smoke_test"]), error=str(exc),
        )
        raise
    print(f"Predictions: {normalized_path}\nManifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
