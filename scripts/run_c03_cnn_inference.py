#!/usr/bin/env python3
"""Run one registered Species checkpoint over a materialized C03 query realization."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from viral_cnn.baselines.preparation import sha256_file, write_method_condition_manifest
from viral_cnn.data.encoding import encode_sequence_tokens
from viral_cnn.models.registry import REGISTERED_VARIANTS, build_registered_model, get_model_spec
from viral_cnn.utils.logging_utils import log_artifact_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--query-manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--variant", choices=[k for k, v in REGISTERED_VARIANTS.items() if v.rank == "species"], required=True)
    parser.add_argument("--reference-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()
    for path in (args.queries, args.query_manifest, args.checkpoint, args.reference_manifest):
        if not path.is_file():
            raise FileNotFoundError(path)
    qmeta = json.loads(args.query_manifest.read_text(encoding="utf-8"))
    if qmeta.get("baseline_protocol") != "c03_paired_v1" or qmeta.get("noise_protocol") != "resample_including_original_v1":
        raise ValueError("Query manifest is not a C03 resample-including-original realization")
    if sha256_file(args.queries) != qmeta.get("query_content_sha256"):
        raise ValueError("Materialized query CSV hash does not match query manifest")
    log_artifact_path("materialized CNN queries", args.queries)
    frame = pd.read_csv(args.queries, dtype={"query_id": str})
    if frame["query_id"].isna().any() or not frame["query_id"].is_unique:
        raise ValueError("Query IDs must be unique and non-null")
    if frame.empty:
        raise ValueError("C03 inference requires at least one materialized query")
    spec = get_model_spec(args.variant)
    if spec.rank != "species" or spec.num_classes != 13683:
        raise ValueError(f"C03 requires a registered 13,683-class Species model, got {args.variant}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rate = float(qmeta["noise_rate"])
    method = args.variant
    tag = f"noise_{round(rate * 100):02d}"
    output_path = args.output_dir / f"predictions_{method}_{tag}.csv"
    method_manifest_path = args.output_dir / f"manifest_{method}_{tag}.json"
    if output_path.exists() or method_manifest_path.exists():
        raise FileExistsError(f"Refusing to overwrite C03 condition output: {output_path}")
    ref_record = json.loads(args.reference_manifest.read_text(encoding="utf-8"))
    reference = {"path": str(args.reference_manifest.resolve()), "sha256": sha256_file(args.reference_manifest), "source_reference_sha256": ref_record.get("train_sha256", "unknown")}
    shared_manifest = dict(
        method=method, noise_rate=rate, query_manifest=args.query_manifest,
        class_order_sha256=qmeta["class_order_sha256"], reference=reference, tool_version=f"torch {torch.__version__}",
        command=sys.argv, abstention_policy="CNN dense top1/top5; no abstention",
        is_smoke_test=bool(qmeta.get("is_smoke_test")), model_variant=args.variant, checkpoint_path=args.checkpoint,
        execution_details={"device": args.device, "batch_size": args.batch_size, "precision": "float32", "class_count": 13683,
                           "query_csv_path": str(args.queries.resolve()), "query_csv_sha256": sha256_file(args.queries),
                           "input_representation": "sequence strings encoded by viral_cnn.data.encoding to 512 A/C/G/T/null tokens"},
    )
    write_method_condition_manifest(method_manifest_path, **shared_manifest, completion="RUNNING")
    try:
        model = build_registered_model(args.variant, checkpoint_path=args.checkpoint, device=args.device)
        model.eval()
        topk_chunks = []
        with torch.inference_mode():
            for start in range(0, len(frame), args.batch_size):
                seqs = frame["sequence"].iloc[start:start + args.batch_size].astype(str).tolist()
                tokens = torch.tensor([encode_sequence_tokens(seq, max_len=512) for seq in seqs], dtype=torch.long, device=args.device)
                logits = model(tokens)
                topk_chunks.append(torch.topk(logits, k=5, dim=1).indices)
        topk = torch.cat(topk_chunks).cpu().numpy()
        result = frame[["query_id", "label"]].copy()
        result["prediction"] = topk[:, 0]
        for idx in range(5):
            result[f"top{idx + 1}"] = topk[:, idx]
        result.to_csv(output_path, index=False)
        write_method_condition_manifest(method_manifest_path, **shared_manifest, completion="COMPLETE", output_path=output_path)
    except Exception as exc:
        write_method_condition_manifest(method_manifest_path, **shared_manifest, completion="FAILED", error=str(exc))
        raise
    print(f"Predictions: {output_path}\nManifest: {method_manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
