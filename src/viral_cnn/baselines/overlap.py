"""Sequence overlap auditing between test queries and training references."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, Union
import numpy as np
import pandas as pd

from viral_cnn.utils.logging_utils import log_artifact_path


@dataclass
class BenchmarkArtifacts:
    """Prepared baseline benchmark data including overlap audit and predictions."""

    audit_df: pd.DataFrame
    strata_masks: Dict[str, np.ndarray]
    predictions: Dict[str, Dict[float, np.ndarray]]
    top_candidates: Dict[str, Dict[float, Optional[np.ndarray]]]
    runtime: Optional[pd.DataFrame] = None
    method_manifests: Optional[List[Dict[str, Any]]] = None


DEFAULT_METHOD_ALIASES: Dict[str, List[str]] = {
    "resnet50_species": ["resnet50_species", "ResNet50", "resnet_clean_species", "resnet50"],
    "resnet_noise_species": ["resnet_noise_species", "ResNet_F", "resnet_noise"],
    "convnext_zero_species": ["convnext_zero_species", "ConvNeXt_Z", "convnext_zero"],
    "convnext_transfer_species": ["convnext_transfer_species", "ConvNeXt_T", "convnext_transfer"],
    "convnext_noise_species": ["convnext_noise_species", "ConvNeXt_F", "convnext_noise"],
    "Kraken2-flat": ["Kraken2-flat", "kraken2_flat", "kraken2", "Kraken2"],
    "BLASTN": ["BLASTN", "blastn", "blast"],
}


def reverse_complement(seq: str) -> str:
    """Return the reverse complement for A/C/G/T/N benchmark sequences."""
    return seq.translate(str.maketrans("ACGTN", "TGCAN"))[::-1]


def canonical_sequence_hash(seq: str) -> str:
    """Hash the lexicographically smaller uppercase forward/reverse-complement sequence."""
    clean = seq.strip().upper()
    canonical = min(clean, reverse_complement(clean))
    return hashlib.sha256(canonical.encode("ascii")).hexdigest()


def _sequence_digest(value: str, *, precomputed: bool = False) -> str:
    """Interpret sequence text as DNA unless its digest type is explicitly declared."""
    if precomputed:
        if len(value) != 64 or any(c not in "0123456789abcdefABCDEF" for c in value):
            raise ValueError(f"Invalid precomputed SHA-256 digest: {value!r}")
        return value.lower()
    return canonical_sequence_hash(value)


def _strict_bool(value: Any, *, field: str) -> bool:
    """Parse actual boolean encodings and reject ambiguous/missing values."""
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)) and value in (0, 1):
        return bool(value)
    if isinstance(value, (float, np.floating)) and np.isfinite(value) and value in (0.0, 1.0):
        return bool(value)
    if isinstance(value, str) and value.strip().lower() in {"true", "false", "1", "0"}:
        return value.strip().lower() in {"true", "1"}
    raise ValueError(f"Invalid or unknown {field}: {value!r}")


def _sha256_path(path: Path) -> str:
    """Hash a file incrementally to keep checkpoint and result verification memory-safe."""
    digest = hashlib.sha256()
    if path.is_dir():
        files = sorted(candidate for candidate in path.rglob("*") if candidate.is_file())
        if not files:
            raise FileNotFoundError(f"No files to hash under {path}")
        for candidate in files:
            digest.update(candidate.relative_to(path).as_posix().encode("utf-8"))
            digest.update(b"\0")
            with candidate.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
        return digest.hexdigest()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_path_hash(path: Path, expected: Any, cache: Dict[str, str], *, context: str) -> None:
    """Verify one file digest, reusing checks for shared inputs within a benchmark load."""
    if not path.is_file() and not path.is_dir():
        raise FileNotFoundError(f"{context} is unavailable: {path}")
    key = str(path.resolve())
    if key not in cache:
        cache[key] = _sha256_path(path)
    if cache[key] != str(expected).lower():
        raise ValueError(f"{context} SHA-256 mismatch: {path}")


def _first_series(df: pd.DataFrame, col_name: str) -> pd.Series:
    """Return a 1D Series for col_name, selecting the first matching column if duplicated."""
    col_obj = df[col_name]
    return col_obj.iloc[:, 0] if isinstance(col_obj, pd.DataFrame) else col_obj


def audit_exact_overlap(
    test_queries: Union[pd.DataFrame, Iterable[Tuple[str, str, int]]],
    train_sequences: Union[Iterable[str], Iterable[Tuple[str, str, int]], Set[str]],
) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
    """Audit exact sequence overlap between test queries and training references.

    Args:
      test_queries: DataFrame with ['query_id', 'sequence' (or 'sequence_padded'/'padded_Representation'), 'label']
        or iterable of (qid, seq, label).
      train_sequences: iterable of training sequence strings, tuples, or set of precomputed hashes.

    Returns:
      audit_df: DataFrame with query_id, label, train_count, has_overlap.
      masks: dict with boolean masks for 'all', 'overlap', 'no_exact_overlap'.
    """
    if isinstance(test_queries, pd.DataFrame):
        seq_col = next(
            (c for c in ("sequence_padded", "padded_Representation", "sequence") if c in test_queries.columns),
            None,
        )
        if seq_col is None:
            raise KeyError(
                "test_queries DataFrame must contain one of ('sequence_padded', 'padded_Representation', 'sequence')"
            )
        qids = _first_series(test_queries, "query_id").tolist()
        seqs = _first_series(test_queries, seq_col).astype(str).tolist()
        labels = [int(lbl) for lbl in _first_series(test_queries, "label").tolist()]
        records = list(zip(qids, seqs, labels))
    else:
        records = list(test_queries)

    # Build canonical reference hash counts; only explicit tuples may carry typed digests.
    train_counts: Dict[str, int] = {}
    for item in train_sequences:
        if isinstance(item, str):
            h = _sequence_digest(item)
        elif isinstance(item, (tuple, list)):
            if len(item) >= 3 and item[2] == "sha256_min_forward_rc_v1":
                h = _sequence_digest(str(item[1]), precomputed=True)
            else:
                h = _sequence_digest(str(item[1]))
        else:
            raise TypeError(f"Unsupported training sequence record: {type(item).__name__}")
        train_counts[h] = train_counts.get(h, 0) + 1

    rows = []
    for qid, seq, label in records:
        h = canonical_sequence_hash(seq)
        count = train_counts.get(h, 0)
        rows.append({
            "query_id": qid,
            "label": label,
            "train_count": count,
            "has_overlap": count > 0,
        })

    audit_df = pd.DataFrame(rows)
    has_overlap_arr = audit_df["has_overlap"].to_numpy()

    masks = {
        "all": np.ones(len(audit_df), dtype=bool),
        "overlap": has_overlap_arr,
        "no_exact_overlap": ~has_overlap_arr,
    }

    return audit_df, masks


def load_prepared_baseline_benchmark(
    benchmark_dir: Union[str, Path],
    methods: Optional[List[str]] = None,
    noise_levels: Optional[List[float]] = None,
) -> BenchmarkArtifacts:
    """Load and validate prepared baseline benchmark artifacts from cluster directory.

    Args:
      benchmark_dir: Root directory containing exact_overlap.csv (or truth.parquet)
        and comparison_results/ directory.
      methods: List of method names to load (default: 5 CNN variants, Kraken2, BLASTN).
      noise_levels: Nominal noise levels to load (default: [0.0, 0.02, 0.05, 0.10, 0.15]).

    Returns:
      BenchmarkArtifacts with audit_df, strata_masks, predictions, top_candidates, and runtime.

    Raises:
      FileNotFoundError: If benchmark_dir or essential artifacts are missing.
      ValueError: If query_ids are duplicate or misaligned with predictions.
    """
    bench_p = Path(benchmark_dir).resolve()
    if not bench_p.exists():
        raise FileNotFoundError(f"Baseline benchmark directory not found: {bench_p}")

    # 1. Locate truth / exact overlap audit table
    truth_candidates = [
        bench_p / "exact_overlap.csv",
        bench_p / "exact_overlap.parquet",
        bench_p / "truth.parquet",
        bench_p / "queries_truth.csv",
        bench_p / "comparison_results" / "exact_overlap.csv",
    ]
    truth_path = next((p for p in truth_candidates if p.exists()), None)
    if truth_path is None:
        raise FileNotFoundError(
            f"No exact overlap or truth table found in {bench_p}. Candidates: {[str(c.name) for c in truth_candidates]}"
        )
    log_artifact_path("baseline truth/overlap table", truth_path)

    if truth_path.suffix == ".parquet":
        audit_df = pd.read_parquet(truth_path)
    else:
        audit_df = pd.read_csv(truth_path)

    if "query_id" not in audit_df.columns:
        raise KeyError(f"Truth/overlap table {truth_path} must contain 'query_id' column")
    if "label" not in audit_df.columns:
        raise KeyError(f"Truth/overlap table {truth_path} must contain canonical zero-based 'label'")
    truth_labels = pd.to_numeric(audit_df["label"], errors="coerce")
    if truth_labels.isna().any() or (truth_labels < 0).any() or (truth_labels >= 13683).any() or not np.equal(truth_labels, np.floor(truth_labels)).all():
        raise ValueError(f"Truth labels must be canonical zero-based Species IDs 0..13682: {truth_path}")
    audit_df["label"] = truth_labels.astype(np.int64)

    if not audit_df["query_id"].is_unique:
        raise ValueError(f"Duplicate query_id found in truth table {truth_path}")

    # Determine overlap mask
    if "has_overlap" in audit_df.columns:
        has_overlap_arr = np.asarray([_strict_bool(v, field="has_overlap") for v in audit_df["has_overlap"]], dtype=bool)
        if "train_count" in audit_df.columns:
            counts = pd.to_numeric(audit_df["train_count"], errors="coerce")
            if counts.isna().any() or (counts < 0).any() or not np.equal(counts, np.floor(counts)).all():
                raise ValueError(f"Unknown or invalid train_count in {truth_path}")
            if not np.array_equal(has_overlap_arr, counts.to_numpy(dtype=np.int64) > 0):
                raise ValueError(f"has_overlap and train_count disagree in {truth_path}")
    elif "train_count" in audit_df.columns:
        counts = pd.to_numeric(audit_df["train_count"], errors="coerce")
        if counts.isna().any() or (counts < 0).any() or not np.equal(counts, np.floor(counts)).all():
            raise ValueError(f"Unknown or invalid train_count in {truth_path}")
        has_overlap_arr = (counts.astype(np.int64) > 0).to_numpy()
        audit_df["has_overlap"] = has_overlap_arr
    else:
        raise KeyError(f"Truth/overlap table {truth_path} must contain 'has_overlap' or 'train_count'")

    strata_masks = {
        "all": np.ones(len(audit_df), dtype=bool),
        "overlap": has_overlap_arr,
        "no_exact_overlap": ~has_overlap_arr,
    }

    truth_qids = audit_df["query_id"].tolist()
    truth_qids_set = set(truth_qids)

    # 2. Results directory
    comparison_dir = bench_p / "comparison_results" if (bench_p / "comparison_results").is_dir() else bench_p

    target_methods = methods if methods is not None else list(DEFAULT_METHOD_ALIASES.keys())
    target_noise = noise_levels if noise_levels is not None else [0.0, 0.02, 0.05, 0.10, 0.15]

    manifest_paths = sorted(comparison_dir.glob("manifest_*.json"))
    if not manifest_paths:
        raise FileNotFoundError(f"No per-method condition manifests found in {comparison_dir}")
    manifests: Dict[Tuple[str, float], Dict[str, Any]] = {}
    required = {"baseline_protocol", "method", "noise_rate", "noise_protocol", "query_manifest", "class_order_sha256", "reference", "tool_version", "command", "abstention_policy", "output_sha256", "completion", "is_smoke_test"}
    for manifest_path in manifest_paths:
        with manifest_path.open("r", encoding="utf-8") as handle:
            record = json.load(handle)
        missing = required - set(record)
        if missing:
            continue
        record["__manifest_path"] = str(manifest_path.resolve())
        record["__manifest_sha256"] = _sha256_path(manifest_path)
        key = (str(record["method"]), float(record["noise_rate"]))
        if key in manifests:
            raise ValueError(f"Conflicting method/condition manifests for {key}: {manifest_path}")
        manifests[key] = record

    class_orders: Set[str] = set()
    training_reference_hashes: Set[str] = set()
    reference_checks: Dict[str, str] = {}
    validated_class_order_files: Set[str] = set()
    manifest_records: List[Dict[str, Any]] = []

    predictions: Dict[str, Dict[float, np.ndarray]] = {m: {} for m in target_methods}
    top_candidates: Dict[str, Dict[float, Optional[np.ndarray]]] = {m: {} for m in target_methods}

    for method in target_methods:
        aliases = DEFAULT_METHOD_ALIASES.get(method, [method])
        for noise in target_noise:
            tag = f"noise_{round(noise * 100):02d}"
            # Locate prediction file trying aliases
            prediction_candidates: List[Path] = []
            for alias in aliases:
                p_cand = comparison_dir / f"predictions_{alias}_{tag}.parquet"
                if p_cand.exists():
                    prediction_candidates.append(p_cand)
                p_csv = comparison_dir / f"predictions_{alias}_{tag}.csv"
                if p_csv.exists():
                    prediction_candidates.append(p_csv)

            if not prediction_candidates:
                raise FileNotFoundError(
                    f"Prediction file for method '{method}' (aliases={aliases}) at noise {noise} ({tag}) not found in {comparison_dir}"
                )
            if len(prediction_candidates) != 1:
                raise ValueError(f"Conflicting prediction aliases for method={method}, noise={noise}: {prediction_candidates}")
            pred_file = prediction_candidates[0]

            matching = [m for (name, rate), m in manifests.items() if name in aliases and math.isclose(rate, float(noise), abs_tol=1e-12)]
            if len(matching) != 1:
                raise ValueError(f"Expected exactly one complete provenance manifest for method={method}, noise={noise}; found {len(matching)}")
            manifest = matching[0]
            if manifest.get("baseline_protocol") != "c03_paired_v1":
                raise ValueError(f"Unsupported baseline protocol for {method}/{noise}: {manifest.get('baseline_protocol')}")
            if manifest.get("output_path") and Path(str(manifest["output_path"])).resolve() != pred_file.resolve():
                raise ValueError(f"Manifest output path does not identify the discovered prediction file: {pred_file}")
            manifest_records.append({"method": method, "noise_rate": float(noise),
                                     "path": manifest["__manifest_path"],
                                     "sha256": manifest["__manifest_sha256"],
                                     "output_path": manifest.get("output_path", str(pred_file.resolve())),
                                     "output_sha256": manifest.get("output_sha256"),
                                     "query_manifest": manifest.get("query_manifest"),
                                     "class_order_sha256": manifest.get("class_order_sha256"),
                                     "reference": manifest.get("reference"),
                                     "model_variant": manifest.get("model_variant", manifest.get("variant")),
                                     "checkpoint": manifest.get("checkpoint")})
            if manifest["completion"] != "COMPLETE":
                raise ValueError(f"Manifest for {method}/{noise} is {manifest['completion']}, not COMPLETE")
            for field in ("reference", "tool_version", "command", "abstention_policy"):
                value = manifest.get(field)
                if value is None or value == "" or value == "unknown" or value == [] or value == {}:
                    raise ValueError(f"Empirical manifest field {field} is unknown/empty for {method}/{noise}")
            reference_info = manifest["reference"]
            if not isinstance(reference_info, dict) or not reference_info.get("path") or not reference_info.get("sha256"):
                raise ValueError(f"Empirical reference must include path and SHA-256: {method}/{noise}")
            if len(str(reference_info["sha256"])) != 64:
                raise ValueError(f"Invalid reference SHA-256 for {method}/{noise}")
            reference_path = Path(str(reference_info["path"]))
            _verify_path_hash(reference_path, reference_info["sha256"], reference_checks, context=f"Reference manifest for {method}/{noise}")
            reference_manifest = json.loads(reference_path.read_text(encoding="utf-8"))
            source_manifest = reference_manifest
            if reference_manifest.get("reference_manifest"):
                nested_path = Path(reference_manifest["reference_manifest"])
                _verify_path_hash(nested_path, reference_manifest.get("reference_manifest_sha256"), reference_checks, context="Nested reference manifest")
                source_manifest = json.loads(nested_path.read_text(encoding="utf-8"))
            source_path = source_manifest.get("train_path")
            source_hash = source_manifest.get("train_sha256")
            if not source_path or not source_hash:
                raise ValueError(f"Reference manifest has no identified training partition: {reference_path}")
            _verify_path_hash(Path(source_path), source_hash, reference_checks, context="Training reference partition")
            training_reference_hashes.add(str(source_hash).lower())
            if "reference_taxonomy" in source_manifest:
                for field, filename in (("kraken_reference_fasta", "train_kraken.fasta"), ("blast_reference_fasta", "train_blast.fasta"), ("nodes_dmp", "nodes.dmp"), ("names_dmp", "names.dmp")):
                    source = Path(source_manifest[field])
                    _verify_path_hash(source, source_manifest.get("sha256", {}).get(filename), reference_checks, context="Reference source artifact")
            db_info = reference_info.get("database_build_manifest")
            if db_info:
                db_path = Path(str(db_info.get("path", "")))
                _verify_path_hash(db_path, db_info.get("sha256"), reference_checks, context="Database build manifest")
                db_record = json.loads(db_path.read_text(encoding="utf-8"))
                if db_record.get("state") != "COMPLETE" or db_record.get("is_smoke_test") is not False:
                    raise ValueError(f"Database build is incomplete or synthetic: {db_path}")
                if db_record.get("reference_manifest_sha256") != reference_info["sha256"]:
                    raise ValueError(f"Database build did not use the declared reference manifest: {db_path}")
                expected_db_path = db_record.get("kraken_db" if method == "Kraken2-flat" else "blast_db_prefix")
                if not expected_db_path or Path(expected_db_path).resolve() != Path(str(reference_info.get("database_path", ""))).resolve():
                    raise ValueError(f"Method database path differs from the completed build manifest: {db_path}")
                for db_output, db_hash in db_record.get("output_sha256", {}).items():
                    _verify_path_hash(Path(db_output), db_hash, reference_checks, context="Built search database")
            if manifest["noise_protocol"] != "resample_including_original_v1":
                raise ValueError(f"Incompatible noise protocol for {method}/{noise}: {manifest['noise_protocol']}")
            if _strict_bool(manifest["is_smoke_test"], field="is_smoke_test"):
                raise ValueError(f"Synthetic smoke artifact cannot be ingested as empirical: {method}/{noise}")
            qmanifest_ref = manifest["query_manifest"]
            if isinstance(qmanifest_ref, dict):
                qmanifest_path = qmanifest_ref.get("path")
                qmanifest_sha = qmanifest_ref.get("sha256")
            else:
                qmanifest_path, qmanifest_sha = str(qmanifest_ref), None
            if not qmanifest_path:
                raise ValueError(f"Manifest has no query_manifest path for {method}/{noise}")
            qmanifest_path = Path(qmanifest_path)
            if not qmanifest_path.is_absolute():
                qmanifest_path = comparison_dir / qmanifest_path
            if not qmanifest_path.is_file():
                raise FileNotFoundError(f"Query manifest referenced by {method}/{noise} is missing: {qmanifest_path}")
            log_artifact_path("baseline query manifest", qmanifest_path, extra=f"method={method}, noise={noise}")
            if qmanifest_sha:
                _verify_path_hash(qmanifest_path, qmanifest_sha, reference_checks, context="Query manifest")
            with qmanifest_path.open("r", encoding="utf-8") as handle:
                qmanifest = json.load(handle)
            if qmanifest.get("baseline_protocol") != "c03_paired_v1" or not math.isclose(float(qmanifest.get("noise_rate", -1)), float(noise), abs_tol=1e-12):
                raise ValueError(f"Query manifest protocol/condition mismatch: {qmanifest_path}")
            if qmanifest.get("noise_protocol") != "resample_including_original_v1" or qmanifest.get("is_smoke_test") is not False:
                raise ValueError(f"Query manifest is incompatible or synthetic: {qmanifest_path}")
            prep_ref = qmanifest.get("preparation_manifest")
            if not isinstance(prep_ref, dict) or not prep_ref.get("path") or not prep_ref.get("sha256"):
                raise ValueError(f"Query manifest does not identify its preparation/source inventory: {qmanifest_path}")
            prep_path = Path(prep_ref["path"])
            if not prep_path.is_absolute():
                prep_path = qmanifest_path.parent / prep_path
            _verify_path_hash(prep_path, prep_ref["sha256"], reference_checks, context="C03 preparation manifest")
            prep_meta = json.loads(prep_path.read_text(encoding="utf-8"))
            if prep_meta.get("baseline_protocol") != "c03_paired_v1" or prep_meta.get("is_smoke_test") is not False:
                raise ValueError(f"Preparation manifest is not empirical C03: {prep_path}")
            if str(prep_meta.get("class_order_sha256", "")).lower() != str(qmanifest.get("class_order_sha256", "")).lower():
                raise ValueError(f"Preparation/query class order differs: {prep_path}")
            for field in ("source_train", "source_test", "class_order_file", "query_identity", "clean_queries", "exact_overlap"):
                artifact = qmanifest.get(field)
                if not isinstance(artifact, dict) or not artifact.get("path") or not artifact.get("sha256"):
                    raise ValueError(f"Query manifest is missing {field} identity: {qmanifest_path}")
                artifact_path = Path(artifact["path"])
                if not artifact_path.is_absolute():
                    artifact_path = qmanifest_path.parent / artifact_path
                _verify_path_hash(artifact_path, artifact["sha256"], reference_checks, context=f"C03 {field}")
                if field == "class_order_file" and str(artifact_path.resolve()) not in validated_class_order_files:
                    class_frame = pd.read_csv(artifact_path)
                    if list(class_frame.columns) != ["species", "id"] or len(class_frame) != 13683 or not np.array_equal(class_frame["id"].to_numpy(), np.arange(1, 13684)):
                        raise ValueError(f"Species class-order table has an invalid ID order: {artifact_path}")
                    observed_order = hashlib.sha256("\n".join(class_frame["species"].astype(str)).encode()).hexdigest()
                    if observed_order != qmanifest.get("class_order_sha256"):
                        raise ValueError(f"Species class-order table digest mismatch: {artifact_path}")
                    validated_class_order_files.add(str(artifact_path.resolve()))
            if Path(qmanifest["exact_overlap"]["path"]).resolve() != truth_path.resolve():
                raise ValueError(f"Loaded overlap audit is not the source preparation audit: {truth_path}")
            identity_path = Path(qmanifest["query_identity"]["path"])
            identity_df = pd.read_csv(identity_path, dtype={"query_id": str})
            if not identity_df["query_id"].is_unique or set(identity_df["query_id"]) != truth_qids_set:
                raise ValueError(f"Query identity table does not match truth IDs: {identity_path}")
            identity_labels = audit_df.set_index("query_id").loc[identity_df["query_id"], "label"].to_numpy(dtype=np.int64)
            if not np.array_equal(identity_df["label"].to_numpy(dtype=np.int64), identity_labels):
                raise ValueError(f"Query identity labels differ from truth: {identity_path}")
            clean_path = Path(qmanifest["clean_queries"]["path"])
            clean_df = pd.read_csv(clean_path, dtype={"query_id": str})
            if set(clean_df["query_id"]) != truth_qids_set or "sequence" not in clean_df.columns:
                raise ValueError(f"Clean query table does not match source query identity: {clean_path}")
            if not clean_df["query_id"].is_unique:
                raise ValueError(f"Duplicate query ID in clean query table: {clean_path}")
            identity_by_id = identity_df.set_index("query_id")["sequence_sha256"].astype(str)
            clean_hashes = clean_df.set_index("query_id")["sequence"].astype(str).str.upper().map(lambda sequence: hashlib.sha256(sequence.encode("ascii")).hexdigest())
            if not identity_by_id.sort_index().equals(clean_hashes.sort_index()):
                raise ValueError(f"Clean source sequences differ from selected query identities: {clean_path}")
            if math.isclose(float(noise), 0.0, abs_tol=1e-12):
                clean_by_id = clean_df.set_index("query_id")["sequence"].astype(str).str.upper()
                noisy_by_id = pd.read_csv(qmanifest_path.parent / qmanifest_path.name.replace("query_manifest_", "queries_").replace(".json", ".csv"), dtype={"query_id": str}).set_index("query_id")["sequence"].astype(str).str.upper()
                if not clean_by_id.sort_index().equals(noisy_by_id.sort_index()):
                    raise ValueError(f"epsilon=0 query content differs from the selected clean queries: {clean_path}")
            if not qmanifest.get("fasta_sha256") or len(str(qmanifest["fasta_sha256"])) != 64:
                raise ValueError(f"Query manifest has no valid FASTA hash: {qmanifest_path}")
            if qmanifest.get("class_order_sha256") is None:
                raise ValueError(f"Query manifest has no class-order digest: {qmanifest_path}")
            if str(source_manifest.get("class_order_sha256", "")).lower() != str(qmanifest["class_order_sha256"]).lower():
                raise ValueError(f"Reference and query manifests declare different class orders: {reference_path}")
            query_fasta = qmanifest_path.parent / qmanifest_path.name.replace("query_manifest_", "queries_").replace(".json", ".fasta")
            _verify_path_hash(query_fasta, qmanifest["fasta_sha256"], reference_checks, context="Materialized query FASTA")
            query_table = qmanifest_path.parent / qmanifest_path.name.replace("query_manifest_", "queries_").replace(".json", ".csv")
            _verify_path_hash(query_table, qmanifest.get("query_content_sha256"), reference_checks, context="Materialized query content")
            log_artifact_path("materialized baseline queries", query_table, extra=f"method={method}, noise={noise}")
            query_df = pd.read_csv(query_table, dtype={"query_id": str})
            if not query_df["query_id"].is_unique or set(query_df["query_id"]) != truth_qids_set:
                raise ValueError(f"Materialized queries do not exactly match truth IDs: {query_table}")
            if "label" not in query_df:
                raise ValueError(f"Materialized query table has no label column: {query_table}")
            query_truth = audit_df.set_index("query_id").loc[query_df["query_id"], "label"].to_numpy(dtype=np.int64)
            if not np.array_equal(query_df["label"].to_numpy(dtype=np.int64), query_truth):
                raise ValueError(f"Materialized query labels diverge from truth: {query_table}")
            expected_qid_digest = hashlib.sha256("\n".join(query_df["query_id"].tolist()).encode()).hexdigest()
            if expected_qid_digest != qmanifest.get("query_ids_sha256"):
                raise ValueError(f"Query ID digest mismatch: {query_table}")
            expected_hash = str(manifest["output_sha256"]).lower()
            actual_hash = _sha256_path(pred_file)
            if expected_hash != actual_hash:
                raise ValueError(f"Prediction hash mismatch for {pred_file}: manifest={expected_hash}, actual={actual_hash}")
            execution = manifest.get("execution_details", {})
            if execution.get("query_csv_sha256") != qmanifest.get("query_content_sha256"):
                raise ValueError(f"Result does not identify the shared query CSV content: {pred_file}")
            if Path(str(execution.get("query_csv_path", ""))).resolve() != query_table.resolve():
                raise ValueError(f"Result command consumed a different query CSV path: {pred_file}")
            if method in {"Kraken2-flat", "BLASTN"}:
                if execution.get("query_fasta_sha256") != qmanifest.get("fasta_sha256"):
                    raise ValueError(f"Result does not identify the shared query FASTA content: {pred_file}")
                if Path(str(execution.get("query_fasta_path", ""))).resolve() != query_fasta.resolve():
                    raise ValueError(f"Result command consumed a different query FASTA path: {pred_file}")

            if pred_file.suffix == ".parquet":
                log_artifact_path("baseline predictions", pred_file, extra=f"method={method}, noise={noise}")
                pred_df = pd.read_parquet(pred_file)
            else:
                log_artifact_path("baseline predictions", pred_file, extra=f"method={method}, noise={noise}")
                pred_df = pd.read_csv(pred_file)

            if "query_id" not in pred_df.columns:
                raise KeyError(f"Missing 'query_id' column in {pred_file}")

            if not pred_df["query_id"].is_unique:
                raise ValueError(f"Duplicate 'query_id' detected in prediction file: {pred_file}")

            if set(pred_df["query_id"]) != truth_qids_set:
                raise ValueError(
                    f"Query ID mismatch between truth ({len(truth_qids_set)}) and predictions in {pred_file} ({len(pred_df['query_id'])})"
                )
            if "label" not in pred_df.columns:
                raise KeyError(f"Prediction file must carry canonical truth labels for validation: {pred_file}")
            truth_labels = audit_df.set_index("query_id").loc[pred_df["query_id"], "label"].to_numpy()
            if not np.array_equal(pred_df["label"].to_numpy(dtype=np.int64), truth_labels.astype(np.int64)):
                raise ValueError(f"Truth label mismatch in {pred_file}")

            pred_col = next(
                (c for c in ("prediction", "y_pred", "pred", "label_pred") if c in pred_df.columns),
                None,
            )
            if pred_col is None:
                raise KeyError(f"No prediction column ('prediction', 'y_pred') found in {pred_file}")

            aligned_series = pred_df.set_index("query_id").loc[truth_qids, pred_col]
            if aligned_series.isna().any():
                raise ValueError(f"Missing predictions (NaN) found in {pred_file}")

            predictions[method][noise] = aligned_series.to_numpy(dtype=np.int64)
            if method not in {"Kraken2-flat", "BLASTN"}:
                variant = manifest.get("model_variant", manifest.get("variant"))
                expected_variant = {
                    "resnet50_species": "resnet_clean_species",
                    "resnet_clean_species": "resnet_clean_species",
                    "resnet_noise_species": "resnet_noise_species",
                    "convnext_zero_species": "convnext_zero_species",
                    "convnext_transfer_species": "convnext_transfer_species",
                    "convnext_noise_species": "convnext_noise_species",
                }.get(method)
                if expected_variant is None or variant != expected_variant:
                    raise ValueError(f"CNN method/checkpoint variant mismatch for {pred_file}: method={method}, variant={variant}")
                checkpoint_hash = manifest.get("checkpoint_sha256")
                checkpoint_info = manifest.get("checkpoint")
                if isinstance(checkpoint_info, dict):
                    checkpoint_hash = checkpoint_info.get("sha256", checkpoint_hash)
                    checkpoint_path = checkpoint_info.get("path")
                    if checkpoint_path and Path(checkpoint_path).is_file():
                        log_artifact_path("baseline model checkpoint", checkpoint_path, extra=f"variant={variant}")
                        if _sha256_path(Path(checkpoint_path)) != str(checkpoint_hash).lower():
                            raise ValueError(f"Checkpoint hash mismatch for {pred_file}: {checkpoint_path}")
                if not variant or not checkpoint_hash:
                    raise ValueError(f"CNN manifest must identify model_variant and checkpoint_sha256: {pred_file}")
                if not isinstance(checkpoint_info, dict) or not checkpoint_info.get("path"):
                    raise ValueError(f"CNN checkpoint path is missing from manifest: {pred_file}")
                _verify_path_hash(Path(checkpoint_info["path"]), checkpoint_hash, reference_checks, context=f"CNN checkpoint {variant}")
                if np.any(predictions[method][noise] == -1):
                    raise ValueError(f"Dense CNN prediction unexpectedly abstains: {pred_file}")
                if len(str(checkpoint_hash)) != 64 or any(char not in "0123456789abcdefABCDEF" for char in str(checkpoint_hash)):
                    raise ValueError(f"CNN manifest has invalid checkpoint_sha256: {pred_file}")
            labels = predictions[method][noise]
            if np.any((labels < -1) | (labels >= 13683)):
                raise ValueError(f"Prediction outside -1..13682 in {pred_file}")
            order_hash = manifest["class_order_sha256"]
            if not isinstance(order_hash, str) or len(order_hash) != 64:
                raise ValueError(f"Invalid class_order_sha256 in manifest for {pred_file}")
            if any(char not in "0123456789abcdefABCDEF" for char in order_hash):
                raise ValueError(f"Invalid class_order_sha256 in manifest for {pred_file}")
            if str(qmanifest.get("class_order_sha256", "")).lower() != order_hash.lower():
                raise ValueError(f"Class-order digest differs between result and query manifest: {pred_file}")
            class_orders.add(order_hash.lower())

            # Top-5 candidates if available
            top_cols = [f"top{k}" for k in range(1, 6)]
            if all(c in pred_df.columns for c in top_cols):
                top_arr = pred_df.set_index("query_id").loc[truth_qids, top_cols].to_numpy(dtype=np.int64)
                if np.any((top_arr < -1) | (top_arr >= 13683)):
                    raise ValueError(f"Top-k label outside -1..13682 in {pred_file}")
                for row_index, row in enumerate(top_arr):
                    valid = row[row >= 0]
                    if len(valid) != len(np.unique(valid)):
                        raise ValueError(f"Duplicate Top-k species candidate in row {row_index} of {pred_file}")
                top_candidates[method][noise] = top_arr
                if method not in {"Kraken2-flat", "BLASTN"} and not np.array_equal(top_arr[:, 0], predictions[method][noise]):
                    raise ValueError(f"CNN Top-1 differs from prediction column in {pred_file}")
            else:
                top_candidates[method][noise] = None
            if method in {"Kraken2-flat", "BLASTN"}:
                if "abstention_reason" not in pred_df.columns:
                    raise KeyError(f"Tool predictions must preserve explicit abstention_reason: {pred_file}")
                is_abstention = predictions[method][noise] == -1
                reasons = pred_df.set_index("query_id").loc[truth_qids, "abstention_reason"].fillna("").astype(str).to_numpy()
                if np.any(is_abstention & (reasons == "")) or np.any((~is_abstention) & (reasons != "")):
                    raise ValueError(f"Tool abstention labels/reasons disagree in {pred_file}")
                if method == "BLASTN":
                    if not all(c in pred_df.columns for c in top_cols):
                        raise ValueError(f"BLASTN output must preserve the five unique species candidates: {pred_file}")
                    aligned_top1 = pred_df.set_index("query_id").loc[truth_qids, "top1"].to_numpy(dtype=np.int64)
                    if not np.array_equal(aligned_top1, predictions[method][noise]):
                        raise ValueError(f"BLASTN Top-1 does not match the first ranked species: {pred_file}")

    # 3. Optional runtime records
    if len(class_orders) != 1:
        raise ValueError(f"Methods/conditions do not share one declared canonical class order: {sorted(class_orders)}")
    if len(training_reference_hashes) != 1:
        raise ValueError(f"Methods do not share one identified train reference partition: {sorted(training_reference_hashes)}")
    runtime_path = comparison_dir / "runtime.jsonl"
    runtime_df: Optional[pd.DataFrame] = None
    if runtime_path.exists():
        try:
            records = []
            with runtime_path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        records.append(json.loads(line))
            if records:
                runtime_df = pd.DataFrame(records)
        except Exception:
            runtime_df = None

    return BenchmarkArtifacts(
        audit_df=audit_df,
        strata_masks=strata_masks,
        predictions=predictions,
        top_candidates=top_candidates,
        runtime=runtime_df,
        method_manifests=manifest_records,
    )

