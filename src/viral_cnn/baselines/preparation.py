"""Opt-in preparation of identified short-fragment baseline queries."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd

from viral_cnn.baselines.overlap import audit_exact_overlap, canonical_sequence_hash
from viral_cnn.utils.logging_utils import log_artifact_path

BASELINE_PROTOCOL = "c03_paired_v1"
NOISE_PROTOCOL = "resample_including_original_v1"
NOISE_RATES = (0.0, 0.02, 0.05, 0.10, 0.15)


def sha256_file(path: Path) -> str:
    """Return a content-addressed digest for a file or a directory of artifacts."""
    if path.is_dir():
        digest = hashlib.sha256()
        files = sorted(candidate for candidate in path.rglob("*") if candidate.is_file())
        if not files:
            raise FileNotFoundError(f"No files to hash under {path}")
        for candidate in files:
            digest.update(candidate.relative_to(path).as_posix().encode("utf-8"))
            digest.update(b"\0")
            with candidate.open("rb") as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(block)
        return digest.hexdigest()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_method_condition_manifest(
    path: Path,
    *,
    method: str,
    noise_rate: float,
    query_manifest: Path,
    class_order_sha256: str,
    reference: Dict[str, Any],
    tool_version: str,
    command: Sequence[str],
    abstention_policy: str,
    completion: str,
    is_smoke_test: bool,
    output_path: Path | None = None,
    model_variant: str | None = None,
    checkpoint_path: Path | None = None,
    error: str | None = None,
    execution_details: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Persist one method/condition state and observed provenance; COMPLETE requires a real output file."""
    states = {"PLANNED", "RUNNING", "COMPLETE", "FAILED"}
    if completion not in states:
        raise ValueError(f"completion must be one of {sorted(states)}")
    if not query_manifest.is_file():
        raise FileNotFoundError(f"Query manifest is missing: {query_manifest}")
    if len(class_order_sha256) != 64:
        raise ValueError("class_order_sha256 must be a SHA-256 digest")
    if completion == "COMPLETE" and (output_path is None or not output_path.is_file()):
        raise ValueError("COMPLETE method-condition requires an existing output file")
    if completion == "FAILED" and not error:
        raise ValueError("FAILED method-condition requires an error description")
    record: Dict[str, Any] = {
        "baseline_protocol": BASELINE_PROTOCOL,
        "method": method,
        "noise_rate": float(noise_rate),
        "noise_protocol": NOISE_PROTOCOL,
        "query_manifest": {"path": str(query_manifest.resolve()), "sha256": sha256_file(query_manifest)},
        "class_order_sha256": class_order_sha256,
        "reference": reference,
        "tool_version": tool_version,
        "command": list(command),
        "abstention_policy": abstention_policy,
        "output_sha256": sha256_file(output_path) if completion == "COMPLETE" and output_path else None,
        "output_path": str(output_path.resolve()) if output_path else None,
        "completion": completion,
        "is_smoke_test": bool(is_smoke_test),
    }
    if model_variant:
        record["model_variant"] = model_variant
    if checkpoint_path:
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"CNN checkpoint does not exist: {checkpoint_path}")
        record["checkpoint"] = {"path": str(checkpoint_path.resolve()), "sha256": sha256_file(checkpoint_path)}
    if error:
        record["error"] = error
    if execution_details:
        record["execution_details"] = execution_details
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return record


def _read_table(path: Path) -> pd.DataFrame:
    """Read an explicitly identified CSV or Parquet input."""
    log_artifact_path("baseline input dataset", path)
    return pd.read_parquet(path) if path.is_dir() or path.suffix.lower() == ".parquet" else pd.read_csv(path)


def _dataset_available(path: Path) -> bool:
    """Accept a single supported table or a Parquet partition directory."""
    return path.is_file() or (path.is_dir() and any(path.rglob("*.parquet")))


def inventory_inputs(train_path: Path, test_path: Path, methods: Sequence[str]) -> Dict[str, Any]:
    """Report available inputs and expected method-condition outputs without doing work."""
    expected = [{"method": m, "noise_rate": e} for m in methods for e in NOISE_RATES]
    return {
        "baseline_protocol": BASELINE_PROTOCOL,
        "inputs": {"train": {"path": str(train_path.resolve()), "exists": _dataset_available(train_path)},
                   "test": {"path": str(test_path.resolve()), "exists": _dataset_available(test_path)}},
        "expected_conditions": expected,
        "available": bool(_dataset_available(train_path) and _dataset_available(test_path)),
        "actions_performed": [],
    }


def _sequence_column(frame: pd.DataFrame) -> str:
    for name in ("sequence", "sequence_padded", "padded_Representation"):
        if name in frame.columns:
            return name
    raise KeyError("Input table needs a sequence, sequence_padded, or padded_Representation column")


def _priority(query_id: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{query_id}".encode("utf-8")).hexdigest()


def _noisy_sequence(sequence: str, epsilon: float, rng: np.random.Generator) -> tuple[str, int, int]:
    bases = np.frombuffer(sequence.upper().encode("ascii"), dtype=np.uint8).copy()
    valid = np.isin(bases, np.frombuffer(b"ACGT", dtype=np.uint8))
    selected = (rng.random(len(bases)) < epsilon) & valid
    eligible = int(valid.sum())
    changes = 0
    if selected.any():
        original = bases[selected].copy()
        bases[selected] = rng.choice(np.frombuffer(b"ACGT", dtype=np.uint8), size=int(selected.sum()))
        changes = int(np.count_nonzero(bases[selected] != original))
    return bases.tobytes().decode("ascii"), eligible, changes


def bind_query_manifests(output_dir: Path) -> None:
    """Link each condition manifest to its finalized preparation/source inventory."""
    prep_path = output_dir / "preparation_manifest.json"
    prep = json.loads(prep_path.read_text(encoding="utf-8"))
    shared = {
        "preparation_manifest": {"path": str(prep_path.resolve()), "sha256": sha256_file(prep_path)},
        "query_identity": {"path": prep["query_identity_path"], "sha256": prep["query_identity_sha256"]},
        "clean_queries": {"path": str((output_dir / "queries_clean.csv").resolve()), "sha256": prep["clean_queries_sha256"]},
        "exact_overlap": {"path": str((output_dir / "exact_overlap.csv").resolve()), "sha256": prep["exact_overlap_sha256"]},
        "source_train": {"path": prep["train_path"], "sha256": prep["train_sha256"]},
        "source_test": {"path": prep["test_path"], "sha256": prep["test_sha256"]},
        "class_order_file": {"path": prep["class_order_path"], "sha256": prep["class_order_file_sha256"]},
    }
    for item in prep["conditions"]:
        manifest_path = Path(item["query_manifest"])
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest.update(shared)
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def prepare_queries(train_path: Path, test_path: Path, output_dir: Path, *, seed: int = 42,
                    per_species: int = 20, class_order_path: Path | None = None,
                    is_smoke_test: bool = True,
                    tool_sequence_policy: str = "historical_padding_to_N_v1") -> Dict[str, Any]:
    """Select deterministic test queries, compute clean overlap, and materialize shared noise FASTAs."""
    if seed != 42 or per_species != 20:
        # Other values are allowed for explicit sensitivity scenarios, but must be named in output.
        pass
    output_dir.mkdir(parents=True, exist_ok=False)
    train, test = _read_table(train_path), _read_table(test_path)
    if class_order_path is None:
        class_order_path = Path(__file__).resolve().parents[3] / "dataset_kraken/species_name_ids.csv"
    log_artifact_path("Species class-order mapping", class_order_path)
    class_order = pd.read_csv(class_order_path)
    if list(class_order.columns) != ["species", "id"] or len(class_order) != 13683 or not np.array_equal(class_order["id"].to_numpy(), np.arange(1, 13684)):
        raise ValueError(f"Species class-order mapping must contain ordered IDs 1..13683: {class_order_path}")
    class_order_sha256 = hashlib.sha256("\n".join(class_order["species"].astype(str)).encode()).hexdigest()
    train_seq_col, test_seq_col = _sequence_column(train), _sequence_column(test)
    for split_name, frame, column in (("train", train, train_seq_col), ("test", test, test_seq_col)):
        lengths = frame[column].dropna().astype(str).str.len()
        if not lengths.eq(512).all():
            raise ValueError(f"{split_name} C03 inputs must be fixed 512-character fragments; observed lengths {sorted(lengths.unique().tolist())[:10]}")
    if "label" not in test or "label" not in train:
        raise KeyError("Train and test inputs must contain canonical zero-based label")
    for split_name, frame in (("train", train), ("test", test)):
        labels = pd.to_numeric(frame["label"], errors="coerce")
        if labels.isna().any() or (labels < 0).any() or (labels >= len(class_order)).any() or not np.equal(labels, np.floor(labels)).all():
            raise ValueError(f"{split_name} labels are not canonical Species indices 0..13682")
        frame["label"] = labels.astype(np.int64)
    if "query_id" not in test:
        test = test.copy()
        test["query_id"] = [
            hashlib.sha256(f"{test_path.name}:{i}:{test.iloc[i]['label']}:{canonical_sequence_hash(str(test.iloc[i][test_seq_col]))}".encode()).hexdigest()
            for i in range(len(test))
        ]
    test["query_id"] = test["query_id"].astype("string")
    if test["query_id"].isna().any() or not test["query_id"].is_unique or test["query_id"].str.strip().eq("").any() or test["query_id"].str.contains(r"\s", regex=True).any():
        raise ValueError("Test query_id values must be non-null, unique, nonempty, and whitespace-free for FASTA identity")
    test = test.copy()
    test["_priority"] = test["query_id"].astype(str).map(lambda q: _priority(q, seed))
    eligible = test[test[test_seq_col].notna() & test["label"].notna()].copy()
    eligible["_source_row_ordinal"] = eligible.index.to_numpy(dtype=np.int64)
    selected = eligible.sort_values(["label", "_priority", "query_id"]).groupby("label", sort=True).head(per_species)
    selected = selected.sort_values(["label", "_priority", "query_id"]).drop(columns="_priority").reset_index(drop=True)
    train_sequences = train[train_seq_col].dropna().astype(str).tolist()
    overlap, _ = audit_exact_overlap(selected.rename(columns={test_seq_col: "sequence"}), train_sequences)
    truth = selected[["query_id", "label"]].copy()
    truth["query_id"] = truth["query_id"].astype(str)
    truth = truth.merge(overlap[["query_id", "train_count", "has_overlap"]], on="query_id", validate="one_to_one")
    truth["genome_in_reference_database"] = pd.NA
    truth.to_csv(output_dir / "exact_overlap.csv", index=False)
    (output_dir / "queries_clean.csv").write_text(selected[["query_id", "label", test_seq_col]].rename(columns={test_seq_col:"sequence"}).to_csv(index=False), encoding="utf-8")
    identity = selected[["query_id", "label", test_seq_col, "_source_row_ordinal"]].copy().rename(columns={test_seq_col: "sequence", "_source_row_ordinal": "source_row_ordinal"})
    identity["sequence_sha256"] = identity["sequence"].astype(str).map(lambda value: hashlib.sha256(value.upper().encode("ascii")).hexdigest())
    identity.drop(columns="sequence").to_csv(output_dir / "query_identity.csv", index=False)
    if tool_sequence_policy != "historical_padding_to_N_v1":
        raise ValueError(f"Unsupported explicit tool sequence policy: {tool_sequence_policy}")
    reference_dir = output_dir / "references"
    reference_dir.mkdir()
    blast_fasta = reference_dir / "train_blast.fasta"
    kraken_fasta = reference_dir / "train_kraken.fasta"
    nodes_path = reference_dir / "nodes.dmp"
    names_path = reference_dir / "names.dmp"
    with blast_fasta.open("w", encoding="ascii") as blast_handle, kraken_fasta.open("w", encoding="ascii") as kraken_handle:
        for ordinal, (_, ref) in enumerate(train.iterrows()):
            if pd.isna(ref[train_seq_col]) or pd.isna(ref["label"]):
                continue
            raw = str(ref[train_seq_col]).upper()
            sequence = raw.rstrip("_").replace("_", "N")
            label = int(ref["label"])
            blast_handle.write(f">train_{ordinal}_label_{label}\n{sequence}\n")
            kraken_handle.write(f">train_{ordinal}|kraken:taxid|{label + 2}\n{sequence}\n")
    with nodes_path.open("w", encoding="ascii") as nodes_handle, names_path.open("w", encoding="utf-8") as names_handle:
        nodes_handle.write("1\t|\t1\t|\tno rank\t|\t\t|\t0\t|\t0\t|\t0\t|\t0\t|\t0\t|\t0\t|\t0\t|\t0\t|\t\t|\n")
        for label, species_name in enumerate(class_order["species"].astype(str)):
            taxid = label + 2
            nodes_handle.write(f"{taxid}\t|\t1\t|\tspecies\t|\t\t|\t0\t|\t0\t|\t0\t|\t0\t|\t0\t|\t0\t|\t0\t|\t0\t|\t\t|\n")
            safe_name = species_name.replace("\t", " ").replace("|", " ")
            names_handle.write(f"{taxid}\t|\t{safe_name}\t|\t\t|\tscientific name\t|\n")
    ref_manifest = {
        "baseline_protocol": BASELINE_PROTOCOL, "reference_taxonomy": "Kraken2-flat",
        "custom_taxid_formula": "label + 2", "class_order_sha256": class_order_sha256,
        "train_path": str(train_path.resolve()), "train_sha256": sha256_file(train_path),
        "source_rows": len(train), "sequence_policy": tool_sequence_policy,
        "kraken_reference_fasta": str(kraken_fasta.resolve()), "blast_reference_fasta": str(blast_fasta.resolve()),
        "nodes_dmp": str(nodes_path.resolve()), "names_dmp": str(names_path.resolve()),
        "sha256": {p.name: sha256_file(p) for p in (kraken_fasta, blast_fasta, nodes_path, names_path)},
        "is_smoke_test": bool(is_smoke_test),
    }
    ref_manifest_path = reference_dir / "reference_manifest.json"
    ref_manifest_path.write_text(json.dumps(ref_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    condition_records = []
    for rate in NOISE_RATES:
        rate_tag = f"noise_{round(rate * 100):02d}"
        query_rows, fasta = [], []
        total_eligible = total_changes = 0
        for _, row in selected.iterrows():
            qid, label = str(row["query_id"]), int(row["label"])
            sequence = str(row[test_seq_col]).upper()
            # Independent deterministic stream per rate and query keeps sampling stable under row order changes.
            qseed = int(hashlib.sha256(f"{seed}:{rate:.2f}:{qid}".encode()).hexdigest()[:16], 16)
            noisy, eligible_count, change_count = _noisy_sequence(sequence, rate, np.random.default_rng(qseed))
            query_rows.append({"query_id": qid, "label": label, "sequence": noisy})
            fasta.extend([f">{qid}", noisy.rstrip("_").replace("_", "N")])
            total_eligible += eligible_count
            total_changes += change_count
        query_path = output_dir / f"queries_{rate_tag}.csv"
        fasta_path = output_dir / f"queries_{rate_tag}.fasta"
        pd.DataFrame(query_rows).to_csv(query_path, index=False)
        fasta_path.write_text("\n".join(fasta) + "\n", encoding="ascii")
        qmanifest = {"baseline_protocol": BASELINE_PROTOCOL, "noise_rate": rate,
                     "noise_protocol": NOISE_PROTOCOL, "seed": seed,
                     "seed_derivation": "sha256(seed:noise_rate:query_id) first 64 bits; numpy.default_rng",
                     "sampling_order": "Bernoulli position mask, then uniform A/C/G/T resample including original",
                     "tool_sequence_policy": tool_sequence_policy,
                     "positions_eligible": total_eligible, "effective_changes_observed": total_changes,
                     "effective_change_rate_observed": total_changes / total_eligible if total_eligible else None,
                     "query_ids_sha256": hashlib.sha256("\n".join(r["query_id"] for r in query_rows).encode()).hexdigest(),
                     "query_content_sha256": sha256_file(query_path), "fasta_sha256": sha256_file(fasta_path),
                     "class_order_sha256": class_order_sha256, "is_smoke_test": bool(is_smoke_test)}
        manifest_path = output_dir / f"query_manifest_{rate_tag}.json"
        manifest_path.write_text(json.dumps(qmanifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        condition_records.append({"noise_rate": rate, "query_manifest": str(manifest_path), "query_content_sha256": qmanifest["query_content_sha256"]})
    run_manifest = {"baseline_protocol": BASELINE_PROTOCOL, "seed": seed, "per_species": per_species,
                    "train_path": str(train_path.resolve()), "train_sha256": sha256_file(train_path),
                    "test_path": str(test_path.resolve()), "test_sha256": sha256_file(test_path),
                    "train_rows": len(train), "test_rows": len(test), "eligible_test_rows": len(eligible),
                    "class_order_path": str(class_order_path.resolve()), "class_order_sha256": class_order_sha256,
                    "selected_queries": len(selected), "selected_per_species_cap": per_species,
                    "excluded_rows": int(len(test) - len(eligible)),
                    "exclusions_by_reason": {"missing_sequence_or_label": int(len(test) - len(eligible))},
                    "eligible_population_by_species": {str(int(k)): int(v) for k, v in eligible.groupby("label").size().items()},
                    "selected_by_species": {str(int(k)): int(v) for k, v in selected.groupby("label").size().items()},
                    "query_identity_path": str((output_dir / "query_identity.csv").resolve()),
                    "query_identity_sha256": sha256_file(output_dir / "query_identity.csv"),
                    "clean_queries_sha256": sha256_file(output_dir / "queries_clean.csv"),
                    "clean_queries_path": str((output_dir / "queries_clean.csv").resolve()),
                    "exact_overlap_sha256": sha256_file(output_dir / "exact_overlap.csv"),
                    "reference_manifest": str(ref_manifest_path.resolve()),
                    "reference_manifest_sha256": sha256_file(ref_manifest_path),
                    "class_order_file_sha256": sha256_file(class_order_path),
                    "conditions": condition_records,
                    "is_smoke_test": bool(is_smoke_test)}
    (output_dir / "preparation_manifest.json").write_text(json.dumps(run_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    bind_query_manifests(output_dir)
    return run_manifest

