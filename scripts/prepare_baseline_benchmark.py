#!/usr/bin/env python3
"""Inventory or explicitly prepare C03 paired baseline benchmark inputs."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from viral_cnn.baselines.preparation import bind_query_manifests, inventory_inputs, prepare_queries, sha256_file


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/baseline_c03.yaml")
    parser.add_argument("--mode", choices=("inventory", "export", "plan", "build-databases"), default="inventory")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument("--prepared-dir", type=Path, default=None)
    parser.add_argument("--threads", type=int, default=16)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    config_path = args.config.resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"C03 config not found: {config_path}")
    with config_path.open(encoding="utf-8") as handle:
        config: dict[str, Any] = yaml.safe_load(handle)
    paths = config["inputs"]
    data_root = Path(os.environ.get("DATA_ROOT", config.get("data_root", "."))).expanduser()
    train_path = (data_root / paths["train"]).resolve()
    test_path = (data_root / paths["test"]).resolve()
    methods = config["methods"]
    inventory = inventory_inputs(train_path, test_path, methods)
    inventory["config_path"] = str(config_path)
    if args.mode == "inventory":
        print(json.dumps(inventory, indent=2, sort_keys=True))
        return 0 if inventory["available"] else 2
    if args.mode == "build-databases":
        if args.prepared_dir is None:
            raise ValueError("--prepared-dir is required for --mode build-databases")
        prepared_dir = args.prepared_dir.resolve()
        ref_dir = prepared_dir / "references"
        ref_manifest = ref_dir / "reference_manifest.json"
        if not ref_manifest.is_file():
            raise FileNotFoundError(f"Reference export not found: {ref_manifest}; run --mode export first")
        ref_data = json.loads(ref_manifest.read_text(encoding="utf-8"))
        kraken_db = ref_dir / "kraken_db"
        blast_prefix = ref_dir / "blast_db" / "train"
        db_manifest_path = ref_dir / "database_build_manifest.json"
        if db_manifest_path.exists():
            raise FileExistsError(f"Refusing to overwrite database build record: {db_manifest_path}")
        if kraken_db.exists() or any(Path(str(blast_prefix) + ext).exists() for ext in (".nhr", ".nin", ".nsq")):
            raise FileExistsError(f"Database output paths already exist; use a fresh C03 run: {ref_dir}")
        commands = [
            ["makeblastdb", "-in", ref_data["blast_reference_fasta"], "-dbtype", "nucl", "-parse_seqids", "-out", str(blast_prefix)],
            ["kraken2-build", "--add-to-library", ref_data["kraken_reference_fasta"], "--db", str(kraken_db)],
            ["kraken2-build", "--build", "--db", str(kraken_db), "--threads", str(args.threads)],
        ]
        versions = {}
        db_manifest_path.write_text(json.dumps({"baseline_protocol": "c03_paired_v1", "state": "RUNNING",
                                                 "commands": commands, "reference_manifest": str(ref_manifest),
                                                 "is_smoke_test": bool(ref_data.get("is_smoke_test"))}, indent=2) + "\n", encoding="utf-8")
        try:
            kraken_db.mkdir(parents=True, exist_ok=True)
            (kraken_db / "taxonomy").mkdir(exist_ok=True)
            shutil.copy2(ref_data["nodes_dmp"], kraken_db / "taxonomy/nodes.dmp")
            shutil.copy2(ref_data["names_dmp"], kraken_db / "taxonomy/names.dmp")
            (kraken_db / "library").mkdir(exist_ok=True)
            for name, command in (("makeblastdb", ["makeblastdb", "-version"]), ("kraken2-build", ["kraken2-build", "--version"])):
                version_proc = subprocess.run(command, text=True, capture_output=True, check=False)
                if version_proc.returncode != 0:
                    raise RuntimeError(f"Version probe failed for {name}: {version_proc.stderr[-4000:]}")
                versions[name] = (version_proc.stdout or version_proc.stderr).strip()
            for command in commands:
                proc = subprocess.run(command, check=False, text=True, capture_output=True)
                if proc.returncode != 0:
                    raise RuntimeError(f"Database command failed ({proc.returncode}): {command}\n{proc.stderr[-4000:]}")
            expected_db_files = [Path(str(blast_prefix) + ext) for ext in (".nhr", ".nin", ".nsq")] + [kraken_db / name for name in ("hash.k2d", "opts.k2d", "taxo.k2d")]
            missing_outputs = [str(path) for path in expected_db_files if not path.is_file()]
            if missing_outputs:
                raise RuntimeError(f"Database builder returned success but outputs are missing: {missing_outputs}")
            output_hashes = {str(path.resolve()): sha256_file(path) for path in expected_db_files}
            state = "COMPLETE"
            error = None
        except Exception as exc:
            state, error = "FAILED", str(exc)
            output_hashes = {}
        db_manifest = {"baseline_protocol": "c03_paired_v1", "state": state, "commands": commands,
                       "tool_versions": versions, "reference_manifest": str(ref_manifest),
                       "reference_manifest_sha256": __import__("hashlib").sha256(ref_manifest.read_bytes()).hexdigest(),
                       "kraken_db": str(kraken_db), "blast_db_prefix": str(blast_prefix), "threads": args.threads,
                       "output_sha256": output_hashes, "is_smoke_test": bool(ref_data.get("is_smoke_test")), "error": error}
        db_manifest_path.write_text(json.dumps(db_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"Database build state: {state}\nManifest: {db_manifest_path}")
        return 0 if state == "COMPLETE" else 1
    if args.mode == "export" and not inventory["available"]:
        print(json.dumps(inventory, indent=2, sort_keys=True))
        return 2
    from viral_cnn.notebook_runtime import reserve_run_directory
    run_id = args.run_id or f"c03-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    run_dir = reserve_run_directory(args.project_root.resolve(), run_id, "c03-baseline-preparation")
    if args.mode == "export":
        remote_mode = os.environ.get("REMOTE_SERVER", "true").strip().lower() not in {"0", "false", "no", "off", "local"}
        smoke_mode = os.environ.get("SMOKE_TEST", "").strip().lower() in {"1", "true", "yes"} or not remote_mode
        status_path = run_dir / "preparation_state.json"
        status_path.write_text(json.dumps({"baseline_protocol": "c03_paired_v1", "state": "RUNNING", "is_smoke_test": smoke_mode}, indent=2) + "\n", encoding="utf-8")
        try:
            manifest = prepare_queries(train_path, test_path, run_dir / "prepared",
                                       seed=int(config.get("seed", 42)),
                                       per_species=int(config.get("per_species", 20)),
                                       class_order_path=(ROOT / config.get("class_order", "dataset_kraken/species_name_ids.csv")).resolve(),
                                       is_smoke_test=smoke_mode,
                                       tool_sequence_policy=config.get("tool_sequence_policy", "historical_padding_to_N_v1"))
            manifest.update({"run_id": run_id, "mode": "export", "config_path": str(config_path),
                             "config_sha256": __import__("hashlib").sha256(config_path.read_bytes()).hexdigest()})
            (run_dir / "prepared" / "preparation_manifest.json").write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            bind_query_manifests(run_dir / "prepared")
            status_path.write_text(json.dumps({"baseline_protocol": "c03_paired_v1", "state": "COMPLETE", "is_smoke_test": smoke_mode}, indent=2) + "\n", encoding="utf-8")
        except Exception as exc:
            status_path.write_text(json.dumps({"baseline_protocol": "c03_paired_v1", "state": "FAILED", "is_smoke_test": smoke_mode, "error": str(exc)}, indent=2) + "\n", encoding="utf-8")
            raise
    else:
        tools = config.get("tools", {})
        tool_status = {name: shutil.which(name) for name in ("kraken2", "kraken2-build", "blastn", "makeblastdb")}
        model_config = yaml.safe_load((ROOT / "configs/models.yaml").read_text(encoding="utf-8"))["models"]
        prepared = "$PREPARED_DIR"
        comparison = f"{prepared}/comparison_results"
        commands = [
            {"stage": "build_databases", "command": ["python3", "scripts/prepare_baseline_benchmark.py", "--mode", "build-databases", "--prepared-dir", prepared]},
        ]
        for rate in (0.0, 0.02, 0.05, 0.10, 0.15):
            tag = f"noise_{round(rate * 100):02d}"
            query_csv, query_fasta = f"{prepared}/queries_{tag}.csv", f"{prepared}/queries_{tag}.fasta"
            qmanifest = f"{prepared}/query_manifest_{tag}.json"
            for method in ("Kraken2-flat", "BLASTN"):
                commands.append({"stage": "classify", "method": method, "noise_rate": rate,
                                 "command": ["python3", "scripts/run_c03_tool_baseline.py", "--method", method,
                                             "--queries", query_csv, "--fasta", query_fasta, "--query-manifest", qmanifest,
                                             "--reference-manifest", f"{prepared}/references/reference_manifest.json",
                                             "--database", f"{prepared}/references/{'kraken_db' if method == 'Kraken2-flat' else 'blast_db/train'}",
                                             "--output-dir", comparison]})
            for variant in ("resnet_clean_species", "resnet_noise_species", "convnext_zero_species", "convnext_transfer_species", "convnext_noise_species"):
                commands.append({"stage": "cnn_inference", "method": variant, "noise_rate": rate,
                                 "command": ["python3", "scripts/run_c03_cnn_inference.py", "--variant", variant,
                                             "--queries", query_csv, "--query-manifest", qmanifest,
                                             "--checkpoint", str((data_root / model_config[variant]["checkpoint_rel_path"]).resolve()),
                                             "--reference-manifest", f"{prepared}/preparation_manifest.json",
                                             "--output-dir", comparison]})
        checkpoint_status = {name: {"path": str((data_root / record["checkpoint_rel_path"]).resolve()),
                                    "exists": (data_root / record["checkpoint_rel_path"]).is_file()}
                             for name, record in model_config.items() if record.get("rank") == "species"}
        manifest = {"baseline_protocol": "c03_paired_v1", "mode": "plan", "commands": commands,
                    "tool_paths": tool_status, "tools_available": all(tool_status.values()),
                    "checkpoint_inventory": checkpoint_status,
                    "inputs": inventory["inputs"], "conditions": inventory["expected_conditions"],
                    "is_smoke_test": True,
                    "note": "No exports, database builds, classifications, or CNN inference were run."}
    manifest["run_id"] = run_id
    manifest["config_path"] = str(config_path)
    manifest["config_sha256"] = __import__("hashlib").sha256(config_path.read_bytes()).hexdigest()
    target = run_dir / "preparation_plan.json"
    target.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Run directory: {run_dir}\nManifest: {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
