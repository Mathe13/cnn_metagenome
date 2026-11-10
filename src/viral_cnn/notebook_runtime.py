"""
Notebook runtime manager, remote JupyterHub topology resolution, and smoke-test synthesis.

Scientific and engineering invariants (AD-005, AD-006):
- Remote JupyterHub data and checkpoint topology resolved via DATA_ROOT without hardcoded personal paths.
- Deterministic random seeding (seed=42) across Python, NumPy, PyTorch CPU, and PyTorch CUDA.
- Transparent fallback to synthetic deterministic partitions (SMOKE_TEST=1) for structural testing.
- Reproducible execution manifests serializable to JSON and YAML.
"""

from __future__ import annotations

import dataclasses
import gc
import json
import logging
import os
import platform
import hashlib
import heapq
import threading
import random
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd
try:
    import pyarrow.parquet as pq
except ImportError:  # Lightweight inventory and run reservation do not need Parquet support.
    pq = None
import torch
import yaml

from viral_cnn.models.registry import REGISTERED_VARIANTS
from viral_cnn.utils.logging_utils import log_artifact_path

logger = logging.getLogger(__name__)


@dataclass
class NotebookRunManifest:
    notebook_id: str
    timestamp_utc: str
    git_commit: str
    seed: int
    data_root: str
    is_smoke_test: bool
    device: str
    library_versions: Dict[str, str] = field(default_factory=dict)
    checkpoints: Dict[str, str] = field(default_factory=dict)
    input_files: List[str] = field(default_factory=list)
    metrics_summary: Dict[str, Any] = field(default_factory=dict)
    schema_version: int = 1
    run_id: str = "legacy"
    configuration: Dict[str, Any] = field(default_factory=dict)
    evidence: Dict[str, Any] = field(default_factory=dict)
    outputs: List[Dict[str, Any]] = field(default_factory=list)
    inputs: List[Dict[str, Any]] = field(default_factory=list)
    git_dirty: Optional[bool] = None


@dataclass
class RuntimeContext:
    data_root: Path
    seed: int
    device: torch.device
    is_smoke_test: bool
    topology: JupyterHubDataTopology
    library_versions: Dict[str, str] = field(default_factory=dict)
    run_id: Optional[str] = None
    output_dir: Optional[Path] = None


class JupyterHubDataTopology:
    """Manages canonical paths for short and long sequence datasets and checkpoints."""

    def __init__(self, data_root: Union[str, Path]):
        self.root = Path(data_root).resolve()
        self.paths: Dict[str, Any] = self._build_paths()

    def _build_paths(self) -> Dict[str, Any]:
        p: Dict[str, Any] = {
            "root": str(self.root),
            "short": {
                "species_parquet_dir": str(self.root / "cnn_kraken" / "processed_kraken_species_splited_parquet"),
                "genus_parquet_dir": str(self.root / "cnn_kraken" / "processed_kraken_genus_splited_parquet"),
                "species_csv": str(self.root / "dataset_kraken" / "species_name_ids.csv"),
                "genus_csv": str(self.root / "dataset_kraken" / "genus_name_ids.csv"),
                "kraken_blast_dir": str(self.root / "cnn_kraken" / "benchmark_kraken_blast_20_per_species"),
                "ictv_master": str(self.root / "short" / "ICTV_Master_Species_List_2025_MSL41.v1.xlsx"),
                "hybrid_taxonomy_parquet": str(self.resolve_short_taxonomy_path()),
            },
            "long": {
                "ncbi_parquet_dir": str(
                    self.resolve_long_artifact_path("processed_ncbi_splited_parquet")
                    or (self.root / "processed_ncbi_splited_parquet")
                ),
                "mapping_species_final": str(
                    self.resolve_long_artifact_path("mapping_species_final.csv")
                    or (self.root / "mapping_species_final.csv")
                ),
                "mapping_genus_final": str(
                    self.resolve_long_artifact_path("mapping_genus_final.csv")
                    or (self.root / "mapping_genus_final.csv")
                ),
                "raw_database_final": str(
                    self.resolve_long_artifact_path("raw_database/final.csv")
                    or self.resolve_long_artifact_path("final.csv")
                    or (self.root / "raw_database" / "final.csv")
                ),
                "synthetic_reads": str(
                    self.resolve_long_artifact_path("dataset_reads_sinteticos_CLEAN.parquet")
                    or (self.root / "dataset_reads_sinteticos_CLEAN.parquet")
                ),
                "taxonomy_recovered": str(
                    self.resolve_long_artifact_path("taxonomia_recuperada.csv")
                    or (self.root / "taxonomia_recuperada.csv")
                ),
            },
            "checkpoints": {},
        }
        for name, spec in REGISTERED_VARIANTS.items():
            resolved = self.resolve_checkpoint_path(name)
            p["checkpoints"][name] = str(resolved if resolved is not None else self.root / spec.checkpoint_rel_path)
        return p

    def resolve_short_taxonomy_path(self) -> Path:
        """Find the ICTV mapping, preferring curated data over the original hybrid.

        ``TAXONOMY_PATH`` can name a Parquet file or a directory containing it.
        An unresolved path is returned for the notebook's explicit missing-input
        error and for the metadata-only preflight.
        """
        filenames = (
            "main_taxonomy_hybrid_CLEANSED.parquet",
            "main_taxonomy_hybrid_ictv_ncbi.parquet",
        )
        configured = os.getenv("TAXONOMY_PATH")
        if configured:
            configured_path = Path(configured).expanduser()
            if configured_path.is_dir():
                return next(
                    (path for name in filenames if (path := configured_path / name).is_file()),
                    configured_path / filenames[0],
                )
            return configured_path

        project_root = Path(__file__).resolve().parents[2]
        directories = (
            self.root / "cnn_kraken" / "data_taxonomy_parquet",
            self.root / "short",
            project_root / "results_reproduction" / "taxonomy",
            project_root / "short",
        )
        return next(
            (path for name in filenames for directory in directories
             if (path := directory / name).is_file()),
            self.root / "short" / filenames[0],
        )

    def resolve_long_artifact_path(self, artifact_name: str) -> Optional[Path]:
        """
        Resolves existing path for a long sequence artifact, checking
        self.root, subfolders long/, cnn_long/, cnn_kraken/, and project root.
        """
        candidates = [
            self.root / artifact_name,
            self.root / "long" / artifact_name,
            self.root / "cnn_long" / artifact_name,
            self.root / "cnn_kraken" / artifact_name,
            self.root.parent / "long" / artifact_name,
            self.root.parent / "cnn_long" / artifact_name,
            self.root.parent / "cnn_kraken" / artifact_name,
            self.root.parent / artifact_name,
            Path.cwd() / "long" / artifact_name,
            Path.cwd() / artifact_name,
        ]
        for c in candidates:
            if c.exists():
                return c
        return None

    def resolve_checkpoint_path(self, variant_name: str) -> Optional[Path]:
        """
        Resolves existing checkpoint path for a registered model variant,
        checking canonical path and common directory variations.
        """
        spec = REGISTERED_VARIANTS.get(variant_name)
        if not spec:
            return None
        rel_p = Path(spec.checkpoint_rel_path)
        candidates = [
            self.root / rel_p,
            self.root / rel_p.parts[0] / rel_p.name,  # e.g. cnn_kraken_species/best_resnet... (without checkpoints/ subdir)
            self.root.parent / rel_p,
            self.root.parent / rel_p.parts[0] / rel_p.name,
            self.root / "cnn_kraken" / rel_p.name,
            self.root / rel_p.name,
        ]
        for c in candidates:
            if c.exists():
                return c
        return self.root / rel_p


def get_git_commit() -> str:
    """Retrieves current Git commit hash or fallback."""
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode("ascii").strip()
        return commit
    except Exception:
        return "git_commit_unavailable"


def seed_everything(seed: int = 42) -> None:
    """Deterministically seeds Python, NumPy, PyTorch CPU, and PyTorch CUDA."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_library_versions() -> Dict[str, str]:
    """Collects installed versions of key scientific libraries."""
    versions = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "cuda": str(torch.version.cuda) if torch.version.cuda else "unavailable",
        "dask": "unavailable",
        "cudf": "unavailable",
    }
    try:
        import pyarrow
        versions["pyarrow"] = pyarrow.__version__
    except ImportError:
        versions["pyarrow"] = "unavailable"
    try:
        import dask
        versions["dask"] = str(dask.__version__)
    except ImportError:
        pass
    try:
        import cudf
        if hasattr(cudf, "__version__"):
            versions["cudf"] = str(cudf.__version__)
    except Exception:
        pass
    return versions


NOTEBOOK_REQUIREMENTS: Dict[str, List[Dict[str, Any]]] = {
    "short/01": [{"path": "short/species_parquet_dir", "kind": "parquet", "columns": ["sequence_padded", "label"]}, {"path": "checkpoints/resnet_clean_species", "kind": "file"}],
    "short/02": [{"path": "short/species_parquet_dir", "kind": "parquet", "columns": ["sequence_padded", "label"]}],
    "short/03": [{"path": "short/species_parquet_dir", "kind": "parquet", "columns": ["sequence_padded", "label"]}, {"path": "checkpoints/resnet_clean_species", "kind": "file"}],
    "short/04": [{"path": "short/species_parquet_dir", "kind": "parquet", "columns": ["sequence_padded", "label"]}, {"path": "checkpoints/convnext_transfer_species", "kind": "file"}],
    "short/05": [{"path": "short/ictv_master", "kind": "file"}, {"path": "short/species_csv", "kind": "file"}],
    "short/06": [{"path": "short/hybrid_taxonomy_parquet", "kind": "parquet"}, {"path": "checkpoints/resnet_clean_species", "kind": "file"}],
    "short/07": [{"path": "checkpoints/resnet_clean_species", "kind": "file"}],
    "short/08": [{"path": "short/species_parquet_dir", "kind": "parquet", "columns": ["sequence_padded"]}, {"path": "results_reproduction/motifs/manifest_07_motifs.json", "kind": "file", "external": True}],
    "short/09": [{"path": "short/kraken_blast_dir", "kind": "directory"}, {"path": "short/species_parquet_dir", "kind": "parquet", "columns": ["sequence_padded", "label"]}, {"path": "checkpoints/resnet_clean_species", "kind": "file"}],
    "short/10": [{"path": "short/species_parquet_dir", "kind": "parquet", "columns": ["sequence_padded", "label"]}],
    "short/11": [{"path": "short/ictv_master", "kind": "file"}, {"path": "short/hybrid_taxonomy_parquet", "kind": "parquet"}],
    "long/01": [{"path": "long/ncbi_parquet_dir", "kind": "parquet", "columns": ["sequence"]}],
    "long/02": [{"path": "long/ncbi_parquet_dir", "kind": "parquet", "columns": ["sequence"]}, {"path": "long/mapping_species_final", "kind": "file"}, *[{"path": f"checkpoints/{v}", "kind": "file"} for v in ("resnet_clean_species", "resnet_noise_species", "convnext_zero_species", "convnext_transfer_species", "convnext_noise_species")]],
    "long/03": [
        {"path": "results_reproduction/long_voter/manifest_02_voter_inference.json", "kind": "file", "external": True},
    ],
    "long/04": [{"path": "long/ncbi_parquet_dir", "kind": "parquet", "columns": ["sequence"]}],
}


def preflight_latent_inputs(
    split_files: Dict[str, Sequence[Union[str, Path]]],
    taxonomy: pd.DataFrame,
    checkpoint_paths: Dict[str, Union[str, Path]],
    required_variants: Sequence[str],
) -> Dict[str, Any]:
    """Fail before latent metrics if any empirical split, taxonomy label, or checkpoint is absent."""
    required_splits = ("train", "validation", "test")
    for split in required_splits:
        files = [Path(path).expanduser().resolve() for path in split_files.get(split, [])]
        if not files:
            raise FileNotFoundError(f"Required nonempty {split} split has no parquet partitions")
        missing = [path for path in files if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"Missing {split} partition: {missing[0]}")
        if len({str(path) for path in files}) != len(files):
            raise ValueError(f"Duplicate partition path in {split} split")
    split_paths = {split: {str(Path(p).expanduser().resolve()) for p in split_files[split]} for split in required_splits}
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        overlap = split_paths[left].intersection(split_paths[right])
        if overlap:
            raise ValueError(f"Partition alias between {left} and {right}: {sorted(overlap)[0]}")
    needed_columns = {"model_label", "Genus", "Order"}
    absent = needed_columns - set(taxonomy.columns)
    if absent:
        raise ValueError(f"Taxonomy missing required columns: {sorted(absent)}")
    if taxonomy.empty or taxonomy["model_label"].isna().any():
        raise ValueError("Taxonomy is empty or contains missing model_label values")
    for rank in ("Genus", "Order"):
        if taxonomy[rank].isna().any():
            first_label = int(taxonomy.loc[taxonomy[rank].isna(), "model_label"].iloc[0])
            raise ValueError(f"Taxonomy has missing {rank} mapping for model label {first_label}")
    if taxonomy["model_label"].duplicated().any():
        raise ValueError("Taxonomy contains duplicate model_label values")
    for name in required_variants:
        path = Path(checkpoint_paths.get(name, "")).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"Required checkpoint for '{name}' not found at: {path}")
    return {"status": "READY", "splits": {k: [str(Path(p).resolve()) for p in split_files[k]] for k in required_splits},
            "taxonomy_labels": int(taxonomy.model_label.nunique()), "variants": list(required_variants)}


def _resolve_requirement(topology: JupyterHubDataTopology, key: str) -> Path:
    """Resolve a dotted requirement key from the canonical topology."""
    if key.startswith("results_reproduction/"):
        project_root = Path(__file__).resolve().parents[2]
        return (project_root / key).resolve()
    if key.startswith("long/") and key.count("/") == 1:
        resolved = topology.resolve_long_artifact_path(key.split("/", 1)[1])
        return (resolved or (topology.root / key.split("/", 1)[1])).resolve()
    parts = key.split("/")
    node: Any = topology.paths
    for part in parts:
        node = node[part]
    return Path(node).expanduser().resolve()


def preflight_notebook(notebook_id: str, data_root: Union[str, Path], *, smoke_test: bool = False) -> Dict[str, Any]:
    """Inspect declared notebook prerequisites using only filesystem metadata."""
    if notebook_id not in NOTEBOOK_REQUIREMENTS and notebook_id != "short/06b":
        raise ValueError(f"Unknown notebook_id {notebook_id!r}; choose one of {sorted(NOTEBOOK_REQUIREMENTS)}")
    if notebook_id == "short/09" and not smoke_test:
        from viral_cnn.baselines.overlap import inspect_c03_package, resolve_c03_directory
        topology = JupyterHubDataTopology(data_root)
        source = os.getenv("C03_PREPARED_DIR", str(topology.paths["short"]["kraken_blast_dir"]))
        return {"notebook_id": notebook_id, "is_smoke_test": False, **inspect_c03_package(resolve_c03_directory(source, data_root))}
    if notebook_id == "short/06b":
        source_str = os.getenv("LATENT_SOURCE_DIR")
        source = Path(source_str).resolve() if source_str else None
        ok = False
        detail = "missing LATENT_SOURCE_DIR"
        if source and source.is_dir():
            has_umap = (source / "umap_comparison.json").is_file()
            has_manifest = (source / "manifest.json").is_file()
            has_embeddings = bool(list(source.glob("*.pt")))
            if has_umap or (has_manifest and has_embeddings):
                ok = True
                detail = f"valid latent source at {source}"
            else:
                detail = f"latent source {source} missing umap_comparison.json or embeddings"
        elif source:
            detail = f"latent source does not exist: {source}"
        elif smoke_test:
            ok = True
            detail = "smoke test mode for short/06b"
        return {
            "notebook_id": notebook_id,
            "status": "READY" if ok else "BLOCKED",
            "is_smoke_test": smoke_test,
            "scientific_reproduction_certified": False,
            "requirements": [{"requirement": "latent_source_dir", "path": str(source) if source else "", "status": "READY" if ok else "BLOCKED", "detail": detail, "external": True}],
        }
    topology = JupyterHubDataTopology(data_root)
    results = []
    for spec in NOTEBOOK_REQUIREMENTS[notebook_id]:
        path = _resolve_requirement(topology, spec["path"])
        ok = path.exists() and os.access(path, os.R_OK)
        detail = "readable" if ok else ("missing" if not path.exists() else "unreadable")
        if ok and spec["kind"] == "parquet":
            files = [path] if path.is_file() else sorted(path.glob("*.parquet"))
            if path.is_dir() and not files:
                files = sorted(path.glob("*/**/*.parquet"))
            if not files:
                ok, detail = False, "no parquet files"
            else:
                try:
                    if pq is None:
                        raise ImportError("pyarrow is required to inspect Parquet schema")
                    schema = pq.read_schema(files[0])
                    absent = sorted(set(spec.get("columns", [])) - set(schema.names))
                    if absent:
                        ok, detail = False, f"missing columns: {absent}"
                    else:
                        detail = f"parquet metadata OK; {len(files)} file(s); schema from {files[0]}"
                except Exception as exc:
                    ok, detail = False, f"parquet metadata unreadable: {exc}"
        results.append({"requirement": spec["path"], "path": str(path), "status": "READY" if ok else "BLOCKED", "detail": detail, "external": spec.get("external", False)})
    ready = all(x["status"] == "READY" for x in results)
    return {"notebook_id": notebook_id, "status": "READY" if ready else "BLOCKED", "is_smoke_test": smoke_test, "scientific_reproduction_certified": False, "requirements": results}


_reservation_lock = threading.Lock()


def reserve_run_directory(project_root: Union[str, Path], run_id: str, notebook_id: str) -> Path:
    """Atomically reserve a unique per-run notebook output directory."""
    if not run_id or run_id in (".", "..") or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_." for c in run_id):
        raise ValueError("run_id may contain only letters, digits, dot, underscore, and hyphen")
    step = Path(notebook_id)
    if not notebook_id or step.is_absolute() or any(p in (".", "..") for p in notebook_id.split("/")):
        raise ValueError("notebook_id must be a relative path without traversal")
    base = Path(project_root).resolve() / "results_reproduction" / "runs"
    target = base / run_id / step
    if not target.resolve().is_relative_to(base.resolve() / run_id):
        raise ValueError("run output escapes its reserved namespace")
    target.parent.mkdir(parents=True, exist_ok=True)
    with _reservation_lock:
        try:
            target.mkdir()
        except FileExistsError as exc:
            raise FileExistsError(f"Run output already reserved: {target}") from exc
    return target


def reserve_c02_long_run(project_root: Union[str, Path], notebook_id: str, run_id: Optional[str] = None) -> tuple[str, Path]:
    """Reserve a collision-free namespace for one C02 notebook execution."""
    import uuid
    resolved_id = run_id or os.getenv("C02_LONG_RUN_ID") or f"c02-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}-{uuid.uuid4().hex[:8]}"
    return resolved_id, reserve_run_directory(project_root, resolved_id, notebook_id)


def file_identity(path: Union[str, Path]) -> Dict[str, Any]:
    """Return observed path, size, and SHA-256 for an artifact file."""
    p = Path(path).resolve()
    if not p.is_file():
        return {"path": str(p), "sha256": "unknown", "size_bytes": "unknown", "quality": "unknown"}
    digest = hashlib.sha256()
    with p.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"path": str(p), "sha256": digest.hexdigest(), "size_bytes": p.stat().st_size, "quality": "observed"}


def empirical_manifest_gate(manifest: Union[NotebookRunManifest, Dict[str, Any]]) -> Dict[str, Any]:
    """Reject synthetic, legacy, or identity-incomplete runs as empirical evidence."""
    data = asdict(manifest) if dataclasses.is_dataclass(manifest) else manifest
    missing = []
    if data.get("is_smoke_test") is True: missing.append("is_smoke_test=true")
    for key in ("run_id", "notebook_id", "git_commit", "inputs", "outputs", "evidence"):
        if not data.get(key): missing.append(key)
    if data.get("schema_version") != 1: missing.append("schema_version=1")
    if data.get("evidence", {}).get("status") != "observed": missing.append("evidence.status=observed")
    return {"accepted": not missing, "rejected_fields": missing}


def is_remote_server() -> bool:
    """
    Checks if running in remote server environment mode (default: True).
    Returns False if either REMOTE_SERVER or remote_server is explicitly set to
    'false', '0', 'no', 'f', 'off', or 'local'.
    """
    raw_val = None
    for k in ("remote_server", "REMOTE_SERVER"):
        if k in os.environ:
            raw_val = os.environ[k]
            break
    if raw_val is not None:
        return raw_val.strip().lower() not in ("false", "0", "no", "f", "off", "local")
    return True


def setup_notebook_runtime(
    data_root: Optional[Union[str, Path]] = None,
    seed: int = 42,
    smoke_test: Optional[bool] = None,
    auto_smoke_fallback: bool = True,
) -> RuntimeContext:
    """
    Initializes notebook execution runtime, seeds PRNGs, and determines environment.
    Follows AD-017 for remote server vs local execution contracts.
    """
    seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass
    remote_mode = is_remote_server()

    # Detect repository project_root by walking up
    curr = Path.cwd().resolve()
    project_root = curr
    for r in [curr, curr.parent, curr.parent.parent, curr.parent.parent.parent, curr.parent.parent.parent.parent]:
        if (r / "src" / "viral_cnn").exists():
            project_root = r
            break

    # Resolve DATA_ROOT (AD-017)
    if data_root is not None:
        resolved_root = Path(data_root).resolve()
    elif "DATA_ROOT" in os.environ:
        resolved_root = Path(os.environ["DATA_ROOT"]).resolve()
    elif remote_mode:
        # Remote server topology: check parent workspace containing cnn_kraken/ or standard /home/mlima/workspace
        if (project_root.parent / "cnn_kraken").exists():
            resolved_root = project_root.parent.resolve()
        elif Path("/home/mlima/workspace").exists():
            resolved_root = Path("/home/mlima/workspace").resolve()
        else:
            resolved_root = project_root
    else:
        resolved_root = project_root

    # Auto-climb: If resolved_root doesn't contain cnn_kraken, but its parent workspace does, adopt parent
    if not (resolved_root / "cnn_kraken").exists() and (resolved_root.parent / "cnn_kraken").exists():
        resolved_root = resolved_root.parent.resolve()

    # Determine smoke test mode
    if smoke_test is not None:
        is_smoke = smoke_test
    elif "SMOKE_TEST" in os.environ:
        is_smoke = os.environ["SMOKE_TEST"].strip().lower() in ("1", "true", "yes")
    elif not remote_mode:
        # Local development / test mode defaults to lightweight smoke test
        is_smoke = True
    elif auto_smoke_fallback:
        # Remote mode: verify that species dataset exists at resolved_root
        topology = JupyterHubDataTopology(resolved_root)
        species_dir = Path(topology.paths["short"]["species_parquet_dir"])
        is_smoke = not species_dir.exists()
    else:
        is_smoke = False

    topology = JupyterHubDataTopology(resolved_root)
    lib_versions = get_library_versions()

    return RuntimeContext(
        data_root=resolved_root,
        seed=seed,
        device=device,
        is_smoke_test=is_smoke,
        topology=topology,
        library_versions=lib_versions,
    )


def ensure_smoke_test_artifacts(
    data_root: Union[str, Path],
    num_samples_per_split: int = 16,
) -> Dict[str, Any]:
    """
    Generates synthetic deterministic Parquet and CSV artifacts for testing.
    """
    root = Path(data_root).resolve()
    short_dir = root / "cnn_kraken" / "processed_kraken_species_splited_parquet"
    long_dir = root / "processed_ncbi_splited_parquet"
    tax_dir = root / "dataset_kraken"

    short_dir.mkdir(parents=True, exist_ok=True)
    long_dir.mkdir(parents=True, exist_ok=True)
    tax_dir.mkdir(parents=True, exist_ok=True)

    bases = ["A", "C", "G", "T"]
    random.seed(42)

    # 1. Short partitions (sequence_padded, label)
    short_test_file = short_dir / "test.parquet"
    seqs = ["".join(random.choices(bases, k=512)) for _ in range(num_samples_per_split)]
    labels = [i % 5 for i in range(num_samples_per_split)]
    pd.DataFrame({"sequence_padded": seqs, "label": labels}).to_parquet(short_test_file, index=False)

    short_train_file = short_dir / "train.parquet"
    pd.DataFrame({"sequence_padded": seqs, "label": labels}).to_parquet(short_train_file, index=False)

    short_val_file = short_dir / "val.parquet"
    pd.DataFrame({"sequence_padded": seqs, "label": labels}).to_parquet(short_val_file, index=False)

    genus_dir = root / "cnn_kraken" / "processed_kraken_genus_splited_parquet"
    genus_dir.mkdir(parents=True, exist_ok=True)
    genus_test_file = genus_dir / "test.parquet"
    genus_labels = [i % 3 for i in range(num_samples_per_split)]
    pd.DataFrame({"sequence_padded": seqs, "label": genus_labels}).to_parquet(genus_test_file, index=False)

    genus_train_file = genus_dir / "train.parquet"
    pd.DataFrame({"sequence_padded": seqs, "label": genus_labels}).to_parquet(genus_train_file, index=False)

    genus_val_file = genus_dir / "val.parquet"
    pd.DataFrame({"sequence_padded": seqs, "label": genus_labels}).to_parquet(genus_val_file, index=False)

    # 2. Long partitions (species_id, genus_id, origin_len, read_len, sequence)
    long_test_file = long_dir / "test.parquet"
    long_seqs = ["".join(random.choices(bases, k=1024)) for _ in range(num_samples_per_split)]
    pd.DataFrame({
        "species_id": [i % 5 + 1 for i in range(num_samples_per_split)],
        "genus_id": [1 for _ in range(num_samples_per_split)],
        "origin_len": [1024 for _ in range(num_samples_per_split)],
        "read_len": [1024 for _ in range(num_samples_per_split)],
        "sequence": long_seqs,
    }).to_parquet(long_test_file, index=False)

    # 3. Taxonomy CSVs (1-based ID) - only create synthetic fallback if not present
    tax_file = tax_dir / "species_name_ids.csv"
    if not tax_file.exists():
        pd.DataFrame({
            "species": [f"Taxon_{i+1}" for i in range(5)],
            "id": [i + 1 for i in range(5)],
        }).to_csv(tax_file, index=False)

    genus_tax_file = tax_dir / "genus_name_ids.csv"
    if not genus_tax_file.exists():
        pd.DataFrame({
            "genus": [f"Genus_{i+1}" for i in range(3)],
            "id": [i + 1 for i in range(3)],
        }).to_csv(genus_tax_file, index=False)

    return {
        "short": {
            "test_parquet": str(short_test_file),
            "genus_test_parquet": str(genus_test_file),
            "species_csv": str(tax_file),
            "genus_csv": str(genus_tax_file),
        },
        "long": {
            "test_parquet": str(long_test_file),
        },
    }


def write_notebook_manifest(
    notebook_id: str,
    data_root: Union[str, Path],
    is_smoke_test: bool,
    metrics_summary: Optional[Dict[str, Any]] = None,
    checkpoints: Optional[Dict[str, str]] = None,
    input_files: Optional[List[str]] = None,
    seed: int = 42,
    output_json_path: Optional[Union[str, Path]] = None,
    output_yaml_path: Optional[Union[str, Path]] = None,
    run_id: Optional[str] = None,
    configuration: Optional[Dict[str, Any]] = None,
    outputs: Optional[List[Union[str, Path]]] = None,
    command: Optional[str] = None,
) -> NotebookRunManifest:
    """
    Constructs and serializes a NotebookRunManifest.
    """
    output_records = [file_identity(p) for p in (outputs or [])]
    input_records = [file_identity(p) for p in (input_files or [])]
    checkpoint_records: Dict[str, Any] = {}
    for variant, value in (checkpoints or {}).items():
        path_value = value.get("path") if isinstance(value, dict) else value
        identity = file_identity(path_value) if path_value and Path(path_value).is_file() else {"path": str(path_value or "unknown"), "sha256": "unknown", "quality": "unknown"}
        spec = REGISTERED_VARIANTS.get(variant)
        checkpoint_records[variant] = {**identity, "rank": spec.rank if spec else "unknown", "variant": variant}
    try:
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], stderr=subprocess.DEVNULL).strip())
    except Exception:
        dirty = None
    manifest = NotebookRunManifest(
        notebook_id=notebook_id,
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
        git_commit=get_git_commit(),
        seed=seed,
        data_root=str(Path(data_root).resolve()),
        is_smoke_test=is_smoke_test,
        device="cuda" if torch.cuda.is_available() else "cpu",
        library_versions=get_library_versions(),
        checkpoints=checkpoint_records,
        input_files=input_files or [],
        metrics_summary=metrics_summary or {},
        run_id=run_id or os.getenv("RUN_ID", "legacy"),
        configuration={**(configuration or {}), "command": command or f"jupyter notebook {notebook_id}"},
        outputs=output_records,
        inputs=input_records,
        evidence={"status": "synthetic" if is_smoke_test else "observed", "inputs": input_records, "quality": "observed" if input_records else "unknown"},
        git_dirty=dirty,
    )

    data = asdict(manifest)

    if output_json_path:
        out_j = Path(output_json_path)
        out_j.parent.mkdir(parents=True, exist_ok=True)
        if out_j.exists():
            raise FileExistsError(f"Refusing to overwrite manifest: {out_j}")
        with open(out_j, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    if output_yaml_path:
        out_y = Path(output_yaml_path)
        out_y.parent.mkdir(parents=True, exist_ok=True)
        if out_y.exists():
            raise FileExistsError(f"Refusing to overwrite manifest: {out_y}")
        with open(out_y, "w", encoding="utf-8") as f:
            yaml.dump(data, f, default_flow_style=False)

    return manifest


def resolve_split_files(
    base_dir: Union[str, Path, Sequence[Union[str, Path]]],
    split_name: str = "test",
) -> List[str]:
    """
    Resolves Parquet partition files for a specific dataset split ('train', 'val'/'validation', 'test').
    Prevents accidental merging of train + validation + test subdirectories via rglob.
    """
    if not isinstance(base_dir, (str, Path)):
        return [str(p) for p in base_dir]

    base = Path(base_dir)
    if base.is_file():
        return [str(base)]

    split_key = split_name.strip().lower()
    if split_key in ("val", "validation", "valid"):
        candidates = ["validation", "val", "valid"]
    else:
        candidates = [split_key]

    # 1. Check split subdirectories (e.g. base/test/*.parquet, base/validation/*.parquet)
    for cand in candidates:
        sub = base / cand
        if sub.is_dir():
            files = sorted(sub.glob("*.parquet"))
            if not files:
                files = sorted(sub.rglob("*.parquet"))
            if files:
                return [str(f) for f in files]

    # 2. Check direct split files (e.g. base/test.parquet, base/val.parquet)
    for cand in candidates:
        f_cand = base / f"{cand}.parquet"
        if f_cand.exists():
            return [str(f_cand)]

    # 3. Check glob matching split name in base directory
    for cand in candidates:
        matched = sorted(base.glob(f"*{cand}*.parquet"))
        if matched:
            return [str(f) for f in matched]

    # 4. Fallback: direct *.parquet files in base (without descending into unrelated split dirs)
    direct = sorted(base.glob("*.parquet"))
    if direct:
        return [str(f) for f in direct]

    return [str(base)]


def read_parquet_sample(
    path_or_files: Union[str, Path, Sequence[Union[str, Path]]],
    columns: Optional[List[str]] = None,
    max_rows: Optional[int] = 1000,
    split: str = "test",
) -> pd.DataFrame:
    """
    Streams up to `max_rows` rows from a Parquet file, split directory, or list of Parquet files
    using PyArrow batch iteration without loading multi-gigabyte partitions into RAM.
    """
    files = resolve_split_files(path_or_files, split_name=split)
    log_artifact_path(
        "ParquetSample",
        files,
        extra=f"split={split}, max_rows={max_rows}",
    )

    chunks: List[pd.DataFrame] = []
    collected = 0
    batch_sz = 8192 if max_rows is None else max(1, min(max_rows, 8192))

    for fp in files:
        p = Path(fp)
        if not p.exists() or p.is_dir():
            continue
        if pq is None:
            raise ImportError("pyarrow is required to read Parquet samples")
        pf = pq.ParquetFile(p)
        schema_names = pf.schema_arrow.names
        valid_cols = None
        if columns is not None:
            valid_cols = [c for c in columns if c in schema_names]
            if not valid_cols:
                valid_cols = None

        for batch in pf.iter_batches(batch_size=batch_sz, columns=valid_cols):
            df_batch = batch.to_pandas()
            chunks.append(df_batch)
            collected += len(df_batch)
            if max_rows is not None and collected >= max_rows:
                break
        if max_rows is not None and collected >= max_rows:
            break

    if not chunks:
        return pd.DataFrame()
    out_df = pd.concat(chunks, ignore_index=True)
    if max_rows is not None and len(out_df) > max_rows:
        out_df = out_df.iloc[:max_rows].copy()
    return out_df


def read_parquet_hash_sample(
    path_or_files: Union[str, Path, Sequence[Union[str, Path]]],
    *,
    columns: Sequence[str],
    limit: Optional[int],
    seed: int = 42,
    split: str = "train",
    sequence_column: Optional[str] = None,
) -> tuple[pd.DataFrame, Dict[str, Any]]:
    """Stream a deterministic bottom-k SHA-256 sample from Parquet partitions."""
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive or None for a census")
    files = resolve_split_files(path_or_files, split_name=split)
    log_artifact_path("ParquetHashSample", files, extra=f"split={split}, limit={limit}, seed={seed}")
    records: list[tuple[int, str, Any]] = []
    heap: list[tuple[int, int, str, Any]] = []
    available = 0
    eligible = 0
    ordinal = 0
    common_dir = Path(os.path.commonpath([str(Path(name).resolve().parent) for name in files])) if files else Path.cwd()
    for file_name in files:
        parquet_path = Path(file_name)
        if pq is None:
            raise ImportError("pyarrow is required to sample Parquet files")
        pf = pq.ParquetFile(parquet_path)
        available += int(pf.metadata.num_rows)
        present = set(pf.schema_arrow.names)
        selected_columns = [c for c in columns if c in present]
        if not selected_columns:
            continue
        seq_col = sequence_column if sequence_column in selected_columns else selected_columns[0]
        relative = parquet_path.resolve().relative_to(common_dir).as_posix()
        for batch in pf.iter_batches(batch_size=8192, columns=selected_columns):
            values = batch.column(selected_columns.index(seq_col)).to_pylist()
            for value in values:
                row_id = f"{relative}#row={ordinal}"
                ordinal += 1
                if value is None:
                    continue
                eligible += 1
                score = int.from_bytes(hashlib.sha256(f"{seed}:{row_id}".encode()).digest(), "big")
                if limit is None:
                    records.append((score, row_id, value))
                elif limit > 0:
                    entry = (-score, -eligible, row_id, value)
                    if len(heap) < limit:
                        heapq.heappush(heap, entry)
                    elif score < -heap[0][0]:
                        heapq.heapreplace(heap, entry)
    if limit is not None:
        records = [(-neg_score, row_id, value) for neg_score, _, row_id, value in heap]
    records.sort(key=lambda row: (row[0], row[1]))
    result = pd.DataFrame({"query_id": [r[1] for r in records], "sequence": [r[2] for r in records]})
    metadata = {
        "kind": "census" if limit is None or len(records) == eligible else "sample",
        "available_count": int(available),
        "population_count": int(eligible),
        "excluded_count": int(available-eligible),
        "planned_count": limit,
        "realized_count": int(len(records)),
        "criterion": "lowest SHA-256(seed, partition filename, row ordinal) scores",
        "seed": int(seed),
        "selected_query_ids": [r[1] for r in records],
    }
    return result, metadata


def create_optimized_dataloader(
    dataset: Any,
    is_smoke_test: bool = False,
    device: Optional[Union[str, torch.device]] = None,
    num_workers: Optional[int] = None,
) -> torch.utils.data.DataLoader:
    """
    Builds a PyTorch DataLoader for pre-batched IterableDatasets (`batch_size=None`)
    tuned to maximize GPU and CPU pipeline throughput without exhausting system RAM.
    """
    use_cuda = (
        str(device).startswith("cuda")
        if device is not None
        else torch.cuda.is_available()
    )
    num_files = len(getattr(dataset, "file_paths", []))

    if num_workers is not None:
        workers = max(0, int(num_workers))
    elif is_smoke_test or not use_cuda or num_files <= 1:
        workers = 0
    else:
        cpu_cnt = os.cpu_count() or 4
        workers = min(8, max(1, cpu_cnt // 2), num_files)

    kwargs: Dict[str, Any] = {
        "batch_size": None,
        "num_workers": workers,
        "pin_memory": bool(use_cuda and not is_smoke_test),
    }
    if workers > 0:
        kwargs["prefetch_factor"] = 2
        kwargs["persistent_workers"] = True

    return torch.utils.data.DataLoader(dataset, **kwargs)


def free_gpu_memory(*objs: Any) -> None:
    """Releases Python references, triggers garbage collection, and clears PyTorch CUDA cache."""
    del objs
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


