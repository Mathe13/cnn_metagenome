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
import json
import logging
import os
import platform
import random
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
import yaml

from viral_cnn.models.registry import REGISTERED_VARIANTS

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


@dataclass
class RuntimeContext:
    data_root: Path
    seed: int
    device: torch.device
    is_smoke_test: bool
    topology: JupyterHubDataTopology
    library_versions: Dict[str, str] = field(default_factory=dict)


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
            },
            "long": {
                "ncbi_parquet_dir": str(self.root / "processed_ncbi_splited_parquet"),
                "mapping_species_final": str(self.root / "mapping_species_final.csv"),
                "mapping_genus_final": str(self.root / "mapping_genus_final.csv"),
                "raw_database_final": str(self.root / "raw_database" / "final.csv"),
                "synthetic_reads": str(self.root / "dataset_reads_sinteticos_CLEAN.parquet"),
                "taxonomy_recovered": str(self.root / "taxonomia_recuperada.csv"),
            },
            "checkpoints": {},
        }
        for name, spec in REGISTERED_VARIANTS.items():
            p["checkpoints"][name] = str(self.root / spec.checkpoint_rel_path)
        return p


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
    }
    try:
        import pyarrow
        versions["pyarrow"] = pyarrow.__version__
    except ImportError:
        pass
    try:
        import cudf
        if hasattr(cudf, "__version__"):
            versions["cudf"] = str(cudf.__version__)
    except Exception:
        pass
    return versions


def setup_notebook_runtime(
    data_root: Optional[Union[str, Path]] = None,
    seed: int = 42,
    smoke_test: Optional[bool] = None,
    auto_smoke_fallback: bool = True,
) -> RuntimeContext:
    """
    Initializes notebook execution runtime, seeds PRNGs, and determines environment.
    """
    seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Resolve DATA_ROOT
    if data_root is not None:
        resolved_root = Path(data_root).resolve()
    elif "DATA_ROOT" in os.environ:
        resolved_root = Path(os.environ["DATA_ROOT"]).resolve()
    else:
        # Auto-detect repository root by walking up looking for src/viral_cnn
        curr = Path.cwd().resolve()
        candidate = curr
        for r in [curr, curr.parent, curr.parent.parent, curr.parent.parent.parent, curr.parent.parent.parent.parent]:
            if (r / "src" / "viral_cnn").exists():
                candidate = r
                break
        resolved_root = candidate

    # Determine smoke test mode
    if smoke_test is not None:
        is_smoke = smoke_test
    elif os.environ.get("SMOKE_TEST", "0") in ("1", "true", "True"):
        is_smoke = True
    elif auto_smoke_fallback:
        # Check if actual remote data exists
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

    genus_dir = root / "cnn_kraken" / "processed_kraken_genus_splited_parquet"
    genus_dir.mkdir(parents=True, exist_ok=True)
    genus_test_file = genus_dir / "test.parquet"
    genus_labels = [i % 3 for i in range(num_samples_per_split)]
    pd.DataFrame({"sequence_padded": seqs, "label": genus_labels}).to_parquet(genus_test_file, index=False)

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
) -> NotebookRunManifest:
    """
    Constructs and serializes a NotebookRunManifest.
    """
    manifest = NotebookRunManifest(
        notebook_id=notebook_id,
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
        git_commit=get_git_commit(),
        seed=seed,
        data_root=str(Path(data_root).resolve()),
        is_smoke_test=is_smoke_test,
        device="cuda" if torch.cuda.is_available() else "cpu",
        library_versions=get_library_versions(),
        checkpoints=checkpoints or {},
        input_files=input_files or [],
        metrics_summary=metrics_summary or {},
    )

    data = asdict(manifest)

    if output_json_path:
        out_j = Path(output_json_path)
        out_j.parent.mkdir(parents=True, exist_ok=True)
        with open(out_j, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    if output_yaml_path:
        out_y = Path(output_yaml_path)
        out_y.parent.mkdir(parents=True, exist_ok=True)
        with open(out_y, "w", encoding="utf-8") as f:
            yaml.dump(data, f, default_flow_style=False)

    return manifest

