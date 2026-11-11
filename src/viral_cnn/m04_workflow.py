"""M04 coordination: first-layer motif perturbation lifecycle, explicit inputs and atomic manifests."""
from __future__ import annotations

import json
import os
import subprocess
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import yaml
from viral_cnn.notebook_runtime import (
    file_identity,
    get_git_commit,
    get_library_versions,
    reserve_run_directory,
    setup_notebook_runtime,
    ensure_smoke_test_artifacts,
    JupyterHubDataTopology,
)
from viral_cnn.run_planning import load_run_plan
from viral_cnn.utils.logging_utils import log_artifact_path

PROTOCOL_ID = "m04_motifs_audit_v1"
BIOLOGICAL_DEMARCATION_TEXT = (
    "COMPOSITIONAL BASE DETECTORS ONLY: First-layer 7-mers act as local GC/nucleotide composition detectors. "
    "They are not verified functional biological transcription factor binding sites or open reading frame codons."
)


def _atomic(path: Path, data: dict[str, Any]) -> None:
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(data, indent=2, allow_nan=False, default=str) + "\n")
    os.replace(temp, path)


def _resolved(value: str | Path, root: Path) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else root / path).resolve()


class M04Run:
    """Reserved run whose guard records Python failures without altering input state."""

    def __init__(
        self,
        output: Path,
        manifest: Path,
        configuration: dict[str, Any],
        run_id: str,
        parent_run_id: str | None,
        stage: str,
    ) -> None:
        self.output_dir = output
        self.manifest_path = manifest
        self.configuration = configuration
        smoke = configuration["is_smoke_test"]
        data = output / "fixtures" if smoke else Path(configuration["data_root"])
        self.ctx = setup_notebook_runtime(data_root=data, smoke_test=smoke, auto_smoke_fallback=False)
        self.ctx.data_root = Path(data).resolve()
        self.ctx.topology = JupyterHubDataTopology(self.ctx.data_root)
        self.run_id = run_id
        try:
            dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], stderr=subprocess.DEVNULL).strip())
        except (OSError, subprocess.CalledProcessError):
            dirty = None
        self.manifest = {
            "protocol_id": PROTOCOL_ID,
            "status": "RUNNING",
            "notebook_id": stage,
            "run_id": run_id,
            "parent_run_id": parent_run_id,
            "is_smoke_test": smoke,
            "population_id": configuration.get("population_id", "unknown"),
            "split_id": configuration.get("split_id", "unknown"),
            "seed": 42,
            "git_commit": get_git_commit(),
            "git_dirty": dirty,
            "configuration": configuration,
            "library_versions": get_library_versions(),
            "inputs": [],
            "checkpoints": {},
            "outputs": [],
            "scientific_reproduction_certified": False,
            "biological_demarcation": BIOLOGICAL_DEMARCATION_TEXT,
            "evidence": {"status": "synthetic" if smoke else "observed"},
        }
        _atomic(manifest, self.manifest)

    def prepare_smoke(self) -> None:
        """Create synthetic test fixtures under this run's isolated directory."""
        if not self.ctx.is_smoke_test:
            raise ValueError("Cannot generate synthetic smoke fixtures in empirical mode (SMOKE_TEST=0)")
        ensure_smoke_test_artifacts(self.ctx.data_root, num_samples_per_split=32)

    @contextmanager
    def guard(self) -> Iterator[None]:
        """Record FAILED for exceptions in notebook execution."""
        try:
            yield
        except BaseException as exc:
            self.manifest.update(status="FAILED", error=str(exc))
            _atomic(self.manifest_path, self.manifest)
            raise

    def complete(
        self,
        *,
        input_files: list[str | Path] | None = None,
        checkpoints: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """Hash real inputs and products, then atomically publish COMPLETE."""
        if self.manifest["status"] != "RUNNING":
            raise RuntimeError("Run is not in RUNNING state")
        records = []
        for value in input_files or []:
            path = Path(value).resolve()
            log_artifact_path("M04 input", path)
            if not path.exists():
                raise FileNotFoundError(path)
            for p in sorted(path.rglob("*")) if path.is_dir() else [path]:
                if p.is_file():
                    records.append(file_identity(p))
        ck = {}
        for variant, value in (checkpoints or {}).items():
            p = Path(value).resolve()
            log_artifact_path("M04 checkpoint", p)
            if not p.is_file():
                raise FileNotFoundError(p)
            ck[variant] = file_identity(p)
        products = [
            {**file_identity(p), "relative_path": str(p.relative_to(self.output_dir))}
            for p in sorted(self.output_dir.rglob("*"))
            if p.is_file()
            and p != self.manifest_path
            and not p.name.startswith(".m04")
            and not p.is_relative_to(self.output_dir / "fixtures")
        ]
        self.manifest.update(inputs=records, checkpoints=ck, outputs=products, status="COMPLETE")
        _atomic(self.manifest_path, self.manifest)
        return self.manifest


def begin_m04(
    notebook_id: str = "short/07",
    *,
    project_root: str | Path | None = None,
    settings: dict[str, Any] | None = None,
    run_plan: str | Path | None = None,
) -> M04Run:
    """Resolve explicit environment, configuration or M00 plan inputs and reserve destination."""
    if notebook_id != "short/07":
        raise ValueError(f"Unknown M04 stage: {notebook_id!r}; must be 'short/07'")
    root = Path(project_root or os.getenv("M04_PROJECT_ROOT", Path(__file__).resolve().parents[2])).resolve()
    config = {}
    if os.getenv("M04_CONFIG"):
        p = Path(os.environ["M04_CONFIG"])
        log_artifact_path("M04 config", p)
        config = yaml.safe_load(p.read_text()) or {}
        config["config_identity"] = file_identity(p)
    config.update(settings or {})

    source_plan = run_plan or os.getenv("RUN_PLAN")
    plan = load_run_plan(source_plan) if source_plan else None
    stage = None
    if plan:
        if plan.get("protocol_id") != "m00_run_plan_v1":
            raise ValueError("Expected M00 plan format (protocol_id: m00_run_plan_v1)")
        matches = [s for s in plan.get("stages", []) if s["notebook_id"] == notebook_id]
        if len(matches) != 1 or matches[0].get("action") != "run":
            raise ValueError(f"Stage {notebook_id} must be selected once with action 'run' in M00 plan")
        stage = matches[0]
        if plan.get("seed", 42) != 42:
            raise ValueError("M00 seed conflict: M04 preserves seed 42")
        pinned = {
            **stage.get("m04", {}),
            "data_root": plan["data_root"],
            "is_smoke_test": plan["is_smoke_test"],
            "population_id": plan.get("population_id", "unknown"),
            "split_id": plan.get("split_id", "unknown"),
        }
        for key, env in [("data_root", "DATA_ROOT")]:
            val = os.getenv(env, config.get(key))
            if val is not None and (key not in pinned or _resolved(val, root) != _resolved(pinned[key], root)):
                raise ValueError(f"M00 input conflict for {key}: env={val} != plan={pinned.get(key)}")
        if os.getenv("SMOKE_TEST") is not None:
            env_smoke = os.environ["SMOKE_TEST"].lower() in ("1", "true", "yes")
            if env_smoke != pinned["is_smoke_test"]:
                raise ValueError("M00 smoke mode conflict with SMOKE_TEST environment variable")
        pinned["requirements"] = stage.get("requirements", [])
        config = pinned
        config["run_plan_identity"] = file_identity(source_plan)
    else:
        ctx = setup_notebook_runtime(data_root=config.get("data_root"), auto_smoke_fallback=False)
        config["data_root"] = str(_resolved(os.getenv("DATA_ROOT", config.get("data_root", ctx.data_root)), root))
        config.setdefault("is_smoke_test", ctx.is_smoke_test)

    if config.get("seed", 42) != 42:
        raise ValueError("M04 preserves seed 42")

    if stage:
        run_id = stage["run_id"]
        if (
            not run_id
            or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_." for c in run_id)
            or run_id in (".", "..")
        ):
            raise ValueError(f"Invalid M00 run_id: {run_id!r}")
        parent = stage["parent_run_id"]
        output = Path(stage["output_dir"]).resolve()
        expected = (root / "results_reproduction/runs" / run_id / notebook_id).resolve()
        manifest = Path(stage["manifest"]).resolve()
        if output != expected or manifest.parent != output or parent != plan["parent_run_id"]:
            raise ValueError(f"Invalid M00 namespace: output={output}, expected={expected}")
        if not output.is_dir() or any(output.iterdir()):
            raise FileExistsError(f"Destination directory already contains files: {output}")
        with (output / ".m04_claim").open("x") as f:
            f.write(run_id)
    else:
        run_id = (
            config.get("run_id")
            or os.getenv("M04_RUN_ID")
            or os.getenv("MOTIFS_RUN_ID")
            or ("m04-" + uuid.uuid4().hex[:16])
        )
        output = reserve_run_directory(root, run_id, notebook_id)
        parent = None
        manifest = output / "manifest.json"

    return M04Run(output, manifest, config, run_id, parent, notebook_id)
