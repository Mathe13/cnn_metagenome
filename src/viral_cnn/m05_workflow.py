"""M05 coordination: exploratory triplet composition, cooccurrence synergy lifecycle, explicit inputs and atomic manifests."""
from __future__ import annotations

import json
import math
import os
import subprocess
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, List, Optional

import numpy as np
import pandas as pd
import yaml

from viral_cnn.notebook_runtime import (
    file_identity,
    get_git_commit,
    get_library_versions,
    reserve_run_directory,
    setup_notebook_runtime,
    ensure_smoke_test_artifacts,
    read_parquet_sample,
    JupyterHubDataTopology,
)
from viral_cnn.run_planning import load_run_plan
from viral_cnn.utils.logging_utils import log_artifact_path

PROTOCOL_ID = "m05_triplets_audit_v1"
BIOLOGICAL_DEMARCATION_TEXT = (
    "EXPLORATORY TRIPLET COMPOSITION ONLY: DNA fragments lack verified biological open reading frames (ORFs). "
    "Triplet scores and RSCU profiles reflect local nucleotide/codon composition, not demonstrated translation, "
    "translation initiation (ATG) or termination (STOP) mechanisms, nor causal protein synthesis."
)


def _atomic(path: Path, data: dict[str, Any]) -> None:
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(data, indent=2, allow_nan=False, default=str) + "\n")
    os.replace(temp, path)


def _resolved(value: str | Path, root: Path) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else root / path).resolve()


class M05Run:
    """Reserved run whose guard records Python failures without altering input state."""

    def __init__(
        self,
        output: Path,
        manifest: Path,
        configuration: dict[str, Any],
        run_id: str,
        parent_run_id: str | None,
        stage: str,
        motifs_07_dir: Path | None = None,
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
        self.motifs_07_dir = motifs_07_dir
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
    ) -> dict[str, Any]:
        """Hash real inputs and products, then atomically publish COMPLETE."""
        if self.manifest["status"] != "RUNNING":
            raise RuntimeError("Run is not in RUNNING state")
        records = []
        for value in input_files or []:
            path = Path(value).resolve()
            log_artifact_path("M05 input", path)
            if not path.exists():
                raise FileNotFoundError(path)
            for p in sorted(path.rglob("*")) if path.is_dir() else [path]:
                if p.is_file():
                    records.append(file_identity(p))
        products = [
            {**file_identity(p), "relative_path": str(p.relative_to(self.output_dir))}
            for p in sorted(self.output_dir.rglob("*"))
            if p.is_file()
            and p != self.manifest_path
            and not p.name.startswith(".m05")
            and not p.is_relative_to(self.output_dir / "fixtures")
        ]
        self.manifest.update(inputs=records, outputs=products, status="COMPLETE")
        _atomic(self.manifest_path, self.manifest)
        return self.manifest


def sample_deterministic_test_sequences(
    parquet_path: str | Path | None,
    max_rows: int = 1000,
    seed: int = 42,
    is_smoke_test: bool = False,
) -> List[str]:
    """Sample test sequences deterministically under seed 42 with known count."""
    if is_smoke_test:
        bases = ["A", "C", "G", "T"]
        rng = np.random.default_rng(seed)
        seqs = []
        for _ in range(min(max_rows, 50)):
            seqs.append("".join(rng.choice(bases, size=512)))
        return seqs

    if parquet_path is None or not Path(parquet_path).exists():
        raise FileNotFoundError(f"Parquet de teste ausente: {parquet_path}")

    log_artifact_path("Test sequences sample", parquet_path, extra=f"max_rows={max_rows}, seed={seed}")
    df_test = read_parquet_sample(
        parquet_path,
        columns=["sequence_padded", "padded_Representation", "sequence"],
        max_rows=max_rows * 2,
        split="test",
    )
    col = next(
        (name for name in ("sequence_padded", "padded_Representation", "sequence") if name in df_test.columns),
        None,
    )
    if col is None:
        raise ValueError("A amostra de teste não contém coluna de sequência")
    all_seqs = df_test[col].dropna().tolist()
    if not all_seqs:
        raise ValueError("Nenhuma sequência disponível no split de teste")

    if len(all_seqs) <= max_rows:
        return all_seqs

    rng = np.random.default_rng(seed)
    chosen_indices = sorted(rng.choice(len(all_seqs), size=max_rows, replace=False))
    return [all_seqs[i] for i in chosen_indices]


def generate_network_edge_table(
    jaccard_df: pd.DataFrame,
    top_k: int = 200,
    seed: int = 42,
) -> pd.DataFrame:
    """Derive deterministic 2D network coordinates and edges satisfying C06 network_edges schema."""
    motifs = list(jaccard_df.index)
    N = len(motifs)
    edges = []
    for i in range(N):
        for j in range(i + 1, N):
            val = float(jaccard_df.iloc[i, j])
            if val > 0:
                edges.append({
                    "motif_a": motifs[i],
                    "motif_b": motifs[j],
                    "jaccard": val,
                })
    edges_df = pd.DataFrame(edges)
    if edges_df.empty:
        return pd.DataFrame(columns=["x1", "y1", "x2", "y2", "motif_a", "motif_b", "jaccard"])

    top_edges = edges_df.sort_values("jaccard", ascending=False, kind="stable").head(top_k).copy()
    unique_motifs = sorted(set(top_edges["motif_a"]).union(set(top_edges["motif_b"])))
    num_nodes = len(unique_motifs)

    # Deterministic circular layout coordinates in [-1, 1]
    node_coords = {}
    for idx, motif in enumerate(unique_motifs):
        angle = 2.0 * math.pi * idx / max(num_nodes, 1)
        node_coords[motif] = (round(math.cos(angle), 4), round(math.sin(angle), 4))

    top_edges["x1"] = top_edges["motif_a"].map(lambda m: node_coords[m][0])
    top_edges["y1"] = top_edges["motif_a"].map(lambda m: node_coords[m][1])
    top_edges["x2"] = top_edges["motif_b"].map(lambda m: node_coords[m][0])
    top_edges["y2"] = top_edges["motif_b"].map(lambda m: node_coords[m][1])

    return top_edges[["x1", "y1", "x2", "y2", "motif_a", "motif_b", "jaccard"]].reset_index(drop=True)


def render_figure_04_rscu(
    rscu_frame0: dict[str, Any],
    rscu_3f: dict[str, Any],
    rscu_6f: dict[str, Any],
    output_path: Path,
) -> Path:
    """Render Figure 4 RSCU profile comparing reading frames with horizontal unit reference."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 4))
    codons_subset = list(rscu_frame0["rscu"].keys())[:32]
    x = np.arange(len(codons_subset))
    ax.bar(x - 0.2, [rscu_frame0["rscu"][c] for c in codons_subset], width=0.4, label="Frame 0 (Histórico)", alpha=0.8)
    ax.bar(x + 0.2, [rscu_6f["rscu"][c] for c in codons_subset], width=0.4, label="6-Frames (Composicional)", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(codons_subset, rotation=90, fontsize=8)
    ax.set_ylabel("RSCU Exploratório")
    ax.set_title("Figura 4: Perfil de tripletos sem quadro de leitura biológico confirmado")
    ax.axhline(1.0, color="red", linestyle="--", alpha=0.5)
    ax.legend()
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close(fig)
    log_artifact_path("Figure 4", output_path)
    return output_path


def render_figure_05_network(
    edge_table_df: pd.DataFrame,
    output_path: Path,
) -> Path:
    """Render Figure 5 network layout from edge table with coordinates."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 7))
    if not edge_table_df.empty:
        for _, edge in edge_table_df.iterrows():
            ax.plot(
                [edge["x1"], edge["x2"]],
                [edge["y1"], edge["y2"]],
                color="gray",
                alpha=float(min(max(edge["jaccard"] * 1.5, 0.1), 0.8)),
                linewidth=float(max(edge["jaccard"] * 3.0, 0.5)),
            )
        all_x = list(edge_table_df["x1"]) + list(edge_table_df["x2"])
        all_y = list(edge_table_df["y1"]) + list(edge_table_df["y2"])
        ax.scatter(all_x, all_y, s=30, color="#1f77b4", zorder=3)
        ax.set_title(f"Figura 5: Rede de Coocorrência de Motivos (N={len(edge_table_df)} arestas)")
    else:
        ax.text(0.5, 0.5, "Nenhuma aresta de coocorrência positiva", ha="center", va="center")
        ax.set_title("Figura 5: Rede de Coocorrência (Sem arestas)")

    ax.axis("off")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close(fig)
    log_artifact_path("Figure 5 Network", output_path)
    return output_path


def render_figure_05_cooccurrence_synergy(
    cooccurrence_results: dict[str, Any],
    output_path: Path,
) -> Path:
    """Render bar chart of exploratory synergy metrics."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 5))
    if cooccurrence_results["status"] == "VALID":
        p_val = cooccurrence_results.get("p_perm", np.nan)
        p_str = f"{p_val:.4f}" if np.isfinite(p_val) else "N/A"
        ax.bar(
            ["r bidirecional", "ICC"],
            [cooccurrence_results["r_bidirectional"], cooccurrence_results["icc_symmetric"]],
            color=["forestgreen", "teal"],
            alpha=0.7,
        )
        ax.set_title(f"Assortatividade composicional exploratória (p = {p_str})")
    else:
        ax.text(0.5, 0.5, f"Não estimável: {cooccurrence_results['status']}", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Coocorrência de motivos: estimativa indisponível")
    ax.set_ylabel("Associação entre escores de motivos")
    ax.set_ylim(-1.0, 1.0)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close(fig)
    log_artifact_path("Figure 5 Synergy", output_path)
    return output_path


def render_figure_05_permutation_null(
    cooccurrence_results: dict[str, Any],
    output_path: Path,
) -> Path:
    """Render empirical permutation null distribution with marked observed statistic."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4))
    null_dist = cooccurrence_results.get("null_distribution", [])
    obs_r = cooccurrence_results.get("r_bidirectional", np.nan)
    p_val = cooccurrence_results.get("p_perm", np.nan)

    if null_dist and np.isfinite(obs_r):
        ax.hist(null_dist, bins=30, color="lightgray", edgecolor="black", alpha=0.7, density=True, label="Nulo de Permutação")
        ax.axvline(obs_r, color="red", linestyle="--", linewidth=2, label=f"Observado r={obs_r:.3f}")
        p_str = f"{p_val:.4f}" if np.isfinite(p_val) else "N/A"
        ax.set_title(f"Distribuição Nula de Permutação de Nós (p = {p_str})")
        ax.set_xlabel("Correlação Bidirecional Nula")
        ax.set_ylabel("Densidade")
        ax.legend()
    else:
        ax.text(0.5, 0.5, "Distribuição nula de permutação indisponível", ha="center", va="center")
        ax.set_title("Distribuição Nula (Indisponível)")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close(fig)
    log_artifact_path("Figure 5 Permutation Null", output_path)
    return output_path


def export_reconciled_table_11b_diagnostics(
    matrices: dict[str, pd.DataFrame],
    output_path: Path,
) -> pd.DataFrame:
    """Export descriptive comparison of Jaccard matrices across model variants."""
    pairs = [
        ("resnet50_species", "resnet_noise_species"),
        ("resnet50_species", "convnext_zero_species"),
        ("resnet50_species", "convnext_transfer_species"),
        ("resnet50_species", "convnext_noise_species"),
        ("convnext_transfer_species", "convnext_noise_species"),
    ]
    records = []
    for v1, v2 in pairs:
        row: dict[str, Any] = {
            "variant_first": v1,
            "variant_second": v2,
            "status": "NOT_AVAILABLE",
            "n_motifs_first": np.nan,
            "n_motifs_second": np.nan,
            "n_common_motifs": np.nan,
            "motif_set_jaccard": np.nan,
            "n_unique_pairs": np.nan,
            "n_positive_pairs_first": np.nan,
            "n_positive_pairs_second": np.nan,
            "off_diagonal_identical": pd.NA,
            "max_abs_difference": np.nan,
        }
        if v1 in matrices and v2 in matrices:
            a, b = matrices[v1], matrices[v2]
            names_a, names_b = set(a.index), set(b.index)
            common = sorted(names_a & names_b)
            union = names_a | names_b
            row.update(
                n_motifs_first=len(names_a),
                n_motifs_second=len(names_b),
                n_common_motifs=len(common),
                motif_set_jaccard=round(len(common) / len(union), 4) if union else np.nan,
                n_unique_pairs=len(common) * (len(common) - 1) // 2,
            )
            if len(common) < 2:
                row["status"] = "INSUFFICIENT_SHARED_MOTIFS"
            else:
                x = a.loc[common, common].to_numpy(dtype=float)
                y = b.loc[common, common].to_numpy(dtype=float)
                upper = np.triu_indices(len(common), k=1)
                x_vals, y_vals = x[upper], y[upper]
                row.update(
                    status="DESCRIPTIVE_SHARED_INPUT_CHECK",
                    n_positive_pairs_first=int(np.count_nonzero(x_vals)),
                    n_positive_pairs_second=int(np.count_nonzero(y_vals)),
                    off_diagonal_identical=bool(np.array_equal(x_vals, y_vals)),
                    max_abs_difference=float(np.max(np.abs(x_vals - y_vals))),
                )
        records.append(row)

    df_out = pd.DataFrame(records)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(output_path, index=False)
    log_artifact_path("Table 11b Diagnostics", output_path)
    return df_out


def begin_m05(
    notebook_id: str = "short/08",
    *,
    project_root: str | Path | None = None,
    settings: dict[str, Any] | None = None,
    run_plan: str | Path | None = None,
) -> M05Run:
    """Resolve explicit environment, configuration or M00 plan inputs and reserve destination."""
    if notebook_id != "short/08":
        raise ValueError(f"Unknown M05 stage: {notebook_id!r}; must be 'short/08'")
    root = Path(project_root or os.getenv("M05_PROJECT_ROOT", Path(__file__).resolve().parents[2])).resolve()
    config = {}
    if os.getenv("M05_CONFIG"):
        p = Path(os.environ["M05_CONFIG"])
        log_artifact_path("M05 config", p)
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
            raise ValueError("M00 seed conflict: M05 preserves seed 42")
        pinned = {
            **stage.get("m05", {}),
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
        raise ValueError("M05 preserves seed 42")

    # Resolve 07 motifs inputs directory
    motifs_07_env = os.getenv("M05_MOTIFS_07_DIR") or os.getenv("MOTIFS_07_DIR") or config.get("motifs_07_dir")
    motifs_07_dir = Path(motifs_07_env).resolve() if motifs_07_env else (root / "results_reproduction" / "motifs").resolve()

    if not config["is_smoke_test"]:
        manifest_07 = motifs_07_dir / "manifest.json"
        if not manifest_07.exists():
            manifest_07 = motifs_07_dir / "manifest_07_motifs.json"
        if not manifest_07.exists():
            raise FileNotFoundError(f"Missing 07 motifs manifest in empirical mode at: {motifs_07_dir}")

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
        with (output / ".m05_claim").open("x") as f:
            f.write(run_id)
    else:
        run_id = (
            config.get("run_id")
            or os.getenv("M05_RUN_ID")
            or os.getenv("TRIPLETS_RUN_ID")
            or ("m05-" + uuid.uuid4().hex[:16])
        )
        output = reserve_run_directory(root, run_id, notebook_id)
        parent = None
        manifest = output / "manifest.json"

    return M05Run(output, manifest, config, run_id, parent, notebook_id, motifs_07_dir)
