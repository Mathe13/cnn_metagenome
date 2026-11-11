"""M06 coordination: long sequence evaluation, strata analysis, and contig profiling."""
from __future__ import annotations

import json
import math
import os
import subprocess
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple, Union

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
from viral_cnn.inference.voter import slice_sequence, DEFAULT_WINDOW_SIZE, DEFAULT_STRIDE

PROTOCOL_ID = "m06_long_strata_audit_v1"
SCIENTIFIC_DEMARCATION_TEXT = (
    "DESCRIPTIVE LONG SEQUENCE EVALUATION ONLY: NCBI contigs evaluated without verified accession numbers have "
    "genome provenance unknown. Differences across length or window strata and ConvNeXt drop mechanisms "
    "constitute hypotheses and observational findings, not causal evidence of intrinsic architectural "
    "superiority without controlled multi-seed retraining."
)


def _clean_for_json(obj: Any) -> Any:
    if isinstance(obj, float):
        if math.isinf(obj):
            return "inf" if obj > 0 else "-inf"
        if math.isnan(obj):
            return None
        return obj
    if isinstance(obj, dict):
        return {k: _clean_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean_for_json(x) for x in obj]
    return obj


def _atomic(path: Path, data: dict[str, Any]) -> None:
    temp = path.with_suffix(".tmp")
    cleaned = _clean_for_json(data)
    temp.write_text(json.dumps(cleaned, indent=2, allow_nan=True, default=str) + "\n")
    os.replace(temp, path)


def _resolved(value: str | Path, root: Path) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else root / path).resolve()


def reserve_m06_run(
    project_root: Union[str, Path],
    stage: str,
    run_id: Optional[str] = None,
) -> Tuple[str, Path]:
    """Atomically reserve a collision-free run output directory under results_reproduction/runs/<run_id>/long/<stage>/."""
    notebook_id = f"long/{stage}" if not stage.startswith("long/") else stage
    resolved_id = run_id or os.getenv("M06_LONG_RUN_ID") or os.getenv("C02_LONG_RUN_ID")
    if not resolved_id:
        from datetime import datetime, timezone
        resolved_id = f"m06-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}-{uuid.uuid4().hex[:8]}"
    out_dir = reserve_run_directory(project_root, resolved_id, notebook_id)
    log_artifact_path(f"M06 stage {stage} run directory", out_dir)
    return resolved_id, out_dir


class M06Run:
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
            "notebook_id": stage if stage.startswith("long/") else f"long/{stage}",
            "run_id": run_id,
            "parent_run_id": parent_run_id,
            "is_smoke_test": smoke,
            "population_id": configuration.get("population_id", "unknown"),
            "split_id": configuration.get("split_id", "unknown"),
            "seed": 42,
            "git_commit": get_git_commit(),
            "git_dirty": dirty,
            "configuration": configuration,
            "scientific_demarcation": SCIENTIFIC_DEMARCATION_TEXT,
            "scientific_reproduction_certified": False,
            "libraries": get_library_versions(),
            "inputs": {},
            "metrics_summary": {},
            "artifacts": {},
        }
        _atomic(self.manifest_path, self.manifest)

    @contextmanager
    def guard(self) -> Iterator["M06Run"]:
        try:
            yield self
        except Exception as exc:
            self.manifest["status"] = "FAILED"
            self.manifest["failure"] = {
                "error_type": type(exc).__name__,
                "error_message": str(exc),
            }
            _atomic(self.manifest_path, self.manifest)
            raise
        else:
            if self.manifest["status"] == "RUNNING":
                self.manifest["status"] = "COMPLETE"
            _atomic(self.manifest_path, self.manifest)


def begin_m06(
    stage: str,
    run_id: str | None = None,
    parent_run_id: str | None = None,
    config_override: dict[str, Any] | None = None,
    data_root: Path | str | None = None,
    smoke_test: bool | None = None,
    motifs_07_dir: Path | None = None,
    project_root: Path | str | None = None,
) -> M06Run:
    """Entrypoint: reserves collision-free run output directory and writes initial manifest."""
    root = Path(project_root).resolve() if project_root else Path(__file__).resolve().parents[2]
    cfg_path = Path(__file__).resolve().parents[2] / "configs" / "reproduction_m06.yaml"
    cfg: dict[str, Any] = {}
    if cfg_path.exists():
        with open(cfg_path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}

    if config_override:
        cfg.update(config_override)

    if smoke_test is not None:
        cfg["is_smoke_test"] = smoke_test
    else:
        env_smoke = os.getenv("SMOKE_TEST")
        if env_smoke is not None:
            cfg["is_smoke_test"] = env_smoke.lower() in ("1", "true", "yes")
        elif "is_smoke_test" not in cfg:
            cfg["is_smoke_test"] = False

    effective_data_root = data_root or os.getenv("DATA_ROOT") or root / "data"
    cfg["data_root"] = str(effective_data_root)

    resolved_id, output_dir = reserve_m06_run(root, stage, run_id=run_id)

    if cfg["is_smoke_test"]:
        fixtures = output_dir / "fixtures"
        fixtures.mkdir(parents=True, exist_ok=True)
        ensure_smoke_test_artifacts(fixtures)

    manifest_path = output_dir / "manifest.json"
    run = M06Run(
        output=output_dir,
        manifest=manifest_path,
        configuration=cfg,
        run_id=resolved_id,
        parent_run_id=parent_run_id,
        stage=stage,
    )
    return run


def sample_deterministic_contigs(
    df_long: pd.DataFrame,
    min_length: int = 2000,
    n_contigs: int = 3,
    seed: int = 42,
) -> pd.DataFrame:
    """Deterministically samples exemplary contigs meeting minimum length criteria."""
    if "sequence" not in df_long.columns:
        raise ValueError("DataFrame must contain 'sequence' column")

    df_filtered = df_long.copy()
    df_filtered["length"] = df_filtered["sequence"].str.len()
    df_filtered = df_filtered[df_filtered["length"] >= min_length].reset_index(drop=True)

    if len(df_filtered) == 0:
        # Fallback if no sequences reach min_length (e.g. in small fixtures)
        df_filtered = df_long.copy().reset_index(drop=True)
        df_filtered["length"] = df_filtered["sequence"].str.len()

    rng = np.random.RandomState(seed)
    n_samples = min(n_contigs, len(df_filtered))
    chosen_indices = sorted(rng.choice(len(df_filtered), size=n_samples, replace=False))

    sampled = df_filtered.iloc[chosen_indices].copy().reset_index(drop=True)
    if "query_id" not in sampled.columns:
        sampled["query_id"] = [f"contig_{i:03d}" for i in range(len(sampled))]

    gc_pcts = []
    for seq in sampled["sequence"]:
        s = seq.upper()
        gc = 100.0 * sum(b in "GC" for b in s) / len(s) if len(s) > 0 else 0.0
        gc_pcts.append(round(gc, 2))
    sampled["gc_percent"] = gc_pcts

    return sampled


def load_top_motifs_for_long_analysis(
    motifs_07_dir: Optional[Union[str, Path]] = None,
    default_motifs: Optional[List[str]] = None,
    max_motifs: int = 4,
) -> List[str]:
    """Loads top learned 7-mer motifs from short/07 outputs, falling back cleanly if missing."""
    fallback = default_motifs or ["CCCCCCC", "GGGGGGG", "ATATATA", "TTTTTTT"]
    if motifs_07_dir is None:
        motifs_env = os.getenv("M05_MOTIFS_07_DIR") or os.getenv("MOTIFS_07_DIR")
        if motifs_env:
            motifs_07_dir = Path(motifs_env)

    if motifs_07_dir is not None:
        p_dir = Path(motifs_07_dir)
        summary_p = p_dir / "motif_summary.parquet"
        if summary_p.is_file():
            try:
                df = pd.read_parquet(summary_p)
                if "motif" in df.columns:
                    motifs = [str(m).strip() for m in df["motif"].dropna().unique() if len(str(m).strip()) == 7]
                    if motifs:
                        return motifs[:max_motifs]
            except Exception:
                pass

        # Try manifest
        for m_name in ("manifest_07_motifs.json", "manifest.json"):
            man_p = p_dir / m_name
            if man_p.is_file():
                try:
                    with open(man_p, encoding="utf-8") as f:
                        data = json.load(f)
                    extracted = data.get("motifs_extracted") or data.get("top_motifs")
                    if isinstance(extracted, list):
                        motifs = [str(m).strip() for m in extracted if len(str(m).strip()) == 7]
                        if motifs:
                            return motifs[:max_motifs]
                except Exception:
                    pass

    return fallback[:max_motifs]


def slice_contig_and_track_motifs(
    contig_seq: str,
    motifs_tracked: Sequence[str],
    window_size: int = DEFAULT_WINDOW_SIZE,
    stride: int = DEFAULT_STRIDE,
    handle_short: str = "pad",
) -> pd.DataFrame:
    """Slices a long sequence into 512 bp windows and counts occurrences of specified 7-mers."""
    windows = slice_sequence(contig_seq, window_size=window_size, stride=stride, handle_short=handle_short)
    token_to_base = ["A", "C", "G", "T", "N"]

    records = []
    for idx, win_tokens in enumerate(windows):
        win = "".join(token_to_base[t] if 0 <= t < 4 else "N" for t in win_tokens)
        gc_pct = 100.0 * sum(b in "GC" for b in win) / len(win) if len(win) > 0 else 0.0
        rec: Dict[str, Any] = {
            "window_idx": idx,
            "start_pos": idx * stride,
            "end_pos": idx * stride + len(win),
            "gc_percent": round(gc_pct, 2),
        }
        for m in motifs_tracked:
            rec[f"motif_{m}"] = win.count(m)
        records.append(rec)

    return pd.DataFrame(records)


def stratify_long_errors_with_windows(
    df_results: pd.DataFrame,
    length_bins: Optional[List[Tuple[int, float, str]]] = None,
    window_bins: Optional[List[Tuple[int, float, str]]] = None,
    ambiguity_bins: Optional[List[Tuple[float, float, str]]] = None,
    taxon_col: str = "taxon_id",
) -> Dict[str, Any]:
    """Stratifies prediction accuracy across sequence length bins, window counts, and ambiguity."""
    if length_bins is None:
        length_bins = [
            (0, 512, "len_lt_512"),
            (512, 2000, "len_512_to_2000"),
            (2000, 5000, "len_2000_to_5000"),
            (5000, 10001, "len_5000_to_10000"),
            (10001, float("inf"), "len_gt_10000"),
        ]

    if window_bins is None:
        window_bins = [
            (1, 2, "1_window"),
            (2, 11, "2_to_10_windows"),
            (11, 51, "11_to_50_windows"),
            (51, float("inf"), "gt_50_windows"),
        ]

    if ambiguity_bins is None:
        ambiguity_bins = [
            (0.0, 0.0, "ambig_0_pct"),
            (0.0, 0.05, "ambig_0_to_5_pct"),
            (0.05, 1.0, "ambig_gt_5_pct"),
        ]

    df = df_results.copy()
    if "n_windows" not in df.columns:
        if "length" in df.columns:
            # Estimate window count if not explicit
            df["n_windows"] = np.maximum(1, np.ceil((df["length"] - 512) / 256) + 1).astype(int)
        else:
            df["n_windows"] = 1

    mean_correct = df["truth_label"] == df["pred_mean_logits"]
    majority_correct = df["truth_label"] == df["pred_majority_vote"]

    strata: Dict[str, Any] = {
        "overall": {
            "sample_size_n": int(len(df)),
            "correct_mean_logits": int(mean_correct.sum()),
            "correct_majority_vote": int(majority_correct.sum()),
            "accuracy_mean_logits": float(mean_correct.mean()) if len(df) else None,
            "accuracy_majority_vote": float(majority_correct.mean()) if len(df) else None,
        },
        "by_length": {},
        "by_windows": {},
        "by_ambiguity": {},
    }

    # Length bins
    for low, high, label in length_bins:
        mask = (df["length"] >= low) & (df["length"] < high)
        subset = df[mask]
        count = int(len(subset))
        acc_mean = float((subset["truth_label"] == subset["pred_mean_logits"]).mean()) if count > 0 else None
        acc_maj = float((subset["truth_label"] == subset["pred_majority_vote"]).mean()) if count > 0 else None
        strata["by_length"][label] = {
            "range": [low, high],
            "sample_size_n": count,
            "correct_mean_logits": int((subset["truth_label"] == subset["pred_mean_logits"]).sum()),
            "correct_majority_vote": int((subset["truth_label"] == subset["pred_majority_vote"]).sum()),
            "accuracy_mean_logits": acc_mean,
            "accuracy_majority_vote": acc_maj,
        }

    # Window count bins
    for low, high, label in window_bins:
        mask = (df["n_windows"] >= low) & (df["n_windows"] < high)
        subset = df[mask]
        count = int(len(subset))
        acc_mean = float((subset["truth_label"] == subset["pred_mean_logits"]).mean()) if count > 0 else None
        acc_maj = float((subset["truth_label"] == subset["pred_majority_vote"]).mean()) if count > 0 else None
        strata["by_windows"][label] = {
            "range": [low, high],
            "sample_size_n": count,
            "correct_mean_logits": int((subset["truth_label"] == subset["pred_mean_logits"]).sum()),
            "correct_majority_vote": int((subset["truth_label"] == subset["pred_majority_vote"]).sum()),
            "accuracy_mean_logits": acc_mean,
            "accuracy_majority_vote": acc_maj,
        }

    # Ambiguity bins
    for low, high, label in ambiguity_bins:
        if low == high == 0.0:
            mask = df.get("ambiguity_ratio", pd.Series([0.0]*len(df))) == 0.0
        else:
            r = df.get("ambiguity_ratio", pd.Series([0.0]*len(df)))
            mask = (r > low) & (r <= high)
        subset = df[mask]
        count = int(len(subset))
        acc_mean = float((subset["truth_label"] == subset["pred_mean_logits"]).mean()) if count > 0 else None
        acc_maj = float((subset["truth_label"] == subset["pred_majority_vote"]).mean()) if count > 0 else None
        strata["by_ambiguity"][label] = {
            "range": [low, high],
            "sample_size_n": count,
            "correct_mean_logits": int((subset["truth_label"] == subset["pred_mean_logits"]).sum()),
            "correct_majority_vote": int((subset["truth_label"] == subset["pred_majority_vote"]).sum()),
            "accuracy_mean_logits": acc_mean,
            "accuracy_majority_vote": acc_maj,
        }

    return strata


def export_stratified_long_errors_csv(strata: Dict[str, Any], output_path: Union[str, Path]) -> pd.DataFrame:
    """Exports structured stratified errors dictionary to tabular CSV with explicit sample size N."""
    rows = []
    for cat in ("by_length", "by_windows", "by_ambiguity"):
        for label, stats in strata.get(cat, {}).items():
            r = stats.get("range")
            range_str = f"[{r[0]}, {r[1]})" if r else "all"
            rows.append({
                "stratum_category": cat,
                "stratum_label": label,
                "range": range_str,
                "sample_size_n": stats.get("sample_size_n", stats.get("count", 0)),
                "correct_mean_logits": stats.get("correct_mean_logits", 0),
                "correct_majority_vote": stats.get("correct_majority_vote", 0),
                "accuracy_mean_logits": stats.get("accuracy_mean_logits"),
                "accuracy_majority_vote": stats.get("accuracy_majority_vote"),
            })
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    log_artifact_path("Stratified Long Errors CSV", output_path)
    return df


def render_figure_long_accuracy_by_length_and_windows(
    stratified_data: Union[Dict[str, Any], pd.DataFrame],
    output_path: Union[str, Path],
) -> Path:
    """Generates a 2-panel figure showing accuracy across sequence length bins and window bins with N."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    # Convert to dict if DataFrame
    if isinstance(stratified_data, pd.DataFrame):
        df = stratified_data
        by_len = df[df["stratum_category"] == "by_length"]
        by_win = df[df["stratum_category"] == "by_windows"]
        len_labels = by_len["stratum_label"].tolist()
        len_acc = by_len["accuracy_mean_logits"].fillna(0.0).tolist()
        len_n = by_len["sample_size_n"].tolist()
        win_labels = by_win["stratum_label"].tolist()
        win_acc = by_win["accuracy_mean_logits"].fillna(0.0).tolist()
        win_n = by_win["sample_size_n"].tolist()
    else:
        by_len = stratified_data.get("by_length", {})
        len_labels = list(by_len.keys())
        len_acc = [stats.get("accuracy_mean_logits") or 0.0 for stats in by_len.values()]
        len_n = [stats.get("sample_size_n", stats.get("count", 0)) for stats in by_len.values()]

        by_win = stratified_data.get("by_windows", {})
        win_labels = list(by_win.keys())
        win_acc = [stats.get("accuracy_mean_logits") or 0.0 for stats in by_win.values()]
        win_n = [stats.get("sample_size_n", stats.get("count", 0)) for stats in by_win.values()]

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Length panel
    x1 = np.arange(len(len_labels))
    bars1 = ax1.bar(x1, len_acc, color="#1f77b4", alpha=0.85, width=0.55)
    ax1.set_xticks(x1)
    ax1.set_xticklabels([f"{lbl}\n(N={n})" for lbl, n in zip(len_labels, len_n)], rotation=25, ha="right")
    ax1.set_ylabel("Acurácia (Mean Logits)")
    ax1.set_title("Acurácia por Estrato de Comprimento (bp)")
    ax1.set_ylim(0, 1.05)
    ax1.grid(True, linestyle="--", alpha=0.4, axis="y")
    for bar in bars1:
        yval = bar.get_height()
        if yval > 0:
            ax1.text(bar.get_x() + bar.get_width() / 2.0, yval + 0.02, f"{yval:.2f}", ha="center", va="bottom", fontsize=9)

    # Windows panel
    x2 = np.arange(len(win_labels))
    bars2 = ax2.bar(x2, win_acc, color="#2ca02c", alpha=0.85, width=0.55)
    ax2.set_xticks(x2)
    ax2.set_xticklabels([f"{lbl}\n(N={n})" for lbl, n in zip(win_labels, win_n)], rotation=25, ha="right")
    ax2.set_ylabel("Acurácia (Mean Logits)")
    ax2.set_title("Acurácia por Contagem de Janelas Deslizantes")
    ax2.set_ylim(0, 1.05)
    ax2.grid(True, linestyle="--", alpha=0.4, axis="y")
    for bar in bars2:
        yval = bar.get_height()
        if yval > 0:
            ax2.text(bar.get_x() + bar.get_width() / 2.0, yval + 0.02, f"{yval:.2f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig(out, dpi=300)
    plt.close(fig)
    log_artifact_path("Figure Long Accuracy by Length and Windows", out)
    return out


def render_figure_long_paired_vote_vs_mean_logits(
    table_12_df: pd.DataFrame,
    output_path: Union[str, Path],
) -> Path:
    """Renders paired comparison between majority vote and mean logits across the 5 variants."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    df = table_12_df.copy()
    variants = df["variant"].unique()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(variants))
    width = 0.35

    mean_acc = []
    vote_acc = []
    support_n = []

    for v in variants:
        v_rows = df[df["variant"] == v]
        m_row = v_rows[v_rows["aggregation"] == "mean_logits"]
        v_row = v_rows[v_rows["aggregation"] == "majority_vote"]

        m_val = float(m_row["overall_accuracy"].iloc[0]) if len(m_row) else 0.0
        v_val = float(v_row["overall_accuracy"].iloc[0]) if len(v_row) else 0.0
        n_val = int(m_row["covered_queries"].iloc[0]) if len(m_row) else 0

        mean_acc.append(m_val)
        vote_acc.append(v_val)
        support_n.append(n_val)

    bars_mean = ax.bar(x - width/2, mean_acc, width, label="Mean Logits (Pré-Softmax)", color="#1f77b4", alpha=0.85)
    bars_vote = ax.bar(x + width/2, vote_acc, width, label="Majority Vote (Voto Majoritário)", color="#ff7f0e", alpha=0.85)

    ax.set_ylabel("Acurácia Global")
    ax.set_title("Comparação Pareada: Média de Logits vs. Voto Majoritário por Variante (AD-010, AD-020)")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{v}\n(N={n})" for v, n in zip(variants, support_n)], rotation=20, ha="right")
    ax.set_ylim(0, 1.08)
    ax.legend(loc="upper right")
    ax.grid(True, linestyle="--", alpha=0.4, axis="y")

    for bm, bv in zip(bars_mean, bars_vote):
        ym = bm.get_height()
        yv = bv.get_height()
        if ym > 0:
            ax.text(bm.get_x() + bm.get_width()/2.0, ym + 0.015, f"{ym:.2f}", ha="center", va="bottom", fontsize=8)
        if yv > 0:
            ax.text(bv.get_x() + bv.get_width()/2.0, yv + 0.015, f"{yv:.2f}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    plt.savefig(out, dpi=300)
    plt.close(fig)
    log_artifact_path("Figure Long Paired Vote vs Mean Logits", out)
    return out


def render_figure_long_motifs_gc(
    df_motifs: pd.DataFrame,
    output_path: Union[str, Path],
    title: Optional[str] = None,
) -> Path:
    """Renders dual-panel GC profile and 7-mer motif occurrence along a long sequence."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 6), sharex=True)

    positions = df_motifs["start_pos"]
    ax1.plot(positions, df_motifs["gc_percent"], color="teal", marker="o", linewidth=2, markersize=4)
    ax1.set_ylabel("Teor de GC (%)")
    ax1.set_title(title or "Perfil de GC e Ativação de Motivos em Sequência Longa Viral (AD-010, AD-020)")
    ax1.grid(True, linestyle="--", alpha=0.5)

    motif_cols = [c for c in df_motifs.columns if c.startswith("motif_")]
    colors = ["#1f77b4", "#2ca02c", "#d62728", "#9467bd", "#ff7f0e"]

    offset = -30
    for i, col in enumerate(motif_cols[:4]):
        motif_name = col.replace("motif_", "")
        color = colors[i % len(colors)]
        ax2.bar(positions + offset, df_motifs[col], width=50, label=motif_name, color=color, alpha=0.75)
        offset += 25

    ax2.set_xlabel("Posição na Sequência Longa (bp)")
    ax2.set_ylabel("Contagem de Ocorrências")
    ax2.legend(loc="upper right")
    ax2.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(out, dpi=300)
    plt.close(fig)
    log_artifact_path("Figure Long Motifs and GC Profile", out)
    return out
