"""Render paired latent coordinates without changing selection or fitting models."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from viral_cnn.eval.latent import _sha256_file
from viral_cnn.latent_resume import _atomic
from viral_cnn.paired_umap import MODEL_NAMES, PairedSample
from viral_cnn.utils.logging_utils import log_artifact_path

RENDER_PROTOCOL = 'c12_large_taxonomy_render_v1'
DISPLAY_NAMES = dict(zip(MODEL_NAMES, ('ResNet50', 'ResNet50_F', 'ConvNeXt_Z', 'ConvNeXt_T', 'ConvNeXt_F')))


def _record(path: Path) -> dict:
    return {'path': str(path.resolve()), 'sha256': _sha256_file(path)}


def _validate(frames: dict, method: str) -> pd.DataFrame:
    if set(frames) != set(MODEL_NAMES) or method not in ('UMAP', 'PCA'):
        raise ValueError('Expected five paired UMAP/PCA projections')
    base = frames[MODEL_NAMES[0]][['row_id', 'label', 'Genus']].reset_index(drop=True)
    if base.empty or base.isna().any().any() or not base.row_id.is_unique:
        raise ValueError('Empty, null or duplicated projection identities')
    for frame in frames.values():
        if not frame[['row_id', 'label', 'Genus']].reset_index(drop=True).equals(base):
            raise ValueError('Projection IDs/labels/genera differ between models')
        if (not np.isfinite(frame[['umap_1', 'umap_2']].to_numpy(dtype=float)).all()
                or not frame['projection_method'].eq(method).all()
                or not frame['is_qualitative_only'].eq(True).all()):
            raise ValueError('Invalid coordinates, method or qualitative marker')
    return base


def load_saved_comparison(source_dir: Path | str, output_dir: Path | str) -> tuple[dict, list[dict]]:
    """Verify and copy coordinate CSVs only; no embedding or taxonomy access."""
    source, output = Path(source_dir).resolve(), Path(output_dir).resolve()
    if source == output:
        raise ValueError('Render output must differ from its source')
    manifest_path = source / 'umap_comparison.json'
    log_artifact_path('saved projection comparison', manifest_path)
    metadata = json.loads(manifest_path.read_text())
    if metadata.get('status') != 'COMPLETE' or metadata.get('protocol') != 'c10_paired_umap_v1':
        raise ValueError('Incomplete or unsupported paired comparison')
    if set(metadata.get('projections', {})) != set(MODEL_NAMES):
        raise ValueError('Missing model projection records')
    frames, inputs, copies = {}, [_record(manifest_path)], []
    for name in MODEL_NAMES:
        record = metadata['projections'][name]
        # Relocation is supported, while embedded absolute paths cannot escape source.
        path = source / Path(record['path']).name
        log_artifact_path('saved coordinate CSV', path)
        content = path.read_bytes()
        digest = hashlib.sha256(content).hexdigest()
        if digest != record['sha256']:
            raise ValueError(f'Corrupt saved projection: {path}')
        import io
        frame = pd.read_csv(io.BytesIO(content), comment='#', dtype={'row_id': str, 'Genus': str}, float_precision='round_trip')
        frames[name] = frame
        inputs.append({'path': str(path), 'sha256': digest})
        copies.append((path.name, content))
    base = _validate(frames, metadata['projection_method'])
    selection_sha = hashlib.sha256(json.dumps(base.to_dict(orient='records'), sort_keys=True).encode()).hexdigest()
    if selection_sha != metadata.get('selection_sha256'):
        raise ValueError('Saved selection fingerprint differs from coordinate identities')
    if len(base) != metadata['selection']['selected_count']:
        raise ValueError('Saved selection count differs from coordinate rows')
    output.mkdir(parents=True, exist_ok=True)
    for filename, content in copies:
        _atomic(output / filename, lambda p, data=content: p.write_bytes(data))
    _atomic(output / 'umap_sample_selection.csv', lambda p: base.to_csv(p, index=False))
    metadata['render_source'] = inputs[0]
    return {'projections': frames, 'sample': PairedSample(base, {}, metadata['selection']), 'metadata': metadata}, inputs


def render_comparison(comparison: dict, output_dir: Path | str) -> dict:
    """Adaptive static overview plus offline searchable/zoomable HTML explorer.

    Coordinates and rows are unchanged. A fixed drawing permutation avoids a
    genus-by-genus painting order hiding earlier categories in dense regions.
    Thousands of colors cannot be perceptually distinguished: search highlights
    one genus against the complete population instead of relying on the palette.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_hex
    from matplotlib.patches import Patch

    output = Path(output_dir).resolve()
    frames, metadata = comparison['projections'], comparison['metadata']
    method = metadata['projection_method']
    base = _validate(frames, method)
    genera = sorted(base.Genus.unique())
    # Preserve the original C10 color identity; only rendering changes here.
    colors = metadata.get('genus_colors', {})
    if set(colors) != set(genera):
        palette = plt.get_cmap('tab20' if len(genera) <= 20 else 'turbo')
        colors = {g: to_hex(palette(i / max(1, len(genera) - 1))) for i, g in enumerate(genera)}
    n, large = len(base), len(genera) > 20
    order = np.random.default_rng(42).permutation(n)
    point_size = 2 if n > 10000 else (5 if n > 1000 else 12)
    alpha = 0.45 if n > 10000 else 0.7
    point_colors = base.Genus.map(colors).to_numpy()[order]
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    try:
        for ax, name in zip(axes.flat, MODEL_NAMES):
            xy = frames[name][['umap_1', 'umap_2']].to_numpy()[order]
            ax.scatter(xy[:, 0], xy[:, 1], c=point_colors, s=point_size, alpha=alpha,
                       linewidths=0, rasterized=True)
            ax.set(title=f'{DISPLAY_NAMES[name]} · N={n:,}', xlabel=f'{method} 1', ylabel=f'{method} 2')
            ax.set_aspect('equal', adjustable='box')
            ax.margins(0.05)
            ax.spines[['top', 'right']].set_visible(False)
        info = axes.flat[-1]
        info.axis('off')
        if not large:
            info.legend(handles=[Patch(color=colors[g], label=g) for g in genera],
                        loc='center', frameon=False, title=f'Genus ({len(genera)})', fontsize=8)
        else:
            info.text(0.05, 0.75, f'{len(genera):,} gêneros · {n:,} fragmentos\n\n'
                      'Todos os pontos estão representados.\n'
                      'Consulte um gênero no HTML interativo:\n'
                      'busca, destaque nos cinco modelos e zoom.\n\n'
                      'Cores e contagens: umap_genus_colors.csv\n'
                      'Milhares de cores não são distinguíveis.',
                      transform=info.transAxes, va='top', fontsize=11, linespacing=1.7)
        fig.suptitle(f'{method} — cinco modelos, mesma amostra de teste', fontsize=16, y=0.98)
        fig.text(0.5, 0.02, 'QUALITATIVE ONLY · Eixos/ajustes independentes; sem inferência quantitativa ou filogenética.',
                 ha='center', fontsize=10)
        fig.subplots_adjust(left=0.06, right=0.98, bottom=0.09, top=0.90, hspace=0.38, wspace=0.27)
        _atomic(output / 'umap_comparison.png', lambda p: fig.savefig(p, format='png', dpi=160))
    finally:
        plt.close(fig)
    counts = base.Genus.value_counts()
    table = pd.DataFrame({'Genus': genera, 'color': [colors[g] for g in genera],
                          'selected_count': [int(counts[g]) for g in genera]})
    _atomic(output / 'umap_genus_colors.csv', lambda p: table.to_csv(p, index=False))

    # Highlighted genera & selective panel
    sorted_genera = sorted(genera, key=lambda g: (-int(counts[g]), str(g)))
    max_highlight = min(10, len(sorted_genera))
    highlighted_set = set(sorted_genera[:max_highlight])

    selected_genera_df = pd.DataFrame([
        {
            'Genus': g,
            'color': colors.get(g, '#d0d0d0'),
            'sample_count': int(counts[g]),
            'selection_rank': i + 1,
            'is_highlighted': g in highlighted_set,
        }
        for i, g in enumerate(sorted_genera)
    ])
    _atomic(output / 'selected_genera.csv', lambda p: selected_genera_df.to_csv(p, index=False))

    sampled_queries_df = pd.DataFrame({
        'query_id': base['row_id'].to_numpy(dtype=str),
        'label': base['label'].to_numpy(dtype=int),
        'Genus': base['Genus'].to_numpy(dtype=str),
        'is_highlighted': base['Genus'].isin(highlighted_set).to_numpy(dtype=bool),
    })
    _atomic(output / 'sampled_query_ids.csv', lambda p: sampled_queries_df.to_csv(p, index=False))

    fig_sel, axes_sel = plt.subplots(1, 5, figsize=(22, 5))
    try:
        sorted_highlighted = sorted(highlighted_set)
        for ax, name in zip(axes_sel, MODEL_NAMES):
            frame = frames[name]
            bg = frame[~frame['Genus'].isin(highlighted_set)]
            if not bg.empty:
                ax.scatter(bg['umap_1'], bg['umap_2'], color='#d0d0d0', s=10, alpha=0.35, label='_nolegend_')
            for genus in sorted_highlighted:
                subset = frame[frame['Genus'] == genus]
                if not subset.empty:
                    ax.scatter(subset['umap_1'], subset['umap_2'], color=colors[genus], s=20, alpha=0.85)
            ax.set_title(f"{DISPLAY_NAMES[name]}\n{method} (selective) · N={len(frame)}")
            ax.set_xlabel(f"{method} 1")
            ax.set_ylabel(f"{method} 2")
            ax.set_aspect('equal', adjustable='box')
            ax.margins(0.05)
            ax.spines[['top', 'right']].set_visible(False)
        handles = [Patch(color=colors[g], label=g) for g in sorted_highlighted]
        if any((~frames[m]['Genus'].isin(highlighted_set)).any() for m in MODEL_NAMES):
            handles.append(Patch(color='#d0d0d0', label='Outros gêneros (fundo)'))
        fig_sel.legend(handles=handles, loc='lower center', ncol=min(6, len(handles)), title='Gêneros selecionados')
        fig_sel.suptitle(f"{method} — amostra pareada; gêneros em destaque sobre fundo neutro (qualitativo)")
        fig_sel.subplots_adjust(bottom=0.26, top=0.78, wspace=0.35)
        _atomic(output / 'umap_selective_panel.png', lambda p: fig_sel.savefig(p, format='png', dpi=160))
    finally:
        plt.close(fig_sel)

    payload = {'method': method, 'genera': genera, 'colors': [colors[g] for g in genera],
               'counts': table.selected_count.tolist(), 'ids': base.row_id.tolist(),
               'labels': base.label.tolist(), 'genus': pd.Categorical(base.Genus, categories=genera).codes.tolist(),
               'order': order.tolist(), 'models': [
                   {'name': DISPLAY_NAMES[name], 'xy': frames[name][['umap_1', 'umap_2']].to_numpy().tolist()}
                   for name in MODEL_NAMES]}
    # JSON is data, never executable markup, including untrusted taxon/row names.
    encoded = json.dumps(payload, ensure_ascii=True, allow_nan=False).replace('<', '\\u003c').replace('>', '\\u003e').replace('&', '\\u0026')
    template = Path(__file__).with_name('latent_explorer.html').read_text()
    _atomic(output / 'umap_comparison.html', lambda p: p.write_text(template.replace('__PROJECTION_DATA__', encoded), encoding='utf-8'))
    for name, record in metadata['projections'].items():
        record.update(_record(output / Path(record['path']).name))
    metadata.update({'genus_colors': colors, 'panel_file': _record(output / 'umap_comparison.png'),
                     'sample_file': _record(output / 'umap_sample_selection.csv'),
                     'rendering': {'protocol': RENDER_PROTOCOL, 'code_sha256': _sha256_file(__file__),
                                   'template_sha256': _sha256_file(Path(__file__).with_name('latent_explorer.html')),
                                   'draw_seed': 42, 'point_size': point_size, 'alpha': alpha,
                                   'legend': 'external searchable' if large else 'dedicated panel',
                                   'coordinates_changed': False, 'rows_displayed_per_model': n,
                                   'interactive_file': _record(output / 'umap_comparison.html'),
                                   'colors_file': _record(output / 'umap_genus_colors.csv')}})
    _atomic(output / 'umap_comparison.json', lambda p: p.write_text(json.dumps(metadata, indent=2) + '\n'))
    return metadata['rendering']


def render_quantitative_summary(
    table5_records: list[dict[str, Any]],
    probe_records: list[dict[str, Any]] | None,
    output_dir: Path | str,
) -> tuple[Path, Path]:
    """Render high-dimensional original space metrics (KNN, Silhouette, Probe) to CSV and PNG."""
    import matplotlib.pyplot as plt

    output = Path(output_dir).resolve()
    csv_path = output / 'latent_quantitative_summary.csv'
    png_path = output / 'latent_quantitative_summary.png'

    t5_df = pd.DataFrame(table5_records) if isinstance(table5_records, list) else pd.DataFrame(table5_records)
    probe_df = pd.DataFrame(probe_records) if probe_records else pd.DataFrame()

    combined_rows = []
    model_col = 'Model' if 'Model' in t5_df.columns else ('model' if 'model' in t5_df.columns else None)
    models = t5_df[model_col].unique() if model_col else list(DISPLAY_NAMES.values())

    for m in models:
        row = {'Model': m}
        sub = t5_df[t5_df[model_col] == m] if model_col else pd.DataFrame()
        if 'Rank' in sub.columns or 'rank' in sub.columns:
            rcol = 'Rank' if 'Rank' in sub.columns else 'rank'
            sp_sub = sub[sub[rcol].astype(str).str.lower() == 'species']
            if not sp_sub.empty:
                acc_cols = [c for c in sp_sub.columns if 'knn' in c.lower() or 'accuracy' in c.lower()]
                if acc_cols:
                    row['KNN_Species_Acc'] = float(sp_sub[acc_cols[0]].iloc[0])
                sil_cols = [c for c in sp_sub.columns if 'silhouette' in c.lower()]
                if sil_cols:
                    row['Silhouette_Species'] = float(sp_sub[sil_cols[0]].iloc[0])
            gen_sub = sub[sub[rcol].astype(str).str.lower() == 'genus']
            if not gen_sub.empty:
                acc_cols = [c for c in gen_sub.columns if 'knn' in c.lower() or 'accuracy' in c.lower()]
                if acc_cols:
                    row['KNN_Genus_Acc'] = float(gen_sub[acc_cols[0]].iloc[0])
        elif not sub.empty:
            for c in sub.columns:
                if c != model_col:
                    row[c] = sub[c].iloc[0]

        if not probe_df.empty:
            p_mcol = 'Model' if 'Model' in probe_df.columns else ('model' if 'model' in probe_df.columns else None)
            if p_mcol:
                p_sub = probe_df[probe_df[p_mcol] == m]
                if not p_sub.empty:
                    if 'Linear_Probe_Test_Acc' in p_sub.columns:
                        row['Linear_Probe_Acc'] = float(p_sub['Linear_Probe_Test_Acc'].iloc[0])
                    if 'Native_Head_Accuracy' in p_sub.columns:
                        row['Native_Head_Acc'] = float(p_sub['Native_Head_Accuracy'].iloc[0])
        combined_rows.append(row)

    summary_df = pd.DataFrame(combined_rows)
    _atomic(csv_path, lambda p: summary_df.to_csv(p, index=False))

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    try:
        ax1 = axes[0]
        knn_cols = [c for c in summary_df.columns if 'knn' in c.lower()]
        if knn_cols:
            summary_df.plot(x='Model', y=knn_cols, kind='bar', ax=ax1, rot=25)
            ax1.set_title('KNN Cosseno (k=5) no Espaço Original')
            ax1.set_ylabel('Acurácia')
            ax1.grid(True, linestyle='--', alpha=0.5)
        else:
            ax1.text(0.5, 0.5, 'Dados de KNN não disponíveis', ha='center', va='center')

        ax2 = axes[1]
        sil_cols = [c for c in summary_df.columns if 'silhouette' in c.lower()]
        if sil_cols:
            summary_df.plot(x='Model', y=sil_cols, kind='bar', ax=ax2, rot=25, color='teal')
            ax2.set_title('Silhouette Score no Espaço Original')
            ax2.set_ylabel('Score [-1, 1]')
            ax2.grid(True, linestyle='--', alpha=0.5)
        else:
            ax2.text(0.5, 0.5, 'Dados de Silhouette não disponíveis', ha='center', va='center')

        ax3 = axes[2]
        probe_cols = [c for c in summary_df.columns if 'probe' in c.lower() or 'native' in c.lower()]
        if probe_cols:
            summary_df.plot(x='Model', y=probe_cols, kind='bar', ax=ax3, rot=25, color=['coral', 'steelblue'][:len(probe_cols)])
            ax3.set_title('Sonda Linear vs Classificador Nativo')
            ax3.set_ylabel('Acurácia em Test')
            ax3.grid(True, linestyle='--', alpha=0.5)
        else:
            ax3.text(0.5, 0.5, 'Dados de Sonda não disponíveis', ha='center', va='center')

        fig.suptitle('Métricas Quantitativas do Espaço Latente Original (Pré-Classificador)', fontsize=14, y=0.98)
        fig.tight_layout()
        _atomic(png_path, lambda p: fig.savefig(p, format='png', dpi=160))
    finally:
        plt.close(fig)

    return csv_path, png_path

