"""Paired, bounded qualitative projections of the five identified test embeddings.

This workflow stays outside eval.latent to preserve historical extractor hashes.
Quantitative evaluations continue to use the original high-dimensional space.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from viral_cnn.eval import latent
from viral_cnn.latent_resume import LatentResumeCache, StageResult, _atomic
from viral_cnn.utils.logging_utils import log_artifact_path

PROTOCOL = 'c10_paired_umap_v1'
MODEL_NAMES = ('resnet_clean_species', 'resnet_noise_species', 'convnext_zero_species',
               'convnext_transfer_species', 'convnext_noise_species')
DISCLAIMER = '# QUALITATIVE ONLY: independently fitted 2D projections; axes are not aligned.\n'
INTERPRETATION = '# Do not infer quantitative cluster separation or phylogenetic distance.\n'


@dataclass(frozen=True)
class ProjectionConfig:
    """Sampling limits for a shared genus-stratified visualization population."""

    max_samples: int = 100
    max_genera: int = 10
    seed: int = 42

    def __post_init__(self) -> None:
        for name, minimum in (('max_samples', 3), ('max_genera', 1)):
            value = getattr(self, name)
            if type(value) is not int or value < minimum:
                raise ValueError(f'{name} must be an integer >= {minimum}')
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError('seed must be a non-negative integer')


@dataclass
class PairedSample:
    """Canonical selected rows, their model-specific positions and coverage."""

    frame: pd.DataFrame
    indices: dict[str, np.ndarray]
    audit: dict[str, Any]


def select_paired_sample(labels: dict[str, np.ndarray], row_ids: dict[str, np.ndarray],
                        species_to_genus: dict[int, str], config: ProjectionConfig) -> PairedSample:
    """Select balanced genus quotas once, then verify each selected ID in all models.

    Sorting IDs before seeded permutation makes sampling independent of storage
    order. Only selected IDs need to align; no silent intersection is performed.
    """
    for name in MODEL_NAMES:
        if name not in labels or name not in row_ids or len(labels[name]) != len(row_ids[name]):
            raise ValueError(f'Missing or misaligned test labels/IDs: {name}')
    base = MODEL_NAMES[0]
    classes, counts = np.unique(labels[base], return_counts=True)
    genus_species: dict[str, list[int]] = {}
    eligible = 0
    for label, count in zip(classes, counts):
        genus = species_to_genus.get(int(label))
        if genus is not None and pd.notna(genus) and str(genus).strip() and not str(genus).startswith('Unassigned'):
            genus_species.setdefault(str(genus), []).append(int(label))
            eligible += int(count)
    rng = np.random.default_rng(config.seed)
    genera = sorted(genus_species)
    chosen_count = min(config.max_genera, len(genera), config.max_samples)
    chosen = sorted(rng.choice(genera, size=chosen_count, replace=False).tolist()) if chosen_count else []
    candidates = {}
    for genus in chosen:
        positions = np.flatnonzero(np.isin(labels[base], genus_species[genus]))
        positions = positions[np.argsort(row_ids[base][positions], kind='stable')]
        candidates[genus] = positions[rng.permutation(len(positions))]
    selected: list[tuple[int, str]] = []
    offset = 0
    while len(selected) < config.max_samples:
        added = False
        for genus in chosen:
            if offset < len(candidates[genus]) and len(selected) < config.max_samples:
                selected.append((int(candidates[genus][offset]), genus))
                added = True
        if not added:
            break
        offset += 1
    if len(selected) < 3:
        raise ValueError(f'Paired projection requires at least 3 selected eligible fragments; found {len(selected)}')
    frame = pd.DataFrame({'row_id': [str(row_ids[base][i]) for i, _ in selected],
                          'label': [int(labels[base][i]) for i, _ in selected],
                          'Genus': [genus for _, genus in selected]})
    frame = frame.sort_values('row_id', kind='stable').reset_index(drop=True)
    if not frame['row_id'].is_unique:
        raise ValueError('Duplicate selected row ID in reference model')
    wanted = frame['row_id'].to_numpy(dtype=str)
    indices = {}
    for name in MODEL_NAMES:
        matches: dict[str, int] = {}
        for start in range(0, len(row_ids[name]), 8192):
            batch = row_ids[name][start:start + 8192]
            for local in np.flatnonzero(np.isin(batch, wanted)):
                row_id = str(batch[local])
                if row_id in matches:
                    raise ValueError(f'Duplicate selected ID in {name}: {row_id}')
                matches[row_id] = start + int(local)
        missing = set(wanted) - matches.keys()
        if missing:
            raise ValueError(f'Missing selected ID in {name}: {sorted(missing)[0]}')
        positions = np.array([matches[key] for key in wanted], dtype=np.int64)
        if not np.array_equal(labels[name][positions], frame['label'].to_numpy()):
            raise ValueError(f'Selected species labels differ in {name}')
        indices[name] = positions
    audit = {'population_available': len(labels[base]), 'eligible_count': eligible,
             'excluded_missing_genus': len(labels[base]) - eligible,
             'eligible_genera': len(genera), 'selected_genera': chosen,
             'eligible_not_selected': eligible - len(frame), 'selected_count': len(frame),
             'selected_counts': {str(k): int(v) for k, v in frame['Genus'].value_counts().sort_index().items()},
             'populations_by_model': {name: len(labels[name]) for name in MODEL_NAMES},
             'configuration': asdict(config), 'sampling_rule': 'seeded genus choice; balanced round-robin quotas without replacement; canonical row IDs'}
    return PairedSample(frame, indices, audit)


def projection_method() -> str:
    """Identify the same import-based backend decision as the historical exporter."""
    try:
        from umap import UMAP  # noqa: F401 — match export_umap_projection's optional import
    except ImportError:
        return 'PCA'
    return 'UMAP'


def _csv(frame: pd.DataFrame, path: Path, *, projection: bool = False) -> None:
    def write(temp: Path) -> None:
        with temp.open('w', encoding='utf-8') as stream:
            if projection:
                stream.write(DISCLAIMER)
                stream.write(INTERPRETATION)
            frame.to_csv(stream, index=False)
    _atomic(path, write)


def _file_record(path: Path) -> dict[str, str]:
    return {'path': str(path.resolve()), 'sha256': latent._sha256_file(path)}


def _panel(frames: dict[str, pd.DataFrame], names: dict[str, str], method: str,
           path: Path) -> dict[str, str]:
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_hex
    from matplotlib.patches import Patch

    genera = sorted(frames[MODEL_NAMES[0]]['Genus'].unique())
    palette = plt.get_cmap('tab20' if len(genera) <= 20 else 'turbo')
    colors = {genus: to_hex(palette(i / max(1, len(genera) - 1))) for i, genus in enumerate(genera)}
    fig, axes = plt.subplots(1, 5, figsize=(20, 5))
    try:
        for ax, name in zip(axes, MODEL_NAMES):
            frame = frames[name]
            for genus in genera:
                subset = frame[frame['Genus'] == genus]
                ax.scatter(subset['umap_1'], subset['umap_2'], color=colors[genus], s=14, alpha=0.8)
            ax.set_title(f'{names[name]}\n{method} · N={len(frame)}')
            ax.set_xlabel(f'{method} 1')
            ax.set_ylabel(f'{method} 2')
        fig.legend(handles=[Patch(color=colors[g], label=g) for g in genera], loc='lower center',
                   ncol=min(5, len(genera)), title='Genus')
        fig.suptitle('QUALITATIVE ONLY — independent fits and axes; no quantitative or phylogenetic inference')
        fig.subplots_adjust(bottom=0.26, top=0.78, wspace=0.35)
        _atomic(path, lambda p: fig.savefig(p, format='png', dpi=150))
    finally:
        plt.close(fig)
    return colors


def _selective_panel(frames: dict[str, pd.DataFrame], names: dict[str, str], method: str,
                     path: Path, highlighted: set[str], colors: dict[str, str]) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    sorted_highlighted = sorted(highlighted)
    fig, axes = plt.subplots(1, 5, figsize=(22, 5))
    try:
        for ax, name in zip(axes, MODEL_NAMES):
            frame = frames[name]
            bg = frame[~frame['Genus'].isin(highlighted)]
            if not bg.empty:
                ax.scatter(bg['umap_1'], bg['umap_2'], color='#d0d0d0', s=10, alpha=0.35, label='_nolegend_')
            for genus in sorted_highlighted:
                subset = frame[frame['Genus'] == genus]
                if not subset.empty:
                    ax.scatter(subset['umap_1'], subset['umap_2'], color=colors[genus], s=20, alpha=0.85)
            ax.set_title(f'{names[name]}\n{method} (selective) · N={len(frame)}')
            ax.set_xlabel(f'{method} 1')
            ax.set_ylabel(f'{method} 2')
        handles = [Patch(color=colors[g], label=g) for g in sorted_highlighted]
        if any((~frames[m]['Genus'].isin(highlighted)).any() for m in MODEL_NAMES):
            handles.append(Patch(color='#d0d0d0', label='Outros gêneros (fundo)'))
        fig.legend(handles=handles, loc='lower center', ncol=min(6, len(handles)), title='Gêneros selecionados')
        fig.suptitle('QUALITATIVE ONLY — amostra pareada; gêneros selecionados sobre fundo neutro; sem inferência filogenética.')
        fig.subplots_adjust(bottom=0.26, top=0.78, wspace=0.35)
        _atomic(path, lambda p: fig.savefig(p, format='png', dpi=150))
    finally:
        plt.close(fig)


def run_paired_umap(model_specs: list[dict[str, Any]], features: dict[str, Any],
                    labels: dict[str, np.ndarray], row_ids: dict[str, np.ndarray],
                    species_to_genus: dict[int, str], resume: LatentResumeCache,
                    environment: dict[str, Any], *, config: ProjectionConfig = ProjectionConfig(),
                    embedding_inputs: dict[str, Any] | None = None) -> dict[str, Any]:
    """Generate/restore all five bounded projections and publish a complete comparison.

    Selection and label alignment finish before any fit. Per-model stage records
    can survive interrupted panel/manifest publication. The comparison JSON is
    published last; its absence denotes an incomplete comparison.
    """
    if len(model_specs) != 5 or {m['name'] for m in model_specs} != set(MODEL_NAMES):
        raise ValueError('Paired comparison requires exactly the five registered species variants')
    for name, source in (embedding_inputs or {}).items():
        log_artifact_path(f'projection source embedding for {name}', source['path'])
    sample = select_paired_sample(labels, row_ids, species_to_genus, config)
    method = projection_method()
    params = {'n_neighbors': min(15, len(sample.frame) - 1), 'min_dist': 0.1,
              'random_state': config.seed, 'metric': 'cosine'}
    records = sample.frame.to_dict(orient='records')
    selection_sha = hashlib.sha256(json.dumps(records, sort_keys=True).encode()).hexdigest()
    sample_path = resume.output_dir / 'umap_sample_selection.csv'
    _csv(sample.frame, sample_path)
    frames, outputs = {}, {}
    specs = {m['name']: m for m in model_specs}
    for name in MODEL_NAMES:
        filename = 'umap_projection_convnext_f.csv' if name == 'convnext_noise_species' else f'umap_projection_{name}.csv'
        path = resume.output_dir / filename
        identity = dict(environment, protocol=PROTOCOL, variant=name,
                        projection_code_sha256=latent._sha256_file(__file__),
                        exporter_code_sha256=latent._sha256_file(latent.__file__),
                        embedding_sha256=specs[name]['test_artifact_sha256'],
                        selection=records, selection_sha256=selection_sha,
                        sampling=asdict(config), projection_method=method, parameters=dict(params, n_components=2))

        def compute() -> StageResult:
            selected_features = features[name][sample.indices[name]]
            selected_labels = sample.frame['label'].to_numpy()
            if method == 'UMAP':
                # The legacy exporter catches ImportError during fit as well as import,
                # but does not report fallback. Keep fitting errors explicit here.
                from umap import UMAP
                coordinates = UMAP(n_components=2, **params).fit_transform(selected_features)
                frame = pd.DataFrame({'umap_1': coordinates[:, 0], 'umap_2': coordinates[:, 1], 'label': selected_labels})
            else:
                frame = latent.export_umap_projection(selected_features, selected_labels, **params)
            frame['row_id'] = sample.frame['row_id'].to_numpy()
            frame['Genus'] = sample.frame['Genus'].to_numpy()
            frame['projection_method'] = method
            frame['is_qualitative_only'] = True
            _csv(frame, path, projection=True)
            return StageResult(frame, [path])

        def validate(frame: pd.DataFrame) -> None:
            if not isinstance(frame, pd.DataFrame) or len(frame) != len(sample.frame):
                raise ValueError('Projection row count differs from selected sample')
            for column in ('row_id', 'label', 'Genus'):
                if frame[column].tolist() != sample.frame[column].tolist():
                    raise ValueError(f'Projection differs from shared selection: {column}')
            if (not frame['projection_method'].eq(method).all() or not frame['is_qualitative_only'].eq(True).all()
                    or not np.isfinite(frame[['umap_1', 'umap_2']].to_numpy(dtype=float)).all()):
                raise ValueError('Projection method/coordinates/qualitative marker invalid')

        frame = resume.run_stage(f'umap_{name}', identity, compute, validate=validate)
        if not path.exists():
            _csv(frame, path, projection=True)
        frames[name] = frame
        outputs[name] = dict(_file_record(path), embedding_sha256=specs[name]['test_artifact_sha256'])
    # Deterministic highlighted genera selection and CSV exports
    genera = sorted(frames[MODEL_NAMES[0]]['Genus'].unique())
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_hex
    palette = plt.get_cmap('tab20' if len(genera) <= 20 else 'turbo')
    colors = {genus: to_hex(palette(i / max(1, len(genera) - 1))) for i, genus in enumerate(genera)}

    counts = sample.frame['Genus'].value_counts()
    sorted_genera = sorted(counts.index, key=lambda g: (-int(counts[g]), str(g)))
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
    selected_genera_path = resume.output_dir / 'selected_genera.csv'
    _csv(selected_genera_df, selected_genera_path)

    sampled_queries_df = pd.DataFrame({
        'query_id': sample.frame['row_id'].to_numpy(dtype=str),
        'label': sample.frame['label'].to_numpy(dtype=int),
        'Genus': sample.frame['Genus'].to_numpy(dtype=str),
        'is_highlighted': sample.frame['Genus'].isin(highlighted_set).to_numpy(dtype=bool),
    })
    sampled_queries_path = resume.output_dir / 'sampled_query_ids.csv'
    _csv(sampled_queries_df, sampled_queries_path)

    selective_panel_path = resume.output_dir / 'umap_selective_panel.png'
    _selective_panel(frames, {m['name']: m['display'] for m in model_specs}, method, selective_panel_path, highlighted_set, colors)

    panel_path = resume.output_dir / 'umap_comparison.png'
    _panel(frames, {m['name']: m['display'] for m in model_specs}, method, panel_path)

    metadata = {'protocol': PROTOCOL, 'status': 'COMPLETE', 'projection_method': method,
                'parameters': dict(params, n_components=2), 'selection': sample.audit,
                'selection_sha256': selection_sha, 'sample_file': _file_record(sample_path),
                'panel_file': _file_record(panel_path),
                'selective_panel_file': _file_record(selective_panel_path),
                'selected_genera_file': _file_record(selected_genera_path),
                'sampled_query_ids_file': _file_record(sampled_queries_path),
                'genus_colors': colors,
                'projections': outputs, 'embedding_inputs': embedding_inputs or {},
                'environment': environment, 'is_qualitative_only': True,
                'interpretation': 'Independent fits/axes; sample is not the full taxonomy; no quantitative or phylogenetic inference.'}
    _atomic(resume.output_dir / 'umap_comparison.json', lambda p: p.write_text(json.dumps(metadata, indent=2) + '\n'))
    return {'projections': frames, 'sample': sample, 'metadata': metadata}
