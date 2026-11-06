"""Paired visualization contracts; all fixtures run offline on CPU."""
import ast
import contextlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from viral_cnn.eval.latent import _sha256_file
from viral_cnn.latent_resume import LatentResumeCache
from viral_cnn.paired_umap import ProjectionConfig, select_paired_sample, run_paired_umap, MODEL_NAMES

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / 'notebooks/reproduction/short/06_ictv_latent_space_and_hierarchy.ipynb'


class FakeUMAP:
    calls = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def fit_transform(self, features):
        self.calls.append((self.kwargs, features.copy()))
        return np.asarray(features[:, :2], dtype=np.float32)


class SelectedOnlyFeatures:
    def __init__(self, values, maximum):
        self.values, self.maximum, self.shape = values, maximum, values.shape
        self.accesses = []

    def __array__(self, *args, **kwargs):
        raise AssertionError('Full feature materialization')

    def __getitem__(self, indices):
        if not isinstance(indices, np.ndarray) or len(indices) > self.maximum:
            raise AssertionError('Feature gather exceeds selected rows')
        self.accesses.append(indices.copy())
        return self.values[indices]


class PairedUMAPTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.models = [{'name': name, 'display': name, 'test_artifact_sha256': f'fixture-{name}'} for name in MODEL_NAMES]
        y = np.repeat(np.arange(4), [2, 8, 10, 2]).astype(np.int64)
        ids = np.asarray([f'part:{i:03d}' for i in range(len(y))])
        self.labels = {name: y.copy() for name in MODEL_NAMES}
        self.ids = {name: ids.copy() for name in MODEL_NAMES}
        self.features = {name: np.column_stack((np.arange(len(y)), np.arange(len(y)) + i, np.ones(len(y)))).astype(np.float32)
                         for i, name in enumerate(MODEL_NAMES)}
        self.genera = {0: 'Alpha', 1: 'Beta', 2: 'Gamma'}
        self.config = ProjectionConfig(max_samples=12, max_genera=3)
        FakeUMAP.calls = []
        self.fake = patch.dict(sys.modules, {'umap': types.SimpleNamespace(UMAP=FakeUMAP)})
        self.fake.start()
        self.addCleanup(self.fake.stop)

    def directory(self, run):
        p = self.root / 'runs' / run / '06_ictv_latent_space_and_hierarchy'
        p.mkdir(parents=True)
        return p

    def run_comparison(self, run='first', **kwargs):
        output = self.directory(run)
        cache = LatentResumeCache(output, **kwargs.pop('cache_options', {}))
        result = run_paired_umap(self.models, self.features, self.labels, self.ids, self.genera,
                                 cache, {'mode': 'synthetic', 'seed': 42}, config=kwargs.pop('config', self.config), **kwargs)
        return result, cache

    def sample(self):
        return select_paired_sample(self.labels, self.ids, self.genera, self.config)

    def test_configuration_boundary(self):
        first = next(''.join(c['source']) for c in json.loads(NOTEBOOK.read_text())['cells'] if c['cell_type'] == 'code')
        tree = ast.parse(first)
        for name, expected in (('LATENT_UMAP_MAX_SAMPLES', 100), ('LATENT_UMAP_MAX_GENERA', 10)):
            node = next(n for n in tree.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in n.targets))
            with patch.dict(os.environ, {}, clear=True):
                scope = {'os': os}
                exec(compile(ast.Module(body=[node], type_ignores=[]), 'first', 'exec'), scope)
            self.assertEqual(scope[name], expected)
        for name, value in (('LATENT_UMAP_MAX_SAMPLES', '2'), ('LATENT_UMAP_MAX_SAMPLES', '3.5'),
                            ('LATENT_UMAP_MAX_SAMPLES', 'bad'), ('LATENT_UMAP_MAX_GENERA', '0')):
            with self.subTest(name=name, value=value), patch.dict(os.environ, {name: value}), patch('viral_cnn.notebook_runtime.setup_notebook_runtime') as setup:
                with self.assertRaises(ValueError):
                    exec(first, {})
                setup.assert_not_called()
        for config in ({'max_samples': 2}, {'max_samples': 3.5}, {'max_genera': 0}, {'max_genera': True}):
            with self.assertRaises(ValueError):
                ProjectionConfig(**config)
        with patch.dict(os.environ, {'LATENT_UMAP_MAX_SAMPLES': '17', 'LATENT_UMAP_MAX_GENERA': '4'}):
            scope = {'os': os}
            for n in tree.body:
                if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id in ('LATENT_UMAP_MAX_SAMPLES', 'LATENT_UMAP_MAX_GENERA') for t in n.targets):
                    exec(compile(ast.Module(body=[n], type_ignores=[]), 'first', 'exec'), scope)
            self.assertEqual((scope['LATENT_UMAP_MAX_SAMPLES'], scope['LATENT_UMAP_MAX_GENERA']), (17, 4))

    def test_coverage_and_balanced_quotas(self):
        before = self.labels[MODEL_NAMES[0]].copy()
        sample = self.sample()
        self.assertEqual(sample.audit['population_available'], 22)
        self.assertEqual(sample.audit['eligible_count'], 20)
        self.assertEqual(sample.audit['excluded_missing_genus'], 2)
        self.assertEqual(sample.frame['Genus'].value_counts().to_dict(), {'Beta': 5, 'Gamma': 5, 'Alpha': 2})
        self.assertEqual(len(sample.frame), 12)
        self.assertTrue(sample.frame['row_id'].is_unique)
        pd.testing.assert_frame_equal(sample.frame, self.sample().frame)
        np.testing.assert_array_equal(self.labels[MODEL_NAMES[0]], before)
        limited = select_paired_sample(self.labels, self.ids, self.genera, ProjectionConfig(max_samples=7, max_genera=2))
        self.assertLessEqual(len(limited.frame), 7)
        self.assertLessEqual(limited.frame['Genus'].nunique(), 2)

    def test_permuted_models_project_identical_rows(self):
        original = self.sample().frame
        rng = np.random.default_rng(7)
        for name in MODEL_NAMES:
            order = rng.permutation(len(self.labels[name]))
            self.labels[name], self.ids[name], self.features[name] = self.labels[name][order], self.ids[name][order], self.features[name][order]
        sample = self.sample()
        pd.testing.assert_frame_equal(sample.frame, original)
        result, _ = self.run_comparison()
        self.assertEqual(set(result['projections']), set(MODEL_NAMES))
        self.assertEqual(len(FakeUMAP.calls), 5)
        for (name, frame), (_, features) in zip(result['projections'].items(), FakeUMAP.calls):
            self.assertEqual(frame['row_id'].tolist(), original['row_id'].tolist())
            self.assertEqual(frame['label'].tolist(), original['label'].tolist())
            expected_positions = sample.indices[name]
            np.testing.assert_array_equal(features, self.features[name][expected_positions])

    def test_identity_errors_before_projection(self):
        selected = self.sample().frame.iloc[0]['row_id']
        target = MODEL_NAMES[-1]
        original_ids, original_labels = self.ids[target].copy(), self.labels[target].copy()
        selected_index = int(np.flatnonzero(original_ids == selected)[0])
        for case in ('missing', 'duplicate', 'label'):
            self.ids[target], self.labels[target] = original_ids.copy(), original_labels.copy()
            if case == 'missing':
                self.ids[target][selected_index] = 'missing'
            elif case == 'duplicate':
                self.ids[target][(selected_index + 1) % len(original_ids)] = selected
            else:
                self.labels[target][selected_index] = 999
            with self.subTest(case=case), self.assertRaises(ValueError):
                self.run_comparison(case)
            self.assertEqual(FakeUMAP.calls, [])

    def test_projection_parameters_and_small_sample(self):
        for n in (5, 20):
            self.run_comparison(str(n), config=ProjectionConfig(max_samples=n, max_genera=3))
            self.assertEqual(len(FakeUMAP.calls[-5:]), 5)
            for params, features in FakeUMAP.calls[-5:]:
                self.assertEqual(params, {'n_neighbors': min(15, n - 1), 'min_dist': 0.1,
                                          'n_components': 2, 'metric': 'cosine', 'random_state': 42})
                self.assertEqual(len(features), n)

    def test_insufficient_sample(self):
        for name in MODEL_NAMES:
            self.ids[name], self.labels[name], self.features[name] = self.ids[name][:2], self.labels[name][:2], self.features[name][:2]
        with self.assertRaisesRegex(ValueError, 'at least 3'):
            self.run_comparison(config=ProjectionConfig(max_samples=3))
        self.assertEqual(FakeUMAP.calls, [])
        self.assertFalse(list(self.root.rglob('umap_comparison.json')))

    def test_selected_feature_access_and_quantitative_contracts(self):
        self.features = {name: SelectedOnlyFeatures(values, 12) for name, values in self.features.items()}
        self.run_comparison()
        self.assertTrue(all(len(values.accesses) == 1 for values in self.features.values()))
        code = '\n'.join(''.join(c['source']) for c in json.loads(NOTEBOOK.read_text())['cells'] if c['cell_type'] == 'code')
        self.assertIn('k=5', code)
        self.assertIn('200 if ctx.is_smoke_test else 5000', code)
        self.assertIn('features=raw_feats', code)

    def test_outputs_provenance_and_panel(self):
        from matplotlib.figure import Figure
        figures = []
        save = Figure.savefig
        def capture(fig, *args, **kwargs):
            figures.append(fig)
            return save(fig, *args, **kwargs)
        with patch.object(Figure, 'savefig', new=capture):
            result, cache = self.run_comparison()
        meta = json.loads((cache.output_dir / 'umap_comparison.json').read_text())
        self.assertEqual(meta['protocol'], 'c10_paired_umap_v1')
        self.assertEqual(meta['status'], 'COMPLETE')
        self.assertEqual(meta['selection']['selected_counts'], {'Alpha': 2, 'Beta': 5, 'Gamma': 5})
        self.assertEqual(len(meta['projections']), 5)
        for name, output in meta['projections'].items():
            self.assertEqual(_sha256_file(output['path']), output['sha256'])
            frame = pd.read_csv(output['path'], comment='#')
            self.assertTrue({'row_id', 'label', 'Genus', 'projection_method', 'is_qualitative_only'}.issubset(frame.columns))
            self.assertEqual(frame['row_id'].tolist(), result['sample'].frame['row_id'].tolist())
        for key in ('sample_file', 'panel_file'):
            self.assertEqual(_sha256_file(meta[key]['path']), meta[key]['sha256'])
        fig = figures[-1]
        self.assertEqual(len(fig.axes), 5)
        self.assertEqual([t.get_text() for t in fig.legends[0].texts], ['Alpha', 'Beta', 'Gamma'])
        self.assertTrue(any('QUALITATIVE' in t.get_text() and 'independent' in t.get_text().lower() for t in fig.texts))
        for ax in fig.axes:
            self.assertIn('UMAP', ax.get_title())
        for genus_index in range(3):
            colors = [ax.collections[genus_index].get_facecolors()[0] for ax in fig.axes]
            for color in colors[1:]:
                np.testing.assert_array_equal(color, colors[0])

    def test_pca_fallback_is_explicit(self):
        from matplotlib.figure import Figure
        figures = []
        save = Figure.savefig
        def capture(fig, *args, **kwargs):
            figures.append(fig)
            return save(fig, *args, **kwargs)
        with patch.dict(sys.modules, {'umap': None}), patch.object(Figure, 'savefig', new=capture):
            result, _ = self.run_comparison()
        self.assertEqual(result['metadata']['projection_method'], 'PCA')
        for frame in result['projections'].values():
            self.assertEqual(set(frame['projection_method']), {'PCA'})
        for ax in figures[-1].axes:
            self.assertIn('PCA', ax.get_title())
            self.assertIn('PCA', ax.get_xlabel())

    def test_resume_and_variant_corruption(self):
        first, cache = self.run_comparison()
        before = {p: p.read_bytes() for p in cache.output_dir.iterdir() if p.is_file()}
        FakeUMAP.calls = []
        second, _ = self.run_comparison('second', cache_options={'reuse_run_id': 'first'})
        self.assertEqual(FakeUMAP.calls, [])
        self.assertEqual(before, {p: p.read_bytes() for p in cache.output_dir.iterdir() if p.is_file()})
        chosen = MODEL_NAMES[2]
        path = Path(first['metadata']['projections'][chosen]['path'])
        path.write_text('corrupt projection')
        corrupted_before = {p: p.read_bytes() for p in cache.output_dir.iterdir() if p.is_file()}
        third, third_cache = self.run_comparison('third', cache_options={'reuse_run_id': 'first'})
        self.assertEqual(len(FakeUMAP.calls), 1)
        self.assertEqual(corrupted_before, {p: p.read_bytes() for p in cache.output_dir.iterdir() if p.is_file()})
        events = [e for e in third_cache.events if e['component'].startswith('umap_') and e['status'] == 'recalculated']
        self.assertEqual([e['component'] for e in events], [f'umap_{chosen}'])
        FakeUMAP.calls = []
        changed, _ = self.run_comparison('selection_changed', cache_options={'reuse_run_id': 'first'},
                                        config=ProjectionConfig(max_samples=11, max_genera=3))
        self.assertEqual(len(FakeUMAP.calls), 5)
        self.assertEqual(len(changed['sample'].frame), 11)
        # The corrupt variant still recalculates; a distinct embedding hash
        # invalidates another variant independently.
        FakeUMAP.calls = []
        self.models[0]['test_artifact_sha256'] = 'changed-embedding'
        self.run_comparison('embedding_changed', cache_options={'reuse_run_id': 'first'})
        self.assertEqual(len(FakeUMAP.calls), 2)
        from viral_cnn.eval import latent
        exporter = latent.export_umap_projection
        with patch.dict(sys.modules, {'umap': None}), patch.object(latent, 'export_umap_projection', wraps=exporter) as export:
            result, _ = self.run_comparison('backend_changed', cache_options={'reuse_run_id': 'first'})
        self.assertEqual(export.call_count, 5)
        self.assertEqual(result['metadata']['projection_method'], 'PCA')

    def test_interrupted_publication(self):
        real_replace = os.replace
        def interrupt(src, target):
            if Path(target).name == 'umap_comparison.json':
                raise OSError('interrupted comparison publication')
            return real_replace(src, target)
        with patch('viral_cnn.latent_resume.os.replace', side_effect=interrupt):
            with self.assertRaises(OSError):
                self.run_comparison()
        self.assertFalse(list(self.root.rglob('umap_comparison.json')))
        self.assertFalse(list(self.root.rglob('*.tmp')))
        FakeUMAP.calls = []
        self.run_comparison('resumed', cache_options={'reuse_run_id': 'first'})
        self.assertEqual(FakeUMAP.calls, [])
        def interrupt_csv(src, target):
            if Path(target).name == 'umap_projection_resnet_noise_species.csv':
                raise OSError('interrupted second variant CSV')
            return real_replace(src, target)
        with patch('viral_cnn.latent_resume.os.replace', side_effect=interrupt_csv):
            with self.assertRaises(OSError):
                self.run_comparison('partial_csv', cache_options={'mode': 'rerun'})
        partial = self.root / 'runs/partial_csv/06_ictv_latent_space_and_hierarchy'
        self.assertFalse((partial / 'umap_comparison.json').exists())
        self.assertEqual(len(list(partial.glob('stage_umap_*.json'))), 1)
        FakeUMAP.calls = []
        self.run_comparison('partial_resumed', cache_options={'reuse_run_id': 'partial_csv'})
        self.assertEqual(len(FakeUMAP.calls), 4)

    def test_umap_fit_import_failure_does_not_claim_umap_success(self):
        class FailingUMAP(FakeUMAP):
            def fit_transform(self, features):
                raise ImportError('missing fitting dependency')
        with patch.dict(sys.modules, {'umap': types.SimpleNamespace(UMAP=FailingUMAP)}):
            with self.assertRaises(ImportError):
                self.run_comparison()
        self.assertFalse(list(self.root.rglob('umap_comparison.json')))


class NotebookPairedUMAPTests(unittest.TestCase):
    def test_notebook_smoke_resume(self):
        from viral_cnn.notebook_runtime import reserve_run_directory
        nb = json.loads(NOTEBOOK.read_text())
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            def reserve(_root, run, name):
                return reserve_run_directory(root, run, name)
            env = {'REMOTE_SERVER': 'False', 'SMOKE_TEST': '1', 'DATA_ROOT': str(root), 'LATENT_EXECUTION_MODE': 'reuse',
                   'LATENT_REUSE_RUN_ID': '', 'LATENT_RAM_BUDGET_MB': '64', 'LATENT_UMAP_MAX_SAMPLES': '100', 'LATENT_UMAP_MAX_GENERA': '10'}
            with patch.dict(os.environ, env), patch('torch.cuda.is_available', return_value=False), patch('viral_cnn.notebook_runtime.reserve_run_directory', side_effect=reserve), contextlib.redirect_stdout(io.StringIO()):
                first = {'__name__': '__main__'}
                for c in nb['cells']:
                    if c['cell_type'] == 'code':
                        exec(''.join(c['source']), first)
                with patch('viral_cnn.eval.latent.export_umap_projection', side_effect=AssertionError('refitted projection')):
                    second = {'__name__': '__main__'}
                    for c in nb['cells']:
                        if c['cell_type'] == 'code':
                            exec(''.join(c['source']), second)
                meta = json.loads((second['output_dir'] / 'umap_comparison.json').read_text())
                self.assertEqual(len(meta['projections']), 5)
                self.assertTrue((second['output_dir'] / 'umap_comparison.png').is_file())
                manifest = json.loads((second['output_dir'] / 'manifest.json').read_text())
                self.assertEqual(manifest['configuration']['umap_protocol'], 'c10_paired_umap_v1')
                self.assertEqual(manifest['metrics_summary']['projection_variants'], 5)
                self.assertEqual(manifest['configuration']['umap_sample_size'], len(second['umap_comparison']['sample'].frame))

    def test_documented_limits(self):
        doc = (ROOT / 'docs/paired_umap.md').read_text()
        for token in ('LATENT_UMAP_MAX_SAMPLES', 'LATENT_UMAP_MAX_GENERA', 'c10_paired_umap_v1',
                      'primeiras 100', 'hash global', 'sintética', 'CPU', 'remota', 'PCA', 'independentes'):
            self.assertIn(token, doc)


if __name__ == '__main__':
    unittest.main()
