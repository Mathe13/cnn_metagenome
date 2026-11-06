"""Scientific contracts of test-only visualization over archived embeddings."""
import contextlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from viral_cnn.paired_umap import MODEL_NAMES
from viral_cnn.visualization_cache import (
    create_visualization_fixtures, open_test_embeddings, load_visualization_taxonomy,
)

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / 'notebooks/reproduction/short/06b_latent_visualizations.ipynb'


class ArchivedVisualizationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        with contextlib.redirect_stdout(io.StringIO()):
            self.source, self.tax = create_visualization_fixtures(Path(self.tmp.name) / 'source')

    def open(self):
        with contextlib.redirect_stdout(io.StringIO()):
            return open_test_embeddings(self.source, smoke=True, batch_size=3)

    def test_selective_read_preserves_rows_labels_and_order(self):
        opened = self.open()
        self.assertEqual(set(opened), set(MODEL_NAMES))
        positions = np.array([35, 1, 1, 34, 0])
        for i, name in enumerate(MODEL_NAMES):
            item = opened[name]
            self.assertEqual(item.features.stats['row_groups_decoded'], [])
            self.assertEqual(item.row_ids[35], 'synthetic-test:35')
            self.assertEqual(item.row_ids[positions].tolist(), [f'synthetic-test:{p}' for p in positions])
            np.testing.assert_array_equal(item.labels, np.arange(48) % 6)
            expected = np.random.default_rng(42 + i).normal(size=(48, 8)).astype(np.float32)
            with contextlib.redirect_stdout(io.StringIO()):
                np.testing.assert_array_equal(item.features[positions], expected[positions])
            self.assertEqual(item.features.stats['row_groups_decoded'], [0, 4])
            self.assertEqual(item.features.stats['selected_rows'], 5)
            with self.assertRaisesRegex(ValueError, 'Full feature'):
                np.asarray(item.features)

    def test_corrupt_bytes_rejected_before_parquet_decode(self):
        path = next(self.source.glob(f'features_{MODEL_NAMES[0]}*.parquet'))
        with path.open('r+b') as f:
            f.seek(12)
            f.write(b'bad')
        with patch('viral_cnn.visualization_cache.pq.ParquetFile', side_effect=AssertionError('decoded corrupt input')):
            with self.assertRaisesRegex(ValueError, 'Corrupt'):
                self.open()

    def test_metadata_read_projects_columns_and_hashes_each_test_once(self):
        from viral_cnn.eval.latent import _sha256_file
        columns = []
        constructor = pq.ParquetFile

        class TrackedParquet:
            def __init__(self, path):
                self.original = constructor(path)

            def __getattr__(self, name):
                return getattr(self.original, name)

            def iter_batches(self, **kwargs):
                columns.append(kwargs['columns'])
                return self.original.iter_batches(**kwargs)

        with patch('viral_cnn.visualization_cache.pq.ParquetFile', TrackedParquet), \
             patch('viral_cnn.visualization_cache._sha256_file', wraps=_sha256_file) as digest:
            self.open()
        self.assertEqual(columns, [['label', 'row_id']] * 5)
        parquet_calls = [Path(c.args[0]) for c in digest.call_args_list if Path(c.args[0]).suffix == '.parquet']
        self.assertEqual(len(parquet_calls), 5)
        self.assertEqual(len(set(parquet_calls)), 5)

    def test_manifest_tampering_rejected(self):
        path = next(self.source.glob(f'features_{MODEL_NAMES[0]}*.json'))
        meta = json.loads(path.read_text())
        meta['class_order'] = list(reversed(meta['class_order']))
        path.write_text(json.dumps(meta))
        with self.assertRaisesRegex(ValueError, 'fingerprint'):
            self.open()

    def test_ambiguous_artifact_rejected(self):
        path = next(self.source.glob(f'features_{MODEL_NAMES[0]}*.json'))
        path.with_name(f'features_{MODEL_NAMES[0]}_test_ambiguous.json').write_text(path.read_text())
        with self.assertRaisesRegex(ValueError, 'exactly one'):
            self.open()

    def test_source_mutation_after_verification_rejected(self):
        item = self.open()[MODEL_NAMES[0]]
        with item.path.open('ab') as f:
            f.write(b'mutated')
        with self.assertRaisesRegex(ValueError, 'changed after checksum'):
            item.features[np.array([1, 4])]

    def test_taxonomy_must_match_archived_label_contract(self):
        _, mapping = load_visualization_taxonomy(self.tax, list(range(6)))
        self.assertEqual(mapping[0], 'SyntheticGenus_0')
        with self.assertRaisesRegex(ValueError, 'class order'):
            load_visualization_taxonomy(self.tax, list(range(5)))
        table = pd.read_parquet(self.tax)
        table['model_label'] += 1
        table.to_parquet(self.tax, index=False)
        with self.assertRaisesRegex(ValueError, 'id - 1'):
            load_visualization_taxonomy(self.tax, list(range(6)))


class VisualizationNotebookTests(unittest.TestCase):
    def execute(self):
        scope = {'__name__': '__main__'}
        for cell in json.loads(NOTEBOOK.read_text())['cells']:
            if cell['cell_type'] == 'code':
                exec(compile(''.join(cell['source']), str(NOTEBOOK), 'exec'), scope)
        return scope

    def test_smoke_five_projections_and_validated_resume_without_models(self):
        from viral_cnn.notebook_runtime import reserve_run_directory
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            def reserve(_root, run, name):
                return reserve_run_directory(root, run, name)
            env = {'REMOTE_SERVER': 'False', 'SMOKE_TEST': '1', 'DATA_ROOT': str(root),
                   'LATENT_VIS_MAX_GENERA': 'all', 'LATENT_VIS_MAX_SAMPLES': 'auto',
                   'LATENT_VIS_EXECUTION_MODE': 'reuse', 'LATENT_VIS_REUSE_RUN_ID': '',
                   'LATENT_VIS_PROJECTION_SOURCE_DIR': ''}
            with patch.dict(os.environ, env), patch('torch.cuda.is_available', return_value=False), \
                 patch('viral_cnn.notebook_runtime.reserve_run_directory', side_effect=reserve), \
                 patch('viral_cnn.eval.latent.extract_latent_embeddings', side_effect=AssertionError('extraction forbidden')), \
                 patch('viral_cnn.eval.latent.LinearProbeClassifier.fit', side_effect=AssertionError('training forbidden')), \
                 contextlib.redirect_stdout(io.StringIO()):
                first = self.execute()
                with patch('viral_cnn.visualization_cache.SelectedParquetFeatures.__getitem__', side_effect=AssertionError('decoded features on reuse')), \
                     patch('viral_cnn.eval.latent.export_umap_projection', side_effect=AssertionError('refit on reuse')):
                    second = self.execute()
                with patch.dict(os.environ, {'LATENT_VIS_PROJECTION_SOURCE_DIR': str(first['output_dir'])}), \
                     patch('viral_cnn.visualization_cache.open_test_embeddings', side_effect=AssertionError('opened embeddings on redraw')), \
                     patch('viral_cnn.paired_umap.run_paired_umap', side_effect=AssertionError('refit on redraw')), \
                     patch('viral_cnn.visualization_cache.create_visualization_fixtures', side_effect=AssertionError('created embeddings on redraw')):
                    third = self.execute()
            for scope in (first, second, third):
                manifest = json.loads((scope['output_dir'] / 'manifest.json').read_text())
                self.assertEqual(manifest['metrics_summary']['projection_variants'], 5)
                self.assertEqual(manifest['metrics_summary']['models_loaded'], 0)
                self.assertFalse(manifest['metrics_summary']['train_validation_loaded'])
                self.assertEqual(manifest['evidence']['status'], 'synthetic')
                self.assertEqual(len(scope['comparison']['sample'].frame), 30)
                self.assertEqual(scope['comparison']['sample'].frame['Genus'].nunique(), 3)
                self.assertTrue((scope['output_dir'] / 'umap_comparison.png').is_file())
                self.assertTrue((scope['output_dir'] / 'umap_comparison.html').is_file())
                self.assertEqual(len(list(scope['output_dir'].glob('umap_projection*.csv'))), 5)
                self.assertFalse(any('_train_' in p['path'] or '_validation_' in p['path'] for p in manifest['inputs']))
            self.assertTrue(all(x.features.stats['selected_rows'] == 30 for x in first['embeddings'].values()))
            self.assertTrue(all(x.features.stats['selected_rows'] == 0 for x in second['embeddings'].values()))
            self.assertEqual(third['embeddings'], {})
            for name in MODEL_NAMES:
                source = Path(first['comparison']['metadata']['projections'][name]['path'])
                copied = Path(third['comparison']['metadata']['projections'][name]['path'])
                self.assertEqual(source.read_bytes(), copied.read_bytes())


if __name__ == '__main__':
    unittest.main()
