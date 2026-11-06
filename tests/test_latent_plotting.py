"""Coordinate rendering preserves paired scientific identities and population."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from viral_cnn.eval.latent import _sha256_file
from viral_cnn.latent_plotting import load_saved_comparison, render_comparison
from viral_cnn.paired_umap import MODEL_NAMES


class SavedCoordinateTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.source = Path(self.tmp.name) / 'source'
        self.output = Path(self.tmp.name) / 'output'
        self.source.mkdir()
        self.base = pd.DataFrame({'row_id': [f'row:{i}' for i in range(48)], 'label': np.arange(48),
                                  'Genus': [f'Genus_{i // 2:02d}' for i in range(48)]})
        self.metadata = {'status': 'COMPLETE', 'protocol': 'c10_paired_umap_v1', 'projection_method': 'UMAP',
                         'selection': {'selected_count': 48, 'configuration': {'max_samples': 48, 'max_genera': 24, 'seed': 42}},
                         'selection_sha256': hashlib.sha256(json.dumps(self.base.to_dict(orient='records'), sort_keys=True).encode()).hexdigest(),
                         'projections': {}}
        for i, name in enumerate(MODEL_NAMES):
            frame = self.base.assign(umap_1=np.arange(48) / 7 + i, umap_2=np.arange(48)[::-1] / 11,
                                     projection_method='UMAP', is_qualitative_only=True)
            filename = 'umap_projection_convnext_f.csv' if name == MODEL_NAMES[-1] else f'umap_projection_{name}.csv'
            path = self.source / filename
            frame.to_csv(path, index=False)
            self.metadata['projections'][name] = {'path': str(path), 'sha256': _sha256_file(path)}
        self.publish()

    def publish(self):
        (self.source / 'umap_comparison.json').write_text(json.dumps(self.metadata))

    def test_redraw_preserves_all_coordinate_csvs_and_identity(self):
        comparison, inputs = load_saved_comparison(self.source, self.output)
        before = {name: frame.copy() for name, frame in comparison['projections'].items()}
        rendering = render_comparison(comparison, self.output)
        self.assertEqual(len(inputs), 6)
        self.assertEqual(rendering['rows_displayed_per_model'], 48)
        self.assertFalse(rendering['coordinates_changed'])
        self.assertEqual(rendering['legend'], 'external searchable')
        colors = pd.read_csv(self.output / 'umap_genus_colors.csv')
        self.assertEqual(colors.selected_count.sum(), 48)
        for name in MODEL_NAMES:
            pd.testing.assert_frame_equal(before[name], comparison['projections'][name])
            filename = Path(self.metadata['projections'][name]['path']).name
            self.assertEqual((self.source / filename).read_bytes(), (self.output / filename).read_bytes())
        html = (self.output / 'umap_comparison.html').read_text()
        data = json.loads(html.split('<script id="data" type="application/json">')[1].split('</script>')[0])
        self.assertEqual(data['ids'], self.base.row_id.tolist())
        self.assertEqual(sorted(data['order']), list(range(48)))
        self.assertEqual([len(model['xy']) for model in data['models']], [48] * 5)
        self.assertEqual(len(data['genera']), 24)

    def test_corrupt_coordinate_csv_is_rejected(self):
        path = Path(self.metadata['projections'][MODEL_NAMES[-1]]['path'])
        path.write_text(path.read_text() + '\ncorrupt')
        with self.assertRaisesRegex(ValueError, 'Corrupt saved projection'):
            load_saved_comparison(self.source, self.output)
        self.assertFalse(self.output.exists())

    def test_valid_checksum_with_misaligned_labels_is_rejected(self):
        record = self.metadata['projections'][MODEL_NAMES[-1]]
        path = Path(record['path'])
        frame = pd.read_csv(path)
        frame.loc[0, 'label'] = 99
        frame.to_csv(path, index=False)
        record['sha256'] = _sha256_file(path)
        self.publish()
        with self.assertRaisesRegex(ValueError, 'differ between models'):
            load_saved_comparison(self.source, self.output)
        self.assertFalse(self.output.exists())


if __name__ == '__main__':
    unittest.main()
