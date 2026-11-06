"""Scientific provenance, interruption and restart contracts for notebook 06."""
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
import torch

from viral_cnn.eval.latent import LatentMemoryBudget, save_embedding_artifact, _sha256_file
from viral_cnn.latent_resume import (
    LatentResumeCache, StageResult, snapshot_probe, restore_probe, infer_native_predictions,
    notebook_code_sha256,
)
from viral_cnn.eval.latent import LinearProbeClassifier
from tests.test_c01_latent_audit import artifact

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / 'notebooks/reproduction/short/06_ictv_latent_space_and_hierarchy.ipynb'


class ResumeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.runs = Path(self.tmp.name) / 'runs'
        self.old = self.run_dir('old')
        self.new = self.run_dir('new')
        self.a = artifact('train', ['a:0', 'a:1', 'a:2', 'a:3'])
        self.a.manifest.update(population_available=4, extracted_count=4)

    def run_dir(self, run):
        path = self.runs / run / '06_ictv_latent_space_and_hierarchy'
        path.mkdir(parents=True)
        return path

    def cache(self, output=None, **kw):
        return LatentResumeCache(output or self.new, **kw)

    def save(self, directory=None, value=None):
        return save_embedding_artifact(value or self.a, (directory or self.old) / 'features')

    def test_modes_and_sources(self):
        import ast
        cell = next(''.join(c['source']) for c in json.loads(NOTEBOOK.read_text())['cells']
                    if c['cell_type'] == 'code')
        assignment = next(n for n in ast.parse(cell).body if isinstance(n, ast.Assign)
                          and any(isinstance(t, ast.Name) and t.id == 'LATENT_EXECUTION_MODE' for t in n.targets))
        scope = {'os': os}
        with patch.dict(os.environ, {}, clear=True):
            exec(compile(ast.Module(body=[assignment], type_ignores=[]), 'first_cell', 'exec'), scope)
        self.assertEqual(scope['LATENT_EXECUTION_MODE'], 'reuse')
        with patch.dict(os.environ, {'LATENT_EXECUTION_MODE': 'invalid'}):
            with patch('viral_cnn.notebook_runtime.setup_notebook_runtime') as setup:
                with patch('viral_cnn.eval.latent.extract_latent_embeddings') as extract:
                    with self.assertRaisesRegex(ValueError, 'LATENT_EXECUTION_MODE'):
                        exec(cell, {})
                extract.assert_not_called()
            setup.assert_not_called()
        source = self.run_dir('recent')
        os.utime(self.old, (10, 10))
        os.utime(source, (20, 20))
        self.assertEqual(self.cache().mode, 'reuse')
        self.assertEqual(self.cache().sources, [source, self.old])
        self.assertEqual(self.cache(reuse_run_id='old').sources, [self.old])
        for mode in ('bad', '', True):
            with self.assertRaises(ValueError):
                self.cache(mode=mode)
        for run in ('missing', '../old', '/tmp', '..'):
            with self.assertRaises((ValueError, FileNotFoundError)):
                self.cache(reuse_run_id=run)

    def test_rerun_and_isolation(self):
        path = self.save()
        before = {p: p.read_bytes() for p in self.old.iterdir()}
        producer = self.cache(self.old, mode='rerun')
        producer.run_stage('unit', {'seed': 42}, lambda: {'value': 8})
        before = {p: p.read_bytes() for p in self.old.iterdir()}
        rerun = self.cache(mode='rerun')
        self.assertIsNone(rerun.load_embedding(self.a.manifest, 4))
        self.assertEqual(rerun.run_stage('unit', {'seed': 42}, lambda: {'value': 9}), {'value': 9})
        other = self.run_dir('other')
        c = self.cache(other, reuse_run_id='old')
        with patch('builtins.print'):
            self.assertEqual(c.run_stage('unit', {'seed': 42}, lambda: self.fail('recomputed')), {'value': 8})
            self.assertIsNotNone(c.load_embedding(self.a.manifest, 4))
        self.assertEqual(before, {p: p.read_bytes() for p in self.old.iterdir()})
        self.assertNotEqual(other, self.new)
        self.assertTrue(path.is_file())

    def test_embedding_identity_and_integrity(self):
        path = self.save()
        fields = {'variant': 'other', 'split': 'test', 'partitions': ['other'],
                  'partition_sha256': ['bad'], 'checkpoint': 'other.pt', 'checkpoint_sha256': 'bad',
                  'class_order': [1, 0], 'encoding': {'channels': ['T', 'G', 'C', 'A']},
                  'sampling': {'seed': 9}, 'mode': 'empirical', 'extractor_code_sha256': 'other'}
        for key, value in fields.items():
            with self.subTest(key=key):
                expected = dict(self.a.manifest, **{key: value})
                self.assertIsNone(self.cache().load_embedding(expected, 4))
        original = path.read_bytes()
        path.write_bytes(original[:-8] + b'corrupt!')
        self.assertIsNone(self.cache().load_embedding(self.a.manifest, 4))
        path.write_bytes(original)
        meta = path.with_suffix('.json')
        content = json.loads(meta.read_text())
        content['fingerprint'] = 'bad'
        meta.write_text(json.dumps(content))
        self.assertIsNone(self.cache().load_embedding(self.a.manifest, 4))

    def test_embedding_fallback_and_shape(self):
        good = self.save()
        recent = self.run_dir('recent')
        os.utime(self.old, (10, 10))
        corrupt = self.save(recent)
        corrupt.with_suffix('.json').unlink()
        cache = self.cache()
        result = cache.load_embedding(self.a.manifest, 4)
        self.assertEqual(result[1], good)
        self.assertTrue(any(e['status'] == 'discarded' for e in cache.events))
        self.assertIsNone(self.cache().load_embedding(self.a.manifest, 5))
        wrong_count = dict(self.a.manifest, extracted_count=5)
        self.assertIsNone(self.cache().load_embedding(wrong_count, 4))
        # A checksum-valid artifact whose manifest lies about its count is also invalid.
        broken = artifact('train', ['b:0', 'b:1'])
        broken.manifest.update(population_available=4, extracted_count=4)
        good.unlink()
        good.with_suffix('.json').unlink()
        self.save(self.old, broken)
        self.assertIsNone(self.cache().load_embedding(broken.manifest, 4))

    def test_embedding_budget_and_no_extraction(self):
        self.save()
        manager = LatentMemoryBudget(0, Path(self.tmp.name) / 'spill', observe_system_memory=False)
        self.addCleanup(manager.close)
        with patch('viral_cnn.eval.latent.extract_latent_embeddings', side_effect=AssertionError('extracted')):
            loaded, _ = self.cache().load_embedding(self.a.manifest, 4, memory_manager=manager, batch_size=2)
        self.assertIsInstance(loaded.features, np.memmap)
        np.testing.assert_array_equal(loaded.features, self.a.features)

    def test_stage_atomic_and_integrity(self):
        extra = self.old / 'neighbors.parquet'
        extra.write_bytes(b'identified neighbors')
        cache = self.cache(self.old, mode='rerun')
        value = {'frame': pd.DataFrame({'label': [0, 1], 'score': [0.2, np.nan]}),
                 'array': np.array([1, 2]), 'tensor': torch.tensor([3.0])}
        cache.run_stage('knn', {'k': 5}, lambda: StageResult(value, [extra]))
        restored = self.cache().run_stage('knn', {'k': 5}, lambda: self.fail('recomputed'))
        pd.testing.assert_frame_equal(restored['frame'], value['frame'])
        np.testing.assert_array_equal(restored['array'], [1, 2])
        extra.write_bytes(b'corrupt')
        self.assertEqual(self.cache().run_stage('knn', {'k': 5}, lambda: 42), 42)
        fresh = self.run_dir('failed')
        with patch('viral_cnn.latent_resume.os.replace', side_effect=OSError('disk failure')):
            with self.assertRaises(OSError):
                self.cache(fresh).run_stage('partial', {}, lambda: 10)
        self.assertFalse(list(fresh.glob('stage_*.json')))
        self.assertFalse(list(fresh.glob('*.tmp')))
        data = next(self.old.glob('stage_*.pt'))
        data.write_bytes(b'broken weights')
        self.assertEqual(self.cache(self.run_dir('repair'), reuse_run_id='old').run_stage('knn', {'k': 5}, lambda: 11), 11)

    def test_stage_dependency_invalidation(self):
        c = self.cache(self.old, mode='rerun')
        c.run_stage('knn', {'embedding': 'a', 'taxonomy': 't'}, lambda: 1)
        c.run_stage('umap', {'embedding': 'a'}, lambda: 2)
        reader = self.cache()
        self.assertEqual(reader.run_stage('knn', {'embedding': 'a', 'taxonomy': 'changed'}, lambda: 3), 3)
        self.assertEqual(reader.run_stage('umap', {'embedding': 'a'}, lambda: self.fail('recomputed')), 2)
        for changed in ({'embedding': 'b'}, {'embedding': 'a', 'seed': 99}):
            self.assertEqual(self.cache(self.run_dir(str(len(list(self.runs.iterdir()))))).run_stage('umap', changed, lambda: 4), 4)

    def test_sealed_probe_roundtrip(self):
        train = self.a
        validation = artifact('validation', ['v:0', 'v:1', 'v:2', 'v:3'])
        probe = LinearProbeClassifier(4, 2)
        selection = probe.fit_select_and_seal(train, validation, epochs=1, batch_size=4)
        payload = snapshot_probe(probe, selection)
        restored, selected = restore_probe(payload, train, validation)
        self.assertEqual(restored._sealed_sha256, selection['weights_sha256'])
        self.assertEqual(selected, selection)
        test = artifact('test', ['t:0', 't:1'])
        restored.evaluate_artifact(test)
        with self.assertRaises(RuntimeError):
            restored.evaluate_artifact(test)
        restored, _ = restore_probe(payload, train, validation)
        test.manifest['partition_sha256'] = train.manifest['partition_sha256']
        with self.assertRaises(ValueError):
            restored.evaluate_artifact(test)
        tampered = snapshot_probe(probe, selection)
        tampered['weights']['fc.bias'] += 1
        with self.assertRaises(ValueError):
            restore_probe(tampered, train, validation)
        with self.assertRaises(ValueError):
            restore_probe(payload, validation, train)

    def test_native_predictions_identity(self):
        model = torch.nn.Linear(4, 2)
        test = artifact('test', ['t:0', 't:1', 't:2', 't:3'])
        batches = [(torch.from_numpy(test.features[:2]), torch.from_numpy(test.labels[:2]), test.row_ids[:2]),
                   (torch.from_numpy(test.features[2:]), torch.from_numpy(test.labels[2:]), test.row_ids[2:])]
        pred = infer_native_predictions(model, batches, test, device='cpu')
        np.testing.assert_array_equal(pred, model(torch.from_numpy(test.features)).argmax(1).detach().numpy())
        batches[0] = (batches[0][0], batches[0][1], ['wrong', 't:1'])
        with self.assertRaises(ValueError):
            infer_native_predictions(model, batches, test, device='cpu')

    def test_legacy_stage_not_admitted(self):
        (self.old / 'umap_projection_convnext_f.csv').write_text('x,y\n1,2\n')
        cache = self.cache()
        self.assertEqual(cache.run_stage('umap', {}, lambda: 8), 8)
        self.assertTrue(any('legacy' in e.get('reason', '').lower() for e in cache.events))

    def test_stage_interrupted_manifest_and_operational_failure(self):
        cache = self.cache(self.old, mode='rerun')
        real_replace = os.replace
        def interrupt(source, target):
            if Path(target).name.startswith('stage_') and Path(target).suffix == '.json':
                raise OSError('interrupted before completion marker')
            return real_replace(source, target)
        with patch('viral_cnn.latent_resume.os.replace', side_effect=interrupt):
            with self.assertRaises(OSError):
                cache.run_stage('interrupted', {}, lambda: 10)
        self.assertTrue(list(self.old.glob('stage_*.pt')))
        self.assertFalse(list(self.old.glob('stage_*.json')))
        self.assertFalse(list(self.old.glob('*.tmp')))
        self.assertEqual(self.cache().run_stage('interrupted', {}, lambda: 11), 11)
        self.save()
        with patch('viral_cnn.latent_resume.load_embedding_artifact', side_effect=PermissionError('access denied')):
            with self.assertRaises(PermissionError):
                self.cache().load_embedding(self.a.manifest, 4)

    def test_explicit_transitive_reuse_and_probe_initialization(self):
        self.save()
        self.cache(self.old, mode='rerun').run_stage('unit', {}, lambda: 21)
        c = self.cache(reuse_run_id='old')
        c.load_embedding(self.a.manifest, 4)
        c.run_stage('unit', {}, lambda: self.fail('recomputed'))
        third = self.cache(self.run_dir('third'), reuse_run_id='new')
        self.assertIsNotNone(third.load_embedding(self.a.manifest, 4))
        self.assertEqual(third.run_stage('unit', {}, lambda: self.fail('recomputed')), 21)
        # Restore validation cannot change the RNG used by later fresh probes.
        probe = LinearProbeClassifier(4, 2)
        val = artifact('validation', ['v:0', 'v:1'])
        selected = probe.fit_select_and_seal(self.a, val, epochs=1)
        saved = snapshot_probe(probe, selected)
        before = torch.get_rng_state().clone()
        restore_probe(saved, self.a, val)
        torch.testing.assert_close(before, torch.get_rng_state())

    def test_initialization_protocol_is_explicit(self):
        import ast
        notebook = json.loads(NOTEBOOK.read_text())
        code = '\n'.join(''.join(c['source']) for c in notebook['cells'] if c['cell_type'] == 'code')
        tree = ast.parse(code)
        fit = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == 'fit_probe')
        self.assertEqual(ast.unparse(fit.body[0]), 'torch.manual_seed(ctx.seed)')
        self.assertIn('c09_seeded_probe_init_v1', code)
        self.assertIn('ctx.run_id = "c09-"', code)

    def test_editing_reuse_selectors_does_not_change_scientific_identity(self):
        def notebook(code):
            path = Path(self.tmp.name) / 'example.ipynb'
            path.write_text(json.dumps({'cells': [{'cell_type': 'code', 'source': code}]}))
            return notebook_code_sha256(path)
        before = notebook('LATENT_EXECUTION_MODE="reuse"\nLATENT_REUSE_RUN_ID=None\nk=5')
        self.assertEqual(before, notebook('LATENT_EXECUTION_MODE="rerun"\nLATENT_REUSE_RUN_ID="old"\nk=5'))
        self.assertNotEqual(before, notebook('LATENT_EXECUTION_MODE="reuse"\nLATENT_REUSE_RUN_ID=None\nk=4'))

    def test_malformed_json_and_completion_record_fingerprint(self):
        path = self.save()
        path.with_suffix('.json').write_text('[]')
        self.assertIsNone(self.cache().load_embedding(self.a.manifest, 4))
        producer = self.cache(self.old, mode='rerun')
        producer.run_stage('unit', {}, lambda: 4)
        record = next(self.old.glob('stage_unit_*.json'))
        original = record.read_text()
        for content in ('[]', '{broken', json.dumps(dict(json.loads(original), files=[{'path': 'bad'}]))):
            record.write_text(content)
            # Restrict to the malformed source: earlier iterations create valid fallbacks.
            reader = self.cache(self.run_dir(str(len(list(self.runs.iterdir())))), reuse_run_id='old')
            self.assertEqual(reader.run_stage('unit', {}, lambda: 7), 7)
            self.assertTrue(any(e['status'] == 'discarded' for e in reader.events))


class NotebookResumeTests(unittest.TestCase):
    def test_smoke_resume_and_manifest(self):
        from viral_cnn.notebook_runtime import reserve_run_directory
        notebook = json.loads(NOTEBOOK.read_text())
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            def reserve(_root, run, name):
                return reserve_run_directory(root, run, name)
            env = {'REMOTE_SERVER': 'False', 'SMOKE_TEST': '1', 'DATA_ROOT': str(root),
                   'LATENT_EXECUTION_MODE': 'reuse', 'LATENT_REUSE_RUN_ID': '', 'LATENT_RAM_BUDGET_MB': '64'}
            with patch.dict(os.environ, env), patch('viral_cnn.notebook_runtime.reserve_run_directory', side_effect=reserve), contextlib.redirect_stdout(io.StringIO()):
                first = {'__name__': '__main__'}
                for cell in notebook['cells']:
                    if cell['cell_type'] == 'code':
                        exec(''.join(cell['source']), first)
                first_dir = first['output_dir']
                before = {p: _sha256_file(p) for p in first_dir.iterdir() if p.is_file()}
                blocked = ['extract_latent_embeddings', 'evaluate_cosine_knn', 'evaluate_silhouette_with_audit',
                           'export_umap_projection']
                with contextlib.ExitStack() as stack:
                    for name in blocked:
                        stack.enter_context(patch('viral_cnn.eval.latent.' + name, side_effect=AssertionError('recomputed ' + name)))
                    stack.enter_context(patch.object(LinearProbeClassifier, 'fit_select_and_seal', side_effect=AssertionError('refitted probe')))
                    stack.enter_context(patch.object(LinearProbeClassifier, 'evaluate_artifact', side_effect=AssertionError('reevaluated probe')))
                    stack.enter_context(patch('viral_cnn.latent_resume.infer_native_predictions', side_effect=AssertionError('native inference')))
                    second = {'__name__': '__main__'}
                    for cell in notebook['cells']:
                        if cell['cell_type'] == 'code':
                            exec(''.join(cell['source']), second)
                self.assertNotEqual(first_dir, second['output_dir'])
                self.assertEqual(before, {p: _sha256_file(p) for p in first_dir.iterdir() if p.is_file()})
                pd.testing.assert_frame_equal(first['table5_df'], second['table5_df'])
                pd.testing.assert_frame_equal(first['probe_df'], second['probe_df'])
                csv = (second['output_dir'] / 'umap_projection_convnext_f.csv').read_text()
                self.assertTrue(csv.startswith('# QUALITATIVE ONLY:'))
                self.assertIn('# Do not infer quantitative cluster separation or phylogenetic distance.', csv)
                manifest = json.loads((second['output_dir'] / 'manifest.json').read_text())
                self.assertEqual(manifest['configuration']['latent_execution_mode'], 'reuse')
                report = json.loads((second['output_dir'] / 'resume_report.json').read_text())
                reused = [e for e in report['events'] if e['status'] == 'reused']
                self.assertEqual(sum(e['component'].startswith('embedding:') for e in reused), 15)
                self.assertTrue(any('knn_' in e['component'] for e in reused))
                self.assertTrue(all(e.get('sha256') for e in reused))

    def test_documented_limits(self):
        doc = (ROOT / 'docs/latent_resume.md').read_text()
        for token in ('LATENT_EXECUTION_MODE', 'LATENT_REUSE_RUN_ID', 'reuse', 'rerun', 'legados', 'sintéticos', 'remota', '5.000', '100'):
            self.assertIn(token, doc)


if __name__ == '__main__':
    unittest.main()
