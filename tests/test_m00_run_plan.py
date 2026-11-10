"""Proofs derived from the approved M00 obligations; synthetic/offline only."""
import copy
import csv
import hashlib
import json
import os
import subprocess
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import yaml

from viral_cnn.notebook_runtime import reserve_run_directory
from viral_cnn.run_planning import build_run_plan, inspect_run_stage, inventory_reference

ROOT = Path(__file__).resolve().parents[1]


class RunPlanTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.config = {"reference_roots": ["reference"], "stages": [
            {"notebook_id": "short/07", "manifest_name": "manifest_07_motifs.json", "requirements": []},
            {"notebook_id": "short/08", "requirements": [], "dependencies": [
                {"source": "short/07", "kind": "planned"}]}]}
        (self.root / 'reference').mkdir()
        (self.root / 'reference' / 'keep.txt').write_text('historical')

    def build(self, run='round-a', config=None, smoke=False):
        path = self.root / 'config.yaml'
        path.write_text(yaml.safe_dump(config or self.config))
        return build_run_plan(path, self.root, data_root=self.root, run_id=run, smoke_test=smoke)

    def cli(self, script, *args):
        return subprocess.run([str(ROOT / '.venv/bin/python3'), str(ROOT / 'scripts' / script), *map(str,args)],
                              cwd=ROOT, env={**os.environ, 'REMOTE_SERVER': 'False', 'LOG_PATH':'0'},
                              text=True, capture_output=True)

    def test_inventory_records(self):
        ref = self.root/'reference'
        records = inventory_reference([ref, ref/'keep.txt', self.root/'missing'])
        self.assertEqual([r['path'] for r in records], sorted(set(r['path'] for r in records)))
        row = next(r for r in records if r['path'].endswith('keep.txt'))
        self.assertEqual(row['size_bytes'], 10)
        self.assertEqual(row['origin_status'], 'present')
        self.assertEqual(row['hash_status'], 'hashed')

    def test_inventory_hash_boundary(self):
        p = self.root/'exact'; p.write_bytes(b'abcd')
        rows = inventory_reference([p], hash_max_bytes=4)
        self.assertEqual(rows[0]['sha256'], hashlib.sha256(b'abcd').hexdigest())
        self.assertEqual(inventory_reference([p], hash_max_bytes=3)[0]['hash_status'], 'skipped_size')

    def test_inventory_exceptions(self):
        big=self.root/'big'; big.write_bytes(b'12345')
        bad=self.root/'bad'; bad.write_text('secret')
        link=self.root/'link'; link.symlink_to(self.root/'reference', target_is_directory=True)
        original=Path.open
        def fail(path, *a, **kw):
            if path == bad: raise PermissionError('denied')
            return original(path,*a,**kw)
        with patch.object(Path,'open',fail):
            rows=inventory_reference([big,bad,link,self.root/'absent'],hash_max_bytes=4)
        self.assertEqual({r['hash_status'] for r in rows}, {'skipped_size','unreadable','symlink','missing'})
        self.assertTrue(all(r['sha256']=='unknown' for r in rows))
        self.assertFalse(any('keep.txt' in r['path'] for r in rows))
        self.assertEqual(big.read_bytes(),b'12345')

    def test_exclusive_rounds(self):
        first=self.build(); second=self.build('round-b')
        self.assertNotEqual(first['stages'][0]['output_dir'],second['stages'][0]['output_dir'])
        self.assertEqual((self.root/'reference/keep.txt').read_text(),'historical')
        for plan in (first,second):
            self.assertTrue(Path(plan['plan_path']).is_file())
            with open(Path(plan['plan_path']).parent/'reference_inventory.csv') as f:
                self.assertTrue(list(csv.DictReader(f)))

    def test_reservation_rejection(self):
        def reserve():
            try: return reserve_run_directory(self.root,'same','short/07')
            except FileExistsError: return None
        with ThreadPoolExecutor(max_workers=2) as pool:
            self.assertEqual(sum(p is not None for p in pool.map(lambda _:reserve(),range(2))),1)
        sentinel=self.root/'results_reproduction/runs/same/short/07/keep';sentinel.write_text('keep')
        for run,step in [('..','x'),('.','x'),('ok','../../escape'),('ok','/tmp/escape')]:
            with self.subTest(run=run,step=step),self.assertRaises(ValueError):
                reserve_run_directory(self.root,run,step)
        with self.assertRaises(FileExistsError): reserve_run_directory(self.root,'same','short/07')
        self.assertEqual(sentinel.read_text(),'keep')
        self.build()
        with self.assertRaises(FileExistsError): self.build()

    def test_plan_identity(self):
        plan=self.build()
        self.assertEqual(plan['protocol_id'],'m00_run_plan_v1')
        self.assertEqual(plan['schema_version'],1)
        self.assertEqual(plan['seed'],42)
        self.assertFalse(plan['is_smoke_test'])
        self.assertEqual(plan['population_id'],'unknown');self.assertEqual(plan['split_id'],'unknown')
        self.assertEqual(plan['configuration']['sha256'],hashlib.sha256((self.root/'config.yaml').read_bytes()).hexdigest())
        self.assertIn('git_commit',plan);self.assertIn('git_dirty',plan)
        self.assertEqual(plan['stages'][0]['resources'],{'device':'unknown','memory_gb':'unknown','storage_gb':'unknown'})
        self.assertEqual(plan['stages'][0]['parent_run_id'],'round-a')

    def test_default_stages(self):
        config=yaml.safe_load((ROOT/'configs/reproduction_m00.yaml').read_text())
        stages={s['notebook_id']:s for s in config['stages']}
        self.assertEqual(set(stages),{f'short/{i:02}' for i in range(1,12)}|{'short/06b'}|{f'long/{i:02}' for i in range(1,5)})
        for key in ['short/03','short/04']:
            self.assertEqual(stages[key]['action'],'skip');self.assertIn('checkpoint',stages[key]['reason'])

    def test_dependency_paths(self):
        config=yaml.safe_load((ROOT/'configs/reproduction_m00.yaml').read_text())
        plan=self.build(config=config)
        stages={s['notebook_id']:s for s in plan['stages']}
        for source,target in [('short/07','short/08'),('short/06','short/06b'),('long/01','long/02'),('long/02','long/03'),('long/03','long/04')]:
            dep=next(d for d in stages[target]['dependencies'] if d['source']==source)
            self.assertEqual(dep['manifest'],stages[source]['manifest'])
            self.assertEqual(dep['kind'],'planned')
        dep=next(d for d in stages['short/09']['dependencies'] if d['source']=='c03')
        self.assertEqual(dep['kind'],'external')
        self.assertTrue(dep['manifest'].endswith('preparation_manifest.json'))

    def test_invalid_configuration_cli(self):
        cases=[]
        c=copy.deepcopy(self.config);c['stages'][1]['dependencies']=[];cases.append(c)
        c=copy.deepcopy(self.config);c['stages'].append(c['stages'][0]);cases.append(c)
        c=copy.deepcopy(self.config);c['stages'][0]['dependencies']=[{'source':'short/08','kind':'planned'}];cases.append(c)
        c=copy.deepcopy(self.config);c['stages'][1]['dependencies'][0]['source']='unknown';cases.append(c)
        for i,c in enumerate(cases):
            config=self.root/f'bad{i}.yaml';config.write_text(yaml.safe_dump(c))
            result=self.cli('plan_reproduction_run.py','--config',config,'--project-root',self.root,'--run-id',f'bad{i}')
            self.assertEqual(result.returncode,2,result.stderr)
            self.assertIn('error',json.loads(result.stdout))
        self.assertFalse(list(self.root.rglob('run_plan.json')))

    def external(self):
        manifest=self.root/'external.json';manifest.write_text(json.dumps({'status':'COMPLETE','is_smoke_test':False}))
        c=copy.deepcopy(self.config)
        c['stages'][1]['dependencies']=[{'source':'short/07','kind':'external','manifest':str(manifest)}]
        return c,manifest

    def test_external_identity(self):
        c,path=self.external();before=path.read_bytes();plan=self.build(config=c)
        dep=plan['stages'][1]['dependencies'][0]
        self.assertEqual(dep['sha256'],hashlib.sha256(before).hexdigest())
        self.assertEqual(dep['manifest'],str(path.resolve()))
        self.assertEqual(inspect_run_stage(plan,'short/08')['status'],'READY')
        self.assertEqual(path.read_bytes(),before)

    def test_resume_pending(self):
        path=self.root/'resume.json';path.write_text('{}')
        c={'stages':[{'notebook_id':'short/06','requirements':[],'resume_manifest':str(path)}]}
        plan=self.build(config=c);stage=plan['stages'][0]
        self.assertEqual(stage['resume']['validation'],'PENDING_C09')
        self.assertNotEqual(Path(stage['output_dir']),path.parent)
        self.assertEqual(inspect_run_stage(plan,'short/06')['status'],'BLOCKED')
        self.assertEqual(path.read_text(),'{}')
        c['stages'] = [{'notebook_id': 'short/06b', 'requirements': [],
                        'resume_manifest': str(path), 'dependencies': [
                            {'source': 'short/06', 'kind': 'external', 'manifest': str(path)}]}]
        second = self.build('resume-06b', c)
        stage_b = second['stages'][0]
        self.assertEqual(stage_b['resume']['validation'], 'PENDING_C09')
        self.assertNotEqual(Path(stage_b['output_dir']), path.parent)
        self.assertEqual(inspect_run_stage(second, 'short/06b')['status'], 'BLOCKED')
        self.assertEqual(path.read_text(), '{}')

    def test_metadata_requirements(self):
        data=self.root/'input.parquet';pd.DataFrame({'wrong':[1]}).to_parquet(data)
        c={'stages':[{'notebook_id':'short/06b','requirements':[
            {'path':str(data),'kind':'parquet','columns':['sequence','label']},
            {'path':str(self.root/'missing.csv'),'kind':'csv','columns':['id']}],
            'dependencies':[{'source':'short/06','kind':'external','manifest':str(self.root/'missing-manifest.json')}]}]}
        report=inspect_run_stage(self.build(config=c),'short/06b')
        self.assertEqual(report['status'],'BLOCKED')
        self.assertEqual(len(report['requirements']),2)
        self.assertIn('sequence',report['requirements'][0]['detail'])
        self.assertIn('missing',report['requirements'][1]['detail'])
        invalid_xlsx = self.root / 'invalid.xlsx'
        invalid_xlsx.write_text('not an XLSX container')
        extra = {'stages': [{'notebook_id': 'short/05', 'requirements': [
            {'path': str(invalid_xlsx), 'kind': 'xlsx', 'sheets': ['MSL']},
            {'path': str(self.root / 'absent.csv'), 'kind': 'csv', 'columns': ['id']}]}]}
        malformed = inspect_run_stage(self.build('xlsx-invalid', extra), 'short/05')
        self.assertEqual(malformed['status'], 'BLOCKED')
        self.assertEqual(len(malformed['requirements']), 2)
        self.assertEqual([r['status'] for r in malformed['requirements']], ['BLOCKED', 'BLOCKED'])
        self.assertIn('zip', malformed['requirements'][0]['detail'].lower())
        self.assertIn('missing', malformed['requirements'][1]['detail'])

    def test_dependency_gate(self):
        planned=self.build();self.assertEqual(inspect_run_stage(planned,'short/08')['status'],'BLOCKED')
        c,path=self.external()
        for i,payload in enumerate([{'status':'FAILED','is_smoke_test':False},{'status':'COMPLETE','is_smoke_test':True},{}]):
            path.write_text(json.dumps(payload));plan=self.build(f'dep-{i}',c)
            self.assertEqual(inspect_run_stage(plan,'short/08')['status'],'BLOCKED')
        path.write_text(json.dumps({'status':'COMPLETE','is_smoke_test':False}))
        plan=self.build('hash',c);path.write_text(path.read_text()+' ')
        report=inspect_run_stage(plan,'short/08')
        self.assertEqual(report['status'],'BLOCKED');self.assertIn('hash',report['dependencies'][0]['detail'])

    def test_metadata_only(self):
        data=self.root/'input.parquet';pd.DataFrame({'sequence':['ACGT'],'label':[0]}).to_parquet(data)
        c={'stages':[{'notebook_id':'short/01','requirements':[{'path':str(data),'kind':'parquet','columns':['sequence','label']}]}]}
        with patch('pandas.read_parquet',side_effect=AssertionError('no rows')), \
             patch('torch.load',side_effect=AssertionError('no weights')), \
             patch('viral_cnn.notebook_runtime.ensure_smoke_test_artifacts',side_effect=AssertionError('no fixtures')):
            plan=self.build(config=c)
            self.assertEqual(inspect_run_stage(plan,'short/01')['status'],'READY')
        self.assertFalse(list(self.root.rglob('*.pt')))
        self.assertEqual(list(self.root.rglob('*.parquet')),[data])

    def test_atomic_failure(self):
        with patch('viral_cnn.run_planning.os.replace',side_effect=OSError('disk failure')):
            with self.assertRaises(OSError):self.build()
        self.assertFalse(list(self.root.rglob('run_plan.json')))
        failure=list(self.root.rglob('failure.json'))
        self.assertEqual(len(failure),1)
        self.assertEqual(json.loads(failure[0].read_text())['status'],'FAILED')

    def test_skip_no_products(self):
        c={'stages':[{'notebook_id':'short/03','action':'skip','reason':'existing checkpoint','requirements':[]}]}
        plan=self.build(config=c);stage=plan['stages'][0]
        self.assertEqual(inspect_run_stage(plan,'short/03')['status'],'SKIPPED')
        self.assertFalse(Path(stage['output_dir']).exists())
        self.assertFalse(Path(stage['manifest']).exists())

    def test_preflight_cli(self):
        plan=self.build()
        ready=self.cli('preflight_notebook.py','short/07','--run-plan',plan['plan_path'])
        blocked=self.cli('preflight_notebook.py','short/08','--run-plan',plan['plan_path'])
        self.assertEqual(ready.returncode,0,ready.stderr);self.assertEqual(json.loads(ready.stdout)['status'],'READY')
        self.assertEqual(blocked.returncode,2);self.assertEqual(json.loads(blocked.stdout)['status'],'BLOCKED')
        legacy=self.cli('preflight_notebook.py','short/01','--data-root',self.root,'--smoke-test')
        self.assertEqual(legacy.returncode,2);self.assertTrue(json.loads(legacy.stdout)['is_smoke_test'])
        data = self.root / 'cnn_kraken/processed_kraken_species_splited_parquet'
        data.mkdir(parents=True)
        pd.DataFrame({'sequence_padded': ['ACGT'], 'label': [0]}).to_parquet(data / 'test.parquet')
        checkpoint = self.root / 'cnn_kraken_species/checkpoints/best_resnet_classifier__fast_phase4.pt'
        checkpoint.parent.mkdir(parents=True)
        checkpoint.write_bytes(b'synthetic metadata-only placeholder; never loaded as weights')
        legacy_ready = self.cli('preflight_notebook.py', 'short/01', '--data-root', self.root, '--smoke-test')
        self.assertEqual(legacy_ready.returncode, 0, legacy_ready.stderr)
        self.assertEqual(json.loads(legacy_ready.stdout)['status'], 'READY')
        self.assertTrue(json.loads(legacy_ready.stdout)['is_smoke_test'])

    def test_documented_limits(self):
        text=(ROOT/'README.md').read_text()
        for literal in ['plan_reproduction_run.py','reproduction_m00.yaml','--run-plan','REMOTE_SERVER=False','preflight remoto','reprodução empírica','PENDING_C09']:
            self.assertIn(literal,text)


class RunPlanEdgeTests(unittest.TestCase):
    """Additional provenance rejection cases beyond the frozen named proofs."""

    def test_reference_exclusion_and_directory_symlinks(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'round').mkdir()
            (root / 'round' / 'current.json').write_text('{}')
            (root / 'old.json').write_text('{}')
            rows = inventory_reference([root], exclude=[root / 'round'])
            self.assertEqual([r['path'] for r in rows], [str(root / 'old.json')])

    def test_csv_and_planned_manifest_inspection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            csv_path = root / 'classes.csv'
            csv_path.write_text('id,species\n1,Virus\n')
            config = {'stages': [
                {'notebook_id': 'short/07', 'requirements': []},
                {'notebook_id': 'short/08', 'requirements': [
                    {'path': str(csv_path), 'kind': 'csv', 'columns': ['id', 'species']}],
                 'dependencies': [{'source': 'short/07', 'kind': 'planned'}]}]}
            config_path = root / 'config.yaml'
            config_path.write_text(yaml.safe_dump(config))
            plan = build_run_plan(config_path, root, data_root=root)
            predecessor = Path(plan['stages'][0]['manifest'])
            predecessor.write_text(json.dumps({'status': 'COMPLETE', 'is_smoke_test': False}))
            self.assertEqual(inspect_run_stage(plan, 'short/08')['status'], 'READY')
            predecessor.write_text(json.dumps({'evidence': None}))
            self.assertEqual(inspect_run_stage(plan, 'short/08')['status'], 'BLOCKED')
            csv_path.write_text('id\n1\n')
            report = inspect_run_stage(plan, 'short/08')
            self.assertEqual(report['requirements'][0]['status'], 'BLOCKED')
            self.assertIn('species', report['requirements'][0]['detail'])

    def test_all_partition_schemas_and_paths_are_logged(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            partitions = root / 'test'
            partitions.mkdir()
            pd.DataFrame({'sequence': ['ACGT'], 'label': [0]}).to_parquet(partitions / 'a.parquet')
            pd.DataFrame({'sequence': ['ACGT']}).to_parquet(partitions / 'b.parquet')
            config_path = root / 'config.yaml'
            config_path.write_text(yaml.safe_dump({'stages': [
                {'notebook_id': 'short/01', 'requirements': [
                    {'path': str(root), 'kind': 'parquet', 'split': 'test', 'columns': ['sequence', 'label']}]}]}))
            with patch('viral_cnn.run_planning.log_artifact_path') as logged:
                plan = build_run_plan(config_path, root, data_root=root)
                report = inspect_run_stage(plan, 'short/01')
            self.assertEqual(report['status'], 'BLOCKED')
            self.assertIn('b.parquet', report['requirements'][0]['detail'])
            self.assertTrue(any(call.args[1] == partitions / 'b.parquet' for call in logged.call_args_list))

    def test_reservation_rejects_symlink_alias(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runs = root / 'results_reproduction/runs'
            (runs / 'historical').mkdir(parents=True)
            (runs / 'alias').symlink_to(runs / 'historical', target_is_directory=True)
            with self.assertRaises(ValueError):
                reserve_run_directory(root, 'alias', 'short/01')
            self.assertEqual(list((runs / 'historical').iterdir()), [])
