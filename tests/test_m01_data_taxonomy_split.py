"""M01 scientific contracts, derived from the approved checks before implementation."""
import contextlib
import csv
import hashlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ['REMOTE_SERVER'] = 'False'
import pandas as pd
import yaml

from viral_cnn.m01_workflow import resolve_m01_inputs, run_m01
from viral_cnn.taxonomy.curation import validate_taxonomy_identity

ROOT = Path(__file__).resolve().parents[1]


class M01Tests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.env = patch.dict(os.environ, {'REMOTE_SERVER': 'False', 'LOG_PATH': '0'}, clear=True)
        self.env.start()
        self.addCleanup(self.env.stop)
        self.mapping = self.root / 'classes.csv'
        pd.DataFrame({'id':[1,2], 'species':['Alpha','Beta']}).to_csv(self.mapping,index=False)
        self.taxonomy = self.root / 'taxonomy.parquet'
        self.tax().to_parquet(self.taxonomy,index=False)
        # Direct only: AACC; RC only: CATT (train AATG); both: ACGT; neither: GAGA.
        self.paths = {}
        for split,seqs,labels in [
            ('train',['AACC','AATG','ACGT'],[0,1,0]),
            ('validation',['____','AN','A'*513],[0,1,1]),
            ('test',['aacc___','CATT','ACGT','GAGA','____','NNNN'],[0,1,0,1,0,1])]:
            p = self.root / (split+'.csv')
            pd.DataFrame({'sequence_padded':seqs,'label':labels}).to_csv(p,index=False)
            self.paths[split]=str(p)
        self.settings = {'data_root':str(self.root),'original_mapping':str(self.mapping),
                         'taxonomy_path':str(self.taxonomy),'splits':self.paths}

    def tax(self):
        return pd.DataFrame({'id':[1,2],'model_label':[0,1],'species':['Alpha','Beta'],
                             'Genus':['G1','G2'],'Order':['O1','O2'],'Family':['F1','F2']})

    def run_stage(self, stage='short/10', **kwargs):
        return run_m01(stage, self.root, settings={**self.settings,**kwargs})

    def overlap(self):
        return self.run_stage()

    def audit(self, **kwargs):
        m=self.overlap()
        return self.run_stage('short/11',overlap_manifest=str(Path(m['output_dir'])/'manifest.json'),**kwargs)

    def workbook(self):
        p=self.root/'msl.xlsx'
        master=self.tax().rename(columns={'species':'Species'})
        for rank in ['Realm','Kingdom','Phylum','Class']:
            master[rank]='R'
        with pd.ExcelWriter(p) as writer:
            master.to_excel(writer,sheet_name='MSL',index=False)
            pd.DataFrame(columns=['Change','Old Name','New Name']).to_excel(writer,sheet_name='Taxa Renamed or Abolished',index=False)
        return p

    def execute_notebook(self, stem, settings, smoke=False):
        path=next((ROOT/'notebooks/reproduction/short').glob(stem+'*.ipynb'))
        config=self.root/'notebook.yaml';config.write_text(yaml.safe_dump(settings))
        ns={'__name__':'__main__'}
        with patch.dict(os.environ, {'M01_CONFIG':str(config),'M01_PROJECT_ROOT':str(self.root),
                                     'SMOKE_TEST':'1' if smoke else '0'}):
            for c in json.loads(path.read_text())['cells']:
                if c['cell_type']=='code':exec(''.join(c['source']),ns)
        return ns['manifest']

    def test_paths(self):
        config=self.root/'config.yaml';config.write_text(yaml.safe_dump(self.settings))
        alternate=self.root/'alternate.csv';alternate.write_bytes(self.mapping.read_bytes())
        alternate_tax=self.root/'alternate.parquet';alternate_tax.write_bytes(self.taxonomy.read_bytes())
        workbook=self.workbook()
        selected={'ORIGINAL_MAPPING_PATH':str(alternate),'TAXONOMY_PATH':str(alternate_tax),
                  'ICTV_MASTER_PATH':str(workbook),'DATA_ROOT':str(self.root),'LOG_PATH':'true'}
        with patch.dict(os.environ,selected),contextlib.redirect_stderr(io.StringIO()) as log:
            resolved=resolve_m01_inputs('short/10',self.root,config_path=config)
            self.assertEqual(resolved['original_mapping'],str(alternate))
            self.assertEqual(resolved['splits']['validation'],self.paths['validation'])
            ten=self.execute_notebook('10_',self.settings)
            source=str(Path(ten['output_dir'])/'manifest.json')
            with patch.dict(os.environ,{'OVERLAP_MANIFEST':source}):
                eleven=self.execute_notebook('11_',self.settings)
                five=self.execute_notebook('05_',self.settings)
            for manifest in [ten,eleven,five]:
                self.assertEqual(manifest['configuration']['original_mapping'],str(alternate))
                self.assertEqual(manifest['configuration']['taxonomy_path'],str(alternate_tax))
                self.assertEqual(manifest['configuration']['ictv_master'],str(workbook))
            self.assertEqual(eleven['configuration']['overlap_manifest'],source)
            self.assertTrue(eleven['metrics_summary']['mapping_integrity_ok'])
            self.assertEqual(five['protocol_id'],'m01_ictv_curate_v1')
            for path in [str(alternate),str(alternate_tax),str(workbook),source]:self.assertIn(path,log.getvalue())
        # Default topology is resolved from DATA_ROOT, without relying on any legacy outputs.
        base=self.root/'cnn_kraken/processed_kraken_species_splited_parquet';base.mkdir(parents=True)
        for split,name in [('train','train'),('validation','val'),('test','test')]:
            pd.read_csv(self.paths[split],keep_default_na=False).to_parquet(base/(name+'.parquet'),index=False)
        for stage in ['short/10','short/11','short/05']:
            defaults=resolve_m01_inputs(stage,self.root,settings={'data_root':str(self.root)})
            self.assertEqual(defaults['original_mapping'],str(self.root/'dataset_kraken/species_name_ids.csv'))
            self.assertEqual(defaults['ictv_master'],str(self.root/'short/ICTV_Master_Species_List_2025_MSL41.v1.xlsx'))
            if stage=='short/10':self.assertEqual(defaults['splits']['validation'],[str(base/'val.parquet')])
        # Selected M00 sources beat configuration/environment overrides at all three entries.
        from viral_cnn.run_planning import build_run_plan
        spec={'data_root':str(self.root),'reference_roots':[],'stages':[]}
        for stage in ['short/10','short/11','short/05']:
            values=dict(self.settings)
            if stage=='short/05':values['mode']='audit_existing'
            item={'notebook_id':stage,'m01':values,'requirements':[]}
            if stage!='short/10':item['dependencies']=[{'source':'short/10','kind':'planned'}]
            spec['stages'].append(item)
        cfg=self.root/'pinned.yaml';cfg.write_text(yaml.safe_dump(spec))
        plan=build_run_plan(cfg,self.root,run_id='pinned')
        for stage in ['short/10','short/11','short/05']:
            m=run_m01(stage,self.root,run_plan=plan['plan_path'])
            self.assertEqual(m['configuration']['original_mapping'],str(self.mapping))
            self.assertEqual(m['configuration']['taxonomy_path'],str(self.taxonomy))
            with patch.dict(os.environ,{'ORIGINAL_MAPPING_PATH':str(alternate)}),self.assertRaisesRegex(ValueError,'Override conflicts'):
                resolve_m01_inputs(stage,self.root,run_plan=plan['plan_path'])

    def test_isolation(self):
        sentinel=self.root/'results_reproduction/historical/report.json'
        sentinel.parent.mkdir(parents=True);sentinel.write_text('historical evidence')
        sources=[self.mapping,self.taxonomy,sentinel,*map(Path,self.paths.values())]
        before={str(p):p.read_bytes() for p in sources}
        for stage in ['short/10','short/11','short/05']:
            if stage=='short/10':a,b=self.overlap(),self.overlap()
            elif stage=='short/11':a,b=self.audit(),self.audit()
            else:a,b=self.run_stage(stage,mode='audit_existing'),self.run_stage(stage,mode='audit_existing')
            self.assertNotEqual(a['output_dir'],b['output_dir'])
            self.assertEqual(before,{str(p):p.read_bytes() for p in sources})

    def make_plan(self):
        target=self.root/'results_reproduction/runs/child/short/10';target.mkdir(parents=True)
        plan={'protocol_id':'m00_run_plan_v1','schema_version':1,'parent_run_id':'parent',
              'data_root':str(self.root),'is_smoke_test':False,'seed':42,'stages':[
                  {'notebook_id':'short/10','run_id':'child','parent_run_id':'parent',
                   'output_dir':str(target),'manifest':str(target/'manifest.json'),'action':'run',
                   'requirements':[{'path':str(self.mapping),'kind':'csv','role':'original_mapping'}],
                   'dependencies':[],'m01':self.settings}]}
        p=self.root/'plan.json';p.write_text(json.dumps(plan));return p,target

    def test_plan_reservation(self):
        p,target=self.make_plan()
        m=run_m01('short/10',self.root,run_plan=p)
        self.assertEqual(m['run_id'],'child');self.assertEqual(m['parent_run_id'],'parent')
        self.assertEqual(Path(m['output_dir']),target)
        with self.assertRaises(FileExistsError):run_m01('short/10',self.root,run_plan=p)

    def test_rejection(self):
        with self.assertRaises(FileNotFoundError):self.run_stage('short/05',ictv_master='missing.xlsx')
        failed=list(self.root.glob('results_reproduction/runs/**/manifest.json'))
        self.assertTrue(failed)
        self.assertTrue(all(json.loads(p.read_text())['status']=='FAILED' for p in failed))
        self.assertFalse(list(self.root.glob('results_reproduction/runs/**/*.parquet')))
        p,_=self.make_plan()
        with self.assertRaises(ValueError):
            run_m01('short/10',self.root,run_plan=p,settings={'original_mapping':'conflicting.csv'})

    def test_manifests(self):
        protocols={'short/10':'m01_fragment_split_audit_v1','short/11':'m01_taxonomy_split_audit_v1',
                   'short/05':'m01_existing_taxonomy_audit_v1'}
        expected_outputs={'short/10':{'decision.json','per_species.csv','split_counts.csv','split_class_counts.csv','split_lengths.csv'},
                          'short/11':{'taxonomy_audit.json','per_taxon.csv','per_genus.csv','per_family.csv','per_order.csv','rank_validation.csv'},
                          'short/05':{'curation_audit.json','rank_validation.csv','class_validation.csv'}}
        for stage in ['short/10','short/11','short/05']:
            if stage=='short/11':m=self.audit()
            elif stage=='short/05':m=self.run_stage(stage,mode='audit_existing')
            else:m=self.overlap()
            self.assertEqual(m['status'],'COMPLETE');self.assertEqual(m['notebook_id'],stage)
            for key in ['protocol_id','configuration','library_versions','git_commit','git_dirty','seed','inputs','outputs']:
                self.assertIn(key,m)
            self.assertEqual(m['seed'],42)
            self.assertFalse(m['is_smoke_test']);self.assertEqual(m['evidence']['status'],'observed')
            self.assertEqual(m['protocol_id'],protocols[stage])
            self.assertEqual(m['configuration']['original_mapping'],str(self.mapping))
            self.assertEqual(m['configuration']['data_root'],str(self.root))
            self.assertEqual(m['configuration']['mode'],'audit_existing' if stage=='short/05' else 'curate')
            self.assertEqual(m['library_versions']['pandas'],pd.__version__)
            self.assertEqual(json.loads((Path(m['output_dir'])/'manifest.json').read_text()),m)
            self.assertTrue(m['inputs']);self.assertTrue(m['outputs'])
            self.assertTrue(all(x['sha256']!='unknown' for x in m['inputs']))
            self.assertTrue(all(x['sha256']!='unknown' for x in m['outputs']))
            self.assertEqual({Path(r['path']).name for r in m['outputs']},expected_outputs[stage])
            for record in m['inputs']+m['outputs']:
                path=Path(record['path'])
                self.assertTrue(path.is_absolute());self.assertTrue(path.is_file())
                self.assertEqual(record['sha256'],hashlib.sha256(path.read_bytes()).hexdigest())
                self.assertEqual(record['size_bytes'],path.stat().st_size)
            self.assertIn(str(self.mapping),[r['path'] for r in m['inputs']])
            if stage=='short/10':self.assertTrue(set(self.paths.values()).issubset(r['path'] for r in m['inputs']))
            else:self.assertIn(str(self.taxonomy),[r['path'] for r in m['inputs']])

    def test_split_counts(self):
        m=self.execute_notebook('10_',self.settings)
        out=Path(m['output_dir'])
        with (out/'split_counts.csv').open() as handle:
            rows=list(csv.DictReader(handle))
        self.assertEqual({r['split']:int(r['rows']) for r in rows},{'train':3,'validation':3,'test':6})
        val=next(r for r in rows if r['split']=='validation')
        self.assertEqual(int(val['empty_after_padding']),1)
        cls=pd.read_csv(out/'split_class_counts.csv');self.assertEqual(cls['rows'].sum(),12)
        self.assertEqual({(r.split,r.model_label):r.rows for r in cls.itertuples()},
                         {('train',0):2,('train',1):1,('validation',0):1,('validation',1):2,('test',0):3,('test',1):3})
        self.assertEqual({r['split']:(int(r['empty_after_padding']),int(r['no_acgt']),int(r['over_512_bp']),int(r['benchmark_compatible'])) for r in rows},
                         {'train':(0,0,0,3),'validation':(1,0,1,1),'test':(1,1,0,4)})
        lengths=pd.read_csv(out/'split_lengths.csv')
        self.assertEqual(int(lengths.loc[(lengths.split=='validation')&(lengths.length==513),'rows'].iloc[0]),1)
        self.assertEqual({(r.split,r.length):r.rows for r in lengths.itertuples()},
                         {('train',4):3,('validation',0):1,('validation',2):1,('validation',513):1,('test',0):1,('test',4):5})

    def test_orientations(self):
        d=self.overlap()['metrics_summary']
        self.assertEqual((d['direct_matches'],d['reverse_complement_matches'],d['both_matches'],d['exact_test_matches_train']),(2,2,1,3))

    def test_denominator(self):
        d=self.overlap()['metrics_summary']
        self.assertEqual(d['test_rows'],6);self.assertEqual(d['exact_overlap_pct'],50.)
        self.assertEqual(d['benchmark_compatible_test_rows'],4)
        self.assertEqual(d['benchmark_compatible_exact_overlap_pct'],75.)

    def test_provenance(self):
        self.assertEqual(self.overlap()['metrics_summary']['status'],'PROVENANCE_REQUIRED')

    def test_pilot(self):
        d=self.run_stage(max_test_rows=2)['metrics_summary']
        self.assertEqual(d['status'],'INCONCLUSIVE_PARTIAL_TEST');self.assertEqual(d['test_rows'],2)
        self.assertEqual(d['scope'],'first_2_test_rows_pilot')

    def test_population(self):
        m=self.overlap();self.assertEqual(m['protocol_id'],'m01_fragment_split_audit_v1')
        self.assertEqual(m['metrics_summary']['population'],'fragment_test_vs_train_not_c03_queries')

    def test_mapping(self):
        for field,values in [('id',[1,1]),('id',[1,3]),('model_label',[1,0]),('species',['Changed','Beta'])]:
            with self.subTest(field=field,values=values):
                df=self.tax();df[field]=values
                report=validate_taxonomy_identity(df,pd.read_csv(self.mapping))
                self.assertFalse(report['mapping_integrity_ok'])
        self.assertFalse(validate_taxonomy_identity(self.tax().iloc[:1],pd.read_csv(self.mapping))['mapping_integrity_ok'])
        self.tax().assign(model_label=[1,0]).to_parquet(self.taxonomy,index=False)
        self.assertFalse(self.audit()['metrics_summary']['mapping_integrity_ok'])

    def test_dependency(self):
        m=self.overlap();path=Path(m['output_dir'])/'manifest.json'
        decision=Path(m['output_dir'])/'decision.json'
        decision.write_text(decision.read_text()+' ')
        with self.assertRaises(ValueError):self.run_stage('short/11',overlap_manifest=str(path))
        m=self.run_stage(max_test_rows=1)
        with self.assertRaises(ValueError):self.run_stage('short/11',overlap_manifest=str(Path(m['output_dir'])/'manifest.json'))
        m=self.overlap();species=Path(m['output_dir'])/'per_species.csv'
        df=pd.read_csv(species);df['test_rows']+=1;df.to_csv(species,index=False)
        with self.assertRaises(ValueError):self.run_stage('short/11',overlap_manifest=str(Path(m['output_dir'])/'manifest.json'))

        for field,message in [('test_rows','test total'),('exact_test_matches_train','match total')]:
            m=self.overlap();output=Path(m['output_dir'])
            decision=output/'decision.json';d=json.loads(decision.read_text());d[field]+=1;decision.write_text(json.dumps(d))
            manifest_path=output/'manifest.json';selected=json.loads(manifest_path.read_text())
            for record in selected['outputs']:
                if record['path']==str(decision):
                    record['sha256']=hashlib.sha256(decision.read_bytes()).hexdigest()
                    record['size_bytes']=decision.stat().st_size
            manifest_path.write_text(json.dumps(selected))
            with self.subTest(field=field),self.assertRaisesRegex(ValueError,message):
                self.run_stage('short/11',overlap_manifest=str(manifest_path))

    def test_ranks(self):
        for rank in ['Genus','Order']:
            for bad in [None,'','Unassigned_'+rank,'Unclassified Parent']:
                df=self.tax();df.loc[0,rank]=bad
                df.to_parquet(self.taxonomy,index=False)
                m=self.audit();self.assertFalse(m['metrics_summary']['hierarchy_ready'])
                self.assertIn(0,m['metrics_summary']['unresolved_ranks'][rank])
                m=self.run_stage('short/05',mode='audit_existing')
                self.assertFalse(m['metrics_summary']['hierarchy_ready'])
                self.assertFalse(list(Path(m['output_dir']).glob('*.parquet')))

    def test_strata(self):
        df=self.tax();df.loc[0,'Genus']=None;df.to_parquet(self.taxonomy,index=False)
        m10=self.overlap()
        m=self.execute_notebook('11_',dict(self.settings,overlap_manifest=str(Path(m10['output_dir'])/'manifest.json')))
        for rank in ['genus','family','order']:
            grouped=pd.read_csv(Path(m['output_dir'])/('per_'+rank+'.csv'))
            self.assertEqual(grouped['test_rows'].sum(),6)
            self.assertEqual(grouped['exact_matches'].sum(),3)
        groups=pd.read_csv(Path(m['output_dir'])/'per_genus.csv')
        self.assertIn('[unresolved]',groups['Genus'].values)

    def test_existing(self):
        before=self.taxonomy.read_bytes()
        m=self.run_stage('short/05',mode='audit_existing')
        self.assertEqual(m['protocol_id'],'m01_existing_taxonomy_audit_v1')
        self.assertEqual(before,self.taxonomy.read_bytes());self.assertFalse(list(Path(m['output_dir']).glob('*.parquet')))
        self.assertFalse(m['metrics_summary']['reconstructed_from_msl'])

    def test_missing_workbook(self):
        with patch('viral_cnn.taxonomy.curation.generate_synthetic_taxonomy',side_effect=AssertionError('forbidden')):
            with self.assertRaises(FileNotFoundError):self.execute_notebook('05_',dict(self.settings,ictv_master='missing.xlsx'))
        self.assertFalse(list(self.root.glob('results_reproduction/runs/**/*.parquet')))
        self.assertTrue(all(json.loads(p.read_text())['status']=='FAILED' for p in self.root.glob('results_reproduction/runs/**/manifest.json')))

    def test_invalid_workbook(self):
        for case in ['missing_sheet','missing_rank','invalid_ids']:
            p=self.workbook()
            if case=='missing_sheet':pd.DataFrame({'Species':['Alpha']}).to_excel(p,index=False)
            elif case=='missing_rank':
                with pd.ExcelWriter(p) as w:
                    pd.DataFrame({'Species':['Alpha']}).to_excel(w,sheet_name='MSL',index=False)
                    pd.DataFrame(columns=['Change','Old Name','New Name']).to_excel(w,sheet_name='Taxa Renamed or Abolished',index=False)
            else:pd.DataFrame({'id':[1,1],'species':['Alpha','Beta']}).to_csv(self.mapping,index=False)
            with self.subTest(case=case),self.assertRaises(ValueError):self.run_stage('short/05',ictv_master=str(p))
        self.assertFalse(list(self.root.glob('results_reproduction/runs/**/*.parquet')))
        self.assertTrue(all(json.loads(p.read_text())['status']=='FAILED' for p in self.root.glob('results_reproduction/runs/**/manifest.json')))

    def test_curate(self):
        p=self.workbook();m=self.run_stage('short/05',ictv_master=str(p))
        self.assertEqual(m['protocol_id'],'m01_ictv_curate_v1')
        out=Path(m['output_dir'])/'main_taxonomy_hybrid_CLEANSED.parquet'
        df=pd.read_parquet(out)
        self.assertEqual(df.model_label.tolist(),[0,1]);self.assertEqual(df.species.tolist(),['Alpha','Beta'])
        self.assertEqual(df.Genus.tolist(),['G1','G2']);self.assertTrue(m['metrics_summary']['hierarchy_ready'])
        self.assertIn(str(p),[i['path'] for i in m['inputs']])

        master=pd.read_excel(p,sheet_name='MSL').iloc[:1]
        with pd.ExcelWriter(p,mode='a',if_sheet_exists='replace',engine='openpyxl') as writer:
            master.to_excel(writer,sheet_name='MSL',index=False)
        ncbi=self.root/'ncbi_local.csv'
        pd.DataFrame({'species':['Beta'],'Genus':['LocalGenus'],'Order':['LocalOrder'],'Family':['LocalFamily']}).to_csv(ncbi,index=False)
        m=self.run_stage('short/05',ictv_master=str(p),ncbi_taxonomy=str(ncbi))
        df=pd.read_parquet(Path(m['output_dir'])/'main_taxonomy_hybrid_CLEANSED.parquet')
        self.assertEqual(df.model_label.tolist(),[0,1]);self.assertEqual(df.species.tolist(),['Alpha','Beta'])
        self.assertEqual(df.Genus.tolist(),['G1','LocalGenus'])
        self.assertEqual(df.Order.tolist(),['O1','LocalOrder'])
        self.assertEqual(df.taxonomy_source.tolist(),['ICTV Master','NCBI Fallback'])
        sources={r['path']:r for r in m['inputs']}
        for path in [self.mapping,p,ncbi]:
            self.assertIn(str(path),sources)
            self.assertEqual(sources[str(path)]['sha256'],hashlib.sha256(path.read_bytes()).hexdigest())

    def test_synthetic(self):
        m=self.execute_notebook('05_',self.settings,smoke=True)
        self.assertTrue(m['is_smoke_test']);self.assertEqual(m['evidence']['status'],'synthetic')
        ten=run_m01('short/10',self.root,settings=self.settings,is_smoke_test=True)
        source=str(Path(ten['output_dir'])/'manifest.json')
        with self.assertRaises(ValueError):self.run_stage('short/11',overlap_manifest=source)
        eleven=run_m01('short/11',self.root,settings=dict(self.settings,overlap_manifest=source),is_smoke_test=True)
        self.assertTrue(eleven['is_smoke_test']);self.assertTrue(eleven['dependencies'][0]['is_smoke_test'])

    def test_unknown_genome(self):
        for mode in ['audit_existing','curate']:
            m=self.run_stage('short/05',mode=mode,ictv_master=str(self.workbook()))
            self.assertEqual(m['metrics_summary']['genome_provenance'],'unknown')

    def test_logging(self):
        workbook=self.workbook();ncbi=self.root/'fallback.csv'
        pd.DataFrame({'species':['Beta'],'Genus':['LocalGenus'],'Order':['LocalOrder']}).to_csv(ncbi,index=False)
        for enabled in [None,'true','0','false']:
            stream=io.StringIO()
            with patch.dict(os.environ,{'LOG_PATH':enabled or 'true'}):
                if enabled is None:os.environ.pop('LOG_PATH')
                with contextlib.redirect_stderr(stream):
                    ten=self.overlap();out=Path(ten['output_dir']);source=str(out/'manifest.json')
                    self.run_stage('short/11',overlap_manifest=source)
                    self.run_stage('short/05',ictv_master=str(workbook),ncbi_taxonomy=str(ncbi))
                    self.run_stage('short/05',mode='audit_existing',overlap_manifest=source)
            text=stream.getvalue()
            if enabled in (None,'true'):
                for path in [str(self.mapping),*self.paths.values(),str(self.taxonomy),source,
                             str(out/'decision.json'),str(out/'per_species.csv'),str(workbook),str(ncbi)]:
                    self.assertIn(path,text)
            else:self.assertNotIn('[LOG_PATH]',text)

    def test_real_m00_chain(self):
        from viral_cnn.run_planning import build_run_plan
        config={'data_root':str(self.root),'reference_roots':[], 'stages':[]}
        for key in ['short/10','short/11','short/05']:
            stage={'notebook_id':key,'m01':dict(self.settings), 'requirements':[]}
            if key != 'short/10':stage['dependencies']=[{'source':'short/10','kind':'planned'}]
            if key == 'short/05':stage['m01']['mode']='audit_existing'
            config['stages'].append(stage)
        cfg=self.root/'m00.yaml';cfg.write_text(yaml.safe_dump(config))
        plan=build_run_plan(cfg,self.root,run_id='round')
        for key in ['short/10','short/11','short/05']:
            m=run_m01(key,self.root,run_plan=plan['plan_path'])
            self.assertEqual(m['parent_run_id'],'round');self.assertEqual(m['status'],'COMPLETE')
            if key=='short/11':self.assertTrue(m['metrics_summary']['mapping_integrity_ok'])

    def test_invalid_fragment_label_and_shared_file(self):
        pd.DataFrame({'sequence':['ACGT'],'label':[8]}).to_csv(self.paths['validation'],index=False)
        with self.assertRaises(ValueError):self.overlap()
        pd.DataFrame({'sequence':['ACGT'],'label':[0]}).to_csv(self.paths['validation'],index=False)
        with self.assertRaises(ValueError):self.run_stage(splits=dict(self.paths,test=self.paths['train']))

    def test_iupac_orientation_and_large_identity(self):
        pd.DataFrame({'sequence':['AARY','AC?G'],'label':[0,1]}).to_csv(self.paths['train'],index=False)
        pd.DataFrame({'sequence':['RYTT','AC?G','NN','____'],'label':[0,1,0,1]}).to_csv(self.paths['test'],index=False)
        m=self.run_stage(hash_max_bytes=0)
        self.assertEqual(m['metrics_summary']['reverse_complement_matches'],1)
        self.assertEqual(m['metrics_summary']['direct_matches'],1)
        self.assertEqual(m['metrics_summary']['exact_overlap_pct'],50.)
        self.assertTrue(all(r['sha256']=='unknown' and r['hash_status']=='skipped_size' for r in m['fragment_inventory']))

    def test_synthetic_reference_is_not_empirical(self):
        self.tax().assign(taxonomy_source='Synthetic').to_parquet(self.taxonomy,index=False)
        with self.assertRaises(ValueError):self.run_stage('short/05',mode='audit_existing')
        m=self.overlap()
        with self.assertRaises(ValueError):self.run_stage('short/11',overlap_manifest=str(Path(m['output_dir'])/'manifest.json'))

    def test_documentation(self):
        text=(ROOT/'docs/m01_data_taxonomy_split.md').read_text()
        for value in ['M01_CONFIG','M01_RUN_PLAN','short/10','short/11','short/05','manifest.json','REMOTE_SERVER=False','SMOKE_TEST=1','preflight','empírica']:
            self.assertIn(value,text)


if __name__=='__main__':unittest.main()
