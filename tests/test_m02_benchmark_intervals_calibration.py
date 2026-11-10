"""M02 acceptance proofs. All data are temporary protocol fixtures, never evidence."""
import contextlib
import hashlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ['REMOTE_SERVER'] = 'False'
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
torch.set_num_threads(1)
# Initialize the CPU autograd engine before unittest emits proof lines.
torch.tensor(1., requires_grad=True).square().backward()
from viral_cnn.baselines.preparation import prepare_queries, bind_query_manifests, write_method_condition_manifest
from viral_cnn.baselines.overlap import inspect_c03_package, load_prepared_baseline_benchmark

ROOT = Path(__file__).resolve().parents[1]
METHODS = ['resnet50_species','resnet_noise_species','convnext_zero_species',
           'convnext_transfer_species','convnext_noise_species','Kraken2-flat','BLASTN']
RATES = [0.,.02,.05,.10,.15]


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def put(p, obj):
    Path(p).write_text(json.dumps(obj, indent=2))


def package(root):
    """Build a full schema fixture with dummy checkpoint/db bytes, no executable model."""
    mapping = root/'classes.csv'
    pd.DataFrame({'species':[f's{i}' for i in range(13683)],'id':range(1,13684)}).to_csv(mapping,index=False)
    train, test = root/'train.csv', root/'test.csv'
    pd.DataFrame({'label':[0,1],'sequence':['A'*512,'C'*512]}).to_csv(train,index=False)
    pd.DataFrame({'query_id':['q0','q1','q2','q3'],'label':[0,0,1,1],
                  'sequence':['A'*512,'AG'*256,'C'*512,'AC'*256]}).to_csv(test,index=False)
    out=root/'prepared'
    meta=prepare_queries(train,test,out,class_order_path=mapping,is_smoke_test=False)
    meta['status']='COMPLETE';put(out/'preparation_manifest.json',meta);bind_query_manifests(out)
    ref=Path(meta['reference_manifest'])
    # Six tiny files stand in for database outputs. They never execute a search.
    db=ref.parent/'database_build_manifest.json';kr=ref.parent/'kraken';kr.mkdir()
    bp=ref.parent/'blast'
    files=[kr/n for n in ('hash.k2d','opts.k2d','taxo.k2d')]+[Path(str(bp)+e) for e in ('.nhr','.nin','.nsq')]
    for p in files:p.write_bytes(b'protocol fixture')
    put(db,{'state':'COMPLETE','baseline_protocol':'c03_paired_v1','is_smoke_test':False,
            'reference_manifest_sha256':sha(ref),'reference_manifest':str(ref),
            'kraken_db':str(kr),'blast_db_prefix':str(bp),'output_sha256':{str(p):sha(p) for p in files}})
    ck=root/'checkpoint.pt';ck.write_bytes(b'not a model; no torch.load permitted')
    comp=out/'comparison_results';comp.mkdir()
    for m in METHODS:
        for rate in RATES:
            tag=f'noise_{round(rate*100):02d}'
            qm=out/f'query_manifest_{tag}.json';q=json.loads(qm.read_text())
            frame=pd.read_csv(out/f'queries_{tag}.csv')[['query_id','label']]
            frame['prediction']=frame.label
            if m in METHODS[-2:]:frame['abstention_reason']=''
            for k in range(1,6):frame[f'top{k}']=(frame.label+k-1)%13683
            dest=comp/f'predictions_{m}_{tag}.csv';frame.to_csv(dest,index=False)
            reference={'path':str(ref),'sha256':sha(ref)}
            if m in METHODS[-2:]:reference.update(database_path=str(kr if m=='Kraken2-flat' else bp),
                    database_build_manifest={'path':str(db),'sha256':sha(db)})
            execution={'query_csv_path':str(out/f'queries_{tag}.csv'),'query_csv_sha256':q['query_content_sha256'],
                       'query_fasta_path':str(out/f'queries_{tag}.fasta'),'query_fasta_sha256':q['fasta_sha256']}
            write_method_condition_manifest(comp/f'manifest_{m}_{tag}.json',method=m,noise_rate=rate,
                query_manifest=qm,class_order_sha256=q['class_order_sha256'],reference=reference,
                tool_version='protocol fixture',command=['fixture-only'],
                abstention_policy=('Kraken U/root => -1' if m=='Kraken2-flat' else 'completed BLASTN with no hit => -1' if m=='BLASTN' else 'dense CNN; no abstention'),
                completion='COMPLETE',is_smoke_test=False,output_path=dest,
                model_variant=('resnet_clean_species' if m=='resnet50_species' else m) if m not in METHODS[-2:] else None,
                checkpoint_path=ck if m not in METHODS[-2:] else None,execution_details=execution)
    return out


class M02Tests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
        cpu=patch('torch.cuda.is_available',return_value=False)
        cpu.start();self.addCleanup(cpu.stop)
        env=patch.dict(os.environ,{'REMOTE_SERVER':'False','LOG_PATH':'0'},clear=True)
        env.start();self.addCleanup(env.stop)

    def complete(self):
        return package(self.root)

    def mutate_manifest(self,out,field,value,m='BLASTN'):
        p=out/'comparison_results'/f'manifest_{m}_noise_00.json'
        obj=json.loads(p.read_text());obj[field]=value;put(p,obj)

    def test_c03_complete_fixture(self):
        out=self.complete();r=inspect_c03_package(out)
        self.assertEqual(r['status'],'READY');self.assertEqual(r['method_condition_count'],35)
        self.assertEqual(r['n'],4);self.assertFalse(r['scientific_reproduction_certified'])

    def test_c03_completeness_matrix(self):
        out=self.complete()
        paths=[out/'preparation_manifest.json',next(out.rglob('manifest_BLASTN_noise_00.json')),
               Path(json.loads(next(out.rglob('manifest_BLASTN_noise_00.json')).read_text())['reference']['database_build_manifest']['path'])]
        for p in paths:
            original=p.read_bytes()
            with self.subTest(missing=p):
                p.unlink();self.assertEqual(inspect_c03_package(out)['status'],'BLOCKED');p.write_bytes(original)
            obj=json.loads(original);obj['status']='RUNNING';obj['state']='RUNNING';obj['completion']='RUNNING'
            with self.subTest(incomplete=p):
                put(p,obj);self.assertEqual(inspect_c03_package(out)['status'],'BLOCKED');p.write_bytes(original)
        # The database link itself is mandatory for each tool, even with files present.
        for m in METHODS[-2:]:
            p=out/'comparison_results'/f'manifest_{m}_noise_00.json';old=p.read_bytes()
            obj=json.loads(old);obj['reference'].pop('database_build_manifest');put(p,obj)
            self.assertEqual(inspect_c03_package(out)['status'],'BLOCKED');p.write_bytes(old)

        # Every one of the 35 conditions must independently block the common gate.
        for p in sorted((out/'comparison_results').glob('manifest_*.json')):
            original=p.read_bytes();obj=json.loads(original)
            p.unlink();self.assertEqual(inspect_c03_package(out)['status'],'BLOCKED');p.write_bytes(original)
            for field,value in [('completion','FAILED'),('output_sha256','0'*64)]:
                changed={**obj,field:value};put(p,changed)
                with self.subTest(condition=p.name,field=field):self.assertEqual(inspect_c03_package(out)['status'],'BLOCKED')
                p.write_bytes(original)
        # Both public preflight paths must reach the same guard with a pinned preparation.
        from viral_cnn.notebook_runtime import preflight_notebook
        from viral_cnn.run_planning import inspect_run_stage, load_run_plan
        plan_path=self.plan_stage('short/09','preflight-parent',smoke=False,c03=out)
        p=out/'comparison_results/manifest_BLASTN_noise_15.json';original=p.read_bytes();obj=json.loads(original);obj['completion']='RUNNING';put(p,obj)
        with patch.dict(os.environ,{'C03_PREPARED_DIR':str(out)}):
            self.assertEqual(preflight_notebook('short/09',self.root)['status'],'BLOCKED')
            self.assertEqual(inspect_run_stage(load_run_plan(plan_path),'short/09')['status'],'BLOCKED')
        p.write_bytes(original)
        for p in [out/'preparation_manifest.json',paths[2]]:
            original=p.read_bytes();p.write_bytes(original+b' ')
            self.assertEqual(inspect_c03_package(out)['status'],'BLOCKED');p.write_bytes(original)

    def test_c03_query_content_identity(self):
        out=self.complete();p=out/'comparison_results/predictions_BLASTN_noise_00.csv'
        mp=out/'comparison_results/manifest_BLASTN_noise_00.json';original=p.read_bytes();manifest=mp.read_bytes()
        for col,value in [('query_id','other'),('label',2)]:
            frame=pd.read_csv(p);frame.loc[0,col]=value;frame.to_csv(p,index=False)
            obj=json.loads(manifest);obj['output_sha256']=sha(p);put(mp,obj)
            self.assertEqual(inspect_c03_package(out)['status'],'BLOCKED');p.write_bytes(original);mp.write_bytes(manifest)
        # IDs remain identical while the selected sequence changes, with a refreshed table hash.
        q=out/'queries_noise_00.csv';frame=pd.read_csv(q);frame.loc[0,'sequence']='T'*512;frame.to_csv(q,index=False)
        self.assertEqual(inspect_c03_package(out)['status'],'BLOCKED')

        # Coherent hashes and identical IDs do not permit a second noisy realization.
        # Reuse the independently specified case, not only the stale-hash mutation above.
        q.write_bytes((self.root/'prepared/queries_clean.csv').read_bytes())
        # The separate case builds its own fresh namespace to avoid the earlier mutations.
        with tempfile.TemporaryDirectory() as other:
            original_root=self.root;self.root=Path(other)
            try:self.test_c03_rejects_second_noise_realization()
            finally:self.root=original_root

    def test_c03_classes_reference_abstention(self):
        out=self.complete();p=out/'comparison_results/manifest_BLASTN_noise_00.json';original=p.read_bytes()
        for field,value in [('class_order_sha256','f'*64),('abstention_policy','discard unclassified'),('reference',{'path':'absent','sha256':'f'*64})]:
            self.mutate_manifest(out,field,value)
            self.assertEqual(inspect_c03_package(out)['status'],'BLOCKED');p.write_bytes(original)

    def test_c03_synthetic_dependency(self):
        out=self.complete();self.mutate_manifest(out,'is_smoke_test',True)
        self.assertEqual(inspect_c03_package(out)['status'],'BLOCKED')

        # Fresh packages for each dependency role; update enclosing hashes so mode is tested.
        for role in ['reference','nested_reference','build','preparation','query']:
            with tempfile.TemporaryDirectory() as tmp:
                out=package(Path(tmp));manifests=sorted((out/'comparison_results').glob('manifest_*.json'))
                prep=json.loads((out/'preparation_manifest.json').read_text());ref=Path(prep['reference_manifest'])
                db=ref.parent/'database_build_manifest.json'
                if role=='reference':
                    data=json.loads(ref.read_text());data['is_smoke_test']=True;put(ref,data)
                elif role=='nested_reference':
                    data=json.loads(ref.read_text());data['is_smoke_test']=True
                    nested=ref.parent/'nested.json';put(nested,data)
                    wrapper={**json.loads(ref.read_text()),'reference_manifest':str(nested),'reference_manifest_sha256':sha(nested)};put(ref,wrapper)
                elif role=='build':
                    data=json.loads(db.read_text());data['is_smoke_test']=True;put(db,data)
                elif role=='preparation':
                    prep['is_smoke_test']=True;put(out/'preparation_manifest.json',prep);bind_query_manifests(out)
                elif role=='query':
                    q=out/'query_manifest_noise_05.json';data=json.loads(q.read_text());data['is_smoke_test']=True;put(q,data)
                if role in ('reference','nested_reference'):
                    data=json.loads(db.read_text());data['reference_manifest_sha256']=sha(ref);put(db,data)
                for mp in manifests:
                    obj=json.loads(mp.read_text());obj['reference']['sha256']=sha(ref)
                    if obj['reference'].get('database_build_manifest'):obj['reference']['database_build_manifest']['sha256']=sha(db)
                    qp=Path(obj['query_manifest']['path']);obj['query_manifest']['sha256']=sha(qp);put(mp,obj)
                with self.subTest(role=role):self.assertEqual(inspect_c03_package(out)['status'],'BLOCKED')

    def test_complete_with_corrupted_product(self):
        out=self.complete();p=out/'comparison_results/predictions_BLASTN_noise_00.csv'
        p.write_text(p.read_text()+'\n')
        r=inspect_c03_package(out);self.assertEqual(r['status'],'BLOCKED');self.assertIn('hash',str(r).lower())

    def test_c03_preflight_entrypoints(self):
        from viral_cnn.notebook_runtime import preflight_notebook
        from viral_cnn.run_planning import inspect_run_stage
        out=self.complete()
        with patch.dict(os.environ,{'C03_PREPARED_DIR':str(out)}):
            self.assertEqual(preflight_notebook('short/09',self.root)['status'],'READY')
        plan={'protocol_id':'m00_run_plan_v1','schema_version':1,'is_smoke_test':False,'stages':[
            {'notebook_id':'short/09','run_id':'child','parent_run_id':'parent','action':'run','requirements':[],
             'dependencies':[{'source':'c03','kind':'external','manifest':str(out/'preparation_manifest.json'),
                              'sha256':sha(out/'preparation_manifest.json')}]}]}
        self.assertEqual(inspect_run_stage(plan,'short/09')['status'],'READY')
        (out/'comparison_results/manifest_BLASTN_noise_00.json').unlink()
        self.assertEqual(inspect_run_stage(plan,'short/09')['status'],'BLOCKED')

    def test_c03_missing_package(self):
        r=inspect_c03_package(self.root/'absent')
        self.assertEqual(r['status'],'BLOCKED');self.assertFalse((self.root/'absent').exists())

    def begin(self,stage='short/01',smoke=True,**settings):
        from viral_cnn.m02_workflow import begin_m02
        return begin_m02(stage,project_root=self.root,settings={'data_root':str(self.root),'is_smoke_test':smoke,**settings})

    def test_source_paths_and_hashes(self):
        p=self.root/'source.json';put(p,{'status':'COMPLETE'})
        with patch.dict(os.environ,{'C03_PREPARED_DIR':str(self.root)}):
            run=self.begin('short/09');self.assertEqual(run.configuration['c03_prepared_dir'],str(self.root))
            run.complete(input_files=[p]);m=json.loads(run.manifest_path.read_text())
            self.assertEqual(m['inputs'][0]['sha256'],sha(p));self.assertEqual(m['inputs'][0]['path'],str(p))

        from viral_cnn.notebook_runtime import preflight_notebook
        from viral_cnn.m02_workflow import begin_m02
        out=self.complete()
        with patch.dict(os.environ,{'C03_PREPARED_DIR':'prepared','LOG_PATH':'true'}), contextlib.redirect_stderr(io.StringIO()) as log:
            pre=preflight_notebook('short/09',self.root)
            selected=begin_m02('short/09',project_root=self.root,settings={'data_root':str(self.root)})
            self.assertEqual(pre['status'],'READY');self.assertEqual(pre['path'],str(out))
            self.assertEqual(selected.configuration['c03_prepared_dir'],str(out))
            self.assertIn(str(out),log.getvalue());self.assertIn(sha(out/'preparation_manifest.json'),log.getvalue())
        plan_path=self.plan_stage('short/09','selected-parent',smoke=False,c03=out)
        with patch.dict(os.environ,{'RUN_PLAN':str(plan_path)}):
            ns=self.execute09(out,stop=1)
            self.assertEqual(ns['run'].configuration['c03_prepared_dir'],str(out))
            self.assertEqual(ns['run'].configuration['run_plan_identity']['sha256'],sha(plan_path))

    def test_conflicting_plan_source(self):
        from viral_cnn.m02_workflow import begin_m02
        output=self.root/'results_reproduction/runs/child/short/09';output.mkdir(parents=True)
        plan={'protocol_id':'m00_run_plan_v1','schema_version':1,'parent_run_id':'parent','data_root':str(self.root),
              'is_smoke_test':False,'seed':42,'stages':[{'notebook_id':'short/09','run_id':'child','parent_run_id':'parent',
              'action':'run','output_dir':str(output),'manifest':str(output/'manifest_09_baselines.json'),
              'requirements':[],'dependencies':[{'source':'c03','manifest':str(self.root/'prepared/preparation_manifest.json')}]}]}
        path=self.root/'plan.json';put(path,plan)
        with patch.dict(os.environ,{'C03_PREPARED_DIR':str(self.root/'different')}),self.assertRaisesRegex(ValueError,'conflict'):
            begin_m02('short/09',project_root=self.root,run_plan=path)
        self.assertEqual(list(output.iterdir()),[])

    def test_three_notebook_reservations(self):
        for stage in ['short/01','short/02','short/09']:
            a=self.begin(stage);b=self.begin(stage)
            self.assertNotEqual(a.output_dir,b.output_dir)
            self.assertTrue(a.ctx.data_root.is_relative_to(a.output_dir))
            self.assertNotEqual(a.ctx.data_root,self.root)

        for stem in ['01','02','09']:
            first=self.execute09(smoke=True,stop=1) if stem=='09' else self.notebook(stem,stop=2)
            second=self.execute09(smoke=True,stop=1) if stem=='09' else self.notebook(stem,stop=2)
            self.assertNotEqual(first['run'].output_dir,second['run'].output_dir)
            pp=self.plan_stage('short/'+stem,'adopt-'+stem,smoke=True)
            with patch.dict(os.environ,{'RUN_PLAN':str(pp)}):
                ns=self.execute09(smoke=True,stop=1) if stem=='09' else self.notebook(stem,stop=2)
            self.assertEqual(ns['run'].run_id,'adopt-'+stem)
            self.assertEqual(ns['run'].manifest['parent_run_id'],'parent')
            self.assertEqual(ns['run'].output_dir,self.root/f'results_reproduction/runs/adopt-{stem}/short/{stem}')

    def test_collision_and_missing_inputs(self):
        from viral_cnn.m02_workflow import begin_m02
        a=self.begin(run_id='same');(a.output_dir/'keep').write_bytes(b'keep')
        with self.assertRaises(FileExistsError):self.begin(run_id='same')
        with self.assertRaises(FileNotFoundError):
            with a.guard():a.complete(input_files=[self.root/'missing'])
        self.assertEqual(json.loads(a.manifest_path.read_text())['status'],'FAILED')
        self.assertEqual((a.output_dir/'keep').read_bytes(),b'keep')

        for stem in ['01','02','09']:
            ns=self.execute09(smoke=True,stop=1) if stem=='09' else self.notebook(stem,stop=2)
            run=ns['run'];snapshot={str(p):p.read_bytes() for p in run.output_dir.rglob('*') if p.is_file()}
            with patch.dict(os.environ,{'M02_RUN_ID':run.run_id}),self.assertRaises(FileExistsError):
                self.execute09(smoke=True,stop=1) if stem=='09' else self.notebook(stem,stop=2)
            self.assertEqual(snapshot,{str(p):p.read_bytes() for p in run.output_dir.rglob('*') if p.is_file()})
            with self.assertRaises((FileNotFoundError,ValueError)):
                self.execute09() if stem=='09' else self.notebook(stem,smoke=False,stop=2)

    def test_manifest_identity_and_lineage(self):
        src=self.root/'source';src.write_bytes(b'input');ck=self.root/'ck';ck.write_bytes(b'weights')
        run=self.begin(population_id='fixture-pop',split_id='fixture-split')
        product=run.output_dir/'data.csv';product.write_bytes(b'a\n1\n')
        run.complete(input_files=[src],checkpoints={'fixture':str(ck)})
        m=json.loads(run.manifest_path.read_text())
        for key in ['protocol_id','status','notebook_id','run_id','parent_run_id','is_smoke_test','population_id',
                    'split_id','seed','git_commit','git_dirty','configuration','library_versions','inputs','checkpoints','outputs','statistical_scope']:
            self.assertIn(key,m)
        self.assertEqual(m['status'],'COMPLETE');self.assertEqual(m['protocol_id'],'m02_benchmark_audit_v1')
        self.assertEqual(m['population_id'],'fixture-pop');self.assertEqual(m['split_id'],'fixture-split');self.assertEqual(m['seed'],42)
        self.assertEqual(m['inputs'][0]['sha256'],sha(src));self.assertEqual(m['checkpoints']['fixture']['sha256'],sha(ck))
        rec=next(r for r in m['outputs'] if r['path']==str(product));self.assertEqual(rec['sha256'],sha(product));self.assertEqual(rec['relative_path'],'data.csv')

    def test_failed_run_manifest(self):
        for stage in ['short/01','short/02','short/09']:
            r=self.begin(stage)
            with self.assertRaisesRegex(RuntimeError,'intentional'):
                with r.guard():raise RuntimeError('intentional')
            m=json.loads(r.manifest_path.read_text());self.assertEqual(m['status'],'FAILED');self.assertEqual(m['error'],'intentional')

        for stem in ['01','02','09']:
            ns=self.execute09(smoke=True,stop=1) if stem=='09' else self.notebook(stem,stop=2)
            path=next((ROOT/'notebooks/reproduction/short').glob(stem+'_*.ipynb'))
            cells=[c for c in json.loads(path.read_text())['cells'] if c['cell_type']=='code']
            target=cells[1] if stem=='09' else cells[2]
            with patch('pandas.DataFrame.to_csv',side_effect=RuntimeError('forced product failure')),          patch.dict(ns,{'build_registered_model':lambda *a,**kw:(_ for _ in ()).throw(RuntimeError('forced model failure'))}),          self.assertRaises(RuntimeError), contextlib.redirect_stdout(io.StringIO()):
                exec(''.join(target['source']),ns)
            self.assertEqual(json.loads(ns['run'].manifest_path.read_text())['status'],'FAILED')

    def test_reservation_preserves_bytes(self):
        r=self.begin(run_id='immutable');p=r.output_dir/'keep';p.write_bytes(b'previous')
        before={str(p):p.read_bytes() for p in r.output_dir.rglob('*') if p.is_file()}
        with self.assertRaises(FileExistsError):self.begin(run_id='immutable')
        self.assertEqual(before,{str(p):p.read_bytes() for p in r.output_dir.rglob('*') if p.is_file()})

    def test_empirical_failure_never_creates_fixtures(self):
        before=set(self.root.rglob('*.parquet'))
        r=self.begin(smoke=False)
        with self.assertRaises(FileNotFoundError):
            with r.guard():r.complete(input_files=[self.root/'missing.parquet'])
        self.assertEqual(before,set(self.root.rglob('*.parquet')))

        sentinel=self.root/'historical.parquet';sentinel.write_bytes(b'untouched history')
        before={str(p):p.read_bytes() for p in self.root.rglob('*.parquet')}
        for stem in ['01','02','09']:
            with self.assertRaises((FileNotFoundError,ValueError)):
                self.execute09() if stem=='09' else self.notebook(stem,smoke=False,stop=2)
            self.assertEqual(before,{str(p):p.read_bytes() for p in self.root.rglob('*.parquet')})

    def plot_data(self):
        truth=np.array([0,0,1,1]);pred={m:{r:truth.copy() for r in RATES} for m in METHODS}
        pred['BLASTN'][0.]=np.array([0,-1,0,-1])
        masks={'all':np.ones(4,dtype=bool),'overlap':np.array([1,0,1,0],dtype=bool),
               'no_exact_overlap':np.array([0,1,0,1],dtype=bool)}
        return truth,pred,masks

    def test_full_metric_grid(self):
        from viral_cnn.baselines.benchmark_plotting import benchmark_summary
        truth,pred,masks=self.plot_data();df=benchmark_summary(truth,pred,masks)
        self.assertEqual(len(df),105);self.assertEqual(set(df.method),set(METHODS))
        row=df[(df.method=='BLASTN')&(df.noise_rate==0)&(df.stratum=='all')].iloc[0]
        self.assertEqual((row.n,row.n_covered,row.accuracy,row.coverage,row.conditional_accuracy),(4,2,.25,.5,.5))

        self.assertEqual(set(zip(df.method,df.noise_rate,df.stratum)),{(m,r,s) for m in METHODS for r in RATES for s in masks})
        self.assertFalse(df.duplicated(['method','noise_rate','stratum']).any())
        for row in df.itertuples():
            expected_n=4 if row.stratum=='all' else 2
            self.assertEqual(row.n,expected_n)
            expected=(.25,.5,.5) if row.method=='BLASTN' and row.noise_rate==0 and row.stratum=='all' else (.5,1.,.5) if row.method=='BLASTN' and row.noise_rate==0 and row.stratum=='overlap' else (0.,0.,np.nan) if row.method=='BLASTN' and row.noise_rate==0 else (1.,1.,1.)
            np.testing.assert_allclose([row.accuracy,row.coverage,row.conditional_accuracy],expected,equal_nan=True)

    def test_abstention_denominators(self):
        from viral_cnn.baselines.benchmark_plotting import benchmark_summary
        truth,pred,masks=self.plot_data()
        for rate in RATES:pred['BLASTN'][rate]=np.full(4,-1)
        df=benchmark_summary(truth,pred,masks);rows=df[df.method=='BLASTN']
        self.assertTrue((rows.accuracy==0).all());self.assertTrue((rows.coverage==0).all())
        self.assertTrue(rows.conditional_accuracy.isna().all());self.assertEqual(rows.n.sum(),40)

    def test_accuracy_coverage_panels(self):
        from viral_cnn.baselines.benchmark_plotting import benchmark_summary, render_curves
        import matplotlib.pyplot as plt
        truth,pred,masks=self.plot_data();df=benchmark_summary(truth,pred,masks)
        src=self.root/'summary.csv';df.to_csv(src,index=False)
        saved=[]
        from matplotlib.figure import Figure
        orig=Figure.savefig
        def capture(fig,*a,**kw):
            saved.append([(ax.get_title(),len(ax.lines),[line.get_label() for line in ax.lines]) for ax in fig.axes]);return orig(fig,*a,**kw)
        with patch.object(Figure,'savefig',capture):records=render_curves(src,self.root)
        self.assertEqual(len(records),2)
        for panels in saved:
            self.assertEqual(len(panels),3)
            for title,count,labels in panels:
                self.assertIn('N=',title);self.assertIn('Wilson 95%',title);self.assertEqual(count,7);self.assertEqual(set(labels),set(METHODS))
        self.assertFalse(plt.get_fignums())

        point=df[(df.method=='BLASTN')&(df.noise_rate==0)&(df.stratum=='all')].iloc[0]
        self.assertAlmostEqual(point.accuracy_ci_low,0.045586062644636216)
        self.assertAlmostEqual(point.accuracy_ci_high,0.6993639475573634)
        from matplotlib.figure import Figure
        # Assert actual plotted coordinates and Wilson polygon bounds, not only legends.
        orig=Figure.savefig
        seen=[]
        def coordinates(fig,*a,**kw):
            metric='coverage' if 'coverage' in str(a[0]) else 'accuracy'
            for ax,stratum in zip(fig.axes,['all','overlap','no_exact_overlap']):
                self.assertIn('N='+str(4 if stratum=='all' else 2),ax.get_title())
                self.assertEqual(len(ax.collections),7)
                for line,band in zip(ax.lines,ax.collections):
                    expected=df[(df.method==line.get_label())&(df.stratum==stratum)].sort_values('noise_rate')
                    np.testing.assert_allclose(line.get_xdata(),expected.noise_rate)
                    np.testing.assert_allclose(line.get_ydata(),expected[metric])
                    vertices=band.get_paths()[0].vertices
                    for rate,lo,hi in zip(expected.noise_rate,expected[metric+'_ci_low'],expected[metric+'_ci_high']):
                        ys=vertices[np.isclose(vertices[:,0],rate),1]
                        self.assertAlmostEqual(ys.min(),lo);self.assertAlmostEqual(ys.max(),hi)
            seen.append(metric);return orig(fig,*a,**kw)
        with patch.object(Figure,'savefig',coordinates):render_curves(src,self.root)
        self.assertEqual(set(seen),{'accuracy','coverage'})

    def test_csv_figure_regeneration_and_hashes(self):
        from viral_cnn.baselines.benchmark_plotting import benchmark_summary, render_curves
        truth,pred,masks=self.plot_data();run=self.begin('short/09')
        src=run.output_dir/'summary.csv';benchmark_summary(truth,pred,masks).to_csv(src,index=False)
        records=render_curves(src,run.output_dir)
        run.complete(input_files=[src]);m=json.loads(run.manifest_path.read_text());outputs={r['path']:r for r in m['outputs']}
        for r in records:
            self.assertEqual(r['source_sha256'],sha(src))
            for k,magic in [('png',b'\x89PNG'),('pdf',b'%PDF')]:
                p=Path(r[k]);self.assertTrue(p.read_bytes().startswith(magic));self.assertEqual(outputs[str(p)]['sha256'],sha(p))

        # The notebook must persist the links, not merely return a Python dict.
        ns=self.execute09(smoke=True);manifest=json.loads(ns['run'].manifest_path.read_text())
        products={r['path']:r for r in manifest['outputs']};links=ns['output_dir']/'figure_sources.json'
        self.assertEqual(products[str(links)]['sha256'],sha(links))
        for record in json.loads(links.read_text()):
            self.assertEqual(record['source_sha256'],sha(record['source_csv']))
            self.assertEqual(products[record['source_csv']]['sha256'],sha(record['source_csv']))
            for kind in ['png','pdf']:self.assertEqual(products[record[kind]]['sha256'],sha(record[kind]))

    def test_paired_difference_grid(self):
        from viral_cnn.baselines.benchmark_plotting import paired_differences
        truth,pred,masks=self.plot_data();df=paired_differences(truth,pred,masks)
        self.assertEqual(len(df),150)
        row=df[(df.method_a=='resnet50_species')&(df.method_b=='BLASTN')&(df.noise_rate==0)&(df.stratum=='all')].iloc[0]
        self.assertEqual(row.diff_mean_pp,75.);self.assertEqual(row.resampling_unit,'species')
        self.assertEqual(row.repeats,2000);self.assertEqual(row.seed,42)
        self.assertEqual(row.ci_status,'available');self.assertLessEqual(row.ci_lower_pp,75.);self.assertGreaterEqual(row.ci_upper_pp,75.)
        self.assertEqual(set(zip(df.method_a,df.method_b)),{(a,b) for a in METHODS[:5] for b in METHODS[-2:]})

        self.assertEqual(set(zip(df.method_a,df.method_b,df.noise_rate,df.stratum)),{(a,b,r,s) for a in METHODS[:5] for b in METHODS[-2:] for r in RATES for s in masks})
        self.assertFalse(df.duplicated(['method_a','method_b','noise_rate','stratum']).any())
        self.assertEqual((row.ci_lower_pp,row.ci_upper_pp),(50.,100.))
        for cell in df.itertuples():
            expected=75. if cell.method_b=='BLASTN' and cell.noise_rate==0 and cell.stratum=='all' else 50. if cell.method_b=='BLASTN' and cell.noise_rate==0 and cell.stratum=='overlap' else 100. if cell.method_b=='BLASTN' and cell.noise_rate==0 else 0.
            self.assertEqual(cell.diff_mean_pp,expected)
            self.assertEqual((cell.repeats,cell.seed,cell.resampling_unit),(2000,42,'species'))
        # Negative direction and unequal species support use literal expected bounds.
        truth=np.array([0,0,0,1]);pred={m:{r:truth.copy() for r in RATES} for m in METHODS}
        for rate in RATES:pred[METHODS[0]][rate]=np.array([1,1,1,1])
        masks={s:np.ones(4,dtype=bool) for s in ['all','overlap','no_exact_overlap']}
        negative=paired_differences(truth,pred,masks)
        cell=negative[(negative.method_a==METHODS[0])&(negative.method_b=='BLASTN')].iloc[0]
        self.assertEqual((cell.diff_mean_pp,cell.ci_lower_pp,cell.ci_upper_pp),(-75.,-100.,0.))

    def test_paired_unavailable_intervals(self):
        from viral_cnn.baselines.benchmark_plotting import paired_differences
        truth,pred,masks=self.plot_data();masks['overlap']=np.zeros(4,dtype=bool);masks['no_exact_overlap']=truth==0
        df=paired_differences(truth,pred,masks)
        empty=df[df.stratum=='overlap'];single=df[df.stratum=='no_exact_overlap']
        self.assertTrue((empty.ci_reason=='empty_stratum').all());self.assertTrue(empty.diff_mean_pp.isna().all())
        self.assertTrue((single.ci_reason=='fewer_than_two_species').all());self.assertTrue(single.diff_mean_pp.notna().all())
        self.assertTrue(df[df.stratum!='all'].ci_lower_pp.isna().all());self.assertTrue(df[df.stratum!='all'].ci_upper_pp.isna().all())

    def test_paired_figure_from_csv(self):
        from viral_cnn.baselines.benchmark_plotting import paired_differences, render_differences
        truth,pred,masks=self.plot_data();src=self.root/'deltas.csv';paired_differences(truth,pred,masks).to_csv(src,index=False)
        r=render_differences(src,self.root)
        self.assertEqual(r['source_sha256'],sha(src));self.assertTrue(Path(r['png']).read_bytes().startswith(b'\x89PNG'))
        self.assertTrue(Path(r['pdf']).read_bytes().startswith(b'%PDF'))

    def notebook(self,stem,*,smoke=True,stop=None,settings=None):
        """Execute actual cells with a tiny backbone; never use shared DATA_ROOT."""
        import torch
        torch.set_num_threads(1)
        from viral_cnn.models.resnet import OneHotEmbedding
        class Tiny(torch.nn.Module):
            def __init__(self,k):
                super().__init__();self.emb=OneHotEmbedding();self.layer4=torch.nn.Conv1d(4,8,3,padding=1);self.fc=torch.nn.Linear(8,k)
            def forward(self,x):return self.fc(self.layer4(self.emb(x)).mean(-1))
        path=next((ROOT/'notebooks/reproduction/short').glob(stem+'_*.ipynb'))
        config=self.root/'m02.yaml'
        import yaml
        config.write_text(yaml.safe_dump({'data_root':str(self.root),'is_smoke_test':smoke,**(settings or {})}))
        ns={'__name__':'__main__'}
        with patch.dict(os.environ,{'M02_PROJECT_ROOT':str(self.root),'M02_CONFIG':str(config),'SMOKE_TEST':'1' if smoke else '0'}), \
             patch('viral_cnn.models.registry.build_registered_model',side_effect=lambda n,**kw:Tiny(3 if n.endswith('_genus') else 5)), \
             patch('viral_cnn.training.profiler.profile_inference_model',return_value={'total_params':1,'avg_latency_ms':.1}), \
             contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
            for i,c in enumerate(json.loads(path.read_text())['cells']):
                if stop is not None and i>stop:break
                if c['cell_type']=='code':exec(''.join(c['source']),ns)
        return ns

    def test_c04_interval_contract(self):
        from viral_cnn.eval.uncertainty import bootstrap_classification_intervals
        obj=bootstrap_classification_intervals([0,0,1,1],[0,1,1,1],2,repetitions=2000,seed=42,confidence=.95)
        row=obj['estimates']['accuracy']
        self.assertEqual((row['resampling_unit'],row['strata'],row['repetitions'],row['seed'],row['ci_level']),
                         ('fragment','true_taxon_with_fixed_support',2000,42,.95));self.assertEqual(row['estimate'],.75)

        ns=self.notebook('01');data=json.loads((ns['output_dir']/'c04_uncertainty.json').read_text())
        self.assertEqual(data['protocol'],'c04_stratified_fragment_v1');self.assertEqual(len(data['rows']),35)
        for result in data['rows']:
            for metric in ['accuracy','macro_f1','weighted_f1','top5_accuracy']:
                published=result['uncertainty']['estimates'][metric]
                self.assertEqual((published['resampling_unit'],published['strata'],published['repetitions'],published['seed'],published['ci_level']),
                                 ('fragment','true_taxon_with_fixed_support',2000,42,.95))
                self.assertEqual(published['n'],64)
                field={'accuracy':'top1_accuracy','macro_f1':'macro_f1','weighted_f1':'weighted_f1','top5_accuracy':'top5_accuracy'}[metric]
                expected=result[field]/100 if metric in ('accuracy','top5_accuracy') else result[field]
                self.assertAlmostEqual(published['estimate'],expected)

    def test_notebook01_complete(self):
        ns=self.notebook('01');m=json.loads(ns['run'].manifest_path.read_text())
        self.assertEqual(m['status'],'COMPLETE');self.assertTrue(m['is_smoke_test']);self.assertEqual(len(ns['summary_rows']),35)
        self.assertEqual(m['statistical_scope']['noise_realizations_per_rate'],1)
        self.assertFalse(m['statistical_scope']['training_variability_estimated'])
        self.assertEqual(len(ns['c04_evidence_records']),35)
        self.assertFalse((self.root/'cnn_kraken').exists())
        self.assertTrue(all(r['sha256']==sha(r['path']) for r in m['outputs']))

    def test_notebook01_failure_isolated(self):
        with self.assertRaises(FileNotFoundError):self.notebook('01',smoke=False,stop=2)
        paths=list(self.root.glob('results_reproduction/runs/*/short/01/manifest.json'))
        self.assertEqual(len(paths),1);self.assertEqual(json.loads(paths[0].read_text())['status'],'FAILED')
        self.assertFalse(list(self.root.glob('cnn_kraken/**/*.parquet')))

    def test_m00_reserved_notebook01(self):
        from viral_cnn.m02_workflow import begin_m02
        output=self.root/'results_reproduction/runs/child/short/01';output.mkdir(parents=True)
        alternate=self.root/'alternate.parquet';pd.DataFrame({'sequence_padded':['A'*512],'label':[0]}).to_parquet(alternate,index=False)
        plan={'protocol_id':'m00_run_plan_v1','schema_version':1,'parent_run_id':'parent','data_root':str(self.root),
              'is_smoke_test':False,'seed':42,'population_id':'p','split_id':'s','stages':[
                  {'notebook_id':'short/01','run_id':'child','parent_run_id':'parent','action':'run','output_dir':str(output),
                   'manifest':str(output/'manifest.json'),'requirements':[{'path':str(alternate),'split':'test','topology_key':'short/species_parquet_dir'}],'dependencies':[]}]}
        pp=self.root/'plan.json';put(pp,plan)
        r=begin_m02('short/01',project_root=self.root,run_plan=pp)
        self.assertEqual(r.split_files('species','test'),[str(alternate)])
        self.assertEqual(r.run_id,'child');self.assertEqual(r.manifest['parent_run_id'],'parent')
        self.assertEqual(r.manifest['population_id'],'p');self.assertEqual(r.manifest['split_id'],'s')
        with self.assertRaises(FileExistsError):begin_m02('short/01',project_root=self.root,run_plan=pp)

    def test_validation_temperature_seal(self):
        import torch
        from viral_cnn.eval.calibration import TemperatureScaler
        logits=torch.tensor([[2.,0.],[0.,2.]])
        with self.assertRaises(ValueError):TemperatureScaler().fit(logits,[0,1],split_name='test')
        with self.assertRaises(ValueError):TemperatureScaler().fit(logits,[0,1],validation_provenance={'split':'validation','identity':'same'},test_provenance={'identity':'same'})
        s=TemperatureScaler().fit(logits,[0,1],validation_provenance={'split':'validation','identity':'validation-fixture'})
        s.seal();out=s.transform(logits)
        self.assertEqual(out['temperature_state'],'SEALED');self.assertEqual(s.provenance['criterion'],'validation_clean_cross_entropy')
        with self.assertRaises(RuntimeError):s.fit(logits,[0,1])

        ns,events=self.literal_calibration(watch=True)
        self.assertEqual(events.count('fit'),2);self.assertEqual(events.count('seal'),2);self.assertEqual(events.count('test'),2)
        self.assertEqual(events,['fit','seal','test','fit','seal','test'])
        self.assertTrue(all(r['temperature_state']=='SEALED' for r in ns['cal_rows']))
        with self.assertRaisesRegex(ValueError,'overlap'):self.literal_calibration(overlap=True)

    def test_reliability_support_and_n(self):
        from viral_cnn.eval.calibration import compute_reliability_diagram, save_reliability_comparison
        from matplotlib.figure import Figure
        bins=[b.to_dict() for b in compute_reliability_diagram(np.array([[.9,.1],[.7,.3]]),[0,1],n_bins=3)]
        titles=[];orig=Figure.savefig
        def capture(fig,*a,**kw):titles.append(fig.axes[0].get_title());return orig(fig,*a,**kw)
        with patch.object(Figure,'savefig',capture):r=save_reliability_comparison(bins,bins,self.root,title='fixture',filename_prefix='reliability',metadata={'n':2})
        self.assertTrue(all('N=2' in t for t in titles));self.assertTrue(r['png']);self.assertTrue(r['pdf'])
        df=pd.read_csv(r['csv']);self.assertEqual(df.groupby('state')['count'].sum().to_dict(),{'after':2,'before':2})
        with self.assertRaises(ValueError):save_reliability_comparison(bins,[],self.root,title='bad',filename_prefix='bad')

        ns,_=self.literal_calibration()
        m=json.loads(ns['run'].manifest_path.read_text());identities={r['path']:r for r in m['inputs']}
        for path in ns['species_test_files']+ns['species_val_files']:
            self.assertEqual(identities[str(Path(path).resolve())]['sha256'],sha(path))
        for row in ns['cal_rows']:
            self.assertAlmostEqual(row['ece'],.25,places=6);self.assertAlmostEqual(row['brier'],.625,places=6)
            self.assertAlmostEqual(row['nll'],-.5*(np.log(.75)+np.log(.25)),places=6)
            t=row['temperature_provenance']['temperature'];prob=1/(1+np.exp(-np.log(3)/t))
            self.assertAlmostEqual(row['calibrated_ece'],prob-.5,places=5)
            self.assertAlmostEqual(row['calibrated_brier'],(1-prob)**2+prob**2,places=5)
            self.assertAlmostEqual(row['calibrated_nll'],-.5*(np.log(prob)+np.log(1-prob)),places=5)
            self.assertEqual(sum(b['count'] for b in row['reliability_bins']['before']),64)
            self.assertEqual(sum(b['count'] for b in row['reliability_bins']['after']),64)
        outputs={r['path']:r for r in m['outputs']}
        for record in ns['figure_records']:
            meta=json.loads(Path(record['json']).read_text())['metadata'];self.assertEqual(meta['n'],64)
            for key in ['csv','json','png','pdf']:self.assertEqual(outputs[record[key]]['sha256'],sha(record[key]))

    def test_gradcam_smoke_only(self):
        import torch
        from types import SimpleNamespace
        p=next((ROOT/'notebooks/reproduction/short').glob('02_*.ipynb'))
        source=''.join(json.loads(p.read_text())['cells'][5]['source'])
        run=self.begin('short/02');ns={'run':run,'ctx':SimpleNamespace(is_smoke_test=False),'model':object(),'free_gpu_memory':lambda x:None,'json':json,'output_dir':run.output_dir}
        with patch('torch.randint',side_effect=AssertionError('random empirical GradCAM')),contextlib.redirect_stdout(io.StringIO()):exec(source,ns)
        self.assertEqual(json.loads((run.output_dir/'gradcam_status.json').read_text())['status'],'omitted_empirical')
        self.assertNotIn('sample_input',ns)

        ns=self.notebook('02',stop=5)
        self.assertIn('sample_input',ns);self.assertEqual(tuple(ns['sample_input'].shape),(1,512))
        self.assertEqual(json.loads((ns['output_dir']/'gradcam_status.json').read_text())['status'],'synthetic_demonstration')

    def test_notebook02_complete(self):
        ns=self.notebook('02');m=json.loads(ns['run'].manifest_path.read_text())
        self.assertEqual(m['status'],'COMPLETE');self.assertEqual(len(ns['cal_rows']),4)
        self.assertTrue(all(r['temperature_state']=='SEALED' for r in ns['cal_rows']))
        self.assertTrue(all(r['temperature_provenance']['criterion']=='validation_clean_cross_entropy' for r in ns['cal_rows']))
        self.assertTrue(all(r['n_samples']==64 for r in ns['cal_rows']))
        for row in ns['cal_rows']:
            for k in ['ece','brier','nll']:self.assertIn(k,row)
        self.assertEqual(json.loads((ns['output_dir']/'gradcam_status.json').read_text())['status'],'synthetic_demonstration')
        self.assertEqual(len(ns['figure_records']),4);self.assertTrue(all(r['png'] and r['pdf'] for r in ns['figure_records']))
        self.assertFalse((self.root/'cnn_kraken').exists())

    def test_uncertainty_scope(self):
        for stage in ['short/01','short/02','short/09']:
            run=self.begin(stage);run.complete();scope=json.loads(run.manifest_path.read_text())['statistical_scope']
            self.assertEqual(scope['noise_realizations_per_rate'],1);self.assertFalse(scope['training_variability_estimated'])
            self.assertEqual(scope['confidence'],.95);self.assertEqual(scope['seed'],42)

        for stem in ['01','02','09']:
            ns=self.execute09(smoke=True) if stem=='09' else self.notebook(stem)
            actual=json.loads(ns['run'].manifest_path.read_text())['statistical_scope']
            self.assertEqual(actual['noise_realizations_per_rate'],1);self.assertFalse(actual['training_variability_estimated'])
            self.assertEqual(actual['confidence'],.95);self.assertEqual(actual['seed'],42)

    def execute09(self,source=None,smoke=False,stop=None):
        path=next((ROOT/'notebooks/reproduction/short').glob('09_*.ipynb'))
        ns={'__name__':'__main__'}
        env={'M02_PROJECT_ROOT':str(self.root),'DATA_ROOT':str(self.root),'SMOKE_TEST':'1' if smoke else '0'}
        if source:env['C03_PREPARED_DIR']=str(source)
        with patch.dict(os.environ,env),contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
            for i,c in enumerate(json.loads(path.read_text())['cells']):
                if stop is not None and i>stop:break
                if c['cell_type']=='code':exec(''.join(c['source']),ns)
            self.assertEqual(os.environ['DATA_ROOT'],str(self.root));self.assertEqual(os.environ['SMOKE_TEST'],env['SMOKE_TEST'])
        return ns

    def test_notebook09_preserves_environment(self):
        ns=self.execute09(smoke=True)
        m=json.loads(ns['run'].manifest_path.read_text());self.assertEqual(m['status'],'COMPLETE');self.assertTrue(m['is_smoke_test'])
        self.assertEqual(ns['total_q'],40);self.assertEqual(len(ns['df_stratified']),105);self.assertEqual(len(ns['df_bootstrap']),150)

    def test_notebook09_consumes_without_inference(self):
        import torch
        out=self.complete()
        with patch('viral_cnn.models.registry.build_registered_model',side_effect=AssertionError('no model inference')), \
             patch('torch.load',side_effect=AssertionError('no checkpoint load')):
            ns=self.execute09(out)
        m=json.loads(ns['run'].manifest_path.read_text());self.assertEqual(m['status'],'COMPLETE')
        self.assertEqual(len(ns['method_records']),35);self.assertEqual(len(ns['df_stratified']),105)
        self.assertTrue((ns['df_stratified'].accuracy==1.).all());self.assertTrue((ns['df_bootstrap'].diff_mean_pp==0.).all())
        self.assertEqual(len(ns['figure_records']),3)
        inputs={r['path']:r for r in m['inputs']}
        self.assertEqual(inputs[str(out/'preparation_manifest.json')]['sha256'],sha(out/'preparation_manifest.json'))
        self.assertEqual(len(m['checkpoints']),5)

    def test_notebook09_invalid_package_fails(self):
        out=self.complete();self.mutate_manifest(out,'completion','FAILED')
        with self.assertRaises(ValueError):self.execute09(out)
        manifests=list(self.root.glob('results_reproduction/runs/*/short/09/manifest.json'))
        self.assertEqual(len(manifests),1);self.assertEqual(json.loads(manifests[0].read_text())['status'],'FAILED')
        self.assertFalse(list(self.root.glob('results_reproduction/runs/*/short/09/benchmark_metrics_stratified.csv')))

    def test_notebook09_missing_source(self):
        with self.assertRaises(FileNotFoundError):self.execute09()
        self.assertEqual(len(list(self.root.glob('results_reproduction/runs/*/short/09/manifest.json'))),1)
        self.assertFalse((self.root/'cnn_kraken').exists())

    def test_c03_rejects_second_noise_realization(self):
        import shutil
        out=self.complete();alt=self.root/'alternate';shutil.copytree(out,alt)
        table=alt/'queries_noise_05.csv';frame=pd.read_csv(table);frame.loc[0,'sequence']='T'*512;frame.to_csv(table,index=False)
        fasta=alt/'queries_noise_05.fasta';fasta.write_text(''.join(f'>{r.query_id}\n{r.sequence}\n' for r in frame.itertuples()))
        qm=alt/'query_manifest_noise_05.json';q=json.loads(qm.read_text());q.update(query_content_sha256=sha(table),fasta_sha256=sha(fasta));put(qm,q)
        mp=out/'comparison_results/manifest_BLASTN_noise_05.json';obj=json.loads(mp.read_text())
        obj['query_manifest']={'path':str(qm),'sha256':sha(qm)}
        obj['execution_details'].update(query_csv_path=str(table),query_csv_sha256=sha(table),query_fasta_path=str(fasta),query_fasta_sha256=sha(fasta));put(mp,obj)
        report=inspect_c03_package(out);self.assertEqual(report['status'],'BLOCKED');self.assertIn('shared',report['error'].lower())

    def plan_stage(self,stage,run_id,*,smoke,c03=None,seed=42):
        output=self.root/'results_reproduction/runs'/run_id/stage;output.mkdir(parents=True)
        manifest_name='manifest_09_baselines.json' if stage=='short/09' else 'manifest.json'
        plan={'protocol_id':'m00_run_plan_v1','schema_version':1,'parent_run_id':'parent','data_root':str(self.root),
              'is_smoke_test':smoke,'seed':seed,'population_id':'fixture-pop','split_id':'fixture-split','stages':[
                  {'notebook_id':stage,'run_id':run_id,'parent_run_id':'parent','action':'run','output_dir':str(output),
                   'manifest':str(output/manifest_name),'requirements':[],'dependencies':[]} ]}
        if c03:plan['stages'][0]['dependencies']=[{'source':'c03','kind':'external','manifest':str(c03/'preparation_manifest.json'),'sha256':sha(c03/'preparation_manifest.json')}]
        path=self.root/(run_id+'.json');put(path,plan);return path

    def literal_calibration(self,*,watch=False,overlap=False):
        """Execute notebook setup/calibration/publication with identified binary logits."""
        import torch
        from viral_cnn.eval.calibration import TemperatureScaler
        import viral_cnn.eval.calibration as calibration
        ns=self.notebook('02',stop=1);run=ns['run'];orig_prepare=run.prepare_smoke
        def fixtures():
            orig_prepare()
            for split in ['test','val']:
                path=run.ctx.data_root/f'cnn_kraken/processed_kraken_species_splited_parquet/{split}.parquet'
                frame=pd.read_parquet(path);frame['label']=np.arange(64)%2;frame.to_parquet(path,index=False)
        run.prepare_smoke=fixtures
        nb=json.loads(next((ROOT/'notebooks/reproduction/short').glob('02_*.ipynb')).read_text())
        with contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):exec(''.join(nb['cells'][2]['source']),ns)
        if overlap:
            shared=next(iter(ns['test_loader'].dataset._partition_ids.values()))
            ns['val_loader'].dataset._partition_ids['additional-partition']=shared
        class Literal(torch.nn.Module):
            def forward(self,x):return torch.tensor([np.log(3),0.],dtype=torch.float32).expand(len(x),-1)
        ns['build_registered_model']=lambda *a,**kw:Literal()
        ns.update(top_k_confused=[],species_model_variants=[],eval_model_name='resnet_clean_species',tax_path=None)
        events=[];last=[None];phase=['validation']
        original_fit=TemperatureScaler.fit;original_seal=TemperatureScaler.seal;original_noise=calibration.inject_noise
        def fit(s,*a,**kw):
            self.assertEqual(kw['split_name'],'validation')
            self.assertEqual(kw['validation_provenance']['split'],'validation')
            events.append('fit');last[0]=s;return original_fit(s,*a,**kw)
        def seal(s):
            result=original_seal(s);events.append('seal');return result
        class Observed:
            def __init__(self,loader,name):self.loader=loader;self.dataset=loader.dataset;self.name=name
            def __len__(self):return len(self.loader)
            def __iter__(inner):
                phase[0]=inner.name
                if inner.name=='test':
                    self.assertIsNotNone(last[0]);self.assertEqual(last[0].state,'SEALED');events.append('test')
                yield from inner.loader
        def noise(*a,**kw):
            self.assertEqual(phase[0],'test');return original_noise(*a,**kw)
        if watch:
            ns['val_loader']=Observed(ns['val_loader'],'validation');ns['test_loader']=Observed(ns['test_loader'],'test')
        with patch.object(TemperatureScaler,'fit',fit),patch.object(TemperatureScaler,'seal',seal),patch.object(calibration,'inject_noise',noise if watch else original_noise),contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
            exec(''.join(nb['cells'][7]['source']),ns)
            exec(''.join(nb['cells'][8]['source']),ns)
        return ns,events

    def test_pinned_seed_and_data_root(self):
        from viral_cnn.m02_workflow import begin_m02
        path=self.plan_stage('short/01','wrong-seed',smoke=False,seed=43)
        with self.assertRaisesRegex(ValueError,'seed'):begin_m02('short/01',project_root=self.root,run_plan=path)
        (self.root/'cnn_kraken').mkdir()
        child=self.root/'explicit-child';child.mkdir()
        run=begin_m02('short/01',project_root=self.root,settings={'data_root':str(child),'is_smoke_test':False})
        self.assertEqual(run.ctx.data_root,child);self.assertEqual(run.ctx.topology.root,child)
        self.assertEqual(run.configuration['data_root'],str(child));self.assertFalse((child/'cnn_kraken').exists())
