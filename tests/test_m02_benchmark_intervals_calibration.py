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
                  'sequence':['A'*512,'G'*512,'C'*512,'T'*512]}).to_csv(test,index=False)
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

    def test_c03_classes_reference_abstention(self):
        out=self.complete();p=out/'comparison_results/manifest_BLASTN_noise_00.json';original=p.read_bytes()
        for field,value in [('class_order_sha256','f'*64),('abstention_policy','discard unclassified'),('reference',{'path':'absent','sha256':'f'*64})]:
            self.mutate_manifest(out,field,value)
            self.assertEqual(inspect_c03_package(out)['status'],'BLOCKED');p.write_bytes(original)

    def test_c03_synthetic_dependency(self):
        out=self.complete();self.mutate_manifest(out,'is_smoke_test',True)
        self.assertEqual(inspect_c03_package(out)['status'],'BLOCKED')

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

    def test_collision_and_missing_inputs(self):
        from viral_cnn.m02_workflow import begin_m02
        a=self.begin(run_id='same');(a.output_dir/'keep').write_bytes(b'keep')
        with self.assertRaises(FileExistsError):self.begin(run_id='same')
        with self.assertRaises(FileNotFoundError):
            with a.guard():a.complete(input_files=[self.root/'missing'])
        self.assertEqual(json.loads(a.manifest_path.read_text())['status'],'FAILED')
        self.assertEqual((a.output_dir/'keep').read_bytes(),b'keep')

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
