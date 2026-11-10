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

    def test_paired_difference_grid(self):
        from viral_cnn.baselines.benchmark_plotting import paired_differences
        truth,pred,masks=self.plot_data();df=paired_differences(truth,pred,masks)
        self.assertEqual(len(df),150)
        row=df[(df.method_a=='resnet50_species')&(df.method_b=='BLASTN')&(df.noise_rate==0)&(df.stratum=='all')].iloc[0]
        self.assertEqual(row.diff_mean_pp,75.);self.assertEqual(row.resampling_unit,'species')
        self.assertEqual(row.repeats,2000);self.assertEqual(row.seed,42)
        self.assertEqual(row.ci_status,'available');self.assertLessEqual(row.ci_lower_pp,75.);self.assertGreaterEqual(row.ci_upper_pp,75.)
        self.assertEqual(set(zip(df.method_a,df.method_b)),{(a,b) for a in METHODS[:5] for b in METHODS[-2:]})

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

    def test_gradcam_smoke_only(self):
        import torch
        from types import SimpleNamespace
        p=next((ROOT/'notebooks/reproduction/short').glob('02_*.ipynb'))
        source=''.join(json.loads(p.read_text())['cells'][5]['source'])
        run=self.begin('short/02');ns={'run':run,'ctx':SimpleNamespace(is_smoke_test=False),'model':object(),'free_gpu_memory':lambda x:None,'json':json,'output_dir':run.output_dir}
        with patch('torch.randint',side_effect=AssertionError('random empirical GradCAM')),contextlib.redirect_stdout(io.StringIO()):exec(source,ns)
        self.assertEqual(json.loads((run.output_dir/'gradcam_status.json').read_text())['status'],'omitted_empirical')
        self.assertNotIn('sample_input',ns)

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

    def execute09(self,source=None,smoke=False):
        path=next((ROOT/'notebooks/reproduction/short').glob('09_*.ipynb'))
        ns={'__name__':'__main__'}
        env={'M02_PROJECT_ROOT':str(self.root),'DATA_ROOT':str(self.root),'SMOKE_TEST':'1' if smoke else '0'}
        if source:env['C03_PREPARED_DIR']=str(source)
        with patch.dict(os.environ,env),contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
            for c in json.loads(path.read_text())['cells']:
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
