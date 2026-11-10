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
