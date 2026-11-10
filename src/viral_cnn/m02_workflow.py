"""M02 coordination: isolated runs, explicit inputs and atomic manifests."""
from __future__ import annotations

import json
import os
import subprocess
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import yaml
from viral_cnn.notebook_runtime import (file_identity, get_git_commit, get_library_versions,
    reserve_run_directory, setup_notebook_runtime, resolve_split_files, ensure_smoke_test_artifacts)
from viral_cnn.run_planning import load_run_plan
from viral_cnn.utils.logging_utils import log_artifact_path

STATISTICAL_SCOPE = {'noise_realizations_per_rate':1, 'training_variability_estimated':False,
    'c04':'stratified fragment bootstrap; fixed true-taxon support',
    'c03':'paired species cluster bootstrap; Wilson individual intervals',
    'confidence':.95, 'seed':42, 'repetitions':2000, 'genome_identity':'unknown'}


def _atomic(path: Path, data: dict[str, Any]) -> None:
    temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(data,indent=2,allow_nan=False,default=str)+'\n')
    os.replace(temp,path)


def _resolved(value: str | Path, root: Path) -> Path:
    path=Path(value).expanduser()
    return (path if path.is_absolute() else root/path).resolve()


class M02Run:
    """Reserved run whose guard records Python failures without changing inputs."""

    def __init__(self, output: Path, manifest: Path, configuration: dict[str, Any],
                 run_id: str, parent_run_id: str | None, stage: str) -> None:
        self.output_dir=output;self.manifest_path=manifest;self.configuration=configuration
        smoke=configuration['is_smoke_test']
        data=output/'fixtures' if smoke else Path(configuration['data_root'])
        self.ctx=setup_notebook_runtime(data_root=data,smoke_test=smoke,auto_smoke_fallback=False)
        for req in configuration.get('requirements',[]):
            key=req.get('topology_key')
            if key and not smoke:
                section,name=key.split('/')
                if not req.get('split'):self.ctx.topology.paths[section][name]=req['path']
        self.run_id=run_id
        try:
            dirty=bool(subprocess.check_output(['git','status','--porcelain'],stderr=subprocess.DEVNULL).strip())
        except (OSError,subprocess.CalledProcessError):dirty=None
        self.manifest={'protocol_id':'m02_benchmark_audit_v1','status':'RUNNING','notebook_id':stage,
            'run_id':run_id,'parent_run_id':parent_run_id,'is_smoke_test':smoke,
            'population_id':configuration.get('population_id','unknown'),'split_id':configuration.get('split_id','unknown'),
            'seed':42,'git_commit':get_git_commit(),'git_dirty':dirty,'configuration':configuration,
            'library_versions':get_library_versions(),'inputs':[],'checkpoints':{},'outputs':[],
            'scientific_reproduction_certified':False,'statistical_scope':STATISTICAL_SCOPE,
            'evidence':{'status':'synthetic' if smoke else 'observed'}}
        _atomic(manifest,self.manifest)

    def split_files(self, rank: str, split: str) -> list[str]:
        """Resolve a split selected in the plan, or the configured topology."""
        normalized='validation' if split=='val' else split
        key=f'short/{rank}_parquet_dir'
        for req in self.configuration.get('requirements',[]):
            if req.get('topology_key')==key and req.get('split') in (split,normalized):
                return resolve_split_files(req['path'],split)
        explicit=self.configuration.get('inputs',{}).get(f'{rank}_{normalized}')
        path=explicit or self.ctx.topology.paths['short'][f'{rank}_parquet_dir']
        return resolve_split_files(path,split)

    def prepare_smoke(self) -> None:
        """Create fixtures only under this run, with distinct validation identity."""
        if not self.ctx.is_smoke_test:raise ValueError('No synthetic artifacts in empirical mode')
        ensure_smoke_test_artifacts(self.ctx.data_root, num_samples_per_split=64)
        import pandas as pd
        for rank in ('species','genus'):
            path=self.ctx.data_root/f'cnn_kraken/processed_kraken_{rank}_splited_parquet/val.parquet'
            log_artifact_path("M02 smoke validation",path)
            frame=pd.read_parquet(path)
            frame['sequence_padded']=frame.sequence_padded.str.translate(str.maketrans('ACGT','CGTA'))
            frame.to_parquet(path,index=False)

    @contextmanager
    def guard(self) -> Iterator[None]:
        """Record FAILED for exceptions in a notebook cell or workflow."""
        try:yield
        except BaseException as exc:
            self.manifest.update(status='FAILED',error=str(exc));_atomic(self.manifest_path,self.manifest)
            raise

    def complete(self, *, input_files: list[str | Path] | None = None,
                 checkpoints: dict[str,str] | None = None) -> dict[str, Any]:
        """Hash real inputs and all products, then atomically publish COMPLETE."""
        if self.manifest['status']!='RUNNING':raise RuntimeError('Run is not RUNNING')
        records=[]
        for value in input_files or []:
            path=Path(value).resolve();log_artifact_path('M02 input',path)
            if not path.exists():raise FileNotFoundError(path)
            for p in sorted(path.rglob('*')) if path.is_dir() else [path]:
                if p.is_file():records.append(file_identity(p))
        ck={}
        for variant,value in (checkpoints or {}).items():
            p=Path(value);log_artifact_path('M02 checkpoint',p)
            if not p.is_file():raise FileNotFoundError(p)
            ck[variant]=file_identity(p)
        products=[{**file_identity(p),'relative_path':str(p.relative_to(self.output_dir))}
                  for p in sorted(self.output_dir.rglob('*')) if p.is_file()
                  and p!=self.manifest_path and not p.name.startswith('.m02')
                  and not p.is_relative_to(self.output_dir/'fixtures')]
        self.manifest.update(inputs=records,checkpoints=ck,outputs=products,status='COMPLETE')
        _atomic(self.manifest_path,self.manifest)
        return self.manifest


def begin_m02(notebook_id: str, *, project_root: str | Path | None = None,
              settings: dict[str, Any] | None = None, run_plan: str | Path | None = None) -> M02Run:
    """Resolve explicit environment/configuration or pinned M00 inputs and reserve.

    Smoke fixtures always use the newly reserved namespace, never shared DATA_ROOT.
    """
    if notebook_id not in ('short/01','short/02','short/09'):raise ValueError('Unknown M02 stage')
    root=Path(project_root or os.getenv('M02_PROJECT_ROOT',Path(__file__).resolve().parents[2])).resolve()
    config={}
    if os.getenv('M02_CONFIG'):
        p=Path(os.environ['M02_CONFIG']);log_artifact_path('M02 config',p)
        config=yaml.safe_load(p.read_text()) or {}
        config['config_identity']=file_identity(p)
    config.update(settings or {})
    source_plan=run_plan or os.getenv('RUN_PLAN')
    plan=load_run_plan(source_plan) if source_plan else None
    stage=None
    if plan:
        if plan.get('protocol_id')!='m00_run_plan_v1':raise ValueError('Expected M00 plan')
        matches=[s for s in plan['stages'] if s['notebook_id']==notebook_id]
        if len(matches)!=1 or matches[0].get('action')!='run':raise ValueError('Stage must be selected once with action run')
        stage=matches[0]
        pinned={**stage.get('m02',{}),'data_root':plan['data_root'],'is_smoke_test':plan['is_smoke_test'],
                'population_id':plan.get('population_id','unknown'),'split_id':plan.get('split_id','unknown')}
        for d in stage.get('dependencies',[]):
            if d.get('source')=='c03':
                pinned['c03_prepared_dir']=str(Path(d['manifest']).parent)
                if d.get('sha256','unknown')!='unknown' and file_identity(d['manifest'])['sha256']!=d['sha256']:
                    raise ValueError('M00 C03 manifest hash conflict')
        for key,env in [('data_root','DATA_ROOT'),('c03_prepared_dir','C03_PREPARED_DIR')]:
            val=os.getenv(env,config.get(key))
            if val is not None and (key not in pinned or _resolved(val,root)!=_resolved(pinned[key],root)):
                raise ValueError(f'M00 input conflict: {key}')
        if os.getenv('SMOKE_TEST') is not None and (os.environ['SMOKE_TEST'].lower() in ('1','true','yes'))!=pinned['is_smoke_test']:
            raise ValueError('M00 smoke mode conflict')
        pinned['requirements']=stage.get('requirements',[])
        config=pinned;config['run_plan_identity']=file_identity(source_plan)
    else:
        # Runtime supplies the remote topology default; fallback is explicitly disabled.
        ctx=setup_notebook_runtime(data_root=config.get('data_root'),auto_smoke_fallback=False)
        config['data_root']=str(_resolved(os.getenv('DATA_ROOT',config.get('data_root',ctx.data_root)),root))
        config.setdefault('is_smoke_test',ctx.is_smoke_test)
        if os.getenv('C03_PREPARED_DIR'):config['c03_prepared_dir']=str(_resolved(os.environ['C03_PREPARED_DIR'],Path(config['data_root'])))
    if config.get('c03_prepared_dir'):
        config['c03_prepared_dir']=str(_resolved(config['c03_prepared_dir'],Path(config['data_root'])))
    if config.get('seed',42)!=42:raise ValueError('M02 preserves seed 42')
    if config.get('c03_prepared_dir'):log_artifact_path('M02 selected C03',config['c03_prepared_dir'])
    if stage:
        run_id=stage['run_id']
        if not run_id or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_.' for c in run_id) or run_id in ('.','..'):
            raise ValueError('Invalid M00 run_id')
        parent=stage['parent_run_id'];output=Path(stage['output_dir']).resolve()
        expected=(root/'results_reproduction/runs'/run_id/notebook_id).resolve()
        manifest=Path(stage['manifest']).resolve()
        if output!=expected or manifest.parent!=output or parent!=plan['parent_run_id']:raise ValueError('Invalid M00 namespace')
        if not output.is_dir() or any(output.iterdir()):raise FileExistsError(output)
        with (output/'.m02_claim').open('x') as f:f.write(run_id)
    else:
        run_id=config.get('run_id') or os.getenv('M02_RUN_ID') or os.getenv('C04_RUN_ID') or 'm02-'+uuid.uuid4().hex[:16]
        output=reserve_run_directory(root,run_id,notebook_id);parent=None;manifest=output/'manifest.json'
    return M02Run(output,manifest,config,run_id,parent,notebook_id)
