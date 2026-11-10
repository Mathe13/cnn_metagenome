"""M01 notebook workflows: explicit inputs, isolated outputs, honest provenance.

This workflow layer reuses the scientific auditors. It never downloads data,
loads a checkpoint, repairs class identities or silently changes populations.
"""
from __future__ import annotations

import csv
import json
import os
import subprocess
import uuid
from importlib.metadata import PackageNotFoundError, version
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from viral_cnn.data.retraining_audit import (
    _normalized, _iter_rows, _schema, _sequence_quality,
    audit_retraining_need, split_files,
)
from viral_cnn.data.taxonomy_split_audit import audit_taxonomy_against_overlap
from viral_cnn.notebook_runtime import (
    JupyterHubDataTopology, file_identity, get_git_commit, get_library_versions,
    reserve_run_directory, resolve_split_files, setup_notebook_runtime,
)
from viral_cnn.run_planning import inventory_reference, load_run_plan
from viral_cnn.taxonomy.curation import (
    ICTV_RANKS, build_hybrid_taxonomy, generate_synthetic_taxonomy,
    validate_taxonomy_identity,
)
from viral_cnn.utils.logging_utils import log_artifact_path

ENV_PATHS = {'original_mapping':'ORIGINAL_MAPPING_PATH', 'taxonomy_path':'TAXONOMY_PATH',
             'ictv_master':'ICTV_MASTER_PATH', 'overlap_manifest':'OVERLAP_MANIFEST',
             'overlap_dir':'OVERLAP_DIR', 'work_dir':'AUDIT_WORK_DIR', 'ncbi_taxonomy':'NCBI_TAXONOMY_PATH'}
PROTOCOLS = {'short/10':'m01_fragment_split_audit_v1', 'short/11':'m01_taxonomy_split_audit_v1'}


def _path(value: str | Path, root: Path) -> str:
    p = Path(value).expanduser()
    return str((p if p.is_absolute() else root / p).resolve())


def _json(path: str | Path) -> dict[str, Any]:
    log_artifact_path('M01 JSON', Path(path).resolve())
    return json.loads(Path(path).read_text(encoding='utf-8'))


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + '.tmp')
    try:
        with temporary.open('x', encoding='utf-8') as f:
            json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False)
            f.write('\n')
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _table(path: str | Path) -> pd.DataFrame:
    log_artifact_path('M01 table', Path(path).resolve())
    return pd.read_csv(path) if Path(path).suffix.lower() == '.csv' else pd.read_parquet(path)


def _reject_synthetic_taxonomy(taxonomy: pd.DataFrame, smoke: bool) -> None:
    if (not smoke and 'taxonomy_source' in taxonomy
            and taxonomy['taxonomy_source'].astype(str).str.contains('synthetic',case=False).any()):
        raise ValueError('Synthetic taxonomy cannot be an empirical M01 dependency')


def resolve_m01_inputs(
    notebook_id: str, project_root: str | Path, *, config_path: str | Path | None = None,
    settings: dict[str, Any] | None = None, run_plan: str | Path | None = None,
) -> dict[str, Any]:
    """Resolve configuration and environment; a selected M00 plan pins inputs.

    Relative input paths use DATA_ROOT; output reservations use project_root.
    No fixtures or output directories are created by resolution.
    """
    if notebook_id not in ('short/10', 'short/11', 'short/05'):
        raise ValueError(f'Unsupported M01 notebook: {notebook_id}')
    root = Path(project_root).resolve()
    config: dict[str, Any] = {}
    if config_path:
        log_artifact_path('M01 configuration', config_path)
        config = yaml.safe_load(Path(config_path).read_text())
        if not isinstance(config, dict):
            raise ValueError('M01 configuration must be a mapping')
    config.update(settings or {})
    plan = load_run_plan(run_plan) if run_plan else None
    stage = None
    if plan:
        if plan.get('protocol_id') != 'm00_run_plan_v1':
            raise ValueError('Expected M00 run plan')
        matches = [s for s in plan['stages'] if s['notebook_id'] == notebook_id]
        if len(matches) != 1 or matches[0].get('action', 'run') != 'run':
            raise ValueError('M01 stage must be selected exactly once with action=run')
        stage = matches[0]
        if plan.get('seed',42) != 42:
            raise ValueError('M01 preserves seed 42; M00 plan conflicts')
        pinned = dict(stage.get('m01', {}))
        pinned['data_root'] = plan['data_root']
        for req in stage.get('requirements', []):
            if req.get('role'):
                if req['role'] in ('train','validation','test'):
                    files = resolve_split_files(req['path'], req['split']) if req.get('split') else [req['path']]
                    pinned.setdefault('splits', {})[req['role']] = files
                else:
                    pinned[req['role']] = req['path']
        for dep in stage.get('dependencies', []):
            if dep.get('source') == 'short/10':
                pinned['overlap_manifest'] = dep['manifest']
        overrides = dict(config)
        overrides.update({k:os.environ[e] for k,e in ENV_PATHS.items() if os.getenv(e)})
        if os.getenv('DATA_ROOT'):overrides['data_root'] = os.environ['DATA_ROOT']
        if os.getenv('M01_MODE'):overrides['mode'] = os.environ['M01_MODE']
        if os.getenv('MAX_TEST_ROWS'):overrides['max_test_rows'] = int(os.environ['MAX_TEST_ROWS'])
        for split, env in [('train','TRAIN_SPLIT_PATH'),('validation','VALIDATION_SPLIT_PATH'),('test','TEST_SPLIT_PATH')]:
            if os.getenv(env):
                if pinned.get('splits',{}).get(split) != os.environ[env]:
                    raise ValueError(f'Override conflicts with M00 pinned split: {split}')
        for k,v in overrides.items():
            if k not in pinned or v != pinned[k]:
                raise ValueError(f'Override conflicts with M00 pinned input: {k}')
        config = pinned
    else:
        config.update({k:os.environ[e] for k,e in ENV_PATHS.items() if os.getenv(e)})
        config['data_root'] = os.getenv('DATA_ROOT', str(config.get('data_root', '.')))
        if os.getenv('M01_MODE'):config['mode'] = os.environ['M01_MODE']
        if os.getenv('MAX_TEST_ROWS'):config['max_test_rows'] = int(os.environ['MAX_TEST_ROWS'])
    data = Path(_path(config.get('data_root', '.'), root))
    topology = JupyterHubDataTopology(data)
    defaults = {'original_mapping':topology.paths['short']['species_csv'],
                'taxonomy_path':topology.paths['short']['hybrid_taxonomy_parquet'],
                'ictv_master':topology.paths['short']['ictv_master']}
    for key, default in defaults.items():config[key] = _path(config.get(key,default), data)
    for key in ('overlap_manifest','overlap_dir','work_dir','ncbi_taxonomy'):
        if config.get(key):config[key] = _path(config[key],data)
    if Path(config['taxonomy_path']).is_dir():
        directory = Path(config['taxonomy_path'])
        candidates = [directory/n for n in ('main_taxonomy_hybrid_CLEANSED.parquet','main_taxonomy_hybrid_ictv_ncbi.parquet')]
        config['taxonomy_path'] = str(next((p for p in candidates if p.is_file()),candidates[0]))
    if notebook_id == 'short/10':
        splits = dict(config.get('splits', {}))
        for split, env in [('train','TRAIN_SPLIT_PATH'),('validation','VALIDATION_SPLIT_PATH'),('test','TEST_SPLIT_PATH')]:
            if not plan and os.getenv(env):splits[split] = os.environ[env]
            if split not in splits:
                splits[split] = resolve_split_files(topology.paths['short']['species_parquet_dir'],split)
            values = splits[split] if isinstance(splits[split],list) else [splits[split]]
            resolved = [_path(v,data) for v in values]
            splits[split] = resolved if isinstance(splits[split],list) else resolved[0]
        config['splits'] = splits
    config['data_root'] = str(data)
    config.setdefault('mode','curate');config.setdefault('seed',42)
    if config['seed'] != 42:raise ValueError('M01 preserves seed 42')
    if stage:config['_stage'] = stage;config['_plan'] = plan
    for key in ('original_mapping','taxonomy_path','ictv_master','overlap_manifest','overlap_dir','ncbi_taxonomy'):
        if config.get(key):log_artifact_path('M01 selected '+key,config[key])
    return config


def _reserve(root: Path, key: str, config: dict[str, Any]) -> tuple[str, str | None, Path]:
    stage = config.get('_stage')
    if stage:
        run_id = stage['run_id']
        if not run_id or run_id in ('.','..') or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_.' for c in run_id):
            raise ValueError('Invalid M00 run_id')
        output = Path(stage['output_dir']).resolve()
        expected = root/'results_reproduction/runs'/run_id/key
        if (not output.is_relative_to((root/'results_reproduction/runs').resolve())
                or Path(stage['manifest']).resolve() != output/'manifest.json'
                or stage['parent_run_id'] != config['_plan']['parent_run_id']):
            raise ValueError('M00 stage namespace/manifest/parent identity mismatch')
        if output != expected.resolve() or not output.is_dir() or any(output.iterdir()):
            raise FileExistsError(f'M00 reservation is nonempty, missing or outside namespace: {output}')
        # Exclusive claim prevents two processes adopting the same empty reservation.
        with (output/'.m01_claim').open('x') as f:f.write(run_id)
        return run_id,stage['parent_run_id'],output
    run_id = f'm01-{uuid.uuid4().hex[:16]}'
    return run_id,None,reserve_run_directory(root,run_id,key)


def _files(paths: str | list[str]) -> list[Path]:
    values = paths if isinstance(paths,list) else [paths]
    return [f for path in values for f in split_files(path)]


def _split_counts(config: dict[str, Any], output: Path, original: pd.DataFrame) -> list[Path]:
    counts,classes,lengths,files = [],[],[],[]
    allowed = set((original.id - 1).astype(int))
    seen: set[Path] = set()
    resolved = {split:_files(paths) for split,paths in config['splits'].items()}
    schemas = {split:_schema(partitions) for split,partitions in resolved.items()}
    for partitions in resolved.values():
        if seen.intersection(partitions):raise ValueError('A file is assigned to multiple splits')
        if len(partitions) != len(set(partitions)):raise ValueError('Duplicate partition within split')
        seen.update(partitions);files.extend(partitions)
    for split,partitions in resolved.items():
        quality: Counter[str] = Counter(); by_class: Counter[int] = Counter(); by_length: Counter[int] = Counter()
        n = 0
        for raw in _iter_rows(partitions,schemas[split]):
            seq,label,_ = _normalized(*raw)
            try:idx = int(label)
            except ValueError as exc:raise ValueError(f'Invalid model label {label}') from exc
            if str(idx) != label or idx not in allowed:raise ValueError(f'Class label outside original mapping: {label}')
            n+=1;by_class[idx]+=1;by_length[len(seq)]+=1;_sequence_quality(seq,quality)
        counts.append({'split':split,'rows':n,**{k:quality[k] for k in ['empty_after_padding','no_acgt','over_512_bp','outside_ACGTN','outside_IUPAC','benchmark_compatible']}})
        classes.extend({'split':split,'model_label':idx,'rows':by_class[idx]} for idx in sorted(allowed))
        lengths.extend({'split':split,'length':length,'rows':n} for length,n in sorted(by_length.items()))
    for name,rows in [('split_counts',counts),('split_class_counts',classes),('split_lengths',lengths)]:
        pd.DataFrame(rows).to_csv(output/(name+'.csv'),index=False)
    return files


def _predecessor(config: dict[str, Any], smoke: bool) -> tuple[Path, Path, dict[str, Any]]:
    selected = config.get('overlap_manifest')
    if not selected and config.get('overlap_dir'):
        selected = str(Path(config['overlap_dir'])/'manifest.json')
    if not selected:raise ValueError('Select overlap_manifest from short/10 (legacy reports are not M01 evidence)')
    source = Path(selected);m = _json(source)
    if m.get('status') != 'COMPLETE' or m.get('protocol_id') != PROTOCOLS['short/10']:
        raise ValueError('Expected COMPLETE M01 short/10 manifest')
    if m.get('is_smoke_test') and not smoke:raise ValueError('Synthetic dependency forbidden in empirical M01')
    records = {Path(r['path']).name:r for r in m['outputs']}
    for record in m['outputs']:
        actual = file_identity(record['path'])
        if actual['sha256'] == 'unknown' or actual['sha256'] != record['sha256']:
            raise ValueError(f'Overlap dependency identity mismatch: {record["path"]}')
    original = file_identity(config['original_mapping'])
    if not any(r.get('sha256') == original['sha256'] and r.get('path') == original['path'] for r in m['inputs']):
        raise ValueError('Overlap uses a different original class mapping')
    for dep in config.get('_stage',{}).get('dependencies',[]):
        if dep.get('source') == 'short/10' and dep.get('sha256','unknown') != 'unknown':
            if file_identity(source)['sha256'] != dep['sha256']:raise ValueError('M00 predecessor manifest changed')
    decision,per = Path(records['decision.json']['path']),Path(records['per_species.csv']['path'])
    if _json(decision).get('scope') != 'full_test':raise ValueError('Taxonomic audit requires full_test, not pilot')
    return decision,per,{**file_identity(source),'is_smoke_test':m['is_smoke_test']}


def _curate(config: dict[str, Any], output: Path, smoke: bool) -> tuple[dict[str, Any],list[Path]]:
    from viral_cnn.data.audit import audit_partition_duplicates

    mode = config['mode']
    if mode not in ('curate','audit_existing'):raise ValueError('M01_MODE must be curate or audit_existing')
    inputs: list[Path] = []
    if smoke and mode == 'curate':
        taxonomy = generate_synthetic_taxonomy(100)
        original = taxonomy[['id','species']].copy()
        _reject_synthetic_taxonomy(taxonomy,smoke)
        report = validate_taxonomy_identity(taxonomy,original)
    else:
        mapping = Path(config['original_mapping']);original = _table(mapping);inputs.append(mapping)
        identity = validate_taxonomy_identity(original,original,require_model_label=False)
        if not identity['mapping_integrity_ok']:raise ValueError(identity['mapping_errors'])
        if mode == 'audit_existing':
            path = Path(config['taxonomy_path']);taxonomy = _table(path);inputs.append(path)
        else:
            path = Path(config['ictv_master']);log_artifact_path('ICTV workbook',path)
            if not path.is_file():raise FileNotFoundError(f'ICTV workbook missing: {path}; select audit_existing explicitly to audit a reference')
            inputs.append(path)
            master = pd.read_excel(path,sheet_name='MSL')
            renames = pd.read_excel(path,sheet_name='Taxa Renamed or Abolished')
            if not set(ICTV_RANKS).issubset(master.columns):raise ValueError('ICTV MSL is missing required rank columns')
            ncbi = _table(config['ncbi_taxonomy']) if config.get('ncbi_taxonomy') else None
            if ncbi is not None:inputs.append(Path(config['ncbi_taxonomy']))
            taxonomy = build_hybrid_taxonomy(original,master,renames,ncbi_df=ncbi,sanitize=False)
            if 'model_label' not in taxonomy:taxonomy['model_label'] = taxonomy.id - 1
        _reject_synthetic_taxonomy(taxonomy,smoke)
        report = validate_taxonomy_identity(taxonomy,original)
    report.update(num_species=len(taxonomy),reconstructed_from_msl=mode=='curate' and not smoke,
                  genome_provenance='unknown',genome_provenance_complete=False,genome_disjoint_test_classes=None)
    if config.get('overlap_manifest') or config.get('overlap_dir'):
        decision,per,dep = _predecessor(config,smoke)
        inputs += [decision,per,Path(dep['path'])]
        report['selected_decision'] = str(decision)
        # Fragment provenance belongs to the selected decision, not to taxonomy.
        report['fragment_audit_genome_provenance_complete'] = _json(decision).get('genome_provenance_complete',False)
    pd.DataFrame([{'rank':rank,'model_label':idx,'status':'unresolved'}
                  for rank,labels in report['unresolved_ranks'].items() for idx in labels],
                 columns=['rank','model_label','status']).to_csv(output/'rank_validation.csv',index=False)
    pd.DataFrame([{'model_label':row.get('model_label',row['id']-1),'id':row['id'],'species':row['species'],
                   'mapping_integrity_ok':report['mapping_integrity_ok']} for _,row in taxonomy.iterrows()]).to_csv(output/'class_validation.csv',index=False)
    if mode == 'curate':
        if not report['hierarchy_ready']:raise ValueError(f'Taxonomy cannot be published: {report}')
        taxonomy.to_parquet(output/'main_taxonomy_hybrid_CLEANSED.parquet',index=False)
    # Duplicate audit remains a diagnostic in 05 when explicit split input is supplied.
    if config.get('splits',{}).get('test'):
        values = config['splits']['test'];values = values if isinstance(values,list) else [values]
        files = _files([_path(p,Path(config['data_root'])) for p in values])
        limit = config.get('hash_max_bytes',16_777_216)
        report['duplicate_fragment_inventory'] = inventory_reference(files,hash_max_bytes=limit)
        inputs.extend(p for p in files if p.stat().st_size <= limit)
        cols = _schema(files)
        report['duplicate_audit'] = audit_partition_duplicates(files,seq_col=cols[0],label_col=cols[1],sample_size=100 if smoke else 50000)
        report['duplication_rate_pct'] = report['duplicate_audit']['duplication_rate_pct']
    else:report['duplication_rate_pct'] = None
    _atomic_json(output/'curation_audit.json',report)
    return report,inputs


def run_m01(
    notebook_id: str, project_root: str | Path, *, config_path: str | Path | None = None,
    settings: dict[str, Any] | None = None, run_plan: str | Path | None = None,
    is_smoke_test: bool = False,
) -> dict[str, Any]:
    """Execute one M01 analysis, publishing COMPLETE only after hashing products.

    Failures in a reserved namespace produce FAILED manifests and propagate the
    original exception. This does not certify real-data scientific reproduction.
    """
    root = Path(project_root).resolve()
    config = resolve_m01_inputs(notebook_id,root,config_path=config_path,settings=settings,run_plan=run_plan)
    if config.get('_plan') and bool(config['_plan']['is_smoke_test']) != is_smoke_test:
        raise ValueError('Mode conflicts with M00 plan')
    run_id,parent,output = _reserve(root,notebook_id,config)
    protocol = PROTOCOLS.get(notebook_id, 'm01_existing_taxonomy_audit_v1' if config['mode']=='audit_existing' else 'm01_ictv_curate_v1')
    clean_config = {k:v for k,v in config.items() if not k.startswith('_')}
    try:
        dirty = bool(subprocess.check_output(['git','status','--porcelain'],cwd=root,stderr=subprocess.DEVNULL).strip())
    except (OSError,subprocess.CalledProcessError):dirty = None
    manifest: dict[str,Any] = {'schema_version':1,'protocol_id':protocol,'notebook_id':notebook_id,
        'run_id':run_id,'parent_run_id':parent,'output_dir':str(output),'status':'FAILED',
        'configuration':clean_config,'seed':42,'is_smoke_test':is_smoke_test,
        'evidence':{'status':'synthetic' if is_smoke_test else 'observed'},
        'git_commit':get_git_commit(),'git_dirty':dirty,'library_versions':get_library_versions(),
        'inputs':[],'outputs':[],'dependencies':[]}
    for package in ('openpyxl','rapidfuzz','PyYAML'):
        try:manifest['library_versions'][package] = version(package)
        except PackageNotFoundError:manifest['library_versions'][package] = 'unavailable'
    try:
        input_paths: list[Path] = []
        if config_path:input_paths.append(Path(config_path))
        if run_plan:input_paths.append(Path(run_plan))
        if notebook_id == 'short/10':
            original = _table(config['original_mapping'])
            check = validate_taxonomy_identity(original,original,require_model_label=False)
            if not check['mapping_integrity_ok']:raise ValueError(check['mapping_errors'])
            files = _split_counts(config,output,original)
            # Explicit lists preserve split boundaries without copying partitions.
            train = _files(config['splits']['train']);test = _files(config['splits']['test'])
            report = audit_retraining_need(train,test,output,
                                          max_test_rows=config.get('max_test_rows'),work_dir=config.get('work_dir'))
            input_paths.append(Path(config['original_mapping']))
            manifest['fragment_inventory'] = inventory_reference(files,hash_max_bytes=config.get('hash_max_bytes',16_777_216))
            input_paths.extend(p for p in files if p.stat().st_size <= config.get('hash_max_bytes',16_777_216))
        elif notebook_id == 'short/11':
            decision,per,dep = _predecessor(config,is_smoke_test);manifest['dependencies'] = [dep]
            _reject_synthetic_taxonomy(_table(config['taxonomy_path']),is_smoke_test)
            report = audit_taxonomy_against_overlap(config['taxonomy_path'],per,decision,output,
                                                    original_mapping_path=config['original_mapping'],strict=True)
            input_paths.extend([Path(config['taxonomy_path']),Path(config['original_mapping']),decision,per,Path(dep['path'])])
            pd.DataFrame([{'rank':rank,'model_label':idx,'status':'unresolved'} for rank,ids in report['unresolved_ranks'].items() for idx in ids],
                         columns=['rank','model_label','status']).to_csv(output/'rank_validation.csv',index=False)
        else:
            report,paths = _curate(config,output,is_smoke_test);input_paths.extend(paths)
        manifest['metrics_summary'] = report
        manifest['inputs'] = [file_identity(p) for p in dict.fromkeys(input_paths)]
        manifest['outputs'] = [file_identity(p) for p in sorted(output.iterdir()) if p.is_file() and not p.name.startswith('.')]
        manifest['status'] = 'COMPLETE'
        _atomic_json(output/'manifest.json',manifest)
        return manifest
    except Exception as exc:
        manifest['status'] = 'FAILED';manifest['error'] = f'{type(exc).__name__}: {exc}'
        if not (output/'manifest.json').exists():_atomic_json(output/'manifest.json',manifest)
        raise


def run_m01_notebook(notebook_id: str, project_root: str | Path) -> dict[str, Any]:
    """Notebook entry point retaining AD-017 remote/local mode semantics."""
    # Explicit smoke only: missing remote data must never enable synthetic mode.
    smoke = os.getenv('SMOKE_TEST','0').strip().lower() in ('1','true','yes')
    ctx = setup_notebook_runtime(smoke_test=smoke, auto_smoke_fallback=False)
    root = os.getenv('M01_PROJECT_ROOT',str(project_root))
    plan = os.getenv('M01_RUN_PLAN')
    config = os.getenv('M01_CONFIG')
    settings = None if config or plan else {'data_root':str(ctx.data_root)}
    return run_m01(notebook_id,root,config_path=config,run_plan=plan,settings=settings,is_smoke_test=ctx.is_smoke_test)
