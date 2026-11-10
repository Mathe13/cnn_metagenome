"""M00: immutable administrative run plans; never executes scientific workloads.

Paths in manifests are explicit, while input roots in configuration are portable.
READY only describes the metadata inspected here, never scientific reproduction.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
import uuid
from xml.etree.ElementTree import ParseError
from zipfile import BadZipFile
from pathlib import Path
from typing import Any, Iterable

import yaml

from viral_cnn.notebook_runtime import (
    JupyterHubDataTopology, NOTEBOOK_REQUIREMENTS,
    reserve_run_directory,
)
from viral_cnn.utils.logging_utils import log_artifact_path

PROTOCOL = 'm00_run_plan_v1'
DEFAULT_HASH_MAX_BYTES = 16_777_216
# Required predecessor selections, independent of whether the producer is in this plan.
PREDECESSORS = {'short/08': ['short/07'], 'short/06b': ['short/06'],
                'short/09': ['c03'], 'long/02': ['long/01'],
                'long/03': ['long/02'], 'long/04': ['long/03']}


def _sha256(path: Path) -> str:
    log_artifact_path('identity', path)
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _path(value: str | Path, root: Path) -> Path:
    path = Path(value).expanduser()
    return Path(os.path.abspath(path if path.is_absolute() else root / path))


def inventory_reference(
    roots: Iterable[str | Path], *, exclude: Iterable[str | Path] = (),
    hash_max_bytes: int = DEFAULT_HASH_MAX_BYTES,
) -> list[dict[str, Any]]:
    """Inventory sources without following symlinks or copying file contents.

    Args:
        roots: Explicit roots to enumerate; absent roots remain visible.
        exclude: Subtrees excluded from enumeration, including this round.
        hash_max_bytes: Maximum regular file size whose full SHA-256 is read.

    Returns:
        Rows ordered and deduplicated by absolute lexical path.
    """
    if isinstance(hash_max_bytes, bool) or not isinstance(hash_max_bytes, int) or hash_max_bytes < 0:
        raise ValueError('hash_max_bytes must be a nonnegative integer')
    excluded = [Path(os.path.abspath(p)) for p in exclude]
    rows: dict[str, dict[str, Any]] = {}

    def visit(path: Path) -> None:
        if any(path == ex or ex in path.parents for ex in excluded):
            return
        key = str(path)
        if key in rows:
            return
        row = {'path': key, 'size_bytes': 'unknown', 'sha256': 'unknown',
               'hash_status': 'unknown', 'origin_status': 'unknown'}
        try:
            if path.is_symlink():
                row.update(origin_status='symlink', hash_status='symlink')
            elif not path.exists():
                row.update(origin_status='missing', hash_status='missing')
            elif path.is_dir():
                for child in sorted(path.iterdir()):
                    visit(child)
                return
            elif path.is_file():
                row.update(size_bytes=path.stat().st_size, origin_status='present')
                # Opening without reading distinguishes an unreadable large file
                # from one whose hash was deliberately omitted for size.
                with path.open('rb'):
                    pass
                if row['size_bytes'] > hash_max_bytes:
                    row['hash_status'] = 'skipped_size'
                else:
                    row.update(sha256=_sha256(path), hash_status='hashed')
            else:
                row.update(origin_status='unsupported', hash_status='unsupported')
        except OSError:
            row.update(origin_status='unreadable', hash_status='unreadable')
        rows[key] = row

    for root in roots:
        visit(Path(os.path.abspath(root)))
    return [rows[key] for key in sorted(rows)]


def _read_json(path: Path) -> dict[str, Any]:
    log_artifact_path('manifest', path)
    result = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(result, dict):
        raise ValueError(f'Expected JSON object: {path}')
    return result


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('x', encoding='utf-8') as stream:
        json.dump(payload, stream, indent=2, allow_nan=False)
        stream.write('\n')
    os.replace(temporary, path)


def _validate_config(config: dict[str, Any]) -> None:
    if config.get('protocol_id', PROTOCOL) != PROTOCOL:
        raise ValueError('Unsupported M00 configuration protocol')
    stages = config.get('stages')
    if not isinstance(stages, list) or not stages:
        raise ValueError('stages must be a nonempty list')
    ids = [s['notebook_id'] for s in stages]
    if len(set(ids)) != len(ids):
        raise ValueError('duplicate notebook_id')
    for stage in stages:
        key = stage['notebook_id']
        if key not in NOTEBOOK_REQUIREMENTS and key != 'short/06b':
            raise ValueError(f'Unknown notebook_id: {key}')
        action = stage.get('action', 'run')
        if action not in ('run', 'skip'):
            raise ValueError(f'Invalid action for {key}')
        if action == 'skip' and not str(stage.get('reason', '')).strip():
            raise ValueError(f'skip requires reason: {key}')
        manifest = stage.get('manifest_name', 'manifest.json')
        if not isinstance(manifest, str) or Path(manifest).name != manifest or manifest in ('.', '..', ''):
            raise ValueError('manifest_name must be a filename')
        deps = stage.get('dependencies', [])
        sources = [d['source'] for d in deps]
        if len(set(sources)) != len(sources):
            raise ValueError(f'duplicate dependency for {key}')
        if action == 'run' and set(PREDECESSORS.get(key, [])) - set(sources):
            raise ValueError(f'missing predecessor selection for {key}')
        for dep in deps:
            if dep.get('kind') == 'planned':
                if dep['source'] not in ids:
                    raise ValueError(f'unknown dependency: {dep["source"]}')
                producer = next(s for s in stages if s['notebook_id'] == dep['source'])
                if producer.get('action', 'run') == 'skip':
                    raise ValueError('planned dependency cannot use skipped producer')
            elif dep.get('kind') == 'external':
                if not dep.get('manifest'):
                    raise ValueError('external dependency requires explicit manifest')
            else:
                raise ValueError('dependency kind must be planned or external')
        resume = stage.get('resume_manifest')
        if resume and key not in ('short/06', 'short/06b'):
            raise ValueError('resume_manifest supported only for 06/06b')
    graph = {s['notebook_id']: [d['source'] for d in s.get('dependencies', [])
                               if d['kind'] == 'planned'] for s in stages}
    visited: set[str] = set()
    visiting: set[str] = set()

    def visit(key: str) -> None:
        if key in visiting:
            raise ValueError('dependency cycle')
        if key in visited:
            return
        visiting.add(key)
        for source in graph[key]:
            visit(source)
        visiting.remove(key)
        visited.add(key)
    for key in graph:
        visit(key)


def _input_requirements(stage: dict[str, Any], topology: JupyterHubDataTopology) -> list[dict[str, Any]]:
    # Fixed-directory predecessor requirements are replaced by explicit dependency manifests.
    specs = stage.get('requirements', [r for r in NOTEBOOK_REQUIREMENTS.get(stage['notebook_id'], [])
                                       if not r.get('external')])
    result = []
    for spec in specs:
        spec = dict(spec)
        if 'topology' in spec:
            section, name = spec.pop('topology').split('/')
            path = Path(topology.paths[section][name])
        elif spec.get('path', '').startswith(('short/', 'long/', 'checkpoints/')):
            section, name = spec['path'].split('/')
            path = Path(topology.paths[section][name])
        else:
            path = _path(spec['path'], topology.root)
        spec['path'] = str(path)
        if spec.get('kind') not in ('file', 'directory', 'parquet', 'csv', 'xlsx'):
            raise ValueError(f'unsupported requirement kind: {spec.get("kind")}')
        result.append(spec)
    return result


def _inspect_requirement(spec: dict[str, Any]) -> dict[str, Any]:
    path = Path(spec['path'])
    log_artifact_path('preflight input', path)
    detail = 'readable'
    try:
        if not path.exists():
            raise ValueError('missing')
        kind = spec['kind']
        if kind == 'directory':
            if not path.is_dir() or not os.access(path, os.R_OK | os.X_OK):
                raise ValueError('unreadable directory')
        elif kind == 'parquet':
            import pyarrow.parquet as pq
            if path.is_file():
                files = [path]
            else:
                split = spec.get('split')
                base = path / split if split else path
                files = sorted(base.rglob('*.parquet'))
                if split and not files:
                    direct = path / f'{split}.parquet'
                    files = [direct] if direct.is_file() else []
            if not files:
                raise ValueError('no parquet files for requested split')
            for file in files:
                log_artifact_path('Parquet metadata', file)
                schema = pq.read_schema(file)
                absent = set(spec.get('columns', [])) - set(schema.names)
                if absent:
                    raise ValueError(f'missing columns in {file}: {sorted(absent)}')
            detail = f'metadata OK; {len(files)} partitions; no rows read'
        elif kind == 'csv':
            with path.open(newline='', encoding='utf-8-sig') as stream:
                columns = next(csv.reader(stream), [])
            absent = set(spec.get('columns', [])) - set(columns)
            if absent:
                raise ValueError(f'missing columns: {sorted(absent)}')
        elif kind == 'xlsx':
            # Workbook metadata only, no worksheet rows loaded.
            from zipfile import ZipFile
            from xml.etree import ElementTree
            with ZipFile(path) as book:
                document = ElementTree.fromstring(book.read('xl/workbook.xml'))
            sheets = {node.attrib['name'] for node in document.iter() if node.tag.endswith('}sheet')}
            absent = set(spec.get('sheets', [])) - sheets
            if absent:
                raise ValueError(f'missing sheets: {sorted(absent)}')
        else:
            if not path.is_file():
                raise ValueError('not a regular file')
            with path.open('rb'):
                pass
        return {**spec, 'status': 'READY', 'detail': detail}
    except (OSError, ValueError, ImportError, KeyError, ParseError, BadZipFile) as exc:
        return {**spec, 'status': 'BLOCKED', 'detail': str(exc)}


def inspect_run_stage(plan: dict[str, Any], notebook_id: str) -> dict[str, Any]:
    """Inspect one persisted stage without reading sequence rows or model weights.

    Args:
        plan: M00 plan loaded from its explicit path.
        notebook_id: Stage selected for inspection.

    Returns:
        READY/BLOCKED/SKIPPED report with every requirement and dependency.
    """
    if plan.get('protocol_id') != PROTOCOL or plan.get('schema_version') != 1:
        raise ValueError('Unsupported M00 plan protocol/schema')
    matches = [s for s in plan['stages'] if s['notebook_id'] == notebook_id]
    if len(matches) != 1:
        raise ValueError(f'Unknown or duplicate plan stage: {notebook_id}')
    stage = matches[0]
    report = {'notebook_id': notebook_id, 'run_id': stage['run_id'],
              'parent_run_id': stage['parent_run_id'], 'is_smoke_test': plan['is_smoke_test'],
              'scientific_reproduction_certified': False, 'requirements': [], 'dependencies': []}
    if stage['action'] == 'skip':
        return {**report, 'status': 'SKIPPED', 'reason': stage['reason']}
    report['requirements'] = [_inspect_requirement(s) for s in stage['requirements']]
    for dep in stage['dependencies']:
        path = Path(dep['manifest'])
        result = {**dep, 'status': 'BLOCKED'}
        try:
            if not path.is_file():
                raise ValueError('planned manifest pending' if dep['kind'] == 'planned' else 'missing external manifest')
            actual_hash = _sha256(path)
            if dep['kind'] == 'external' and dep['sha256'] == 'unknown':
                raise ValueError('external identity unpinned; create a new plan after selecting available input')
            if dep['sha256'] != 'unknown' and dep['sha256'] != actual_hash:
                raise ValueError('manifest hash mismatch')
            manifest = _read_json(path)
            evidence = manifest.get('evidence')
            evidence_status = evidence.get('status') if isinstance(evidence, dict) else None
            if manifest.get('status', evidence_status) != 'COMPLETE':
                raise ValueError('manifest incomplete: explicit COMPLETE required')
            if not plan['is_smoke_test'] and manifest.get('is_smoke_test') is not False:
                raise ValueError('synthetic or unknown manifest mode')
            if notebook_id == 'short/09' and dep.get('source') == 'c03' and not plan['is_smoke_test']:
                from viral_cnn.baselines.overlap import inspect_c03_package
                specialized = inspect_c03_package(path.parent)
                result['c03_inspection'] = specialized
                if specialized['status'] != 'READY':
                    raise ValueError(specialized.get('error', 'C03 package blocked'))
            result.update(status='READY', detail='manifest verified; see specialized inspection when present',
                          observed_sha256=actual_hash)
        except (OSError, ValueError, TypeError) as exc:
            result['detail'] = str(exc)
        report['dependencies'].append(result)
    resume = stage.get('resume')
    if resume:
        report['dependencies'].append({**resume, 'status': 'BLOCKED',
                                       'detail': 'PENDING_C09: specialized cache validation required'})
    checks = report['requirements'] + report['dependencies']
    report['status'] = 'READY' if all(s['status'] == 'READY' for s in checks) else 'BLOCKED'
    return report


def build_run_plan(
    config_path: str | Path, project_root: str | Path, *, data_root: str | Path | None = None,
    run_id: str | None = None, smoke_test: bool = False,
) -> dict[str, Any]:
    """Reserve a new round and atomically publish its plan, inventory and preflight.

    Args:
        config_path: Explicit YAML configuration, never inferred from prior runs.
        project_root: Repository/output root.
        data_root: Override of DATA_ROOT and configured data_root.
        run_id: Round ID; generated from UUID when omitted.
        smoke_test: Marks administrative outputs synthetic; never creates fixtures.

    Returns:
        The persisted run plan. BLOCKED plans are inspectable but cannot run.
    """
    root = Path(project_root).resolve()
    config_file = Path(config_path).resolve()
    log_artifact_path('M00 configuration', config_file)
    config = yaml.safe_load(config_file.read_text(encoding='utf-8'))
    if not isinstance(config, dict):
        raise ValueError('configuration must be a mapping')
    _validate_config(config)
    selected_root = data_root or os.getenv('DATA_ROOT') or config.get('data_root', '.')
    data = _path(selected_root, root).resolve()
    topology = JupyterHubDataTopology(data)
    # Validate shapes and resolve inputs before reserving any output.
    requirements = {s['notebook_id']: _input_requirements(s, topology) for s in config['stages']}
    hash_limit = config.get('hash_max_bytes', DEFAULT_HASH_MAX_BYTES)
    if isinstance(hash_limit, bool) or not isinstance(hash_limit, int) or hash_limit < 0:
        raise ValueError('hash_max_bytes must be a nonnegative integer')
    parent = run_id if run_id is not None else f'm00-{uuid.uuid4().hex[:16]}'
    output = reserve_run_directory(root, parent, 'm00')
    try:
        stages = []
        for spec in config['stages']:
            key = spec['notebook_id']
            child = f'{parent}-{key.replace("/", "-")}'
            target = root / 'results_reproduction' / 'runs' / child / key
            if spec.get('action', 'run') != 'skip':
                target = reserve_run_directory(root, child, key)
            resources = {'device': 'unknown', 'memory_gb': 'unknown', 'storage_gb': 'unknown'}
            resources.update(config.get('resources', {}))
            resources.update(spec.get('resources', {}))
            stage = {'notebook_id': key, 'run_id': child, 'parent_run_id': parent,
                     'output_dir': str(target), 'manifest': str(target / spec.get('manifest_name', 'manifest.json')),
                     'action': spec.get('action', 'run'), 'reason': spec.get('reason', ''),
                     'resources': resources, 'requirements': requirements[key], 'dependencies': []}
            if spec.get('m01') is not None:
                stage['m01'] = spec['m01']
            if spec.get('resume_manifest'):
                source = _path(spec['resume_manifest'], data)
                stage['resume'] = {'manifest': str(source), 'sha256': _sha256(source) if source.is_file() else 'unknown',
                                   'validation': 'PENDING_C09', 'kind': 'external'}
            stages.append(stage)
        by_id = {s['notebook_id']: s for s in stages}
        for spec, stage in zip(config['stages'], stages):
            for dep in spec.get('dependencies', []):
                path = (Path(by_id[dep['source']]['manifest']) if dep['kind'] == 'planned'
                        else _path(dep['manifest'], data))
                stage['dependencies'].append({**dep, 'manifest': str(path),
                                               'sha256': _sha256(path) if path.is_file() else 'unknown'})
        try:
            dirty = bool(subprocess.check_output(['git', 'status', '--porcelain'], cwd=root,
                                                stderr=subprocess.DEVNULL).strip())
            commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root,
                                             stderr=subprocess.DEVNULL).decode().strip()
        except (OSError, subprocess.CalledProcessError):
            dirty, commit = None, 'unknown'
        plan = {'protocol_id': PROTOCOL, 'schema_version': 1, 'parent_run_id': parent,
                'plan_path': str(output / 'run_plan.json'), 'data_root': str(data),
                'is_smoke_test': smoke_test, 'seed': config.get('seed', 42),
                'git_commit': commit, 'git_dirty': dirty,
                'population_id': config.get('population_id') or 'unknown',
                'split_id': config.get('split_id') or 'unknown',
                'configuration': {'path': str(config_file), 'sha256': _sha256(config_file)},
                'stages': stages, 'scientific_reproduction_certified': False}
        roots = [_path(p, root) for p in config.get('reference_roots', ['paper', 'results_reproduction'])]
        roots += [_path(p, data) for p in config.get('external_reference_roots', [])]
        excluded = [output.parent] + [Path(s['output_dir']).parents[len(Path(s['notebook_id']).parts)-1] for s in stages]
        rows = inventory_reference(roots, exclude=excluded, hash_max_bytes=hash_limit)
        inventory = output / 'reference_inventory.csv'
        temporary = output / 'reference_inventory.csv.tmp'
        with temporary.open('x', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=['path', 'size_bytes', 'sha256', 'hash_status', 'origin_status'])
            writer.writeheader(); writer.writerows(rows)
        os.replace(temporary, inventory)
        plan['reference_inventory'] = {'path': str(inventory), 'sha256': _sha256(inventory),
                                       'hash_max_bytes': hash_limit}
        reports = [inspect_run_stage(plan, s['notebook_id']) for s in stages]
        plan['status'] = 'READY' if all(r['status'] in ('READY', 'SKIPPED') for r in reports) else 'BLOCKED'
        plan['preflight_summary'] = {r['notebook_id']: r['status'] for r in reports}
        _atomic_json(output / 'preflight_report.json', {'status': plan['status'], 'stages': reports})
        # Publish plan last: its presence means every administrative product was written.
        _atomic_json(output / 'run_plan.json', plan)
        return plan
    except Exception as exc:
        # An exclusive failure marker does not depend on the failed atomic-write primitive.
        with (output / 'failure.json').open('x', encoding='utf-8') as stream:
            json.dump({'status': 'FAILED', 'error': str(exc), 'parent_run_id': parent}, stream)
        raise


def load_run_plan(path: str | Path) -> dict[str, Any]:
    """Load an explicitly selected administrative plan with path logging."""
    return _read_json(Path(path).expanduser().resolve())
