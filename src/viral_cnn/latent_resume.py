"""Read-only reuse of validated notebook 06 artifacts across exclusive run directories.

Orchestration lives outside eval.latent: its source hash is part of historical
embedding identities and must not change merely to add restart support.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import ast
import hashlib
import json
import os
from pathlib import Path
import pickle
import re
import tempfile
from typing import Any, Callable

import numpy as np
import pandas as pd
import torch

from viral_cnn.eval.latent import (
    EmbeddingArtifact, LatentMemoryBudget, LinearProbeClassifier,
    _sha256_file, load_embedding_artifact,
)
from viral_cnn.utils.logging_utils import log_artifact_path

PROTOCOL = 'c09_latent_resume_v1'


def notebook_code_sha256(path: Path | str) -> str:
    """Hash executed code structure excluding the two non-scientific reuse knobs.

    Editing a run selector must not invalidate an otherwise identical analysis.
    All other assignments and code remain in this identity.
    """
    notebook = json.loads(Path(path).read_text())
    code = '\n'.join(''.join(c['source']) for c in notebook['cells'] if c['cell_type'] == 'code')
    tree = ast.parse(code)
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in
                {'LATENT_EXECUTION_MODE', 'LATENT_REUSE_RUN_ID'} for t in node.targets):
            node.value = ast.Constant(value='c09-runtime-selector')
    return hashlib.sha256(ast.dump(tree, include_attributes=False).encode()).hexdigest()


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def _atomic(path: Path, write: Callable[[Path], None]) -> None:
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix='.tmp', delete=False) as stream:
        temp = Path(stream.name)
    try:
        write(temp)
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def _pack(value: Any) -> Any:
    """Use tensors/primitives only so torch.load can enforce weights_only=True."""
    if isinstance(value, pd.DataFrame):
        return {'__c09__': 'frame', 'columns': list(value.columns),
                'data': _pack(value.to_numpy().tolist()), 'dtypes': [str(t) for t in value.dtypes]}
    if isinstance(value, np.ndarray):
        return {'__c09__': 'array', 'tensor': torch.from_numpy(value.copy())}
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _pack(v) for key, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_pack(v) for v in value]
    if value is None or type(value) in (str, int, float, bool):
        return value
    raise TypeError(f'Unsupported restart payload: {type(value)}')


def _unpack(value: Any) -> Any:
    if isinstance(value, dict):
        if value.get('__c09__') == 'array':
            return value['tensor'].numpy()
        if value.get('__c09__') == 'frame':
            result = pd.DataFrame(_unpack(value['data']), columns=value['columns'])
            return result.astype(dict(zip(value['columns'], value['dtypes'])))
        return {key: _unpack(v) for key, v in value.items()}
    if isinstance(value, list):
        return [_unpack(v) for v in value]
    return value


@dataclass
class StageResult:
    """A small stage result and the external files required to restore it."""

    value: Any
    files: list[Path | str] = field(default_factory=list)


class LatentResumeCache:
    """Select compatible historical artifacts without ever modifying their runs.

    Invalid candidates are discarded from selection, not physically deleted.
    Operational I/O failures propagate; absence and content-validation failures
    cause fallback. Each caller must reserve its own output directory first.
    """

    def __init__(self, output_dir: Path | str, mode: str = 'reuse',
                 reuse_run_id: str | None = None) -> None:
        if mode not in ('reuse', 'rerun'):
            raise ValueError('LATENT_EXECUTION_MODE must be reuse or rerun')
        self.output_dir = Path(output_dir).resolve()
        self.mode = mode
        self.events: list[dict[str, Any]] = []
        runs = self.output_dir.parent.parent
        self.sources: list[Path] = []
        if reuse_run_id:
            if (not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', reuse_run_id)
                    or reuse_run_id in ('.', '..')):
                raise ValueError(f'Invalid LATENT_REUSE_RUN_ID: {reuse_run_id!r}')
            source = runs / reuse_run_id / self.output_dir.name
            if not source.is_dir() or source.resolve() == self.output_dir:
                raise FileNotFoundError(f'LATENT_REUSE_RUN_ID not available: {reuse_run_id}')
            if mode == 'reuse':
                self.sources = [source]
        elif mode == 'reuse':
            self.sources = sorted(
                (p for p in runs.glob(f'*/{self.output_dir.name}')
                 if p.is_dir() and p.resolve() != self.output_dir),
                key=lambda p: (p.stat().st_mtime_ns, str(p)), reverse=True)
        print(f'[LATENT_RESUME] mode={mode}; sources={len(self.sources)}; output={self.output_dir}')
        self._report()

    def _report(self) -> None:
        report = {'protocol': PROTOCOL, 'mode': self.mode,
                  'sources': [str(p) for p in self.sources], 'events': self.events}
        _atomic(self.output_dir / 'resume_report.json',
                lambda p: p.write_text(json.dumps(report, indent=2) + '\n'))

    def record(self, component: str, status: str, path: Path | str | None = None,
               reason: str | None = None, sha256: str | None = None) -> None:
        """Persist the decision immediately, including rejections before a failure."""
        event = {'component': component, 'status': status}
        if path is not None:
            event['path'] = str(Path(path).resolve())
        if reason:
            event['reason'] = reason
        if sha256:
            event['sha256'] = sha256
        self.events.append(event)
        print(f'[LATENT_RESUME] {component}: {status}' + (f' — {reason}' if reason else ''))
        self._report()

    def load_embedding(self, expected: dict[str, Any], dimension: int, *,
                       memory_manager: LatentMemoryBudget | None = None,
                       batch_size: int = 4096) -> tuple[EmbeddingArtifact, Path] | None:
        """Load a fully compatible historical split, falling back on bad content."""
        component = f"embedding:{expected['variant']}:{expected['split']}"
        manager = memory_manager or LatentMemoryBudget.from_environment()
        if self.mode == 'rerun':
            return None
        for source in self.sources:
            stem = f"features_{expected['variant']}_{expected['split']}_"
            candidates = sorted({p.with_suffix('.parquet') for p in source.glob(stem + '*')
                                 if p.suffix in ('.json', '.parquet')}, reverse=True)
            report_path = source / 'resume_report.json'
            if report_path.is_file():
                try:
                    log_artifact_path('latent resume references', report_path)
                    report = json.loads(report_path.read_text())
                    candidates += [Path(e['path']) for e in report['events']
                                   if e['component'] == component and e['status'] == 'reused']
                except (ValueError, KeyError, TypeError) as exc:
                    self.record(component, 'discarded', report_path, str(exc))
            candidates = list(dict.fromkeys(candidates))
            for path in candidates:
                # Cheap manifest comparison avoids hashing/reading incompatible huge files.
                loaded = None
                prior_allocations = len(manager._arrays)
                try:
                    log_artifact_path('embedding resume manifest', path.with_suffix('.json'))
                    manifest = json.loads(path.with_suffix('.json').read_text())
                    if not isinstance(manifest, dict):
                        raise ValueError('Embedding manifest must be an object')
                    if any(manifest.get(k) != v for k, v in expected.items()):
                        raise ValueError('Embedding identity/configuration mismatch')
                    loaded = load_embedding_artifact(path, expected, memory_manager=manager, batch_size=batch_size)
                    count = expected.get('extracted_count', expected.get('population_available'))
                    if not len(loaded.features) or (count is not None and len(loaded.features) != count):
                        raise ValueError('Embedding row count does not match population/extracted_count')
                    if loaded.features.shape[1] != dimension:
                        raise ValueError('Embedding dimension does not match classifier input')
                    for start in range(0, len(loaded.labels), batch_size):
                        if not np.isin(loaded.labels[start:start + batch_size], expected['class_order']).all():
                            raise ValueError('Embedding labels outside declared class order')
                except (FileNotFoundError, ValueError, KeyError, TypeError) as exc:
                    manager.release(*manager._arrays[prior_allocations:])
                    self.record(component, 'discarded', path, str(exc))
                    continue
                self.record(component, 'reused', path, sha256=loaded.manifest['artifact_sha256'])
                return loaded, path
        return None

    def run_stage(self, name: str, identity: dict[str, Any], compute: Callable[[], Any],
                  *, validate: Callable[[Any], None] | None = None) -> Any:
        """Restore or atomically checkpoint one complete dependency-identified unit.

        ``validate`` is run on restored content before accepting it. Computation
        errors propagate and never publish a completion record.
        """
        if not re.fullmatch(r'[A-Za-z0-9_-]+', name):
            raise ValueError(f'Invalid stage name: {name}')
        expected = dict(identity, resume_protocol=PROTOCOL, resume_code_sha256=_sha256_file(__file__))
        fingerprint = _digest(expected)
        stem = f'stage_{name}_{fingerprint}'
        for source in self.sources:
            records = sorted(source.glob(f'stage_{name}_*.json'))
            report_path = source / 'resume_report.json'
            if report_path.is_file():
                try:
                    log_artifact_path('latent stage references', report_path)
                    report = json.loads(report_path.read_text())
                    records += [Path(e['path']).with_suffix('.json') for e in report['events']
                                if e['component'] == name and e['status'] == 'reused']
                    records = list(dict.fromkeys(records))
                except (ValueError, KeyError, TypeError) as exc:
                    self.record(name, 'discarded', report_path, str(exc))
            if not records and any(source.iterdir()):
                self.record(name, 'discarded', source, 'Legacy outputs have no verified stage completion record')
            for record in records:
                try:
                    log_artifact_path('latent stage manifest', record)
                    meta = json.loads(record.read_text())
                    if not isinstance(meta, dict):
                        raise ValueError('Stage completion record must be an object')
                    if (meta.get('protocol') != PROTOCOL or meta.get('record_sha256') !=
                            _digest({k: v for k, v in meta.items() if k != 'record_sha256'})):
                        raise ValueError('Stage completion record fingerprint mismatch')
                    if meta['identity'] != expected or meta['fingerprint'] != fingerprint:
                        raise ValueError('Stage dependency/configuration mismatch')
                    payload = Path(meta.get('payload_path', record.with_suffix('.pt')))
                    files = [dict(path=str(payload), sha256=meta['payload_sha256'])] + meta['files']
                    for item in files:
                        log_artifact_path('latent stage artifact', item['path'])
                        if _sha256_file(item['path']) != item['sha256']:
                            raise ValueError(f"Stage checksum mismatch: {item['path']}")
                    value = _unpack(torch.load(payload, map_location='cpu', weights_only=True))
                    if validate:
                        validate(value)
                except (FileNotFoundError, ValueError, KeyError, TypeError, EOFError, RuntimeError, pickle.UnpicklingError) as exc:
                    self.record(name, 'discarded', record, str(exc))
                    continue
                # Link every dependency, including transitive external files, into this run.
                self.record(name, 'reused', payload, sha256=meta['payload_sha256'])
                for item in meta['files']:
                    self.record(name + ':dependency', 'reused', item['path'], sha256=item['sha256'])
                return value
        result = compute()
        bundle = result if isinstance(result, StageResult) else StageResult(result)
        if validate:
            validate(bundle.value)
        payload = self.output_dir / (stem + '.pt')
        record = self.output_dir / (stem + '.json')
        if payload.exists() or record.exists():
            raise FileExistsError(f'Stage output already exists: {payload}; start a fresh notebook run')
        files = []
        for source_path in bundle.files:
            source_path = Path(source_path).resolve()
            files.append({'path': str(source_path), 'sha256': _sha256_file(source_path)})
        _atomic(payload, lambda p: torch.save(_pack(bundle.value), p))
        meta = {'protocol': PROTOCOL, 'identity': expected, 'fingerprint': fingerprint,
                'payload_path': str(payload),
                'payload_sha256': _sha256_file(payload), 'files': files}
        meta['record_sha256'] = _digest(meta)
        _atomic(record, lambda p: p.write_text(json.dumps(meta, indent=2) + '\n'))
        self.record(name, 'recalculated', payload, sha256=meta['payload_sha256'])
        return bundle.value


def snapshot_probe(probe: LinearProbeClassifier, selection: dict[str, Any]) -> dict[str, Any]:
    """Capture selected weights before test evaluation, without serializing a module."""
    if probe._audit_state != 'SEALED' or probe._state_hash(probe) != selection['weights_sha256']:
        raise ValueError('Probe must be sealed with the selected weights')
    return {'weights': {k: v.detach().cpu().clone() for k, v in probe.state_dict().items()},
            'selection': selection, 'in_features': probe.fc.in_features, 'num_classes': probe.fc.out_features}


def restore_probe(payload: dict[str, Any], train: EmbeddingArtifact,
                  validation: EmbeddingArtifact) -> tuple[LinearProbeClassifier, dict[str, Any]]:
    """Rebuild the sealed audit state only for the same verified train/validation."""
    train.validate()
    validation.validate()
    selection = payload['selection']
    if train.manifest['split'] != 'train' or validation.manifest['split'] != 'validation':
        raise ValueError('Probe restoration requires train and validation')
    if selection.get('state') != 'SEALED' or selection.get('probe_protocol') != 'c01_train_val_frozen_v1':
        raise ValueError('Missing sealed probe selection')
    if selection['train_manifest'] != train.manifest or selection['validation_manifest'] != validation.manifest:
        raise ValueError('Probe embedding dependencies changed')
    tr = set(train.manifest['partition_sha256'])
    va = set(validation.manifest['partition_sha256'])
    if not tr or not va or tr.intersection(va):
        raise ValueError('Probe train/validation identities overlap or are missing')
    # Initializing a restored module must not perturb RNG for subsequent fresh probes.
    with torch.random.fork_rng(devices=[]):
        probe = LinearProbeClassifier(payload['in_features'], payload['num_classes'])
    probe.load_state_dict(payload['weights'])
    if probe._state_hash(probe) != selection['weights_sha256']:
        raise ValueError('Restored probe weights do not match sealed SHA-256')
    if probe.fc.in_features != train.features.shape[1] or probe.fc.in_features != validation.features.shape[1]:
        raise ValueError('Restored probe dimension mismatch')
    probe._training_partition_hashes = tr
    probe._validation_partition_hashes = va
    probe._selection_history = list(selection['selection_history'])
    probe._sealed_sha256 = selection['weights_sha256']
    probe._audit_state = 'SEALED'
    probe.eval()
    return probe, selection


def infer_native_predictions(model: torch.nn.Module, loader: Any, test: EmbeddingArtifact,
                             *, device: str | torch.device = 'cpu') -> np.ndarray:
    """Run the native backbone/head and match every prediction to cached test rows.

    Follows extract_latent_embeddings inference precision (no autocast). This is
    necessary for old test artifacts which did not store native predictions.
    """
    model.eval()
    predictions = np.empty(len(test.labels), dtype=np.int64)
    offset = 0
    with torch.inference_mode():
        for inputs, labels, ids in loader:
            size = len(labels)
            stop = offset + size
            values = labels.detach().cpu().numpy() if isinstance(labels, torch.Tensor) else np.asarray(labels)
            if (stop > len(test.labels) or not np.array_equal(np.asarray(ids, dtype=str), test.row_ids[offset:stop])
                    or not np.array_equal(values, test.labels[offset:stop])):
                raise ValueError('Native inference rows do not match cached test identities/labels')
            predictions[offset:stop] = model(inputs.to(device)).argmax(dim=1).detach().cpu().numpy()
            offset = stop
    if offset != len(test.labels):
        raise ValueError('Native inference row count does not match cached test')
    return predictions
