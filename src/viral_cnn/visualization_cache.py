"""Read archived test embeddings for visualization without loading full features.

Inputs are immutable Parquet/JSON artifacts, not raw sequence datasets or model
checkpoints. A full SHA-256 check precedes selective column/row-group decoding.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import time
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from viral_cnn.eval.latent import EmbeddingArtifact, _sha256_file, save_embedding_artifact
from viral_cnn.paired_umap import MODEL_NAMES
from viral_cnn.utils.logging_utils import log_artifact_path


def _signature(path: Path) -> tuple[int, int, int, int]:
    stat = path.stat()
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns


class CompactRowIDs:
    """Keep UTF-8 Arrow buffers; materialize only indexed/sliced IDs as NumPy."""

    def __init__(self, chunks: list[pa.Array]) -> None:
        self.values = pa.chunked_array(chunks, type=pa.string())

    def __len__(self) -> int:
        return len(self.values)

    def __getitem__(self, index: Any) -> Any:
        if isinstance(index, slice):
            start, stop, step = index.indices(len(self))
            if step != 1:
                return self[np.arange(start, stop, step)]
            return np.asarray(self.values.slice(start, max(0, stop - start)).to_pylist(), dtype=str)
        if np.isscalar(index):
            return self.values[int(index)].as_py()
        return np.asarray(self.values.take(pa.array(np.asarray(index, dtype=np.int64))).to_pylist(), dtype=str)


class SelectedParquetFeatures:
    """Decode only row groups containing requested rows, using bounded batches."""

    def __init__(self, path: Path, parquet: pq.ParquetFile, dimension: int,
                 signature: tuple[int, int, int, int], batch_size: int) -> None:
        self.path, self.parquet, self.signature = path, parquet, signature
        self.shape = (parquet.metadata.num_rows, dimension)
        self.batch_size = batch_size
        self.ends = np.cumsum([parquet.metadata.row_group(i).num_rows for i in range(parquet.num_row_groups)])
        self.stats: dict[str, Any] = {'row_groups_total': parquet.num_row_groups,
                                      'row_groups_decoded': [], 'selected_rows': 0}

    def _unchanged(self) -> None:
        if _signature(self.path) != self.signature:
            raise ValueError(f'Embedding changed after checksum validation: {self.path}')

    def __array__(self, *args: Any, **kwargs: Any) -> np.ndarray:
        raise ValueError('Full feature materialization is disabled in notebook 06b')

    def __getitem__(self, positions: np.ndarray) -> np.ndarray:
        positions = np.asarray(positions)
        if positions.ndim != 1 or positions.dtype.kind not in 'iu' or np.any(positions < 0) or np.any(positions >= self.shape[0]):
            raise ValueError('Feature selection requires valid one-dimensional integer row indices')
        self._unchanged()
        output = np.empty((len(positions), self.shape[1]), dtype=np.float32)
        order = np.argsort(positions, kind='stable')
        sorted_positions = positions[order]
        groups = np.searchsorted(self.ends, sorted_positions, side='right')
        wanted_groups = np.unique(groups)
        log_artifact_path('selected test embedding features', self.path,
                          extra=f'{len(positions)} selected rows; {len(wanted_groups)}/{len(self.ends)} row groups')
        written = 0
        for group in wanted_groups:
            begin = int(self.ends[group - 1]) if group else 0
            selected = np.flatnonzero(groups == group)
            local = sorted_positions[selected] - begin
            offset = 0
            self.stats['row_groups_decoded'].append(int(group))
            for batch in self.parquet.iter_batches(batch_size=self.batch_size, row_groups=[int(group)], columns=['features']):
                stop = offset + batch.num_rows
                left, right = np.searchsorted(local, [offset, stop])
                if right > left:
                    values = batch.column(0).flatten().to_numpy(zero_copy_only=False).reshape(batch.num_rows, self.shape[1])
                    target = order[selected[left:right]]
                    output[target] = values[local[left:right] - offset]
                    written += right - left
                offset = stop
                if offset > local[-1]:
                    break
        self._unchanged()
        if written != len(positions):
            raise ValueError(f'Incomplete selected feature read: {written}/{len(positions)}')
        if not np.isfinite(output).all():
            raise ValueError(f'Non-finite selected embedding features: {self.path}')
        self.stats['selected_rows'] += len(positions)
        return output


@dataclass
class VisualizationEmbedding:
    """Verified source identity, compact row metadata and lazy selected features."""

    path: Path
    manifest: dict[str, Any]
    labels: np.ndarray
    row_ids: CompactRowIDs
    features: SelectedParquetFeatures
    inputs: list[dict[str, Any]]


def open_test_embeddings(source_dir: Path | str, *, smoke: bool = False,
                         batch_size: int = 4096) -> dict[str, VisualizationEmbedding]:
    """Verify five archive artifacts, decoding only label/row_id columns initially.

    Historical extraction/checkpoint identities are recorded, not re-executed or
    checked against unavailable raw data/weights. No source artifact is modified.
    """
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError('Read batch_size must be a positive integer')
    root = Path(source_dir).resolve()
    embeddings = {}
    class_order = None
    for name in MODEL_NAMES:
        candidates = sorted(root.glob(f'features_{name}_test_*.json'))
        if len(candidates) != 1:
            raise ValueError(f'Expected exactly one test embedding manifest for {name} in {root}; found {len(candidates)}')
        meta_path = candidates[0]
        path = meta_path.with_suffix('.parquet')
        meta_signature = _signature(meta_path)
        log_artifact_path('archived test embedding manifest', meta_path)
        meta = json.loads(meta_path.read_text())
        required = {'verified', 'variant', 'split', 'checkpoint', 'checkpoint_sha256', 'class_order',
                    'partitions', 'partition_sha256', 'encoding', 'extractor_code_sha256', 'sampling',
                    'mode', 'extracted_count', 'population_available', 'artifact_sha256', 'fingerprint'}
        if not isinstance(meta, dict) or required - meta.keys():
            raise ValueError(f'Incomplete test embedding provenance: {meta_path}')
        payload = {k: v for k, v in meta.items() if k not in {'schema_version', 'artifact_sha256', 'fingerprint'}}
        fingerprint = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
        if meta['fingerprint'] != fingerprint or not path.stem.endswith('_' + fingerprint):
            raise ValueError(f'Embedding manifest fingerprint mismatch: {meta_path}')
        if meta['verified'] is not True or meta['variant'] != name or meta['split'] != 'test':
            raise ValueError(f'Embedding variant/split is invalid: {meta_path}')
        if meta['mode'] != ('smoke' if smoke else 'empirical'):
            raise ValueError(f'Smoke/empirical embedding mode mismatch: {meta_path}')
        if meta['encoding'].get('channels') != ['A', 'C', 'G', 'T'] or meta['encoding'].get('length') != 512:
            raise ValueError(f'Incompatible nucleotide encoding: {meta_path}')
        declared = meta['class_order']
        if not declared or any(type(x) is not int or x < 0 for x in declared) or len(set(declared)) != len(declared):
            raise ValueError(f'Invalid class order: {meta_path}')
        if class_order is not None and declared != class_order:
            raise ValueError(f'Class order differs between model variants: {meta_path}')
        class_order = declared
        signature = _signature(path)
        log_artifact_path('test embedding SHA-256 verification', path, extra=f'{signature[2] / 2**30:.2f} GiB; full byte scan')
        started = time.perf_counter()
        digest = _sha256_file(path)
        if digest != meta['artifact_sha256'] or _signature(path) != signature:
            raise ValueError(f'Corrupt or changing embedding artifact: {path}')
        print(f'[VIS_CACHE] {name}: checksum OK in {time.perf_counter() - started:.1f}s; reading labels/IDs only')
        parquet = pq.ParquetFile(path)
        field = parquet.schema_arrow.field('features').type
        if not pa.types.is_fixed_size_list(field) or field.value_type != pa.float32() or field.list_size < 2:
            raise ValueError(f'Expected fixed-size float32 feature vectors: {path}')
        count = parquet.metadata.num_rows
        if count < 1 or meta['extracted_count'] != count or meta['population_available'] != count:
            raise ValueError(f'Embedding row count differs from provenance: {path}')
        if parquet.schema_arrow.field('label').type != pa.int64() or parquet.schema_arrow.field('row_id').type != pa.string():
            raise ValueError(f'Invalid labels/row IDs schema: {path}')
        labels = np.empty(count, dtype=np.int64)
        chunks, offset = [], 0
        for batch in parquet.iter_batches(batch_size=batch_size, columns=['label', 'row_id']):
            y, ids = batch.column(0), batch.column(1)
            lengths = pc.utf8_length(ids)
            if y.null_count or ids.null_count or pc.max(lengths).as_py() > 256 or pc.min(lengths).as_py() < 1:
                raise ValueError(f'Null labels/IDs or empty/overlong operational ID: {path}')
            labels[offset:offset + batch.num_rows] = y.to_numpy(zero_copy_only=False)
            chunks.append(ids)
            offset += batch.num_rows
        if offset != count or not np.isin(np.unique(labels), class_order).all() or _signature(path) != signature:
            raise ValueError(f'Invalid labels or changing row metadata: {path}')
        records = [{'path': str(path), 'sha256': digest, 'size_bytes': signature[2], 'quality': 'observed'},
                   {'path': str(meta_path), 'sha256': _sha256_file(meta_path), 'size_bytes': meta_path.stat().st_size, 'quality': 'observed'}]
        if _signature(meta_path) != meta_signature:
            raise ValueError(f'Embedding manifest changed during validation: {meta_path}')
        embeddings[name] = VisualizationEmbedding(path, meta, labels, CompactRowIDs(chunks),
            SelectedParquetFeatures(path, parquet, int(field.list_size), signature, batch_size), records)
    return embeddings


def load_visualization_taxonomy(path: Path | str, class_order: list[int]) -> tuple[pd.DataFrame, dict[int, str]]:
    """Load an explicit label→Genus mapping without reindexing classifier labels."""
    path = Path(path).resolve()
    log_artifact_path('visualization taxonomy', path)
    table = pd.read_parquet(path)
    if not {'id', 'Genus'}.issubset(table.columns):
        raise ValueError('Visualization taxonomy requires id and Genus columns')
    ids = pd.to_numeric(table['id'], errors='raise')
    if ids.isna().any() or (ids < 1).any() or (ids % 1 != 0).any() or ids.duplicated().any():
        raise ValueError('Taxonomy IDs must be unique positive integers')
    labels = ids.astype(np.int64) - 1
    if 'model_label' in table and not pd.to_numeric(table['model_label'], errors='raise').eq(labels).all():
        raise ValueError('Taxonomy model_label must equal id - 1')
    if set(labels) != set(class_order):
        raise ValueError('Taxonomy IDs and archived classifier class order differ')
    table = table.copy()
    table['model_label'] = labels
    names = table['Genus'].astype('string').str.strip()
    missing = names.isna() | names.eq('').fillna(False) | names.str.startswith('Unassigned', na=False)
    table['Genus'] = names.mask(missing, pd.NA)
    mapping = {int(row['model_label']): str(row['Genus']) for _, row in table.iterrows() if pd.notna(row['Genus'])}
    if not mapping:
        raise ValueError('No known genus in visualization taxonomy')
    return table, mapping


def create_visualization_fixtures(directory: Path | str, *, seed: int = 42) -> tuple[Path, Path]:
    """Create tiny, explicitly synthetic test embeddings; never initialize a model."""
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    labels = np.arange(48, dtype=np.int64) % 6
    ids = np.asarray([f'synthetic-test:{i}' for i in range(48)], dtype=str)
    code_hash = _sha256_file(__file__)
    for index, name in enumerate(MODEL_NAMES):
        features = np.random.default_rng(seed + index).normal(size=(48, 8)).astype(np.float32)
        manifest = {'verified': True, 'variant': name, 'split': 'test', 'mode': 'smoke',
                    'checkpoint': 'synthetic-fixture-no-model', 'checkpoint_sha256': 'synthetic',
                    'class_order': list(range(6)), 'partitions': ['synthetic-shared-test'],
                    'partition_sha256': [hashlib.sha256(b'synthetic-shared-test').hexdigest()],
                    'encoding': {'channels': ['A', 'C', 'G', 'T'], 'length': 512},
                    'extractor_code_sha256': code_hash, 'extractor': 'synthetic-embedding-fixture:v1',
                    'sampling': {'rule': 'synthetic', 'seed': seed}, 'population_available': 48, 'extracted_count': 48}
        save_embedding_artifact(EmbeddingArtifact(features, labels, ids, manifest), directory / 'features', batch_size=8)
    tax_path = directory / 'taxonomy.parquet'
    pd.DataFrame({'id': np.arange(1, 7), 'model_label': np.arange(6),
                  'Genus': [f'SyntheticGenus_{i // 2}' for i in range(6)]}).to_parquet(tax_path, index=False)
    return directory, tax_path
