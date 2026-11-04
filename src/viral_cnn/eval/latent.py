"""
Latent space analysis, pre-classifier embedding extraction, cosine KNN,
Silhouette scoring with subsampling audit, hierarchical taxonomic correlation,
and independent linear probing.
"""

from __future__ import annotations

import math
import atexit
import hashlib
import json
import os
import re
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import mpmath
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from viral_cnn.utils.logging_utils import log_artifact_path


class LatentMemoryBudget:
    """Allocate latent-analysis arrays in RAM up to a shared budget, then spill to disk."""

    def __init__(
        self,
        budget_bytes: int,
        spill_dir: Union[str, Path],
        *,
        observe_system_memory: bool = True,
    ) -> None:
        if not isinstance(budget_bytes, int) or budget_bytes < 0:
            raise ValueError("Latent RAM budget must be a non-negative integer number of bytes")
        self.budget_bytes = budget_bytes
        self.spill_dir = Path(spill_dir)
        self.observe_system_memory = observe_system_memory
        self.ram_bytes = 0
        self.spill_bytes = 0
        self.spill_paths: List[Path] = []
        self._arrays: List[np.ndarray] = []
        self._closed = False
        atexit.register(self.close)

    @staticmethod
    def available_ram_bytes() -> int:
        """Return currently available host RAM using procfs or the POSIX fallback."""
        meminfo = Path("/proc/meminfo")
        if meminfo.is_file():
            for line in meminfo.read_text().splitlines():
                if line.startswith("MemAvailable:"):
                    available = int(line.split()[1]) * 1024
                    break
            else:
                available = 0
            for current_path, maximum_path in (
                (Path("/sys/fs/cgroup/memory.current"), Path("/sys/fs/cgroup/memory.max")),
                (Path("/sys/fs/cgroup/memory/memory.usage_in_bytes"), Path("/sys/fs/cgroup/memory/memory.limit_in_bytes")),
            ):
                if current_path.is_file() and maximum_path.is_file():
                    try:
                        current = int(current_path.read_text().strip())
                        raw_maximum = maximum_path.read_text().strip()
                        if raw_maximum != "max":
                            maximum = int(raw_maximum)
                            if maximum > 0:
                                available = min(available, max(0, maximum - current))
                    except (OSError, ValueError):
                        pass
            if available:
                return available
        try:
            return int(os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE"))
        except (AttributeError, OSError, ValueError) as exc:
            raise RuntimeError("Cannot determine available host RAM; set LATENT_RAM_BUDGET_MB explicitly") from exc

    @classmethod
    def from_environment(cls, *, spill_dir: Optional[Union[str, Path]] = None) -> "LatentMemoryBudget":
        """Build the execution-wide memory budget from the notebook environment."""
        raw = os.environ.get("LATENT_RAM_BUDGET_MB")
        if raw is None:
            budget_bytes = int(cls.available_ram_bytes() * 0.30)
        else:
            if not re.fullmatch(r"\d+", raw.strip()):
                raise ValueError("LATENT_RAM_BUDGET_MB must be a non-negative integer")
            budget_bytes = int(raw) * 1024 * 1024
        target = Path(spill_dir or os.environ.get("LATENT_SPILL_DIR", tempfile.gettempdir()))
        return cls(budget_bytes, target)

    @property
    def spill_count(self) -> int:
        return len(self.spill_paths)

    def release(self, *arrays: np.ndarray) -> None:
        """Release arrays no longer needed and delete only their owned spill files."""
        for array in arrays:
            index = next((i for i, owned in enumerate(self._arrays) if owned is array), None)
            if index is None:
                continue
            self._arrays.pop(index)
            nbytes = int(array.nbytes)
            if isinstance(array, np.memmap):
                array.flush()
                path = Path(array.filename)
                mmap = getattr(array, "_mmap", None)
                if mmap is not None:
                    mmap.close()
                path.unlink(missing_ok=True)
                self.spill_bytes -= nbytes
                self.spill_paths = [item for item in self.spill_paths if item != path]
            else:
                self.ram_bytes -= nbytes

    def report(self) -> Dict[str, Any]:
        """Return the effective budget and current managed storage totals."""
        return {
            "budget_bytes": self.budget_bytes,
            "ram_bytes": self.ram_bytes,
            "spill_bytes": self.spill_bytes,
            "spill_count": self.spill_count,
            "spill_dir": str(self.spill_dir.resolve()),
        }

    def allocate(self, shape: Tuple[int, ...], dtype: Any, name: str) -> np.ndarray:
        """Allocate a zero-initialized array in RAM or in an exclusive disk-backed memmap."""
        if self._closed:
            raise RuntimeError("Latent memory budget is closed")
        dtype = np.dtype(dtype)
        nbytes = int(np.prod(shape, dtype=np.int64)) * dtype.itemsize
        pressure = self.observe_system_memory and self.available_ram_bytes() < self.budget_bytes
        if nbytes == 0 or (not pressure and self.ram_bytes + nbytes <= self.budget_bytes):
            try:
                array = np.empty(shape, dtype=dtype)
                self.ram_bytes += nbytes
                self._arrays.append(array)
                return array
            except MemoryError:
                pass

        self.spill_dir.mkdir(parents=True, exist_ok=True)
        safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", name)[:48] or "array"
        path = self.spill_dir / f"latent_{os.getpid()}_{uuid.uuid4().hex}_{safe_name}.mmap"
        try:
            with path.open("xb") as stream:
                if nbytes:
                    os.posix_fallocate(stream.fileno(), 0, nbytes)
            array = np.memmap(path, mode="r+", dtype=dtype, shape=shape)
        except (OSError, AttributeError) as exc:
            path.unlink(missing_ok=True)
            raise OSError(f"Cannot allocate latent spill array at {path}: {exc}") from exc
        array.fill(0)
        self.spill_paths.append(path)
        self.spill_bytes += nbytes
        self._arrays.append(array)
        log_artifact_path("latent spill array", path, extra=f"shape={shape}, dtype={dtype}, bytes={nbytes}")
        return array

    def close(self) -> None:
        """Flush and remove only temporary arrays owned by this budget."""
        if self._closed:
            return
        for array in self._arrays:
            if isinstance(array, np.memmap):
                array.flush()
                mmap = getattr(array, "_mmap", None)
                if mmap is not None:
                    mmap.close()
        for path in self.spill_paths:
            path.unlink(missing_ok=True)
        self._arrays.clear()
        self._closed = True


@dataclass
class EmbeddingArtifact:
    """Embeddings plus the provenance needed by the audited evaluation path."""

    features: np.ndarray
    labels: np.ndarray
    row_ids: np.ndarray
    manifest: Dict[str, Any]

    def validate(self) -> None:
        n = len(self.features)
        if len(self.labels) != n or len(self.row_ids) != n:
            raise ValueError("Embedding artifact features, labels, and row identities have different lengths")
        if not self.manifest.get("verified"):
            raise ValueError("Embedding artifact provenance is not verified")
        if self.manifest.get("split") not in {"train", "validation", "test"}:
            raise ValueError(f"Invalid embedding split: {self.manifest.get('split')!r}")


def _sha256_file(path: Union[str, Path]) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_embedding_artifact(
    artifact: EmbeddingArtifact,
    cache_prefix: Union[str, Path],
    *,
    batch_size: int = 4096,
) -> Path:
    """Write an immutable split artifact by row groups without duplicating all features in RAM."""
    artifact.validate()
    if batch_size < 1:
        raise ValueError("Artifact write batch_size must be a positive integer")
    prefix = Path(cache_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    fingerprint = hashlib.sha256(json.dumps(artifact.manifest, sort_keys=True).encode()).hexdigest()
    parquet_path = prefix.with_name(f"{prefix.name}_{artifact.manifest['variant']}_{artifact.manifest['split']}_{fingerprint}.parquet")
    manifest_path = parquet_path.with_suffix(".json")
    if parquet_path.exists() or manifest_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing embedding artifact: {parquet_path}")
    import pyarrow as pa
    import pyarrow.parquet as pq

    print(f"[LOG_PATH] Saving embedding artifact to: {parquet_path.resolve()}")
    with tempfile.NamedTemporaryFile(dir=prefix.parent, suffix=".parquet", delete=False) as temp:
        temp_path = Path(temp.name)
    temp_manifest = manifest_path.with_suffix(".json.tmp")
    try:
        feature_type = pa.list_(pa.float32(), list_size=int(artifact.features.shape[1]))
        schema = pa.schema([("features", feature_type), ("label", pa.int64()), ("row_id", pa.string())])
        with pq.ParquetWriter(temp_path, schema=schema) as writer:
            for start in range(0, len(artifact.features), batch_size):
                stop = min(start + batch_size, len(artifact.features))
                feature_values = np.asarray(artifact.features[start:stop], dtype=np.float32)
                flat = pa.array(feature_values.reshape(-1), type=pa.float32())
                feature_array = pa.FixedSizeListArray.from_arrays(flat, list_size=feature_values.shape[1])
                table = pa.Table.from_arrays(
                    [feature_array, pa.array(artifact.labels[start:stop], type=pa.int64()),
                     pa.array(artifact.row_ids[start:stop].astype(str), type=pa.string())],
                    schema=schema,
                )
                writer.write_table(table, row_group_size=batch_size)
        os.replace(temp_path, parquet_path)
        manifest = dict(artifact.manifest, schema_version=1, artifact_sha256=_sha256_file(parquet_path), fingerprint=fingerprint)
        temp_manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        os.replace(temp_manifest, manifest_path)
    except Exception:
        parquet_path.unlink(missing_ok=True)
        manifest_path.unlink(missing_ok=True)
        temp_manifest.unlink(missing_ok=True)
        raise
    finally:
        temp_path.unlink(missing_ok=True)
    return parquet_path


def load_embedding_artifact(
    path: Union[str, Path],
    expected: Dict[str, Any],
    *,
    memory_manager: Optional[LatentMemoryBudget] = None,
    batch_size: int = 4096,
) -> EmbeddingArtifact:
    """Load a verified cache into budget-managed arrays, reading Parquet in batches."""
    required_identity = {
        "variant", "split", "partitions", "partition_sha256", "checkpoint", "checkpoint_sha256",
        "class_order", "encoding", "sampling", "mode", "extractor_code_sha256",
    }
    missing = required_identity - set(expected)
    if missing:
        raise ValueError(f"Cache identity is incomplete; missing expected fields: {sorted(missing)}")
    path = Path(path)
    print(f"[LOG_PATH] Loading embedding artifact from: {path.resolve()}")
    manifest_path = path.with_suffix(".json")
    if not path.is_file() or not manifest_path.is_file():
        raise ValueError(f"Legacy or incomplete embedding cache (data/manifest missing): {path}")
    manifest = json.loads(manifest_path.read_text())
    missing_manifest = required_identity - set(manifest)
    if missing_manifest:
        raise ValueError(f"Cache manifest is incomplete: {sorted(missing_manifest)}")
    fingerprint_payload = {key: value for key, value in manifest.items() if key not in {"schema_version", "artifact_sha256", "fingerprint"}}
    actual_fingerprint = hashlib.sha256(json.dumps(fingerprint_payload, sort_keys=True).encode()).hexdigest()
    if manifest.get("fingerprint") != actual_fingerprint:
        raise ValueError(f"Embedding cache fingerprint mismatch: {path}")
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(f"Stale embedding cache {path}: {key} expected {value!r}, found {manifest.get(key)!r}")
    if manifest.get("artifact_sha256") != _sha256_file(path):
        raise ValueError(f"Corrupt embedding cache checksum: {path}")
    import pyarrow as pa
    import pyarrow.parquet as pq

    if batch_size < 1:
        raise ValueError("Artifact read batch_size must be a positive integer")
    parquet = pq.ParquetFile(path)
    n_rows = parquet.metadata.num_rows
    feature_field = parquet.schema_arrow.field("features").type
    dimension = int(feature_field.list_size) if pa.types.is_fixed_size_list(feature_field) else None
    if dimension is None:
        first = next(parquet.iter_batches(batch_size=1, columns=["features"]), None)
        if first is None:
            raise ValueError(f"Embedding artifact is empty: {path}")
        dimension = len(first.column(0)[0].as_py())
    manager = memory_manager or LatentMemoryBudget.from_environment()
    features = manager.allocate((n_rows, dimension), np.float32, "loaded_features")
    labels = manager.allocate((n_rows,), np.int64, "loaded_labels")
    row_ids = manager.allocate((n_rows,), "U256", "loaded_row_ids")
    offset = 0
    for record_batch in parquet.iter_batches(batch_size=batch_size):
        size = record_batch.num_rows
        feature_column = record_batch.column(record_batch.schema.get_field_index("features"))
        if pa.types.is_fixed_size_list(feature_column.type):
            feature_values = feature_column.values.to_numpy(zero_copy_only=False).reshape(size, dimension)
        else:
            feature_values = np.stack([np.asarray(value.as_py(), dtype=np.float32) for value in feature_column])
        features[offset:offset + size] = feature_values
        labels[offset:offset + size] = record_batch.column(record_batch.schema.get_field_index("label")).to_numpy()
        ids = record_batch.column(record_batch.schema.get_field_index("row_id")).to_pylist()
        if any(len(value) > 256 for value in ids):
            raise ValueError("Embedding row identity exceeds the supported 256-character storage width")
        row_ids[offset:offset + size] = ids
        offset += size
    if offset != n_rows:
        raise RuntimeError(f"Embedding artifact row count changed while reading {path}: {offset} != {n_rows}")
    artifact = EmbeddingArtifact(features, labels, row_ids, manifest)
    artifact.validate()
    return artifact


# ---------------------------------------------------------------------------
# Statistical Correlation Helpers (Pure NumPy/mpmath, zero external C-deps)
# ---------------------------------------------------------------------------

def compute_pearson_r(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """Computes Pearson correlation coefficient r and two-tailed p-value."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n = len(x)
    if n < 3:
        return float("nan"), float("nan")

    mx = x - np.mean(x)
    my = y - np.mean(y)
    denom = np.sqrt(np.sum(mx ** 2) * np.sum(my ** 2))
    if denom == 0 or np.isnan(denom):
        return float("nan"), float("nan")

    r = float(np.sum(mx * my) / denom)
    r = max(-1.0, min(1.0, r))

    if abs(r) == 1.0:
        return r, 0.0

    # Student-t statistic: t = r * sqrt((n - 2) / (1 - r^2))
    df = n - 2
    t = r * math.sqrt(df / (1.0 - r ** 2))
    # Two-tailed p-value via regularized incomplete beta function I_x(df/2, 1/2)
    x_val = df / (df + t ** 2)
    p_val = float(mpmath.betainc(0.5 * df, 0.5, 0, x_val, regularized=True))
    return r, p_val


def compute_spearman_rho(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """Computes Spearman rank correlation coefficient rho and two-tailed p-value."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n = len(x)
    if n < 3:
        return float("nan"), float("nan")

    def rankdata(a: np.ndarray) -> np.ndarray:
        sorter = np.argsort(a)
        inv = np.empty(n, dtype=np.intp)
        inv[sorter] = np.arange(n)

        a_sorted = a[sorter]
        obs = np.r_[True, a_sorted[1:] != a_sorted[:-1]]
        dense = obs.cumsum()[inv]

        # Cumulative counts of each unique value
        count = np.r_[np.nonzero(obs)[0], n]
        return 0.5 * (count[dense] + count[dense - 1] + 1)

    rx = rankdata(x)
    ry = rankdata(y)
    return compute_pearson_r(rx, ry)


# ---------------------------------------------------------------------------
# S1: Pre-Classifier Embedding Extraction & Cosine KNN
# ---------------------------------------------------------------------------

def extract_latent_embeddings(
    model: nn.Module,
    dataloader: Any,
    device: str = "cpu",
    *,
    return_identities: bool = False,
    return_logits: bool = False,
    return_predictions: bool = False,
    memory_manager: Optional[LatentMemoryBudget] = None,
) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, ...]]:
    """
    Extracts high-dimensional pre-classifier embeddings by registering a forward
    pre-hook on `model.fc`. Returns (features, labels) as NumPy arrays.
    """
    model.eval()
    dev = torch.device(device)
    target_model = getattr(model, "_orig_mod", model)

    if not hasattr(target_model, "fc") or not isinstance(target_model.fc, nn.Linear):
        raise AttributeError("Model must have an `fc` attribute of type `nn.Linear`")

    if return_logits and return_predictions:
        raise ValueError("Request either full logits or Top-1 predictions, not both")
    manager = memory_manager or LatentMemoryBudget.from_environment()

    dataset = getattr(dataloader, "dataset", None)
    expected_rows = getattr(dataset, "total_samples", None)
    if expected_rows is None and dataset is not None:
        try:
            expected_rows = len(dataset)
        except TypeError:
            expected_rows = None
    if expected_rows is None and isinstance(dataloader, (list, tuple)):
        expected_rows = sum(len(batch[1]) for batch in dataloader)
    if expected_rows is None:
        try:
            batch_count = len(dataloader)
            batch_size = getattr(dataloader, "batch_size", None)
            if batch_size is not None:
                expected_rows = batch_count * batch_size
        except TypeError:
            expected_rows = None
    if expected_rows is None:
        raise ValueError("Latent extraction requires a dataset with a known sample count for bounded storage")
    expected_rows = int(expected_rows)
    if expected_rows < 0:
        raise ValueError("Latent extraction sample count cannot be negative")

    features = manager.allocate((expected_rows, target_model.fc.in_features), np.float32, "extracted_features")
    labels_out = manager.allocate((expected_rows,), np.int64, "extracted_labels")
    identities_out = manager.allocate((expected_rows,), "U256", "extracted_row_ids") if return_identities else None
    logits_out = manager.allocate((expected_rows, target_model.fc.out_features), np.float32, "extracted_logits") if return_logits else None
    predictions_out = manager.allocate((expected_rows,), np.int64, "extracted_top1") if return_predictions else None
    offset = 0
    captured_features: Optional[torch.Tensor] = None

    def pre_hook_fn(module: nn.Module, inputs: Tuple[torch.Tensor, ...]) -> None:
        nonlocal captured_features
        captured_features = inputs[0].detach()

    handle = target_model.fc.register_forward_pre_hook(pre_hook_fn)

    try:
        with torch.inference_mode():
            for batch in dataloader:
                if isinstance(batch, (tuple, list)) and len(batch) >= 2:
                    inputs, labels = batch[0], batch[1]
                    row_ids = batch[2] if len(batch) >= 3 else None
                elif isinstance(batch, dict):
                    inputs = batch["sequence"] if "sequence" in batch else batch["inputs"]
                    labels = batch["label"] if "label" in batch else batch["labels"]
                    row_ids = batch.get("row_id", batch.get("row_ids"))
                else:
                    raise ValueError(f"Unrecognized batch format from dataloader: {type(batch)}")

                inputs = inputs.to(dev)
                logits = model(inputs)
                if captured_features is None:
                    raise RuntimeError("The classifier pre-hook did not capture an embedding batch")
                batch_features = captured_features.to(device="cpu", dtype=torch.float32).numpy()
                batch_labels = labels.detach().cpu().numpy() if isinstance(labels, torch.Tensor) else np.asarray(labels)
                batch_size = len(batch_labels)
                stop = offset + batch_size
                if stop > expected_rows:
                    raise RuntimeError(f"Latent extraction produced more than {expected_rows} declared rows")
                features[offset:stop] = batch_features
                labels_out[offset:stop] = batch_labels
                if return_identities:
                    if row_ids is None:
                        raise ValueError("Identity-aware extraction requires row IDs from the source dataset")
                    batch_ids = np.asarray(row_ids, dtype=str)
                    if len(batch_ids) != batch_size or any(len(value) > 256 for value in batch_ids):
                        raise ValueError("Row identities must align with the batch and fit within 256 characters")
                    identities_out[offset:stop] = batch_ids
                if return_logits:
                    logits_out[offset:stop] = logits.detach().to(device="cpu", dtype=torch.float32).numpy()
                elif return_predictions:
                    predictions_out[offset:stop] = logits.argmax(dim=1).detach().cpu().numpy()
                captured_features = None
                offset = stop
    finally:
        handle.remove()

    if offset != expected_rows:
        raise RuntimeError(f"Latent extraction produced {offset} rows; expected {expected_rows}")
    result: Tuple[Any, ...] = (features, labels_out)
    if return_identities:
        result += (identities_out,)
    if return_logits:
        result += (logits_out,)
    if return_predictions:
        result += (predictions_out,)
    return result


def _knn_workspace_bytes(query_rows: int, reference_rows: int, dimension: int, k: int) -> int:
    """Conservative bound for explicit gather, normalization, score and key tensors."""
    return 12 * dimension * (query_rows + reference_rows) + 64 * query_rows * reference_rows + 64 * query_rows * k


def _knn_ordered_keys(similarity: torch.Tensor, original_indices: torch.Tensor) -> torch.Tensor:
    """Encode float32 score and smallest original index as one sortable int64 key."""
    bits = similarity.contiguous().view(torch.int32).to(torch.int64) & 0xFFFFFFFF
    ordered = torch.where((bits & 0x80000000) != 0, (~bits) & 0xFFFFFFFF, bits ^ 0x80000000)
    ordered = torch.where(similarity == 0, 0x80000000, ordered) - 0x80000000
    return ordered * (1 << 32) + (0xFFFFFFFF - original_indices[None, :])


def _hash_ordered_ids(ids: np.ndarray, indices: np.ndarray) -> str:
    """Hash ordered identities without allocating their full selected array."""
    digest = hashlib.sha256()
    for start in range(0, len(indices), 8192):
        digest.update(np.asarray(ids[indices[start:start + 8192]]).tobytes())
    return digest.hexdigest()


def save_knn_identity_audit(evaluations: List[Dict[str, Any]], path: Union[str, Path]) -> Path:
    """Atomically publish the compact C08 KNN audit JSON for one notebook run."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise FileExistsError(f"Refusing to overwrite KNN audit: {target}")
    with tempfile.NamedTemporaryFile(dir=target.parent, suffix=".tmp", delete=False) as temp:
        temporary = Path(temp.name)
    try:
        temporary.write_text(json.dumps({"schema_version": "c08_knn_audit_v1", "evaluations": evaluations}, indent=2) + "\n")
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    log_artifact_path("KNN identity audit", target)
    return target


def evaluate_cosine_knn(
    features: np.ndarray,
    labels: np.ndarray,
    k: int = 5,
    test_size: float = 0.2,
    random_state: int = 42,
    batch_size: int = 1000,
    device: str = "cpu",
    taxonomic_rank: str = "Species",
    row_ids: Optional[np.ndarray] = None,
    *,
    workspace_mb: int = 1024,
    query_batch_size: Optional[int] = None,
    reference_batch_size: Optional[int] = None,
    selected_indices: Optional[np.ndarray] = None,
    compact: bool = False,
    audit_output_path: Optional[Union[str, Path]] = None,
    embedding_artifact_sha256: Optional[str] = None,
) -> Dict[str, Any]:
    """Evaluate exact cosine KNN with bounded reference/query tiles and C08 audit."""
    if not isinstance(workspace_mb, int) or isinstance(workspace_mb, bool) or workspace_mb < 1:
        raise ValueError("workspace_mb must be a positive integer number of MiB")
    if query_batch_size is None:
        query_batch_size = batch_size
    if not isinstance(query_batch_size, int) or isinstance(query_batch_size, bool) or query_batch_size < 1:
        raise ValueError("query_batch_size must be a positive integer")
    if reference_batch_size is not None and (
        not isinstance(reference_batch_size, int) or isinstance(reference_batch_size, bool) or reference_batch_size < 1
    ):
        raise ValueError("reference_batch_size must be a positive integer")
    if not isinstance(k, int) or k < 1:
        raise ValueError("k must be a positive integer")
    if audit_output_path is not None and not embedding_artifact_sha256:
        raise ValueError("embedding_artifact_sha256 is required for a persisted KNN audit")
    if len(features.shape) != 2 or features.shape[1] < 1:
        raise ValueError("KNN features must have shape (rows, dimensions)")
    if len(features) >= 1 << 32:
        raise ValueError("C08 tie ordering supports fewer than 2^32 feature rows")

    chosen = np.arange(len(features), dtype=np.intp) if selected_indices is None else np.asarray(selected_indices, dtype=np.intp)
    if chosen.ndim != 1 or np.any(chosen < 0) or np.any(chosen >= len(features)) or len(np.unique(chosen)) != len(chosen):
        raise ValueError("selected_indices must be unique valid feature row indices")
    labels = np.asarray(labels)
    if len(labels) != len(chosen):
        raise ValueError("Length mismatch between selected features and labels")
    ids = np.asarray(row_ids if row_ids is not None else [f"row:{i}" for i in range(len(features))], dtype=str)
    if len(ids) != len(features):
        raise ValueError("row_ids length must match feature rows")

    n_samples = len(chosen)
    rng = np.random.default_rng(random_state)
    unique_classes, counts = np.unique(labels, return_counts=True)
    sorted_positions = np.argsort(labels, kind="stable")
    ref_local: List[int] = []
    query_local: List[int] = []
    offset = 0
    for count in counts:
        class_positions = sorted_positions[offset:offset + count].copy()
        rng.shuffle(class_positions)
        n_query = max(1, int(round(count * test_size))) if count > 1 else 0
        n_query = min(n_query, count - 1)
        query_local.extend(class_positions[:n_query])
        ref_local.extend(class_positions[n_query:])
        offset += count
    ref_local_arr = np.asarray(ref_local, dtype=np.intp)
    query_local_arr = np.asarray(query_local, dtype=np.intp)
    ref_idx_arr = chosen[ref_local_arr]
    query_idx_arr = chosen[query_local_arr]
    selected_lookup = None
    if selected_indices is not None:
        selected_lookup = np.full(len(features), -1, dtype=np.intp)
        selected_lookup[chosen] = np.arange(n_samples, dtype=np.intp)
    if np.intersect1d(ref_idx_arr, query_idx_arr).size:
        raise RuntimeError("Reference and query index partitions must be strictly disjoint")
    seen_ids = set()
    for start in range(0, len(ref_idx_arr), 8192):
        seen_ids.update(str(value) for value in ids[ref_idx_arr[start:start + 8192]])
    for start in range(0, len(query_idx_arr), 8192):
        for value in ids[query_idx_arr[start:start + 8192]]:
            if str(value) in seen_ids:
                raise ValueError(f"Operational row identities overlap between KNN reference and query: {value}")

    n_query_classes = len(np.unique(labels[query_local_arr]))
    common = {
        "taxonomic_rank": taxonomic_rank,
        "k_requested": k,
        "n_reference": len(ref_idx_arr),
        "n_query": len(query_idx_arr),
        "n_classes_total": len(unique_classes),
        "n_classes_with_query": n_query_classes,
        "n_classes_reference_only": len(unique_classes) - n_query_classes,
        "reference_indices": ref_idx_arr,
        "query_indices": query_idx_arr,
        "identity_overlap_count": 0,
        "zero_self_neighbor": True,
        "top5_definition": "true label appears among the requested nearest-neighbor labels",
        "protocol": "c08_tiled_cosine_v1",
        "device": str(torch.device(device)),
        "workspace_mb": workspace_mb,
    }
    ref_hash = _hash_ordered_ids(ids, ref_idx_arr)
    query_hash = _hash_ordered_ids(ids, query_idx_arr)
    unavailable = len(query_idx_arr) == 0 or len(ref_idx_arr) < k
    if unavailable:
        result = dict(common, available=False, k=0, global_knn_accuracy=float("nan"),
                      global_knn_top5_accuracy=float("nan"),
                      per_cluster_metrics=pd.DataFrame(columns=["class_label", "test_samples", "knn_accuracy", "knn_top5_accuracy", "taxonomic_rank"]),
                      unavailable_reason="no_stratified_queries" if len(query_idx_arr) == 0 else "fewer_than_requested_neighbors",
                      self_neighbor_count=0)
        if not compact:
            result.update(reference_ids=ids[ref_idx_arr], query_ids=ids[query_idx_arr])
        result["audit_summary"] = {
            "model": None, "rank": taxonomic_rank, "available": False,
            "protocol": "c08_tiled_cosine_v1", "embedding_artifact_sha256": embedding_artifact_sha256,
            "n_reference": len(ref_idx_arr), "n_query": len(query_idx_arr), "k_requested": k,
            "reference_ids_sha256": ref_hash, "query_ids_sha256": query_hash,
            "neighbor_ids_sha256": hashlib.sha256().hexdigest(), "self_neighbor_count": 0,
            "neighbor_parquet_path": None, "neighbor_parquet_sha256": None,
            "device": str(torch.device(device)), "workspace_mb": workspace_mb,
            "query_batch_size_effective": 0, "reference_batch_size_effective": 0,
        }
        return result

    dimension = int(features.shape[1])
    requested_bytes = workspace_mb * 1024 * 1024
    minimum_bytes = _knn_workspace_bytes(1, k, dimension, k)
    if requested_bytes < minimum_bytes:
        raise RuntimeError(f"KNN workspace requested {requested_bytes} bytes; minimum {minimum_bytes} bytes for one query and {k} references")
    query_rows = min(query_batch_size, len(query_idx_arr))
    while query_rows > 1 and _knn_workspace_bytes(query_rows, k, dimension, k) > requested_bytes:
        query_rows = max(1, query_rows // 2)
    ref_rows = (requested_bytes - 12 * dimension * query_rows - 64 * query_rows * k) // (12 * dimension + 64 * query_rows)
    ref_rows = max(1, min(int(ref_rows), len(ref_idx_arr)))
    if reference_batch_size is not None:
        ref_rows = min(ref_rows, reference_batch_size)
    peak_estimate = _knn_workspace_bytes(query_rows, ref_rows, dimension, k)
    if peak_estimate > requested_bytes:
        raise RuntimeError(f"KNN workspace requested {requested_bytes} bytes; minimum {minimum_bytes} bytes")

    dev = torch.device(device)
    y_true = np.asarray(labels[query_local_arr], dtype=np.int64)
    y_pred = np.empty(len(query_idx_arr), dtype=np.int64)
    top5_correct = np.empty(len(query_idx_arr), dtype=bool)
    neighbor_indices = np.empty((len(query_idx_arr), k), dtype=np.intp) if not compact else None
    neighbor_hash = hashlib.sha256()
    self_neighbor_count = 0
    audit_path = Path(audit_output_path) if audit_output_path is not None else None
    temporary = None
    writer = None
    if audit_path is not None:
        import pyarrow as pa
        import pyarrow.parquet as pq

        audit_path.parent.mkdir(parents=True, exist_ok=True)
        if audit_path.exists():
            raise FileExistsError(f"Refusing to overwrite KNN neighbor audit: {audit_path}")
        with tempfile.NamedTemporaryFile(dir=audit_path.parent, suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
        schema = pa.schema([("query_index", pa.int64()), ("neighbor_indices", pa.list_(pa.int64(), list_size=k))])
    try:
        if temporary is not None:
            writer = pq.ParquetWriter(temporary, schema=schema)
        for start in range(0, len(query_idx_arr), query_rows):
            stop = min(start + query_rows, len(query_idx_arr))
            batch_q = query_idx_arr[start:stop]
            query_tensor = F.normalize(torch.as_tensor(np.asarray(features[batch_q]), dtype=torch.float32, device=dev), p=2, dim=1)
            best_keys = None
            for ref_start in range(0, len(ref_idx_arr), ref_rows):
                batch_r = ref_idx_arr[ref_start:ref_start + ref_rows]
                ref_tensor = F.normalize(torch.as_tensor(np.asarray(features[batch_r]), dtype=torch.float32, device=dev), p=2, dim=1)
                similarity = torch.mm(query_tensor, ref_tensor.T)
                reference_indices = torch.as_tensor(batch_r, dtype=torch.int64, device=dev)
                keys = _knn_ordered_keys(similarity, reference_indices)
                tile_keys = torch.topk(keys, k=min(k, len(batch_r)), dim=1).values
                candidates = tile_keys if best_keys is None else torch.cat((best_keys, tile_keys), dim=1)
                best_keys = torch.topk(candidates, k=min(k, candidates.shape[1]), dim=1).values
                del ref_tensor, similarity, reference_indices, keys, tile_keys, candidates
            batch_neighbors = (0xFFFFFFFF - (best_keys & 0xFFFFFFFF)).to("cpu").numpy().astype(np.intp, copy=False)
            self_neighbor_count += int(np.any(batch_neighbors == batch_q[:, None], axis=1).sum())
            neighbor_hash.update(np.asarray(ids[batch_neighbors]).tobytes())
            neighbor_local = selected_lookup[batch_neighbors] if selected_lookup is not None else batch_neighbors
            neighbor_labels = np.asarray(labels[neighbor_local], dtype=np.int64)
            votes = (neighbor_labels[:, :, None] == neighbor_labels[:, None, :]).sum(axis=2)
            winning = votes == votes.max(axis=1, keepdims=True)
            y_pred[start:stop] = np.where(winning, neighbor_labels, np.iinfo(np.int64).max).min(axis=1)
            top5_correct[start:stop] = (neighbor_labels == y_true[start:stop, None]).any(axis=1)
            if neighbor_indices is not None:
                neighbor_indices[start:stop] = batch_neighbors
            if writer is not None:
                flat = pa.array(batch_neighbors.astype(np.int64, copy=False).reshape(-1), type=pa.int64())
                table = pa.Table.from_arrays([
                    pa.array(batch_q.astype(np.int64, copy=False), type=pa.int64()),
                    pa.FixedSizeListArray.from_arrays(flat, list_size=k),
                ], schema=schema)
                writer.write_table(table, row_group_size=query_rows)
        if writer is not None:
            writer.close()
            writer = None
            os.replace(temporary, audit_path)
    finally:
        if writer is not None:
            writer.close()
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    if self_neighbor_count:
        if audit_path is not None:
            audit_path.unlink(missing_ok=True)
        raise RuntimeError("A query identity appeared among its own KNN reference neighbors")

    correct = y_pred == y_true
    cluster_summary = (pd.DataFrame({"y_true": y_true, "correct": correct, "top5_correct": top5_correct})
                       .groupby("y_true")
                       .agg(test_samples=("correct", "count"), knn_accuracy=("correct", "mean"),
                            knn_top5_accuracy=("top5_correct", "mean"))
                       .reset_index().rename(columns={"y_true": "class_label"}))
    cluster_summary["taxonomic_rank"] = taxonomic_rank
    audit_summary = {
        "rank": taxonomic_rank, "available": True, "protocol": "c08_tiled_cosine_v1",
        "embedding_artifact_sha256": embedding_artifact_sha256,
        "n_reference": len(ref_idx_arr), "n_query": len(query_idx_arr), "k_requested": k,
        "reference_ids_sha256": ref_hash, "query_ids_sha256": query_hash,
        "neighbor_ids_sha256": neighbor_hash.hexdigest(), "self_neighbor_count": 0,
        "identity_overlap_count": 0, "neighbor_parquet_path": str(audit_path.resolve()) if audit_path else None,
        "neighbor_parquet_sha256": _sha256_file(audit_path) if audit_path else None,
        "device": str(dev), "workspace_mb": workspace_mb,
        "query_batch_size_effective": query_rows, "reference_batch_size_effective": ref_rows,
    }
    result = dict(common, available=True, k=k, global_knn_accuracy=float(np.mean(correct)),
                  global_knn_top5_accuracy=float(np.mean(top5_correct)),
                  per_cluster_metrics=cluster_summary, self_neighbor_count=0,
                  reference_fraction=len(ref_idx_arr) / n_samples,
                  query_fraction=len(query_idx_arr) / n_samples, y_true=y_true, y_pred=y_pred,
                  query_batch_size_effective=query_rows, reference_batch_size_effective=ref_rows,
                  workspace_peak_estimated_bytes=peak_estimate, audit_summary=audit_summary)
    if not compact:
        result.update(reference_ids=ids[ref_idx_arr], query_ids=ids[query_idx_arr],
                      neighbor_indices=neighbor_indices, neighbor_ids=ids[neighbor_indices])
    if audit_path is not None:
        log_artifact_path("KNN neighbor audit", audit_path)
    return result


# ---------------------------------------------------------------------------
# S2: Genome-Disjoint KNN Evaluation
# ---------------------------------------------------------------------------

def evaluate_genome_disjoint_knn(
    features: np.ndarray,
    labels: np.ndarray,
    genome_ids: np.ndarray,
    k: int = 5,
    random_state: int = 42,
    batch_size: int = 1000,
    device: str = "cpu",
    taxonomic_rank: str = "Species",
) -> Dict[str, Any]:
    """
    Evaluates KNN where reference and query sets are partitioned by genome accessions,
    guaranteeing zero genome accession overlap.
    """
    labels = np.asarray(labels)
    genome_ids = np.asarray(genome_ids)
    features = np.asarray(features, dtype=np.float32)

    if len(features) != len(labels) or len(labels) != len(genome_ids):
        raise ValueError("Length mismatch between features, labels, and genome_ids")

    # Map species to their unique genome accessions
    species_to_genomes: Dict[Any, List[Any]] = {}
    for sp, gid in zip(labels, genome_ids):
        species_to_genomes.setdefault(sp, set()).add(gid)

    # Isolate multi-genome species vs single-genome species
    multi_genome_species = {sp for sp, gids in species_to_genomes.items() if len(gids) >= 2}
    single_genome_species = {sp for sp, gids in species_to_genomes.items() if len(gids) < 2}

    rng = np.random.default_rng(random_state)
    ref_indices: List[int] = []
    query_indices: List[int] = []

    # Assign genomes of multi-genome species: 1 genome to query, remaining to ref
    for sp in multi_genome_species:
        gids = sorted(list(species_to_genomes[sp]))
        rng.shuffle(gids)
        query_gid = gids[0]
        ref_gids = set(gids[1:])

        sp_indices = np.where(labels == sp)[0]
        for idx in sp_indices:
            gid = genome_ids[idx]
            if gid == query_gid:
                query_indices.append(idx)
            elif gid in ref_gids:
                ref_indices.append(idx)

    # For single-genome species, put all into reference (closed-set requirement)
    for sp in single_genome_species:
        sp_indices = np.where(labels == sp)[0]
        ref_indices.extend(sp_indices)

    ref_arr = np.array(ref_indices, dtype=np.intp)
    query_arr = np.array(query_indices, dtype=np.intp)

    if len(query_arr) == 0:
        raise ValueError("No query samples available for genome-disjoint evaluation (all single-genome)")

    # Assert zero accession overlap between reference and query
    ref_accessions = set(genome_ids[ref_arr])
    query_accessions = set(genome_ids[query_arr])
    accession_overlap = ref_accessions.intersection(query_accessions)
    if len(accession_overlap) > 0:
        raise RuntimeError(f"Genome accession leak detected: {len(accession_overlap)} overlapping accessions")

    # Compute genome-disjoint KNN
    dev = torch.device(device)
    X_ref = F.normalize(torch.as_tensor(features[ref_arr], dtype=torch.float32).to(dev), p=2, dim=1)
    y_ref = torch.as_tensor(labels[ref_arr], dtype=torch.long).to(dev)

    X_query = torch.as_tensor(features[query_arr], dtype=torch.float32)
    y_query = torch.as_tensor(labels[query_arr], dtype=torch.long)

    k_eff = min(k, len(X_ref))
    preds: List[torch.Tensor] = []

    for i in range(0, len(X_query), batch_size):
        bx = F.normalize(X_query[i : i + batch_size].to(dev), p=2, dim=1)
        sim = torch.mm(bx, X_ref.T)
        topk = torch.topk(sim, k=k_eff, dim=1, largest=True)
        mode_preds, _ = torch.mode(y_ref[topk.indices], dim=1)
        preds.append(mode_preds.cpu())

    y_pred_disjoint = torch.cat(preds).numpy()
    y_true_disjoint = y_query.numpy()

    disjoint_acc = float(np.mean(y_pred_disjoint == y_true_disjoint))

    # Also compute fragment-split KNN on the identical multi-genome subset for comparison
    multi_mask = np.isin(labels, list(multi_genome_species))
    frag_res = evaluate_cosine_knn(
        features[multi_mask],
        labels[multi_mask],
        k=k,
        random_state=random_state,
        batch_size=batch_size,
        device=device,
        taxonomic_rank=taxonomic_rank,
    )
    fragment_acc = frag_res["global_knn_accuracy"]
    acc_drop = fragment_acc - disjoint_acc

    return {
        "taxonomic_rank": taxonomic_rank,
        "genome_disjoint_accuracy": disjoint_acc,
        "fragment_split_accuracy": fragment_acc,
        "accuracy_drop_percentage_points": acc_drop * 100.0,
        "n_multi_genome_species": len(multi_genome_species),
        "n_single_genome_species_excluded": len(single_genome_species),
        "excluded_species_labels": sorted(single_genome_species, key=str),
        "n_reference_samples": len(X_ref),
        "n_query_samples": len(X_query),
        "zero_accession_overlap": True,
    }


# ---------------------------------------------------------------------------
# S3: High-Dimensional Silhouette Score & Subsampling Audit
# ---------------------------------------------------------------------------

def compute_cosine_silhouette_fast(X: np.ndarray, y: np.ndarray) -> Tuple[float, np.ndarray]:
    """
    Exact vectorized computation of Silhouette Score under Cosine Distance.
    O(N * K) time and O(N + K * D) memory, numerically identical to scikit-learn.
    """
    X = np.asarray(X, dtype=np.float32)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    X_norm = X / norms

    y = np.asarray(y)
    unique_clusters, cluster_counts = np.unique(y, return_counts=True)
    K = len(unique_clusters)
    N = len(X)

    if K <= 1:
        return float("nan"), np.zeros(N, dtype=np.float32)

    cluster_sums = np.zeros((K, X.shape[1]), dtype=np.float32)
    for idx, c in enumerate(unique_clusters):
        cluster_sums[idx] = np.sum(X_norm[y == c], axis=0)

    # Precompute cluster map
    c_to_idx = {c: i for i, c in enumerate(unique_clusters)}
    sample_c_indices = np.array([c_to_idx[c] for c in y], dtype=np.intp)

    a_vals = np.zeros(N, dtype=np.float32)
    b_vals = np.zeros(N, dtype=np.float32)

    for i in range(N):
        c_idx = sample_c_indices[i]
        cnt = cluster_counts[c_idx]

        if cnt > 1:
            sum_excl = cluster_sums[c_idx] - X_norm[i]
            mean_sim = np.dot(X_norm[i], sum_excl) / (cnt - 1)
            a_vals[i] = 1.0 - mean_sim
        else:
            a_vals[i] = 0.0

        # Mean distance to all other clusters: 1.0 - max dot product with centroid
        other_indices = [idx for idx in range(K) if idx != c_idx]
        sims = np.dot(cluster_sums[other_indices] / cluster_counts[other_indices, None], X_norm[i])
        b_vals[i] = 1.0 - np.max(sims)

    max_ab = np.maximum(a_vals, b_vals)
    sil_samples = np.zeros(N, dtype=np.float32)
    valid_mask = max_ab > 0
    sil_samples[valid_mask] = (b_vals[valid_mask] - a_vals[valid_mask]) / max_ab[valid_mask]
    global_score = float(np.mean(sil_samples))

    return global_score, sil_samples


def evaluate_silhouette_with_audit(
    features: np.ndarray,
    labels: np.ndarray,
    max_samples: int = 50000,
    sampling_strategy: str = "random",
    random_state: int = 42,
    taxonomic_rank: str = "Species",
    selected_indices: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Computes Cosine Silhouette Score in the original high-dimensional feature space,
    auditing class retention and dropping singletons (<2 samples).
    """
    if selected_indices is None:
        features = np.asarray(features, dtype=np.float32)
        selected = None
    else:
        selected = np.asarray(selected_indices, dtype=np.intp)
        if selected.ndim != 1 or np.any(selected < 0) or np.any(selected >= len(features)):
            raise ValueError("Silhouette selected_indices must be valid feature row indices")
    labels = np.asarray(labels)
    N = len(features) if selected is None else len(selected)
    if len(labels) != N:
        raise ValueError("Silhouette feature selection and label counts differ")
    rng = np.random.default_rng(random_state)

    if N <= max_samples:
        sample_indices = np.arange(N)
    elif sampling_strategy == "stratified":
        # Stratified sampling: at least 2 samples per eligible class
        unique_classes = np.unique(labels)
        sample_indices_list: List[int] = []
        n_per_class = max(2, max_samples // len(unique_classes))

        for c in unique_classes:
            idx = np.where(labels == c)[0]
            if len(idx) >= 2:
                chosen = rng.choice(idx, size=min(len(idx), n_per_class), replace=False)
                sample_indices_list.extend(chosen)
            else:
                sample_indices_list.extend(idx)
        sample_indices = np.array(sample_indices_list[:max_samples], dtype=np.intp)
    else:
        # Uniform random subsampling
        sample_indices = rng.choice(N, size=max_samples, replace=False)

    selected_sample = sample_indices if selected is None else selected[sample_indices]
    sub_features = np.asarray(features[selected_sample], dtype=np.float32)
    sub_labels = labels[sample_indices]

    # Audit classes
    all_classes, sample_counts = np.unique(sub_labels, return_counts=True)
    valid_class_mask = sample_counts >= 2
    valid_classes = set(all_classes[valid_class_mask])
    excluded_classes = set(all_classes[~valid_class_mask])

    mask = np.isin(sub_labels, list(valid_classes))
    retained_sample_count = int(np.sum(mask))
    excluded_sample_count = len(sub_labels) - retained_sample_count
    retained_sample_rate = retained_sample_count / len(sub_labels) if len(sub_labels) > 0 else 0.0

    if len(valid_classes) >= 2 and retained_sample_count > len(valid_classes):
        global_sil, sample_sil = compute_cosine_silhouette_fast(sub_features[mask], sub_labels[mask])

        df_sil = pd.DataFrame({"class_label": sub_labels[mask], "silhouette": sample_sil})
        per_cluster = (
            df_sil.groupby("class_label")
            .agg(
                silhouette_score=("silhouette", "mean"),
                sample_count=("silhouette", "count"),
            )
            .reset_index()
        )
        per_cluster["taxonomic_rank"] = taxonomic_rank
    else:
        global_sil = float("nan")
        per_cluster = pd.DataFrame(columns=["class_label", "silhouette_score", "sample_count", "taxonomic_rank"])

    return {
        "taxonomic_rank": taxonomic_rank,
        "global_silhouette_score": global_sil,
        "valid_class_rate": len(valid_classes) / len(all_classes) if len(all_classes) else 0.0,
        "sampling_strategy": sampling_strategy,
        "target_sample_size": max_samples,
        "actual_sample_size": len(sub_labels),
        "total_classes_sampled": len(all_classes),
        "valid_classes_count": len(valid_classes),
        "excluded_classes_count": len(excluded_classes),
        "retained_sample_count": retained_sample_count,
        "excluded_sample_count": excluded_sample_count,
        "retained_sample_rate": retained_sample_rate,
        "per_cluster_metrics": per_cluster,
    }


# ---------------------------------------------------------------------------
# S4: Hierarchical Taxonomic Correspondence and Correlation
# ---------------------------------------------------------------------------

def calculate_taxonomic_correlations(
    child_df: pd.DataFrame,
    parent_df: pd.DataFrame,
    taxonomy_df: pd.DataFrame,
    child_tax_col: str = "species",
    parent_tax_col: str = "Genus",
    child_level: str = "Species",
    parent_level: str = "Genus",
    min_children: int = 3,
) -> Dict[str, Any]:
    """
    Calculates parent-child correlation for KNN Accuracy and Silhouette Score.
    Maintains independent valid parent subsets (N_parents_KNN and N_parents_Sil).
    """
    mapping = (
        taxonomy_df[[child_tax_col, parent_tax_col]]
        .drop_duplicates()
        .dropna()
        .rename(columns={child_tax_col: "child_name", parent_tax_col: "parent_name"})
    )

    # Standardize child and parent columns
    child = child_df.copy()
    if "Taxon_Name" in child.columns:
        child = child.rename(columns={"Taxon_Name": "child_name"})
    elif "class_label" in child.columns:
        child = child.rename(columns={"class_label": "child_name"})

    parent = parent_df.copy()
    if "Taxon_Name" in parent.columns:
        parent = parent.rename(columns={"Taxon_Name": "parent_name"})
    elif "class_label" in parent.columns:
        parent = parent.rename(columns={"class_label": "parent_name"})

    # Rename metric columns to standard keys
    if "KNN_Accuracy" in child.columns:
        child = child.rename(columns={"KNN_Accuracy": "knn_accuracy"})
    if "Silhouette_Score" in child.columns:
        child = child.rename(columns={"Silhouette_Score": "silhouette_score"})
    if "KNN_Accuracy" in parent.columns:
        parent = parent.rename(columns={"KNN_Accuracy": "parent_knn"})
    elif "knn_accuracy" in parent.columns:
        parent = parent.rename(columns={"knn_accuracy": "parent_knn"})
    if "Silhouette_Score" in parent.columns:
        parent = parent.rename(columns={"Silhouette_Score": "parent_sil"})
    elif "silhouette_score" in parent.columns:
        parent = parent.rename(columns={"silhouette_score": "parent_sil"})

    merged_child = child.merge(mapping, on="child_name", how="inner")
    merged = merged_child.merge(parent, on="parent_name", how="inner")

    # 1. KNN correlation: independent valid subset
    if "knn_accuracy" in merged.columns and "parent_knn" in merged.columns:
        knn_data = merged.dropna(subset=["knn_accuracy", "parent_knn"])
        knn_summary = (
            knn_data.groupby("parent_name")
            .agg(
                child_knn_mean=("knn_accuracy", "mean"),
                parent_knn=("parent_knn", "first"),
                n_children=("child_name", "nunique"),
            )
            .reset_index()
        )
        knn_valid = knn_summary[knn_summary["n_children"] >= min_children]

        if len(knn_valid) >= 3:
            r_knn, p_r_knn = compute_pearson_r(knn_valid["child_knn_mean"].values, knn_valid["parent_knn"].values)
            rho_knn, p_rho_knn = compute_spearman_rho(knn_valid["child_knn_mean"].values, knn_valid["parent_knn"].values)
        else:
            r_knn = p_r_knn = rho_knn = p_rho_knn = float("nan")
    else:
        knn_valid = pd.DataFrame(columns=["parent_name", "child_knn_mean", "parent_knn", "n_children"])
        r_knn = p_r_knn = rho_knn = p_rho_knn = float("nan")

    # 2. Silhouette correlation: independent valid subset
    if "silhouette_score" in merged.columns and "parent_sil" in merged.columns:
        sil_data = merged.dropna(subset=["silhouette_score", "parent_sil"])
        sil_summary = (
            sil_data.groupby("parent_name")
            .agg(
                child_sil_mean=("silhouette_score", "mean"),
                parent_sil=("parent_sil", "first"),
                n_children=("child_name", "nunique"),
            )
            .reset_index()
        )
        sil_valid = sil_summary[sil_summary["n_children"] >= min_children]

        if len(sil_valid) >= 3:
            r_sil, p_r_sil = compute_pearson_r(sil_valid["child_sil_mean"].values, sil_valid["parent_sil"].values)
            rho_sil, p_rho_sil = compute_spearman_rho(sil_valid["child_sil_mean"].values, sil_valid["parent_sil"].values)
        else:
            r_sil = p_r_sil = rho_sil = p_rho_sil = float("nan")
    else:
        sil_valid = pd.DataFrame(columns=["parent_name", "child_sil_mean", "parent_sil", "n_children"])
        r_sil = p_r_sil = rho_sil = p_rho_sil = float("nan")

    return {
        "child_level": child_level,
        "parent_level": parent_level,
        "n_parents_knn": len(knn_valid),
        "n_parents_silhouette": len(sil_valid),
        "pearson_r_knn": r_knn,
        "pearson_p_knn": p_r_knn,
        "spearman_rho_knn": rho_knn,
        "spearman_p_knn": p_rho_knn,
        "pearson_r_silhouette": r_sil,
        "pearson_p_silhouette": p_r_sil,
        "spearman_rho_silhouette": rho_sil,
        "spearman_p_silhouette": p_rho_sil,
        "knn_summary_df": knn_valid,
        "silhouette_summary_df": sil_valid,
    }


# ---------------------------------------------------------------------------
# S5: ConvNeXt Head Bottleneck & Independent Linear Probe
# ---------------------------------------------------------------------------

class LinearProbeClassifier(nn.Module):
    """
    Linear probing classification head trained strictly on frozen representations
    from training or validation partitions to evaluate true linear separability.
    """

    def __init__(self, in_features: int, num_classes: int) -> None:
        super().__init__()
        self.fc = nn.Linear(in_features, num_classes)
        self._audit_state = "UNFITTED"
        self._sealed_sha256: Optional[str] = None
        self._test_evaluated = False
        self._selection_history: List[Dict[str, float]] = []
        self._training_partition_hashes: set = set()
        self._validation_partition_hashes: set = set()

    @staticmethod
    def _state_hash(module: nn.Module) -> str:
        digest = hashlib.sha256()
        for name, tensor in sorted(module.state_dict().items()):
            digest.update(name.encode())
            digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
        return digest.hexdigest()

    def fit_select_and_seal(
        self,
        train: EmbeddingArtifact,
        validation: EmbeddingArtifact,
        *,
        epochs: int = 10,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        batch_size: int = 256,
        seed: int = 42,
        device: str = "cpu",
    ) -> Dict[str, Any]:
        """Fit only on verified train rows, select by validation accuracy, then freeze the protocol."""
        if self._audit_state != "UNFITTED":
            raise RuntimeError(f"Probe cannot be fitted in state {self._audit_state}")
        if batch_size < 1:
            raise ValueError("Training and validation batch_size must be a positive integer")
        train.validate()
        validation.validate()
        if train.manifest["split"] != "train" or validation.manifest["split"] != "validation":
            raise ValueError("Audited probe requires train and validation embedding artifacts")
        if train.manifest.get("variant") != validation.manifest.get("variant"):
            raise ValueError("Train and validation artifacts belong to different model variants")
        train_hashes = set(train.manifest.get("partition_sha256", []))
        validation_hashes = set(validation.manifest.get("partition_sha256", []))
        if not train_hashes or not validation_hashes or train_hashes.intersection(validation_hashes):
            raise ValueError("Train and validation partition identities are missing or overlap")
        if train.features.shape[1] != self.fc.in_features or validation.features.shape[1] != self.fc.in_features:
            raise ValueError("Embedding dimensions do not match the probe input dimension")
        if np.any(train.labels < 0) or np.any(train.labels >= self.fc.out_features):
            raise ValueError("Train labels are outside the probe class range")
        if np.any(validation.labels < 0) or np.any(validation.labels >= self.fc.out_features):
            raise ValueError("Validation labels are outside the probe class range")
        torch.manual_seed(seed)
        dev = torch.device(device)
        self.to(dev)
        X = torch.as_tensor(train.features, dtype=torch.float32)
        y = torch.as_tensor(train.labels, dtype=torch.long)
        loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X, y), batch_size=batch_size, shuffle=True)
        optimizer = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=weight_decay)
        criterion = nn.CrossEntropyLoss()
        best_accuracy = -1.0
        best_state = None
        y_val = torch.as_tensor(validation.labels, dtype=torch.long, device=dev)
        for epoch in range(epochs):
            self.train()
            for bx, by in loader:
                bx, by = bx.to(dev), by.to(dev)
                optimizer.zero_grad(set_to_none=True)
                criterion(self(bx), by).backward()
                optimizer.step()
            self.eval()
            correct_count = torch.zeros((), dtype=torch.int64, device=dev)
            with torch.inference_mode():
                for start in range(0, len(validation.features), batch_size):
                    stop = min(start + batch_size, len(validation.features))
                    batch_x = torch.as_tensor(validation.features[start:stop], dtype=torch.float32, device=dev)
                    batch_pred = self(batch_x).argmax(dim=1)
                    correct_count += (batch_pred == y_val[start:stop]).sum()
            accuracy = int(correct_count.cpu()) / len(y_val) if len(y_val) else 0.0
            self._selection_history.append({"epoch": epoch + 1, "validation_accuracy": accuracy})
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_state = {name: value.detach().cpu().clone() for name, value in self.state_dict().items()}
        if best_state is None:
            raise ValueError("Validation artifact is empty; cannot select probe weights")
        self.load_state_dict(best_state)
        self._training_partition_hashes = train_hashes
        self._validation_partition_hashes = validation_hashes
        self.eval()
        self._audit_state = "SEALED"
        self._sealed_sha256 = self._state_hash(self)
        return {
            "probe_protocol": "c01_train_val_frozen_v1",
            "state": self._audit_state,
            "selected_epoch": int(max(self._selection_history, key=lambda item: item["validation_accuracy"])['epoch']),
            "selection_rule": "maximum validation Accuracy; earliest epoch wins ties",
            "selection_history": list(self._selection_history),
            "weights_sha256": self._sealed_sha256,
            "train_manifest": dict(train.manifest),
            "validation_manifest": dict(validation.manifest),
        }

    def evaluate_artifact(
        self,
        test: EmbeddingArtifact,
        device: str = "cpu",
        *,
        batch_size: int = 256,
        include_row_ids: bool = True,
    ) -> Dict[str, Any]:
        """Evaluate once on a verified test artifact after the selected weights are sealed."""
        if self._test_evaluated:
            raise RuntimeError("Sealed probe test evaluation may occur only once")
        if self._audit_state != "SEALED" or self._sealed_sha256 is None:
            raise RuntimeError("Probe must be selected and sealed before test evaluation")
        test.validate()
        if test.manifest["split"] != "test":
            raise ValueError("Final probe evaluation requires a test embedding artifact")
        test_hashes = set(test.manifest.get("partition_sha256", []))
        if (not test_hashes or test_hashes.intersection(self._training_partition_hashes)
                or test_hashes.intersection(self._validation_partition_hashes)):
            raise ValueError("Test partition identities overlap train/validation or are missing")
        if self._state_hash(self) != self._sealed_sha256:
            raise RuntimeError("Sealed probe weights changed before test evaluation")
        result = self.evaluate(test.features, test.labels, device=device, batch_size=batch_size)
        identity_hash = hashlib.sha256()
        for start in range(0, len(test.row_ids), 65536):
            identity_hash.update(np.asarray(test.row_ids[start:start + 65536]).tobytes())
        result.update({"row_ids_sha256": identity_hash.hexdigest(), "weights_sha256": self._sealed_sha256,
                       "supported_classes": int(np.unique(test.labels).size), "probe_protocol": "c01_train_val_frozen_v1"})
        if include_row_ids:
            result["row_ids"] = test.row_ids.tolist()
        self._test_evaluated = True
        self._audit_state = "TEST_EVALUATED"
        return result

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        split_origin: str = "train",
        epochs: int = 10,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        batch_size: int = 256,
        device: str = "cpu",
    ) -> "LinearProbeClassifier":
        """
        Trains linear probe exclusively on training or validation split embeddings.
        Guarantees that test embeddings are never accessed.
        """
        if split_origin.lower().strip() == "test":
            raise ValueError(
                "Violation of scientific invariant: LinearProbeClassifier cannot be trained on test set embeddings"
            )
        if self._audit_state != "UNFITTED":
            raise RuntimeError(f"Probe cannot be fitted in state {self._audit_state}")

        dev = torch.device(device)
        self.to(dev)
        self.train()

        X_t = torch.as_tensor(X_train, dtype=torch.float32)
        y_t = torch.as_tensor(y_train, dtype=torch.long)

        dataset = torch.utils.data.TensorDataset(X_t, y_t)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        optimizer = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=weight_decay)
        criterion = nn.CrossEntropyLoss()

        for _ in range(epochs):
            for bx, by in loader:
                bx = bx.to(dev)
                by = by.to(dev)
                optimizer.zero_grad()
                logits = self(bx)
                loss = criterion(logits, by)
                loss.backward()
                optimizer.step()

        self.eval()
        return self

    def evaluate(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        device: str = "cpu",
        *,
        batch_size: int = 256,
    ) -> Dict[str, float]:
        """Evaluate test embeddings in bounded batches with global class counts."""
        if batch_size < 1:
            raise ValueError("Evaluation batch_size must be a positive integer")
        self.eval()
        dev = torch.device(device)
        self.to(dev)

        y_true = np.asarray(y_test, dtype=np.int64)
        if len(X_test) != len(y_true):
            raise ValueError("Test feature and label counts differ")
        num_classes = self.fc.out_features
        if np.any(y_true < 0) or np.any(y_true >= num_classes):
            raise ValueError("Test labels are outside the probe class range")
        tp = torch.zeros(num_classes, dtype=torch.int64, device=dev)
        pred_counts = torch.zeros(num_classes, dtype=torch.int64, device=dev)
        true_counts = torch.zeros(num_classes, dtype=torch.int64, device=dev)
        correct_count = torch.zeros((), dtype=torch.int64, device=dev)
        with torch.no_grad():
            for start in range(0, len(y_true), batch_size):
                stop = min(start + batch_size, len(y_true))
                bx = torch.as_tensor(X_test[start:stop], dtype=torch.float32, device=dev)
                by = torch.as_tensor(y_true[start:stop], dtype=torch.long, device=dev)
                pred = self(bx).argmax(dim=1)
                correct = pred == by
                correct_count += correct.sum()
                true_counts += torch.bincount(by, minlength=num_classes)
                pred_counts += torch.bincount(pred, minlength=num_classes)
                tp += torch.bincount(by[correct], minlength=num_classes)
        tp_np = tp.cpu().numpy()
        pred_counts_np = pred_counts.cpu().numpy()
        true_counts_np = true_counts.cpu().numpy()
        denom = pred_counts_np + true_counts_np
        f1_per_class = np.zeros(num_classes, dtype=np.float64)
        valid = denom > 0
        f1_per_class[valid] = 2.0 * tp_np[valid] / denom[valid]
        macro_f1 = float(np.mean(f1_per_class[true_counts_np > 0])) if np.any(true_counts_np > 0) else 0.0
        accuracy = int(correct_count.cpu()) / len(y_true) if len(y_true) else 0.0

        return {
            "top1_accuracy": accuracy,
            "macro_f1": macro_f1,
            "n_test_samples": len(y_true),
        }


def audit_classifier_bottleneck(
    native_model_accuracy: float,
    linear_probe_accuracy: float,
    transductive_knn_accuracy: float,
) -> Dict[str, Any]:
    """
    Constructs a comparative diagnostic table resolving the ConvNeXt head bottleneck hypothesis.
    """
    statement = (
        f"Transductive test 5-NN accuracy ({transductive_knn_accuracy:.2%}) utilizes test-set reference "
        "representations and non-parametric local metric lookup. It does NOT demonstrate undercapacity of the "
        f"parametric classifier head. The true linear separability of the frozen representations evaluated via an "
        f"independent linear probe trained on train/val data is {linear_probe_accuracy:.2%}, compared to the native "
        f"classifier head's accuracy of {native_model_accuracy:.2%}."
    )

    summary_df = pd.DataFrame(
        [
            {"evaluation_protocol": "Native Trained Classifier Head", "accuracy": native_model_accuracy, "training_source": "Training set (parametric)"},
            {"evaluation_protocol": "Independent Linear Probe", "accuracy": linear_probe_accuracy, "training_source": "Train/Val embeddings (parametric)"},
            {"evaluation_protocol": "Transductive Test 5-NN", "accuracy": transductive_knn_accuracy, "training_source": "80% test reference (non-parametric)"},
        ]
    )

    return {
        "diagnostic_statement": statement,
        "summary_table": summary_df,
        "proves_head_bottleneck": False,
    }


# ---------------------------------------------------------------------------
# S6: UMAP Qualitative Projection
# ---------------------------------------------------------------------------

def export_umap_projection(
    features: np.ndarray,
    labels: np.ndarray,
    output_path: Optional[Union[str, Path]] = None,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    random_state: int = 42,
    metric: str = "cosine",
) -> pd.DataFrame:
    """
    Computes 2D UMAP projection for qualitative visualization only.
    Exports DataFrame with explicit qualitative disclaimer.
    """
    try:
        from umap import UMAP

        reducer = UMAP(
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            n_components=2,
            metric=metric,
            random_state=random_state,
        )
        embedding = reducer.fit_transform(features)
    except ImportError:
        # Graceful fallback: 2D PCA projection if UMAP is not installed in environment
        centered = features - np.mean(features, axis=0)
        u, s, vt = np.linalg.svd(centered, full_matrices=False)
        embedding = centered @ vt[:2].T

    df_umap = pd.DataFrame(
        {
            "umap_1": embedding[:, 0],
            "umap_2": embedding[:, 1],
            "label": labels,
            "is_qualitative_only": True,
        }
    )

    if output_path is not None:
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        # Include disclaimer header if saving as CSV
        if p.suffix.lower() == ".csv":
            with open(p, "w", encoding="utf-8") as f:
                f.write("# QUALITATIVE ONLY: 2D UMAP projection is strictly for exploratory visualization.\n")
                f.write("# Do not infer quantitative cluster separation or phylogenetic distance.\n")
                df_umap.to_csv(f, index=False)
        elif p.suffix.lower() in (".parquet", ".pq"):
            df_umap.to_parquet(p, index=False)

    return df_umap
