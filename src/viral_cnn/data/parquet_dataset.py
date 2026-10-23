"""
Parquet batch iterable dataset for large-scale multi-partition genomic sequence loading.

Scientific and engineering invariants (AGENTS.md, AD-003, AD-006):
- Strictly yields pre-batched tensors (batch_size, maxlen) compatible with DataLoader(..., batch_size=None)
- 4-channel nucleotide tokenization: A=0, C=1, G=2, T=3; padding/ambiguous=4
- Deterministic query ordering when shuffle=False (critical for paired evaluations with N=1,706,270)
- Disjoint partition splitting across PyTorch worker processes via get_worker_info()
- Support for CPU (pyarrow/pandas) and GPU (cudf) backends
"""

from __future__ import annotations

import logging
import math
import random
from typing import Iterator, List, Optional, Sequence, Tuple

import pandas as pd
import pyarrow.parquet as pq
import torch
from torch.utils.data import IterableDataset, get_worker_info

from viral_cnn.data.encoding import NUCLEOTIDES, NUCLEOTIDE_MAPPING, NUM_FEATURES

logger = logging.getLogger(__name__)


class ParquetBatchIterableDataset(IterableDataset):
    """
    High-performance IterableDataset for loading genomic sequences from Parquet partitions.

    Supports both single-worker and multi-worker loading with disjoint file distribution,
    transparent column name fallback ('sequence_padded', 'padded_Representation', 'sequence'),
    fast row counting via PyArrow metadata, and deterministic vs reproducible shuffle modes.
    """

    def __init__(
        self,
        file_paths: Sequence[str],
        padded_column_name: str = "sequence_padded",
        label_column: str = "label",
        maxlen: int = 512,
        nucleotides: Sequence[str] = NUCLEOTIDES,
        batch_size: int = 1024,
        backend: str = "cpu",
        shuffle: bool = False,
        seed: int = 42,
        legacy_global_rng: bool = False,
        fallback_to_cpu: bool = True,
        debug: bool = False,
    ):
        super().__init__()
        self.file_paths = [str(p) for p in file_paths]
        self.padded_column_name = padded_column_name
        self.label_column = label_column
        self.maxlen = maxlen
        self.nucleotides = list(nucleotides)
        self.batch_size = batch_size
        self.backend = backend.lower().strip()
        self.shuffle = shuffle
        self.seed = seed
        self.legacy_global_rng = legacy_global_rng
        self.fallback_to_cpu = fallback_to_cpu
        self.debug = debug

        self.mapping = {n: i for i, n in enumerate(self.nucleotides)}
        self.num_features = len(self.nucleotides)

        # Fast CPU metadata read
        self.total_samples = self._calculate_total_samples_fast()

    def _calculate_total_samples_fast(self) -> int:
        """Reads only Parquet metadata footers across file partitions."""
        count = 0
        try:
            for f in self.file_paths:
                count += pq.read_metadata(f).num_rows
            return count
        except Exception as e:
            if self.debug:
                logger.warning(f"Failed to read parquet metadata via PyArrow: {e}")
            return 0

    def __len__(self) -> int:
        """Returns the total number of batches."""
        if not self.total_samples or self.total_samples <= 0:
            return 1
        return math.ceil(self.total_samples / self.batch_size)

    def _resolve_sequence_column(self, available_columns: Sequence[str]) -> str:
        """Resolves sequence column name with transparent fallback."""
        candidates = [self.padded_column_name, "sequence_padded", "padded_Representation", "sequence"]
        for c in candidates:
            if c in available_columns:
                return c
        raise KeyError(
            f"None of the candidate sequence columns {candidates} found in partition columns: {list(available_columns)}"
        )

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        worker_info = get_worker_info()

        if worker_info is None:
            worker_id = 0
            my_files = list(self.file_paths)
        else:
            worker_id = worker_info.id
            per_worker = int(math.ceil(len(self.file_paths) / float(worker_info.num_workers)))
            iter_start = worker_id * per_worker
            iter_end = min(iter_start + per_worker, len(self.file_paths))
            my_files = list(self.file_paths[iter_start:iter_end])

        if self.shuffle:
            if self.legacy_global_rng:
                # Historical behaviour in short/utils/dna_dataset_2.py
                random.shuffle(my_files)
            else:
                worker_rng = random.Random(self.seed + worker_id)
                worker_rng.shuffle(my_files)

        for file_path in my_files:
            if self.backend == "cudf":
                try:
                    import cudf
                    use_cudf = True
                except ImportError as e:
                    if not self.fallback_to_cpu:
                        raise ImportError(f"cuDF backend requested but cudf is not installed: {e}") from e
                    logger.warning("cuDF not available; falling back to CPU pyarrow backend.")
                    use_cudf = False
            else:
                use_cudf = False

            if use_cudf:
                yield from self._iter_cudf(file_path)
            else:
                yield from self._iter_cpu(file_path)

    def _iter_cpu(self, file_path: str) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        try:
            # Inspect columns to handle fallback
            parquet_schema = pq.read_schema(file_path)
            seq_col = self._resolve_sequence_column(parquet_schema.names)
            cols_to_load = [seq_col]
            if self.label_column in parquet_schema.names:
                cols_to_load.append(self.label_column)

            partition_df = pd.read_parquet(file_path, engine="pyarrow", columns=cols_to_load)
        except Exception as e:
            logger.error(f"Failed to read parquet partition {file_path}: {e}")
            return

        num_rows = len(partition_df)
        has_label = self.label_column in partition_df.columns

        for i in range(0, num_rows, self.batch_size):
            end_index = min(i + self.batch_size, num_rows)
            batch_df = partition_df.iloc[i:end_index]

            inputs_list = batch_df[seq_col].tolist()
            if has_label:
                labels_raw = batch_df[self.label_column].to_numpy()
                labels_tensor = torch.as_tensor(labels_raw, dtype=torch.long)
            else:
                labels_tensor = torch.zeros(len(inputs_list), dtype=torch.long)

            # Tokenize sequence to integer matrix [0..4]
            tokenized_list = [
                [self.mapping.get(base, self.num_features) for base in seq[: self.maxlen]]
                + [self.num_features] * max(0, self.maxlen - len(seq))
                for seq in inputs_list
            ]
            inputs_tensor = torch.tensor(tokenized_list, dtype=torch.long)

            yield inputs_tensor, labels_tensor

    def _iter_cudf(self, file_path: str) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        try:
            import cudf
            partition_df = cudf.read_parquet(file_path)
            if not hasattr(partition_df, "columns") or len(partition_df.columns) == 0:
                raise ImportError("Mocked or invalid cudf module")
        except Exception as e:
            if not self.fallback_to_cpu:
                raise ImportError(f"cuDF failed to read parquet partition {file_path}: {e}") from e
            logger.warning(f"cuDF read failed ({e}); falling back to CPU pyarrow backend.")
            yield from self._iter_cpu(file_path)
            return

        seq_col = self._resolve_sequence_column(partition_df.columns)
        has_label = self.label_column in partition_df.columns
        num_rows = len(partition_df)

        for i in range(0, num_rows, self.batch_size):
            end_index = min(i + self.batch_size, num_rows)
            batch_df = partition_df.iloc[i:end_index]

            inputs_list = batch_df[seq_col].to_arrow().to_pylist()
            if has_label:
                labels_raw = batch_df[self.label_column].to_numpy()
                labels_tensor = torch.as_tensor(labels_raw, dtype=torch.long)
            else:
                labels_tensor = torch.zeros(len(inputs_list), dtype=torch.long)

            tokenized_list = [
                [self.mapping.get(base, self.num_features) for base in seq[: self.maxlen]]
                + [self.num_features] * max(0, self.maxlen - len(seq))
                for seq in inputs_list
            ]
            inputs_tensor = torch.tensor(tokenized_list, dtype=torch.long)

            yield inputs_tensor, labels_tensor

