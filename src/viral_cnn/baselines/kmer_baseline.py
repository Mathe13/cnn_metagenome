"""k-mer frequency centroid classifier for sequence taxonomic baseline."""

from __future__ import annotations

import itertools
from typing import Dict, Iterable, List, Optional, Tuple, Union
import numpy as np


class KmerCentroidClassifier:
    """Taxonomic classifier based on k-mer frequency cosine centroids."""

    def __init__(self, k: int = 6):
        self.k = k
        # Map all 4^k canonical k-mers to indices
        bases = ["A", "C", "G", "T"]
        self.kmer_to_idx = {
            "".join(kmer): i for i, kmer in enumerate(itertools.product(bases, repeat=k))
        }
        self.num_kmers = len(self.kmer_to_idx)
        self.centroids: Optional[np.ndarray] = None
        self.classes_: Optional[np.ndarray] = None
        self.class_to_row: Dict[int, int] = {}

    def count_kmers(self, sequence: str) -> np.ndarray:
        """Count k-mers in uppercase sequence, ignoring non-ACGT characters."""
        seq = sequence.strip().upper()
        counts = np.zeros(self.num_kmers, dtype=np.float32)
        k = self.k
        for i in range(len(seq) - k + 1):
            kmer = seq[i : i + k]
            idx = self.kmer_to_idx.get(kmer)
            if idx is not None:
                counts[idx] += 1.0
        norm = np.linalg.norm(counts)
        if norm > 0:
            counts /= norm
        return counts

    def fit(
        self,
        sequences: Iterable[str],
        labels: Iterable[int],
    ) -> "KmerCentroidClassifier":
        """Fit class centroids from training sequences and labels."""
        seq_list = list(sequences)
        lbl_list = list(labels)
        if len(seq_list) != len(lbl_list):
            raise ValueError("Mismatched counts for sequences and labels")

        unique_classes = np.unique(lbl_list)
        self.classes_ = unique_classes
        self.class_to_row = {int(c): i for i, c in enumerate(unique_classes)}
        n_classes = len(unique_classes)

        centroids = np.zeros((n_classes, self.num_kmers), dtype=np.float32)
        counts = np.zeros(n_classes, dtype=np.float32)

        for seq, lbl in zip(seq_list, lbl_list):
            c_idx = self.class_to_row[int(lbl)]
            vec = self.count_kmers(seq)
            centroids[c_idx] += vec
            counts[c_idx] += 1.0

        for i in range(n_classes):
            if counts[i] > 0:
                centroids[i] /= counts[i]
            norm = np.linalg.norm(centroids[i])
            if norm > 0:
                centroids[i] /= norm

        self.centroids = centroids
        return self

    def predict(self, sequences: Iterable[str]) -> np.ndarray:
        """Predict class label for sequences via cosine similarity to centroids."""
        if self.centroids is None or self.classes_ is None:
            raise RuntimeError("Classifier is not fitted")

        predictions = []
        for seq in sequences:
            vec = self.count_kmers(seq)
            if np.all(vec == 0.0):
                predictions.append(-1)
                continue
            sims = self.centroids.dot(vec)
            best_idx = int(np.argmax(sims))
            predictions.append(int(self.classes_[best_idx]))

        return np.asarray(predictions, dtype=np.int64)

