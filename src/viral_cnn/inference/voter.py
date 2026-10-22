"""
Sliding-window consensus voter for long-sequence taxonomic inference.

Faithfully implements the historical Voter protocol evaluated in manuscript Section 3.5
(Table 12) from `long/cnn-long-voter.ipynb`, with additions for robust contig handling,
deterministic tie-breaking, and class-dimension validation.
"""

from dataclasses import dataclass
from typing import Literal, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

# Canonical nucleotide to token mapping
DNA_MAP = {"A": 0, "C": 1, "G": 2, "T": 3}
NUM_FEATURES = 4
DEFAULT_WINDOW_SIZE = 512
DEFAULT_STRIDE = 256
DEFAULT_BATCH_SIZE = 2048


@dataclass
class LongConsensusResult:
    """
    Structured outcome of sliding-window consensus over a long genomic sequence.
    """
    pred_mean_logits: int
    pred_majority_vote: int
    num_windows: int
    top5_mean_indices: list[int]
    top5_mean_probs: list[float]
    window_top5_indices: list[list[int]]
    window_top5_probs: list[list[float]]


def slice_sequence(
    sequence: str,
    window_size: int = DEFAULT_WINDOW_SIZE,
    stride: int = DEFAULT_STRIDE,
    handle_short: Literal["skip", "pad"] = "pad",
) -> list[list[int]]:
    """
    Slices a nucleotide sequence into overlapping token windows of length `window_size`.

    - Uppercase normalization (`.upper()`) ensures case invariance.
    - Canonical nucleotides A, C, G, T map to 0, 1, 2, 3.
    - Ambiguous bases (IUPAC, N), unknown characters, and padding map to token 4 (null vector).
    - If `len(sequence) < window_size`:
        - if `handle_short == 'skip'`: returns [] (matches historical notebook behavior).
        - if `handle_short == 'pad'`: pads right with token 4 up to `window_size` (returns 1 window).
    """
    seq_upper = sequence.upper().strip()
    seq_len = len(seq_upper)

    if seq_len < window_size:
        if handle_short == "skip":
            return []
        elif handle_short == "pad":
            tokens = [DNA_MAP.get(b, NUM_FEATURES) for b in seq_upper]
            tokens.extend([NUM_FEATURES] * (window_size - seq_len))
            return [tokens]
        else:
            raise ValueError(f"Invalid handle_short mode: {handle_short}. Choose 'skip' or 'pad'.")

    windows: list[list[int]] = []
    for start in range(0, seq_len - window_size + 1, stride):
        window_str = seq_upper[start : start + window_size]
        indices = [DNA_MAP.get(base, NUM_FEATURES) for base in window_str]
        windows.append(indices)

    return windows


def aggregate_window_logits(
    logits: torch.Tensor,
    tie_break: Literal["min_index"] = "min_index",
) -> LongConsensusResult:
    """
    Aggregates window-level model logits into sequence-level consensus predictions.

    - `pred_mean_logits`: Average pooling of raw logits before softmax (Table 12 Mean Logits).
    - `pred_majority_vote`: Mode of window-level argmax predictions (Table 12 Majority Vote).
    - `tie_break`: Deterministic policy for majority vote ties.
      `torch.mode` returns the smallest class index among maximum-count ties (`min_index`).
    """
    if logits.ndim != 2 or logits.size(0) == 0:
        raise ValueError(f"Expected 2D logits tensor of shape (N_windows, N_classes), got {logits.shape}")

    num_windows = logits.size(0)
    num_classes = logits.size(1)
    k = min(5, num_classes)

    # 1. Pre-softmax Logit Averaging (Average Pooling)
    mean_logits = logits.mean(dim=0)
    top5_mean_val, top5_mean_idx = torch.topk(F.softmax(mean_logits, dim=-1), k=k)
    pred_mean_logits = mean_logits.argmax().item()

    # 2. Hard Majority Voting (Mode across window winners)
    window_winners = logits.argmax(dim=1)
    # torch.mode returns (values, indices) where values breaks ties by smallest index
    majority_class = torch.mode(window_winners).values.item()

    # 3. Window-level top-5 probabilities and indices
    window_probs = F.softmax(logits, dim=1)
    top5_probs, top5_indices = torch.topk(window_probs, k=k, dim=1)

    return LongConsensusResult(
        pred_mean_logits=pred_mean_logits,
        pred_majority_vote=majority_class,
        num_windows=num_windows,
        top5_mean_indices=top5_mean_idx.tolist(),
        top5_mean_probs=top5_mean_val.tolist(),
        window_top5_indices=top5_indices.cpu().tolist(),
        window_top5_probs=top5_probs.cpu().tolist(),
    )


class LongSequenceVoter:
    """
    Sliding-window voter orchestrator for variable-length genomic sequences.
    """

    def __init__(
        self,
        model: nn.Module,
        window_size: int = DEFAULT_WINDOW_SIZE,
        stride: int = DEFAULT_STRIDE,
        handle_short: Literal["skip", "pad"] = "pad",
        batch_size_inference: int = DEFAULT_BATCH_SIZE,
        device: str = "cpu",
        expected_classes: Optional[int] = None,
    ):
        self.model = model
        self.window_size = window_size
        self.stride = stride
        self.handle_short = handle_short
        self.batch_size_inference = batch_size_inference
        self.device = torch.device(device)

        # Validate classification head dimension if specified
        if expected_classes is not None:
            self._validate_head_classes(expected_classes)

        self.model.eval()
        self.model.to(self.device)

    def _validate_head_classes(self, expected_classes: int) -> None:
        """
        Validates that model classification head matches expected class count.
        """
        head_classes = None
        if hasattr(self.model, "fc") and hasattr(self.model.fc, "out_features"):
            head_classes = self.model.fc.out_features
        elif hasattr(self.model, "head") and hasattr(self.model.head, "out_features"):
            head_classes = self.model.head.out_features

        if head_classes is not None and head_classes != expected_classes:
            raise ValueError(
                f"Model classification head dimension ({head_classes}) does not match "
                f"expected class count ({expected_classes})."
            )

    def predict_sequence(self, sequence: str) -> Optional[LongConsensusResult]:
        """
        Performs sliding-window inference and consensus aggregation for a single sequence.
        Returns None if sequence produces no windows (under handle_short='skip').
        """
        windows = slice_sequence(
            sequence,
            window_size=self.window_size,
            stride=self.stride,
            handle_short=self.handle_short,
        )

        if not windows:
            return None

        # Convert to token tensor of shape (N_windows, window_size)
        batch_tensor = torch.tensor(windows, dtype=torch.long)
        num_windows = batch_tensor.size(0)
        all_logits = []

        with torch.no_grad():
            for i in range(0, num_windows, self.batch_size_inference):
                mini_batch = batch_tensor[i : i + self.batch_size_inference].to(self.device)
                logits = self.model(mini_batch)
                all_logits.append(logits.cpu())

        full_logits = torch.cat(all_logits, dim=0)
        return aggregate_window_logits(full_logits)

