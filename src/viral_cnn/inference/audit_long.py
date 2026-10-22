"""
Audit utilities for long-sequence inference, logit equivalence, and training overlap.
"""

from typing import Any, Optional
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from .engine import predict_sequence
from .voter import LongSequenceVoter

DNA_COMPLEMENT = str.maketrans("ACGTacgt", "TGCAtgca")


def reverse_complement(seq: str) -> str:
    """Returns the reverse complement of a DNA sequence."""
    return seq.translate(DNA_COMPLEMENT)[::-1]


def canonical_kmer(kmer: str) -> str:
    """Returns lexicographically smallest between forward and reverse complement kmer."""
    rc = reverse_complement(kmer)
    return min(kmer, rc)


def verify_window_logit_equivalence(
    model: nn.Module,
    sequence_512: str,
    device: str = "cpu",
) -> float:
    """
    Verifies that a 512 bp window produces numerically identical logits
    when processed via the single-fragment engine vs the LongSequenceVoter.
    Returns the maximum absolute difference between logit vectors.
    """
    if len(sequence_512) != 512:
        raise ValueError(f"Expected sequence of exactly 512 bp, got {len(sequence_512)}")

    # 1. Single-fragment engine inference
    single_res = predict_sequence(model, sequence_512, device=device, max_len=512)
    single_logits = torch.tensor(single_res.logits, dtype=torch.float32)

    # 2. Voter sliding-window inference with window=512, stride=512
    voter = LongSequenceVoter(
        model=model,
        window_size=512,
        stride=512,
        handle_short="pad",
        device=device,
    )
    voter_res = voter.predict_sequence(sequence_512)
    if voter_res is None:
        raise RuntimeError("Voter returned None for 512 bp sequence")

    # To get raw logits directly from voter pass:
    tokens = torch.tensor(
        voter.model.one_hot_layer.num_features if hasattr(voter.model, "one_hot_layer") else 4
    )
    # Forward pass of voter's exact window
    from .voter import slice_sequence

    windows = slice_sequence(sequence_512, window_size=512, stride=512, handle_short="pad")
    tokens_tensor = torch.tensor(windows, dtype=torch.long, device=device)
    with torch.no_grad():
        voter_logits = voter.model(tokens_tensor).cpu().squeeze(0)

    max_abs_diff = torch.max(torch.abs(single_logits - voter_logits)).item()
    return float(max_abs_diff)


def quantify_training_overlap(
    test_sequences: list[str],
    train_sequences: list[str],
    k: int = 15,
) -> dict[str, float]:
    """
    Quantifies exact sequence and canonical k-mer overlap between test sequences
    and training reference fragments.

    Returns:
    - exact_match_fraction: fraction of test sequences having identical (or RC) match in training.
    - mean_kmer_overlap_fraction: average fraction of test k-mers present in training k-mer set.
    - fraction_with_any_overlap: fraction of test sequences sharing at least 1 canonical k-mer.
    """
    if not test_sequences or not train_sequences:
        return {
            "exact_match_fraction": 0.0,
            "mean_kmer_overlap_fraction": 0.0,
            "fraction_with_any_overlap": 0.0,
        }

    # Build canonical training set
    train_exact = set()
    train_kmers = set()

    for seq in train_sequences:
        seq_clean = seq.upper().strip()
        train_exact.add(seq_clean)
        train_exact.add(reverse_complement(seq_clean))

        if len(seq_clean) >= k:
            for i in range(len(seq_clean) - k + 1):
                kmer = seq_clean[i : i + k]
                if all(c in "ACGT" for c in kmer):
                    train_kmers.add(canonical_kmer(kmer))

    exact_matches = 0
    kmer_overlaps = []
    any_overlap_count = 0

    for test_seq in test_sequences:
        test_clean = test_seq.upper().strip()

        # Check exact sequence match
        if test_clean in train_exact or reverse_complement(test_clean) in train_exact:
            exact_matches += 1

        # Check k-mer overlap
        test_seq_kmers = set()
        if len(test_clean) >= k:
            for i in range(len(test_clean) - k + 1):
                kmer = test_clean[i : i + k]
                if all(c in "ACGT" for c in kmer):
                    test_seq_kmers.add(canonical_kmer(kmer))

        if test_seq_kmers:
            shared = test_seq_kmers.intersection(train_kmers)
            overlap_frac = len(shared) / len(test_seq_kmers)
            kmer_overlaps.append(overlap_frac)
            if len(shared) > 0:
                any_overlap_count += 1
        else:
            kmer_overlaps.append(0.0)

    n_test = len(test_sequences)
    return {
        "exact_match_fraction": float(exact_matches / n_test),
        "mean_kmer_overlap_fraction": float(np.mean(kmer_overlaps) if kmer_overlaps else 0.0),
        "fraction_with_any_overlap": float(any_overlap_count / n_test),
    }


def stratify_long_errors(
    df_results: pd.DataFrame,
    length_bins: Optional[list[tuple[int, int, str]]] = None,
    ambiguity_bins: Optional[list[tuple[float, float, str]]] = None,
) -> dict[str, Any]:
    """
    Stratifies prediction performance across sequence length bins and ambiguity fractions.

    Expects df_results to contain columns:
    - 'length': integer sequence length
    - 'ambiguity_ratio': fraction of non-ACGT characters
    - 'truth_label': integer true class (1-based or 0-based)
    - 'pred_mean_logits': integer predicted class
    - 'pred_majority_vote': integer predicted class
    """
    if length_bins is None:
        length_bins = [
            (512, 2000, "len_512_to_2000"),
            (2000, 5000, "len_2000_to_5000"),
            (5000, 10001, "len_5000_to_10000"),
        ]

    if ambiguity_bins is None:
        ambiguity_bins = [
            (0.0, 0.0, "ambig_0_pct"),
            (1e-6, 0.05, "ambig_0_to_5_pct"),
            (0.05 + 1e-6, 1.0, "ambig_gt_5_pct"),
        ]

    # Calculate correctness flags
    mean_correct = df_results["truth_label"] == df_results["pred_mean_logits"]
    majority_correct = df_results["truth_label"] == df_results["pred_majority_vote"]

    strata: dict[str, Any] = {
        "overall": {
            "total_samples": int(len(df_results)),
            "accuracy_mean_logits": float(mean_correct.mean()) if len(df_results) else 0.0,
            "accuracy_majority_vote": float(majority_correct.mean()) if len(df_results) else 0.0,
        },
        "by_length": {},
        "by_ambiguity": {},
    }

    # Stratify by length
    for low, high, label in length_bins:
        mask = (df_results["length"] >= low) & (df_results["length"] < high)
        subset = df_results[mask]
        count = int(len(subset))
        acc_mean = float((subset["truth_label"] == subset["pred_mean_logits"]).mean()) if count > 0 else 0.0
        acc_maj = float((subset["truth_label"] == subset["pred_majority_vote"]).mean()) if count > 0 else 0.0
        strata["by_length"][label] = {
            "range": [low, high],
            "count": count,
            "accuracy_mean_logits": acc_mean,
            "accuracy_majority_vote": acc_maj,
        }

    # Stratify by ambiguity
    for low, high, label in ambiguity_bins:
        if low == high == 0.0:
            mask = df_results["ambiguity_ratio"] == 0.0
        else:
            mask = (df_results["ambiguity_ratio"] >= low) & (df_results["ambiguity_ratio"] <= high)
        subset = df_results[mask]
        count = int(len(subset))
        acc_mean = float((subset["truth_label"] == subset["pred_mean_logits"]).mean()) if count > 0 else 0.0
        acc_maj = float((subset["truth_label"] == subset["pred_majority_vote"]).mean()) if count > 0 else 0.0
        strata["by_ambiguity"][label] = {
            "range": [low, high],
            "count": count,
            "accuracy_mean_logits": acc_mean,
            "accuracy_majority_vote": acc_maj,
        }

    return strata

