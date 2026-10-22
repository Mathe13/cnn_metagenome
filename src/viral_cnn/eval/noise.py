"""
Controlled noise perturbation engine for genomic sequences.
"""

from typing import Optional
import torch


def inject_noise(
    sequences: torch.Tensor,
    noise_rate: float,
    num_features: int = 4,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """
    Applies Bernoulli position selection with rate epsilon and uniform
    resampling across {0, 1, ..., num_features - 1}.

    Expected effective substitution rate is (1 - 1/num_features) * noise_rate.
    For standard DNA (num_features=4), expected effective rate is 3 * noise_rate / 4.

    Padding tokens and ambiguous bases (values < 0 or >= num_features) are strictly preserved.
    """
    if noise_rate < 0.0 or noise_rate > 1.0:
        raise ValueError(f"noise_rate must be in [0.0, 1.0], got {noise_rate}")

    if noise_rate == 0.0:
        return sequences.clone()

    # Valid nucleotide positions that are eligible for perturbation
    valid_mask = (sequences >= 0) & (sequences < num_features)

    # Bernoulli selection per position
    rand_vals = torch.rand(
        sequences.shape,
        generator=generator,
        device=sequences.device,
    )
    bernoulli_mask = rand_vals < noise_rate

    # Mutate only valid positions selected by Bernoulli mask
    mutation_mask = bernoulli_mask & valid_mask

    # Uniform resampling among canonical bases [0, num_features - 1]
    random_bases = torch.randint(
        0,
        num_features,
        sequences.shape,
        generator=generator,
        device=sequences.device,
        dtype=sequences.dtype,
    )

    # Where mutation applies, insert resampled base; otherwise preserve original
    noisy_sequences = torch.where(mutation_mask, random_bases, sequences)
    return noisy_sequences


def verify_effective_noise_rate(
    original: torch.Tensor,
    noisy: torch.Tensor,
    epsilon: float,
    tolerance: float = 0.005,
) -> float:
    """
    Computes the empirical substitution rate over valid nucleotide positions
    and verifies that it converges to (3/4) * epsilon within specified tolerance.
    """
    valid_mask = (original >= 0) & (original < 4)
    total_valid = valid_mask.sum().item()
    if total_valid == 0:
        return 0.0

    effective_changes = ((noisy != original) & valid_mask).sum().item()
    actual_rate = effective_changes / total_valid
    expected_rate = 0.75 * epsilon

    if abs(actual_rate - expected_rate) > tolerance:
        raise AssertionError(
            f"Empirical substitution rate {actual_rate:.5f} deviated from expected "
            f"{expected_rate:.5f} by more than tolerance {tolerance} (epsilon={epsilon})"
        )

    return actual_rate
