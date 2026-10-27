"""
DNA sequence encoding and spatial one-hot representation.

Invariant rules (AGENTS.md):
- 512 bp windows are encoded in four channels in the strict order:
    Channel 0: A
    Channel 1: C
    Channel 2: G
    Channel 3: T
- Padding and unknown/ambiguous bases generate an all-zero vector [0.0, 0.0, 0.0, 0.0].
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

NUCLEOTIDES = ["A", "C", "G", "T"]
NUCLEOTIDE_MAPPING = {n: i for i, n in enumerate(NUCLEOTIDES)}
NUM_FEATURES = len(NUCLEOTIDES)  # 4


def encode_sequence_tokens(
    sequence: str, max_len: int = 512, pad_char: str = "_"
) -> list[int]:
    """
    Encodes a DNA string into a list of integer tokens of length `max_len`.
    Canonical nucleotides:
      'A' -> 0, 'C' -> 1, 'G' -> 2, 'T' -> 3
    Any other character (including padding, 'N', IUPAC ambiguities, gaps) -> NUM_FEATURES (4).

    Sequences longer than `max_len` are truncated.
    Sequences shorter than `max_len` are padded with token 4 up to `max_len`.
    """
    seq_upper = sequence.upper().strip()
    tokens = [NUCLEOTIDE_MAPPING.get(base, NUM_FEATURES) for base in seq_upper[:max_len]]
    if len(tokens) < max_len:
        tokens.extend([NUM_FEATURES] * (max_len - len(tokens)))
    return tokens


class OneHotEmbedding(nn.Module):
    """
    1D Spatial One-Hot Embedding layer matching historical `short/utils/models.py`.

    Takes input tensor of shape (batch_size, seq_len) of token indices [0..num_features].
    Produces output tensor of shape (batch_size, num_features, seq_len).
    Indices >= num_features or < 0 produce null vectors [0.0, 0.0, 0.0, 0.0].
    """

    def __init__(self, num_features: int = NUM_FEATURES):
        super(OneHotEmbedding, self).__init__()
        self.num_features = num_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_clean = torch.clamp(
            torch.where(x < 0, self.num_features, x),
            min=0,
            max=self.num_features,
        )

        one_hot = F.one_hot(x_clean.long(), num_classes=self.num_features + 1).float()
        one_hot = one_hot[:, :, : self.num_features]
        one_hot = one_hot.permute(0, 2, 1)
        return one_hot


def encode_spatial_one_hot(
    sequences: list[str] | str, max_len: int = 512, device: str = "cpu"
) -> torch.Tensor:
    """
    High-level convenience function: encodes sequence(s) directly into a
    (batch_size, 4, max_len) spatial one-hot tensor.
    """
    if isinstance(sequences, str):
        sequences = [sequences]

    token_matrix = [encode_sequence_tokens(s, max_len=max_len) for s in sequences]
    tokens_tensor = torch.tensor(token_matrix, dtype=torch.long, device=device)
    embedding_layer = OneHotEmbedding(num_features=NUM_FEATURES).to(device)
    with torch.no_grad():
        return embedding_layer(tokens_tensor)

