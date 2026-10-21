"""
Clean forward inference engine for 1D-CNN models.
"""

from dataclasses import dataclass
import torch
import torch.nn.functional as F
from ..data.encoding import encode_sequence_tokens, NUM_FEATURES


@dataclass
class InferenceResult:
    predicted_class_index: int
    confidence: float
    top5_class_indices: list[int]
    top5_confidences: list[float]
    input_length: int
    channels: int
    logits: list[float]


def predict_sequence(
    model: torch.nn.Module,
    sequence: str,
    device: str = "cpu",
    max_len: int = 512,
    top_k: int = 5,
) -> InferenceResult:
    """
    Executes a clean forward pass for a single DNA sequence.
    """
    model.eval()
    model.to(device)

    tokens = encode_sequence_tokens(sequence, max_len=max_len)
    tokens_tensor = torch.tensor([tokens], dtype=torch.long, device=device)

    with torch.no_grad():
        logits = model(tokens_tensor)
        probs = F.softmax(logits, dim=-1)

    k = min(top_k, logits.size(-1))
    topk_probs, topk_indices = torch.topk(probs, k=k, dim=-1)

    pred_idx = topk_indices[0, 0].item()
    pred_conf = topk_probs[0, 0].item()
    top5_indices = topk_indices[0].tolist()
    top5_confidences = topk_probs[0].tolist()

    return InferenceResult(
        predicted_class_index=pred_idx,
        confidence=pred_conf,
        top5_class_indices=top5_indices,
        top5_confidences=top5_confidences,
        input_length=len(sequence),
        channels=NUM_FEATURES,
        logits=logits[0].tolist(),
    )


def predict_batch(
    model: torch.nn.Module,
    sequences: list[str],
    device: str = "cpu",
    max_len: int = 512,
    top_k: int = 5,
) -> list[InferenceResult]:
    """
    Executes a clean forward pass for a batch of DNA sequences.
    """
    return [
        predict_sequence(model, s, device=device, max_len=max_len, top_k=top_k)
        for s in sequences
    ]

