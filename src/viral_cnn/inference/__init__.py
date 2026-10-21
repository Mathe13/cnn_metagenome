"""Inference engine for 1D-CNN models."""
from .engine import (
    predict_sequence,
    predict_batch,
    InferenceResult,
)

__all__ = [
    "predict_sequence",
    "predict_batch",
    "InferenceResult",
]

