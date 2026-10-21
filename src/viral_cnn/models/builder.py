"""
Model builder and factory function for 1D genomic architectures.
"""

import torch.nn as nn
from .resnet import ResNet50_1D
from .convnext import ConvNeXt_Tiny_1D
from ..data.encoding import NUM_FEATURES

MODEL_REGISTRY = {
    "resnet50_1d": ResNet50_1D,
    "resnet50": ResNet50_1D,
    "convnext_tiny_1d": ConvNeXt_Tiny_1D,
    "convnext": ConvNeXt_Tiny_1D,
}


def build_model(
    model_type: str,
    num_classes: int = 13683,
    max_seq_len: int = 512,
    dropout_rate: float = 0.3,
) -> nn.Module:
    """
    Factory creating a 1D-CNN instance given the architecture name.

    Supported model types:
      - 'resnet50_1d' / 'resnet50'
      - 'convnext_tiny_1d' / 'convnext'
    """
    normalized_type = model_type.lower().strip()
    if normalized_type not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model_type '{model_type}'. Supported: {list(MODEL_REGISTRY.keys())}"
        )

    model_cls = MODEL_REGISTRY[normalized_type]
    return model_cls(
        num_classes=num_classes,
        max_seq_len=max_seq_len,
        num_features=NUM_FEATURES,
        dropout_rate=dropout_rate,
    )

