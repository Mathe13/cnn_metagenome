"""
Model registry, metadata specifications, and unified checkpoint loading for 1D CNN models.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import torch
import torch.nn as nn
import yaml

from .builder import build_model


@dataclass
class ModelVariantSpec:
    name: str
    architecture: str
    rank: str
    num_classes: int
    checkpoint_rel_path: str
    dropout_rate: float
    selection_criterion: str
    status: str
    verified_hyperparameters: Dict[str, Any] = field(default_factory=dict)
    unverified_hyperparameters: Dict[str, Any] = field(default_factory=dict)


# Canonical 7 model variants evaluated in the manuscript
REGISTERED_VARIANTS: Dict[str, ModelVariantSpec] = {
    "resnet_clean_species": ModelVariantSpec(
        name="resnet_clean_species",
        architecture="resnet50_1d",
        rank="species",
        num_classes=13683,
        checkpoint_rel_path="cnn_kraken_species/checkpoints/best_resnet_classifier__fast_phase4.pt",
        dropout_rate=0.5,
        selection_criterion="best_validation_phase4_restart",
        status="verified",
        verified_hyperparameters={
            "optimizer": "AdamW",
            "scheduler": "CosineAnnealingLR",
            "label_smoothing": 0.1,
            "amp": True,
            "max_sequence_length": 512,
            "channels": ["A", "C", "G", "T"],
        },
        unverified_hyperparameters={
            "early_stopping_patience": "unverified - intermediate training logs missing",
        },
    ),
    "resnet_noise_species": ModelVariantSpec(
        name="resnet_noise_species",
        architecture="resnet50_1d",
        rank="species",
        num_classes=13683,
        checkpoint_rel_path="cnn_kraken/checkpoint_finetune_resnet/finetune_noise_14.pt",
        dropout_rate=0.5,
        selection_criterion="final_training_epoch_14",
        status="verified",
        verified_hyperparameters={
            "optimizer": "AdamW",
            "scheduler": "CosineAnnealingLR",
            "label_smoothing": 0.1,
            "amp": True,
            "noise_rate": 0.05,
            "max_lr": 5e-5,
            "weight_decay": 1e-4,
            "epochs": 15,
            "max_sequence_length": 512,
            "channels": ["A", "C", "G", "T"],
        },
        unverified_hyperparameters={},
    ),
    "convnext_zero_species": ModelVariantSpec(
        name="convnext_zero_species",
        architecture="convnext_tiny_1d",
        rank="species",
        num_classes=13683,
        checkpoint_rel_path="cnn_kraken/checkpoint_species_conv_zero/best_convnext_phase3_restarts_v2.pt",
        dropout_rate=0.7,
        selection_criterion="best_validation_phase3_restart",
        status="verified",
        verified_hyperparameters={
            "optimizer": "AdamW",
            "scheduler": "CosineAnnealingLR",
            "label_smoothing": 0.1,
            "amp": True,
            "max_sequence_length": 512,
            "channels": ["A", "C", "G", "T"],
        },
        unverified_hyperparameters={
            "weight_initialization_seed": "unverified - phase 1 init seed missing",
        },
    ),
    "convnext_transfer_species": ModelVariantSpec(
        name="convnext_transfer_species",
        architecture="convnext_tiny_1d",
        rank="species",
        num_classes=13683,
        checkpoint_rel_path="cnn_kraken/checkpoint_transfer_species_conv/best_convnext_classifier_fast_phase4_2_restarts.pt",
        dropout_rate=0.7,
        selection_criterion="best_validation_phase4_restart",
        status="verified",
        verified_hyperparameters={
            "optimizer": "AdamW",
            "scheduler": "CosineAnnealingLR",
            "label_smoothing": 0.1,
            "amp": True,
            "max_sequence_length": 512,
            "channels": ["A", "C", "G", "T"],
        },
        unverified_hyperparameters={
            "intermediate_loss_thresholds": "unverified - phase transfer logs missing",
        },
    ),
    "convnext_noise_species": ModelVariantSpec(
        name="convnext_noise_species",
        architecture="convnext_tiny_1d",
        rank="species",
        num_classes=13683,
        checkpoint_rel_path="cnn_kraken/checkpoint_transfer_species_conv/finetune_noise_7.pt",
        dropout_rate=0.7,
        selection_criterion="best_validation_epoch_7",
        status="verified",
        verified_hyperparameters={
            "optimizer": "AdamW",
            "scheduler": "CosineAnnealingLR",
            "label_smoothing": 0.1,
            "amp": True,
            "noise_rate": 0.05,
            "max_lr": 5e-5,
            "weight_decay": 1e-4,
            "epochs": 15,
            "max_sequence_length": 512,
            "channels": ["A", "C", "G", "T"],
        },
        unverified_hyperparameters={},
    ),
    "resnet_genus": ModelVariantSpec(
        name="resnet_genus",
        architecture="resnet50_1d",
        rank="genus",
        num_classes=2702,
        checkpoint_rel_path="cnn_kraken_genus/best_resnet_classifier__fast_phase4.pt",
        dropout_rate=0.5,
        selection_criterion="best_validation_phase4_restart",
        status="verified",
        verified_hyperparameters={
            "optimizer": "AdamW",
            "scheduler": "CosineAnnealingLR",
            "label_smoothing": 0.1,
            "amp": True,
            "max_sequence_length": 512,
            "channels": ["A", "C", "G", "T"],
        },
        unverified_hyperparameters={
            "early_stopping_patience": "unverified - phase logs missing",
        },
    ),
    "convnext_genus": ModelVariantSpec(
        name="convnext_genus",
        architecture="convnext_tiny_1d",
        rank="genus",
        num_classes=2702,
        checkpoint_rel_path="cnn_kraken/checkpoint_genus_conv/best_convnext_classifier_fast_phase4_2.pt",
        dropout_rate=0.7,
        selection_criterion="best_validation_phase4_restart",
        status="verified",
        verified_hyperparameters={
            "optimizer": "AdamW",
            "scheduler": "CosineAnnealingLR",
            "label_smoothing": 0.1,
            "amp": True,
            "max_sequence_length": 512,
            "channels": ["A", "C", "G", "T"],
        },
        unverified_hyperparameters={
            "early_stopping_patience": "unverified - phase logs missing",
        },
    ),
}

# Synonyms / aliases for flexible resolution
VARIANT_ALIASES: Dict[str, str] = {
    "resnet_species": "resnet_clean_species",
    "resnet_species_clean": "resnet_clean_species",
    "resnet_species_noise": "resnet_noise_species",
    "convnext_species_zero": "convnext_zero_species",
    "convnext_z": "convnext_zero_species",
    "convnext_species_transfer": "convnext_transfer_species",
    "convnext_t": "convnext_transfer_species",
    "convnext_species_fine": "convnext_noise_species",
    "convnext_species_noise": "convnext_noise_species",
    "convnext_f": "convnext_noise_species",
    "resnet_50_genus": "resnet_genus",
    "convnext_tiny_genus": "convnext_genus",
}


def list_model_variants() -> List[str]:
    """Returns the list of all canonical registered model variant keys."""
    return sorted(list(REGISTERED_VARIANTS.keys()))


def get_model_spec(variant_name: str) -> ModelVariantSpec:
    """
    Retrieves the ModelVariantSpec for a given model variant name or alias.
    """
    key = variant_name.lower().strip()
    resolved_key = VARIANT_ALIASES.get(key, key)
    if resolved_key not in REGISTERED_VARIANTS:
        supported = list_model_variants()
        raise ValueError(
            f"Unknown model variant '{variant_name}'. Supported variants: {supported}"
        )
    return REGISTERED_VARIANTS[resolved_key]


def load_model_checkpoint(
    model: nn.Module,
    checkpoint: Union[str, Path, Dict[str, Any]],
    expected_classes: Optional[int] = None,
    map_location: str = "cpu",
    strict: bool = True,
) -> Dict[str, Any]:
    """
    Loads model checkpoint weights with automatic wrapper format handling,
    _orig_mod. compile-prefix stripping, and classification head shape verification.

    Parameters:
      model: Target PyTorch module.
      checkpoint: Filepath (str/Path) or already loaded dictionary.
      expected_classes: Expected number of classes. If None, inferred from model.num_classes if present.
      map_location: PyTorch device string for torch.load.
      strict: Whether to strictly enforce state_dict key matching.

    Returns:
      Dictionary of checkpoint metadata (e.g. 'epoch', 'metrics') if present, or empty dict.
    """
    if isinstance(checkpoint, (str, Path)):
        ckpt_path = Path(checkpoint)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint file not found: {ckpt_path}")
        raw_checkpoint = torch.load(ckpt_path, map_location=map_location)
    elif isinstance(checkpoint, dict):
        raw_checkpoint = checkpoint
    else:
        raise TypeError(
            f"Unsupported checkpoint type: {type(checkpoint)}. Expected path or dict."
        )

    # Handle dictionary wrapper format vs bare state_dict
    metadata: Dict[str, Any] = {}
    if isinstance(raw_checkpoint, dict) and "model_state_dict" in raw_checkpoint:
        state_dict = raw_checkpoint["model_state_dict"]
        metadata = {k: v for k, v in raw_checkpoint.items() if k != "model_state_dict"}
    else:
        state_dict = raw_checkpoint

    # Strip torch.compile _orig_mod. prefix
    clean_state_dict = {
        (k[len("_orig_mod.") :] if k.startswith("_orig_mod.") else k): v
        for k, v in state_dict.items()
    }

    # Head shape / class count verification
    target_classes = expected_classes
    if target_classes is None and hasattr(model, "num_classes"):
        target_classes = getattr(model, "num_classes")

    if "fc.weight" in clean_state_dict and target_classes is not None:
        ckpt_classes = clean_state_dict["fc.weight"].shape[0]
        if ckpt_classes != target_classes:
            raise ValueError(
                f"Classification head class count mismatch: checkpoint has {ckpt_classes} classes "
                f"but model/expected specifies {target_classes} classes."
            )

    model.load_state_dict(clean_state_dict, strict=strict)
    return metadata


def build_registered_model(
    variant_name: str,
    checkpoint_path: Optional[Union[str, Path]] = None,
    device: str = "cpu",
    dropout_rate: Optional[float] = None,
) -> nn.Module:
    """
    Instantiates and optionally loads weights for a registered model variant.
    """
    spec = get_model_spec(variant_name)
    dropout = dropout_rate if dropout_rate is not None else spec.dropout_rate

    model = build_model(
        model_type=spec.architecture,
        num_classes=spec.num_classes,
        dropout_rate=dropout,
    )

    if checkpoint_path is not None:
        load_model_checkpoint(
            model=model,
            checkpoint=checkpoint_path,
            expected_classes=spec.num_classes,
            map_location=device,
        )

    model.to(device)
    return model
