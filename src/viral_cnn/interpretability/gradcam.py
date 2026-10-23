"""
1D Grad-CAM and Score-CAM explainability for convolutional DNA classification models.

Scientific and engineering invariants:
- Heatmaps are 1D arrays of shape (seq_len,) normalized to [0.0, 1.0].
- Support for both ResNet50-1D and ConvNeXt-1D architectures.
- Padded and ambiguous positions (null vectors in OneHotEmbedding) are explicitly zero-masked.
- Decoupled visual plotting with optional headless Figure return (show=False).
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple, Union

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from viral_cnn.taxonomy.mapping import load_taxonomy_mapping, model_label_to_taxonomic_id

logger = logging.getLogger(__name__)


def show_binary_data_as_image(
    data_matrix: np.ndarray,
    title: str = "Binary Representation",
    show: bool = True,
) -> Optional[plt.Figure]:
    """Displays a (512, 4) binary one-hot matrix as an image."""
    if data_matrix.shape != (512, 4):
        raise ValueError(f"Input matrix must have shape (512, 4), got {data_matrix.shape}")

    image_data = (data_matrix * 255).astype(np.uint8)
    scale_factor_width = 100
    new_width = image_data.shape[1] * scale_factor_width
    new_height = image_data.shape[0]

    resized_image = cv2.resize(image_data, (new_width, new_height), interpolation=cv2.INTER_NEAREST)

    fig, ax = plt.subplots(figsize=(new_width / 100, new_height / 100))
    ax.imshow(resized_image, cmap="gray", vmin=0, vmax=255)
    ax.set_title(title)
    ax.axis("off")
    if show:
        plt.show()
    return fig


class GradCAM1D:
    """1D Grad-CAM attribution generator."""

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.gradients: Optional[torch.Tensor] = None
        self.activations: Optional[torch.Tensor] = None

        self.target_layer.register_forward_hook(self._save_activations_hook)
        self.target_layer.register_full_backward_hook(self._save_gradients_hook)

    def _save_activations_hook(self, module: nn.Module, input: Tuple[torch.Tensor, ...], output: torch.Tensor):
        self.activations = output

    def _save_gradients_hook(
        self,
        module: nn.Module,
        grad_input: Tuple[Optional[torch.Tensor], ...],
        grad_output: Tuple[torch.Tensor, ...],
    ):
        self.gradients = grad_output[0]

    def __call__(self, input_tensor: torch.Tensor, target_class: Optional[int] = None) -> np.ndarray:
        """
        Computes 1D Grad-CAM heatmap for input_tensor.

        Args:
            input_tensor: (1, seq_len) token indices or (1, 4, seq_len) one-hot tensor.
            target_class: Target class index. If None, highest predicted class is used.

        Returns:
            1D numpy array of shape (seq_len,) with values in [0.0, 1.0].
        """
        self.model.eval()
        self.model.zero_grad()

        output = self.model(input_tensor)
        if target_class is None:
            target_class = int(output.argmax(dim=1).item())

        one_hot_output = torch.zeros_like(output)
        one_hot_output[0][target_class] = 1.0

        output.backward(gradient=one_hot_output, retain_graph=True)

        if self.gradients is None or self.activations is None:
            raise RuntimeError("Failed to capture activations or gradients from target layer.")

        gradients = self.gradients.data.cpu().numpy()  # (B, C, L)
        activations = self.activations.data.cpu().numpy()  # (B, C, L)

        weights = np.mean(gradients, axis=2)[0]  # (num_channels,)

        cam = np.zeros(activations.shape[2], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * activations[0, i, :]

        cam = np.maximum(cam, 0)
        cam_min, cam_max = np.min(cam), np.max(cam)
        cam = cam - cam_min
        cam = cam / (cam_max + 1e-8)

        original_seq_len = input_tensor.shape[-1]
        cam_2d = cam[:, np.newaxis]
        cam_resized = cv2.resize(
            cam_2d,
            dsize=None,
            fx=1,
            fy=original_seq_len / cam_2d.shape[0],
            interpolation=cv2.INTER_LINEAR,
        ).flatten()

        return cam_resized


class ScoreCAM1D:
    """1D Score-CAM attribution generator."""

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.model.eval()
        self.activations: Optional[torch.Tensor] = None

        self.target_layer.register_forward_hook(self._save_activations_hook)

    def _save_activations_hook(self, module: nn.Module, input: Tuple[torch.Tensor, ...], output: torch.Tensor):
        self.activations = output.detach()

    def _forward_from_one_hot(self, x_one_hot: torch.Tensor) -> torch.Tensor:
        """
        Executes forward pass directly with one-hot tensor.
        Both ResNet50_1D and ConvNeXt_Tiny_1D natively accept 3D (B, 4, L) tensors in forward().
        """
        return self.model(x_one_hot)

    def __call__(self, input_tensor_indices: torch.Tensor, target_class: Optional[int] = None) -> np.ndarray:
        """
        Computes 1D Score-CAM heatmap.

        Args:
            input_tensor_indices: (1, seq_len) token indices.
            target_class: Target class index.
        """
        device = input_tensor_indices.device

        with torch.no_grad():
            output = self.model(input_tensor_indices)

        if target_class is None:
            target_class = int(output.argmax(dim=1).item())

        if self.activations is None:
            raise RuntimeError("Activations were not captured by forward hook.")

        activations = self.activations[0]  # (C, L)
        num_channels = activations.size(0)

        # Get one-hot input
        with torch.no_grad():
            if hasattr(self.model, "one_hot_layer"):
                input_one_hot = self.model.one_hot_layer(input_tensor_indices)
            else:
                # Fallback to standard F.one_hot
                clean = input_tensor_indices.clone()
                clean[clean >= 4] = 4
                one_hot = F.one_hot(clean.long(), num_classes=5).float()[:, :, :4].permute(0, 2, 1)
                input_one_hot = one_hot.to(device)

        weights = torch.zeros(num_channels, device=device)

        for k in range(num_channels):
            feature_map_k = activations[k].unsqueeze(0).unsqueeze(0)  # (1, 1, L_feat)
            mask_resized = F.interpolate(
                feature_map_k,
                size=input_one_hot.size(2),
                mode="linear",
                align_corners=False,
            )

            mask_min = mask_resized.min()
            mask_max = mask_resized.max()
            if mask_max > mask_min:
                mask_norm = (mask_resized - mask_min) / (mask_max - mask_min)
            else:
                mask_norm = torch.zeros_like(mask_resized)

            mask_expanded = mask_norm.expand_as(input_one_hot)
            masked_input = (input_one_hot * mask_expanded).to(device)

            with torch.no_grad():
                score_output = self._forward_from_one_hot(masked_input)

            weights[k] = score_output[0, target_class]

        weights_reshaped = weights.unsqueeze(1)
        cam = torch.sum(weights_reshaped * activations, dim=0).detach().cpu().numpy()
        cam = np.maximum(cam, 0)
        cam_min, cam_max = np.min(cam), np.max(cam)
        cam = cam - cam_min
        cam = cam / (cam_max + 1e-8)

        original_seq_len = input_tensor_indices.size(1)
        cam_2d = cam[:, np.newaxis]
        cam_resized = cv2.resize(
            cam_2d,
            dsize=None,
            fx=1,
            fy=original_seq_len / cam_2d.shape[0],
            interpolation=cv2.INTER_LINEAR,
        ).flatten()

        return cam_resized


def visualize_heatmap(
    original_seq_one_hot_np: np.ndarray,
    heatmap: np.ndarray,
    title: str = "Grad-CAM 1D",
    show: bool = True,
) -> Optional[plt.Figure]:
    """Overlays Grad-CAM heatmap on top of one-hot representation."""
    if original_seq_one_hot_np.shape[1] != 4:
        raise ValueError("One-Hot array must have 4 columns (A, C, G, T).")

    image_data = (original_seq_one_hot_np * 255).astype(np.uint8)
    scale_factor_width = 100
    new_width = image_data.shape[1] * scale_factor_width
    new_height = image_data.shape[0]

    resized_image = cv2.resize(image_data, (new_width, new_height), interpolation=cv2.INTER_NEAREST)

    heat_array = np.array(heatmap)[:, np.newaxis]
    heat_array = np.tile(heat_array, (1, 4))
    resized_heatmap = cv2.resize(heat_array, (new_width, new_height), interpolation=cv2.INTER_NEAREST)

    fig, ax = plt.subplots(figsize=(new_width / 100, new_height / 100 + 1))
    ax.imshow(resized_image, cmap="gray", vmin=0, vmax=255)
    im = ax.imshow(resized_heatmap, cmap="jet", alpha=0.5, vmin=0, vmax=1)
    ax.set_title(title, fontsize=14)

    feature_labels = ["A", "C", "G", "T"]
    ax.set_xticks(
        np.arange(len(feature_labels)) * scale_factor_width + (scale_factor_width / 2),
        feature_labels,
        fontsize=12,
    )
    ax.set_yticks([])
    for i in range(1, 4):
        ax.axvline(x=i * scale_factor_width, color="red", linestyle="-", linewidth=0.5)

    fig.colorbar(im, ax=ax, label="Attribution")
    if show:
        plt.show()
    return fig


def _plot_heatmap_comparison(ax, original_seq_one_hot_np, heatmap, title):
    scale_factor_width = 100
    image_data = (original_seq_one_hot_np * 255).astype(np.uint8)
    new_width = image_data.shape[1] * scale_factor_width
    new_height = image_data.shape[0]

    resized_image = cv2.resize(image_data, (new_width, new_height), interpolation=cv2.INTER_NEAREST)
    heat_array = np.tile(np.array(heatmap)[:, np.newaxis], (1, 4))
    resized_heatmap = cv2.resize(heat_array, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

    ax.imshow(resized_image, cmap="gray", vmin=0, vmax=255)
    im = ax.imshow(resized_heatmap, cmap="jet", alpha=0.5, vmin=0, vmax=1)
    ax.set_title(title, fontsize=12)

    feature_labels = ["A", "C", "G", "T"]
    ax.set_xticks(
        np.arange(len(feature_labels)) * scale_factor_width + (scale_factor_width / 2),
        feature_labels,
        fontsize=10,
    )
    ax.set_yticks([])
    for i in range(1, 4):
        ax.axvline(x=i * scale_factor_width, color="red", linestyle="-", linewidth=0.5)
    return im


def visualize_heat_map_examples_by_class(
    model: nn.Module,
    target_layer: nn.Module,
    class_true_id: int,
    target_pred_id: int,
    examples: List[torch.Tensor],
    score_principal_confusao: float,
    cam_type: str,
    device: torch.device,
    mapping_csv_path: Optional[str] = None,
    col_name: str = "species",
    show: bool = True,
) -> plt.Figure:
    """Visualizes comparative CAM maps side-by-side for confused class pairs."""
    cam_type_upper = cam_type.upper().strip()
    if cam_type_upper == "GRAD":
        generator = GradCAM1D(model, target_layer)
    elif cam_type_upper == "SCORE":
        generator = ScoreCAM1D(model, target_layer)
    else:
        raise ValueError(f"cam_type must be 'GRAD' or 'SCORE', got '{cam_type}'")

    mapping = {}
    if mapping_csv_path:
        try:
            mapping = load_taxonomy_mapping(mapping_csv_path, rank_column=col_name)
        except Exception as e:
            logger.warning(f"Could not load mapping CSV {mapping_csv_path}: {e}")

    true_name = f"{class_true_id} - {mapping.get(class_true_id, 'Taxon')}"
    pred_name = f"{target_pred_id} - {mapping.get(target_pred_id, 'Taxon')}"

    num_examples = len(examples)
    fig, axs = plt.subplots(num_examples, 2, figsize=(12, max(4, num_examples * 5)), squeeze=False)
    fig.suptitle(
        f"Comparative {cam_type_upper}-CAM for Confused Pair\n"
        f"True: {true_name} vs. Pred: {pred_name} (Score: {score_principal_confusao:.4f})",
        fontsize=14,
        y=1.02,
    )

    cbar_im = None
    for i, example in enumerate(examples):
        input_tensor_device = example.unsqueeze(0).to(device)

        heatmap_pred = generator(input_tensor_device, target_class=target_pred_id)
        heatmap_target = generator(input_tensor_device, target_class=class_true_id)

        # Zero out padding positions (tokens >= 4)
        is_padding = (example >= 4).cpu().numpy()
        heatmap_target[is_padding] = 0.0
        heatmap_pred[is_padding] = 0.0

        with torch.no_grad():
            if hasattr(model, "one_hot_layer"):
                one_hot_orig = model.one_hot_layer(input_tensor_device)[0].permute(1, 0).cpu().numpy()
            else:
                clean = input_tensor_device.clone()
                clean[clean >= 4] = 4
                one_hot_orig = F.one_hot(clean.long(), num_classes=5).float()[0, :, :4].cpu().numpy()

        ax_true = axs[i, 0]
        cbar_im = _plot_heatmap_comparison(ax_true, one_hot_orig, heatmap_target, f"Ex {i+1} | True: {true_name}")
        ax_pred = axs[i, 1]
        _plot_heatmap_comparison(ax_pred, one_hot_orig, heatmap_pred, f"Ex {i+1} | Pred: {pred_name}")

    if cbar_im is not None:
        fig.colorbar(cbar_im, ax=axs, orientation="horizontal", fraction=0.05, pad=0.08, label=f"{cam_type_upper}-CAM Attribution")

    fig.tight_layout()
    if show:
        plt.show()
    return fig

