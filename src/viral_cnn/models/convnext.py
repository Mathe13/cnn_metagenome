"""
ConvNeXt-1D model architecture for genomic sequence classification.
Directly equivalent to `short/utils/models.py:convnext_tiny_1d` (HybridConvNeXt).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from ..data.encoding import OneHotEmbedding, NUM_FEATURES


class LayerNorm1d(nn.Module):
    def __init__(self, normalized_shape: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (N, C, L) -> (N, L, C)
        u = x.permute(0, 2, 1)
        out = F.layer_norm(u, (self.normalized_shape,), self.weight, self.bias, self.eps)
        # (N, L, C) -> (N, C, L)
        return out.permute(0, 2, 1)


class ConvNeXtBlock1D(nn.Module):
    expansion = 4

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, downsample=None):
        super(ConvNeXtBlock1D, self).__init__()
        self.downsample = downsample
        dim = in_channels

        # 1. Depthwise Convolution (Spatial mixing, kernel 7, grouped by dim)
        self.dwconv = nn.Conv1d(dim, dim, kernel_size=7, padding=3, groups=dim, stride=stride)

        # 2. Normalization (LayerNorm1d)
        self.norm = LayerNorm1d(dim)

        # 3. Pointwise Convolution 1 (Channel mixing - Expansion to 4x)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()

        # 4. Pointwise Convolution 2 (Projection to target channels * expansion)
        self.pwconv2 = nn.Linear(4 * dim, out_channels * self.expansion)

        # 5. Layer Scale
        self.gamma = nn.Parameter(
            1e-6 * torch.ones((out_channels * self.expansion)), requires_grad=True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_tensor = x

        x = self.dwconv(x)
        x = self.norm(x)

        # (N, C, L) -> (N, L, C) for Linear
        x = x.permute(0, 2, 1)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)

        if self.gamma is not None:
            x = self.gamma * x

        # (N, L, C) -> (N, C, L)
        x = x.permute(0, 2, 1)

        if self.downsample is not None:
            input_tensor = self.downsample(input_tensor)

        x = input_tensor + x
        return x


class ConvNeXt_Tiny_1D(nn.Module):
    def __init__(
        self,
        num_classes: int = 13683,
        max_seq_len: int = 512,
        num_features: int = NUM_FEATURES,
        dropout_rate: float = 0.3,
    ):
        super(ConvNeXt_Tiny_1D, self).__init__()
        self.num_classes = num_classes
        self.max_seq_len = max_seq_len
        self.num_features = num_features

        self.one_hot_layer = OneHotEmbedding(num_features)

        current_in_channels = 64
        self.conv1 = nn.Conv1d(
            num_features, current_in_channels, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = nn.BatchNorm1d(current_in_channels)
        self.relu = nn.ReLU(inplace=False)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        layers = [3, 4, 6, 3]
        self.layer1 = self._make_layer(ConvNeXtBlock1D, 64, layers[0], current_in_channels, stride=1)
        current_in_channels = 64 * ConvNeXtBlock1D.expansion
        self.layer2 = self._make_layer(ConvNeXtBlock1D, 128, layers[1], current_in_channels, stride=2)
        current_in_channels = 128 * ConvNeXtBlock1D.expansion
        self.layer3 = self._make_layer(ConvNeXtBlock1D, 256, layers[2], current_in_channels, stride=2)
        current_in_channels = 256 * ConvNeXtBlock1D.expansion
        self.layer4 = self._make_layer(ConvNeXtBlock1D, 512, layers[3], current_in_channels, stride=2)
        current_in_channels = 512 * ConvNeXtBlock1D.expansion

        self._set_fc_input_features(max_seq_len, num_features)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fc = nn.Linear(self.fc_input_features, num_classes)

    def _make_layer(self, block, out_channels, blocks, in_channels, stride=1):
        downsample = None
        if stride != 1 or in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                LayerNorm1d(in_channels),
                nn.Conv1d(
                    in_channels, out_channels * block.expansion, kernel_size=1, stride=stride, bias=False
                ),
            )

        layers_list = []
        layers_list.append(block(in_channels, out_channels, stride, downsample))
        in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers_list.append(block(in_channels, out_channels))

        return nn.Sequential(*layers_list)

    def _set_fc_input_features(self, max_seq_len, num_features):
        with torch.no_grad():
            dummy_input = torch.randint(0, num_features, (1, max_seq_len))
            x = self.one_hot_layer(dummy_input)
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
            x = F.max_pool1d(x, kernel_size=x.size(2))
            x = torch.flatten(x, 1)
            self.fc_input_features = x.size(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = self.one_hot_layer(x)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = F.max_pool1d(x, kernel_size=x.size(2))
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        return x

