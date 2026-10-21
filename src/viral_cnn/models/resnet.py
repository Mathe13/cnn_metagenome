"""
ResNet50-1D model architecture for genomic sequence classification.
Directly equivalent to `short/utils/models.py:resnet50_1d`.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from ..data.encoding import OneHotEmbedding, NUM_FEATURES


class BottleneckBlock1D(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BottleneckBlock1D, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.conv3 = nn.Conv1d(
            out_channels, out_channels * self.expansion, kernel_size=1, bias=False
        )
        self.bn3 = nn.BatchNorm1d(out_channels * self.expansion)
        self.relu = nn.ReLU(inplace=False)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity
        out = self.relu(out)
        return out


class ResNet50_1D(nn.Module):
    def __init__(
        self,
        num_classes: int = 13683,
        max_seq_len: int = 512,
        num_features: int = NUM_FEATURES,
        dropout_rate: float = 0.3,
    ):
        super(ResNet50_1D, self).__init__()
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
        self.layer1 = self._make_layer(BottleneckBlock1D, 64, layers[0], current_in_channels, stride=1)
        current_in_channels = 64 * BottleneckBlock1D.expansion
        self.layer2 = self._make_layer(BottleneckBlock1D, 128, layers[1], current_in_channels, stride=2)
        current_in_channels = 128 * BottleneckBlock1D.expansion
        self.layer3 = self._make_layer(BottleneckBlock1D, 256, layers[2], current_in_channels, stride=2)
        current_in_channels = 256 * BottleneckBlock1D.expansion
        self.layer4 = self._make_layer(BottleneckBlock1D, 512, layers[3], current_in_channels, stride=2)
        current_in_channels = 512 * BottleneckBlock1D.expansion

        self._set_fc_input_features(max_seq_len, num_features)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fc = nn.Linear(self.fc_input_features, num_classes)

    def _make_layer(self, block, out_channels, blocks, in_channels, stride=1):
        downsample = None
        if stride != 1 or in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(
                    in_channels, out_channels * block.expansion, kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm1d(out_channels * block.expansion),
            )
        layer_list = []
        layer_list.append(block(in_channels, out_channels, stride, downsample))
        in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layer_list.append(block(in_channels, out_channels))
        return nn.Sequential(*layer_list)

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
        # If input is integer tokens (batch, seq_len), pass through one_hot_layer
        if x.dim() == 2:
            x = self.one_hot_layer(x)
        # If input is already (batch, num_features, seq_len), skip one_hot_layer

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

