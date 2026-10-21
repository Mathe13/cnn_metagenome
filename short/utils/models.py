
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys

class ConvNeXtBlock1D(nn.Module):
    expansion = 4 

    def __init__(self, in_channels, out_channels, stride=1, downsample=None, debug=False):
        super(ConvNeXtBlock1D, self).__init__()
        self.debug = debug
        self.downsample = downsample
        
        # Calculate intermediate hidden dimension (Inverted Bottleneck)
        # ConvNeXt typically expands 4x inside the block
        dim = in_channels
        
        # 1. Depthwise Convolution (Spatial mixing)
        # Larger kernel (7), grouped by channels (groups=dim)
        self.dwconv = nn.Conv1d(dim, dim, kernel_size=7, padding=3, groups=dim, stride=stride) 
        
        # 2. Normalization (LayerNorm instead of BatchNorm)
        self.norm = LayerNorm1d(dim)
        
        # 3. Pointwise Convolution 1 (Channel mixing - Expansion)
        # We expand to 4x the input, or match the target output expansion
        self.pwconv1 = nn.Linear(dim, 4 * dim) 
        self.act = nn.GELU()
        
        # 4. Pointwise Convolution 2 (Projection)
        # Project back to the expected output channels (out_channels * expansion)
        self.pwconv2 = nn.Linear(4 * dim, out_channels * self.expansion)
        
        # Layer Scale (Optional but recommended for deep ConvNeXts)
        self.gamma = nn.Parameter(1e-6 * torch.ones((out_channels * self.expansion)), requires_grad=True)

    def forward(self, x):
        input = x
        
        if self.debug: print_tensor_stats(x, "ConvNeXt Block Input", "Activations")

        # --- Depthwise Conv ---
        x = self.dwconv(x)
        if self.debug: print_tensor_stats(x, "Depthwise Conv", "Activations")
        
        # --- LayerNorm ---
        x = self.norm(x)
        
        # --- Pointwise (implemented via Linear for efficiency, requires permutation) ---
        # (N, C, L) -> (N, L, C)
        x = x.permute(0, 2, 1) 
        
        x = self.pwconv1(x)
        x = self.act(x)
        if self.debug: print_tensor_stats(x.permute(0,2,1), "Pointwise Expansion", "Activations")

        x = self.pwconv2(x)
        
        # Apply Layer Scale
        if self.gamma is not None:
            x = self.gamma * x
            
        # (N, L, C) -> (N, C, L)
        x = x.permute(0, 2, 1)

        # --- Residual Connection ---
        if self.downsample is not None:
            input = self.downsample(input)

        x = input + x
        
        # Note: ConvNeXt does NOT put an activation after the addition
        if self.debug: print_tensor_stats(x, "Block Output", "Activations")
        
        return x

class LayerNorm1d(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape
        self.eps = eps

    def forward(self, x):
        # Permute: (N, C, L) -> (N, L, C)
        u = x.permute(0, 2, 1)
        # Apply LayerNorm
        out = F.layer_norm(u, (self.normalized_shape,), self.weight, self.bias, self.eps)
        # Permute back: (N, L, C) -> (N, C, L)
        return out.permute(0, 2, 1)

class OneHotEmbedding(nn.Module):
    def __init__(self, num_features):
        super(OneHotEmbedding, self).__init__()
        self.num_features = num_features
    
    def forward(self, x):
        # x é um tensor de inteiros com shape (batch_size, seq_len)
        
        # 1. Copia o tensor de entrada para não modificar o original
        x_clean = x.clone()
        
        # 2. Mapeia todos os índices inválidos para o índice de padding (`_` ou `4`)
        # O valor de padding é self.num_features
        x_clean[x_clean >= self.num_features] = self.num_features
        x_clean[x_clean < 0] = self.num_features
        
        # 3. Use F.one_hot com o número total de classes, incluindo o padding
        one_hot = F.one_hot(x_clean.long(), num_classes=self.num_features + 1).float()
        
        # 4. Pegue apenas os primeiros 'num_features' canais, ignorando o canal de padding
        one_hot = one_hot[:, :, :self.num_features]
        
        # 5. Permute para o formato esperado pelo Conv1d
        one_hot = one_hot.permute(0, 2, 1)
        
        return one_hot


# Função para imprimir estatísticas de um tensor
def print_tensor_stats(tensor, layer_name, tensor_type):
    if tensor is None or tensor.numel() == 0:
        return
    
    mean = tensor.mean().item()
    std = tensor.std().item()
    min_val = tensor.min().item()
    max_val = tensor.max().item()
    
    print(f"DEBUG: {layer_name:<25} | {tensor_type:<10} | Mean: {mean:.4e} | Std: {std:.4e} | Min: {min_val:.4e} | Max: {max_val:.4e}")

class BottleneckBlock1D(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, downsample=None, debug=False):
        super(BottleneckBlock1D, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.conv3 = nn.Conv1d(out_channels, out_channels * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm1d(out_channels * self.expansion)
        self.relu = nn.ReLU(inplace=False)
        self.downsample = downsample
        self.stride = stride
        self.debug = debug

    def forward(self, x):
        identity = x
        
        if self.debug: print_tensor_stats(x, "Bottleneck Input", "Activations")

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        if self.debug: print_tensor_stats(out, "Bottleneck conv1->bn1->relu", "Activations")

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        if self.debug: print_tensor_stats(out, "Bottleneck conv2->bn2->relu", "Activations")

        out = self.conv3(out)
        out = self.bn3(out)
        if self.debug: print_tensor_stats(out, "Bottleneck conv3->bn3", "Activations")

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        if self.debug: print_tensor_stats(out, "Bottleneck Output", "Activations")
        
        return out

class ResNet50_1D(nn.Module):
    def __init__(self, block, layers, num_features, num_classes, max_seq_len, debug=False, dropout_rate=0.3):
        super(ResNet50_1D, self).__init__()
        
        self.one_hot_layer = OneHotEmbedding(num_features)
        
        current_in_channels = 64
        self.conv1 = nn.Conv1d(num_features, current_in_channels, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(current_in_channels)
        self.relu = nn.ReLU(inplace=False)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0], current_in_channels, stride=1, debug=debug)
        current_in_channels = 64 * block.expansion
        self.layer2 = self._make_layer(block, 128, layers[1], current_in_channels, stride=2, debug=debug)
        current_in_channels = 128 * block.expansion
        self.layer3 = self._make_layer(block, 256, layers[2], current_in_channels, stride=2, debug=debug)
        current_in_channels = 256 * block.expansion
        self.layer4 = self._make_layer(block, 512, layers[3], current_in_channels, stride=2, debug=debug)
        current_in_channels = 512 * block.expansion

        self._set_avgpool_input_features(max_seq_len, num_features)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fc = nn.Linear(self.fc_input_features, num_classes)
        self.debug = debug

    def _make_layer(self, block, out_channels, blocks, in_channels, stride=1, debug=False):
        downsample = None
        if stride != 1 or in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(in_channels, out_channels * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels * block.expansion),
            )
        layers = []
        layers.append(block(in_channels, out_channels, stride, downsample, debug=debug))
        in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(in_channels, out_channels, debug=debug))
        return nn.Sequential(*layers)
    
    def _set_avgpool_input_features(self, max_seq_len, num_features):
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

    def forward(self, x):
        # Mova o tensor para a CPU para verificar (se necessário)
        # if self.debug: print(f"DEBUG: Entrada - Device: {x.device}, Shape: {x.shape}")
        
        x = self.one_hot_layer(x)
        if self.debug: print_tensor_stats(x, "Após OneHotEmbedding", "Ativações")
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if self.debug: print_tensor_stats(x, "Após conv1 -> bn1 -> relu", "Ativações")
        
        x = self.maxpool(x)
        if self.debug: print_tensor_stats(x, "Após maxpool", "Ativações")
        
        x = self.layer1(x)
        if self.debug: print_tensor_stats(x, "Após layer1", "Ativações")
        
        x = self.layer2(x)
        if self.debug: print_tensor_stats(x, "Após layer2", "Ativações")
        
        x = self.layer3(x)
        if self.debug: print_tensor_stats(x, "Após layer3", "Ativações")
        
        x = self.layer4(x)
        if self.debug: print_tensor_stats(x, "Após layer4", "Ativações")
        
        x = F.max_pool1d(x, kernel_size=x.size(2))
        if self.debug: print_tensor_stats(x, "Após Global Max Pooling", "Ativações")
        
        x = torch.flatten(x, 1)
        if self.debug: print_tensor_stats(x, "Após flatten", "Ativações")

        x = self.dropout(x)
        if self.debug: print_tensor_stats(x, "Após dropout", "Ativações")
        
        x = self.fc(x)
        if self.debug: print_tensor_stats(x, "Saída (logits)", "Ativações")
        
        return x
def resnet50_1d(max_seq_len, num_features, num_classes,debug=False,dropout_rate=0.3):
    resnet50_layers = [3, 4, 6, 3]
    return ResNet50_1D(BottleneckBlock1D, resnet50_layers, num_features, num_classes, max_seq_len,debug=debug,dropout_rate=dropout_rate)



class HybridConvNeXt(ResNet50_1D):
    def __init__(self, block, layers, num_features, num_classes, max_seq_len, debug=False, dropout_rate=0.3):
        super().__init__(block, layers, num_features, num_classes, max_seq_len, debug, dropout_rate)
        # Inherits everything, but we override _make_layer to use LayerNorm in downsampling

    def _make_layer(self, block, out_channels, blocks, in_channels, stride=1, debug=False):
        downsample = None
        # Verify if we need to downsample dimensions or spatial resolution
        if stride != 1 or in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                # Use LayerNorm1d instead of BatchNorm1d for the skip connection
                LayerNorm1d(in_channels), 
                nn.Conv1d(in_channels, out_channels * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
            )

        layers = []
        layers.append(block(in_channels, out_channels, stride, downsample, debug=debug))
        in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(in_channels, out_channels, debug=debug))

        return nn.Sequential(*layers)

# Instantiation Function
def convnext_tiny_1d(max_seq_len, num_features, num_classes, debug=False, dropout_rate=0.3):
    # Using ResNet50 stage sizes [3, 4, 6, 3] creates a model roughly equivalent to ConvNeXt-Tiny
    layers = [3, 4, 6, 3] 
    return HybridConvNeXt(
        ConvNeXtBlock1D, 
        layers, 
        num_features, 
        num_classes, 
        max_seq_len, 
        debug=debug, 
        dropout_rate=dropout_rate
    )