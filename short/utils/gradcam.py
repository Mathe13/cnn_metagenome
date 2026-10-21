import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch.nn as nn
from pytorch_grad_cam import GradCAM
import torch
import torch.nn.functional as F

def show_binary_data_as_image(data_matrix, title="Imagem Binária"):
    """
    Exibe uma matriz de dados binários (0s e 1s) como uma imagem no Google Colab/Jupyter.

    Args:
        data_matrix (np.ndarray): Uma matriz NumPy com shape (512, 4)
                                  contendo apenas valores 0 ou 1.
        title (str): O título para a imagem.
    """
    # Verifica se a matriz tem o shape correto
    if data_matrix.shape != (512, 4):
        print(f"Erro: A matriz de entrada deve ter o shape (512, 4), mas tem {data_matrix.shape}")
        return

    # Certifica-se de que os dados são do tipo uint8 e escala para 0-255
    # para que 1 se torne branco e 0 se torne preto na imagem em escala de cinza.
    image_data = (data_matrix * 255).astype(np.uint8)

    # Aumenta a escala da imagem para melhor visualização
    # Uma imagem 512x4 é muito fina. Vamos redimensioná-la.
    # O fator de escala de largura é aumentado para 100x para torná-la visível.
    scale_factor_width = 100
    scale_factor_height = 1 # Manter a altura original (já é 512)

    # Calcule as novas dimensões
    new_width = image_data.shape[1] * scale_factor_width
    new_height = image_data.shape[0] * scale_factor_height

    # Redimensiona a imagem usando interpolação vizinho mais próximo para dados binários.
    # Isso evita valores intermediários (cinza) que não existem nos seus dados 0/1.
    resized_image = cv2.resize(image_data, (new_width, new_height), interpolation=cv2.INTER_NEAREST)

    # --- Exibindo a imagem com Matplotlib (Funciona no Google Colab) ---
    plt.figure(figsize=(new_width / 100, new_height / 100)) # Ajusta o tamanho da figura para ser proporcional
    plt.imshow(resized_image, cmap='gray', vmin=0, vmax=255)
    plt.title(title)
    plt.axis('off') # Remove os eixos para uma visualização limpa
    plt.show()


class GradCAM1D:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # Registrar hooks para extrair gradientes e ativações
        self.target_layer.register_forward_hook(self._save_activations_hook)
        
        # USE register_full_backward_hook AQUI
        self.target_layer.register_full_backward_hook(self._save_gradients_hook)


    def _save_activations_hook(self, module, input, output):
        self.activations = output
    
    def _save_gradients_hook(self, module, grad_input, grad_output):
        # grad_output[0] é o gradiente para a saída do módulo
        self.gradients = grad_output[0]

    def __call__(self, input_tensor, target_class=None):
        """
        Calcula o mapa Grad-CAM para uma entrada específica.

        Args:
            input_tensor (torch.Tensor): O tensor de entrada para o modelo (batch_size, seq_len).
            target_class (int, optional): A classe alvo para a qual calcular o Grad-CAM.
                                          Se None, usará a classe com maior pontuação (predita).

        Returns:
            np.ndarray: O mapa Grad-CAM (heatmap) redimensionado para o comprimento original da sequência,
                        com valores entre 0 e 1.
        """
        self.model.eval() # Modo de avaliação

        # Zerar gradientes anteriores
        self.model.zero_grad()

        # Forward pass
        output = self.model(input_tensor) # output shape: (batch_size, num_classes)
        
        if target_class is None:
            target_class = output.argmax(dim=1).item() # Escolhe a classe com maior score

        # Backward pass para a classe alvo
        one_hot_output = torch.zeros_like(output)
        one_hot_output[0][target_class] = 1 # Para o primeiro item do batch
        
        output.backward(gradient=one_hot_output, retain_graph=True) # Calcule os gradientes

        # Extrair gradientes e ativações
        gradients = self.gradients.data.cpu().numpy() # Shape: (B, C, L)
        activations = self.activations.data.cpu().numpy() # Shape: (B, C, L)

        # Média global dos gradientes (pooling por canal)
        # alpha_k = 1/Z * sum(sum(dY^c / dA^k_ij))
        # Para 1D, fazemos a média ao longo da dimensão de comprimento (L)
        # gradients_mean_across_length = np.mean(gradients, axis=2) # Shape: (B, C)
        
        # O paper original do Grad-CAM sugere a média por feature map (canal)
        # e então multiplicar cada feature map pelo seu peso correspondente.
        # Estamos pegando o gradiente da saída para o feature map (activations)
        # e tirando a média ao longo da dimensão de comprimento (L) para obter um peso por canal.
        
        # gradientes_ponderados = np.mean(gradients, axis=2)[0] # Shape: (C,)
        # Para cada canal, faça a média de seus gradientes ao longo da dimensão espacial.
        # Isso dá um peso (alpha_k) para cada canal de feature map.
        weights = np.mean(gradients, axis=2)[0] # (num_channels,)

        # Gerar o mapa de calor
        # Combine linearmente os feature maps com os pesos calculados (ReLU)
        cam = np.zeros(activations.shape[2], dtype=np.float32) # (L_target,)
        for i, w in enumerate(weights):
            cam += w * activations[0, i, :] # Sum across channels, for the first item in batch

        # Aplicar ReLU para obter apenas as contribuições positivas
        cam = np.maximum(cam, 0)

        # Normalizar para 0-1
        cam = cam - np.min(cam)
        cam = cam / (np.max(cam) + 1e-8) # Adicionar epsilon para evitar divisão por zero

        # Redimensionar para o comprimento original da sequência
        original_seq_len = input_tensor.shape[1]
        cam_2d = cam[:, np.newaxis]
        cam = cv2.resize(
        cam_2d, 
        dsize=None, # Define dsize como None
        fx=1, # Fator de escala de largura (mantemos em 1 coluna)
        fy=original_seq_len / cam_2d.shape[0], # Fator de escala de altura
        interpolation=cv2.INTER_LINEAR
        )
        
        # 3. Retorna o vetor 1D (original_seq_len,)
        cam = cam.flatten()
        
        return cam

class ScoreCAM1D:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.model.eval()
        
        # Buffer para armazenar as ativações da camada alvo
        self.activations = None
        
        # Registrar o forward hook na camada alvo
        self.target_layer.register_forward_hook(self._save_activations_hook)

    def _save_activations_hook(self, module, input, output):
        # Apenas armazena as ativações (não precisa de requires_grad)
        self.activations = output.detach()
    
    def __call__(self, input_tensor_indices, target_class=None):
        """
        Calcula o mapa Score-CAM para uma sequência 1D.

        Args:
            input_tensor_indices (torch.Tensor): O tensor de entrada (1, 512) de índices inteiros.
            target_class (int, optional): A classe alvo para a qual calcular o Score-CAM.
        """
        device = input_tensor_indices.device
        
        # 1. Forward Pass Inicial para obter as ativações (A_k)
        with torch.no_grad():
            output = self.model(input_tensor_indices)
            
        if target_class is None:
            # Seleciona a classe com o score mais alto (predita)
            target_class = output.argmax(dim=1).item()
            
        # O self.activations agora contém A_k (B=1, C=2048, L=16, aproximadamente)
        activations = self.activations[0] # Pega o item do batch: (C, L)
        num_channels = activations.size(0)
        target_len = activations.size(1)

        # 2. Score-CAM: Calcular o peso (alpha_k) para cada canal
        weights = torch.zeros(num_channels, device=device)
        
        # O input_tensor_indices é (1, 512). Precisamos do input_one_hot para mascarar
        # Nota: O forward do modelo faz o OneHot, então precisamos replicar essa conversão
        # para obter a entrada One-Hot que a rede vê.
        with torch.no_grad():
             input_one_hot_B_C_L = self.model.one_hot_layer(input_tensor_indices)
        
        # Itera sobre cada feature map (canal k)
        for k in range(num_channels):
            # 2a. Redimensionar o Feature Map A_k para o tamanho da entrada One-Hot
            # De (1, 16) para (4, 512) - Note que L_final é 16 se a sequência original é 512
            feature_map_k = activations[k].unsqueeze(0).unsqueeze(0) # (1, 1, L_target)
            
            # Redimensiona para o comprimento final (L=512)
            # F.interpolate espera (B, C, L)
            mask_resized_1d = F.interpolate(
                feature_map_k, 
                size=input_one_hot_B_C_L.size(2), 
                mode='linear', 
                align_corners=False
            ) # (1, 1, 512)

            # 2b. Normalizar a máscara (opcional, mas comum no Score-CAM)
            mask_min = mask_resized_1d.min()
            mask_max = mask_resized_1d.max()
            if mask_max > mask_min:
                mask_normalized = (mask_resized_1d - mask_min) / (mask_max - mask_min)
            else:
                mask_normalized = torch.zeros_like(mask_resized_1d)

            # 2c. Aplicar Mascaramento: Mascarar a entrada One-Hot
            # A máscara 1D (1, 1, 512) precisa ser expandida para (1, 4, 512)
            mask_expanded = mask_normalized.expand_as(input_one_hot_B_C_L)
            
            masked_input_B_C_L = input_one_hot_B_C_L * mask_expanded
            
            # 2d. Forward Pass da Entrada Mascarada
            # O modelo espera o input one-hot aqui. Precisamos de uma forward proxy
            # que aceite o tensor one-hot B_C_L (4x512) e ignore o OneHotEmbedding inicial.
            
            # ATENÇÃO: Score-CAM requer a reavaliação do modelo a partir do ponto A_k.
            # Como o mascaramento é feito na entrada (X), precisamos de um forward proxy:
            
            # CUIDADO COM O DISPOSITIVO
            masked_input_B_C_L = masked_input_B_C_L.to(device)
            
            # Passa a entrada one-hot mascarada pelo modelo a partir de conv1
            score_output = self._forward_from_one_hot(masked_input_B_C_L, self.model)
            
            # 2e. Obter o Score e Armazenar o Peso alpha_k
            # Obtém o score (logit) da classe alvo
            score_c = score_output[0, target_class]
            
            # O peso alpha_k é o score
            weights[k] = score_c

        # 3. Gerar o Mapa de Calor
        
        # Combina os Feature Maps (A_k) com os pesos (alpha_k)
        # activations.shape é (C, L), weights.shape é (C,)
        # Reshape weights para (C, 1) para multiplicação broadcast
        weights_reshaped = weights.unsqueeze(1) 
        
        # Combinação: sum_k (alpha_k * A_k)
        cam = torch.sum(weights_reshaped * activations, dim=0).detach().cpu().numpy() # (L_target,)

        # 4. Aplica ReLU (Score-CAM padrão não usa, mas é bom para visualizar importância positiva)
        # Usamos ReLU aqui para manter o resultado em uma faixa de importância positiva.
        cam = np.maximum(cam, 0) 
        
        # 5. Normalização (0 a 1)
        cam = cam - np.min(cam)
        cam = cam / (np.max(cam) + 1e-8) 

        # 6. Redimensionar para o comprimento original (L=512)
        original_seq_len = input_tensor_indices.size(1)
        
        # Redimensionamento 1D (tratando como (L_target, 1) -> (L_original, 1))
        cam_2d_L_x_1 = cam[:, np.newaxis] 
        cam_resized = cv2.resize(
            cam_2d_L_x_1,
            dsize=None,
            fx=1,
            fy=original_seq_len / cam_2d_L_x_1.shape[0],
            interpolation=cv2.INTER_LINEAR
        ).flatten()
        
        return cam_resized

    def _forward_from_one_hot(self, x_one_hot, model):
        """
        Executa o forward pass a partir do tensor One-Hot (ignorando model.one_hot_layer).
        """
        # x_one_hot: (B, C, L)
        
        x = model.conv1(x_one_hot)
        x = model.bn1(x)
        x = model.relu(x)
        x = model.maxpool(x)
        
        x = model.layer1(x)
        x = model.layer2(x)
        x = model.layer3(x)
        
        # O forward deve ser feito ATÉ a camada FC para obter o score final
        x = model.layer4(x)
        
        # Global Pooling (Max ou Avg, dependendo do que o seu modelo usa)
        x = F.max_pool1d(x, kernel_size=x.size(2)) 
        
        x = torch.flatten(x, 1)
        x = model.dropout(x)
        x = model.fc(x)
        
        return x

# --- Função de Visualização do Grad-CAM com a Imagem Original ---
def visualize_heatmap(original_seq_one_hot_np, 
                      heatmap, 
                      original_sequence_string, 
                      title="Grad-CAM 1D"):
    """
    Visualiza a imagem One-Hot da sequência de DNA e sobrepõe o heatmap do Grad-CAM.

    Args:
        original_seq_one_hot_np (np.ndarray): Matriz One-Hot (512, 4) para a imagem de fundo.
        heatmap (np.ndarray): O heatmap do Grad-CAM (seq_len,) com valores entre 0 e 1.
        original_sequence_string (str): A string original da sequência de DNA (para títulos/contexto).
        title (str): Título da plotagem.
    """
    # Certifique-se de que a matriz One-Hot está no formato (L, C)
    if original_seq_one_hot_np.shape[1] != 4:
        print("A matriz One-Hot deve ter 4 colunas para as bases (A, C, G, T).")
        return

    seq_len = original_seq_one_hot_np.shape[0]

    # Preparar a imagem One-Hot (preto e branco)
    image_data_display = (original_seq_one_hot_np * 255).astype(np.uint8)

    # Redimensionar para melhor visualização (similar à show_binary_data_as_image)
    scale_factor_width = 100
    new_width = image_data_display.shape[1] * scale_factor_width
    new_height = image_data_display.shape[0]

    resized_image = cv2.resize(image_data_display, (new_width, new_height), interpolation=cv2.INTER_NEAREST)

    # Redimensionar o heatmap para as dimensões visuais da imagem
    # O heatmap é 1D, então redimensionamos apenas ao longo do eixo de comprimento (altura).
    heat_array = np.array(heatmap)
    heat_array = heat_array[:, np.newaxis]
    heat_array = np.tile(heat_array, (1, 4))
    resized_heatmap = cv2.resize(heat_array, (new_width, new_height), interpolation=cv2.INTER_NEAREST)
    # Plotagem
    plt.figure(figsize=(new_width / 100, new_height / 100 + 1)) # Ajusta o tamanho da figura

    # Plotar a imagem One-Hot de fundo
    plt.imshow(resized_image, cmap='gray', vmin=0, vmax=255)
    
    # Sobrepor o heatmap
    plt.imshow(resized_heatmap, cmap='jet', alpha=0.5,vmin=0, vmax=1) # 'jet' é uma boa escolha de cmap para heatmaps

    plt.title(title, fontsize=16)
    
    feature_labels = ['A', 'C', 'G', 'T']
    plt.xticks(
        np.arange(len(feature_labels)) * scale_factor_width + (scale_factor_width / 2),
        feature_labels,
        fontsize=12
    )
    plt.yticks([]) # Remove os ticks do eixo y (comprimento da sequência)
    
    # Adiciona uma grade para delinear as bases (colunas)
    for i in range(1, original_seq_one_hot_np.shape[1]):
        plt.axvline(x=i * scale_factor_width, color='red', linestyle='-', linewidth=0.5)

    plt.colorbar(label='Importância (Grad-CAM)')
    plt.show()
def get_class_label(mapping_csv_path: str, target_label_0based: int,col_name = ""):
    """
    Carrega o CSV de mapeamento e inspeciona o rótulo do modelo (0-based).
    
    Args:
        mapping_csv_path (str): Caminho para o arquivo CSV de mapeamento.
        target_label_0based (int): O rótulo do modelo a ser inspecionado (0-based).
    """
    if not os.path.exists(mapping_csv_path):
        print(f"ERRO: Arquivo não encontrado no caminho: {mapping_csv_path}")
        return

    # O ID no CSV é 1-based, então somamos 1 ao rótulo do modelo.
    target_id_1based = target_label_0based + 1

    try:
        # Carrega o CSV de mapeamento (assumindo colunas 'id' e 'col_name')
        df = pd.read_csv(mapping_csv_path)

        # Filtra pela coluna 'id'
        target_row = df[df['id'] == target_id_1based]

        if not target_row.empty:
            return target_row[col_name].iloc[0]
        else:
            return "Not Found"

    except Exception as e:
        print(f"ERRO: Falha ao carregar ou processar o CSV: {e}")

CAM_TYPE = "GRAD"

import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
# Importe as classes GradCAM1D e ScoreCAM1D e a função visualize_heatmap daqui

# --- Função Auxiliar de Visualização (Baseada na sua última análise) ---
# Esta lógica é movida para dentro da função principal para evitar dependências externas complexas,
# mas se você a tiver em analize_funcs, pode simplesmente chamá-la.

def _plot_heatmap_comparison(ax, original_seq_one_hot_np, heatmap, title):
    """Lógica de plotagem para um único heatmap (baseada na última pergunta do usuário)."""
    
    # 1. Preparar a imagem One-Hot (Fundo)
    image_data_display = (original_seq_one_hot_np * 255).astype(np.uint8)

    scale_factor_width = 100
    new_width = image_data_display.shape[1] * scale_factor_width
    new_height = image_data_display.shape[0]

    # Interpolação NEAREST para o fundo (mantém pixels puros)
    resized_image = cv2.resize(image_data_display, (new_width, new_height), interpolation=cv2.INTER_NEAREST)

    # 2. Preparar e Redimensionar o Heatmap
    heat_array = np.array(heatmap)
    heat_array = heat_array[:, np.newaxis]
    
    # Tiling: transforma (L, 1) em (L, 4)
    heat_array = np.tile(heat_array, (1, 4))
    
    # Interpolação LINEAR para o heatmap (mantém transições suaves)
    resized_heatmap = cv2.resize(heat_array, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    
    # 3. Plotagem
    ax.imshow(resized_image, cmap='gray', vmin=0, vmax=255)
    ax.imshow(resized_heatmap, cmap='jet', alpha=0.5, vmin=0, vmax=1) 

    ax.set_title(title, fontsize=12)
    
    feature_labels = ['A', 'C', 'G', 'T']
    ax.set_xticks(
        np.arange(len(feature_labels)) * scale_factor_width + (scale_factor_width / 2),
        feature_labels,
        fontsize=10
    )
    ax.set_yticks([]) 
    ax.tick_params(axis='x', length=0) # Remove ticks, mantém labels
    
    # Adiciona a grade
    for i in range(1, original_seq_one_hot_np.shape[1]):
        ax.axvline(x=i * scale_factor_width, color='red', linestyle='-', linewidth=0.5)

    return ax.images[-1] # Retorna o objeto Imagem para a ColorBar

# --- Função Principal ---

def visualize_heat_map_examples_by_class(model,
                                         target_layer,
                                         class_true_id,
                                         target_pred_id,
                                         examples,
                                         score_principal_confusao,
                                         cam_type,
                                         device, # É necessário passar o dispositivo
                                         mapping_csv_path=None,
                                         col_name=""): # Opcional: para nomes de classe
    
    # Inicializa o gerador CAM apropriado
    if cam_type == "GRAD":
        generator = GradCAM1D(model, target_layer)
    elif cam_type == "SCORE":
        generator = ScoreCAM1D(model, target_layer)
    else:
        raise ValueError("cam_type deve ser 'GRAD' ou 'SCORE'.")

    # Obtém nomes legíveis, se houver mapeamento
    def get_name(class_id,mapping_csv_path,col_name):
        if mapping_csv_path:
            label = get_class_label(mapping_csv_path = mapping_csv_path, 
                            target_label_0based = class_id,
                            col_name = col_name)
            return f"{class_id} - {label}"
        return f'ID {class_id}'
        
    true_name = get_name(class_true_id,mapping_csv_path,col_name)
    pred_name = get_name(target_pred_id,mapping_csv_path,col_name)

    print(f"\n--- Análise de Confusão: Verdadeiro ({class_true_id})={true_name} -> Predito ({target_pred_id})={pred_name} | Score Confusão: {score_principal_confusao:.4f} ---")
    
    
    # Determina o tamanho da figura: 2 colunas (Verdadeiro vs Predito) por exemplo.
    num_examples = len(examples)
    fig_width = 12 
    fig_height = num_examples * 6
    
    # Cria a figura e a grade de subplots
    fig, axs = plt.subplots(num_examples, 2, figsize=(fig_width, fig_height), squeeze=False)

    main_title = (
        f"Comparação de Foco {cam_type}-CAM para Confusão\n"
        f"Verdadeiro: {true_name} vs. Predito: {pred_name} (Score Confusão: {score_principal_confusao:.4f})"
    )
    fig.suptitle(main_title, fontsize=16, y=1.02) # y=1.02 coloca o título ligeiramente acima do topo
    
    # Variável para a ColorBar
    cbar_mappable = None

    for i, example in enumerate(examples):
        # 1. Preparação da Sequência (Adicionar Batch e Mover para Device)
        input_tensor_device = example.unsqueeze(0).to(device)
        
        # 2. Geração dos Heatmaps
        # Heatmap da Classe Predita (Foco do Modelo)
        heatmap_pred = generator(input_tensor_device, target_class=target_pred_id)
        
        # Heatmap da Classe Verdadeira (Foco Esperado)
        heatmap_target = generator(input_tensor_device, target_class=class_true_id)
        is_padding = input_tensor_device.sum(dim=1) == 0  # Shape (1, L)
        heatmap_target[is_padding] = 0
        heatmap_pred[is_padding] = 0
        # 3. Geração da Imagem One-Hot de Fundo
        with torch.no_grad():
            # Assumindo que 'example' é o tensor de índices (L,)
            one_hot_original = model.one_hot_layer(input_tensor_device)
            # Retorna para (L, C) e NumPy
            one_hot_original_np = one_hot_original[0].permute(1, 0).cpu().numpy() 

        # 4. Plotagem (Lado a Lado)
        
        # --- Lado Esquerdo: Classe Verdadeira ---
        ax_true = axs[i, 0]
        cbar_mappable = _plot_heatmap_comparison(
            ax_true, 
            one_hot_original_np, 
            heatmap_target, 
            f"Exemplo {i+1} | (Verdadeiro: {true_name})"
        )

        # --- Lado Direito: Classe Predita ---
        ax_pred = axs[i, 1]
        _plot_heatmap_comparison(
            ax_pred, 
            one_hot_original_np, 
            heatmap_pred, 
            f"Exemplo {i+1} | (Predito: {pred_name})"
        )

    # 5. Adicionar ColorBar Global
    if cbar_mappable is not None:
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7]) # [left, bottom, width, height]
        fig.colorbar(cbar_mappable, cax=cbar_ax, label=f'Importância {cam_type}-CAM (0 a 1)')

    plt.tight_layout(rect=[0, 0, 0.9, 1]) # Ajusta layout para a colorbar
    plt.show()
        
    