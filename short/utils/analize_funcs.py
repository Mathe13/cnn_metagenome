import pandas as pd
import os


import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from tqdm.auto import tqdm
import os
import pandas as pd
from math import ceil
import torch
import numpy as np
from tqdm.auto import tqdm
from typing import List, Dict, Any, Tuple


def plot_confusion_matrix_insights(model, test_loader, device, num_classes, top_k_confusion=10, mapping_csv_path=None, col_name = ""):
    """
    Calcula a Matriz de Confusão e visualiza as Top-K classes mais confundidas,
    mostrando o ID numérico e o nome da espécie (rótulo) combinados.
    
    Args:
        model (nn.Module): Modelo treinado.
        test_loader (DataLoader): DataLoader para o conjunto de teste.
        device (torch.device): Dispositivo ('cuda' ou 'cpu').
        num_classes (int): Número total de classes.
        top_k_confusion (int): Número de classes mais confundidas a serem destacadas.
        mapping_csv_path (str): Caminho para o arquivo CSV contendo colunas 'species' e 'id' (ID 1-based).
    """
    print("\n--- Coletando Previsões do Modelo para a Matriz de Confusão ---")
    
    # 0. Lógica de Mapeamento de Labels (NOVO)
    label_to_full_name = {}
    if mapping_csv_path and os.path.exists(mapping_csv_path):
        try:
            mapping_df = pd.read_csv(mapping_csv_path)
            
            # Ajusta o ID para 0-based
            mapping_df['model_label'] = mapping_df['id'] - 1
            
            # Combina ID e Nome da Espécie
            mapping_df['full_label'] = mapping_df['model_label'].astype(str) + ' - ' + mapping_df[col_name]
            
            # Cria o dicionário de mapeamento: {rótulo_modelo (0-based): ID - Nome da Espécie}
            label_to_full_name = mapping_df.set_index('model_label')['full_label'].to_dict()
            print(f"Mapeamento de espécies carregado e ajustado de '{mapping_csv_path}'.")
        except Exception as e:
            print(f"Aviso: Falha ao carregar ou processar o CSV de mapeamento. Exibindo IDs numéricos. Erro: {e}")
            
    model.eval()
    all_labels = []
    all_preds = []

    # Coleta de rótulos e previsões
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Coletando Previsões"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    # Converte listas para arrays NumPy
    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)

    # 1. Cálculo da Matriz de Confusão Normalizada
    cm = confusion_matrix(y_true, y_pred, labels=range(num_classes))
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_normalized = np.nan_to_num(cm_normalized)

    # 2. Análise Top-K de Confusão
    
    confused_pairs = []
    for i in range(num_classes):
        for j in range(num_classes):
            if i != j: # Ignora acertos (diagonal)
                score = cm_normalized[i, j]
                if score > 0:
                    confused_pairs.append((score, i, j))

    confused_pairs.sort(key=lambda x: x[0], reverse=True)
    top_k_confused = confused_pairs[:top_k_confusion]

    # --- Visualização ---

    plt.figure(figsize=(15, 6))

    # Gráfico 1: Matriz Completa (para contexto)
    plt.subplot(1, 2, 1)
    plt.imshow(cm_normalized, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Matriz de Confusão (Normalizada Completa)")
    plt.colorbar()
    plt.xlabel("Rótulo Predito")
    plt.ylabel("Rótulo Verdadeiro")

    # Gráfico 2: Top-K Confusões (Foco principal)
    plt.subplot(1, 2, 2)
    
    labels_bar = []
    scores_bar = []
    
    for score, i, j in top_k_confused:
        # Obtém o rótulo completo (ID - Espécie) ou ID numérico se o mapeamento falhar
        true_label = label_to_full_name.get(i, f'ID {i}')
        predicted_label = label_to_full_name.get(j, f'ID {j}')
        
        labels_bar.append(f"Verdadeiro: {true_label} -> Predito: {predicted_label}")
        scores_bar.append(score)
    
    y_pos = np.arange(len(labels_bar))
    plt.barh(y_pos, scores_bar, align='center', color='darkred')
    plt.yticks(y_pos, labels_bar, fontsize=8) 
    plt.gca().invert_yaxis()
    plt.xlabel('Taxa de Confusão (Normalizada)')
    plt.title(f"Top {top_k_confusion} Pares de Classes Mais Confundidos")
    plt.tight_layout()
    plt.show()
    
    return top_k_confused

def inspect_csv(mapping_csv_path: str, target_label_0based: int,col_name = ""):
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
            print("\n" + "="*70)
            print(f"Inspeção do Rótulo Confundido (Modelo ID: {target_label_0based})")
            print("="*70)
            
            # Exibe todas as informações da linha
            print(target_row.transpose())
            
            print("\n--- Informações Chave ---")
            print(f"Rótulo no Modelo (0-based): {target_label_0based}")
            print(f"ID no CSV (1-based): {target_id_1based}")
            print(f"Nome da Espécie: {target_row[col_name].iloc[0]}")
            print("="*70)
        else:
            print(f"Nenhuma entrada encontrada no CSV para o ID: {target_id_1based}")

    except Exception as e:
        print(f"ERRO: Falha ao carregar ou processar o CSV: {e}")

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


# Define o tipo de dado da entrada para clareza
ConfusedPairInfo = Tuple[np.float64, int, int]

def get_confused_class_samples(
    confused_pairs_info: List[ConfusedPairInfo],
    test_loader: torch.utils.data.DataLoader,
    device: torch.device,
    n_examples_per_class: int = 5
) -> List[Dict[str, Any]]:
    """
    Percorre o DataLoader para coletar 'n' tensores de input para cada classe
    (rótulo verdadeiro) presente nos Top-K pares de confusão fornecidos.

    Args:
        confused_pairs_info: Lista de tuplas [(score, true_id, pred_id)] dos pares mais confundidos.
        test_loader: DataLoader para o conjunto de teste (assume que já contém labels).
        device: Dispositivo ('cuda' ou 'cpu').
        n_examples_per_class: Número de tensores de exemplo (inputs) para coletar por classe verdadeira.

    Returns:
        Lista estruturada com a taxa de confusão e os tensores de input de exemplo.
    """
    
    # 1. Identificar as classes alvo únicas (True IDs) e seus scores de confusão
    
    # Mapeamento: True ID -> [Pred ID, Score, Lista de Exemplos]
    # Usamos um dict para garantir que cada True ID seja processado
    target_classes_map: Dict[int, Dict[str, Any]] = {}
    
    for score, true_id, pred_id in confused_pairs_info:
        true_id = int(true_id)
        pred_id = int(pred_id)
        
        # Inicializa a estrutura para o True ID, se ainda não existir
        if true_id not in target_classes_map:
            target_classes_map[true_id] = {
                "id_true": true_id,
                "total_confused_score": 0.0,
                "examples": [],
                "target_pred_id": pred_id, # Guarda um dos IDs mais confundidos
                "score": score # Guarda o score da principal confusão
            }
            
        # Acumula o score de confusão (se o par for o mesmo)
        target_classes_map[true_id]["total_confused_score"] += score
    
    # 2. Coleta de Exemplos do DataLoader
    print("\nPasso 1: Coletando tensores de input para as classes alvo...")
    
    classes_collected_counts: Dict[int, int] = {cid: 0 for cid in target_classes_map.keys()}
    all_targets_met = False

    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Coletando Amostras"):
            inputs, labels = inputs.to(device), labels.to(device)

            if all_targets_met:
                break
                
            for sample_idx in range(inputs.size(0)):
                true_id = labels[sample_idx].item()
                
                # Verifica se a amostra pertence a uma das classes alvo e se ainda precisamos de exemplos
                if true_id in target_classes_map and classes_collected_counts[true_id] < n_examples_per_class:
                    
                    # Coleta o tensor de input, movendo-o para CPU (para evitar problemas de memória na GPU)
                    input_tensor = inputs[sample_idx].cpu()
                    target_classes_map[true_id]["examples"].append(input_tensor)
                    classes_collected_counts[true_id] += 1
                    
                    # Checa se todos os alvos foram encontrados para interromper a iteração
                    if all(count >= n_examples_per_class for count in classes_collected_counts.values()):
                        all_targets_met = True
                        break


    # 3. Formatar a Saída Final
    print("\nPasso 2: Formatando os resultados finais.")
    final_output: List[Dict[str, Any]] = []

    # Converte o mapa de classes de volta para uma lista estruturada
    for cid, data in target_classes_map.items():
        final_output.append({
            "score_principal_confusao": data["score"],
            "class_true": {
                "id": cid,
                "target_pred_id": data["target_pred_id"],
                "examples": data["examples"]
            }
        })
        
    return final_output

def tensor_to_img(tensor, emb_model):
    single_sequence_ids = tensor
    
    batched_input = single_sequence_ids.unsqueeze(0)
    # --- 4. Apply the Model ---
    output_tensor = emb_model(batched_input)
    return torch.transpose(output_tensor[0],0,1).numpy()


