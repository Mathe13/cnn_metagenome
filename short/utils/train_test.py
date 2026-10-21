import torch
import torch.nn as nn  # Ensure nn is imported for nn.Module and nn.utils
import torch.nn.functional as F  # Ensure F is imported for activations etc.
from tqdm.auto import tqdm
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.metrics import f1_score, accuracy_score
import random
import numpy as np
import pandas as pd  # <--- Esta é a linha que faltou
# Add imports for Mixed Precision Training

from torch.amp import GradScaler, autocast


def print_loss_info(batch_idx, inputs, labels):
    print(
        f"ATENÇÃO: Loss inválida (nan/inf) no lote {batch_idx+1}. Pulando a atualização de pesos."
    )
    print("Inputs problemáticos:", inputs)
    print("Labels problemáticos:", labels)


def salvar_lote_invalido(
    batch_idx, inputs, labels, num_classes=13683, arquivo="lotes_invalidos_debug.txt"
):
    """
    Salva informações de um lote inválido (com loss nan/inf) em um arquivo de texto,
    incluindo uma verificação detalhada de integridade.

    Args:
        batch_idx (int): Índice do lote.
        inputs (torch.Tensor): Tensor de entrada.
        labels (torch.Tensor): Tensor de rótulos.
        num_classes (int, optional): O número total de classes para validar o intervalo dos rótulos.
        arquivo (str): Nome do arquivo onde os dados serão salvos.
    """
    import torch

    # --- Início das Verificações de Integridade ---
    report = []
    report.append("--- Verificação de Integridade ---")

    # 1. Checagem dos Inputs
    report.append(">> Inputs:")
    report.append(f"   - Shape: {inputs.shape}")
    report.append(f"   - Dtype: {inputs.dtype}")
    report.append(f"   - Contém NaN?: {torch.isnan(inputs).any().item()}")
    report.append(f"   - Contém Inf?: {torch.isinf(inputs).any().item()}")

    # Evita erro se o tensor estiver vazio
    if inputs.numel() > 0:
        report.append(
            f"   - Min / Max: {inputs.min().item():.2f} / {inputs.max().item():.2f}"
        )
    else:
        report.append("   - Min / Max: (Tensor vazio)")

    # 2. Checagem dos Labels
    report.append("\n>> Labels:")
    report.append(f"   - Shape: {labels.shape}")
    report.append(f"   - Dtype: {labels.dtype}")
    report.append(f"   - Contém NaN?: {torch.isnan(labels).any().item()}")
    report.append(f"   - Contém Inf?: {torch.isinf(labels).any().item()}")

    if labels.numel() > 0:
        report.append(f"   - Min / Max: {labels.min().item()} / {labels.max().item()}")
        # 3. Checagem CRÍTICA: Rótulos fora do intervalo de classes
        if num_classes is not None:
            # Verifica se algum rótulo é negativo ou maior/igual ao número de classes
            invalid_labels_mask = (labels < 0) | (labels >= num_classes)
            if invalid_labels_mask.any():
                problematic_indices = torch.where(invalid_labels_mask)[0].cpu().numpy()
                report.append(
                    f"   - ALERTA: Rótulos fora do intervalo [0, {num_classes-1}] encontrados!"
                )
                report.append(
                    f"   - Índices no lote com problemas: {problematic_indices}"
                )
            else:
                report.append(f"   - Verificação de intervalo [0, {num_classes-1}]: OK")
    else:
        report.append("   - Min / Max: (Tensor vazio)")

    report_str = "\n".join(report)
    # --- Fim das Verificações de Integridade ---

    # Salva o relatório e os dados no arquivo
    with open(arquivo, "a") as f:
        f.write(f"\n--- Lote Inválido {batch_idx+1} ---\n")
        f.write(report_str)
        f.write("\n\n--- Dados do Lote ---\n")
        f.write("Inputs:\n")
        f.write(str(inputs.cpu().numpy()))
        f.write("\n\nLabels:\n")
        f.write(str(labels.cpu().numpy()))
        f.write("\n-----------------------------\n")


def set_bn_eval_for_freezed_layers(model, freezed_layers_names, debug=False):
    """
    Define o modo de avaliação (eval) para as camadas de Batch Normalization (BN)
    dentro de módulos congelados.

    Args:
        model (nn.Module): O modelo PyTorch.
        freezed_layers_names (list): Lista de strings com os nomes das camadas que
                                      estão congeladas. Ex: ['layer1', 'layer2'].
    """
    model.train()  # Garante que o modelo está em modo de treinamento

    # Nomes completos das camadas congeladas no OptimizedModule
    full_freezed_names = [f"_orig_mod.{name}" for name in freezed_layers_names]

    # Adiciona a camada inicial `conv1` e `bn1` se o modelo for ResNet
    full_freezed_names.append("_orig_mod.conv1")
    full_freezed_names.append("_orig_mod.bn1")

    # Itera sobre todos os módulos do modelo
    for name, module in model.named_modules():
        # Verifica se o módulo é uma camada de BatchNorm
        if isinstance(module, nn.BatchNorm1d):
            # Verifica se o nome da camada de BatchNorm pertence a um módulo congelado
            is_freezed_bn = any(
                name.startswith(full_name) for full_name in full_freezed_names
            )

            if is_freezed_bn:
                module.eval()
                # Opcional: imprimir para confirmação
                if debug:
                    print(
                        f"Camada de BatchNorm '{name}' (congelada) está em modo de avaliação."
                    )


def train_validate_model(
    model,
    train_loader,
    val_loader,
    optimizer,
    criterion,
    num_epochs,
    device,
    best_model_path,
    num_classes,
    lr_scheduler=None,
    grad_clip_norm=None,
    freezed_layers_names=None,
    has_freezed=False,
    patience=None,
    checkpoint_path="last_checkpoint.pth",
    save_interval_batches=None,
    scheduler_by_epoch=True,
    accumulation_steps=1,
):
    """
    Realiza o treinamento e a validação de um modelo PyTorch com checkpointing.
    Adiciona lógica para salvar o lote, retomar o treinamento e pular os lotes já processados.
    """
    best_val_loss = np.inf
    train_losses, train_f1s, train_accuracies = [], [], []
    val_losses, val_f1s, val_accuracies = [], [], []
    scaler = GradScaler()
    patience_counter = 0
    start_epoch = 0
    start_batch_idx = 0
    total_steps = len(train_loader) * num_epochs
    current_step = 0

    if freezed_layers_names == None:
        freezed_layers_names = []

    # --- Lógica para Carregar Checkpoint e Continuar o Treino ---
    if os.path.exists(checkpoint_path):
        print(
            f"*** Checkpoint encontrado! Retomando o treinamento de '{checkpoint_path}' ***"
        )
        checkpoint = torch.load(checkpoint_path,weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        # Load Scaler state (Important for AMP!)
        if "scaler_state_dict" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler_state_dict"])
        start_epoch = checkpoint["epoch"]
        start_batch_idx = checkpoint.get("batch_idx", 0)
        best_val_loss = checkpoint["best_val_loss"]
        current_step = checkpoint.get("current_step", 0)  # Carrega o current_step salvo

        try:
            # 1. Python Random
            if "random_state" in checkpoint:
                random.setstate(checkpoint["random_state"])

            # 2. Numpy Random
            if "numpy_random_state" in checkpoint:
                np.random.set_state(checkpoint["numpy_random_state"])

            # 3. PyTorch CPU
            if "torch_rng_state" in checkpoint:
                torch.set_rng_state(checkpoint["torch_rng_state"])

            # 4. PyTorch CUDA
            if "torch_cuda_rng_state" in checkpoint:
                # If you switched from multi-GPU to single-GPU (or vice versa), this might fail.
                # The check below prevents crashing if the device count changed.
                try:
                    torch.cuda.set_rng_state_all(checkpoint["torch_cuda_rng_state"])
                except Exception as e:
                    print(
                        f"Warning: Failed to restore CUDA RNG state (Device count changed?): {e}"
                    )

            print("Random states (Python, Numpy, Torch CPU/GPU) restored successfully.")

        except Exception as e:
            print(
                f"Warning: Could not restore some random states. Augmentations/Shuffling may differ. Error: {e}"
            )

        # Carrega as métricas do checkpoint
        if "train_losses" in checkpoint:
            train_losses = checkpoint["train_losses"]
            train_f1s = checkpoint["train_f1s"]
            train_accuracies = checkpoint["train_accuracies"]
            val_losses = checkpoint["val_losses"]
            val_f1s = checkpoint["val_f1s"]
            val_accuracies = checkpoint["val_accuracies"]

        if lr_scheduler is not None:
            # Carrega o estado do scheduler
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler_state_dict"])
            # Restaura last_epoch para garantir a continuidade
            # Isso é crucial para CosineAnnealingLR e outros schedulers per-batch
            if "scheduler_last_epoch" in checkpoint:
                lr_scheduler.last_epoch = checkpoint["scheduler_last_epoch"]
                print(
                    f"Estado do scheduler restaurado para o passo {lr_scheduler.last_epoch + 1}."
                )

        print(f"Retomando da Época {start_epoch}, Lote {start_batch_idx}.")

    print("--- Treinamento Iniciado ---")

    for epoch in range(start_epoch, num_epochs):
        model.train()
        train_iter = iter(train_loader)

        if epoch == start_epoch and start_batch_idx > 0:
            print(
                f"Avançando o DataLoader para o lote {start_batch_idx} da época {start_epoch + 1}..."
            )
            try:
                for _ in tqdm(range(start_batch_idx), desc="Avançando Lotes"):
                    next(train_iter)
                print("Avanço concluído com sucesso.")
            except StopIteration:
                print(
                    f"Aviso: O DataLoader terminou antes de alcançar o lote {start_batch_idx}. Começando do início da época."
                )
                start_batch_idx = 0

        if has_freezed:
            set_bn_eval_for_freezed_layers(model, freezed_layers_names, debug=(epoch == start_epoch))

        running_loss = 0.0
        all_labels_train = []
        all_predictions_train = []
        total_train_samples = 0

        print(f"\n--- Época {epoch+1}/{num_epochs} - Treinamento ---")

        tqdm_total = (
            len(train_loader) - start_batch_idx
            if len(train_loader) is not None
            else None
        )
        
        remaining = len(train_loader) - start_batch_idx if len(train_loader) is not None else 0
        tqdm_total = max(1, remaining) # Evita crash se for 0 ou negativo
        
        train_loop = tqdm(train_iter, total=tqdm_total, desc=f"Treino Época {epoch+1}")

        optimizer.zero_grad()
        for batch_idx, (inputs, labels) in enumerate(train_loop, start=start_batch_idx):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            # [FIXED] Mixed Precision Context
            with autocast(device_type=device.type):  # or device_type='cuda'
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                # [NEW] Normalize loss for gradient accumulation
                loss = loss / accumulation_steps

            if torch.isinf(loss) or torch.isnan(loss):
                print_loss_info(batch_idx, inputs, labels)
                salvar_lote_invalido(batch_idx, inputs, labels)
                optimizer.zero_grad()
                continue

            batch_size = inputs.size(0)
            total_train_samples += batch_size

            scaler.scale(loss).backward()

            # [NEW] Gradient Accumulation Step
            if (batch_idx + 1) % accumulation_steps == 0:
                if grad_clip_norm is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)

                scale_before = scaler.get_scale()

                scaler.step(optimizer)
                scaler.update()

                scale_after = scaler.get_scale()

                optimizer_stepped = scale_after >= scale_before

                if optimizer_stepped:
                    if not scheduler_by_epoch:
                        if lr_scheduler is not None and current_step < total_steps:
                            lr_scheduler.step()
                            current_step += 1
                    optimizer.zero_grad()
                else:
                    optimizer.zero_grad()

            running_loss += loss.item() * batch_size * accumulation_steps

            _, predicted = torch.max(outputs.detach(), 1)
            all_labels_train.extend(labels.cpu().numpy())
            all_predictions_train.extend(predicted.cpu().numpy())

            train_loop.set_postfix(loss=running_loss / total_train_samples)

            if (
                save_interval_batches is not None
                and (batch_idx + 1) % save_interval_batches == 0
            ):
                print(
                    f"\nSalvando checkpoint na época {epoch+1}, lote {batch_idx+1}..."
                )
                checkpoint = {
                    "epoch": epoch,
                    "batch_idx": batch_idx + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val_loss": best_val_loss,
                    "lr_scheduler_state_dict": (
                        lr_scheduler.state_dict() if lr_scheduler else None
                    ),
                    "scheduler_last_epoch": (
                        lr_scheduler.last_epoch if lr_scheduler else None
                    ),
                    "current_step": current_step,
                    "train_losses": train_losses,
                    "train_f1s": train_f1s,
                    "train_accuracies": train_accuracies,
                    "val_losses": val_losses,
                    "val_f1s": val_f1s,
                    "val_accuracies": val_accuracies,
                    "random_state": random.getstate(),
                    "numpy_random_state": np.random.get_state(),
                    "torch_rng_state": torch.get_rng_state(),
                    "torch_cuda_rng_state": torch.cuda.get_rng_state_all(),
                }
                torch.save(checkpoint, checkpoint_path)
                print("Checkpoint salvo com sucesso.")

        start_batch_idx = 0

        # Scheduler step (if per-epoch)
        if scheduler_by_epoch and lr_scheduler is not None:
            lr_scheduler.step()

        avg_train_loss = running_loss / total_train_samples
        train_f1 = f1_score(
            all_labels_train, all_predictions_train, average="weighted", zero_division=0
        )
        train_acc = accuracy_score(all_labels_train, all_predictions_train)

        train_losses.append(avg_train_loss)
        train_f1s.append(train_f1)
        train_accuracies.append(train_acc)

        print(
            f"Resultados Treinamento: Loss = {avg_train_loss:.4f}, F1-Score = {train_f1:.4f}, Acurácia = {train_acc:.4f}"
        )

        model.eval()
        val_loss, all_labels_val, all_predictions_val = 0.0, [], []
        total_val_samples = 0
        print(f"\n--- Época {epoch+1}/{num_epochs} - Validação ---")
        val_loop = tqdm(val_loader, desc=f"Validação Época {epoch+1}")
        with torch.no_grad():
            for batch_idx, (inputs_val, labels_val) in enumerate(val_loop):
                inputs_val, labels_val = inputs_val.to(device), labels_val.to(device)
                with autocast(device_type=device.type):
                    outputs_val = model(inputs_val)
                    loss_val = criterion(outputs_val, labels_val)

                if torch.isinf(loss_val) or torch.isnan(loss_val):
                    print_loss_info(batch_idx, inputs_val, labels_val)
                    salvar_lote_invalido(batch_idx, inputs_val, labels_val)
                    continue

                batch_size = inputs_val.size(0)
                val_loss += loss_val.item() * batch_size
                total_val_samples += batch_size

                _, predicted_val = torch.max(outputs_val.detach(), 1)
                all_labels_val.extend(labels_val.cpu().numpy())
                all_predictions_val.extend(predicted_val.cpu().numpy())

                val_loop.set_postfix(loss=val_loss / total_val_samples)

        avg_val_loss = val_loss / total_val_samples
        val_f1 = f1_score(
            all_labels_val, all_predictions_val, average="weighted", zero_division=0
        )
        val_acc = accuracy_score(all_labels_val, all_predictions_val)
        val_losses.append(avg_val_loss)
        val_f1s.append(val_f1)
        val_accuracies.append(val_acc)
        print(
            f"Resultados Validação: Loss = {avg_val_loss:.4f}, F1-Score = {val_f1:.4f}, Acurácia = {val_acc:.4f}"
        )

        print(f"\nSalvando checkpoint da época {epoch+1}...")
        checkpoint = {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_loss": best_val_loss,
            "lr_scheduler_state_dict": (
                lr_scheduler.state_dict() if lr_scheduler else None
            ),
            "scheduler_last_epoch": lr_scheduler.last_epoch if lr_scheduler else None,
            "current_step": current_step,
            "train_losses": train_losses,
            "train_f1s": train_f1s,
            "train_accuracies": train_accuracies,
            "val_losses": val_losses,
            "val_f1s": val_f1s,
            "val_accuracies": val_accuracies,
            "random_state": random.getstate(),
            "numpy_random_state": np.random.get_state(),
            "torch_rng_state": torch.get_rng_state(),
            "torch_cuda_rng_state": torch.cuda.get_rng_state_all(),
        }
        torch.save(checkpoint, checkpoint_path)
        print("Checkpoint salvo com sucesso.")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            print(
                f"*** Modelo salvo em '{best_model_path}': Nova melhor loss de validação ({best_val_loss:.4f}) ***"
            )
        elif patience is not None:
            patience_counter += 1
            print(f"Early Stopping: {patience_counter}/{patience} paciência...")

        if patience is not None and patience_counter >= patience:
            print("Treinamento interrompido devido a Early Stopping.")
            break

    print("\n--- Treinamento Completo ---")
    print(
        f"Melhor Loss de Validação: {best_val_loss:.4f}. Modelo salvo em '{best_model_path}'."
    )
    return (
        train_losses,
        train_f1s,
        train_accuracies,
        val_losses,
        val_f1s,
        val_accuracies,
    )


# Função para calcular Acurácia Top-K
def topk_accuracy(outputs, targets, k=5):
    """Calcula a precisão de acerto se o rótulo verdadeiro estiver entre os k principais."""
    with torch.no_grad():
        max_k = max(k) if isinstance(k, (list, tuple)) else k
        batch_size = targets.size(0)

        # Obtém os top-k valores e índices (dim=1 é a dimensão da classe)
        _, pred = outputs.topk(max_k, 1, True, True)
        pred = pred.t()  # Transpõe para (k, batch_size)

        # pred == targets é True onde a previsão está correta.
        # targets.view(1, -1) expande targets para (1, batch_size)
        correct = pred.eq(targets.view(1, -1).expand_as(pred))

        res = []
        for topk in k if isinstance(k, (list, tuple)) else [k]:
            correct_k = correct[:topk].float().sum()
            res.append(correct_k.mul_(100.0 / batch_size))  # Retorna em porcentagem

        return res if isinstance(k, (list, tuple)) else res[0]


def test_model(model, test_loader, criterion, device, best_model_path, num_classes):
    """
    Avalia o modelo treinado no conjunto de teste.
    Carrega o melhor modelo salvo durante o treinamento para avaliação final.

    Args:
        model (torch.nn.Module): O modelo PyTorch a ser testado.
        test_loader (torch.utils.data.DataLoader): DataLoader para os dados de teste.
        criterion (torch.nn.Module): A função de loss.
        device (torch.device): O dispositivo (CPU ou GPU) a ser usado.
        best_model_path (str): O caminho para o melhor modelo salvo.
        num_classes (int): O número total de classes para cálculo das métricas.
    """
    print("\n--- Iniciando Fase de Teste ---")

    try:
        model.load_state_dict(torch.load(best_model_path))
        print(f"Melhor modelo carregado de '{best_model_path}'.")
    except FileNotFoundError:
        print(
            f"Aviso: '{best_model_path}' não encontrado. O teste usará o modelo no seu estado atual (após a última época de treino)."
        )

    model.eval()
    test_loss = 0.0
    all_labels_test = []
    all_predictions_test = []
    total_test_samples = 0
    top5_correct = 0  # NOVO: Contador para Top-5

    test_loop = tqdm(test_loader, desc="Testando Modelo")

    with torch.no_grad():
        for batch_idx, (inputs_test, labels_test) in enumerate(test_loop):

            inputs_test, labels_test = inputs_test.to(device), labels_test.to(device)

            outputs_test = model(inputs_test)
            loss_test = criterion(outputs_test, labels_test)

            # --- NOVO: Verificação de nan/inf na loss ---
            if torch.isinf(loss_test) or torch.isnan(loss_test):
                print_loss_info(batch_idx, inputs_test, labels_test)
                salvar_lote_invalido(batch_idx, inputs_test, labels_test)
                continue  # Pula para o próximo lote

            batch_size = inputs_test.size(0)
            test_loss += loss_test.item() * batch_size
            total_test_samples += batch_size

            # test_loss += loss_test.item()
            topk_results = topk_accuracy(outputs_test, labels_test, k=[3, 5])

            _, predicted_test = torch.max(outputs_test.detach(), 1)
            all_labels_test.extend(labels_test.cpu().numpy())
            all_predictions_test.extend(predicted_test.cpu().numpy())

            test_loop.set_postfix(loss=test_loss / total_test_samples)

            # --- CÁLCULO DIRETO DA CONTAGEM TOP-K ---
            max_k = 5
            _, pred = outputs_test.topk(max_k, 1, True, True)
            pred = pred.t()
            correct = pred.eq(labels_test.view(1, -1).expand_as(pred))

            # Acumula acertos Top-5: verifica se a verdade está em qualquer um dos 5 primeiros
            top5_correct += correct[:5].reshape(-1).float().sum().item()

    avg_test_loss = test_loss / total_test_samples
    test_f1 = f1_score(
        all_labels_test, all_predictions_test, average="weighted", zero_division=0
    )
    test_accuracy = accuracy_score(all_labels_test, all_predictions_test)
    test_top5_accuracy = (top5_correct / total_test_samples) * 100.0

    print(f"\n--- Resultados Finais do Teste ---")
    print(
        f"Test Loss: {avg_test_loss:.4f}, Test F1-Score (weighted): {test_f1:.4f}, Test Accuracy: {test_accuracy:.4f}"
    )
    return {
        # "all_labels_test" : all_labels_test,
        # "all_predictions_test" : all_predictions_test,
        "avg_test_loss": avg_test_loss,
        "test_f1": test_f1,
        "test_accuracy": test_accuracy,
        "test_top5_accuracy": test_top5_accuracy,  # NOVO RETORNO
    }


def plot_metrics(
    train_losses,
    train_f1s,
    train_accuracies,
    val_losses,
    val_f1s,
    val_accuracies,
    num_epochs,
):
    """
    Gera gráficos da loss, F1-score e Acurácia (treinamento e validação) por época.
    """
    epochs_range = range(1, num_epochs + 1)

    plt.figure(figsize=(18, 5))  # Aumenta a figura para acomodar 3 subplots

    # Gráfico da Loss
    plt.subplot(1, 3, 1)  # 1 linha, 3 colunas, primeiro gráfico
    plt.plot(epochs_range, train_losses, label="Loss de Treino")
    plt.plot(epochs_range, val_losses, label="Loss de Validação")
    plt.title("Loss por Época")
    plt.xlabel("Época")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    # Gráfico do F1-score
    plt.subplot(1, 3, 2)  # 1 linha, 3 colunas, segundo gráfico
    plt.plot(epochs_range, train_f1s, label="F1-Score de Treino")
    plt.plot(epochs_range, val_f1s, label="F1-Score de Validação")
    plt.title("F1-Score por Época")
    plt.xlabel("Época")
    plt.ylabel("F1-Score (ponderado)")
    plt.legend()
    plt.grid(True)

    # Gráfico da Acurácia
    plt.subplot(1, 3, 3)  # 1 linha, 3 colunas, terceiro gráfico
    plt.plot(epochs_range, train_accuracies, label="Acurácia de Treino")
    plt.plot(epochs_range, val_accuracies, label="Acurácia de Validação")
    plt.title("Acurácia por Época")
    plt.xlabel("Época")
    plt.ylabel("Acurácia")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()
