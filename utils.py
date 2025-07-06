import torch
import torch.nn as nn
from typing import List, Tuple, Dict
from metrics import compute_all_metrics
import matplotlib.pyplot as plt
import numpy as np
from loss import ComboLoss
import torch.optim as optim
from stats import plot_metrics_evolution


def train_one_epoch(model: nn.Module, 
                    dataloader: torch.utils.data.DataLoader, 
                    optimizer: torch.optim.Optimizer, 
                    loss_fn: nn.Module, 
                    device: torch.device, 
                    num_classes: int = 4) -> float:
    """Entrena el modelo por una época.
    
    Args:
        model: Modelo a entrenar
        dataloader: DataLoader con datos de entrenamiento
        optimizer: Optimizador
        loss_fn: Función de pérdida
        device: Dispositivo (CPU/GPU)
        num_classes: Número de clases
        
    Returns:
        Pérdida promedio en la época
    """
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    for batch in dataloader:
        # Manejar diferentes formatos de batch (imágenes, máscaras, posiblemente otros datos)
        images = batch[0].to(device)
        masks = batch[1]
        
        # Preprocesar máscaras
        if masks.dim() == 4 and masks.size(1) == 1:
            masks = masks.squeeze(1)
        masks = masks.long().to(device)

        # Validar rango de las máscaras
        if masks.min() < 0 or masks.max() >= num_classes:
            raise ValueError(f"Las etiquetas están fuera de rango: {masks.min()} a {masks.max()}")
        
        # Paso forward
        outputs = model(images)
        loss = loss_fn(outputs, masks)

        # Paso backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1

    return total_loss / max(num_batches, 1)


def evaluate(model: nn.Module, 
             dataloader: torch.utils.data.DataLoader, 
             loss_fn: nn.Module, 
             device: torch.device, 
             num_classes: int = 4) -> Tuple[float, Dict[str, float]]:
    """Evalúa el modelo en el conjunto de validación con todas las métricas.
    
    Args:
        model: Modelo a evaluar
        dataloader: DataLoader con datos de validación
        loss_fn: Función de pérdida
        device: Dispositivo (CPU/GPU)
        num_classes: Número de clases
        
    Returns:
        Tuple con (pérdida promedio, diccionario con todas las métricas)
    """
    model.eval()
    total_loss = 0.0
    all_metrics = {}
    num_batches = 0
    
    with torch.no_grad():
        for batch in dataloader:
            images = batch[0].to(device)
            masks = batch[1]
            
            # Preprocesar máscaras
            if masks.dim() == 4 and masks.size(1) == 1:
                masks = masks.squeeze(1)
            masks = masks.long().to(device)

            # Validar rango de las máscaras
            if masks.min() < 0 or masks.max() >= num_classes:
                raise ValueError(f"Las etiquetas están fuera de rango: {masks.min()} a {masks.max()}")
            
            # Forward pass
            outputs = model(images)
            
            # Calcular pérdida
            total_loss += loss_fn(outputs, masks).item()
            
            # Calcular todas las métricas
            batch_metrics = compute_all_metrics(outputs, masks, num_classes)
            
            # Acumular métricas
            for key, value in batch_metrics.items():
                if key not in all_metrics:
                    all_metrics[key] = []
                all_metrics[key].append(value)
            
            num_batches += 1
    
    # Promediar métricas
    avg_metrics = {}
    for key, values in all_metrics.items():
        # Filtrar valores infinitos para HD
        if 'hd95' in key:
            finite_values = [v for v in values if v != float('inf')]
            avg_metrics[key] = np.mean(finite_values) if finite_values else float('inf')
        else:
            avg_metrics[key] = np.mean(values)
    
    avg_loss = total_loss / max(num_batches, 1)
    
    return avg_loss, avg_metrics


def show_predictions(model: nn.Module, 
                    dataloader: torch.utils.data.DataLoader, 
                    device: torch.device, 
                    num_classes: int = 4,
                    num_examples: int = 3):
    """Muestra ejemplos de predicciones del modelo.
    
    Args:
        model: Modelo entrenado
        dataloader: DataLoader con datos
        device: Dispositivo (CPU/GPU)
        num_classes: Número de clases
        num_examples: Número de ejemplos a mostrar
    """
    model.eval()
    
    try:
        # Obtener un batch de datos
        batch = next(iter(dataloader))
        images = batch[0].to(device)
        masks = batch[1]
        
        # Preprocesar máscaras
        if masks.dim() == 4 and masks.size(1) == 1:
            masks = masks.squeeze(1)
        masks = masks.to(device)

        # Obtener predicciones
        with torch.no_grad():
            preds = model(images)
            preds = torch.argmax(preds, dim=1)  # Convertir a clases predichas

        # Mover datos a CPU para visualización
        images = images.cpu()
        masks = masks.cpu()
        preds = preds.cpu()

        # Mostrar ejemplos
        for i in range(min(num_examples, images.size(0))):
            fig, axs = plt.subplots(1, 3, figsize=(15, 5))

            # Imagen original
            axs[0].imshow(images[i].permute(1, 2, 0))
            axs[0].set_title("Imagen original")
            axs[0].axis("off")

            # Máscara verdadera
            axs[1].imshow(masks[i], cmap="tab10", vmin=0, vmax=num_classes-1)
            axs[1].set_title("Máscara verdadera")
            axs[1].axis("off")

            # Predicción
            axs[2].imshow(preds[i], cmap="tab10", vmin=0, vmax=num_classes-1)
            axs[2].set_title("Predicción")
            axs[2].axis("off")

            plt.tight_layout()
            plt.show()
            
    except Exception as e:
        print(f"Error al visualizar predicciones: {str(e)}")
def train_model(model: nn.Module, 
                train_loader: torch.utils.data.DataLoader, 
                test_loader: torch.utils.data.DataLoader, 
                epochs: int = 10, 
                lr: float = 1e-4,
                num_classes: int = 4) -> Tuple[nn.Module, List[float], List[float], List[Dict[str, float]]]:
    """Entrena y evalúa el modelo con métricas completas.
    
    Args:
        model: Modelo a entrenar
        train_loader: DataLoader de entrenamiento
        test_loader: DataLoader de validación
        epochs: Número de épocas
        lr: Tasa de aprendizaje
        num_classes: Número de clases
        
    Returns:
        Tuple con (modelo entrenado, pérdidas de entrenamiento, pérdidas de validación, métricas de validación)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Inicializar función de pérdida y optimizador
    loss_fn = ComboLoss(alpha=0.5, beta=0.5)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Historial de métricas
    train_losses = []
    val_losses = []
    val_metrics_history = []

    print("🚀 Iniciando entrenamiento con métricas completas...")
    print("📊 Métricas incluidas: Dice, Accuracy, Precision, Recall, IoU, HD95")
    print("-" * 80)

    for epoch in range(epochs):
        print(f"🚀 Época {epoch+1}/{epochs}")
        
        # Fase de entrenamiento
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device, num_classes)
        train_losses.append(train_loss)
        
        # Fase de evaluación
        val_loss, val_metrics = evaluate(model, test_loader, loss_fn, device, num_classes)
        val_losses.append(val_loss)
        val_metrics_history.append(val_metrics)
        
        # Mostrar métricas principales
        print(f"📊 Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(f"📊 Dice: {val_metrics['dice_avg']:.4f} | Accuracy: {val_metrics['pixel_accuracy']:.4f}")
        print(f"📊 Precision: {val_metrics['precision_avg']:.4f} | Recall: {val_metrics['recall_avg']:.4f}")
        print(f"📊 IoU: {val_metrics['iou_avg']:.4f} | HD95: {val_metrics['hd95_avg']:.2f}")
        print("-" * 80)
        
    # Visualizar resultados finales
    print("🎯 Mostrando predicciones finales...")
    show_predictions(model, test_loader, device, num_classes)
    
    # Graficar evolución de métricas
    print("📈 Graficando evolución de métricas...")
    plot_metrics_evolution(val_metrics_history, num_classes)
    
    # Mostrar resumen final
    final_metrics = val_metrics_history[-1]
    print("\n🏆 RESUMEN FINAL DE MÉTRICAS:")
    print(f"  • Dice Score: {final_metrics['dice_avg']:.4f}")
    print(f"  • Pixel Accuracy: {final_metrics['pixel_accuracy']:.4f}")
    print(f"  • Precision: {final_metrics['precision_avg']:.4f}")
    print(f"  • Recall: {final_metrics['recall_avg']:.4f}")
    print(f"  • IoU: {final_metrics['iou_avg']:.4f}")
    print(f"  • HD95: {final_metrics['hd95_avg']:.2f} píxeles")
    
    print("\n📋 MÉTRICAS POR CLASE:")
    for cls in range(num_classes):
        print(f"  Clase {cls}:")
        print(f"    - Dice: {final_metrics[f'dice_class_{cls}']:.4f}")
        print(f"    - Precision: {final_metrics[f'precision_class_{cls}']:.4f}")
        print(f"    - Recall: {final_metrics[f'recall_class_{cls}']:.4f}")
        print(f"    - IoU: {final_metrics[f'iou_class_{cls}']:.4f}")
        hd_val = final_metrics[f'hd95_class_{cls}']
        hd_str = f"{hd_val:.2f}" if hd_val != float('inf') else "∞"
        print(f"    - HD95: {hd_str} píxeles")
    
    return model, train_losses, val_losses, val_metrics_history