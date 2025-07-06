import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from typing import List, Dict
from metrics import compute_all_metrics


def plot_metrics_evolution(metrics_history: List[Dict[str, float]], num_classes: int = 4):
    """Grafica la evolución de todas las métricas durante el entrenamiento.
    
    Args:
        metrics_history: Lista de diccionarios con métricas por época
        num_classes: Número de clases
    """
    epochs = range(1, len(metrics_history) + 1)
    
    # Crear subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    # 1. Dice Score
    dice_avg = [m['dice_avg'] for m in metrics_history]
    axes[0].plot(epochs, dice_avg, 'b-', linewidth=2, label='Promedio')
    for cls in range(num_classes):
        dice_cls = [m[f'dice_class_{cls}'] for m in metrics_history]
        axes[0].plot(epochs, dice_cls, '--', alpha=0.7, label=f'Clase {cls}')
    axes[0].set_title('Dice Score')
    axes[0].set_xlabel('Época')
    axes[0].set_ylabel('Dice Score')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 2. Pixel Accuracy
    accuracy = [m['pixel_accuracy'] for m in metrics_history]
    axes[1].plot(epochs, accuracy, 'g-', linewidth=2)
    axes[1].set_title('Pixel Accuracy')
    axes[1].set_xlabel('Época')
    axes[1].set_ylabel('Accuracy')
    axes[1].grid(True, alpha=0.3)
    
    # 3. Precision
    precision_avg = [m['precision_avg'] for m in metrics_history]
    axes[2].plot(epochs, precision_avg, 'r-', linewidth=2, label='Promedio')
    for cls in range(num_classes):
        prec_cls = [m[f'precision_class_{cls}'] for m in metrics_history]
        axes[2].plot(epochs, prec_cls, '--', alpha=0.7, label=f'Clase {cls}')
    axes[2].set_title('Precision')
    axes[2].set_xlabel('Época')
    axes[2].set_ylabel('Precision')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    # 4. Recall
    recall_avg = [m['recall_avg'] for m in metrics_history]
    axes[3].plot(epochs, recall_avg, 'orange', linewidth=2, label='Promedio')
    for cls in range(num_classes):
        rec_cls = [m[f'recall_class_{cls}'] for m in metrics_history]
        axes[3].plot(epochs, rec_cls, '--', alpha=0.7, label=f'Clase {cls}')
    axes[3].set_title('Recall')
    axes[3].set_xlabel('Época')
    axes[3].set_ylabel('Recall')
    axes[3].legend()
    axes[3].grid(True, alpha=0.3)
    
    # 5. IoU
    iou_avg = [m['iou_avg'] for m in metrics_history]
    axes[4].plot(epochs, iou_avg, 'm-', linewidth=2, label='Promedio')
    for cls in range(num_classes):
        iou_cls = [m[f'iou_class_{cls}'] for m in metrics_history]
        axes[4].plot(epochs, iou_cls, '--', alpha=0.7, label=f'Clase {cls}')
    axes[4].set_title('IoU (Intersection over Union)')
    axes[4].set_xlabel('Época')
    axes[4].set_ylabel('IoU')
    axes[4].legend()
    axes[4].grid(True, alpha=0.3)
    
    # 6. Hausdorff Distance (HD95)
    hd95_avg = [m['hd95_avg'] if m['hd95_avg'] != float('inf') else np.nan for m in metrics_history]
    axes[5].plot(epochs, hd95_avg, 'c-', linewidth=2, label='Promedio')
    for cls in range(num_classes):
        hd_cls = [m[f'hd95_class_{cls}'] if m[f'hd95_class_{cls}'] != float('inf') else np.nan 
                  for m in metrics_history]
        axes[5].plot(epochs, hd_cls, '--', alpha=0.7, label=f'Clase {cls}')
    axes[5].set_title('Hausdorff Distance 95 (HD95)')
    axes[5].set_xlabel('Época')
    axes[5].set_ylabel('HD95 (píxeles)')
    axes[5].legend()
    axes[5].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def save_metrics_to_csv(metrics_history: List[Dict[str, float]], filename: str = "training_metrics.csv"):
    """Guarda las métricas de entrenamiento en un archivo CSV.
    
    Args:
        metrics_history: Lista de diccionarios con métricas por época
        filename: Nombre del archivo CSV
    """
    try:
        import pandas as pd
        
        # Convertir lista de diccionarios a DataFrame
        df = pd.DataFrame(metrics_history)
        
        # Añadir columna de época
        df.insert(0, 'epoch', range(1, len(df) + 1))
        
        # Guardar a CSV
        df.to_csv(filename, index=False)
        print(f"✅ Métricas guardadas en: {filename}")
        
    except ImportError:
        print("⚠️ pandas no está disponible. Guardando métricas en formato texto...")
        
        # Alternativa sin pandas
        with open(filename.replace('.csv', '.txt'), 'w') as f:
            # Escribir encabezados
            headers = ['epoch'] + list(metrics_history[0].keys())
            f.write('\t'.join(headers) + '\n')
            
            # Escribir datos
            for epoch, metrics in enumerate(metrics_history, 1):
                row = [str(epoch)] + [str(metrics[key]) for key in metrics.keys()]
                f.write('\t'.join(row) + '\n')
        
        print(f"✅ Métricas guardadas en formato texto: {filename.replace('.csv', '.txt')}")


def load_metrics_from_csv(filename: str) -> List[Dict[str, float]]:
    """Carga métricas desde un archivo CSV.
    
    Args:
        filename: Nombre del archivo CSV
        
    Returns:
        Lista de diccionarios con métricas por época
    """
    try:
        import pandas as pd
        df = pd.read_csv(filename)
        
        # Convertir DataFrame a lista de diccionarios
        metrics_history = []
        for _, row in df.iterrows():
            metrics = {key: value for key, value in row.items() if key != 'epoch'}
            metrics_history.append(metrics)
        
        print(f"✅ Métricas cargadas desde: {filename}")
        return metrics_history
        
    except ImportError:
        print("⚠️ pandas no está disponible. Use save_metrics_to_csv con formato texto.")
        return []
    except FileNotFoundError:
        print(f"❌ Archivo no encontrado: {filename}")
        return []


def compare_models_metrics(metrics_list: List[List[Dict[str, float]]], 
                          model_names: List[str], 
                          metric_name: str = 'dice_avg'):
    """Compara las métricas de múltiples modelos.
    
    Args:
        metrics_list: Lista de historiales de métricas para cada modelo
        model_names: Nombres de los modelos
        metric_name: Nombre de la métrica a comparar
    """
    plt.figure(figsize=(12, 8))
    
    for i, (metrics_history, name) in enumerate(zip(metrics_list, model_names)):
        epochs = range(1, len(metrics_history) + 1)
        values = [m[metric_name] for m in metrics_history]
        plt.plot(epochs, values, linewidth=2, label=name, marker='o', markersize=4)
    
    plt.title(f'Comparación de {metric_name.replace("_", " ").title()} entre Modelos')
    plt.xlabel('Época')
    plt.ylabel(metric_name.replace("_", " ").title())
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # Mostrar tabla de comparación final
    print(f"\n📊 COMPARACIÓN FINAL - {metric_name.upper()}:")
    print("-" * 50)
    for name, metrics_history in zip(model_names, metrics_list):
        final_value = metrics_history[-1][metric_name]
        print(f"{name:20}: {final_value:.4f}")


def print_metrics_summary(metrics: Dict[str, float], num_classes: int = 4):
    """Imprime un resumen bonito de las métricas.
    
    Args:
        metrics: Diccionario con métricas
        num_classes: Número de clases
    """
    print("\n" + "="*60)
    print("📊 RESUMEN DE MÉTRICAS DE EVALUACIÓN")
    print("="*60)
    
    # Métricas principales
    print(f"🎯 MÉTRICAS GENERALES:")
    print(f"   Dice Score Promedio:    {metrics['dice_avg']:.4f}")
    print(f"   Pixel Accuracy:         {metrics['pixel_accuracy']:.4f}")
    print(f"   Precision Promedio:     {metrics['precision_avg']:.4f}")
    print(f"   Recall Promedio:        {metrics['recall_avg']:.4f}")
    print(f"   IoU Promedio:           {metrics['iou_avg']:.4f}")
    hd_val = metrics['hd95_avg']
    hd_str = f"{hd_val:.2f}" if hd_val != float('inf') else "∞"
    print(f"   HD95 Promedio:          {hd_str} píxeles")
    
    print(f"\n📋 MÉTRICAS DETALLADAS POR CLASE:")
    print("-" * 60)
    
    # Encabezado de tabla
    print(f"{'Clase':<6} {'Dice':<8} {'Prec.':<8} {'Recall':<8} {'IoU':<8} {'HD95':<10}")
    print("-" * 60)
    
    # Métricas por clase
    for cls in range(num_classes):
        dice_val = metrics[f'dice_class_{cls}']
        prec_val = metrics[f'precision_class_{cls}']
        rec_val = metrics[f'recall_class_{cls}']
        iou_val = metrics[f'iou_class_{cls}']
        hd_val = metrics[f'hd95_class_{cls}']
        hd_str = f"{hd_val:.2f}" if hd_val != float('inf') else "∞"
        
        print(f"{cls:<6} {dice_val:<8.4f} {prec_val:<8.4f} {rec_val:<8.4f} {iou_val:<8.4f} {hd_str:<10}")
    
    print("="*60)


# Ejemplo de uso y función de utilidad para evaluación rápida
def quick_evaluate_model(model: nn.Module, 
                        dataloader: torch.utils.data.DataLoader, 
                        device: torch.device, 
                        num_classes: int = 4,
                        save_to_file: bool = False,
                        filename: str = "evaluation_results.csv") -> Dict[str, float]:
    """Evaluación rápida de un modelo con todas las métricas.
    
    Args:
        model: Modelo a evaluar
        dataloader: DataLoader con datos
        device: Dispositivo (CPU/GPU)
        num_classes: Número de clases
        save_to_file: Si guardar resultados en archivo
        filename: Nombre del archivo
        
    Returns:
        Diccionario con todas las métricas
    """
    print("🔍 Evaluando modelo...")
    
    model.eval()
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
            
            # Forward pass
            outputs = model(images)
            
            # Calcular métricas
            batch_metrics = compute_all_metrics(outputs, masks, num_classes)
            
            # Acumular métricas
            for key, value in batch_metrics.items():
                if key not in all_metrics:
                    all_metrics[key] = []
                all_metrics[key].append(value)
            
            num_batches += 1
    
    # Promediar métricas
    final_metrics = {}
    for key, values in all_metrics.items():
        if 'hd95' in key:
            finite_values = [v for v in values if v != float('inf')]
            final_metrics[key] = np.mean(finite_values) if finite_values else float('inf')
        else:
            final_metrics[key] = np.mean(values)
    
    # Mostrar resumen
    print_metrics_summary(final_metrics, num_classes)
    
    # Guardar si se solicita
    if save_to_file:
        save_metrics_to_csv([final_metrics], filename)
    
    return final_metrics