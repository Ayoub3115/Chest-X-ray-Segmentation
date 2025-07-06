import torch
import numpy as np
from scipy import ndimage
from scipy.spatial.distance import directed_hausdorff
from typing import List, Tuple, Dict
import warnings


def dice_score(preds: torch.Tensor, 
               targets: torch.Tensor, 
               num_classes: int = 4, 
               epsilon: float = 1e-6) -> Tuple[float, List[float]]:
    """Calcula el Dice Score promedio para múltiples clases.
    
    Args:
        preds: Tensor de predicciones [B, C, H, W] (logits)
        targets: Tensor de máscaras [B, H, W] con valores en [0, num_classes-1]
        num_classes: Número de clases
        epsilon: Pequeño valor para estabilidad numérica
        
    Returns:
        Tuple con (Dice Score promedio, lista de dice scores por clase)
    """
    preds = torch.argmax(preds, dim=1)  # Convertir logits a clases predichas
    dice_scores = []
    
    for cls in range(num_classes):
        pred_cls = (preds == cls).float()
        true_cls = (targets == cls).float()
        
        intersection = (pred_cls * true_cls).sum().item()
        union = pred_cls.sum().item() + true_cls.sum().item()
        
        # Manejo de casos donde la clase no está presente
        if union > 0:
            dice_scores.append((2. * intersection + epsilon) / (union + epsilon))
        else:
            dice_scores.append(1.0)  # Clase no presente = score perfecto
            
    return sum(dice_scores) / len(dice_scores), dice_scores


def pixel_accuracy(preds: torch.Tensor, targets: torch.Tensor) -> float:
    """Calcula la precisión a nivel de píxel.
    
    Args:
        preds: Tensor de predicciones [B, C, H, W] (logits)
        targets: Tensor de máscaras [B, H, W]
        
    Returns:
        Accuracy promedio
    """
    preds = torch.argmax(preds, dim=1)
    correct = (preds == targets).float()
    return correct.mean().item()


def precision_recall_per_class(preds: torch.Tensor, 
                              targets: torch.Tensor, 
                              num_classes: int = 4, 
                              epsilon: float = 1e-6) -> Tuple[List[float], List[float]]:
    """Calcula precision y recall por clase.
    
    Args:
        preds: Tensor de predicciones [B, C, H, W] (logits)
        targets: Tensor de máscaras [B, H, W]
        num_classes: Número de clases
        epsilon: Pequeño valor para estabilidad numérica
        
    Returns:
        Tuple con (lista de precision por clase, lista de recall por clase)
    """
    preds = torch.argmax(preds, dim=1)
    precisions = []
    recalls = []
    
    for cls in range(num_classes):
        pred_cls = (preds == cls).float()
        true_cls = (targets == cls).float()
        
        # True Positives, False Positives, False Negatives
        tp = (pred_cls * true_cls).sum().item()
        fp = (pred_cls * (1 - true_cls)).sum().item()
        fn = ((1 - pred_cls) * true_cls).sum().item()
        
        # Precision = TP / (TP + FP)
        if (tp + fp) > 0:
            precision = tp / (tp + fp)
        else:
            precision = 1.0 if tp == 0 else 0.0
            
        # Recall = TP / (TP + FN)
        if (tp + fn) > 0:
            recall = tp / (tp + fn)
        else:
            recall = 1.0 if tp == 0 else 0.0
            
        precisions.append(precision)
        recalls.append(recall)
    
    return precisions, recalls


def iou_score(preds: torch.Tensor, 
              targets: torch.Tensor, 
              num_classes: int = 4, 
              epsilon: float = 1e-6) -> Tuple[float, List[float]]:
    """Calcula IoU (Intersection over Union) por clase.
    
    Args:
        preds: Tensor de predicciones [B, C, H, W] (logits)
        targets: Tensor de máscaras [B, H, W]
        num_classes: Número de clases
        epsilon: Pequeño valor para estabilidad numérica
        
    Returns:
        Tuple con (IoU promedio, lista de IoU por clase)
    """
    preds = torch.argmax(preds, dim=1)
    ious = []
    
    for cls in range(num_classes):
        pred_cls = (preds == cls).float()
        true_cls = (targets == cls).float()
        
        intersection = (pred_cls * true_cls).sum().item()
        union = pred_cls.sum().item() + true_cls.sum().item() - intersection
        
        if union > 0:
            iou = (intersection + epsilon) / (union + epsilon)
        else:
            iou = 1.0  # Clase no presente = IoU perfecto
            
        ious.append(iou)
    
    return sum(ious) / len(ious), ious


def hausdorff_distance_per_class(preds: torch.Tensor, 
                                targets: torch.Tensor, 
                                num_classes: int = 4,
                                percentile: int = 95) -> List[float]:
    """Calcula la distancia de Hausdorff por clase.
    
    Args:
        preds: Tensor de predicciones [B, C, H, W] (logits)
        targets: Tensor de máscaras [B, H, W]
        num_classes: Número de clases
        percentile: Percentil para HD (95 es común en literatura médica)
        
    Returns:
        Lista de distancias de Hausdorff por clase
    """
    preds = torch.argmax(preds, dim=1).cpu().numpy()
    targets = targets.cpu().numpy()
    
    hausdorff_distances = []
    
    for cls in range(num_classes):
        hd_class = []
        
        # Procesar cada imagen en el batch
        for b in range(preds.shape[0]):
            pred_cls = (preds[b] == cls).astype(np.uint8)
            true_cls = (targets[b] == cls).astype(np.uint8)
            
            # Obtener contornos usando detección de bordes
            pred_edges = ndimage.binary_erosion(pred_cls) ^ pred_cls
            true_edges = ndimage.binary_erosion(true_cls) ^ true_cls
            
            # Obtener coordenadas de los bordes
            pred_coords = np.argwhere(pred_edges)
            true_coords = np.argwhere(true_edges)
            
            if len(pred_coords) == 0 or len(true_coords) == 0:
                # Si no hay contornos, asignar distancia máxima o 0 según el caso
                if len(pred_coords) == 0 and len(true_coords) == 0:
                    hd_class.append(0.0)  # Ambos vacíos = distancia 0
                else:
                    # Usar diagonal de la imagen como distancia máxima
                    max_dist = np.sqrt(preds.shape[1]**2 + preds.shape[2]**2)
                    hd_class.append(max_dist)
            else:
                try:
                    # Calcular distancia de Hausdorff bidireccional
                    hd1 = directed_hausdorff(pred_coords, true_coords)[0]
                    hd2 = directed_hausdorff(true_coords, pred_coords)[0]
                    
                    # HD percentil (tomar el percentil especificado en lugar del máximo)
                    if percentile == 100:
                        hd = max(hd1, hd2)
                    else:
                        # Para HD95, calculamos las distancias y tomamos el percentil
                        distances = []
                        for pc in pred_coords:
                            distances.append(np.min(np.linalg.norm(true_coords - pc, axis=1)))
                        for tc in true_coords:
                            distances.append(np.min(np.linalg.norm(pred_coords - tc, axis=1)))
                        hd = np.percentile(distances, percentile)
                    
                    hd_class.append(hd)
                except Exception as e:
                    warnings.warn(f"Error calculando HD para clase {cls}: {e}")
                    hd_class.append(float('inf'))
        
        # Promedio de HD para esta clase en el batch
        if hd_class:
            avg_hd = np.mean([h for h in hd_class if h != float('inf')])
            hausdorff_distances.append(avg_hd if not np.isnan(avg_hd) else float('inf'))
        else:
            hausdorff_distances.append(float('inf'))
    
    return hausdorff_distances


def compute_all_metrics(preds: torch.Tensor, 
                       targets: torch.Tensor, 
                       num_classes: int = 4) -> Dict[str, float]:
    """Calcula todas las métricas de evaluación.
    
    Args:
        preds: Tensor de predicciones [B, C, H, W] (logits)
        targets: Tensor de máscaras [B, H, W]
        num_classes: Número de clases
        
    Returns:
        Diccionario con todas las métricas
    """
    metrics = {}
    
    # Dice Score
    dice_avg, dice_per_class = dice_score(preds, targets, num_classes)
    metrics['dice_avg'] = dice_avg
    for i, dice in enumerate(dice_per_class):
        metrics[f'dice_class_{i}'] = dice
    
    # Pixel Accuracy
    metrics['pixel_accuracy'] = pixel_accuracy(preds, targets)
    
    # Precision y Recall
    precisions, recalls = precision_recall_per_class(preds, targets, num_classes)
    metrics['precision_avg'] = np.mean(precisions)
    metrics['recall_avg'] = np.mean(recalls)
    for i, (prec, rec) in enumerate(zip(precisions, recalls)):
        metrics[f'precision_class_{i}'] = prec
        metrics[f'recall_class_{i}'] = rec
    
    # IoU
    iou_avg, iou_per_class = iou_score(preds, targets, num_classes)
    metrics['iou_avg'] = iou_avg
    for i, iou in enumerate(iou_per_class):
        metrics[f'iou_class_{i}'] = iou
    
    # Hausdorff Distance (HD95)
    try:
        hd_distances = hausdorff_distance_per_class(preds, targets, num_classes, percentile=95)
        metrics['hd95_avg'] = np.mean([hd for hd in hd_distances if hd != float('inf')])
        for i, hd in enumerate(hd_distances):
            metrics[f'hd95_class_{i}'] = hd
    except Exception as e:
        warnings.warn(f"Error calculando Hausdorff Distance: {e}")
        metrics['hd95_avg'] = float('inf')
        for i in range(num_classes):
            metrics[f'hd95_class_{i}'] = float('inf')
    
    return metrics

def compute_all_metrics(preds: torch.Tensor, 
                       targets: torch.Tensor, 
                       num_classes: int = 4) -> Dict[str, float]:
    """Calcula todas las métricas de evaluación.
    
    Args:
        preds: Tensor de predicciones [B, C, H, W] (logits)
        targets: Tensor de máscaras [B, H, W]
        num_classes: Número de clases
        
    Returns:
        Diccionario con todas las métricas
    """
    metrics = {}
    
    # Dice Score
    dice_avg, dice_per_class = dice_score(preds, targets, num_classes)
    metrics['dice_avg'] = dice_avg
    for i, dice in enumerate(dice_per_class):
        metrics[f'dice_class_{i}'] = dice
    
    # Pixel Accuracy
    metrics['pixel_accuracy'] = pixel_accuracy(preds, targets)
    
    # Precision y Recall
    precisions, recalls = precision_recall_per_class(preds, targets, num_classes)
    metrics['precision_avg'] = np.mean(precisions)
    metrics['recall_avg'] = np.mean(recalls)
    for i, (prec, rec) in enumerate(zip(precisions, recalls)):
        metrics[f'precision_class_{i}'] = prec
        metrics[f'recall_class_{i}'] = rec
    
    # IoU
    iou_avg, iou_per_class = iou_score(preds, targets, num_classes)
    metrics['iou_avg'] = iou_avg
    for i, iou in enumerate(iou_per_class):
        metrics[f'iou_class_{i}'] = iou
    
    # Hausdorff Distance (HD95)
    try:
        hd_distances = hausdorff_distance_per_class(preds, targets, num_classes, percentile=95)
        metrics['hd95_avg'] = np.mean([hd for hd in hd_distances if hd != float('inf')])
        for i, hd in enumerate(hd_distances):
            metrics[f'hd95_class_{i}'] = hd
    except Exception as e:
        warnings.warn(f"Error calculando Hausdorff Distance: {e}")
        metrics['hd95_avg'] = float('inf')
        for i in range(num_classes):
            metrics[f'hd95_class_{i}'] = float('inf')
    
    return metrics
