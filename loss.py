import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1e-6):
        """Inicializa Dice Loss para segmentación multiclase.
        
        Args:
            smooth: Valor pequeño para evitar división por cero
        """
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Calcula Dice Loss.
        
        Args:
            inputs: Logits del modelo [B, C, H, W]
            targets: Máscaras de verdad [B, H, W] con valores en [0, C-1]
            
        Returns:
            Valor del loss
        """
        # Convertir logits a probabilidades
        inputs = torch.softmax(inputs, dim=1)
        
        # Convertir máscaras a one-hot encoding
        if targets.dim() == 4 and targets.size(1) == 1:
            targets = targets.squeeze(1)  # Eliminar dimensión de canal si existe
            
        targets_one_hot = torch.nn.functional.one_hot(targets, num_classes=inputs.shape[1])
        targets_one_hot = targets_one_hot.permute(0, 3, 1, 2).float()

        # Calcular Dice Loss
        intersection = (inputs * targets_one_hot).sum(dim=(2, 3))
        union = inputs.sum(dim=(2, 3)) + targets_one_hot.sum(dim=(2, 3))
        dice = (2 * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1 - dice.mean()
        
        return dice_loss


class ComboLoss(nn.Module):
    def __init__(self, alpha: float = 0.5, beta: float = 0.5):
        """Combina Dice Loss y CrossEntropy Loss.
        
        Args:
            alpha: Peso para Dice Loss
            beta: Peso para CrossEntropy Loss
        """
        super(ComboLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.dice_loss = DiceLoss()
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Calcula la pérdida combinada.
        
        Args:
            inputs: Logits del modelo [B, C, H, W]
            targets: Máscaras de verdad [B, H, W] con valores en [0, C-1]
            
        Returns:
            Valor del loss combinado
        """
        dice = self.dice_loss(inputs, targets)
        ce = self.ce_loss(inputs, targets)
        return self.alpha * dice + self.beta * ce