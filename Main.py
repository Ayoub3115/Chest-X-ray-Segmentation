import torch
import matplotlib.pyplot as plt
from models import UNET
from utils import train_model, quick_evaluate_model, show_predictions, save_metrics_to_csv
from data import train_loader, test_loader  



# ======================== CONFIGURACIÓN INICIAL ========================
print("Comprobando GPU...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {device}")

# Crear el modelo (asumiendo que ya tienes UNET definido)
model = UNET(in_channels=3, out_channels=4).to(device)
# model = SwinUNet(in_channels=3, out_channels=4).to(device)  # Si usas SwinUNet
# model = UNetPlusPlus(in_channels=3, out_channels=4).to(device)  # Si usas UNet++

# Mostrar información del modelo
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Tamaño del modelo: {total_params:,} parámetros")
print(f"Tamaño del modelo en MB: {total_params * 4 / (1024 ** 2):.2f} MB")

# ======================== ENTRENAMIENTO ========================
print("\n🚀 Iniciando entrenamiento...")

# Entrenar el modelo (asumiendo que ya tienes train_loader y test_loader)
trained_model, train_losses, val_losses, metrics_history = train_model(
    model=model,
    train_loader=train_loader,
    test_loader=test_loader,
    epochs=45,
    lr=1e-4,
    num_classes=4
)

print("✅ Entrenamiento completado!")

# ======================== OPCIONES ADICIONALES ========================

# 1. Guardar las métricas en CSV
save_metrics_to_csv(metrics_history, "my_training_metrics.csv")

# 2. Evaluación rápida final
print("\n🔍 Evaluación final completa...")
final_metrics = quick_evaluate_model(
    model=trained_model,
    dataloader=test_loader,
    device=device,
    num_classes=4,
    save_to_file=True,
    filename="final_evaluation.csv"
)

# 3. Mostrar predicciones adicionales
print("\n🖼️ Mostrando más ejemplos de predicciones...")
show_predictions(
    model=trained_model,
    dataloader=test_loader,
    device=device,
    num_classes=4,
    num_examples=5
)

# 4. Guardar el modelo entrenado
torch.save({
    'model_state_dict': trained_model.state_dict(),
    'train_losses': train_losses,
    'val_losses': val_losses,
    'metrics_history': metrics_history,
    'final_metrics': final_metrics
}, 'trained_unet_model.pth')

print("💾 Modelo guardado como 'trained_unet_model.pth'")

# ======================== ANÁLISIS DE RESULTADOS ========================

# Mostrar gráfico de pérdidas
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.title('Evolución de la Pérdida')
plt.xlabel('Época')
plt.ylabel('Loss')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
dice_scores = [m['dice_avg'] for m in metrics_history]
plt.plot(dice_scores, 'g-', linewidth=2)
plt.title('Evolución del Dice Score')
plt.xlabel('Época')
plt.ylabel('Dice Score')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("\n🎉 ¡Proceso completo terminado!")