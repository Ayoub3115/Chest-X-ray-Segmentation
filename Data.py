import os
import torch
import numpy as np
import shutil
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import re
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import random

# Comprobar GPU
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = 'cpu'
print(f"Usando dispositivo: {device}")

class SegmentationDataset(Dataset):
    def __init__(self, img_dir, mask_dir, transform=None, mask_transform=None):
        """
        Dataset para segmentación de imágenes médicas.
        
        Args:
            img_dir: Directorio donde se encuentran las imágenes
            mask_dir: Directorio donde se encuentran las máscaras
            transform: Transformaciones para las imágenes
            mask_transform: Transformaciones para las máscaras
        """
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.mask_transform = mask_transform
        
        # Obtener nombres de archivos
        self.img_files = sorted([f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.png', '.jpeg'))])
        self.mask_files = sorted([f for f in os.listdir(mask_dir) if f.endswith(('.jpg', '.png', '.jpeg'))])
        
        print(f"Encontrados {len(self.img_files)} archivos de imágenes y {len(self.mask_files)} archivos de máscaras")
        
    def __len__(self):
        return len(self.img_files)
    
    def __getitem__(self, idx):
        # Cargar imagen
        img_path = os.path.join(self.img_dir, self.img_files[idx])
        img = Image.open(img_path).convert('RGB') 
        
        # Cargar máscara
        mask_path = os.path.join(self.mask_dir, self.mask_files[idx])
        mask = Image.open(mask_path).convert('RGB')  # Cambiar a RGB para trabajar con colores
        
        # Aplicar transformaciones a la imagen
        if self.transform:
            img = self.transform(img)
        else:
            img = transforms.ToTensor()(img)
        # Aplicar transformaciones a la máscara (redimensionamiento)
        if self.mask_transform:
            mask = self.mask_transform(mask)
        
        # Convertir máscara a numpy array
        mask_array = np.array(mask)
        
        # Colores de referencia para las clases
        reference_colors = np.array([
            [0, 0, 0],      # Clase 0: Fondo (negro)
            [0, 0, 255],    # Clase 1: Azul
            [0, 255, 0],    # Clase 2: Verde
            [255, 0, 0]     # Clase 3: Rojo
        ])
        
        # Aplanar la máscara para procesamiento eficiente
        height, width = mask_array.shape[:2]
        pixels = mask_array.reshape(-1, 3)
        
        # Pre-calcular las distancias para todos los píxeles a todos los colores de referencia
        # Redimensionar para broadcasting: (n_pixels, 1, 3) - (1, n_classes, 3)
        diff = pixels[:, np.newaxis, :] - reference_colors[np.newaxis, :, :]
        distances = np.sqrt(np.sum(diff**2, axis=2))  # Distancia euclidiana
        
        # Para cada píxel, encuentra la clase con menor distancia
        closest_classes = np.argmin(distances, axis=1)
        
        # Reshape de vuelta a la forma original
        mask_mapped = closest_classes.reshape(height, width)
        
        # Convertir a tensor de tipo long
        mask_tensor = torch.from_numpy(mask_mapped.astype(np.int64))
        
        return img, mask_tensor, self.img_files[idx], self.mask_files[idx]

    def get_original_item(self, idx):
        """Devuelve las imágenes originales (no transformadas) para visualización"""
        img_path = os.path.join(self.img_dir, self.img_files[idx])
        mask_path = os.path.join(self.mask_dir, self.mask_files[idx])
        
        img = np.array(Image.open(img_path).convert('RGB'))
        mask = np.array(Image.open(mask_path).convert('RGB'))  # Cambiar a RGB para trabajar con colores
        
        return img, mask, img_path, mask_path


def organizar_dataset(ruta_base="data", train_dir="dataset/train", test_dir="dataset/test", seed=42):
    """
    Organiza el dataset en carpetas de entrenamiento y prueba.
    Usa exactamente 1000 imágenes para entrenamiento y 200 para test.
    
    Args:
        ruta_base: Ruta donde están las carpetas 'images' y 'masks'
        train_dir: Directorio para datos de entrenamiento
        test_dir: Directorio para datos de prueba
        seed: Semilla para reproducibilidad
    
    Returns:
        train_img_dir, train_mask_dir, test_img_dir, test_mask_dir: Rutas a las carpetas creadas
    """
    # Rutas de entrada
    images_dir = os.path.join(ruta_base, "images")
    masks_dir = os.path.join(ruta_base, "masks")
    
    # Crear directorios y limpiarlos si ya existen
    train_img_dir = os.path.join(train_dir, "images")
    train_mask_dir = os.path.join(train_dir, "masks")
    test_img_dir = os.path.join(test_dir, "images")
    test_mask_dir = os.path.join(test_dir, "masks")
    
    # Limpiar directorios existentes
    if os.path.exists(train_img_dir):
        shutil.rmtree(train_img_dir)
    if os.path.exists(train_mask_dir):
        shutil.rmtree(train_mask_dir)
    if os.path.exists(test_img_dir):
        shutil.rmtree(test_img_dir)
    if os.path.exists(test_mask_dir):
        shutil.rmtree(test_mask_dir)
    
    # Crear directorios limpios
    os.makedirs(train_img_dir, exist_ok=True)
    os.makedirs(train_mask_dir, exist_ok=True)
    os.makedirs(test_img_dir, exist_ok=True)
    os.makedirs(test_mask_dir, exist_ok=True)
    
    print("Directorios creados:")
    print(f"- Entrenamiento: {train_dir}")
    print(f"- Prueba: {test_dir}")
    
    # Mapear imágenes y máscaras por ID
    def extract_id(filename):
        match = re.search(r'(ID\d+)_(?:mask_)?(\d+)', filename)
        if match:
            base_id, num = match.groups()
            return f"{base_id}_{num}"
        return filename
    
    # Encontrar archivos coincidentes
    img_files = [f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
    mask_files = [f for f in os.listdir(masks_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
    
    # Agrupar por ID
    image_ids = {extract_id(f): f for f in img_files}
    mask_ids = {extract_id(f): f for f in mask_files}
    
    # Encontrar coincidencias
    common_ids = set(image_ids.keys()) & set(mask_ids.keys())
    print(f"Encontrados {len(common_ids)} pares válidos de imagen-máscara")
    
    # Dividir en entrenamiento y prueba con números fijos
    random.seed(seed)
    all_ids = list(common_ids)
    random.shuffle(all_ids)
    
    # Tomar exactamente 200 para test y 1000 para train
    test_ids = set(all_ids[:200])  # Primeros 200 para test
    train_ids = set(all_ids[200:1200])  # Siguientes 1000 para train
    
    print(f"Usando exactamente {len(train_ids)} pares para entrenamiento y {len(test_ids)} pares para prueba")
    
    # Copiar archivos a los directorios correspondientes
    for id in train_ids:
        # Copiar imagen
        src_img = os.path.join(images_dir, image_ids[id])
        dst_img = os.path.join(train_img_dir, image_ids[id])
        shutil.copy2(src_img, dst_img)
        
        # Copiar máscara
        src_mask = os.path.join(masks_dir, mask_ids[id])
        dst_mask = os.path.join(train_mask_dir, mask_ids[id])
        shutil.copy2(src_mask, dst_mask)
    
    for id in test_ids:
        # Copiar imagen
        src_img = os.path.join(images_dir, image_ids[id])
        dst_img = os.path.join(test_img_dir, image_ids[id])
        shutil.copy2(src_img, dst_img)
        
        # Copiar máscara
        src_mask = os.path.join(masks_dir, mask_ids[id])
        dst_mask = os.path.join(test_mask_dir, mask_ids[id])
        shutil.copy2(src_mask, dst_mask)
    
    print(f"Copiados {len(train_ids)} pares al conjunto de entrenamiento")
    print(f"Copiados {len(test_ids)} pares al conjunto de prueba")
    
    return train_img_dir, train_mask_dir, test_img_dir, test_mask_dir

def crear_dataloaders(train_img_dir, train_mask_dir, test_img_dir, test_mask_dir, batch_size=8, img_size=224):
    """
    Crea DataLoaders para entrenamiento y prueba.
    
    Args:
        train_img_dir: Directorio de imágenes de entrenamiento
        train_mask_dir: Directorio de máscaras de entrenamiento
        test_img_dir: Directorio de imágenes de prueba
        test_mask_dir: Directorio de máscaras de prueba
        batch_size: Tamaño del lote
        img_size: Tamaño al que redimensionar las imágenes
    
    Returns:
        train_loader, test_loader: DataLoaders
    """
    # Transformaciones
    image_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Para las máscaras, necesitamos que mantengan valores discretos
    # No usamos ToTensor() directamente porque normaliza a [0,1]
    mask_transform = transforms.Compose([
        transforms.Resize((img_size, img_size), interpolation=transforms.InterpolationMode.NEAREST),
        # No aplicamos transformación a tensor aquí, lo haremos en __getitem__
    ])
    
    # Datasets
    train_dataset = SegmentationDataset(
        train_img_dir, 
        train_mask_dir, 
        transform=image_transform, 
        mask_transform=mask_transform
    )
    
    test_dataset = SegmentationDataset(
        test_img_dir, 
        test_mask_dir, 
        transform=image_transform, 
        mask_transform=mask_transform
    )
    
    # DataLoaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=2,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=2,
        pin_memory=True
    )
    
    return train_loader, test_loader, train_dataset, test_dataset
# Función para ver imágenes y máscaras
def visualizar_imagen_mascara(dataset, idx):
    """
    Visualiza una imagen y su máscara correspondiente.
    
    Args:
        dataset: Dataset de segmentación
        idx: Índice de la imagen a visualizar
    """
    img, mask, img_path, mask_path = dataset.get_original_item(idx)
    
    # Obtener solo los nombres de archivo (sin rutas)
    img_filename = os.path.basename(img_path)
    mask_filename = os.path.basename(mask_path)
    
    # Crear figura
    fig = plt.figure(figsize=(15, 8))
    gs = GridSpec(2, 2, height_ratios=[4, 1], width_ratios=[1, 1])
    
    # Mostrar imagen original
    ax0 = plt.subplot(gs[0, 0])
    ax0.imshow(img)
    ax0.set_title("Imagen original")
    ax0.axis('off')
    
    # Mostrar máscara
    ax1 = plt.subplot(gs[0, 1])
    ax1.imshow(mask, cmap='gray')
    ax1.set_title("Máscara")
    ax1.axis('off')
    
    # Texto con nombres de archivo
    ax2 = plt.subplot(gs[1, :])
    ax2.axis('off')
    ax2.text(0.1, 0.7, f"Nombre de la imagen: {img_filename}", fontsize=12)
    ax2.text(0.1, 0.3, f"Nombre de la máscara: {mask_filename}", fontsize=12)
    
    plt.tight_layout()
    plt.show()

# Ejemplo de uso
if __name__ == "__main__":
    # Organizar el dataset en carpetas de entrenamiento y prueba
    train_img_dir, train_mask_dir, test_img_dir, test_mask_dir = organizar_dataset(
        ruta_base="data",
        train_dir="dataset/train",
        test_dir="dataset/test"
    )
    
    # Crear DataLoaders
    train_loader, test_loader, train_dataset, test_dataset = crear_dataloaders(
        train_img_dir, 
        train_mask_dir, 
        test_img_dir, 
        test_mask_dir,
        batch_size=4
    )
    
    # Visualizar algunas imágenes de prueba
    if len(train_dataset) > 0:
        print("\nVisualizando imagen de entrenamiento:")
        visualizar_imagen_mascara(train_dataset, 0)
    
    if len(test_dataset) > 0:
        print("\nVisualizando imagen de prueba:")
        visualizar_imagen_mascara(test_dataset, 0)