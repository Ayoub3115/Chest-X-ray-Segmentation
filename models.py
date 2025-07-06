from torch_vertex import Grapher
import torch.nn.functional as F
import torch
import torch.nn as nn
from torchvision import transforms as TF
import timm

# UNet implementation using PyTorch
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x):
        return self.conv(x)
    
class UNET(nn.Module):
    def __init__(self, in_channels=3, out_channels=4, features=[64, 128, 256, 512]):
        super(UNET, self).__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Downsampling path
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # Upsampling path
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2)
            )
            self.ups.append(DoubleConv(feature * 2, feature))

        # Bottleneck
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)

        # Final convolution and softmax
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)
        self.softmax = nn.Softmax(dim=1)  # Softmax across channel dimension

    def forward(self, x):
        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx // 2]

            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])

            x = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx + 1](x)

        x = self.final_conv(x)
        return x

# Unet++ implementation using PyTorch
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=False):
        super(ConvBlock, self).__init__()
        layers = [
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ]
        if dropout:
            layers.append(nn.Dropout2d(0.25))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)

class UNetPlusPlus(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, base_filters=64):
        super(UNetPlusPlus, self).__init__()
        nb_filter = [base_filters, base_filters*2, base_filters*4, base_filters*8, base_filters*16]

        self.pool = nn.MaxPool2d(2, 2)

        # Convs
        self.conv0_0 = ConvBlock(in_channels, nb_filter[0])
        self.conv1_0 = ConvBlock(nb_filter[0], nb_filter[1])
        self.conv2_0 = ConvBlock(nb_filter[1], nb_filter[2])
        self.conv3_0 = ConvBlock(nb_filter[2], nb_filter[3])
        self.conv4_0 = ConvBlock(nb_filter[3], nb_filter[4], dropout=True)

        self.up1_0 = nn.ConvTranspose2d(nb_filter[1], nb_filter[0], 2, stride=2)
        self.conv0_1 = ConvBlock(nb_filter[0]*2, nb_filter[0])

        self.up2_0 = nn.ConvTranspose2d(nb_filter[2], nb_filter[1], 2, stride=2)
        self.conv1_1 = ConvBlock(nb_filter[1]*2, nb_filter[1])

        self.up3_0 = nn.ConvTranspose2d(nb_filter[3], nb_filter[2], 2, stride=2)
        self.conv2_1 = ConvBlock(nb_filter[2]*2, nb_filter[2])

        self.up4_0 = nn.ConvTranspose2d(nb_filter[4], nb_filter[3], 2, stride=2)
        self.conv3_1 = ConvBlock(nb_filter[3]*2, nb_filter[3])

        # Nested convs
        self.up1_1 = nn.ConvTranspose2d(nb_filter[1], nb_filter[0], 2, stride=2)
        self.conv0_2 = ConvBlock(nb_filter[0]*3, nb_filter[0])

        self.up2_1 = nn.ConvTranspose2d(nb_filter[2], nb_filter[1], 2, stride=2)
        self.conv1_2 = ConvBlock(nb_filter[1]*3, nb_filter[1])

        self.up3_1 = nn.ConvTranspose2d(nb_filter[3], nb_filter[2], 2, stride=2)
        self.conv2_2 = ConvBlock(nb_filter[2]*3, nb_filter[2])

        self.up1_2 = nn.ConvTranspose2d(nb_filter[1], nb_filter[0], 2, stride=2)
        self.conv0_3 = ConvBlock(nb_filter[0]*4, nb_filter[0])

        self.up2_2 = nn.ConvTranspose2d(nb_filter[2], nb_filter[1], 2, stride=2)
        self.conv1_3 = ConvBlock(nb_filter[1]*4, nb_filter[1])

        self.up1_3 = nn.ConvTranspose2d(nb_filter[1], nb_filter[0], 2, stride=2)
        self.conv0_4 = ConvBlock(nb_filter[0]*5, nb_filter[0])

        self.final = nn.Conv2d(nb_filter[0], out_channels, kernel_size=1)

    def forward(self, x):
        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up1_0(x1_0)], dim=1))

        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up2_0(x2_0)], dim=1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up1_1(x1_1)], dim=1))

        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up3_0(x3_0)], dim=1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up2_1(x2_1)], dim=1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up1_2(x1_2)], dim=1))

        x4_0 = self.conv4_0(self.pool(x3_0))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up4_0(x4_0)], dim=1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up3_1(x3_1)], dim=1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up2_2(x2_2)], dim=1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up1_3(x1_3)], dim=1))

        return self.final(x0_4)


# SwinUNet implementation using PyTorch and timm

class SwinUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, base_filters=64):
        super(SwinUNet, self).__init__()

        # Usamos un modelo Swin Transformer como extractor de características
        self.backbone = timm.create_model(
            'swin_base_patch4_window7_224', pretrained=True, features_only=True
        )
        
        # Obtener información de los canales del encoder
        self.encoder_channels = self.backbone.feature_info.channels()  # [128, 256, 512, 1024]
        #print(f"Encoder channels: {self.encoder_channels}")

        # 1x1 conv para adaptar entrada si es necesario
        if in_channels != 3:
            self.input_proj = nn.Conv2d(in_channels, 3, kernel_size=1)
        else:
            self.input_proj = nn.Identity()

        # Decoder: Upsample y fusionamos características
        # Nota: Los índices van de menor a mayor resolución
        self.up4 = self._upsample_block(self.encoder_channels[3], self.encoder_channels[2])  # 1024 -> 512
        self.up3 = self._upsample_block(self.encoder_channels[2], self.encoder_channels[1])  # 512 -> 256
        self.up2 = self._upsample_block(self.encoder_channels[1], self.encoder_channels[0])  # 256 -> 128
        self.up1 = self._upsample_block(self.encoder_channels[0], base_filters)  # 128 -> 64

        # Upsampling final para restaurar resolución original
        self.final_up = nn.ConvTranspose2d(base_filters, base_filters, kernel_size=2, stride=2)
        
        # Final conv para mapa de salida
        self.final_conv = nn.Conv2d(base_filters, out_channels, kernel_size=1)

    def _upsample_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        # Guardar tamaño original para redimensionar al final
        original_size = x.shape[2:]
        
        # Proyectar entrada si es necesario
        x = self.input_proj(x)  # x: [B, 3, H, W]
        
        # Extraer características con el backbone
        feats = self.backbone(x)  # Lista de 4 características
        
        # Convertir características de NHWC a NCHW si es necesario
        processed_feats = []
        for i, feat in enumerate(feats):
            #print(f"Feature {i} original shape: {feat.shape}")
            # Si la característica tiene 4 dimensiones y la última es mayor que la tercera,
            # probablemente está en formato NHWC
            if len(feat.shape) == 4 and feat.shape[-1] > feat.shape[-2]:
                feat = feat.permute(0, 3, 1, 2)  # NHWC -> NCHW
                #print(f"Feature {i} converted to: {feat.shape}")
            processed_feats.append(feat)
        
        # Decoder path con skip connections
        x4 = processed_feats[3]  # Características de menor resolución
        x3 = self.up4(x4) + processed_feats[2]  # Upsample y suma skip connection
        x2 = self.up3(x3) + processed_feats[1]  # Upsample y suma skip connection
        x1 = self.up2(x2) + processed_feats[0]  # Upsample y suma skip connection
        
        # Upsampling final
        out = self.up1(x1)
        out = self.final_up(out)
        
        # Aplicar convolución final
        out = self.final_conv(out)
        
        # Redimensionar a tamaño original si es necesario
        if out.shape[2:] != original_size:
            out = F.interpolate(out, size=original_size, mode='bilinear', align_corners=False)
        
        #print(f"Output shape: {out.shape}")
        return out

