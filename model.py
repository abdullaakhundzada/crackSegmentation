import torch.nn as nn
import torch

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        
        # Gating path
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        # Skip connection path
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        # Final 1x1 conv
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, g, x):
        # Get input dimensions
        input_size = x.size()[2:]
        
        # Upsample gating signal to match skip connection size
        g_up = nn.functional.interpolate(g, size=input_size, mode="bilinear", align_corners=True)
        
        # Apply convolutions
        g1 = self.W_g(g_up)
        x1 = self.W_x(x)
        
        # Element-wise addition and ReLU
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        
        return x * psi

class AttentionUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super().__init__()
        
        # Encoder
        self.encoder1 = DoubleConv(in_channels, 64)
        self.encoder2 = DoubleConv(64, 128)
        self.encoder3 = DoubleConv(128, 256)
        self.encoder4 = DoubleConv(256, 512)
        
        # Bottleneck
        self.bottleneck = DoubleConv(512, 1024)
        
        # Attention Gates
        self.attention4 = AttentionGate(F_g=1024, F_l=512, F_int=256)
        self.attention3 = AttentionGate(F_g=512, F_l=256, F_int=128)
        self.attention2 = AttentionGate(F_g=256, F_l=128, F_int=64)
        self.attention1 = AttentionGate(F_g=128, F_l=64, F_int=32)
        
        # Decoder
        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.decoder4 = DoubleConv(1024, 512)
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.decoder3 = DoubleConv(512, 256)
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.decoder2 = DoubleConv(256, 128)
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.decoder1 = DoubleConv(128, 64)
        
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)
        self.maxpool = nn.MaxPool2d(2)
        
    def forward(self, x):
        # Encoder
        enc1 = self.encoder1(x)
        enc1_pool = self.maxpool(enc1)
        
        enc2 = self.encoder2(enc1_pool)
        enc2_pool = self.maxpool(enc2)
        
        enc3 = self.encoder3(enc2_pool)
        enc3_pool = self.maxpool(enc3)
        
        enc4 = self.encoder4(enc3_pool)
        enc4_pool = self.maxpool(enc4)
        
        # Bottleneck
        bottleneck = self.bottleneck(enc4_pool)
        
        # Decoder with Attention
        # Upsample bottleneck and apply attention to enc4
        dec4 = self.upconv4(bottleneck)
        att4 = self.attention4(bottleneck, enc4)
        dec4 = torch.cat((dec4, att4), dim=1)
        dec4 = self.decoder4(dec4)
        
        # Upsample dec4 and apply attention to enc3
        dec3 = self.upconv3(dec4)
        att3 = self.attention3(dec4, enc3)
        dec3 = torch.cat((dec3, att3), dim=1)
        dec3 = self.decoder3(dec3)
        
        # Upsample dec3 and apply attention to enc2
        dec2 = self.upconv2(dec3)
        att2 = self.attention2(dec3, enc2)
        dec2 = torch.cat((dec2, att2), dim=1)
        dec2 = self.decoder2(dec2)
        
        # Upsample dec2 and apply attention to enc1
        dec1 = self.upconv1(dec2)
        att1 = self.attention1(dec2, enc1)
        dec1 = torch.cat((dec1, att1), dim=1)
        dec1 = self.decoder1(dec1)
        
        return torch.sigmoid(self.final_conv(dec1))