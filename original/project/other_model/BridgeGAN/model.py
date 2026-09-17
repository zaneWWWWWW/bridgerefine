# -*- coding: utf-8 -*-
"""
BridgeGAN: Diffusion Bridge + GAN Post-Refiner
Stage 1: SelfRDB bridge → coarse MRI (structurally accurate)
Stage 2: Lightweight GAN → refined MRI (texture enhanced)
"""
import torch, torch.nn as nn, torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv1 = nn.Conv2d(ch, ch, 3, padding=1, padding_mode='reflect')
        self.norm1 = nn.InstanceNorm2d(ch)
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=1, padding_mode='reflect')
        self.norm2 = nn.InstanceNorm2d(ch)

    def forward(self, x):
        h = F.relu(self.norm1(self.conv1(x)))
        return x + self.norm2(self.conv2(h))


class RefinerGenerator(nn.Module):
    """
    Lightweight generator: coarse MRI + CT → refined MRI.
    Architecture: small U-Net (2 down, 2 up) with residual blocks.
    Input: coarse_MRI(1ch) + CT(1ch) = 2ch
    Output: refined MRI (1ch)
    """
    def __init__(self, in_ch=2, out_ch=1, base_ch=64):
        super().__init__()
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_ch, base_ch, 7, padding=3, padding_mode='reflect'),
            nn.InstanceNorm2d(base_ch), nn.ReLU(True))
        self.enc2 = nn.Sequential(
            nn.Conv2d(base_ch, base_ch*2, 3, stride=2, padding=1),
            nn.InstanceNorm2d(base_ch*2), nn.ReLU(True))
        self.res = nn.Sequential(*[ResBlock(base_ch*2) for _ in range(4)])
        self.dec1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(base_ch*2, base_ch, 3, padding=1),
            nn.InstanceNorm2d(base_ch), nn.ReLU(True))
        self.out = nn.Sequential(
            nn.Conv2d(base_ch, out_ch, 7, padding=3, padding_mode='reflect'),
            nn.Tanh())

    def forward(self, coarse_mri, ct):
        x = torch.cat([coarse_mri, ct], dim=1)
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        r = self.res(e2)
        d1 = self.dec1(r)
        return self.out(d1)


class PatchGANDisc(nn.Module):
    """PatchGAN discriminator for refined MRI."""
    def __init__(self, in_ch=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 64, 4, 2, 1), nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1), nn.InstanceNorm2d(128), nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 4, 2, 1), nn.InstanceNorm2d(256), nn.LeakyReLU(0.2),
            nn.Conv2d(256, 512, 4, 1, 1), nn.InstanceNorm2d(512), nn.LeakyReLU(0.2),
            nn.Conv2d(512, 1, 4, 1, 1),
        )

    def forward(self, x):
        return self.net(x)
