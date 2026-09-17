# -*- coding: utf-8 -*-
"""
Mask-Guided Fidelity-Constrained CycleGAN (MG-CycleGAN)
Lightweight reproduction for CT→MRI translation.

Key components:
1. Brain mask as soft guide (concatenated to input)
2. Encoder-ResBlock-Decoder generator
3. Fidelity loss via structural features (SSIM + L1)
4. Cycle consistency + adversarial training
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    """Residual block: Conv-InstanceNorm-ReLU-Conv-InstanceNorm + skip"""
    def __init__(self, ch):
        super().__init__()
        self.conv1 = nn.Conv2d(ch, ch, 3, padding=1, padding_mode='reflect')
        self.norm1 = nn.InstanceNorm2d(ch)
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=1, padding_mode='reflect')
        self.norm2 = nn.InstanceNorm2d(ch)

    def forward(self, x):
        h = F.relu(self.norm1(self.conv1(x)))
        h = self.norm2(self.conv2(h))
        return x + h


class Generator(nn.Module):
    """
    Encoder-ResBlock-Decoder for CT→MRI translation.
    Input: CT (1ch) + Mask (1ch) = 2ch
    Output: MRI (1ch)
    """
    def __init__(self, in_ch=2, out_ch=1, base_ch=64, n_res=6):
        super().__init__()
        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_ch, base_ch, 7, padding=3, padding_mode='reflect'),
            nn.InstanceNorm2d(base_ch), nn.ReLU(True))
        self.enc2 = nn.Sequential(
            nn.Conv2d(base_ch, base_ch*2, 3, stride=2, padding=1),
            nn.InstanceNorm2d(base_ch*2), nn.ReLU(True))
        self.enc3 = nn.Sequential(
            nn.Conv2d(base_ch*2, base_ch*4, 3, stride=2, padding=1),
            nn.InstanceNorm2d(base_ch*4), nn.ReLU(True))

        # Residual blocks
        self.res_blocks = nn.Sequential(*[ResBlock(base_ch*4) for _ in range(n_res)])

        # Decoder
        self.dec1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(base_ch*4, base_ch*2, 3, padding=1),
            nn.InstanceNorm2d(base_ch*2), nn.ReLU(True))
        self.dec2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(base_ch*2, base_ch, 3, padding=1),
            nn.InstanceNorm2d(base_ch), nn.ReLU(True))
        self.out_conv = nn.Sequential(
            nn.Conv2d(base_ch, out_ch, 7, padding=3, padding_mode='reflect'),
            nn.Tanh())

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        r = self.res_blocks(e3)
        d1 = self.dec1(r)
        d2 = self.dec2(d1)
        return self.out_conv(d2)


class Discriminator(nn.Module):
    """PatchGAN discriminator (70x70 receptive field)"""
    def __init__(self, in_ch=1, base_ch=64):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_ch, base_ch, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(base_ch, base_ch*2, 4, stride=2, padding=1),
            nn.InstanceNorm2d(base_ch*2), nn.LeakyReLU(0.2, True),
            nn.Conv2d(base_ch*2, base_ch*4, 4, stride=2, padding=1),
            nn.InstanceNorm2d(base_ch*4), nn.LeakyReLU(0.2, True),
            nn.Conv2d(base_ch*4, base_ch*8, 4, stride=1, padding=1),
            nn.InstanceNorm2d(base_ch*8), nn.LeakyReLU(0.2, True),
            nn.Conv2d(base_ch*8, 1, 4, stride=1, padding=1),
        )

    def forward(self, x):
        return self.layers(x)
