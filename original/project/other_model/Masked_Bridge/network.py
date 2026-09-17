# -*- coding: utf-8 -*-
"""
SelfRDB generator and discriminator networks.

Generator G_θ(x_t, t, y, x̂_0_prev) → x̂_0_new:
    U-Net that predicts the target image x_0 from the noisy bridge sample.
    Input: [xt, y, x0_prev] concatenated (3 channels)
    Output: predicted x_0 (1 channel)

Discriminator D_θ(x_{t-1}, t, x_t):
    Timestep-conditioned PatchGAN discriminator for adversarial training.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def get_timestep_embedding(timesteps, embedding_dim):
    """Sinusoidal time embedding (Transformer-style)."""
    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


class ResBlock(nn.Module):
    """ResBlock with GroupNorm + SiLU + Conv + time embedding projection."""
    def __init__(self, in_ch, out_ch, time_emb_dim, dropout=0.0):
        super().__init__()
        num_groups = min(32, in_ch) if in_ch >= 32 else in_ch
        self.norm1 = nn.GroupNorm(num_groups, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.time_proj = nn.Linear(time_emb_dim, out_ch)
        num_groups2 = min(32, out_ch) if out_ch >= 32 else out_ch
        self.norm2 = nn.GroupNorm(num_groups2, out_ch)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t_emb):
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time_proj(F.silu(t_emb))[:, :, None, None]
        h = self.conv2(self.dropout(F.silu(self.norm2(h))))
        return h + self.skip(x)


class SelfAttention(nn.Module):
    """Single-head self-attention for bottleneck."""
    def __init__(self, channels):
        super().__init__()
        num_groups = min(32, channels) if channels >= 32 else channels
        self.norm = nn.GroupNorm(num_groups, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.norm(x)
        q, k, v = self.qkv(h).reshape(B, 3, C, H * W).unbind(1)
        scale = C ** -0.5
        attn = torch.bmm(q.transpose(1, 2), k) * scale
        attn = F.softmax(attn, dim=-1)
        out = torch.bmm(v, attn.transpose(1, 2)).reshape(B, C, H, W)
        return x + self.proj(out)


class X0UNet(nn.Module):
    """
    U-Net for x_0 prediction in diffusion bridge.

    Input:  x_t (1ch) + y (1ch) + x̂_0_prev (1ch) = 3 channels
    Output: x̂_0_new (1 channel)
    """
    def __init__(self, in_ch=3, out_ch=1, base_ch=64, ch_mult=(1, 2, 4),
                 num_res_blocks=2, dropout=0.1, time_emb_dim=None):
        super().__init__()
        self.base_ch = base_ch
        self.num_resolutions = len(ch_mult)
        time_emb_dim = time_emb_dim or base_ch * 4

        self.time_mlp = nn.Sequential(
            nn.Linear(base_ch, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

        self.conv_in = nn.Conv2d(in_ch, base_ch, 3, padding=1)

        # Encoder
        self.down_blocks = nn.ModuleList()
        ch = base_ch
        ch_list = [ch]
        for i_level in range(self.num_resolutions):
            block_ch = base_ch * ch_mult[i_level]
            for _ in range(num_res_blocks):
                self.down_blocks.append(ResBlock(ch, block_ch, time_emb_dim, dropout))
                ch = block_ch
                ch_list.append(ch)
            if i_level < self.num_resolutions - 1:
                self.down_blocks.append(nn.Conv2d(ch, ch, 3, stride=2, padding=1))
                ch_list.append(ch)

        # Bottleneck
        mid_ch = base_ch * ch_mult[-1]
        self.mid_block1 = ResBlock(ch, mid_ch, time_emb_dim, dropout)
        self.mid_attn = SelfAttention(mid_ch)
        self.mid_block2 = ResBlock(mid_ch, mid_ch, time_emb_dim, dropout)

        # Decoder
        self.up_blocks = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block_ch = base_ch * ch_mult[i_level]
            for _ in range(num_res_blocks + 1):
                self.up_blocks.append(ResBlock(ch + ch_list.pop(), block_ch, time_emb_dim, dropout))
                ch = block_ch
            if i_level > 0:
                self.up_blocks.append(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False))
                self.up_blocks.append(nn.Conv2d(ch, ch, 3, padding=1))

        # Output
        num_groups = min(32, ch) if ch >= 32 else ch
        self.norm_out = nn.GroupNorm(num_groups, ch)
        self.conv_out = nn.Conv2d(ch, out_ch, 3, padding=1)

    def forward(self, x, t, y, x0_prev, mask=None):
        # Concatenate inputs: [x_t, y, x̂_0_prev, mask]
        if mask is None:
            mask = torch.zeros_like(y)
        inp = torch.cat([x, y, x0_prev, mask], dim=1)

        t_emb = get_timestep_embedding(t, self.base_ch)
        t_emb = self.time_mlp(t_emb)

        h = self.conv_in(inp)
        skips = [h]
        for layer in self.down_blocks:
            if isinstance(layer, ResBlock):
                h = layer(h, t_emb)
            else:
                h = layer(h)
            skips.append(h)

        h = self.mid_block1(h, t_emb)
        h = self.mid_attn(h)
        h = self.mid_block2(h, t_emb)

        for layer in self.up_blocks:
            if isinstance(layer, ResBlock):
                skip = skips.pop()
                h = torch.cat([h, skip], dim=1)
                h = layer(h, t_emb)
            else:
                h = layer(h)

        return self.conv_out(F.silu(self.norm_out(h)))


class Discriminator(nn.Module):
    """
    Timestep-conditioned PatchGAN discriminator.

    Input:  x_{t-1} (real or fake, 1ch) + x_t (1ch) = 2 channels + time embedding
    Output: Patch-wise real/fake logits.
    """
    def __init__(self, in_ch=2, base_ch=64, ch_mult=(1, 2, 4, 8)):
        super().__init__()
        self.base_ch = base_ch

        self.time_mlp = nn.Sequential(
            nn.Linear(base_ch, base_ch * 4),
            nn.SiLU(),
            nn.Linear(base_ch * 4, base_ch * 4),
        )

        layers = []
        ch = base_ch
        for i, mult in enumerate(ch_mult):
            out_ch = base_ch * mult
            layers.append(nn.Conv2d(in_ch if i == 0 else ch, out_ch, 4, stride=2, padding=1))
            layers.append(nn.GroupNorm(min(32, out_ch), out_ch))
            layers.append(nn.LeakyReLU(0.2))
            ch = out_ch
            in_ch = out_ch  # for next iteration

        layers.append(nn.Conv2d(ch, 1, 4, stride=1, padding=1))
        self.convs = nn.Sequential(*layers)

        # Time embedding projection to match spatial features
        self.time_proj = nn.Linear(base_ch * 4, ch)

    def forward(self, x_prev, t, x_t):
        t_emb = get_timestep_embedding(t, self.base_ch)
        t_emb = self.time_mlp(t_emb)

        inp = torch.cat([x_prev, x_t], dim=1)
        h = inp
        for layer in self.convs[:-1]:
            h = layer(h)
        # Inject time
        t_proj = self.time_proj(t_emb)[:, :, None, None]
        h = h + t_proj[:, :h.shape[1]]
        out = self.convs[-1](h)
        return out
