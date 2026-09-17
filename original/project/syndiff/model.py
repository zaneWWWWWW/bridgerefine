# -*- coding: utf-8 -*-
"""
SynDiff: Unsupervised Medical Image Translation with Adversarial Diffusion Models
Lightweight reproduction — adapted for supervised CT→MRI.

Key components:
1. Non-diffusive module: fast G_nd(CT) → initial MRI estimate
2. Diffusive module: adversarial diffusion with large steps
3. Large-step reverse process (k ≫ 1) for fast sampling
4. Source-conditional adversarial learning
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def get_timestep_embedding(timesteps, embedding_dim):
    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


class ResBlock(nn.Module):
    def __init__(self, ch, time_emb_dim):
        super().__init__()
        self.norm1 = nn.GroupNorm(min(32, ch), ch)
        self.conv1 = nn.Conv2d(ch, ch, 3, padding=1)
        self.time_proj = nn.Linear(time_emb_dim, ch)
        self.norm2 = nn.GroupNorm(min(32, ch), ch)
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=1)

    def forward(self, x, t_emb):
        h = F.silu(self.norm1(self.conv1(x)))
        h = h + self.time_proj(F.silu(t_emb))[:, :, None, None]
        return x + F.silu(self.norm2(self.conv2(h)))


class NonDiffusiveGenerator(nn.Module):
    """Fast CT→MRI translator: encoder-resblock-decoder."""
    def __init__(self, in_ch=1, out_ch=1, base_ch=64):
        super().__init__()
        self.enc1 = nn.Conv2d(in_ch, base_ch, 3, stride=2, padding=1)
        self.enc2 = nn.Conv2d(base_ch, base_ch*2, 3, stride=2, padding=1)
        self.res1 = ResBlock(base_ch*2, base_ch*4)
        self.res2 = ResBlock(base_ch*2, base_ch*4)
        self.dec1 = nn.Sequential(nn.Upsample(scale_factor=2), nn.Conv2d(base_ch*2, base_ch, 3, padding=1))
        self.dec2 = nn.Sequential(nn.Upsample(scale_factor=2), nn.Conv2d(base_ch, out_ch, 3, padding=1))

    def forward(self, x):
        e1 = F.silu(self.enc1(x))
        e2 = F.silu(self.enc2(e1))
        r = self.res2(self.res1(e2, torch.zeros(1, 256, device=x.device)), torch.zeros(1, 256, device=x.device))
        d1 = F.silu(self.dec1(r))
        return torch.tanh(self.dec2(d1))


class DiffusionUNet(nn.Module):
    """
    Lightweight U-Net for large-step adversarial diffusion.
    Predicts x0 from (xt, condition=CT).
    """
    def __init__(self, in_ch=2, out_ch=1, base_ch=64):
        super().__init__()
        self.base_ch = base_ch
        self.time_mlp = nn.Sequential(nn.Linear(base_ch, base_ch*4), nn.SiLU(), nn.Linear(base_ch*4, base_ch*4))
        self.enc1 = nn.Conv2d(in_ch, base_ch, 3, padding=1)
        self.enc2 = nn.Conv2d(base_ch, base_ch*2, 3, stride=2, padding=1)
        self.enc3 = nn.Conv2d(base_ch*2, base_ch*4, 3, stride=2, padding=1)
        self.mid1 = ResBlock(base_ch*4, base_ch*4)
        self.mid2 = ResBlock(base_ch*4, base_ch*4)
        self.dec1 = nn.Sequential(nn.Upsample(scale_factor=2), nn.Conv2d(base_ch*4, base_ch*2, 3, padding=1))
        self.dec2 = nn.Sequential(nn.Upsample(scale_factor=2), nn.Conv2d(base_ch*2, base_ch, 3, padding=1))
        self.out = nn.Conv2d(base_ch, out_ch, 3, padding=1)

    def forward(self, x, t, condition):
        inp = torch.cat([x, condition], dim=1)
        t_emb = self.time_mlp(get_timestep_embedding(t, self.base_ch))
        e1 = F.silu(self.enc1(inp))
        e2 = F.silu(self.enc2(e1))
        e3 = F.silu(self.enc3(e2))
        m = self.mid2(self.mid1(e3, t_emb), t_emb)
        d1 = F.silu(self.dec1(m + e3))
        d2 = F.silu(self.dec2(d1 + e2))
        return self.out(d2 + e1)


class PatchDiscriminator(nn.Module):
    """PatchGAN for adversarial diffusion."""
    def __init__(self, in_ch=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 64, 4, 2, 1), nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1), nn.InstanceNorm2d(128), nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 4, 2, 1), nn.InstanceNorm2d(256), nn.LeakyReLU(0.2),
            nn.Conv2d(256, 1, 4, 1, 1),
        )

    def forward(self, x_tm1, x_t):
        return self.net(torch.cat([x_tm1, x_t], dim=1))


class SynDiff:
    """
    Adversarial Diffusion Model with large-step reverse process.
    """
    def __init__(self, diff_unet, discriminator, nd_generator, n_steps=4, device='cpu'):
        self.diff_unet = diff_unet
        self.discriminator = discriminator
        self.nd_generator = nd_generator
        self.n_steps = n_steps
        self.device = device
        # Linear schedule
        self.betas = torch.linspace(0.1, 0.9, n_steps, device=device)
        self.alphas = 1 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

    def q_sample(self, x0, t):
        """Forward: x_t = sqrt(alpha_bar)*x0 + sqrt(1-alpha_bar)*noise"""
        ab = self.alpha_bars[t].view(-1, 1, 1, 1)
        noise = torch.randn_like(x0)
        return torch.sqrt(ab) * x0 + torch.sqrt(1 - ab) * noise, noise

    def training_loss_g(self, x0, condition):
        """Generator loss: L1(x0_pred, x0) + adversarial."""
        B = x0.shape[0]
        t = torch.randint(0, self.n_steps, (B,), device=self.device)
        xt, _ = self.q_sample(x0, t)
        x0_pred = self.diff_unet(xt, t, condition)
        # Posterior: x_{t-1} from x_t and x0_pred
        t_prev = (t - 1).clamp(min=0)
        ab_t = self.alpha_bars[t].view(-1, 1, 1, 1)
        ab_tm1 = self.alpha_bars[t_prev].view(-1, 1, 1, 1)
        x_tm1_pred = torch.sqrt(ab_tm1) * x0_pred + torch.sqrt(1 - ab_tm1) * torch.randn_like(x0_pred)
        # Adversarial
        d_out = self.discriminator(x_tm1_pred, xt)
        adv_loss = F.softplus(-d_out).mean()
        rec_loss = F.l1_loss(x0_pred, x0)
        return rec_loss + 0.1 * adv_loss

    def training_loss_d(self, x0, condition):
        """Discriminator loss."""
        B = x0.shape[0]
        t = torch.randint(0, self.n_steps, (B,), device=self.device)
        xt, _ = self.q_sample(x0, t)
        with torch.no_grad():
            x0_pred = self.diff_unet(xt, t, condition)
        t_prev = (t - 1).clamp(min=0)
        ab_t = self.alpha_bars[t].view(-1, 1, 1, 1)
        ab_tm1 = self.alpha_bars[t_prev].view(-1, 1, 1, 1)
        # Real
        x_tm1_real, _ = self.q_sample(x0, t_prev)
        real_out = self.discriminator(x_tm1_real, xt)
        # Fake
        x_tm1_fake = torch.sqrt(ab_tm1) * x0_pred + torch.sqrt(1 - ab_tm1) * torch.randn_like(x0_pred)
        fake_out = self.discriminator(x_tm1_fake, xt)
        return F.softplus(real_out).mean() + F.softplus(-fake_out).mean()

    @torch.no_grad()
    def sample(self, condition):
        """Large-step reverse sampling."""
        B, _, H, W = condition.shape
        # Start from CT + noise (non-diffusive provides init)
        x0_init = self.nd_generator(condition)
        xt = torch.sqrt(self.alpha_bars[-1]) * x0_init + torch.sqrt(1 - self.alpha_bars[-1]) * torch.randn_like(x0_init)

        for t_idx in reversed(range(self.n_steps)):
            t = torch.full((B,), t_idx, dtype=torch.long, device=self.device)
            x0_pred = self.diff_unet(xt, t, condition)
            if t_idx > 0:
                ab_t = self.alpha_bars[t_idx]
                ab_tm1 = self.alpha_bars[t_idx - 1]
                xt = torch.sqrt(ab_tm1) * x0_pred + torch.sqrt(1 - ab_tm1) * torch.randn_like(x0_pred)
            else:
                xt = x0_pred
        return xt
