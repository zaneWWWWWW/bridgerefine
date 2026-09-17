# -*- coding: utf-8 -*-
"""
Mask-Guided Adversarial Diffusion Bridge
= SelfRDB bridge + brain mask guidance + adversarial training
"""
import math, torch, torch.nn as nn, torch.nn.functional as F


def get_timestep_embedding(timesteps, embedding_dim):
    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1: emb = F.pad(emb, (0, 1))
    return emb

class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim, dropout=0.0):
        super().__init__()
        ng = min(32, in_ch) if in_ch >= 32 else in_ch
        self.norm1 = nn.GroupNorm(ng, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.time_proj = nn.Linear(time_emb_dim, out_ch)
        ng2 = min(32, out_ch) if out_ch >= 32 else out_ch
        self.norm2 = nn.GroupNorm(ng2, out_ch)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t_emb):
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time_proj(F.silu(t_emb))[:, :, None, None]
        h = self.conv2(self.dropout(F.silu(self.norm2(h))))
        return h + self.skip(x)


class SelfAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        ng = min(32, channels) if channels >= 32 else channels
        self.norm = nn.GroupNorm(ng, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.norm(x)
        q, k, v = self.qkv(h).reshape(B, 3, C, H * W).unbind(1)
        attn = torch.bmm(q.transpose(1, 2), k) * (C ** -0.5)
        attn = F.softmax(attn, dim=-1)
        out = torch.bmm(v, attn.transpose(1, 2)).reshape(B, C, H, W)
        return x + self.proj(out)


class MaskedX0UNet(nn.Module):
    """
    U-Net with mask guidance for x0 prediction.
    Input: xt(1) + y(1) + x0_prev(1) + mask(1) = 4 channels
    """
    def __init__(self, in_ch=4, out_ch=1, base_ch=64, ch_mult=(1, 2, 4),
                 num_res_blocks=2, dropout=0.1, time_emb_dim=None):
        super().__init__()
        self.base_ch = base_ch
        self.num_resolutions = len(ch_mult)
        time_emb_dim = time_emb_dim or base_ch * 4
        self.time_mlp = nn.Sequential(nn.Linear(base_ch, time_emb_dim), nn.SiLU(), nn.Linear(time_emb_dim, time_emb_dim))
        self.conv_in = nn.Conv2d(in_ch, base_ch, 3, padding=1)

        self.down_blocks = nn.ModuleList()
        ch = base_ch; ch_list = [ch]
        for i_level in range(self.num_resolutions):
            block_ch = base_ch * ch_mult[i_level]
            for _ in range(num_res_blocks):
                self.down_blocks.append(ResBlock(ch, block_ch, time_emb_dim, dropout))
                ch = block_ch; ch_list.append(ch)
            if i_level < self.num_resolutions - 1:
                self.down_blocks.append(nn.Conv2d(ch, ch, 3, stride=2, padding=1))
                ch_list.append(ch)

        mid_ch = base_ch * ch_mult[-1]
        self.mid_block1 = ResBlock(ch, mid_ch, time_emb_dim, dropout)
        self.mid_attn = SelfAttention(mid_ch)
        self.mid_block2 = ResBlock(mid_ch, mid_ch, time_emb_dim, dropout)

        self.up_blocks = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block_ch = base_ch * ch_mult[i_level]
            for _ in range(num_res_blocks + 1):
                self.up_blocks.append(ResBlock(ch + ch_list.pop(), block_ch, time_emb_dim, dropout))
                ch = block_ch
            if i_level > 0:
                self.up_blocks.append(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False))
                self.up_blocks.append(nn.Conv2d(ch, ch, 3, padding=1))

        ng = min(32, ch) if ch >= 32 else ch
        self.norm_out = nn.GroupNorm(ng, ch)
        self.conv_out = nn.Conv2d(ch, out_ch, 3, padding=1)

    def forward(self, x, t, y, x0_prev, mask):
        inp = torch.cat([x, y, x0_prev, mask], dim=1)
        t_emb = self.time_mlp(get_timestep_embedding(t, self.base_ch))
        h = self.conv_in(inp); skips = [h]
        for layer in self.down_blocks:
            if isinstance(layer, ResBlock): h = layer(h, t_emb)
            else: h = layer(h)
            skips.append(h)
        h = self.mid_block1(h, t_emb); h = self.mid_attn(h); h = self.mid_block2(h, t_emb)
        for layer in self.up_blocks:
            if isinstance(layer, ResBlock):
                skip = skips.pop(); h = torch.cat([h, skip], dim=1); h = layer(h, t_emb)
            else: h = layer(h)
        return self.conv_out(F.silu(self.norm_out(h)))


class PatchGANDiscriminator(nn.Module):
    """PatchGAN for adversarial bridge posterior sampling."""
    def __init__(self, in_ch=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 64, 4, 2, 1), nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1), nn.InstanceNorm2d(128), nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 4, 2, 1), nn.InstanceNorm2d(256), nn.LeakyReLU(0.2),
            nn.Conv2d(256, 512, 4, 1, 1), nn.InstanceNorm2d(512), nn.LeakyReLU(0.2),
            nn.Conv2d(512, 1, 4, 1, 1),
        )

    def forward(self, x_prev, x_t):
        return self.net(torch.cat([x_prev, x_t], dim=1))


class MaskedAdvDiffusionBridge:
    """Diffusion bridge with mask guidance + adversarial training."""
    def __init__(self, model, discriminator, timesteps=300, gamma=0.1, device='cpu'):
        self.model = model
        self.discriminator = discriminator
        self.T = timesteps; self.gamma = gamma; self.device = device
        t = torch.arange(0, timesteps + 1, dtype=torch.float32, device=device)
        self.a = (timesteps - t) / timesteps
        self.b = t / timesteps
        self.var = gamma * t / timesteps
        self.sqrt_var = torch.sqrt(self.var)
        self.delta_a = self.a[1:] - self.a[:-1]
        self.delta_b = self.b[1:] - self.b[:-1]
        self.step_var = self.var[1:] - self.var[:-1]
        self.sqrt_step_var = torch.sqrt(self.step_var)

    def q_sample(self, x0, y, t):
        a_t = self.a[t].view(-1, 1, 1, 1); b_t = self.b[t].view(-1, 1, 1, 1)
        s_t = self.sqrt_var[t].view(-1, 1, 1, 1)
        noise = torch.randn_like(x0)
        return a_t * x0 + b_t * y + s_t * noise, noise

    def q_posterior(self, xt, t, x0_hat, y):
        B = xt.shape[0]; t_idx = t.clamp(1, self.T)
        da = self.delta_a[t_idx-1].view(B,1,1,1); db = self.delta_b[t_idx-1].view(B,1,1,1)
        ss = self.sqrt_step_var[t_idx-1].view(B,1,1,1)
        return xt + da * x0_hat + db * y + ss * torch.randn_like(xt)

    def training_loss_g(self, x0, y, mask):
        """Generator loss: L1(x0_pred, x0) + adversarial."""
        B = x0.shape[0]
        t = torch.randint(1, self.T + 1, (B,), device=self.device)
        xt, _ = self.q_sample(x0, y, t)
        # x0 prediction (2-stage recursive)
        x0_pred1 = self.model(xt, t, y, torch.zeros_like(xt), mask)
        x0_pred2 = self.model(xt, t, y, x0_pred1.detach(), mask)
        rec_loss = F.l1_loss(x0_pred1, x0) + 0.5 * F.l1_loss(x0_pred2, x0)
        lambda_adv = 0.02
        # Adversarial on posterior sample
        xt_prev_fake = self.q_posterior(xt, t, x0_pred2.detach(), y)
        d_fake = self.discriminator(xt_prev_fake, xt)
        adv_loss = F.binary_cross_entropy_with_logits(d_fake, torch.ones_like(d_fake))
        return rec_loss + lambda_adv * adv_loss

    def training_loss_d(self, x0, y, mask):
        """Discriminator loss: real vs fake posterior samples."""
        B = x0.shape[0]
        t = torch.randint(1, self.T + 1, (B,), device=self.device)
        xt, _ = self.q_sample(x0, y, t)
        with torch.no_grad():
            x0_pred = self.model(xt, t, y, torch.zeros_like(xt), mask)
            x0_pred = self.model(xt, t, y, x0_pred, mask)
            xt_prev_fake = self.q_posterior(xt, t, x0_pred, y)
        xt_prev_real = self.q_posterior(xt, t, x0, y)
        d_real = self.discriminator(xt_prev_real, xt)
        d_fake = self.discriminator(xt_prev_fake, xt)
        return F.binary_cross_entropy_with_logits(d_real, torch.ones_like(d_real)) + \
               F.binary_cross_entropy_with_logits(d_fake, torch.zeros_like(d_fake))

    @torch.no_grad()
    def sample(self, y, mask, num_steps=20, num_recursions=5):
        B, C, H, W = y.shape
        xt = y + math.sqrt(self.gamma) * torch.randn(B, C, H, W, device=self.device)
        step_indices = torch.linspace(self.T, 0, num_steps + 1, dtype=torch.long, device=self.device)
        for k in range(num_steps):
            t_curr = step_indices[k]; t_next = step_indices[k + 1]
            t_batch = t_curr.expand(B)
            # Self-consistent recursive estimation
            x0_est = torch.zeros_like(xt)
            for _ in range(num_recursions):
                x0_new = self.model(xt, t_batch, y, x0_est, mask)
                if (x0_new - x0_est).abs().mean().item() < 1e-4: break
                x0_est = x0_new
            da = (self.a[t_next] - self.a[t_curr]).item()
            db = (self.b[t_next] - self.b[t_curr]).item()
            xt = xt + da * x0_new + db * y
        # Final estimate at t=0
        x0_final = torch.zeros_like(xt)
        for _ in range(num_recursions):
            x0_new = self.model(xt, torch.zeros(B, dtype=torch.long, device=self.device), y, x0_final, mask)
            if (x0_new - x0_final).abs().mean().item() < 1e-4: break
            x0_final = x0_new
        return x0_new
