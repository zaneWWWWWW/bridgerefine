# -*- coding: utf-8 -*-
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


class ZeroConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)
        nn.init.zeros_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)

    def forward(self, x):
        return self.conv(x)


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim, dropout=0.0):
        super().__init__()
        self.norm1 = nn.GroupNorm(min(32, in_ch), in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.time_proj = nn.Linear(time_emb_dim, out_ch)
        self.norm2 = nn.GroupNorm(min(32, out_ch), out_ch)
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
        self.norm = nn.GroupNorm(min(32, channels), channels)
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


class ControlNetEncoder(nn.Module):
    def __init__(self, in_ch=2, base_ch=64, ch_mult=(1,2,4), num_res_blocks=2, dropout=0.0):
        super().__init__()
        self.num_resolutions = len(ch_mult)
        self.conv_in = nn.Conv2d(in_ch, base_ch, 3, padding=1)
        self.down_blocks = nn.ModuleList()
        self.zero_convs = nn.ModuleList()
        ch = base_ch
        time_emb_dim = base_ch * 4
        for i_level in range(self.num_resolutions):
            block_ch = base_ch * ch_mult[i_level]
            for _ in range(num_res_blocks):
                self.down_blocks.append(ResBlock(ch, block_ch, time_emb_dim, dropout))
                ch = block_ch
            self.zero_convs.append(ZeroConv(ch, ch))
            if i_level < self.num_resolutions - 1:
                self.down_blocks.append(nn.Conv2d(ch, ch, 3, stride=2, padding=1))
        self.mid_block1 = ResBlock(ch, ch, time_emb_dim, dropout)
        self.mid_attn = SelfAttention(ch)
        self.mid_block2 = ResBlock(ch, ch, time_emb_dim, dropout)
        self.mid_zero_conv = ZeroConv(ch, ch)

    def forward(self, x):
        t_dummy = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
        t_emb = get_timestep_embedding(t_dummy, self.conv_in.out_channels * 4)
        x = self.conv_in(x)
        features = []
        zero_idx = 0
        for layer in self.down_blocks:
            if isinstance(layer, ResBlock):
                x = layer(x, t_emb)
            elif isinstance(layer, nn.Conv2d):
                features.append(self.zero_convs[zero_idx](x))
                zero_idx += 1
                x = layer(x)
        features.append(self.zero_convs[zero_idx](x))
        x = self.mid_block1(x, t_emb)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t_emb)
        mid_feat = self.mid_zero_conv(x)
        return features, mid_feat


class ControlUNet(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, cond_ch=2, base_ch=64, ch_mult=(1,2,4),
                 num_res_blocks=2, dropout=0.0, time_emb_dim=None):
        super().__init__()
        self.base_ch = base_ch
        time_emb_dim = time_emb_dim or base_ch * 4
        self.num_resolutions = len(ch_mult)
        self.time_mlp = nn.Sequential(
            nn.Linear(base_ch, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )
        self.conv_in = nn.Conv2d(in_ch, base_ch, 3, padding=1)

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
                self.up_blocks.append(nn.Upsample(scale_factor=2, mode='nearest'))
                self.up_blocks.append(nn.Conv2d(ch, ch, 3, padding=1))

        self.norm_out = nn.GroupNorm(min(32, ch), ch)
        self.conv_out = nn.Conv2d(ch, out_ch, 3, padding=1)

        self.control_encoder = ControlNetEncoder(
            in_ch=cond_ch, base_ch=base_ch, ch_mult=ch_mult,
            num_res_blocks=num_res_blocks, dropout=dropout,
        )

    def forward(self, x, t, condition, edge_map=None):
        t_emb = get_timestep_embedding(t, self.base_ch)
        t_emb = self.time_mlp(t_emb)

        if edge_map is not None:
            ctrl_input = torch.cat([condition, edge_map], dim=1)
        else:
            ctrl_input = torch.cat([condition, torch.zeros_like(condition)], dim=1)
        ctrl_features, ctrl_mid = self.control_encoder(ctrl_input)
        ctrl_features = list(reversed(ctrl_features))

        x = self.conv_in(x)
        skips = [x]
        for layer in self.down_blocks:
            if isinstance(layer, ResBlock):
                x = layer(x, t_emb)
            else:
                x = layer(x)
            skips.append(x)

        x = self.mid_block1(x, t_emb)
        x = x + ctrl_mid
        x = self.mid_attn(x)
        x = self.mid_block2(x, t_emb)

        ctrl_idx = 0
        for layer in self.up_blocks:
            if isinstance(layer, ResBlock):
                skip = skips.pop()
                x = torch.cat([x, skip], dim=1)
                x = layer(x, t_emb)
            elif isinstance(layer, nn.Upsample):
                if ctrl_idx < len(ctrl_features):
                    x = x + ctrl_features[ctrl_idx]
                    ctrl_idx += 1
                x = layer(x)
            else:
                x = layer(x)

        return self.conv_out(F.silu(self.norm_out(x)))


class GaussianDiffusion:
    def __init__(self, model, timesteps=1000, beta_start=1e-4, beta_end=0.02, device='cpu'):
        self.model = model
        self.timesteps = timesteps
        self.device = device
        betas = torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float32)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.betas = betas.to(device)
        self.alphas = alphas.to(device)
        self.alphas_cumprod = alphas_cumprod.to(device)
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod).to(device)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod).to(device)

    def q_sample(self, x0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x0)
        sqrt_alpha = self.sqrt_alphas_cumprod[t].view(-1,1,1,1)
        sqrt_one_minus = self.sqrt_one_minus_alphas_cumprod[t].view(-1,1,1,1)
        return sqrt_alpha * x0 + sqrt_one_minus * noise, noise

    @torch.no_grad()
    def ddim_sample(self, condition, edge_map=None, ddim_steps=20, eta=0.0):
        batch_size = condition.shape[0]
        h, w = condition.shape[2], condition.shape[3]
        times = torch.linspace(self.timesteps - 1, 0, ddim_steps, dtype=torch.long, device=self.device)
        x_t = torch.randn(batch_size, 1, h, w, device=self.device)
        for i in range(len(times) - 1):
            t_prev = times[i]
            t_next = times[i + 1]
            t_batch = torch.full((batch_size,), t_prev, dtype=torch.long, device=self.device)
            eps_pred = self.model(x_t, t_batch, condition, edge_map)
            alpha_prev = self.alphas_cumprod[t_prev]
            alpha_next = self.alphas_cumprod[t_next] if t_next >= 0 else torch.tensor(1.0, device=self.device)
            sqrt_alpha_next = torch.sqrt(alpha_next)
            pred_x0 = (x_t - torch.sqrt(1 - alpha_prev) * eps_pred) / torch.sqrt(alpha_prev)
            pred_x0 = pred_x0.clamp(-1, 1)
            dir_xt = torch.sqrt(1 - alpha_next)
            x_t = sqrt_alpha_next * pred_x0 + dir_xt * eps_pred
        return x_t

    def training_loss(self, x0, condition, edge_map=None):
        batch_size = x0.shape[0]
        t = torch.randint(0, self.timesteps, (batch_size,), device=self.device, dtype=torch.long)
        x_t, noise = self.q_sample(x0, t)
        eps_pred = self.model(x_t, t, condition, edge_map)
        return F.mse_loss(eps_pred, noise)
