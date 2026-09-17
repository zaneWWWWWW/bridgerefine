# -*- coding: utf-8 -*-
"""
SelfRDB: Self-Consistent Recursive Diffusion Bridge
Core diffusion bridge mathematics for CT → MRI translation.

Key differences from standard DDPM:
1. Forward process: x_t starts from target (MRI), ends at noisy source (CT+noise)
   x_t = (1 - t/T)·x_0 + (t/T)·y + √(γ·t/T)·ε
2. Monotonically increasing variance: σ_t² = γ·t/T (not peaked at T/2)
3. Self-consistent recursive estimation at each reverse step
4. x_0-prediction instead of ε-prediction

Reference: Arslan et al., "Self-Consistent Recursive Diffusion Bridge for
Medical Image Translation", Medical Image Analysis, 2025.
"""
import math
import torch
import torch.nn.functional as F


class DiffusionBridge:
    """
    SelfRDB Diffusion Bridge.

    Forward:  x_t = a_t·x_0 + b_t·y + c_t·ε
    Reverse:  x_{t-1} = x_t + Δa·x̂_0 + Δb·y + noise
    where x_0 = target (MRI), y = source (CT)
    """
    def __init__(self, model, timesteps=500, gamma=0.1, device='cpu'):
        """
        Args:
            model: x_0-prediction network G_θ(x_t, t, y, x̂_0_prev) → x̂_0_new
            timesteps: total diffusion steps T
            gamma: soft-prior strength (noise level at t=T, endpoint)
            device: torch device
        """
        self.model = model
        self.T = timesteps
        self.gamma = gamma
        self.device = device

        # Precompute schedule coefficients
        t = torch.arange(0, timesteps + 1, dtype=torch.float32, device=device)
        # SelfRDB: monotonically increasing variance
        # With constant diffusion coefficient g(tau)=1:
        #   s_t² = t,  s̄_t² = T-t,  S² = T
        #   a_t = (T-t)/T,  b_t = t/T
        #   σ_t² = γ·t/T
        self.a = (timesteps - t) / timesteps                # (T+1,)
        self.b = t / timesteps                                # (T+1,)
        self.var = gamma * t / timesteps                      # (T+1,) monotonically increasing

        self.sqrt_var = torch.sqrt(self.var)
        # For step size Δ=1: increment in mean per step
        self.delta_a = self.a[1:] - self.a[:-1]               # = -1/T
        self.delta_b = self.b[1:] - self.b[:-1]               # =  1/T
        self.step_var = self.var[1:] - self.var[:-1]           # = γ/T
        self.sqrt_step_var = torch.sqrt(self.step_var)

    def q_sample(self, x0, y, t):
        """Forward: x_t = a_t·x_0 + b_t·y + sqrt(var_t)·ε"""
        a_t = self.a[t].view(-1, 1, 1, 1)
        b_t = self.b[t].view(-1, 1, 1, 1)
        s_t = self.sqrt_var[t].view(-1, 1, 1, 1)
        noise = torch.randn_like(x0)
        xt = a_t * x0 + b_t * y + s_t * noise
        return xt, noise

    def q_posterior(self, xt, t, x0_hat, y):
        """
        Posterior: q(x_{t-1} | x_t, x̂_0, y)

        For constant g, with Δ=1 step:
            x_{t-1} = x_t + (x̂_0 - y)/T + sqrt(γ/T)·ε
        """
        B = xt.shape[0]
        t_idx = t.clamp(1, self.T)  # ensure t >= 1
        delta_a = self.delta_a[t_idx - 1].view(B, 1, 1, 1)
        delta_b = self.delta_b[t_idx - 1].view(B, 1, 1, 1)
        step_std = self.sqrt_step_var[t_idx - 1].view(B, 1, 1, 1)

        mean = xt + delta_a * x0_hat + delta_b * y
        noise = torch.randn_like(xt)
        xt_prev = mean + step_std * noise
        return xt_prev

    @torch.no_grad()
    def sample_ddib(self, y, num_steps=20, num_recursions=5, tol=1e-4, eta=0.0):
        """
        DDIB-accelerated sampling with self-consistent recursive estimation.

        Args:
            y: source image (CT) [B, 1, H, W]
            num_steps: number of accelerated sampling steps (K << T)
            num_recursions: max recursion depth per step
            tol: convergence tolerance for self-consistency
            eta: stochasticity (0=deterministic DDIM-like, 1=full stochastic)

        Returns:
            x0_hat: generated target image (MRI)
        """
        B, C, H, W = y.shape

        # Start from noisy source: x_T = y + sqrt(gamma)·ε
        xt = y + math.sqrt(self.gamma) * torch.randn(B, C, H, W, device=self.device)

        # Choose K evenly-spaced timesteps from T down to 0
        step_indices = torch.linspace(self.T, 0, num_steps + 1, dtype=torch.long, device=self.device)

        for k in range(num_steps):
            t_curr = step_indices[k]
            t_next = step_indices[k + 1]

            # Self-consistent recursive estimation of x_0
            x0_hat = self._recursive_estimate(xt, t_curr, y, num_recursions, tol)

            # Jump from t_curr to t_next using bridge dynamics
            a_curr = self.a[t_curr]
            a_next = self.a[t_next]
            b_curr = self.b[t_curr]
            b_next = self.b[t_next]
            var_curr = self.var[t_curr]
            var_next = self.var[t_next]

            # Deterministic update (generalized DDIM for bridge)
            delta_a = (a_next - a_curr).item()
            delta_b = (b_next - b_curr).item()
            delta_var = (var_next - var_curr).item()

            xt = xt + delta_a * x0_hat + delta_b * y

            if eta > 0 and delta_var > 0:
                xt = xt + eta * math.sqrt(abs(delta_var)) * torch.randn_like(xt)

        # Final estimate at t=0
        x0_hat = self._recursive_estimate(xt, torch.zeros(1, dtype=torch.long, device=self.device), y, num_recursions, tol)
        return x0_hat

    def _recursive_estimate(self, xt, t, y, max_iters, tol):
        """Self-consistent recursive x_0 estimation at timestep t."""
        t_batch = t.expand(xt.shape[0]) if t.dim() == 0 else t

        # Initial estimate from xt (heuristic: denoise with bridge coefficients)
        a_t = self.a[t].view(-1, 1, 1, 1) if t.dim() == 0 else self.a[t].view(-1, 1, 1, 1)
        b_t = self.b[t].view(-1, 1, 1, 1) if t.dim() == 0 else self.b[t].view(-1, 1, 1, 1)
        # Simple initial guess: extrapolate
        x0_prev = torch.zeros_like(xt)  # start from zero

        for r in range(max_iters):
            x0_new = self.model(xt, t_batch, y, x0_prev)

            if r > 0:
                diff = (x0_new - x0_prev).abs().mean().item()
                if diff < tol:
                    break

            x0_prev = x0_new

        return x0_new

    def training_loss(self, x0, y):
        """
        Training loss: two-stage recursive x0 prediction.
        Stage 1: predict from zero init
        Stage 2: refine using stage 1 output
        This teaches the model to iteratively improve predictions.
        """
        B = x0.shape[0]
        t = torch.randint(1, self.T + 1, (B,), device=self.device, dtype=torch.long)
        xt, _ = self.q_sample(x0, y, t)

        # Stage 1: one-shot prediction
        x0_pred1 = self.model(xt, t, y, torch.zeros_like(xt))

        # Stage 2: recursive refinement
        x0_pred2 = self.model(xt, t, y, x0_pred1.detach())

        return F.l1_loss(x0_pred1, x0) + 0.5 * F.l1_loss(x0_pred2, x0)
