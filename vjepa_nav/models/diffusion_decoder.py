"""Conditional diffusion decoder (DiT) mapping latent h_t -> pixel frame.

A V-JEPA-style decoder: a DiT trained to denoise a 256x256 frame conditioned
(via cross-attention) on the latent tokens h_t of that future frame.
ε-prediction with a linear DDPM schedule; DDIM sampling at eval.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
#  Diffusion schedule
# --------------------------------------------------------------------------- #

def linear_betas(num_timesteps: int = 1000, start: float = 1e-4, end: float = 0.02):
    return torch.linspace(start, end, num_timesteps)


def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(half, dtype=torch.float32, device=t.device) / half
    )
    args = t[:, None].float() * freqs[None]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


class GaussianDiffusion:
    def __init__(self, num_timesteps: int = 1000, device="cuda"):
        self.num_timesteps = num_timesteps
        betas = linear_betas(num_timesteps).to(device)
        self.betas = betas
        self.alphas = 1.0 - betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        s = self.sqrt_alphas_cumprod[t][:, None, None, None]
        m = self.sqrt_one_minus_alphas_cumprod[t][:, None, None, None]
        return s * x0 + m * noise

    def ddim_sample(
        self, model: nn.Module, cond: torch.Tensor, shape, steps: int = 50, eta: float = 0.0
    ):
        model.eval()
        device = cond.device
        x = torch.randn(shape, device=device)
        ts = torch.linspace(self.num_timesteps - 1, 0, steps, dtype=torch.long, device=device)
        ts_next = torch.cat([ts[1:], torch.zeros(1, dtype=torch.long, device=device)])
        with torch.no_grad():
            for i in range(steps):
                t = ts[i].expand(shape[0])
                eps = model(x, t, cond)
                alpha_cum = self.alphas_cumprod[ts[i]]
                alpha_cum_next = self.alphas_cumprod[ts_next[i]]
                x0 = (x - torch.sqrt(1 - alpha_cum) * eps) / torch.sqrt(alpha_cum)
                x0 = x0.clamp(-1, 1)
                c1 = eta * torch.sqrt((1 - alpha_cum / alpha_cum_next) * (1 - alpha_cum_next) / (1 - alpha_cum))
                c2 = torch.sqrt((1 - alpha_cum_next) - c1**2)
                x = torch.sqrt(alpha_cum_next) * x0 + c1 * torch.randn_like(x) + c2 * eps
        return x


# --------------------------------------------------------------------------- #
#  DiT blocks
# --------------------------------------------------------------------------- #

class MLP(nn.Module):
    def __init__(self, dim, hidden, act=nn.GELU):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.act = act()
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class AdaLNZeroBlock(nn.Module):
    """Self-attn + cross-attn (condition latents) + MLP, all adaLN-Zero modulated."""

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False)
        self.self_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm_cross = nn.LayerNorm(dim, elementwise_affine=False)
        self.cross_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False)
        self.mlp = MLP(dim, int(dim * mlp_ratio))
        # scale/shift/gate for self-attn, cross-attn, mlp = 9 outputs
        self.adaLN = nn.Sequential(nn.SiLU(), nn.Linear(dim, 9 * dim))

    def forward(self, x, cond, t_emb):
        shift_msa, scale_msa, gate_msa, shift_ca, scale_ca, gate_ca, shift_mlp, scale_mlp, gate_mlp = \
            self.adaLN(t_emb).chunk(9, dim=-1)

        h = modulate(self.norm1(x), shift_msa, scale_msa)
        h, _ = self.self_attn(h, h, h)
        x = x + gate_msa.unsqueeze(1) * h

        h = modulate(self.norm_cross(x), shift_ca, scale_ca)
        h, _ = self.cross_attn(h, cond, cond)
        x = x + gate_ca.unsqueeze(1) * h

        h = modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(h)
        return x


# --------------------------------------------------------------------------- #
#  Conditional DiT
# --------------------------------------------------------------------------- #

class ConditionalDiT(nn.Module):
    def __init__(
        self,
        img_size: int = 256,
        patch_size: int = 8,
        in_chans: int = 3,
        hidden: int = 384,
        depth: int = 8,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        cond_dim: int = 1024,
        cond_len: int = 256,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid = img_size // patch_size
        self.num_patches = self.grid * self.grid
        self.patch_dim = in_chans * patch_size * patch_size
        self.hidden = hidden

        self.patchify = nn.Linear(self.patch_dim, hidden)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, hidden))
        self.t_embed = nn.Sequential(
            nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, hidden)
        )
        self.cond_proj = nn.Linear(cond_dim, hidden)
        self.cond_len = cond_len
        self.cond_pos = nn.Parameter(torch.zeros(1, cond_len, hidden))

        self.blocks = nn.ModuleList(
            [AdaLNZeroBlock(hidden, num_heads, mlp_ratio) for _ in range(depth)]
        )
        self.final_norm = nn.LayerNorm(hidden, elementwise_affine=False)
        self.unpatchify = nn.Linear(hidden, self.patch_dim)

        nn.init.normal_(self.pos_embed, std=0.02)
        nn.init.normal_(self.cond_pos, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor, t: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        x = x.unfold(2, self.patch_size, self.patch_size).unfold(
            3, self.patch_size, self.patch_size
        )
        x = x.permute(0, 2, 3, 1, 4, 5).reshape(B, self.num_patches, self.patch_dim)
        h = self.patchify(x) + self.pos_embed

        t_emb = self.t_embed(timestep_embedding(t, self.hidden))

        cond_tokens = self.cond_proj(cond) + self.cond_pos

        for blk in self.blocks:
            h = blk(h, cond_tokens, t_emb)
        h = self.final_norm(h)
        out = self.unpatchify(h).reshape(
            B, self.grid, self.grid, C, self.patch_size, self.patch_size
        )
        out = out.permute(0, 3, 1, 4, 2, 5).reshape(B, C, H, W)
        return out
