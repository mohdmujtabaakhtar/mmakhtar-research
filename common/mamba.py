"""A dependency-free Mamba (selective state-space) block in plain PyTorch.

Follows the S6 layer of Gu & Dao (2023): input-dependent step size, B and C,
a diagonal state matrix A = -exp(A_log), and a gated output. The scan runs
sequentially over time, which is fine for the utterance lengths used here;
swap in ``mamba_ssm`` for long sequences on GPU.
"""
from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F


class SelectiveSSM(nn.Module):
    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        d_inner = expand * d_model
        self.d_inner, self.d_state = d_inner, d_state
        self.dt_rank = math.ceil(d_model / 16)
        self.in_proj = nn.Linear(d_model, 2 * d_inner, bias=False)
        self.conv = nn.Conv1d(d_inner, d_inner, d_conv, groups=d_inner, padding=d_conv - 1)
        self.x_proj = nn.Linear(d_inner, self.dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, d_inner)
        a = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(d_inner, 1)
        self.A_log = nn.Parameter(torch.log(a))
        self.D = nn.Parameter(torch.ones(d_inner))
        self.out_proj = nn.Linear(d_inner, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:        # (B, T, d_model)
        t = x.shape[1]
        u, z = self.in_proj(x).chunk(2, dim=-1)
        u = F.silu(self.conv(u.transpose(1, 2))[..., :t].transpose(1, 2))   # causal depthwise conv
        dt, b, c = self.x_proj(u).split([self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = F.softplus(self.dt_proj(dt))                                     # (B, T, d_inner)
        a = -torch.exp(self.A_log)                                            # (d_inner, N)
        decay = torch.exp(dt.unsqueeze(-1) * a)                               # (B, T, d_inner, N)
        drive = dt.unsqueeze(-1) * b.unsqueeze(2) * u.unsqueeze(-1)           # (B, T, d_inner, N)
        h = torch.zeros(x.shape[0], self.d_inner, self.d_state, device=x.device, dtype=x.dtype)
        ys = []
        for i in range(t):
            h = decay[:, i] * h + drive[:, i]
            ys.append((h * c[:, i].unsqueeze(1)).sum(-1))
        y = torch.stack(ys, dim=1) + u * self.D
        return self.out_proj(y * F.silu(z))


class MambaBlock(nn.Module):
    """Pre-norm residual block: x + SSM(LN(x)), then + GatedMLP(LN(x))."""

    def __init__(self, d_model: int, d_state: int = 16, dropout: float = 0.1):
        super().__init__()
        self.norm1, self.norm2 = nn.LayerNorm(d_model), nn.LayerNorm(d_model)
        self.ssm = SelectiveSSM(d_model, d_state)
        self.gate = nn.Linear(d_model, 4 * d_model)
        self.down = nn.Linear(2 * d_model, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop(self.ssm(self.norm1(x)))
        a, g = self.gate(self.norm2(x)).chunk(2, dim=-1)
        return x + self.drop(self.down(a * F.silu(g)))
