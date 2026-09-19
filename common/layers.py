"""Small building blocks shared by several papers."""
from __future__ import annotations

import torch
from torch import nn


class ConvAdapter(nn.Module):
    """(B, T, d_in) -> (B, T / 2^blocks, d_out): [Conv1d(k=3) -> BN -> ReLU -> MaxPool(2)] x blocks."""

    def __init__(self, in_dim: int, out_dim: int, blocks: int = 1, pool: bool = True):
        super().__init__()
        layers = []
        for i in range(blocks):
            layers += [nn.Conv1d(in_dim if i == 0 else out_dim, out_dim, 3, padding=1),
                       nn.BatchNorm1d(out_dim), nn.ReLU()]
            if pool:
                layers.append(nn.MaxPool1d(2, ceil_mode=True))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.transpose(1, 2)).transpose(1, 2)


class AttentionPool(nn.Module):
    """Additive attention pooling: alpha_t = softmax(v^T tanh(W x_t)), out = sum_t alpha_t x_t."""

    def __init__(self, dim: int, hidden: int | None = None):
        super().__init__()
        self.proj = nn.Linear(dim, hidden or dim)
        self.v = nn.Linear(hidden or dim, 1, bias=False)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None):
        scores = self.v(torch.tanh(self.proj(x))).squeeze(-1)          # (B, T)
        if mask is not None:
            scores = scores.masked_fill(~mask, float("-inf"))
        weights = scores.softmax(dim=-1)
        return (weights.unsqueeze(-1) * x).sum(1), weights
