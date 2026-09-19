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


def as_vector(x: torch.Tensor) -> torch.Tensor:
    """(B, D) stays as is; (B, T, D) is averaged over time (utterance-level embedding)."""
    return x.mean(dim=1) if x.dim() == 3 else x


class EmbeddingCNN(nn.Module):
    """1-D CNN over an utterance-level embedding treated as a one-channel sequence.

    Many of the fusion papers pool a foundation model's last hidden layer to a single
    vector of size D and then run ``[Conv1d(k) -> ReLU -> MaxPool(2)] x n`` along the
    feature axis. Input (B, D) or (B, T, D); output feature map (B, C_last, L) with
    ``L = D // 2**n``. ``flat_dim`` is the size after flattening.
    """

    def __init__(self, in_dim: int, filters=(64, 128), kernel: int = 3, pool: int = 2,
                 batch_norm: bool = False):
        super().__init__()
        layers, c_in, length = [], 1, in_dim
        for c in filters:
            layers += [nn.Conv1d(c_in, c, kernel, padding=kernel // 2)]
            if batch_norm:
                layers.append(nn.BatchNorm1d(c))
            layers += [nn.ReLU(), nn.MaxPool1d(pool)]
            c_in, length = c, length // pool
        self.net = nn.Sequential(*layers)
        self.channels, self.length = c_in, length
        self.flat_dim = c_in * length

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(as_vector(x).unsqueeze(1))


def dense_block(in_dim: int, dims, dropout: float = 0.3) -> tuple[nn.Sequential, int]:
    """Stack of Linear -> ReLU -> Dropout layers; returns the block and its output width."""
    layers, d = [], in_dim
    for h in dims:
        layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(dropout)]
        d = h
    return nn.Sequential(*layers), d
