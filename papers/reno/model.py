"""PyTorch implementation of RENO from "Are Mamba-Based Audio Foundation Models the Best Fit
for Non-Verbal Emotion Recognition?" (EUSIPCO 2025).

RENO (RENyi AttentiOn Network) fuses two foundation-model representations:

    each FM embedding -> two conv blocks (Conv1d 32, 64, k=3, max-pool)
                      -> multi-head self-attention (2 heads, intra-representation)
                      -> flatten
    Renyi divergence between the two feature distributions:
        L_RD = 1/(beta-1) log sum_j (z_x,j + delta)^beta (z_y,j + delta)^(1-beta),  beta = 2, delta = 0.2
    concatenate -> multi-head self-attention (2 heads) -> dense 512 -> dense 128 -> softmax

    L = lambda L_CE + (1 - lambda) L_RD,  lambda = 0.4

``fusion="concat"`` removes the Renyi loss and both self-attention blocks
(the paper's concatenation baseline).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import nn
import torch.nn.functional as F

from common.divergences import renyi, to_distribution
from common.layers import EmbeddingCNN, dense_block


class SelfAttention(nn.Module):
    """Multi-head self-attention with a residual connection and layer norm, (B, T, d) -> (B, T, d)."""

    def __init__(self, dim: int, heads: int, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x + self.attn(x, x, x)[0])


@dataclass
class RenoConfig:
    input_dims: dict = field(default_factory=lambda: {"fm1": 3840, "fm2": 768})
    num_classes: int = 6
    filters: tuple = (32, 64)    # (paper)
    heads: int = 2               # both self-attention blocks (paper)
    align_dim: int = 256         # common width on which the Renyi divergence is computed
    hidden: tuple = (512, 128)   # (paper)
    dropout: float = 0.3
    beta: float = 2.0            # (paper)
    delta: float = 0.2           # (paper)
    lam: float = 0.4             # (paper)
    fusion: str = "reno"         # reno | concat


class Reno(nn.Module):
    def __init__(self, cfg: RenoConfig):
        super().__init__()
        self.cfg = cfg
        self.streams = list(cfg.input_dims)[:2]
        self.cnn = nn.ModuleDict({s: EmbeddingCNN(cfg.input_dims[s], tuple(cfg.filters)) for s in self.streams})
        self.intra = nn.ModuleDict({s: SelfAttention(cfg.filters[-1], cfg.heads) for s in self.streams})
        self.proj = nn.ModuleDict({s: nn.Linear(self.cnn[s].flat_dim, cfg.align_dim) for s in self.streams})
        self.inter = SelfAttention(cfg.align_dim, cfg.heads)
        self.fc, width = dense_block(2 * cfg.align_dim, cfg.hidden, cfg.dropout)
        self.out = nn.Linear(width, cfg.num_classes)

    def forward(self, batch: dict) -> dict:
        z = {}
        for s in self.streams:
            tokens = self.cnn[s](batch[s]).transpose(1, 2)            # (B, L, C): positions as tokens
            if self.cfg.fusion == "reno":
                tokens = self.intra[s](tokens)
            z[s] = self.proj[s](tokens.flatten(1))
        fused = torch.stack([z[s] for s in self.streams], dim=1)      # (B, 2, align_dim)
        if self.cfg.fusion == "reno":
            fused = self.inter(fused)
        return {"logits": self.out(self.fc(fused.flatten(1))), "z": z}

    def loss(self, out: dict, batch: dict):
        ce = F.cross_entropy(out["logits"], batch["label"])
        if self.cfg.fusion != "reno":
            return ce, {"ce": ce.item()}
        zx, zy = (out["z"][s] for s in self.streams)
        rd = renyi(to_distribution(zx), to_distribution(zy), self.cfg.beta, self.cfg.delta)
        total = self.cfg.lam * ce + (1 - self.cfg.lam) * rd
        return total, {"ce": ce.item(), "renyi": rd.item(), "monitor": ce.item()}
