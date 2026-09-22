"""PyTorch implementation of TRIO from "Source Tracing of Synthetic Speech Systems Through
Paralinguistic Pre-Trained Representations" (EUSIPCO 2025).

TRIO (GaTed Canonical CorRelatIOn Attention Network) fuses two speech pre-trained
model (SPTM) representations for source tracing:

    each SPTM embedding -> two conv blocks (Conv1d 128, 64, k=3, max-pool 2) -> X, Y
    sigmoid gates:   X^ = G_X * X,  Y^ = G_Y * Y
    CCA loss:        L_CCA = tr( S_xx^-1/2 S_xy S_yy^-1/2 ) (maximised)
    concatenate -> multi-head self-attention -> dense 90 -> dense 45 -> softmax

    L = L_CE - lambda * L_CCA,  lambda = 0.3

``fusion="concat"`` removes the gates, the CCA loss and the self-attention block,
which is the paper's concatenation baseline.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import nn
import torch.nn.functional as F

from common.divergences import cca_objective
from common.layers import EmbeddingCNN, dense_block


@dataclass
class TrioConfig:
    input_dims: dict = field(default_factory=lambda: {"fm1": 512, "fm2": 1024})
    num_classes: int = 19
    filters: tuple = (128, 64)   # (paper)
    hidden: tuple = (90, 45)     # (paper)
    heads: int = 4
    cca_dim: int = 16            # width of the projections on which CCA is computed
    cca_reg: float = 1e-3
    lam: float = 0.3             # (paper)
    dropout: float = 0.3
    fusion: str = "trio"         # trio | concat


class Trio(nn.Module):
    def __init__(self, cfg: TrioConfig):
        super().__init__()
        self.cfg = cfg
        self.streams = list(cfg.input_dims)
        self.cnn = nn.ModuleDict({s: EmbeddingCNN(d, tuple(cfg.filters)) for s, d in cfg.input_dims.items()})
        ch = cfg.filters[-1]
        self.gate = nn.ModuleDict({s: nn.Conv1d(ch, ch, 1) for s in self.streams})
        self.cca_proj = nn.ModuleDict({s: nn.Linear(self.cnn[s].flat_dim, cfg.cca_dim) for s in self.streams})
        self.attn = nn.MultiheadAttention(ch, cfg.heads, dropout=cfg.dropout, batch_first=True)
        self.attn_norm = nn.LayerNorm(ch)
        flat = sum(self.cnn[s].flat_dim for s in self.streams)
        self.fc, width = dense_block(flat, cfg.hidden, cfg.dropout)
        self.out = nn.Linear(width, cfg.num_classes)

    def forward(self, batch: dict) -> dict:
        maps = {s: self.cnn[s](batch[s]) for s in self.streams}            # (B, C, L_s)
        out = {}
        if self.cfg.fusion == "trio":
            maps = {s: torch.sigmoid(self.gate[s](m)) * m for s, m in maps.items()}
            out["cca_views"] = [self.cca_proj[s](maps[s].flatten(1)) for s in self.streams]
        tokens = torch.cat([maps[s].transpose(1, 2) for s in self.streams], dim=1)   # (B, L1 + L2, C)
        if self.cfg.fusion == "trio":
            attended, _ = self.attn(tokens, tokens, tokens)
            tokens = self.attn_norm(tokens + attended)          # residual self-attention refinement
        out["logits"] = self.out(self.fc(tokens.flatten(1)))
        return out

    def loss(self, out: dict, batch: dict):
        ce = F.cross_entropy(out["logits"], batch["label"])
        parts = {"ce": ce.item(), "monitor": ce.item()}
        total = ce
        if "cca_views" in out and out["logits"].shape[0] > 2:
            cca = cca_objective(*out["cca_views"], reg=self.cfg.cca_reg)
            total = ce - self.cfg.lam * cca
            parts["cca"] = cca.item()
        return total, parts
