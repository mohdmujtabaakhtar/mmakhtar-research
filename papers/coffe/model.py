"""PyTorch implementation of COFFE from "Towards Source Attribution of Singing Voice
Deepfake with Multimodal Foundation Models" (INTERSPEECH 2025).

COFFE (Fusion using ChernOFF DistancE) fuses two foundation-model representations
for singing-voice deepfake source attribution:

    each FM embedding -> two conv blocks (Conv1d 64, 128, k=3, max-pool 2) -> flatten
    Chernoff distance between the two branches' feature distributions (L_CD)
    concatenate -> dense 128 -> softmax over the source systems

    L = L_CE + lambda * L_CD,   L_CD = -log sum_i p_i^s q_i^(1-s),   s = 0.3, lambda = 0.1

With ``lam = 0`` the same network is the paper's concatenation-fusion baseline.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import nn
import torch.nn.functional as F

from common.divergences import chernoff, to_distribution
from common.layers import EmbeddingCNN, dense_block


@dataclass
class CoffeConfig:
    input_dims: dict = field(default_factory=lambda: {"fm1": 768, "fm2": 1024})
    num_classes: int = 8
    filters: tuple = (64, 128)
    proj_dim: int = 128          # common width of the two branches before the Chernoff distance
    hidden: tuple = (128,)       # FCN block (paper: one dense layer of 128)
    dropout: float = 0.3
    s: float = 0.3               # Chernoff order (paper)
    lam: float = 0.1             # weight of L_CD (paper); 0 = concatenation baseline


class Coffe(nn.Module):
    def __init__(self, cfg: CoffeConfig):
        super().__init__()
        self.cfg = cfg
        self.streams = list(cfg.input_dims)
        self.cnn = nn.ModuleDict({s: EmbeddingCNN(d, tuple(cfg.filters)) for s, d in cfg.input_dims.items()})
        self.proj = nn.ModuleDict({s: nn.Linear(self.cnn[s].flat_dim, cfg.proj_dim) for s in self.streams})
        self.fc, width = dense_block(cfg.proj_dim * len(self.streams), cfg.hidden, cfg.dropout)
        self.out = nn.Linear(width, cfg.num_classes)

    def forward(self, batch: dict) -> dict:
        z = {s: self.proj[s](self.cnn[s](batch[s]).flatten(1)) for s in self.streams}
        fused = torch.cat([z[s] for s in self.streams], dim=-1)
        return {"logits": self.out(self.fc(fused)), "z": z}

    def loss(self, out: dict, batch: dict):
        ce = F.cross_entropy(out["logits"], batch["label"])
        z1, z2 = (out["z"][s] for s in self.streams[:2])
        cd = chernoff(to_distribution(z1), to_distribution(z2), self.cfg.s)
        total = ce + self.cfg.lam * cd
        return total, {"ce": ce.item(), "chernoff": cd.item(), "monitor": ce.item()}
