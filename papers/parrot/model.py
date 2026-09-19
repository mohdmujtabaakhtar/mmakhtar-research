"""PyTorch implementation of PARROT from "PARROT: Synergizing Mamba and Attention-based SSL
Pre-Trained Models via Parallel Branch Hadamard Optimal Transport for Speech Emotion
Recognition" (INTERSPEECH 2025).

Each pre-trained model (PTM) embedding goes through two conv blocks (Conv1d 64, 128,
k=3, ReLU, max-pool), is flattened and linearly projected to 120 dimensions (R_p, R_q).
Two fusion branches run in parallel:

* Hadamard Product Fusion Block (HPFB):  HP = R_p * R_q
* Optimal Transport Fusion Block (OTFB): C = ||R_p - R_q||_2 / max, Gamma = Sinkhorn(C),
  R_p->R_q = Gamma . R_p, R_q->R_p = Gamma^T . R_q,
  F_q = [R_p->R_q, R_q], F_p = [R_q->R_p, R_p]

The branch outputs are concatenated and passed to a dense layer of 128 units and a
softmax output. ``fusion="concat"`` removes both branches (concatenation baseline).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import nn
import torch.nn.functional as F

from common.layers import EmbeddingCNN, dense_block
from common.ot import ot_exchange


@dataclass
class ParrotConfig:
    input_dims: dict = field(default_factory=lambda: {"fm1": 3840, "fm2": 768})
    num_classes: int = 6
    filters: tuple = (64, 128)   # (paper)
    proj_dim: int = 120          # (paper)
    hidden: tuple = (128,)       # (paper)
    dropout: float = 0.3
    ot_epsilon: float = 0.05
    ot_iters: int = 50
    fusion: str = "parrot"       # parrot | hadamard | ot | concat


class Parrot(nn.Module):
    def __init__(self, cfg: ParrotConfig):
        super().__init__()
        self.cfg = cfg
        self.streams = list(cfg.input_dims)[:2]
        self.cnn = nn.ModuleDict({s: EmbeddingCNN(cfg.input_dims[s], tuple(cfg.filters)) for s in self.streams})
        self.proj = nn.ModuleDict({s: nn.Linear(self.cnn[s].flat_dim, cfg.proj_dim) for s in self.streams})
        width = {"parrot": 5, "hadamard": 1, "ot": 4, "concat": 2}[cfg.fusion] * cfg.proj_dim
        self.fc, width = dense_block(width, cfg.hidden, cfg.dropout)
        self.out = nn.Linear(width, cfg.num_classes)

    def fuse(self, rp: torch.Tensor, rq: torch.Tensor) -> torch.Tensor:
        parts = []
        if self.cfg.fusion in ("parrot", "hadamard"):
            parts.append(rp * rq)                                              # HPFB
        if self.cfg.fusion in ("parrot", "ot"):
            q_to_p, p_to_q, _ = ot_exchange(rp, rq, self.cfg.ot_epsilon, self.cfg.ot_iters)
            parts += [p_to_q, rq, q_to_p, rp]                                  # F_q, F_p
        if self.cfg.fusion == "concat":
            parts += [rp, rq]
        return torch.cat(parts, dim=-1)

    def forward(self, batch: dict) -> dict:
        rp, rq = (self.proj[s](self.cnn[s](batch[s]).flatten(1)) for s in self.streams)
        return {"logits": self.out(self.fc(self.fuse(rp, rq)))}

    def loss(self, out: dict, batch: dict):
        ce = F.cross_entropy(out["logits"], batch["label"])
        return ce, {"ce": ce.item()}
