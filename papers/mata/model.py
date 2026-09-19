"""PyTorch implementation of MATA from "Strong Alone, Stronger Together: Synergizing
Modality-Binding Foundation Models with Optimal Transport for Non-Verbal Emotion
Recognition" (ICASSP 2025).

MATA (Intra-Modality Alignment through Transport Attention):

    each FM embedding -> two conv blocks (Conv1d 32, 64, k=3, max-pool) -> flatten
                      -> linear projection to 120-d (x1, x2)
    M = ||x1 - x2||_2 / max,  gamma = Sinkhorn(M)
    x2->x1 = gamma . x2,  x1->x2 = gamma^T . x1
    fused1 = [x2->x1, x1],  fused2 = [x1->x2, x2]
    each fused vector is concatenated with the *opposite* FM's features:
        [fused1, x2], [fused2, x1]  -> concatenated -> multi-head attention (8 heads)
    -> fully connected network -> softmax

Switches: ``use_mha=False`` is the paper's "Fusion with OT" ablation (no attention);
``fusion="concat"`` is the concatenation baseline.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import nn
import torch.nn.functional as F

from common.layers import EmbeddingCNN, dense_block
from common.ot import ot_exchange


@dataclass
class MataConfig:
    input_dims: dict = field(default_factory=lambda: {"fm1": 768, "fm2": 1024})
    num_classes: int = 6
    filters: tuple = (32, 64)    # (paper)
    proj_dim: int = 120          # (paper)
    heads: int = 8               # (paper)
    hidden: tuple = (128,)
    dropout: float = 0.3
    ot_epsilon: float = 0.05
    ot_iters: int = 50
    use_mha: bool = True
    fusion: str = "mata"         # mata | concat


class Mata(nn.Module):
    def __init__(self, cfg: MataConfig):
        super().__init__()
        self.cfg = cfg
        self.streams = list(cfg.input_dims)[:2]
        self.cnn = nn.ModuleDict({s: EmbeddingCNN(cfg.input_dims[s], tuple(cfg.filters)) for s in self.streams})
        self.proj = nn.ModuleDict({s: nn.Linear(self.cnn[s].flat_dim, cfg.proj_dim) for s in self.streams})
        self.attn = nn.MultiheadAttention(cfg.proj_dim, cfg.heads, dropout=cfg.dropout, batch_first=True)
        self.attn_norm = nn.LayerNorm(cfg.proj_dim)
        tokens = 6 if cfg.fusion == "mata" else 2
        self.fc, width = dense_block(tokens * cfg.proj_dim, cfg.hidden, cfg.dropout)
        self.out = nn.Linear(width, cfg.num_classes)

    def forward(self, batch: dict) -> dict:
        x1, x2 = (self.proj[s](self.cnn[s](batch[s]).flatten(1)) for s in self.streams)
        if self.cfg.fusion == "mata":
            x2_to_x1, x1_to_x2, _ = ot_exchange(x1, x2, self.cfg.ot_epsilon, self.cfg.ot_iters)
            # [fused1, x2] and [fused2, x1], kept as a sequence of 120-d tokens for attention
            tokens = torch.stack([x2_to_x1, x1, x2, x1_to_x2, x2, x1], dim=1)      # (B, 6, 120)
            if self.cfg.use_mha:
                attended, _ = self.attn(tokens, tokens, tokens)
                tokens = self.attn_norm(tokens + attended)
        else:
            tokens = torch.stack([x1, x2], dim=1)
        return {"logits": self.out(self.fc(tokens.flatten(1)))}

    def loss(self, out: dict, batch: dict):
        ce = F.cross_entropy(out["logits"], batch["label"])
        return ce, {"ce": ce.item()}
