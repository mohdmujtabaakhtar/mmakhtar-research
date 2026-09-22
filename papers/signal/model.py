"""PyTorch implementation of SIGNAL from "Bridging Attribution and Open-Set Detection using
Graph-Augmented Instance Learning in Synthetic Speech" (EACL 2026).

SIGNAL combines a graph-attention head over learnable generator prototypes (source
attribution) with a distance-weighted KNN over training embeddings (open-set
reasoning):

    z0 = frozen speech foundation model embedding
    z  = f_cnn(z0) in R^64          (two Conv1d + ReLU + max-pool, then a dense projection)

    GNN head:  s = W_s z;  e~_i = e_i + s  (one node per seen generator)
               e~'_i = MultiHeadAttn(e~_i, {e~_j});  l_i = w^T e~'_i;  p_GNN = softmax(l)
               H_attn = -sum_i p_GNN,i log p_GNN,i  (uncertainty signal)
    KNN:       p_KNN = sum_k w_k y_k / sum_k w_k,   w_k = 1 / (||z - z_k||^2 + eps)
    Ensemble:  p_ens = alpha p_GNN + (1 - alpha) p_KNN
    Open set:  "unseen generator" if max(p_ens) < tau   (tau = 0.5)

Only the encoder and the GNN head are trained (cross-entropy over the seen
generators); the KNN bank is filled with training embeddings after training.
``alpha = 1`` gives the GNN-only model and ``alpha = 0`` the KNN-only model.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F

from common.layers import EmbeddingCNN


@dataclass
class SignalConfig:
    input_dim: int = 3840
    num_classes: int = 8          # seen generators
    filters: tuple = (64, 128)
    embed_dim: int = 64           # d (paper)
    heads: int = 4
    k: int = 10                   # neighbours in the KNN branch
    knn_eps: float = 1e-6
    alpha: float = 0.5            # ensemble weight; 1 = GNN only, 0 = KNN only
    tau: float = 0.5              # open-set confidence threshold (paper)
    dropout: float = 0.1


class Signal(nn.Module):
    def __init__(self, cfg: SignalConfig):
        super().__init__()
        self.cfg = cfg
        self.cnn = EmbeddingCNN(cfg.input_dim, tuple(cfg.filters))
        self.embed = nn.Linear(self.cnn.flat_dim, cfg.embed_dim)            # dense projection -> z
        d = cfg.embed_dim
        self.prototypes = nn.Parameter(torch.randn(cfg.num_classes, d))   # e_1..e_N
        self.W_s = nn.Linear(d, d, bias=False)
        self.attn = nn.MultiheadAttention(d, cfg.heads, dropout=cfg.dropout, batch_first=True)
        self.norm = nn.LayerNorm(d)
        self.w = nn.Linear(d, 1)
        self.register_buffer("bank_z", torch.zeros(0, d))
        self.register_buffer("bank_y", torch.zeros(0, dtype=torch.long))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.embed(self.cnn(x).flatten(1))

    def gnn(self, z: torch.Tensor):
        nodes = self.prototypes[None] + self.W_s(z)[:, None]              # (B, N, d): e~_i = e_i + s
        refined = self.norm(nodes + self.attn(nodes, nodes, nodes)[0])    # message passing among class nodes
        logits = self.w(refined).squeeze(-1)                              # l_i = w^T e~'_i
        p = logits.softmax(-1)
        entropy = -(p * p.clamp_min(1e-12).log()).sum(-1)
        return logits, p, entropy

    @torch.no_grad()
    def fit_knn(self, z: torch.Tensor, y: torch.Tensor) -> None:
        """Store the training embeddings and labels used by the KNN branch."""
        self.bank_z, self.bank_y = z.detach().float(), y.detach().long()

    def knn(self, z: torch.Tensor) -> torch.Tensor:
        if len(self.bank_y) == 0:
            return torch.full((z.shape[0], self.cfg.num_classes), 1.0 / self.cfg.num_classes, device=z.device)
        d2 = torch.cdist(z.float(), self.bank_z).pow(2)
        k = min(self.cfg.k, d2.shape[1])
        dist, idx = d2.topk(k, dim=1, largest=False)
        w = 1.0 / (dist + self.cfg.knn_eps)
        onehot = F.one_hot(self.bank_y[idx], self.cfg.num_classes).float()      # (B, k, N)
        return (w[..., None] * onehot).sum(1) / w.sum(1, keepdim=True)

    def forward(self, batch: dict) -> dict:
        z = self.encode(batch["features"])
        logits, p_gnn, entropy = self.gnn(z)
        out = {"z": z, "gnn_logits": logits, "p_gnn": p_gnn, "entropy": entropy}
        if not self.training:
            a = self.cfg.alpha
            out["p_ens"] = a * p_gnn + (1 - a) * self.knn(z) if a < 1 else p_gnn
            out["logits"] = out["p_ens"].clamp_min(1e-12).log()
        return out

    def loss(self, out: dict, batch: dict):
        y = batch["label"]
        seen = y < self.cfg.num_classes
        ce = F.cross_entropy(out["gnn_logits"][seen], y[seen])
        return ce, {"ce": ce.item()}
