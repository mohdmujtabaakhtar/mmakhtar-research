"""PHOENIX-Mamba: Prototypical Hyperbolic Organization for Evidence Normalization
and Inference using eXponential-map.

PyTorch implementation of PHOENIX-Mamba from Akhtar, Girish and Singh, "HCFD:
A Benchmark for Audio Deepfake Detection in Healthcare" (Findings of ACL 2026).

    X (frozen PTM frames, T x D)
      -> token-wise adapter phi            U = phi(X)             (T x d)
      -> Mamba backbone f_theta            Z = f_theta(U)          (T x d)
      -> multi-evidence attention pooling  E = [e_1..e_M]          (M x d)
      -> exponential map to Poincare ball  h_m = Exp_0(W e_m)      (M x h)
      -> prototypes: one real p_-, K fake modes p_{+,k}
         s_-(h) = -d(h, p_-),  s_+(h) = soft-min_k d(h, p_{+,k})
         logits = [mean_m s_-, mean_m s_+]

    L = L_cls + lambda * L_cluster + beta * L_sep
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F

from common.geometry import clip_tangent, make_manifold
from common.layers import ConvAdapter
from common.mamba import MambaBlock


@dataclass
class PhoenixConfig:
    input_dim: int = 768
    adapter_dim: int = 256          # d (paper)
    hyp_dim: int = 128              # h (paper)
    curvature: float = 1.0          # kappa = -1 (paper)
    num_evidence: int = 4           # M glimpses (paper)
    num_modes: int = 4              # K positive (fake) prototypes (paper)
    tau: float = 0.1                # assignment / soft-min temperature (paper)
    lam: float = 1.0                # cluster loss weight (paper)
    beta: float = 0.1               # separation loss weight (paper)
    gamma: float = 0.05             # entropy weight inside L_cluster (paper)
    num_layers: int = 2             # Mamba blocks
    d_state: int = 16               # SSM state size
    dropout: float = 0.1
    clip_norm: float = 1.0          # tangent feature clipping before the exponential map
    backbone: str = "mamba"         # mamba | bigru | cnn  (Table 5 ablation)
    geometry: str = "hyperbolic"    # hyperbolic | euclidean  (PHOENIX-Euc ablation)


class EvidencePooling(nn.Module):
    """M learnable queries attend over time; each yields one evidence vector."""

    def __init__(self, dim: int, num_queries: int):
        super().__init__()
        self.queries = nn.Parameter(torch.randn(num_queries, dim) * dim ** -0.5)
        self.key = nn.Linear(dim, dim)
        self.out = nn.Sequential(nn.Linear(dim, dim), nn.LayerNorm(dim))

    def forward(self, z: torch.Tensor):                       # (B, T, d)
        scores = torch.einsum("md,btd->bmt", self.queries, self.key(z)) / z.shape[-1] ** 0.5
        weights = scores.softmax(-1)                          # a_{m,t}, sums to 1 over t
        return self.out(torch.einsum("bmt,btd->bmd", weights, z)), weights


class PhoenixMamba(nn.Module):
    def __init__(self, cfg: PhoenixConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.adapter_dim
        self.adapter = nn.Sequential(nn.Linear(cfg.input_dim, d), nn.LayerNorm(d))
        if cfg.backbone == "mamba":
            self.backbone = nn.Sequential(*[MambaBlock(d, cfg.d_state, cfg.dropout)
                                            for _ in range(cfg.num_layers)])
        elif cfg.backbone == "bigru":
            self.gru = nn.GRU(d, d // 2, batch_first=True, bidirectional=True)
            self.backbone = lambda u: self.gru(u)[0]
        elif cfg.backbone == "cnn":
            self.backbone = ConvAdapter(d, d, blocks=2, pool=False)
        else:
            raise ValueError(cfg.backbone)
        self.pool = EvidencePooling(d, cfg.num_evidence)
        self.to_manifold = nn.Linear(d, cfg.hyp_dim)
        self.manifold = make_manifold(cfg.geometry, cfg.curvature)
        # Prototypes are parameterised in the tangent space at the origin.
        self.proto_real = nn.Parameter(torch.randn(1, cfg.hyp_dim) * 0.1)
        self.proto_fake = nn.Parameter(torch.randn(cfg.num_modes, cfg.hyp_dim) * 0.1)

    def prototypes(self):
        return self.manifold.from_tangent(self.proto_real), self.manifold.from_tangent(self.proto_fake)

    def forward(self, batch: dict) -> dict:
        cfg, man = self.cfg, self.manifold
        z = self.backbone(self.adapter(batch["features"]))
        evidence, attn = self.pool(z)                                     # (B, M, d)
        # tangent vectors are clipped before the exponential map; otherwise the
        # points saturate at the ball boundary and training stalls.
        h = man.from_tangent(clip_tangent(self.to_manifold(evidence), cfg.clip_norm))   # (B, M, h)
        p_neg, p_pos = self.prototypes()
        d_neg = man.dist(h, p_neg[None])                                  # (B, M)
        d_pos = man.dist(h[:, :, None], p_pos[None, None])                # (B, M, K)
        s_neg = -d_neg
        # soft-min written as tau * logsumexp(-d / tau) so that s_+ is on the
        # same distance scale as s_-.
        s_pos = cfg.tau * torch.logsumexp(-d_pos / cfg.tau, dim=-1)
        logits = torch.stack([s_neg.mean(1), s_pos.mean(1)], dim=-1)      # [S_-, S_+]
        q = (-d_pos / cfg.tau).softmax(-1)                                # q_{m,k}
        return {"logits": logits, "d_pos": d_pos, "q": q, "attention": attn, "evidence": h}

    def loss(self, out: dict, batch: dict):
        cfg, man = self.cfg, self.manifold
        y = batch["label"]
        l_cls = F.cross_entropy(out["logits"], y)
        # L_cluster pulls evidence toward the *fake* modes, so it is applied
        # to fake samples only; pulling real speech toward fake prototypes would
        # contradict the classifier.
        fake = y == 1
        if fake.any():
            q, d = out["q"][fake], out["d_pos"][fake]
            l_cluster = ((q * d).sum(-1) + cfg.gamma * (q * q.clamp_min(1e-8).log()).sum(-1)).mean()
        else:
            l_cluster = out["logits"].sum() * 0
        p_neg, p_pos = self.prototypes()
        k = p_pos.shape[0]
        i, j = torch.triu_indices(k, k, offset=1)
        l_sep = (torch.exp(-man.dist(p_pos[i], p_pos[j])).sum()
                 + torch.exp(-man.dist(p_pos, p_neg.expand_as(p_pos))).sum())
        total = l_cls + cfg.lam * l_cluster + cfg.beta * l_sep
        return total, {"cls": l_cls.item(), "cluster": l_cluster.item(), "sep": l_sep.item(),
                       "monitor": l_cls.item()}

    @torch.no_grad()
    def fake_score(self, batch: dict) -> torch.Tensor:
        return self(batch)["logits"].softmax(-1)[:, 1]
