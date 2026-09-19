"""PyTorch implementation of MiCuNet from "Towards Attribution of Generators and Emotional
Manipulation in Cross-Lingual Synthetic Speech using Geometric Learning"
(Findings of IJCNLP-AACL 2025).

MiCuNet (Mixed-Curvature Network) traces emotionally manipulated speech with three
tasks at once: the original emotion (OE), the current / manipulated emotion (CE)
and the manipulation source (M, the emotional voice conversion model).

1. Encoders: a speech pre-trained model (PTM) embedding goes through a 1-D CNN; a
   hand-crafted spectrogram (e.g. STFT + mel) goes through a 3-D CNN
   (3 blocks, 32 -> 64 -> 128 channels, 3x3x3 conv, batch norm, ReLU, 2x2x2 max-pool).
2. Self-attention inside each modality, bidirectional cross-modal attention, global
   average pooling and concatenation give the joint embedding z0.
3. Mixed-curvature projection: shared non-linear maps send z0 to the Poincare ball
   (hyperbolic), Euclidean space and the unit sphere; log maps (at the origin / the
   north pole) bring each back to a tangent space.
4. A learnable gate (feed-forward network + softmax) weights the three spaces;
   z_fused = sum_k w_k log_k(z_k).
5. Three heads (hidden layer + ReLU + softmax) predict OE, CE and M; the loss is
   the sum of the three cross-entropies.

Ablations (Table 4): ``spaces`` (drop the hyperbolic and / or spherical branch) and
``gating="uniform"`` (no learnable gating); ``fusion="concat"`` is the concatenation baseline.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import nn
import torch.nn.functional as F

from common.geometry import (clip_tangent, expmap0, logmap0, sphere_expmap_north,
                             sphere_logmap_north)
from common.layers import EmbeddingCNN

TASKS = ("oe", "ce", "source")


class Spectro3DCNN(nn.Module):
    """(B, T, F) spectrogram -> (B, tokens, 128).

    The time axis is split into ``depth`` consecutive segments that form the depth
    dimension of a (1, depth, T / depth, F) volume.
    """

    def __init__(self, channels=(32, 64, 128), depth: int = 8):
        super().__init__()
        layers, c_in = [], 1
        for c in channels:
            layers += [nn.Conv3d(c_in, c, 3, padding=1), nn.BatchNorm3d(c), nn.ReLU(),
                       nn.MaxPool3d(2, ceil_mode=True)]
            c_in = c
        self.net = nn.Sequential(*layers)
        self.depth, self.out_dim = depth, c_in

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, f = x.shape
        seg = t // self.depth
        vol = x[:, : seg * self.depth].reshape(b, 1, self.depth, seg, f)
        return self.net(vol).flatten(2).transpose(1, 2)


class AttnBlock(nn.Module):
    """Residual multi-head attention: query tokens attend to context tokens."""

    def __init__(self, dim: int, heads: int, dropout: float):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, q: torch.Tensor, ctx: torch.Tensor | None = None) -> torch.Tensor:
        ctx = q if ctx is None else ctx
        return self.norm(q + self.attn(q, ctx, ctx)[0])


@dataclass
class MiCuNetConfig:
    ptm_dim: int = 1280
    num_classes: dict = field(default_factory=lambda: {"oe": 5, "ce": 5, "source": 8})
    ptm_filters: tuple = (64, 128)
    spec_channels: tuple = (32, 64, 128)
    spec_depth: int = 8
    heads: int = 4
    manifold_dim: int = 128
    curvature: float = 1.0            # initial c, learned
    spaces: tuple = ("hyperbolic", "euclidean", "spherical")
    gating: str = "learned"           # learned | uniform
    head_hidden: int = 128
    dropout: float = 0.3
    fusion: str = "micunet"           # micunet | concat (pooled encoders concatenated, no attention / manifolds)


class MiCuNet(nn.Module):
    def __init__(self, cfg: MiCuNetConfig):
        super().__init__()
        self.cfg = cfg
        self.ptm_cnn = EmbeddingCNN(cfg.ptm_dim, tuple(cfg.ptm_filters))
        self.spec_cnn = Spectro3DCNN(tuple(cfg.spec_channels), cfg.spec_depth)
        d = cfg.ptm_filters[-1]
        if self.spec_cnn.out_dim != d:
            raise ValueError("the last PTM filter count and 3-D CNN channel count must match")
        self.self_attn = nn.ModuleDict({m: AttnBlock(d, cfg.heads, cfg.dropout) for m in ("ptm", "spec")})
        self.cross_attn = nn.ModuleDict({m: AttnBlock(d, cfg.heads, cfg.dropout) for m in ("ptm", "spec")})
        # The PTM "sequence" runs along the embedding's feature axis, so plain averaging
        # would discard which features fired; a linear read-out keeps that information.
        self.ptm_pool = nn.Linear(self.ptm_cnn.flat_dim, d)
        md = cfg.manifold_dim
        self.shared = nn.Sequential(nn.Linear(2 * d, md), nn.ReLU())         # shared non-linear transform
        self.phi = nn.ModuleDict({s: nn.Linear(md, md) for s in cfg.spaces})   # Phi^H, Phi^E, Phi^S
        self.gate = nn.Sequential(nn.Linear(2 * d, md), nn.ReLU(), nn.Linear(md, len(cfg.spaces)))
        self.raw_c = nn.Parameter(torch.tensor(float(cfg.curvature)).expm1().log())
        head_in = md if cfg.fusion == "micunet" else 2 * d
        self.heads = nn.ModuleDict({
            t: nn.Sequential(nn.Linear(head_in, cfg.head_hidden), nn.ReLU(), nn.Dropout(cfg.dropout),
                             nn.Linear(cfg.head_hidden, n)) for t, n in cfg.num_classes.items()})

    @property
    def c(self) -> torch.Tensor:
        return F.softplus(self.raw_c) + 1e-4

    def to_tangent(self, space: str, v: torch.Tensor) -> torch.Tensor:
        """Project onto the manifold, then log-map back to its tangent space."""
        if space == "hyperbolic":
            return logmap0(expmap0(clip_tangent(v), self.c), self.c)
        if space == "spherical":
            return sphere_logmap_north(sphere_expmap_north(clip_tangent(v, 3.0)))
        return v

    def forward(self, batch: dict) -> dict:
        a = self.ptm_cnn(batch["ptm"]).transpose(1, 2)                   # (B, L, 128)
        s = self.spec_cnn(batch["spec"])                                  # (B, V, 128)
        if self.cfg.fusion == "concat":
            z0 = torch.cat([self.ptm_pool(a.flatten(1)), s.mean(1)], dim=-1)
            return {f"logits_{t}": head(z0) for t, head in self.heads.items()}
        a, s = self.self_attn["ptm"](a), self.self_attn["spec"](s)
        a2, s2 = self.cross_attn["ptm"](a, s), self.cross_attn["spec"](s, a)   # bidirectional
        z0 = torch.cat([self.ptm_pool(a2.flatten(1)), s2.mean(1)], dim=-1)
        h = self.shared(z0)
        tangents = torch.stack([self.to_tangent(sp, self.phi[sp](h)) for sp in self.cfg.spaces], 1)
        if self.cfg.gating == "learned":
            w = self.gate(z0).softmax(-1)
        else:
            w = torch.full((z0.shape[0], len(self.cfg.spaces)), 1.0 / len(self.cfg.spaces), device=z0.device)
        fused = (w[..., None] * tangents).sum(1)
        out = {f"logits_{t}": head(fused) for t, head in self.heads.items()}
        out["gate"] = w
        return out

    def loss(self, out: dict, batch: dict):
        parts = {t: F.cross_entropy(out[f"logits_{t}"], batch[t]) for t in self.heads if t in batch}
        total = sum(parts.values())
        return total, {k: v.item() for k, v in parts.items()}
