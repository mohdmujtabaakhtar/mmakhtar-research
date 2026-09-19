"""PyTorch implementation of RHYME from "Curved Worlds, Clear Boundaries: Generalizing Speech
Deepfake Detection using Hyperbolic and Spherical Geometry Spaces" (IJCNLP-AACL 2025).

RHYME fuses hyperbolic and spherical views of one speech representation:

    s(tau) = frozen PTM frame embeddings                  (T, D)
    H = Phi_1D(s)  (1-D conv encoder),  u = GAP_t(H)      (d,)
    alpha = sigmoid(w_g^T u + b_g);  u_h = alpha u,  u_s = (1 - alpha) u
    hyperbolic:  x_h = exp^c_0(u_h)
    spherical:   x_s = u_s / ||u_s||,   y_s = x_s / (1 + sqrt(1 - ||x_s||^2))   (stereographic -> ball)
    barycentre:  z* = exp^c_0( alpha log^c_0(x_h) + (1 - alpha) log^c_0(y_s) )
    r = log^c_0(z*)  -> classifier -> softmax (bona fide / spoof)

Everything, including the curvature c, is trained end to end with cross-entropy.
Ablations (Table 3): ``gating="fixed"`` (alpha = 0.5), ``branches`` = both /
hyperbolic / spherical, ``fusion="euclidean"`` (no manifolds).
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F

from common.geometry import clip_tangent, expmap0, logmap0, stereographic_to_ball, to_sphere


@dataclass
class RhymeConfig:
    input_dim: int = 768
    conv_channels: tuple = (256, 128)   # Phi_1D; the last width is d
    kernel: int = 3
    hidden: int = 128                   # dense layer of the classifier
    num_classes: int = 2
    curvature: float = 1.0              # initial value; c is learned
    sphere_radius: float = 0.9          # sphere point scaled inside the unit ball before the stereographic map
    gating: str = "learned"             # learned | fixed (alpha = 0.5)
    branches: str = "both"              # both | hyperbolic | spherical
    fusion: str = "riemannian"          # riemannian | euclidean
    dropout: float = 0.3


class Rhyme(nn.Module):
    def __init__(self, cfg: RhymeConfig):
        super().__init__()
        self.cfg = cfg
        layers, c_in = [], cfg.input_dim
        for c in cfg.conv_channels:
            layers += [nn.Conv1d(c_in, c, cfg.kernel, padding=cfg.kernel // 2), nn.BatchNorm1d(c), nn.ReLU()]
            c_in = c
        self.phi = nn.Sequential(*layers)
        d = cfg.conv_channels[-1]
        self.gate = nn.Linear(d, 1)                                   # w_g, b_g
        self.raw_c = nn.Parameter(torch.tensor(float(cfg.curvature)).expm1().log())   # softplus^-1(c)
        self.classifier = nn.Sequential(nn.Linear(d, cfg.hidden), nn.ReLU(), nn.Dropout(cfg.dropout),
                                        nn.Linear(cfg.hidden, cfg.num_classes))

    @property
    def c(self) -> torch.Tensor:
        return F.softplus(self.raw_c) + 1e-4

    def forward(self, batch: dict) -> dict:
        x = batch["features"]
        if x.dim() == 2:
            x = x[:, None]
        u = self.phi(x.transpose(1, 2)).mean(-1)                         # GAP over time -> (B, d)
        alpha = (torch.sigmoid(self.gate(u)) if self.cfg.gating == "learned"
                 else torch.full((u.shape[0], 1), 0.5, device=u.device))
        u_h, u_s = alpha * u, (1 - alpha) * u
        c = self.c
        if self.cfg.fusion == "euclidean":
            r = alpha * u_h + (1 - alpha) * u_s
        else:
            x_h = expmap0(clip_tangent(u_h), c)
            y_s = stereographic_to_ball(to_sphere(u_s, self.cfg.sphere_radius / c.sqrt()), c)
            if self.cfg.branches == "hyperbolic":
                z = expmap0(clip_tangent(u), c)
            elif self.cfg.branches == "spherical":
                z = stereographic_to_ball(to_sphere(u, self.cfg.sphere_radius / c.sqrt()), c)
            else:
                z = expmap0(alpha * logmap0(x_h, c) + (1 - alpha) * logmap0(y_s, c), c)   # barycentre
            r = logmap0(z, c)
        return {"logits": self.classifier(r), "alpha": alpha.squeeze(-1)}

    def loss(self, out: dict, batch: dict):
        ce = F.cross_entropy(out["logits"], batch["label"])
        return ce, {"ce": ce.item()}
