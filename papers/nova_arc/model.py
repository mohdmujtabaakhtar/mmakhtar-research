"""NOVA-ARC: NOn-verbal to Verbal Adaptation via hyperbolic Alignment, Radial
calibration and Codebook tokens.

PyTorch implementation of NOVA-ARC from Girish, Akhtar and Singh, "Prosody as
Supervision: Bridging the Non-Verbal-Verbal for Multilingual Speech Emotion
Recognition" (ACL 2026).

Shared forward pass for any utterance (labelled non-verbal source or unlabelled
verbal target):

    z_t = E(x)_t                                   frame features (frozen here)
    x_t = Exp_0(W_p z_t + b_p)                     hyperbolic frames
    q_t = nearest codeword in hyperbolic VQ codebook C (prosody tokens)
    b_t = Exp_0(W_b Log_0(x_t (+) q_t))            Mobius fusion + bottleneck
    b~_t = HEL(b_t)                                radius r -> r^alpha (Hyperbolic Emotion Lens)
    u = sum_t softmax(w . Log_0 b~_t) Log_0 b~_t   attention pooling in the tangent space
    p(y|x) = softmax(W_e u + b_e)

Adaptation: class prototypes mu_c = Frechet mean of source embeddings (refreshed
each epoch); an entropic OT plan between prototypes and a target batch gives
soft labels q_cj = n Pi_cj.

    L = L_S + lambda_OPT <Pi, M> + lambda_OT L_OT-CE + lambda_VQ L_VQ
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F

from common.geometry import clip_tangent, make_manifold
from common.ot import sinkhorn


@dataclass
class NovaArcConfig:
    input_dim: int = 768
    num_classes: int = 5            # happy, anger, disgust, sadness, fear (paper)
    latent_dim: int = 256           # d (paper)
    bottleneck_dim: int = 128       # d_b (paper)
    curvature: float = 1.0          # kappa = -1 (paper)
    codebook_size: int = 256        # K (paper)
    commitment: float = 0.25        # beta (paper)
    lambda_vq: float = 1.0          # (paper)
    alpha_init: float = 1.0         # HEL exponent, learned (paper)
    hel_eps: float = 1e-8           # HEL stabiliser (paper)
    ot_epsilon: float = 0.05        # (paper)
    sinkhorn_iters: int = 50        # (paper)
    lambda_opt: float = 1.0         # (paper)
    lambda_ot: float = 1.0          # (paper)
    clip_norm: float = 1.0          # tangent clipping before the exponential map
    geometry: str = "hyperbolic"    # hyperbolic | euclidean   (Table 3 EUC vs HYP)
    tokens: str = "both"            # both | continuous | discrete  (Table 4 "No VQ" / "Token only")
    fusion: str = "mobius"          # mobius | concat                (Table 4 "Concat/MLP")
    use_hel: bool = True            # Table 4 "No HEL"
    ot_geometry: str = "same"       # same | euclidean               (Table 4 "Euclidean OT")
    adaptation: str = "ot"          # ot | none (source-only training)


class NovaArc(nn.Module):
    def __init__(self, cfg: NovaArcConfig):
        super().__init__()
        self.cfg = cfg
        d, db = cfg.latent_dim, cfg.bottleneck_dim
        self.man = make_manifold(cfg.geometry, cfg.curvature)
        self.ot_man = self.man if cfg.ot_geometry == "same" else make_manifold("euclidean")
        self.proj = nn.Linear(cfg.input_dim, d)                              # W_p, b_p
        self.codebook = nn.Parameter(torch.randn(cfg.codebook_size, d) * 0.3)  # tangent-space codewords
        self.fuse_mlp = (nn.Sequential(nn.Linear(2 * d, d), nn.ReLU(), nn.Linear(d, d))
                         if cfg.fusion == "concat" else None)
        self.bottleneck = nn.Linear(d, db)                                   # W_b, b_b
        self.log_alpha = nn.Parameter(torch.tensor(float(cfg.alpha_init)).log())
        self.attn = nn.Linear(db, 1, bias=False)                             # shared vector w
        self.classifier = nn.Linear(db, cfg.num_classes)                     # W_e, b_e
        self.register_buffer("prototypes", torch.zeros(cfg.num_classes, db))
        self.register_buffer("class_prior", torch.full((cfg.num_classes,), 1.0 / cfg.num_classes))

    def _exp(self, v):
        return self.man.from_tangent(clip_tangent(v, self.cfg.clip_norm) if self.man.name == "hyperbolic" else v)

    def encode(self, z: torch.Tensor) -> dict:
        """Shared forward pass. z: (B, T, D) frame features."""
        cfg, man = self.cfg, self.man
        x = self._exp(self.proj(z))                                          # (B, T, d)
        codes = self._exp(self.codebook)                                     # (K, d)
        flat = x.reshape(-1, x.shape[-1])
        idx = man.pairwise(flat, codes).argmin(-1)
        q = codes[idx].view_as(x)                                            # prosody tokens
        x_t, q_t = man.to_tangent(x), man.to_tangent(q)
        vq = F.mse_loss(q_t, x_t.detach()) + cfg.commitment * F.mse_loss(x_t, q_t.detach())

        if cfg.tokens == "continuous":
            fused_t = x_t
        elif cfg.tokens == "discrete":
            fused_t = x_t + (q_t - x_t).detach()     # straight-through so the projection still learns
        elif cfg.fusion == "concat":
            fused_t = self.fuse_mlp(torch.cat([x_t, q_t], -1))
        else:
            fused_t = man.to_tangent(man.add(x, q))                          # Mobius addition
        b = self._exp(self.bottleneck(fused_t))                              # (B, T, d_b)

        v = man.to_tangent(b)
        if cfg.use_hel:  # radial power-law warp: r -> r^alpha
            r = v.norm(dim=-1, keepdim=True)
            v = (r + cfg.hel_eps).pow(self.log_alpha.exp()) * v / (r + cfg.hel_eps)
        weights = self.attn(v).squeeze(-1).softmax(-1)                      # (B, T)
        u = (weights.unsqueeze(-1) * v).sum(1)                               # u (tangent)
        return {"logits": self.classifier(u), "u": u, "embedding": self._exp(u), "vq": vq}

    def forward(self, batch: dict) -> dict:
        out = {"source": self.encode(batch["source"])}
        if "target" in batch:
            out["target"] = self.encode(batch["target"])
        return out

    def ot_embedding(self, enc: dict) -> torch.Tensor:
        """Where prototypes and OT costs live: the manifold, or the tangent space for Euclidean OT."""
        return enc["embedding"] if self.cfg.ot_geometry == "same" else enc["u"]

    @torch.no_grad()
    def refresh_prototypes(self, embeddings: torch.Tensor, labels: torch.Tensor) -> None:
        """mu_c = Frechet mean of source embeddings of class c; also stores the class prior."""
        for c in range(self.cfg.num_classes):
            if (labels == c).any():
                self.prototypes[c] = self.ot_man.mean(embeddings[labels == c])
        counts = torch.bincount(labels, minlength=self.cfg.num_classes).float()
        self.class_prior.copy_(counts / counts.sum())

    def loss(self, out: dict, batch: dict):
        cfg = self.cfg
        src = out["source"]
        l_s = F.cross_entropy(src["logits"], batch["label"])
        l_vq = src["vq"]
        parts = {"source_ce": l_s.item()}
        total = l_s
        if "target" in out and cfg.adaptation == "ot":
            tgt = out["target"]
            cost = self.ot_man.pairwise(self.prototypes, self.ot_embedding(tgt)).pow(2)   # M_cj (C, n)
            n = cost.shape[1]
            plan = sinkhorn(cost.detach(), self.class_prior, torch.full((n,), 1.0 / n, device=cost.device),
                            cfg.ot_epsilon, cfg.sinkhorn_iters)
            l_opt = (plan * cost).sum()
            soft = (n * plan).T                                              # q_cj, rows sum to 1
            l_otce = -(soft * tgt["logits"].log_softmax(-1)).sum(-1).mean()
            l_vq = l_vq + tgt["vq"]
            total = total + cfg.lambda_opt * l_opt + cfg.lambda_ot * l_otce
            parts.update(opt=l_opt.item(), ot_ce=l_otce.item())
        total = total + cfg.lambda_vq * l_vq
        parts.update(vq=l_vq.item(), alpha=self.log_alpha.exp().item())
        return total, parts
