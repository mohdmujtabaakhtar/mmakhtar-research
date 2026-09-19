"""COBALT: COdebook-Aligned BAndit-weighted hyperboLic proTotypE fusion.

PyTorch implementation of COBALT from Akhtar, Girish, Wadhwa, Singh and Ma,
"From Signals to Patterns: Non-Invasive Tuberculosis Detection from Cough Audio
using Bandit Weighted Hyperbolic Prototypes" (INTERSPEECH 2026).

For each of two streams m (e.g. MFCC frames and PaSST tokens):

    X^(m) -> 1-D CNN adapter g_m           H^(m)   (T' x d)
          -> tokenise to K tokens          Z^(m)   (K x d)
          -> Exp_0(W_m Z^(m))              Y^(m)   (K x d_h) on the Poincare ball
          -> soft assignment to a SHARED hyperbolic codebook of M prototypes
          -> prototype evidence            p^(m) = mean_k A_k   (M,)
Fusion:
    w = softmax(Q / tau_w)  (bandit reliability per prototype)
    f = [p~1, p~2, p~1 * p~2],  p~m = p^(m) * w  ->  MLP  ->  TB+ / TB-

    L = L_task + beta_vq (L_vq^1 + L_vq^2) + lambda H(w)

The bandit scores Q are updated after every step from the reward
r = alpha (L_base - L_COBALT) + beta (M_COBALT - M_base), where "base" is the same
forward pass with uniform prototype weights and M is the confidence margin.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import nn
import torch.nn.functional as F

from common.geometry import clip_tangent, make_manifold
from common.layers import ConvAdapter

STREAMS = ("stream1", "stream2")


@dataclass
class CobaltConfig:
    input_dims: dict = field(default_factory=lambda: {"stream1": 40, "stream2": 768})
    adapter_dim: int = 128        # d, adapter width
    num_tokens: int = 8           # K tokens per stream
    hyp_dim: int = 64             # d_h
    num_prototypes: int = 32      # M shared codebook size
    curvature: float = 1.0        # c
    tau_q: float = 0.1            # assignment temperature
    tau_w: float = 1.0            # reliability temperature
    bandit_lr: float = 0.1        # eta
    reward_alpha: float = 1.0     # alpha
    reward_beta: float = 1.0      # beta
    beta_vq: float = 0.25         # VQ loss weight
    commitment: float = 0.25      # commitment weight inside L_vq
    lam_entropy: float = 0.01     # lambda for H(w)
    hidden: int = 128             # MLP head width (paper: e.g. 128 units)
    dropout: float = 0.3
    clip_norm: float = 1.0        # tangent clipping before the exponential map
    geometry: str = "hyperbolic"  # hyperbolic | euclidean   (COBALT-E, Table 2)
    fusion: str = "cobalt"        # cobalt | mobius | concat  (Table 3 / Table 2 baselines)


class Cobalt(nn.Module):
    def __init__(self, cfg: CobaltConfig):
        super().__init__()
        self.cfg = cfg
        d, dh, m = cfg.adapter_dim, cfg.hyp_dim, cfg.num_prototypes
        self.adapters = nn.ModuleDict({s: ConvAdapter(cfg.input_dims[s], d) for s in STREAMS})
        self.to_manifold = nn.ModuleDict({s: nn.Linear(d, dh) for s in STREAMS})
        self.manifold = make_manifold(cfg.geometry, cfg.curvature)
        self.codebook = nn.Parameter(torch.randn(m, dh) * 0.3)    # tangent-space parameters
        # Bandit scores Q live in a buffer (no gradient). A learnable offset s
        # lets the H(w) term act through gradients, since w = softmax((Q + s) / tau_w).
        self.register_buffer("Q", torch.zeros(m))
        self.offset = nn.Parameter(torch.zeros(m))
        in_dim = {"cobalt": 3 * m, "mobius": dh, "concat": 2 * dh}[cfg.fusion]
        self.head = nn.Sequential(nn.LayerNorm(in_dim), nn.Linear(in_dim, cfg.hidden), nn.ReLU(),
                                  nn.Dropout(cfg.dropout), nn.Linear(cfg.hidden, 2))

    # ------------------------------------------------------------------ pieces
    def tokens(self, s: str, x: torch.Tensor) -> torch.Tensor:
        """Adapter + adaptive average pooling to K tokens, mapped to the manifold."""
        h = self.adapters[s](x)                                                  # (B, T', d)
        z = F.adaptive_avg_pool1d(h.transpose(1, 2), self.cfg.num_tokens).transpose(1, 2)
        return self.manifold.from_tangent(clip_tangent(self.to_manifold[s](z), self.cfg.clip_norm))

    def reliability(self, uniform: bool = False) -> torch.Tensor:
        m = self.cfg.num_prototypes
        if uniform:
            return torch.full((m,), 1.0 / m, device=self.Q.device)
        return ((self.Q + self.offset) / self.cfg.tau_w).softmax(-1)

    def _head_input(self, p: dict, w: torch.Tensor) -> torch.Tensor:
        # evidence and weights are each rescaled by M so that uniform
        # reliability leaves the evidence unchanged; otherwise the fused vector
        # has entries of order 1/M^2 and the head's decision threshold drifts.
        m = self.cfg.num_prototypes
        p1, p2 = m * p["stream1"] * m * w, m * p["stream2"] * m * w
        return torch.cat([p1, p2, p1 * p2], dim=-1)

    # ------------------------------------------------------------------ forward
    def forward(self, batch: dict) -> dict:
        cfg, man = self.cfg, self.manifold
        y_tok = {s: self.tokens(s, batch[s]) for s in STREAMS}                   # (B, K, d_h)
        if cfg.fusion != "cobalt":  # Mobius-addition / concatenation baselines
            pooled = {s: man.from_tangent(man.to_tangent(y_tok[s]).mean(1)) for s in STREAMS}
            if cfg.fusion == "mobius":
                f = man.to_tangent(man.add(pooled["stream1"], pooled["stream2"]))
            else:
                f = torch.cat([man.to_tangent(pooled[s]) for s in STREAMS], dim=-1)
            return {"logits": self.head(f)}

        codes = man.from_tangent(clip_tangent(self.codebook, cfg.clip_norm))    # (M, d_h)
        assign, dists, p = {}, {}, {}
        for s in STREAMS:
            dists[s] = man.dist(y_tok[s][:, :, None], codes[None, None])         # (B, K, M)
            assign[s] = (-dists[s] / cfg.tau_q).softmax(-1)                      # A_j(y)
            p[s] = assign[s].mean(1)                                             # p^(m)  (B, M)
        w = self.reliability()
        logits = self.head(self._head_input(p, w))
        with torch.no_grad():  # baseline pass with uniform weights, for the bandit reward
            base_logits = self.head(self._head_input(p, self.reliability(uniform=True)))
        return {"logits": logits, "base_logits": base_logits, "assign": assign,
                "dists": dists, "tokens": y_tok, "codes": codes, "w": w}

    def loss(self, out: dict, batch: dict):
        cfg = self.cfg
        y = batch["label"]
        l_task = F.cross_entropy(out["logits"], y)
        if cfg.fusion != "cobalt":
            return l_task, {"task": l_task.item(), "monitor": l_task.item()}
        man = self.manifold
        l_vq = 0.0
        for s in STREAMS:  # nearest codeword per token, codebook + commitment terms in tangent space
            nearest = out["codes"][out["dists"][s].argmin(-1)]                   # (B, K, d_h)
            tok_t, code_t = man.to_tangent(out["tokens"][s]), man.to_tangent(nearest)
            l_vq = l_vq + F.mse_loss(code_t, tok_t.detach()) + cfg.commitment * F.mse_loss(tok_t, code_t.detach())
        w = out["w"]
        entropy = -(w * w.clamp_min(1e-12).log()).sum()
        total = l_task + cfg.beta_vq * l_vq + cfg.lam_entropy * entropy
        return total, {"task": l_task.item(), "vq": float(l_vq.detach()), "entropy": entropy.item(),
                       "monitor": l_task.item()}

    # ------------------------------------------------------------------ bandit update
    @staticmethod
    def _margin(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        prob = logits.softmax(-1)
        true = prob.gather(1, y[:, None]).squeeze(1)
        return (true - prob.scatter(1, y[:, None], 0).max(-1).values).mean()

    @torch.no_grad()
    def after_step(self, batch: dict, out: dict) -> None:
        """Q_j <- (1 - eta u_j) Q_j + eta u_j r  after each optimiser step."""
        if self.cfg.fusion != "cobalt":
            return
        cfg, y = self.cfg, batch["label"]
        l_cobalt = F.cross_entropy(out["logits"], y)
        l_base = F.cross_entropy(out["base_logits"], y)
        reward = (cfg.reward_alpha * (l_base - l_cobalt)
                  + cfg.reward_beta * (self._margin(out["logits"], y) - self._margin(out["base_logits"], y)))
        # usage u_j: share of assignment mass on prototype j in this batch (both streams)
        usage = torch.stack([out["assign"][s].mean((0, 1)) for s in STREAMS]).mean(0)
        usage = usage / usage.max().clamp_min(1e-8)
        rate = cfg.bandit_lr * usage
        self.Q.mul_(1 - rate).add_(rate * reward)
