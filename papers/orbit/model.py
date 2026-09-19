"""ORBIT: Optimized Representation Learning via BI-geometric and adversarial
Training for zero-shot cross-lingual Alzheimer detection (reference implementation).

Re-implemented in PyTorch from Section 3.1 of Girish, Akhtar, Sheth, Singh,
Gerard, McClean and Wong-Lin (INTERSPEECH 2026). Not the original experimental
code. Details the paper leaves open are marked ``# [impl]``.

    audio frames S, transcript tokens H  (frozen multilingual PTMs)
      -> light 1-D conv refinement + attention pooling          a, t
      -> bidirectional cross-attention + MLP fusion              f
      -> sphere head      x^S = r W^S f / ||W^S f||
      -> hyperbolic head  x^H = Exp_0(W^H f)
      -> K cluster centres per manifold, soft assignments q_S, q_H,
         product-of-experts consensus q_C
      -> class prototypes per manifold, p_S(y), p_H(y), PoE vote p_vote(y)
      -> language discriminators behind gradient reversal on f, x^S, x^H, [q_S || q_H]

    L = L_cls + lambda_BGCC (L_dec + L_js + L_margin) + sum_Z lambda_Z CE(D_Z(GRL(Z)), language)
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import nn
import torch.nn.functional as F

from common.geometry import clip_tangent, make_manifold
from common.layers import AttentionPool, ConvAdapter
from common.ot import grad_reverse


@dataclass
class OrbitConfig:
    audio_dim: int = 768
    text_dim: int = 768
    num_languages: int = 4
    hidden: int = 256              # [impl] fusion width
    geo_dim: int = 64              # [impl] sphere / ball dimension
    num_clusters: int = 4          # K > 2 centres per manifold  [impl] value
    tau: float = 0.1               # [impl] cluster temperature
    tau_c: float = 0.1             # [impl] prototype temperature
    sphere_radius: float = 1.0     # r  [impl]
    curvature: float = 1.0         # c  [impl]
    margin: float = 0.5            # [impl] prototype margin
    lambda_bgcc: float = 0.1       # [impl]
    lambda_adv: float = 0.1        # [impl] same lambda_Z for every tap
    grl_coeff: float = 1.0         # [impl]
    dropout: float = 0.3
    clip_norm: float = 1.0         # [impl] tangent clipping before the exponential map
    cross_attention: bool = True   # Table 2: ORBIT with / without cross-attention
    use_grl: bool = True           # Table 3: "w/o GRL"
    geometries: list = field(default_factory=lambda: ["sphere", "hyperbolic"])  # Table 3 variants


class GeometryHead(nn.Module):
    """Projects f onto one manifold; K cluster centres and 2 class prototypes live there."""

    def __init__(self, name: str, cfg: OrbitConfig):
        super().__init__()
        self.name, self.cfg = name, cfg
        self.manifold = make_manifold(name, cfg.curvature, cfg.sphere_radius)
        self.proj = nn.Linear(cfg.hidden, cfg.geo_dim)
        self.centres = nn.Parameter(torch.randn(cfg.num_clusters, cfg.geo_dim) * 0.3)
        self.prototypes = nn.Parameter(torch.randn(2, cfg.geo_dim) * 0.3)

    def embed(self, v: torch.Tensor) -> torch.Tensor:
        if self.name == "hyperbolic":
            v = clip_tangent(v, self.cfg.clip_norm)
        return self.manifold.from_tangent(v)

    def forward(self, f: torch.Tensor) -> dict:
        x = self.embed(self.proj(f))
        d_cluster = self.manifold.dist(x[:, None], self.embed(self.centres)[None])   # (B, K)
        d_proto = self.manifold.dist(x[:, None], self.embed(self.prototypes)[None])  # (B, 2)
        return {"x": x, "q": (-d_cluster / self.cfg.tau).softmax(-1),
                "p": (-d_proto / self.cfg.tau_c).softmax(-1), "d_proto": d_proto}


def _normalise(x: torch.Tensor) -> torch.Tensor:
    return x / x.sum(-1, keepdim=True).clamp_min(1e-8)


class Orbit(nn.Module):
    def __init__(self, cfg: OrbitConfig):
        super().__init__()
        self.cfg = cfg
        h = cfg.hidden
        self.enc = nn.ModuleDict({
            "audio": nn.Sequential(nn.Linear(cfg.audio_dim, h), ConvAdapter(h, h, pool=False)),
            "text": nn.Sequential(nn.Linear(cfg.text_dim, h), ConvAdapter(h, h, pool=False)),
        })
        self.pool = nn.ModuleDict({m: AttentionPool(h) for m in ("audio", "text")})
        # [impl] bidirectional cross-attention: the pooled vector of one modality
        # queries the frame/token sequence of the other.
        self.cross = nn.ModuleDict({m: nn.MultiheadAttention(h, 4, batch_first=True)
                                    for m in ("audio", "text")})
        fuse_in = 4 * h if cfg.cross_attention else 2 * h
        self.fuse = nn.Sequential(nn.Linear(fuse_in, h), nn.ReLU(), nn.Dropout(cfg.dropout), nn.Linear(h, h))
        self.heads = nn.ModuleDict({g: GeometryHead(g, cfg) for g in cfg.geometries})
        n_lang = cfg.num_languages

        def disc(d):
            return nn.Sequential(nn.Linear(d, h), nn.ReLU(), nn.Linear(h, n_lang))

        self.discriminators = nn.ModuleDict({"f": disc(h), "Q": disc(cfg.num_clusters * len(cfg.geometries))})
        for g in cfg.geometries:
            self.discriminators[g] = disc(cfg.geo_dim)

    def forward(self, batch: dict) -> dict:
        cfg = self.cfg
        seq = {m: self.enc[m](batch[m]) for m in ("audio", "text")}
        vec = {m: self.pool[m](seq[m])[0] for m in ("audio", "text")}
        parts = [vec["audio"], vec["text"]]
        if cfg.cross_attention:
            a_ctx = self.cross["audio"](vec["audio"][:, None], seq["text"], seq["text"])[0][:, 0]
            t_ctx = self.cross["text"](vec["text"][:, None], seq["audio"], seq["audio"])[0][:, 0]
            parts += [a_ctx, t_ctx]
        f = self.fuse(torch.cat(parts, dim=-1))
        geo = {g: head(f) for g, head in self.heads.items()}

        # Product-of-experts consensus over clusters and vote over classes.
        q_c, p_vote = None, None
        for g in geo.values():
            q_c = g["q"] if q_c is None else q_c * g["q"]
            p_vote = g["p"] if p_vote is None else p_vote * g["p"]
        q_c, p_vote = _normalise(q_c), _normalise(p_vote)

        # Language discriminators behind gradient reversal (multi-tap adversaries).
        coeff = cfg.grl_coeff if cfg.use_grl else 0.0
        taps = {"f": f, "Q": torch.cat([g["q"] for g in geo.values()], -1)}
        taps.update({name: g["x"] for name, g in geo.items()})
        lang_logits = {k: self.discriminators[k](grad_reverse(v, coeff)) for k, v in taps.items()}
        return {"p_vote": p_vote, "q_c": q_c, "geo": geo, "lang_logits": lang_logits, "f": f}

    def loss(self, out: dict, batch: dict):
        cfg, y = self.cfg, batch["label"]
        l_cls = F.nll_loss(out["p_vote"].clamp_min(1e-8).log(), y)

        # Cluster-consensus regularisers.
        q_c = out["q_c"]
        freq = q_c.sum(0, keepdim=True)
        target = _normalise(q_c ** 2 / freq).detach()                         # DEC-style sharpening
        l_dec = F.kl_div(q_c.clamp_min(1e-8).log(), target, reduction="batchmean")
        qs = [g["q"] for g in out["geo"].values()]
        if len(qs) == 2:
            m = 0.5 * (qs[0] + qs[1])
            l_js = 0.5 * (F.kl_div(m.clamp_min(1e-8).log(), qs[0], reduction="batchmean")
                          + F.kl_div(m.clamp_min(1e-8).log(), qs[1], reduction="batchmean"))
        else:
            l_js = q_c.sum() * 0
        # [impl] prototype margin: the true-class prototype must be closer than the
        # other class's by at least `margin`, in every geometry.
        l_margin = 0.0
        for g in out["geo"].values():
            d = g["d_proto"]
            d_true, d_other = d.gather(1, y[:, None]), d.gather(1, (1 - y)[:, None])
            l_margin = l_margin + F.relu(d_true - d_other + cfg.margin).mean()
        l_bgcc = l_dec + l_js + l_margin

        l_adv = sum(F.cross_entropy(v, batch["language"]) for v in out["lang_logits"].values())
        adv_weight = cfg.lambda_adv if cfg.use_grl else 0.0
        total = l_cls + cfg.lambda_bgcc * l_bgcc + adv_weight * l_adv
        return total, {"cls": l_cls.item(), "dec": l_dec.item(), "js": float(l_js.detach()),
                       "margin": float(l_margin.detach()), "adv": l_adv.item(), "monitor": l_cls.item()}
