"""PyTorch implementation of SATYAM from "Indic-CodecFake meets SATYAM: Towards Detecting Neural
Audio Codec Synthesized Speech Deepfakes in Indic Languages" (Findings of ACL 2026).

SATYAM is a hyperbolic audio-language model for codec-fake detection:

1. Semantic (Whisper, e_w) and paralinguistic (TRILLsson, e_t) embeddings each pass a
   lightweight CNN block (Conv1d, kernel 3, max-pool), a projection into a shared
   d-dimensional space (W_w, W_t) and a sigmoid gate.
2. Both are mapped into the Poincare ball: h_w = exp_0^c(e~_w), h_t = exp_0^c(e~_t).
3. Speech-speech alignment:  L_S-S = D_B(h_w, h_t)  (Bhattacharyya distance);
   fusion by Mobius addition: h_f = h_w (+) h_t.
4. A conditioning prompt ("Analyze the speech for unnatural artifacts") is encoded by
   the LM; hidden states of an intermediate layer are mean-pooled, projected and mapped
   into the ball (h_A). Speech-text alignment:  L_S-T = D_B(h_f, h_A);
   h_final = h_f (+) h_A.
5. u_final = log_0^c(h_final); g = W_g u_final is injected as prefix tokens into the
   frozen Qwen2 decoder, followed by the decision prompt. The answer is constrained to
   "Real" or "Fake".

    L = lambda_1 L_S-S + lambda_2 L_S-T + lambda_3 L_LM,   lambdas = 1, 0.5, 1

The alignment losses are computed after mapping back to Euclidean space (log map),
on softmax feature distributions. Ablation switches (Table 2): ``fusion="concat"``
(C: Euclidean concatenation in both fusion stages), ``bd_ss=bd_st=False`` (MA: Mobius
fusion without BD), ``geometry="euclidean"`` (E-BD), ``bd_st=False`` (H-BD-SS: BD on
the speech-speech stage only) and ``bd_ss=False`` (H-BD-ST).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import nn

from common.alm import PrefixLMHead
from common.divergences import bhattacharyya, to_distribution
from common.geometry import clip_tangent, expmap0, logmap0, mobius_add, project_to_ball
from common.layers import EmbeddingCNN

DECISION_PROMPT = 'Determine whether the speech is real or fake. Answer only in one word: "Real" or "Fake".'
CONDITION_PROMPT = "Analyze the speech for unnatural artifacts"


@dataclass
class SatyamConfig:
    input_dims: dict = field(default_factory=lambda: {"whisper": 512, "trillsson": 1024})
    conv_filters: int = 32
    shared_dim: int = 256            # d
    curvature: float = 1.0
    num_prefix: int = 4              # prefix tokens built from g
    geometry: str = "hyperbolic"     # hyperbolic | euclidean (E-BD)
    fusion: str = "satyam"           # satyam | concat (C)
    bd_ss: bool = True               # speech-speech alignment loss L_S-S
    bd_st: bool = True               # speech-prompt alignment loss L_S-T
    lambdas: tuple = (1.0, 0.5, 1.0)  # (paper)
    dropout: float = 0.1
    lm: dict = field(default_factory=lambda: {"backbone": "Qwen/Qwen2-7B"})
    condition_prompt: str = CONDITION_PROMPT
    decision_prompt: str = DECISION_PROMPT
    condition_layer: int | None = None   # default: the middle layer of the LM


class Satyam(nn.Module):
    def __init__(self, cfg: SatyamConfig):
        super().__init__()
        self.cfg = cfg
        self.streams = list(cfg.input_dims)[:2]
        lm_kw = {k: v for k, v in cfg.lm.items() if k != "num_prefix"}
        self.lm = PrefixLMHead(prompt=cfg.decision_prompt, answers=("Real", "Fake"),
                               extra_texts=[cfg.condition_prompt], **lm_kw)
        d, hidden = cfg.shared_dim, self.lm.hidden_size
        self.cnn = nn.ModuleDict({s: EmbeddingCNN(cfg.input_dims[s], (cfg.conv_filters,)) for s in self.streams})
        self.proj = nn.ModuleDict({s: nn.Linear(self.cnn[s].flat_dim, d) for s in self.streams})   # W_w, W_t
        self.gate = nn.ModuleDict({s: nn.Linear(d, d) for s in self.streams})
        # conditioning prompt: intermediate-layer hidden states of the LM, mean-pooled (fixed)
        layer = cfg.condition_layer if cfg.condition_layer is not None else self._middle_layer()
        self.register_buffer("e_A", self.lm.text_representation(cfg.condition_prompt, layer).float())
        self.W_A = nn.Linear(hidden, d)
        width = 3 * d if cfg.fusion == "concat" else d
        self.W_g = nn.Sequential(nn.Dropout(cfg.dropout), nn.Linear(width, cfg.num_prefix * hidden))

    def _middle_layer(self) -> int:
        with torch.no_grad():
            ids = self.lm.prompt_ids[:1][None]
            n = len(self.lm.lm(inputs_embeds=self.lm.embed_ids(ids), output_hidden_states=True).hidden_states)
        return n // 2

    def to_space(self, v: torch.Tensor) -> torch.Tensor:
        return expmap0(clip_tangent(v), self.cfg.curvature) if self.cfg.geometry == "hyperbolic" else v

    def to_euclid(self, h: torch.Tensor) -> torch.Tensor:
        return logmap0(h, self.cfg.curvature) if self.cfg.geometry == "hyperbolic" else h

    def add(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if self.cfg.geometry == "hyperbolic":
            return project_to_ball(mobius_add(x, y, self.cfg.curvature), self.cfg.curvature)
        return x + y

    def forward(self, batch: dict) -> dict:
        e = {}
        for s in self.streams:
            z = self.proj[s](self.cnn[s](batch[s]).flatten(1))
            e[s] = torch.sigmoid(self.gate[s](z)) * z                     # sigmoid gating
        out = {}
        e_A = self.W_A(self.e_A)[None].expand_as(e[self.streams[0]])       # projected prompt representation
        if self.cfg.fusion == "concat":
            u = torch.cat([e[s] for s in self.streams] + [e_A], -1)
        else:
            h_w, h_t = (self.to_space(e[s]) for s in self.streams)
            out["ss"] = (self.to_euclid(h_w), self.to_euclid(h_t))
            h_f = self.add(h_w, h_t)                                       # speech-speech fusion
            h_A = self.to_space(e_A)
            out["st"] = (self.to_euclid(h_f), self.to_euclid(h_A))
            u = self.to_euclid(self.add(h_f, h_A))                         # u_final = log(h_f (+) h_A)
        prefix = self.W_g(u).view(u.shape[0], self.cfg.num_prefix, -1)
        out.update(self.lm(prefix, batch.get("label")))
        return out

    def loss(self, out: dict, batch: dict):
        l1, l2, l3 = self.cfg.lambdas
        parts = {"lm": out["lm_loss"]}
        total = l3 * out["lm_loss"]
        if self.cfg.bd_ss and "ss" in out:
            parts["ss"] = bhattacharyya(*(to_distribution(v) for v in out["ss"]))
            total = total + l1 * parts["ss"]
        if self.cfg.bd_st and "st" in out:
            parts["st"] = bhattacharyya(*(to_distribution(v) for v in out["st"]))
            total = total + l2 * parts["st"]
        logs = {k: v.item() for k, v in parts.items()}
        logs["monitor"] = logs["lm"]
        return total, logs
