"""PyTorch implementation of GARUDA from "Bridging the SEA Gap: An Initial Benchmark for Neural
Audio Codec-Synthesized Speech Deepfakes in South-East Asian Languages" (IJCAI 2026).

GARUDA is a small audio-language model (< 1B parameters) for codec-fake detection:

1. A semantic representation (Whisper encoder) and a prosodic / speaker representation
   (x-vector) each pass a convolutional module (Conv1d, kernel 3, max-pool), are
   flattened and filtered by a sigmoid gate.
2. Jensen-Shannon alignment on temperature-scaled feature distributions:
       p_x = softmax(x / tau), p_y = softmax(y / tau), m = (p_x + p_y) / 2
       L_JS = KL(p_x || m) / 2 + KL(p_y || m) / 2
3. The two are concatenated, passed through a fully connected layer of 90 neurons and
   projected into the LM embedding space as a continuous prefix for Qwen2-0.5B, followed
   by the prompt 'Is the speech sample fake or real? Reply in one word "fake" or "real".'

    L = L_LM + lambda L_JS,   tau = 0.5, lambda = 0.4

Two training formats: GARUDA (projection module only, LM frozen) and GARUDA-FT (also
LoRA adapters, rank 8, alpha 32, on the LM's query and value projections).
Ablations (Table 1): single encoder (``streams``), ``fusion="concat"`` (concatenation
without the JS loss) and ``align="kl"`` (KL divergence instead of JS).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import nn

from common.alm import PrefixLMHead
from common.divergences import js_divergence, to_distribution
from common.layers import EmbeddingCNN

PROMPT = 'Is the speech sample fake or real? Reply in one word "fake" or "real".'


def kl_divergence(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    p, q = p.clamp_min(1e-8), q.clamp_min(1e-8)
    return (p * (p.log() - q.log())).sum(-1).mean()


@dataclass
class GarudaConfig:
    input_dims: dict = field(default_factory=lambda: {"whisper": 512, "xvector": 512})
    conv_filters: int = 32
    align_dim: int = 0               # 0: align the flattened features directly (needs equal sizes)
    hidden: int = 90                 # fully connected layer (paper)
    num_prefix: int = 4
    tau: float = 0.5                 # (paper)
    lam: float = 0.4                 # (paper)
    fusion: str = "garuda"           # garuda | concat
    align: str = "js"                # js | kl
    dropout: float = 0.1
    lm: dict = field(default_factory=lambda: {"backbone": "Qwen/Qwen2-0.5B"})
    prompt: str = PROMPT


class Garuda(nn.Module):
    def __init__(self, cfg: GarudaConfig):
        super().__init__()
        self.cfg = cfg
        self.streams = list(cfg.input_dims)
        self.lm = PrefixLMHead(prompt=cfg.prompt, answers=("real", "fake"), **cfg.lm)
        self.cnn = nn.ModuleDict({s: EmbeddingCNN(cfg.input_dims[s], (cfg.conv_filters,)) for s in self.streams})
        self.gate = nn.ModuleDict({s: nn.Conv1d(cfg.conv_filters, cfg.conv_filters, 1) for s in self.streams})
        flat = {s: self.cnn[s].flat_dim for s in self.streams}
        need_proj = cfg.align_dim or (len(set(flat.values())) > 1)
        width = cfg.align_dim or min(flat.values())
        self.proj = nn.ModuleDict({s: nn.Linear(flat[s], width) if need_proj else nn.Identity()
                                   for s in self.streams})
        widths = [width if need_proj else flat[s] for s in self.streams]
        self.fc = nn.Sequential(nn.Linear(sum(widths), cfg.hidden), nn.ReLU(), nn.Dropout(cfg.dropout))
        self.to_prefix = nn.Linear(cfg.hidden, cfg.num_prefix * self.lm.hidden_size)

    def forward(self, batch: dict) -> dict:
        feats = []
        for s in self.streams:
            m = self.cnn[s](batch[s])
            m = torch.sigmoid(self.gate[s](m)) * m                         # sigmoid gating
            feats.append(self.proj[s](m.flatten(1)))
        h = self.fc(torch.cat(feats, dim=-1))
        prefix = self.to_prefix(h).view(h.shape[0], self.cfg.num_prefix, -1)
        out = self.lm(prefix, batch.get("label"))
        out["feats"] = feats
        return out

    def loss(self, out: dict, batch: dict):
        total, logs = out["lm_loss"], {"lm": out["lm_loss"].item()}
        if self.cfg.fusion == "garuda" and len(out["feats"]) == 2:
            p, q = (to_distribution(f, self.cfg.tau) for f in out["feats"])
            align = js_divergence(p, q) if self.cfg.align == "js" else kl_divergence(p, q)
            total = total + self.cfg.lam * align
            logs[self.cfg.align] = align.item()
        logs["monitor"] = logs["lm"]
        return total, logs
