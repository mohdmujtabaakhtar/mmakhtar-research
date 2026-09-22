"""PyTorch implementation of SNIFR from "SNIFR: Boosting Fine-Grained Child Harmful Content
Detection Through Audio-Visual Alignment with Cascaded Cross-Transformer" (INTERSPEECH 2025).

SNIFR (CrosS-Modality INteractIon Cascaded TransFoRmer):

1. Audio (AST) and visual (VideoMAE) representations each pass through a transformer
   encoder (self-attention -> FFN, with residuals and layer norm): intra-modality
   interaction.
2. A two-stage **cascaded cross-transformer** aligns the modalities. In each stage,
   queries come from one modality and keys/values from the other:
       Z_A <- LN(Z_A + Attn(Q_A, K_B, V_B)),  Z_A <- LN(Z_A + FFN(Z_A))
       Z_B <- LN(Z_B + Attn(Q_B, K_A, V_A)),  Z_B <- LN(Z_B + FFN(Z_B))
3. Z_fused = Concat(Z_A^(2), Z_B^(2)) -> classifier (dense 120 -> softmax over
   safe / sexual / violent / both).

``fusion`` selects the paper's baselines: ``ec`` (early concatenation), ``lc`` (late
concatenation, dense 128 per modality), ``ea`` (element-wise average), ``ep``
(element-wise product), ``ct`` (a single cross-transformer stage) or ``snifr``.
``SnifrUnimodal`` is the audio-only / visual-only model.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import nn
import torch.nn.functional as F


class Tokenizer(nn.Module):
    """Turns features into a token sequence of width d_model.

    Frame-level input (B, T, D) is projected frame by frame. A pooled vector
    ((B, D) or (B, 1, D)) is split into ``num_tokens`` equal chunks, each projected
    to d_model, so attention has a sequence to work on.
    """

    def __init__(self, in_dim: int, d_model: int, num_tokens: int, max_len: int = 512):
        super().__init__()
        if in_dim % num_tokens:
            raise ValueError(f"feature size {in_dim} is not divisible by num_tokens={num_tokens}")
        self.num_tokens = num_tokens
        self.chunk = nn.Linear(in_dim // num_tokens, d_model)
        self.frame = nn.Linear(in_dim, d_model)
        self.pos = nn.Parameter(torch.randn(max_len, d_model) * 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2 or x.shape[1] == 1:
            x = x.reshape(x.shape[0], self.num_tokens, -1)
            tokens = self.chunk(x)
        else:
            tokens = self.frame(x)
        return tokens + self.pos[: tokens.shape[1]]


class CrossBlock(nn.Module):
    """One cross-transformer stage: each modality attends to the other, then an FFN."""

    def __init__(self, d: int, heads: int, ff: int, dropout: float):
        super().__init__()
        self.attn = nn.ModuleDict({m: nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True)
                                   for m in ("a", "b")})
        self.ffn = nn.ModuleDict({m: nn.Sequential(nn.Linear(d, ff), nn.GELU(), nn.Dropout(dropout),
                                                   nn.Linear(ff, d)) for m in ("a", "b")})
        self.norm1 = nn.ModuleDict({m: nn.LayerNorm(d) for m in ("a", "b")})
        self.norm2 = nn.ModuleDict({m: nn.LayerNorm(d) for m in ("a", "b")})

    def forward(self, za: torch.Tensor, zb: torch.Tensor):
        a = self.norm1["a"](za + self.attn["a"](za, zb, zb)[0])     # Q from A, K/V from B
        b = self.norm1["b"](zb + self.attn["b"](zb, za, za)[0])     # Q from B, K/V from A
        return self.norm2["a"](a + self.ffn["a"](a)), self.norm2["b"](b + self.ffn["b"](b))


def encoder(d: int, heads: int, ff: int, dropout: float, layers: int) -> nn.TransformerEncoder:
    layer = nn.TransformerEncoderLayer(d, heads, ff, dropout, batch_first=True)
    return nn.TransformerEncoder(layer, layers)


@dataclass
class SnifrConfig:
    input_dims: dict = field(default_factory=lambda: {"audio": 768, "video": 768})
    num_classes: int = 4
    d_model: int = 256
    num_tokens: int = 12         # tokens per pooled 768-d vector (64 features each)
    heads: int = 4
    ff: int = 512
    encoder_layers: int = 1
    cascades: int = 2            # (paper); 1 = the single cross-transformer baseline "CT"
    hidden: int = 120            # classifier dense layer (paper)
    dropout: float = 0.1
    fusion: str = "snifr"        # snifr | ct | ec | lc | ea | ep


class Snifr(nn.Module):
    def __init__(self, cfg: SnifrConfig):
        super().__init__()
        self.cfg = cfg
        self.a_key, self.b_key = list(cfg.input_dims)[:2]          # modality A = audio, B = visual
        d = cfg.d_model
        self.tok = nn.ModuleDict({m: Tokenizer(cfg.input_dims[m], d, cfg.num_tokens)
                                  for m in (self.a_key, self.b_key)})
        self.enc = nn.ModuleDict({m: encoder(d, cfg.heads, cfg.ff, cfg.dropout, cfg.encoder_layers)
                                  for m in (self.a_key, self.b_key)})
        stages = cfg.cascades if cfg.fusion == "snifr" else 1
        self.cross = nn.ModuleList([CrossBlock(d, cfg.heads, cfg.ff, cfg.dropout) for _ in range(stages)])
        self.late = nn.ModuleDict({m: nn.Sequential(nn.Linear(d, 128), nn.ReLU())
                                   for m in (self.a_key, self.b_key)})
        width = {"snifr": 2 * d, "ct": 2 * d, "ec": 2 * d, "lc": 256, "ea": d, "ep": d}[cfg.fusion]
        self.classifier = nn.Sequential(nn.Linear(width, cfg.hidden), nn.ReLU(), nn.Dropout(cfg.dropout),
                                        nn.Linear(cfg.hidden, cfg.num_classes))

    def forward(self, batch: dict) -> dict:
        za, zb = (self.enc[m](self.tok[m](batch[m])) for m in (self.a_key, self.b_key))
        f = self.cfg.fusion
        if f in ("snifr", "ct"):
            for block in self.cross:
                za, zb = block(za, zb)
        pa, pb = za.mean(1), zb.mean(1)                              # pool tokens
        if f in ("snifr", "ct", "ec"):
            fused = torch.cat([pa, pb], -1)
        elif f == "lc":
            fused = torch.cat([self.late[self.a_key](pa), self.late[self.b_key](pb)], -1)
        elif f == "ea":
            fused = (pa + pb) / 2
        else:
            fused = pa * pb
        return {"logits": self.classifier(fused)}

    def loss(self, out: dict, batch: dict):
        ce = F.cross_entropy(out["logits"], batch["label"])
        return ce, {"ce": ce.item()}


class SnifrUnimodal(nn.Module):
    """Audio-only or visual-only model: transformer encoder -> dense 120 -> softmax."""

    def __init__(self, cfg: SnifrConfig, stream: str):
        super().__init__()
        self.stream = stream
        self.tok = Tokenizer(cfg.input_dims[stream], cfg.d_model, cfg.num_tokens)
        self.enc = encoder(cfg.d_model, cfg.heads, cfg.ff, cfg.dropout, cfg.encoder_layers)
        self.classifier = nn.Sequential(nn.Linear(cfg.d_model, cfg.hidden), nn.ReLU(), nn.Dropout(cfg.dropout),
                                        nn.Linear(cfg.hidden, cfg.num_classes))

    def forward(self, batch: dict) -> dict:
        return {"logits": self.classifier(self.enc(self.tok(batch[self.stream])).mean(1))}

    def loss(self, out: dict, batch: dict):
        ce = F.cross_entropy(out["logits"], batch["label"])
        return ce, {"ce": ce.item()}
