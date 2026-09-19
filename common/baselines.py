"""Downstream baseline heads used across the papers.

Most papers compare their proposed framework against the same two probes
trained on frozen foundation-model features:

* **FCN**: three dense layers (256 -> 128 -> 64) on the time-averaged embedding;
* **CNN**: two 1-D conv blocks (256 and 128 filters, kernel 3, batch norm,
  max-pool 2) followed by the same FCN.

``ConcatFusion`` is the simple multimodal baseline: concatenate the per-modality
representations and apply one of the heads above.
"""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


def mlp(in_dim: int, dims=(256, 128, 64), dropout: float = 0.3) -> nn.Sequential:
    layers, d = [], in_dim
    for h in dims:
        layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(dropout)]
        d = h
    return nn.Sequential(*layers)


class FCNEncoder(nn.Module):
    """(B, T, D) -> (B, 64) via mean pooling over time and an MLP."""

    def __init__(self, in_dim: int, dropout: float = 0.3):
        super().__init__()
        self.net = mlp(in_dim, dropout=dropout)
        self.out_dim = 64

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.mean(dim=1) if x.dim() == 3 else x)


class CNNEncoder(nn.Module):
    """(B, T, D) -> (B, 64) via two conv blocks, flattening and an MLP."""

    def __init__(self, in_dim: int, num_frames: int, dropout: float = 0.3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_dim, 256, 3, padding=1), nn.BatchNorm1d(256), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(256, 128, 3, padding=1), nn.BatchNorm1d(128), nn.ReLU(), nn.MaxPool1d(2),
        )
        self.net = mlp(128 * (num_frames // 4), dropout=dropout)
        self.out_dim = 64

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(self.conv(x.transpose(1, 2)).flatten(1))


class MultiTaskProbe(nn.Module):
    """Encoder per modality + concatenation + one linear head per task.

    With a single modality this is the unimodal FCN/CNN baseline; with several
    it is the concatenation-fusion baseline.
    """

    def __init__(self, modalities: dict[str, tuple[int, int]], tasks: dict[str, int],
                 head: str = "cnn", task_weights: dict[str, float] | None = None):
        super().__init__()
        enc = {
            m: CNNEncoder(d, t) if head == "cnn" else FCNEncoder(d)
            for m, (t, d) in modalities.items()
        }
        self.encoders = nn.ModuleDict(enc)
        width = sum(e.out_dim for e in enc.values())
        self.heads = nn.ModuleDict({k: nn.Linear(width, n) for k, n in tasks.items()})
        self.task_weights = task_weights or {k: 1.0 for k in tasks}

    def forward(self, batch: dict) -> dict:
        h = torch.cat([self.encoders[m](batch[m]) for m in self.encoders], dim=-1)
        return {f"logits_{k}": head(h) for k, head in self.heads.items()}

    def loss(self, out: dict, batch: dict):
        parts = {k: F.cross_entropy(out[f"logits_{k}"], batch[k]) for k in self.heads}
        total = sum(self.task_weights[k] * v for k, v in parts.items())
        return total, {k: v.item() for k, v in parts.items()}
