"""Train and evaluate RENO (or a baseline) for non-verbal emotion recognition.

Examples:

    python -m papers.reno.train --synthetic                          # RENO on generated data
    python -m papers.reno.train --synthetic --model concat           # concatenation fusion
    python -m papers.reno.train --synthetic --model cnn --stream fm1     # single-FM CNN
    python -m papers.reno.train --manifest data/vivae/manifest.csv       # 5-fold CV

Protocol (paper): ASVP-ESD (non-speech), JNV and VIVAE with 5-fold CV (four folds
train, one tests); Adam, lr 1e-3, batch 32, 20 epochs, dropout and early stopping.
Metrics: accuracy and macro-F1. Manifest columns: ``subject_id`` (speaker), ``fm1``,
``fm2`` (pooled embeddings) and ``label`` (emotion).
"""
from __future__ import annotations

from pathlib import Path

from common.baselines import EmbeddingCNNClassifier, EmbeddingFCNClassifier
from common.runner import fusion_cli
from papers.reno.model import Reno, RenoConfig

HERE = Path(__file__).parent


def build(name: str, cfg: dict, stream: str):
    dims, n = {s: v["dim"] for s, v in cfg["features"].items()}, cfg["num_classes"]
    b = cfg["baseline"]
    if name in ("reno", "concat"):
        return (lambda: Reno(RenoConfig(input_dims=dims, num_classes=n,
                                        **{**cfg["model"], "fusion": name}))), list(dims)
    if name == "cnn":
        return (lambda: EmbeddingCNNClassifier(dims[stream], n, b["filters"], b["dense"], stream=stream)), [stream]
    return (lambda: EmbeddingFCNClassifier(dims[stream], n, b["dense"], stream=stream)), [stream]


if __name__ == "__main__":
    fusion_cli(__doc__, "reno", HERE / "configs" / "reno.yaml", build, ["reno", "concat", "cnn", "fcn"], "reno")
