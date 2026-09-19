"""Train and evaluate PARROT (or a baseline) for speech emotion recognition.

Examples:

    python -m papers.parrot.train --synthetic                         # PARROT on generated data
    python -m papers.parrot.train --synthetic --model concat          # concatenation fusion
    python -m papers.parrot.train --synthetic --model cnn --stream fm1    # single-PTM CNN
    python -m papers.parrot.train --manifest data/cremad/manifest.csv     # 5-fold CV

Protocol (paper): CREMA-D, Emo-DB and MESD with speaker-independent 5-fold CV
(four folds train, one tests), Adam, lr 1e-3, batch 32, 50 epochs, dropout and
early stopping. Metrics: accuracy and macro-F1. Manifest columns: ``subject_id``
(speaker), ``fm1``, ``fm2`` (pooled embeddings) and ``label`` (emotion).
"""
from __future__ import annotations

from pathlib import Path

from common.baselines import EmbeddingCNNClassifier, EmbeddingFCNClassifier
from common.runner import fusion_cli
from papers.parrot.model import Parrot, ParrotConfig

HERE = Path(__file__).parent


def build(name: str, cfg: dict, stream: str):
    dims, n = {s: v["dim"] for s, v in cfg["features"].items()}, cfg["num_classes"]
    b = cfg["baseline"]
    if name in ("parrot", "concat", "hadamard", "ot"):
        return (lambda: Parrot(ParrotConfig(input_dims=dims, num_classes=n,
                                            **{**cfg["model"], "fusion": name}))), list(dims)
    if name == "cnn":
        return (lambda: EmbeddingCNNClassifier(dims[stream], n, b["filters"], b["dense"], stream=stream)), [stream]
    return (lambda: EmbeddingFCNClassifier(dims[stream], n, b["dense"], stream=stream)), [stream]


if __name__ == "__main__":
    fusion_cli(__doc__, "parrot", HERE / "configs" / "parrot.yaml", build,
               ["parrot", "concat", "hadamard", "ot", "cnn", "fcn"], "parrot")
