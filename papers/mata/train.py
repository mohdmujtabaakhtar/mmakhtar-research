"""Train and evaluate MATA (or a baseline) for non-verbal emotion recognition.

Examples:

    python -m papers.mata.train --synthetic                          # MATA on generated data
    python -m papers.mata.train --synthetic --set model.use_mha=false    # "Fusion with OT" ablation
    python -m papers.mata.train --synthetic --model concat           # concatenation fusion
    python -m papers.mata.train --synthetic --model cnn --stream fm1     # single-FM CNN
    python -m papers.mata.train --manifest data/asvp_esd/manifest.csv    # 5-fold CV

Protocol (paper): ASVP-ESD (non-speech part), JNV, VIVAE and CREMA-D, 5-fold CV;
Adam, lr 1e-3, batch 32, 50 epochs, cross-entropy, dropout and early stopping.
Metrics: accuracy and macro-F1. Manifest columns: ``subject_id`` (speaker),
``fm1``, ``fm2`` (pooled embeddings) and ``label`` (emotion).
"""
from __future__ import annotations

from pathlib import Path

from common.baselines import EmbeddingCNNClassifier, EmbeddingFCNClassifier
from common.runner import fusion_cli
from papers.mata.model import Mata, MataConfig

HERE = Path(__file__).parent


def build(name: str, cfg: dict, stream: str):
    dims, n = {s: v["dim"] for s, v in cfg["features"].items()}, cfg["num_classes"]
    b = cfg["baseline"]
    if name in ("mata", "concat"):
        return (lambda: Mata(MataConfig(input_dims=dims, num_classes=n,
                                        **{**cfg["model"], "fusion": name}))), list(dims)
    if name == "cnn":
        return (lambda: EmbeddingCNNClassifier(dims[stream], n, b["filters"], b["dense"], stream=stream)), [stream]
    return (lambda: EmbeddingFCNClassifier(dims[stream], n, b["dense"], stream=stream)), [stream]


if __name__ == "__main__":
    fusion_cli(__doc__, "mata", HERE / "configs" / "mata.yaml", build, ["mata", "concat", "cnn", "fcn"], "mata")
