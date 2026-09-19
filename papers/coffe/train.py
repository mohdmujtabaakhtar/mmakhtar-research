"""Train and evaluate COFFE (or a baseline) for singing-voice deepfake source attribution.

Examples:

    python -m papers.coffe.train --synthetic                     # COFFE on generated data
    python -m papers.coffe.train --synthetic --model concat      # concatenation fusion (no L_CD)
    python -m papers.coffe.train --synthetic --model cnn --stream fm1   # single-FM CNN
    python -m papers.coffe.train --manifest data/ctrsvdd/manifest.csv --protocol official

Protocol (paper): CtrSVDD synthetic clips from systems A01-A08; the official train
split trains and the official dev split tests. Adam, lr 1e-3, 50 epochs, early
stopping. Metrics: accuracy, macro-F1 and one-vs-all EER averaged over classes.
Manifest columns: ``subject_id``, ``fm1``, ``fm2`` (paths to pooled embeddings),
``label`` (source system, 0-7) and, for ``--protocol official``, ``split``.
"""
from __future__ import annotations

from pathlib import Path

from common.baselines import EmbeddingCNNClassifier, EmbeddingFCNClassifier
from common.runner import fusion_cli
from papers.coffe.model import Coffe, CoffeConfig

HERE = Path(__file__).parent


def build(name: str, cfg: dict, stream: str):
    dims, n = {s: v["dim"] for s, v in cfg["features"].items()}, cfg["num_classes"]
    b = cfg["baseline"]
    if name == "coffe":
        return (lambda: Coffe(CoffeConfig(input_dims=dims, num_classes=n, **cfg["model"]))), list(dims)
    if name == "concat":
        return (lambda: Coffe(CoffeConfig(input_dims=dims, num_classes=n, **{**cfg["model"], "lam": 0.0}))), list(dims)
    if name == "cnn":
        return (lambda: EmbeddingCNNClassifier(dims[stream], n, b["filters"], b["dense"], stream=stream)), [stream]
    return (lambda: EmbeddingFCNClassifier(dims[stream], n, b["dense"], stream=stream)), [stream]


if __name__ == "__main__":
    fusion_cli(__doc__, "coffe", HERE / "configs" / "coffe.yaml", build, ["coffe", "concat", "cnn", "fcn"], "coffe")
