"""Train and evaluate TRIO (or a baseline) for source tracing of synthetic speech.

Examples:

    python -m papers.trio.train --synthetic                        # TRIO on generated data
    python -m papers.trio.train --synthetic --model concat         # concatenation fusion
    python -m papers.trio.train --synthetic --model cnn --stream fm2   # single-SPTM CNN
    python -m papers.trio.train --manifest data/asv19/manifest.csv                     # 5-fold CV
    python -m papers.trio.train --manifest data/cfad/manifest.csv --protocol official \\
        --set num_classes=12

Protocol (paper): ASVspoof 2019 LA with train/dev/eval merged into 19 source classes
(A01-A19), 5-fold CV; CFAD with its official split. Adam, 50 epochs, batch 32,
early stopping. Metrics: accuracy and one-vs-all EER averaged over classes.
Manifest columns: ``subject_id``, ``fm1``, ``fm2`` (pooled embeddings), ``label`` and,
for ``--protocol official``, ``split``.
"""
from __future__ import annotations

from pathlib import Path

from common.baselines import EmbeddingCNNClassifier, EmbeddingFCNClassifier
from common.runner import fusion_cli
from papers.trio.model import Trio, TrioConfig

HERE = Path(__file__).parent


def build(name: str, cfg: dict, stream: str):
    dims, n = {s: v["dim"] for s, v in cfg["features"].items()}, cfg["num_classes"]
    b = cfg["baseline"]
    if name in ("trio", "concat"):
        fusion = {"fusion": name}
        return (lambda: Trio(TrioConfig(input_dims=dims, num_classes=n, **{**cfg["model"], **fusion}))), list(dims)
    if name == "cnn":
        return (lambda: EmbeddingCNNClassifier(dims[stream], n, b["filters"], b["dense"], stream=stream)), [stream]
    return (lambda: EmbeddingFCNClassifier(dims[stream], n, b["dense"], stream=stream)), [stream]


if __name__ == "__main__":
    fusion_cli(__doc__, "trio", HERE / "configs" / "trio.yaml", build, ["trio", "concat", "cnn", "fcn"], "trio")
