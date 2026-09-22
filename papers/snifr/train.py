"""Train and evaluate SNIFR (or a baseline) for fine-grained harmful-content classification.

Examples:

    python -m papers.snifr.train --synthetic                          # SNIFR on generated data
    python -m papers.snifr.train --synthetic --model ct               # single cross-transformer stage
    python -m papers.snifr.train --synthetic --model ec               # also: lc, ea, ep
    python -m papers.snifr.train --synthetic --model unimodal --stream video
    python -m papers.snifr.train --manifest data/fgchcd/manifest.csv  # 5-fold CV

Protocol (paper): one-second clips labelled safe / sexual / violent / both; AST audio
and VideoMAE visual embeddings (768-d each, average-pooled); 5-fold CV; AdamW,
lr 1e-4, weight decay 1e-5, batch 16, 25 epochs, dropout and early stopping.
Reports class-wise accuracy, F1 and one-vs-rest AUC, and their averages.
Manifest columns: ``subject_id`` (source video, so clips of one video stay in one
fold), ``audio``, ``video`` (pooled embeddings) and ``label`` (0 safe, 1 sexual,
2 violent, 3 both).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score, roc_auc_score

from common.runner import fusion_cli
from papers.snifr.model import Snifr, SnifrConfig, SnifrUnimodal

HERE = Path(__file__).parent
CLASSES = ["safe", "sexual", "violent", "both"]


def classwise_scores(y, probs) -> dict:
    """Per-class accuracy (recall), F1 and one-vs-rest AUC, plus their averages (in %)."""
    pred = probs.argmax(1)
    f1s = f1_score(y, pred, labels=range(probs.shape[1]), average=None, zero_division=0)
    res = {}
    for c in range(probs.shape[1]):
        name = CLASSES[c] if c < len(CLASSES) else str(c)
        mask = y == c
        res[f"{name}_acc"] = 100.0 * float((pred[mask] == c).mean()) if mask.any() else float("nan")
        res[f"{name}_f1"] = 100.0 * float(f1s[c])
        res[f"{name}_auc"] = (100.0 * float(roc_auc_score(mask.astype(int), probs[:, c]))
                              if 0 < mask.sum() < len(y) else float("nan"))
    for m in ("acc", "f1", "auc"):
        res[m] = float(np.nanmean([v for k, v in res.items() if k.endswith(f"_{m}")]))
    return res


def build(name: str, cfg: dict, stream: str):
    dims, n = {s: v["dim"] for s, v in cfg["features"].items()}, cfg["num_classes"]
    mc = SnifrConfig(input_dims=dims, num_classes=n, **cfg["model"])
    if name == "unimodal":
        return (lambda: SnifrUnimodal(mc, stream)), [stream]
    mc.fusion = name
    return (lambda: Snifr(mc)), list(dims)


if __name__ == "__main__":
    fusion_cli(__doc__, "snifr", HERE / "configs" / "snifr.yaml", build,
               ["snifr", "ct", "ec", "lc", "ea", "ep", "unimodal"], "snifr", score_fn=classwise_scores)
