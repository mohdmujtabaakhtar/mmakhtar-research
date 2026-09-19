"""Evaluation metrics shared across papers.

All functions accept NumPy arrays or plain lists and return Python floats,
so results can be logged or serialised to JSON directly.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, roc_curve


def accuracy(y_true, y_pred) -> float:
    """Classification accuracy in percent."""
    return 100.0 * float(accuracy_score(y_true, y_pred))


def macro_f1(y_true, y_pred) -> float:
    """Macro-averaged F1 in percent."""
    return 100.0 * float(f1_score(y_true, y_pred, average="macro", zero_division=0))


def mae(y_true, y_pred) -> float:
    """Mean absolute error."""
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true, y_pred) -> float:
    """Root mean squared error."""
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def auc(y_true, scores) -> float:
    """Area under the ROC curve in percent (binary; ``scores`` higher = positive)."""
    return 100.0 * float(roc_auc_score(np.asarray(y_true), np.asarray(scores)))


def best_threshold(y_true, scores) -> float:
    """Decision threshold on ``scores`` that maximises macro-F1 (pick it on validation data)."""
    y_true, scores = np.asarray(y_true), np.asarray(scores)
    candidates = np.unique(scores)
    if len(candidates) > 200:
        candidates = np.quantile(scores, np.linspace(0, 1, 201))
    f1s = [f1_score(y_true, (scores >= t).astype(int), average="macro", zero_division=0)
           for t in candidates]
    return float(candidates[int(np.argmax(f1s))])


def eer(y_true, scores) -> float:
    """Equal error rate in percent for binary detection.

    ``y_true`` uses 1 for the positive (e.g. spoof / fake) class and ``scores``
    are higher for the positive class. Used by the deepfake-detection papers.
    """
    fpr, tpr, _ = roc_curve(np.asarray(y_true), np.asarray(scores))
    fnr = 1.0 - tpr
    idx = int(np.nanargmin(np.abs(fnr - fpr)))
    return 100.0 * float((fpr[idx] + fnr[idx]) / 2.0)


def summarise(values: list[float]) -> tuple[float, float]:
    """Mean and standard deviation across folds or seeds."""
    arr = np.asarray(values, float)
    return float(arr.mean()), float(arr.std())
