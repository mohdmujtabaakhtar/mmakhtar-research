"""Datasets over pre-extracted foundation-model features.

Every paper in this repository works on *frozen* foundation-model embeddings,
so the pipeline is always two steps:

1. extract features once with ``common/features`` and save them as ``.npy``
   arrays of shape ``(frames, dim)`` (or ``(dim,)`` for utterance-level models);
2. train a light downstream model on those arrays through a CSV manifest.

A manifest is a CSV file with one row per sample. Required columns:

* ``subject_id`` - speaker / participant ID, used for subject-wise splits;
* one column per modality holding the path to its ``.npy`` file
  (column names are chosen in the paper's config, e.g. ``video`` and ``audio``);
* one column per label (e.g. ``diagnosis`` and ``severity``), integer-coded or text
  (text labels such as a ``language`` column are mapped to integers in sorted order).

Relative feature paths are resolved against the manifest's directory.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


def fix_length(x: np.ndarray, num_frames: int) -> np.ndarray:
    """Centre-crop or zero-pad a (frames, dim) array to exactly ``num_frames`` rows."""
    if x.ndim == 1:
        x = x[None, :]
    t = x.shape[0]
    if t >= num_frames:
        start = (t - num_frames) // 2
        return x[start:start + num_frames]
    pad = np.zeros((num_frames - t, x.shape[1]), dtype=x.dtype)
    return np.concatenate([x, pad], axis=0)


class FeatureDataset(Dataset):
    """Loads pre-extracted features and labels listed in a manifest CSV."""

    def __init__(self, manifest: str | Path, modalities: dict[str, int],
                 label_columns: list[str], indices: list[int] | None = None):
        """
        Args:
            manifest: path to the CSV manifest.
            modalities: mapping of manifest column -> number of frames to crop/pad to.
            label_columns: manifest columns holding integer labels.
            indices: optional subset of row indices (for cross-validation folds).
        """
        self.root = Path(manifest).parent
        df = pd.read_csv(manifest)
        missing = [c for c in ["subject_id", *modalities, *label_columns] if c not in df.columns]
        if missing:
            raise ValueError(f"Manifest {manifest} is missing columns: {missing}")
        # Text-valued label columns get one integer code per value, fixed over the
        # whole manifest so that codes agree across cross-validation folds.
        self.label_maps = {c: {v: i for i, v in enumerate(sorted(df[c].astype(str).unique()))}
                           for c in label_columns if not pd.api.types.is_numeric_dtype(df[c])}
        self.df = df.iloc[indices].reset_index(drop=True) if indices is not None else df
        self.modalities = modalities
        self.label_columns = label_columns

    def __len__(self) -> int:
        return len(self.df)

    def _load(self, path: str) -> np.ndarray:
        p = Path(path)
        return np.load(p if p.is_absolute() else self.root / p).astype(np.float32)

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        row = self.df.iloc[i]
        item = {m: torch.from_numpy(fix_length(self._load(row[m]), n))
                for m, n in self.modalities.items()}
        for col in self.label_columns:
            value = self.label_maps[col][str(row[col])] if col in self.label_maps else int(row[col])
            item[col] = torch.tensor(value, dtype=torch.long)
        return item


def subject_kfold(subject_ids, k: int = 5, seed: int = 0, val: bool = True):
    """Subject-wise k-fold splits, optionally with a held-out validation fold.

    Yields ``(train_idx, val_idx, test_idx)`` for each of the ``k`` runs. Fold ``i``
    is the test set; with ``val=True`` fold ``(i + 1) % k`` is the validation set,
    otherwise ``val_idx`` is empty and the other ``k - 1`` folds train. No
    participant ever appears in more than one split of a run.
    """
    subject_ids = np.asarray(subject_ids)
    subjects = np.unique(subject_ids)
    if len(subjects) < k:
        raise ValueError(f"Need at least {k} subjects for {k}-fold CV, got {len(subjects)}")
    rng = np.random.default_rng(seed)
    rng.shuffle(subjects)
    folds = np.array_split(subjects, k)
    for i in range(k):
        test_s, val_s = set(folds[i]), set(folds[(i + 1) % k]) if val else set()
        test = [j for j, s in enumerate(subject_ids) if s in test_s]
        val_idx = [j for j, s in enumerate(subject_ids) if s in val_s]
        train = [j for j, s in enumerate(subject_ids) if s not in test_s | val_s]
        yield train, val_idx, test


def make_synthetic_manifest(out_dir: str | Path, modalities: dict[str, tuple[int, int]],
                            n_subjects: int = 20, samples_per_subject: int = 16,
                            n_diagnosis: int = 3, n_severity: int = 3, seed: int = 0) -> Path:
    """Write a small synthetic dataset with a learnable signal, for smoke tests.

    Args:
        out_dir: directory to write ``manifest.csv`` and ``features/*.npy`` into.
        modalities: column name -> (frames, dim) of the fake features.
    Returns:
        Path to the written manifest.
    """
    out_dir = Path(out_dir)
    (out_dir / "features").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    prototypes = {m: rng.normal(size=(n_diagnosis, d)) for m, (_, d) in modalities.items()}
    rows = []
    for s in range(n_subjects):
        diagnosis = s % n_diagnosis
        for k in range(samples_per_subject):
            severity = int(rng.integers(n_severity))
            row = {"subject_id": f"S{s:03d}", "diagnosis": diagnosis, "severity": severity}
            for m, (t, d) in modalities.items():
                signal = prototypes[m][diagnosis] * (1.0 + 0.5 * severity)
                x = rng.normal(size=(t, d)) + signal
                name = f"features/{m}_S{s:03d}_{k}.npy"
                np.save(out_dir / name, x.astype(np.float32))
                row[m] = name
            rows.append(row)
    path = out_dir / "manifest.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path
