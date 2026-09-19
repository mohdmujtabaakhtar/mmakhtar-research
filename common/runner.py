"""Shared experiment runner for the utterance-level classification papers.

Most papers in this repository follow the same protocol: pre-extracted features
from one or two frozen foundation models, a light downstream network, and either
k-fold cross-validation or the dataset's official train/dev/test split. This
module holds that protocol once so each paper's ``train.py`` only builds its
model:

    results = run_experiment(manifest, {"fm1": 1, "fm2": 1}, build_model, cfg["training"])

Manifest columns: ``subject_id``, one path column per stream, the label column
(default ``label``) and, for the official-split protocol, ``split`` with values
``train`` / ``dev`` / ``test``.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from common import metrics
from common.data import FeatureDataset, subject_kfold
from common.training import fit, move, set_seed


# ----------------------------------------------------------------- synthetic data

def make_synthetic_streams(out_dir, streams: dict[str, tuple[int, int]], n_classes: int,
                           n_subjects: int = 40, per_subject: int = 6, strength: float = 0.8,
                           informative: float = 0.7, official_split: bool = False,
                           seed: int = 0) -> Path:
    """Write a small multi-stream classification dataset for smoke tests.

    Each stream (name -> (frames, dim)) carries the class signal on only a random
    ``informative`` fraction of the samples, independently per stream, so that
    fusing streams helps. ``frames == 1`` stores utterance-level vectors (dim,).
    With ``official_split`` a subject-level ``split`` column (70/15/15) is added.
    """
    rng = np.random.default_rng(seed)
    out_dir = Path(out_dir)
    (out_dir / "features").mkdir(parents=True, exist_ok=True)
    protos = {s: rng.normal(size=(n_classes, d)) for s, (_, d) in streams.items()}
    rows = []
    for subj in range(n_subjects):
        split = "train" if subj % 20 < 14 else ("dev" if subj % 20 < 17 else "test")
        for k in range(per_subject):
            label = int(rng.integers(n_classes))
            row = {"subject_id": f"S{subj:03d}", "label": label}
            if official_split:
                row["split"] = split
            for s, (frames, dim) in streams.items():
                signal = strength * protos[s][label] * (rng.random() < informative)
                x = rng.normal(size=(frames, dim)) + signal
                x = x[0] if frames == 1 else x
                name = f"features/{s}_S{subj:03d}_{k}.npy"
                np.save(out_dir / name, x.astype(np.float32))
                row[s] = name
            rows.append(row)
    path = out_dir / "manifest.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


# ----------------------------------------------------------------- evaluation

@torch.no_grad()
def predict(model, loader, device, label_key: str = "label", logits_key: str = "logits"):
    """Returns (labels (N,), class probabilities (N, C)) as NumPy arrays."""
    model.eval()
    ys, ps = [], []
    for batch in loader:
        batch = move(batch, device)
        out = model(batch)
        ps.append(out[logits_key].float().softmax(-1).cpu())
        ys.append(batch[label_key].cpu())
    return torch.cat(ys).numpy(), torch.cat(ps).numpy()


def eer_one_vs_rest(y, probs) -> float:
    """Average of per-class one-vs-rest EERs (classes absent from ``y`` are skipped)."""
    values = [metrics.eer((y == c).astype(int), probs[:, c]) for c in range(probs.shape[1])
              if 0 < (y == c).sum() < len(y)]
    return float(np.mean(values)) if values else float("nan")


def classification_scores(y, probs) -> dict:
    """Accuracy, macro-F1 and EER (binary: class 1 = positive; multi-class: one-vs-rest)."""
    pred = probs.argmax(1)
    eer = metrics.eer(y, probs[:, 1]) if probs.shape[1] == 2 else eer_one_vs_rest(y, probs)
    return {"acc": metrics.accuracy(y, pred), "f1": metrics.macro_f1(y, pred), "eer": eer}


# ----------------------------------------------------------------- splits

def carve_validation(train_idx, subjects, frac: float = 0.1, seed: int = 0):
    """Hold out ``frac`` of the training subjects for early stopping."""
    train_idx = np.asarray(train_idx)
    subj = np.unique(subjects[train_idx])
    rng = np.random.default_rng(seed)
    held = set(rng.choice(subj, size=max(1, int(round(frac * len(subj)))), replace=False))
    val = [i for i in train_idx if subjects[i] in held]
    train = [i for i in train_idx if subjects[i] not in held]
    return train, val


def iterate_splits(df: pd.DataFrame, protocol: str = "kfold", folds: int = 5, seed: int = 0,
                   val_frac: float = 0.1):
    """Yields (name, train_idx, val_idx, test_idx).

    ``official`` uses the ``split`` column (dev = early-stopping set);
    ``kfold`` uses subject-wise k-fold CV with ``val_frac`` of the training
    subjects held out for early stopping.
    """
    if protocol == "official":
        split = df["split"].astype(str).str.lower().replace({"val": "dev", "valid": "dev", "eval": "test"})
        idx = {s: list(np.where(split == s)[0]) for s in ("train", "dev", "test")}
        yield "official", idx["train"], idx["dev"], idx["test"]
        return
    subjects = df["subject_id"].astype(str).to_numpy()
    for fold, (tr, _, te) in enumerate(subject_kfold(subjects, k=folds, seed=seed, val=False)):
        tr, va = carve_validation(tr, subjects, val_frac, seed * 100 + fold) if val_frac else (tr, [])
        yield f"fold{fold + 1}", tr, va, te


# ----------------------------------------------------------------- experiment

def run_experiment(manifest, streams: dict[str, int], build_model, tc: dict, *, device="cpu",
                   label_key: str = "label", protocol: str = "kfold", score_fn=classification_scores,
                   extra_columns: list[str] | None = None, verbose: bool = True) -> list[dict]:
    """Train and evaluate ``build_model()`` on every split; returns one score dict per split.

    Args:
        streams: manifest column -> number of frames (1 for utterance-level vectors).
        tc: training config with ``lr``, ``batch_size``, ``epochs`` and optionally
            ``patience``, ``folds``, ``seeds``, ``optimizer``, ``weight_decay``.
    """
    df = pd.read_csv(manifest)
    labels = [label_key, *(extra_columns or [])]
    results = []
    for seed in tc.get("seeds", [0]):
        for name, tr, va, te in iterate_splits(df, protocol, tc.get("folds", 5), seed,
                                               tc.get("val_frac", 0.1)):
            set_seed(seed * 100 + len(results))
            make = lambda idx, shuffle: DataLoader(   # noqa: E731
                FeatureDataset(manifest, streams, labels, idx), batch_size=tc["batch_size"],
                shuffle=shuffle, drop_last=shuffle and len(idx) > tc["batch_size"])
            model = build_model()
            fit(model, make(tr, True), make(va, False) if len(va) else None, epochs=tc["epochs"],
                lr=tc["lr"], patience=tc.get("patience", 10), device=device,
                optimizer=tc.get("optimizer", "adam"), weight_decay=tc.get("weight_decay", 0.0),
                verbose=False)
            y, probs = predict(model, make(te, False), device, label_key)
            res = score_fn(y, probs)
            if verbose:
                print(f"seed {seed} {name}: " + "  ".join(f"{k} {v:.2f}" for k, v in res.items()))
            results.append(res)
    return results


def summarise_and_save(results: list[dict], out: str | Path, label: str, cfg: dict) -> dict:
    """Print mean ± std over splits and write a JSON report."""
    summary = {k: metrics.summarise([r[k] for r in results]) for k in results[0]}
    print(f"\n{label}: " + "  ".join(f"{k} {m:.2f} ± {s:.2f}" for k, (m, s) in summary.items()))
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps({"model": label, "config": cfg, "summary": summary,
                                     "runs": results}, indent=2))
    print(f"wrote {out}")
    return summary


# ----------------------------------------------------------------- command line

def fusion_cli(doc: str, name: str, default_config, build, model_choices, default_model: str,
               score_fn=classification_scores) -> None:
    """Command-line entry point shared by the two-representation fusion papers.

    ``build(model_name, cfg, stream) -> (factory, used_streams)`` returns a
    zero-argument model factory and the manifest columns that model reads.
    Config sections: ``features`` (stream -> {dim}), ``num_classes``, ``model``,
    ``baseline``, ``training`` (incl. ``protocol``: kfold | official).
    """
    import argparse

    import torch as _torch

    from common.config import load_config

    ap = argparse.ArgumentParser(description=doc, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=default_config)
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="config overrides")
    ap.add_argument("--manifest", help="CSV manifest of pre-extracted features")
    ap.add_argument("--synthetic", action="store_true", help="run on generated data (smoke test)")
    ap.add_argument("--model", default=default_model, choices=model_choices)
    ap.add_argument("--stream", default="fm1", help="representation used by single-model baselines")
    ap.add_argument("--protocol", choices=["kfold", "official"], help="overrides training.protocol")
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--out", help=f"JSON report (default results/{name}[_<model>].json)")
    ap.add_argument("--device", default="cuda" if _torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    cfg = load_config(args.config, args.set)
    tc = cfg["training"]
    protocol = args.protocol or tc.get("protocol", "kfold")
    dims = {s: v["dim"] for s, v in cfg["features"].items()}
    if args.synthetic:
        syn = dict(cfg.get("synthetic", {}))
        for s, d in syn.pop("dims", {}).items():      # smaller feature sizes keep smoke tests fast
            cfg["features"][s]["dim"] = dims[s] = d
        for key, value in syn.pop("overrides", {}).items():   # e.g. a toy language model
            node = cfg
            *parents, leaf = key.split(".")
            for part in parents:
                node = node[part]
            node[leaf] = value
        manifest = make_synthetic_streams(f"data/synthetic_{name}", {s: (1, d) for s, d in dims.items()},
                                          cfg["num_classes"], official_split=protocol == "official",
                                          **syn)
    elif args.manifest:
        manifest = args.manifest
    else:
        ap.error("pass --manifest or --synthetic")
    if args.epochs:
        tc["epochs"] = args.epochs

    factory, used = build(args.model, cfg, args.stream)
    results = run_experiment(manifest, {s: 1 for s in used}, factory, tc, device=args.device,
                             protocol=protocol, score_fn=score_fn)
    label = args.model if args.model == default_model else f"{args.model} ({'+'.join(used)})"
    out = args.out or (f"results/{name}.json" if args.model == default_model else f"results/{name}_{args.model}.json")
    summarise_and_save(results, out, label, cfg)
