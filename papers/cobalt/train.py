"""Train and evaluate COBALT (or a baseline) for cough-based TB screening.

Examples:

    # smoke test on generated data, no downloads needed
    python -m papers.cobalt.train --synthetic

    # CODA TB DREAM features: MFCC frames + PaSST tokens
    python -m papers.cobalt.train --manifest data/coda/manifest.csv

    # Euclidean variant (COBALT-E) and fusion baselines, same splits
    python -m papers.cobalt.train --manifest ... --set model.geometry=euclidean
    python -m papers.cobalt.train --manifest ... --set model.fusion=mobius
    python -m papers.cobalt.train --manifest ... --model concat_cnn
    python -m papers.cobalt.train --manifest ... --model cnn_probe --streams stream2

Protocol (paper): participant-disjoint 5-fold cross-validation, training on four
folds and evaluating on the fifth, 50 epochs, batch 32, Adam. Reports accuracy,
macro-F1 and AUC. The manifest needs ``subject_id``, ``stream1``, ``stream2``
(paths to (frames, dim) .npy files) and ``label`` (1 = TB positive).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from common import metrics
from common.baselines import MultiTaskProbe
from common.config import load_config
from common.data import FeatureDataset, subject_kfold
from common.training import fit, move, set_seed
from papers.cobalt.model import Cobalt, CobaltConfig

HERE = Path(__file__).parent


def make_synthetic(out_dir, dims: dict, frames: dict, n_subjects: int = 60, per_subject: int = 4,
                   seed: int = 0) -> Path:
    """Two feature streams, each carrying part of the TB signal, so fusing them helps.
    Stream 2 is corrupted by loud noise on a quarter of the recordings (an unreliable stream)."""
    rng = np.random.default_rng(seed)
    out_dir = Path(out_dir)
    (out_dir / "features").mkdir(parents=True, exist_ok=True)
    direction = {s: rng.normal(size=dims[s]) for s in dims}
    rows = []
    for subj in range(n_subjects):
        label = subj % 2
        for k in range(per_subject):
            row = {"subject_id": f"P{subj:03d}", "label": label}
            for s in dims:
                x = rng.normal(size=(frames[s], dims[s]))
                x += (0.6 if label else -0.6) * direction[s] * (rng.random() < 0.75)
                if s == "stream2" and rng.random() < 0.25:
                    x += rng.normal(size=x.shape) * 3
                name = f"features/{s}_P{subj:03d}_{k}.npy"
                np.save(out_dir / name, x.astype(np.float32))
                row[s] = name
            rows.append(row)
    path = out_dir / "manifest.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


@torch.no_grad()
def evaluate(model, loader, device) -> dict:
    model.eval()
    probs, preds, ys = [], [], []
    for batch in loader:
        batch = move(batch, device)
        out = model(batch)
        logits = out["logits"] if "logits" in out else out["logits_label"]
        p = logits.softmax(-1)
        probs.append(p[:, 1].cpu()); preds.append(p.argmax(-1).cpu()); ys.append(batch["label"].cpu())
    s, pred, y = (torch.cat(v).numpy() for v in (probs, preds, ys))
    return {"acc": metrics.accuracy(y, pred), "f1": metrics.macro_f1(y, pred), "auc": metrics.auc(y, s)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=HERE / "configs" / "cobalt.yaml")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="config overrides")
    ap.add_argument("--manifest")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--model", default="cobalt", choices=["cobalt", "concat_cnn", "concat_fcn",
                                                          "cnn_probe", "fcn_probe"])
    ap.add_argument("--streams", default="stream1,stream2", help="streams used by the probe baselines")
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--out", default="results/cobalt.json")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    cfg = load_config(args.config, args.set)
    tc = cfg["training"]
    if args.epochs:
        tc["epochs"] = args.epochs
    frames = {s: v["frames"] for s, v in cfg["features"].items()}
    dims = {s: v["dim"] for s, v in cfg["features"].items()}
    if args.synthetic:
        manifest = make_synthetic("data/synthetic_cobalt", dims, frames)
    elif args.manifest:
        manifest = Path(args.manifest)
    else:
        ap.error("pass --manifest or --synthetic")

    subjects = pd.read_csv(manifest)["subject_id"].to_numpy()
    streams = frames if args.model in ("cobalt", "concat_cnn", "concat_fcn") else \
        {s: frames[s] for s in args.streams.split(",")}
    results = []
    for seed in tc["seeds"]:
        for fold, (tr, _, te) in enumerate(subject_kfold(subjects, k=tc["folds"], seed=seed, val=False)):
            set_seed(seed * 100 + fold)
            train = DataLoader(FeatureDataset(manifest, streams, ["label"], tr), batch_size=tc["batch_size"], shuffle=True)
            test = DataLoader(FeatureDataset(manifest, streams, ["label"], te), batch_size=tc["batch_size"])
            if args.model == "cobalt":
                model = Cobalt(CobaltConfig(input_dims=dims, **cfg["model"]))
            else:
                model = MultiTaskProbe({s: (frames[s], dims[s]) for s in streams}, {"label": 2},
                                       head=args.model.split("_")[-1] if "concat" in args.model
                                       else args.model.split("_")[0])
            fit(model, train, None, epochs=tc["epochs"], lr=tc["lr"], device=args.device, verbose=False)
            res = evaluate(model, test, args.device)
            print(f"seed {seed} fold {fold + 1}: " + "  ".join(f"{k} {v:.2f}" for k, v in res.items()))
            results.append(res)

    summary = {k: metrics.summarise([r[k] for r in results]) for k in results[0]}
    label = args.model if args.model != "cobalt" else f"cobalt ({cfg['model']['fusion']}, {cfg['model']['geometry']})"
    print(f"\n{label}: " + "  ".join(f"{k} {m:.2f} ± {s:.2f}" for k, (m, s) in summary.items()))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"model": label, "config": cfg, "summary": summary,
                                          "runs": results}, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
