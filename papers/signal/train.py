"""Train and evaluate SIGNAL (or a baseline) for source attribution with open-set detection.

Examples:

    python -m papers.signal.train --synthetic                         # SIGNAL (KNN + GNN)
    python -m papers.signal.train --synthetic --set model.alpha=1.0   # GNN only
    python -m papers.signal.train --synthetic --set model.alpha=0.0   # KNN only
    python -m papers.signal.train --synthetic --model cnn             # CNN baseline
    python -m papers.signal.train --manifest data/diffssd/manifest.csv

Protocol (paper): DiffSSD's predefined train/dev/test splits. The generators present
in the training split are the *seen* classes; test clips from any other generator
(ElevenLabs and PlayHT in DiffSSD) are *unseen*. Adam, lr 1e-3, batch 32, up to 50
epochs, early stopping on dev. Three evaluations:

* DEV and TEST: closed-set attribution over the seen generators;
* ID/OOD: the whole test split, where a clip is predicted "unseen" when
  max(p_ens) < tau, scored as (N + 1)-way accuracy / macro-F1, with EER for the
  seen-vs-unseen decision (score 1 - max p_ens).

Manifest columns: ``subject_id``, ``features`` (pooled embedding .npy), ``generator``
(name or ID) and ``split`` (train / dev / test).
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
from common.baselines import EmbeddingCNNClassifier, EmbeddingFCNClassifier
from common.config import load_config
from common.data import FeatureDataset
from common.runner import eer_one_vs_rest
from common.training import fit, move, set_seed
from papers.signal.model import Signal, SignalConfig

HERE = Path(__file__).parent


def make_synthetic(out_dir, dim: int, n_seen: int = 6, n_unseen: int = 2, per_class: int = 60,
                   seed: int = 0) -> Path:
    """Generators as Gaussian clusters; the unseen ones appear only in the test split."""
    rng = np.random.default_rng(seed)
    out_dir = Path(out_dir)
    (out_dir / "features").mkdir(parents=True, exist_ok=True)
    centres = rng.normal(size=(n_seen + n_unseen, dim)) * 0.9
    rows = []
    for g in range(n_seen + n_unseen):
        for k in range(per_class):
            split = "test" if g >= n_seen else ("train" if k < 0.6 * per_class else
                                                 "dev" if k < 0.75 * per_class else "test")
            name = f"features/G{g}_{k}.npy"
            np.save(out_dir / name, (centres[g] + rng.normal(size=dim)).astype(np.float32))
            rows.append({"subject_id": f"G{g}_{k}", "features": name, "generator": f"gen{g}", "split": split})
    path = out_dir / "manifest.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def prepare(manifest: Path, work: Path):
    """Integer labels: seen generators (those in the training split) -> 0..N-1, others -> N."""
    df = pd.read_csv(manifest)
    seen = sorted(df.loc[df["split"] == "train", "generator"].astype(str).unique())
    index = {g: i for i, g in enumerate(seen)}
    df["label"] = [index.get(str(g), len(seen)) for g in df["generator"]]
    df["features"] = [str((manifest.parent / p).resolve()) if not Path(p).is_absolute() else p
                      for p in df["features"]]
    work.mkdir(parents=True, exist_ok=True)
    df.to_csv(work / "manifest_labels.csv", index=False)
    return work / "manifest_labels.csv", df, seen


@torch.no_grad()
def collect(model, loader, device):
    model.eval()
    zs, ys, ps = [], [], []
    for batch in loader:
        batch = move(batch, device)
        out = model(batch)
        if "z" in out:
            zs.append(out["z"].cpu())
        ps.append((out["p_ens"] if "p_ens" in out else out["logits"].softmax(-1)).cpu())
        ys.append(batch["label"].cpu())
    z = torch.cat(zs) if zs else None
    return z, torch.cat(ys).numpy(), torch.cat(ps).numpy()


def closed_set(y, p, n_seen) -> dict:
    m = y < n_seen
    y, p = y[m], p[m]
    pred = p.argmax(1)
    return {"acc": metrics.accuracy(y, pred), "f1": metrics.macro_f1(y, pred), "eer": eer_one_vs_rest(y, p)}


def open_set(y, p, n_seen, tau) -> dict:
    conf = p.max(1)
    pred = np.where(conf < tau, n_seen, p.argmax(1))
    unseen = (y >= n_seen).astype(int)
    eer = metrics.eer(unseen, 1 - conf) if 0 < unseen.sum() < len(y) else float("nan")
    return {"acc": metrics.accuracy(y, pred), "f1": metrics.macro_f1(y, pred), "eer": eer}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=HERE / "configs" / "signal.yaml")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="config overrides")
    ap.add_argument("--manifest")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--model", default="signal", choices=["signal", "cnn", "fcn"])
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--out")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    cfg = load_config(args.config, args.set)
    tc = cfg["training"]
    if args.epochs:
        tc["epochs"] = args.epochs
    if args.synthetic:
        cfg["features"]["dim"] = cfg["synthetic"]["dim"]
        manifest = make_synthetic("data/synthetic_signal", cfg["features"]["dim"])
    elif args.manifest:
        manifest = Path(args.manifest)
    else:
        ap.error("pass --manifest or --synthetic")
    labelled, df, seen = prepare(Path(manifest), Path("results/signal_work"))
    n = len(seen)
    idx = {s: list(np.where(df["split"].astype(str).str.lower() == s)[0]) for s in ("train", "dev", "test")}
    print(f"{n} seen generators; test clips: {sum(df.loc[idx['test'], 'label'] < n)} seen, "
          f"{sum(df.loc[idx['test'], 'label'] >= n)} unseen")

    set_seed(0)
    loader = lambda ids, shuffle=False: DataLoader(   # noqa: E731
        FeatureDataset(labelled, {"features": 1}, ["label"], ids), batch_size=tc["batch_size"], shuffle=shuffle)
    train_seen = [i for i in idx["train"] if df.loc[i, "label"] < n]
    dim = cfg["features"]["dim"]
    if args.model == "signal":
        model = Signal(SignalConfig(input_dim=dim, num_classes=n, **cfg["model"]))
    elif args.model == "cnn":
        b = cfg["baseline"]
        model = EmbeddingCNNClassifier(dim, n, b["filters"], b["dense"], stream="features")
    else:
        model = EmbeddingFCNClassifier(dim, n, cfg["baseline"]["dense"], stream="features")
    fit(model, loader(train_seen, True), loader(idx["dev"]), epochs=tc["epochs"], lr=tc["lr"],
        patience=tc.get("patience", 10), device=args.device, verbose=False)

    if args.model == "signal":                       # fill the KNN bank with training embeddings
        z, y, _ = collect(model, loader(train_seen), args.device)
        model.fit_knn(torch.as_tensor(z).to(args.device), torch.as_tensor(y).to(args.device))
    _, y_dev, p_dev = collect(model, loader(idx["dev"]), args.device)
    _, y_test, p_test = collect(model, loader(idx["test"]), args.device)
    tau = cfg["model"]["tau"]
    results = {"dev": closed_set(y_dev, p_dev, n), "test": closed_set(y_test, p_test, n),
               "id_ood": open_set(y_test, p_test, n, tau)}
    label = args.model if args.model != "signal" else f"signal (alpha={cfg['model']['alpha']})"
    for split, res in results.items():
        print(f"{label} {split:6s}: " + "  ".join(f"{k} {v:.2f}" for k, v in res.items()))
    out = args.out or ("results/signal.json" if args.model == "signal" else f"results/signal_{args.model}.json")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps({"model": label, "config": cfg, "seen_generators": seen,
                                     "results": results}, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
