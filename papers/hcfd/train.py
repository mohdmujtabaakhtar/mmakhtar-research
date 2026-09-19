"""Train and evaluate PHOENIX-Mamba (or a probe baseline) on one HCFK subset.

Examples:

    # smoke test on generated data, no downloads needed
    python -m papers.hcfd.train --synthetic

    # one clinical condition / language, e.g. English depression (DAIC-WOZ)
    python -m papers.hcfd.train --manifest data/hcfk/en_dep/manifest.csv

    # seen/unseen codec protocol: hold two codec families out of training
    python -m papers.hcfd.train --manifest data/hcfk/en_dep/manifest.csv --unseen-codecs snac,audiodec

    # baselines and ablations on the same splits
    python -m papers.hcfd.train --manifest ... --model cnn_probe
    python -m papers.hcfd.train --manifest ... --set model.backbone=bigru
    python -m papers.hcfd.train --manifest ... --set model.num_evidence=1
    python -m papers.hcfd.train --manifest ... --set model.geometry=euclidean

The manifest needs ``subject_id``, ``features`` (path to a (frames, dim) .npy),
``label`` (0 = bona fide, 1 = codec-fake) and, to use the official speaker-disjoint
splits, a ``split`` column with train / dev / test. Without ``split``, subject-wise
5-fold cross-validation is used. An optional ``codec`` column enables
``--unseen-codecs``. Accuracy and macro-F1 use a threshold chosen on dev; EER is
threshold-free.
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
from papers.hcfd.model import PhoenixConfig, PhoenixMamba

HERE = Path(__file__).parent


def make_synthetic(out_dir: str | Path, frames: int = 48, dim: int = 64, n_subjects: int = 30,
                   per_subject: int = 12, seed: int = 0) -> Path:
    """Bona fide vs codec-fake sequences. Each of four 'codecs' leaves a different
    artefact pattern on a few random frames, so fakes form several distinct modes
    and the evidence is local rather than spread over the whole utterance."""
    rng = np.random.default_rng(seed)
    out_dir = Path(out_dir)
    (out_dir / "features").mkdir(parents=True, exist_ok=True)
    codecs = ["encodec", "dac", "snac", "audiodec"]
    artefacts = {c: rng.normal(size=dim) * 1.5 for c in codecs}
    rows = []
    for s in range(n_subjects):
        speaker = rng.normal(size=dim) * 0.5
        split = "train" if s < 0.6 * n_subjects else ("dev" if s < 0.8 * n_subjects else "test")
        for k in range(per_subject):
            x = rng.normal(size=(frames, dim)) + speaker
            label, codec = k % 2, "bonafide"
            if label:
                codec = codecs[int(rng.integers(len(codecs)))]
                hit = rng.choice(frames, size=frames // 6, replace=False)
                x[hit] += artefacts[codec]
            name = f"features/S{s:03d}_{k}.npy"
            np.save(out_dir / name, x.astype(np.float32))
            rows.append({"subject_id": f"S{s:03d}", "features": name, "label": label,
                         "codec": codec, "split": split})
    path = out_dir / "manifest.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


@torch.no_grad()
def scores_and_labels(model, loader, device):
    model.eval()
    scores, labels = [], []
    for batch in loader:
        batch = move(batch, device)
        out = model(batch)
        logits = out["logits"] if "logits" in out else out["logits_label"]
        scores.append(logits.softmax(-1)[:, 1].cpu())
        labels.append(batch["label"].cpu())
    return torch.cat(scores).numpy(), torch.cat(labels).numpy()


def evaluate(model, dev_loader, test_loader, device) -> dict:
    dev_s, dev_y = scores_and_labels(model, dev_loader, device)
    thr = metrics.best_threshold(dev_y, dev_s)
    s, y = scores_and_labels(model, test_loader, device)
    pred = (s >= thr).astype(int)
    return {"acc": metrics.accuracy(y, pred), "f1": metrics.macro_f1(y, pred),
            "eer": metrics.eer(y, s), "auc": metrics.auc(y, s)}


def build(name: str, cfg: dict, frames: int, dim: int):
    if name == "phoenix":
        return PhoenixMamba(PhoenixConfig(input_dim=dim, **cfg["model"]))
    head = name.split("_")[0]  # cnn_probe / fcn_probe
    return MultiTaskProbe({"features": (frames, dim)}, {"label": 2}, head=head)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", help="default: configs/phoenix.yaml (configs/phoenix_smoke.yaml with --synthetic)")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="config overrides")
    ap.add_argument("--manifest")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--model", default="phoenix", choices=["phoenix", "cnn_probe", "fcn_probe"])
    ap.add_argument("--unseen-codecs", default="", help="comma-separated codec names held out of training")
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--out", default="results/phoenix.json")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    config = args.config or HERE / "configs" / ("phoenix_smoke.yaml" if args.synthetic else "phoenix.yaml")
    cfg = load_config(config, args.set)
    tc = cfg["training"]
    if args.epochs:
        tc["epochs"] = args.epochs
    frames, dim = cfg["features"]["frames"], cfg["features"]["dim"]
    if args.synthetic:
        manifest = make_synthetic("data/synthetic_hcfd", frames=frames, dim=dim)
    elif args.manifest:
        manifest = Path(args.manifest)
    else:
        ap.error("pass --manifest or --synthetic")

    df = pd.read_csv(manifest)
    unseen = {c.strip() for c in args.unseen_codecs.split(",") if c.strip()}
    if unseen and "codec" not in df.columns:
        ap.error("--unseen-codecs needs a 'codec' column in the manifest")

    def drop_unseen(idx):  # remove held-out codec families from train / dev
        return [i for i in idx if not unseen or df.loc[i, "codec"] not in unseen]

    if "split" in df.columns:
        runs = [(drop_unseen(df.index[df.split == "train"].tolist()),
                 drop_unseen(df.index[df.split == "dev"].tolist()),
                 df.index[df.split == "test"].tolist())]
    else:
        runs = [(drop_unseen(tr), drop_unseen(va), te)
                for tr, va, te in subject_kfold(df.subject_id, k=tc["folds"], seed=0)]

    results = []
    for seed in tc["seeds"]:
        for r, (tr, va, te) in enumerate(runs):
            set_seed(seed * 100 + r)
            loaders = [DataLoader(FeatureDataset(manifest, {"features": frames}, ["label"], idx),
                                  batch_size=tc["batch_size"], shuffle=(i == 0))
                       for i, idx in enumerate((tr, va, te))]
            model = build(args.model, cfg, frames, dim)
            fit(model, loaders[0], loaders[1], epochs=tc["epochs"], lr=tc["lr"], patience=tc["patience"],
                optimizer="adamw", weight_decay=tc["weight_decay"], betas=tc["betas"],
                grad_clip=tc["grad_clip"], device=args.device, verbose=False)
            res = evaluate(model, loaders[1], loaders[2], args.device)
            print(f"seed {seed} run {r + 1}/{len(runs)}: " + "  ".join(f"{k} {v:.2f}" for k, v in res.items()))
            results.append(res)

    summary = {k: metrics.summarise([r[k] for r in results]) for k in results[0]}
    print(f"\n{args.model}: " + "  ".join(f"{k} {m:.2f} ± {s:.2f}" for k, (m, s) in summary.items()))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"model": args.model, "unseen_codecs": sorted(unseen),
                                          "config": cfg, "summary": summary, "runs": results}, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
