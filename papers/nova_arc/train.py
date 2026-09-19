"""Unsupervised non-verbal -> verbal adaptation with NOVA-ARC.

Examples:

    # smoke test on generated data, no downloads needed
    python -m papers.nova_arc.train --synthetic

    # ASVP-ESD non-verbal (labelled source) -> RAVDESS speech (unlabelled target)
    python -m papers.nova_arc.train --source data/asvp_nv/manifest.csv --target data/ravdess/manifest.csv

    # Table 3 / Table 4 variants on the same data
    python -m papers.nova_arc.train ... --set model.geometry=euclidean
    python -m papers.nova_arc.train ... --set model.use_hel=false
    python -m papers.nova_arc.train ... --set model.adaptation=none      # source-only baseline

Both manifests need ``subject_id``, ``features`` (path to a (frames, dim) .npy) and
``label`` using the shared five-class space 0 happy, 1 anger, 2 disgust,
3 sadness, 4 fear. Target labels are **never used in training**, only to score the
adapted model at the end.
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from common import metrics
from common.config import load_config
from common.data import FeatureDataset
from common.training import make_optimizer, move, set_seed, warmup_cosine
from papers.nova_arc.model import NovaArc, NovaArcConfig

HERE = Path(__file__).parent
EMOTIONS = ["happy", "anger", "disgust", "sadness", "fear"]


def make_synthetic(out_dir, frames: int, dim: int, per_class: int = 40, seed: int = 0):
    """Source ('non-verbal') and target ('verbal') share the emotion structure, but the
    target is shifted, partly rotated and has weaker intensity, so a source-only model
    transfers poorly."""
    rng = np.random.default_rng(seed)
    out_dir = Path(out_dir)
    (out_dir / "features").mkdir(parents=True, exist_ok=True)
    protos = rng.normal(size=(len(EMOTIONS), dim)) * 1.2
    rotation, _ = np.linalg.qr(rng.normal(size=(dim, dim)))
    mix = 0.5 * np.eye(dim) + 0.5 * rotation
    shift = rng.normal(size=dim) * 1.5
    paths = {}
    for domain in ("source", "target"):
        rows = []
        for c in range(len(EMOTIONS)):
            for k in range(per_class):
                centre = protos[c] if domain == "source" else 0.6 * protos[c] @ mix + shift
                x = rng.normal(size=(frames, dim)) + centre
                name = f"features/{domain}_{c}_{k}.npy"
                np.save(out_dir / name, x.astype(np.float32))
                rows.append({"subject_id": f"{domain}{k % 10}", "features": name, "label": c})
        paths[domain] = out_dir / f"{domain}.csv"
        pd.DataFrame(rows).to_csv(paths[domain], index=False)
    return paths["source"], paths["target"]


@torch.no_grad()
def embed_all(model, loader, device):
    model.eval()
    embs, labels, preds = [], [], []
    for batch in loader:
        batch = move(batch, device)
        enc = model.encode(batch["features"])
        embs.append(model.ot_embedding(enc)); labels.append(batch["label"]); preds.append(enc["logits"].argmax(-1))
    return torch.cat(embs), torch.cat(labels), torch.cat(preds)


def train_one(cfg: dict, source_csv, target_csv, frames: int, dim: int, device, seed: int, verbose=True):
    tc = cfg["training"]
    set_seed(seed)
    ds_src = FeatureDataset(source_csv, {"features": frames}, ["label"])
    ds_tgt = FeatureDataset(target_csv, {"features": frames}, ["label"])
    src_loader = DataLoader(ds_src, batch_size=tc["batch_size"], shuffle=True, drop_last=True)
    tgt_loader = DataLoader(ds_tgt, batch_size=tc["batch_size"], shuffle=True, drop_last=True)
    src_eval = DataLoader(ds_src, batch_size=64)
    tgt_eval = DataLoader(ds_tgt, batch_size=64)

    model = NovaArc(NovaArcConfig(input_dim=dim, **cfg["model"])).to(device)
    optim = make_optimizer(model.parameters(), "adamw", tc["lr"], tc["weight_decay"], tc["betas"])
    steps_per_epoch = max(len(src_loader), len(tgt_loader))
    sched = warmup_cosine(optim, tc["epochs"] * steps_per_epoch, tc["warmup_frac"])
    for epoch in range(1, tc["epochs"] + 1):
        model.refresh_prototypes(*embed_all(model, src_eval, device)[:2])   # once per epoch
        model.train()
        src_iter = itertools.cycle(src_loader) if len(src_loader) < steps_per_epoch else iter(src_loader)
        tgt_iter = itertools.cycle(tgt_loader) if len(tgt_loader) < steps_per_epoch else iter(tgt_loader)
        for _ in range(steps_per_epoch):
            s, t = next(src_iter), next(tgt_iter)
            batch = move({"source": s["features"], "label": s["label"], "target": t["features"]}, device)
            loss, parts = model.loss(model(batch), batch)
            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), tc["grad_clip"])
            optim.step()
            sched.step()
        if verbose and (epoch % 5 == 0 or epoch == tc["epochs"]):
            print(f"  epoch {epoch:3d}  " + "  ".join(f"{k} {v:.3f}" for k, v in parts.items()))
    _, y, pred = embed_all(model, tgt_eval, device)
    y, pred = y.cpu().numpy(), pred.cpu().numpy()
    return model, {"acc": metrics.accuracy(y, pred), "f1": metrics.macro_f1(y, pred)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=HERE / "configs" / "nova_arc.yaml")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="config overrides")
    ap.add_argument("--source", help="manifest of labelled non-verbal source data")
    ap.add_argument("--target", help="manifest of verbal target data (labels only used for scoring)")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--out", default="results/nova_arc.json")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    cfg = load_config(args.config, args.set)
    if args.epochs:
        cfg["training"]["epochs"] = args.epochs
    frames, dim = cfg["features"]["frames"], cfg["features"]["dim"]
    if args.synthetic:
        source, target = make_synthetic("data/synthetic_nova_arc", frames, dim)
    elif args.source and args.target:
        source, target = Path(args.source), Path(args.target)
    else:
        ap.error("pass --source and --target, or --synthetic")

    runs = []
    for seed in cfg["training"]["seeds"]:
        _, res = train_one(cfg, source, target, frames, dim, args.device, seed)
        print(f"seed {seed}: target acc {res['acc']:.2f}  f1 {res['f1']:.2f}")
        runs.append(res)
    summary = {k: metrics.summarise([r[k] for r in runs]) for k in ("acc", "f1")}
    m = cfg["model"]
    label = f"NOVA-ARC ({m['geometry']}, adaptation={m['adaptation']})"
    print(f"\n{label}: " + "  ".join(f"{k} {mu:.2f} ± {sd:.2f}" for k, (mu, sd) in summary.items()))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"model": label, "config": cfg, "summary": summary, "runs": runs}, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
