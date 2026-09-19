"""Train and evaluate ORBIT (or a baseline) under zero-shot cross-lingual protocols.

Examples:

    # smoke test on generated data, no downloads needed
    python -m papers.orbit.train --synthetic

    # leave-one-language-out on the four-language corpus
    python -m papers.orbit.train --manifest data/sadd/manifest.csv --protocol lolo

    # leave-two-languages-out: train on English + Chinese, test on Spanish and Greek
    python -m papers.orbit.train --manifest ... --protocol ltlo --train-languages english,chinese

    # ablations and baselines
    python -m papers.orbit.train --manifest ... --set model.cross_attention=false
    python -m papers.orbit.train --manifest ... --set model.use_grl=false
    python -m papers.orbit.train --manifest ... --set "model.geometries=[hyperbolic, euclidean]"
    python -m papers.orbit.train --manifest ... --model concat_cnn

The manifest needs ``subject_id``, ``audio`` and ``text`` (paths to (frames, dim)
and (tokens, dim) .npy files), ``label`` (1 = AD, 0 = healthy control) and
``language``. Test languages are never seen in training, not even their
unlabelled data. 10% of training speakers are held out for early stopping.
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
from common.data import FeatureDataset
from common.training import fit, move, set_seed
from papers.orbit.model import Orbit, OrbitConfig

HERE = Path(__file__).parent
LANGUAGES = ["english", "spanish", "chinese", "greek"]


def make_synthetic(out_dir, frames: dict, dims: dict, per_language: int = 24, per_subject: int = 2,
                   seed: int = 0) -> Path:
    """A shared AD signal in audio and text, plus a strong language-specific offset."""
    rng = np.random.default_rng(seed)
    out_dir = Path(out_dir)
    (out_dir / "features").mkdir(parents=True, exist_ok=True)
    ad_dir = {m: rng.normal(size=dims[m]) for m in dims}
    rows = []
    for lang in LANGUAGES:
        lang_offset = {m: rng.normal(size=dims[m]) * 2 for m in dims}
        for s in range(per_language):
            label = s % 2
            for k in range(per_subject):
                row = {"subject_id": f"{lang[:2]}{s:03d}", "label": label, "language": lang}
                for m in dims:
                    x = rng.normal(size=(frames[m], dims[m])) + lang_offset[m]
                    x += (0.7 if label else -0.7) * ad_dir[m]
                    name = f"features/{m}_{lang[:2]}{s:03d}_{k}.npy"
                    np.save(out_dir / name, x.astype(np.float32))
                    row[m] = name
                rows.append(row)
    path = out_dir / "manifest.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def protocol_runs(df: pd.DataFrame, protocol: str, train_languages: list[str]):
    """Yield (name, train_languages, test_languages)."""
    langs = sorted(df.language.unique())
    if protocol == "lolo":
        for held in langs:
            yield f"test={held}", [l for l in langs if l != held], [held]
    else:
        test = [l for l in langs if l not in train_languages]
        yield f"test={'+'.join(test)}", train_languages, test


def split_val(df, idx, frac: float, seed: int):
    subjects = np.unique(df.loc[idx, "subject_id"])
    rng = np.random.default_rng(seed)
    val_s = set(rng.choice(subjects, size=max(1, int(len(subjects) * frac)), replace=False))
    return ([i for i in idx if df.loc[i, "subject_id"] not in val_s],
            [i for i in idx if df.loc[i, "subject_id"] in val_s])


@torch.no_grad()
def evaluate(model, loader, device) -> dict:
    model.eval()
    preds, ys = [], []
    for batch in loader:
        batch = move(batch, device)
        out = model(batch)
        probs = out["p_vote"] if "p_vote" in out else out["logits_label"].softmax(-1)
        preds.append(probs.argmax(-1).cpu()); ys.append(batch["label"].cpu())
    pred, y = torch.cat(preds).numpy(), torch.cat(ys).numpy()
    return {"acc": metrics.accuracy(y, pred), "f1": metrics.macro_f1(y, pred)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=HERE / "configs" / "orbit.yaml")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="config overrides")
    ap.add_argument("--manifest")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--protocol", default="lolo", choices=["lolo", "ltlo"])
    ap.add_argument("--train-languages", default="english,chinese", help="for --protocol ltlo")
    ap.add_argument("--model", default="orbit", choices=["orbit", "concat_cnn", "concat_fcn"])
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--out", default="results/orbit.json")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    cfg = load_config(args.config, args.set)
    tc = cfg["training"]
    if args.epochs:
        tc["epochs"] = args.epochs
    frames = {m: v["frames"] for m, v in cfg["features"].items()}
    dims = {m: v["dim"] for m, v in cfg["features"].items()}
    if args.synthetic:
        manifest = make_synthetic("data/synthetic_orbit", frames, dims)
    elif args.manifest:
        manifest = Path(args.manifest)
    else:
        ap.error("pass --manifest or --synthetic")

    df = pd.read_csv(manifest)
    df["language"] = df["language"].astype(str)
    n_lang = df.language.nunique()
    labels = ["label", "language"]
    runs = []
    for name, train_langs, test_langs in protocol_runs(df, args.protocol, args.train_languages.split(",")):
        for seed in tc["seeds"]:
            set_seed(seed)
            tr_all = df.index[df.language.isin(train_langs)].tolist()
            tr, va = split_val(df, tr_all, tc["val_fraction"], seed)
            make = lambda idx, shuffle=False: DataLoader(FeatureDataset(manifest, frames, labels, idx),
                                                         batch_size=tc["batch_size"], shuffle=shuffle)
            if args.model == "orbit":
                model = Orbit(OrbitConfig(audio_dim=dims["audio"], text_dim=dims["text"],
                                          num_languages=n_lang, **cfg["model"]))
            else:
                model = MultiTaskProbe({m: (frames[m], dims[m]) for m in frames}, {"label": 2},
                                       head=args.model.split("_")[1])
            fit(model, make(tr, True), make(va), epochs=tc["epochs"], lr=tc["lr"], patience=tc["patience"],
                optimizer="adamw", weight_decay=tc["weight_decay"], device=args.device, verbose=False)
            # Scores are averaged over the held-out languages, as in the paper's LTLO protocol.
            per_lang = [evaluate(model, make(df.index[df.language == l].tolist()), args.device)
                        for l in test_langs]
            res = {k: float(np.mean([r[k] for r in per_lang])) for k in per_lang[0]}
            print(f"{name} seed {seed}: acc {res['acc']:.2f}  f1 {res['f1']:.2f}")
            runs.append({"run": name, "seed": seed, **res})

    summary = {k: metrics.summarise([r[k] for r in runs]) for k in ("acc", "f1")}
    print(f"\n{args.model} {args.protocol.upper()}: " +
          "  ".join(f"{k} {m:.2f} ± {s:.2f}" for k, (m, s) in summary.items()))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"model": args.model, "protocol": args.protocol, "config": cfg,
                                          "summary": summary, "runs": runs}, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
