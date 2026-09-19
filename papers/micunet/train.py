"""Train and evaluate MiCuNet for emotion-manipulation and generator traceback.

Examples:

    python -m papers.micunet.train --synthetic                                   # E -> E and E -> C
    python -m papers.micunet.train --synthetic --set "model.spaces=[euclidean]"  # only Euclidean
    python -m papers.micunet.train --synthetic --set model.gating=uniform        # no learnable gating
    python -m papers.micunet.train --synthetic --model concat                    # concatenation fusion
    python -m papers.micunet.train --synthetic --unseen-sources evc5,evc6        # unseen EVC models
    python -m papers.micunet.train --manifest data/emofake/manifest.csv --train-language english \\
        --test-language chinese

Protocol (paper): EmoFake, English and Chinese subsets, released train / dev / test
splits; Adam, lr 1e-3 with decay, batch 32, 50 epochs, dropout, early stopping on
dev. Cross-lingual runs train on one language and test on the other. Metrics per
task (OE, CE, M): accuracy, macro-F1 and one-vs-rest EER.

Manifest columns: ``subject_id``, ``ptm`` (pooled PTM embedding .npy), ``spec``
(spectrogram .npy, (frames, filters)), ``oe``, ``ce``, ``source``, ``split`` and
``language``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from common.config import load_config
from common.data import FeatureDataset
from common.runner import classification_scores, predict
from common.training import fit, set_seed
from papers.micunet.model import TASKS, MiCuNet, MiCuNetConfig

HERE = Path(__file__).parent
EMOTIONS = ["neutral", "happy", "angry", "sad", "surprise"]


def make_synthetic(out_dir, ptm_dim: int, frames: int, n_filters: int, n_sources: int = 7,
                   per_language: int = 280, seed: int = 0) -> Path:
    """OE lives mostly in the spectrogram, CE and the source mostly in the PTM embedding;
    the two languages share the class structure but differ by a feature offset."""
    rng = np.random.default_rng(seed)
    out_dir = Path(out_dir)
    (out_dir / "features").mkdir(parents=True, exist_ok=True)
    oe_spec = rng.normal(size=(5, n_filters))
    ce_ptm, src_ptm = rng.normal(size=(5, ptm_dim)), rng.normal(size=(n_sources, ptm_dim))
    rows = []
    for lang in ("english", "chinese"):
        shift = rng.normal(size=ptm_dim) * 0.3
        for k in range(per_language):
            oe, ce, src = int(rng.integers(5)), int(rng.integers(5)), int(rng.integers(n_sources))
            ptm = rng.normal(size=ptm_dim) + 0.7 * ce_ptm[ce] + 0.7 * src_ptm[src] + shift
            spec = rng.normal(size=(frames, n_filters)) + 0.5 * oe_spec[oe] + 0.2 * ce_ptm[ce, :n_filters]
            split = "train" if k % 10 < 6 else ("dev" if k % 10 < 8 else "test")
            names = {m: f"features/{m}_{lang}_{k}.npy" for m in ("ptm", "spec")}
            np.save(out_dir / names["ptm"], ptm.astype(np.float32))
            np.save(out_dir / names["spec"], spec.astype(np.float32))
            rows.append({"subject_id": f"{lang}_{k % 10}", **names, "oe": EMOTIONS[oe], "ce": EMOTIONS[ce],
                         "source": f"evc{src}", "split": split, "language": lang})
    path = out_dir / "manifest.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=HERE / "configs" / "micunet.yaml")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="config overrides")
    ap.add_argument("--manifest")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--model", default="micunet", choices=["micunet", "concat"])
    ap.add_argument("--train-language", default="english")
    ap.add_argument("--test-language", help="default: both languages")
    ap.add_argument("--unseen-sources", help="comma-separated sources held out of training (OE/CE on them only)")
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--out")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    cfg = load_config(args.config, args.set)
    tc, fc = cfg["training"], cfg["features"]
    if args.epochs:
        tc["epochs"] = args.epochs
    if args.synthetic:
        fc.update(cfg["synthetic"])
        manifest = make_synthetic("data/synthetic_micunet", fc["ptm_dim"], fc["spec_frames"], fc["spec_filters"])
    elif args.manifest:
        manifest = Path(args.manifest)
    else:
        ap.error("pass --manifest or --synthetic")

    df = pd.read_csv(manifest)
    lang, split = df["language"].astype(str).str.lower(), df["split"].astype(str).str.lower()
    unseen = set((args.unseen_sources or "").split(",")) - {""}
    is_unseen = df["source"].astype(str).isin(unseen)
    pick = lambda l, s, keep: list(np.where((lang == l) & (split == s) & keep)[0])   # noqa: E731
    tr, va = pick(args.train_language, "train", ~is_unseen), pick(args.train_language, "dev", ~is_unseen)
    tests = [args.test_language] if args.test_language else sorted(lang.unique(), key=lambda l: l != args.train_language)
    n_classes = {t: df[t].nunique() for t in TASKS}

    set_seed(0)
    streams = {"ptm": 1, "spec": fc["spec_frames"]}
    loader = lambda ids, shuffle=False: DataLoader(   # noqa: E731
        FeatureDataset(manifest, streams, list(TASKS), ids), batch_size=tc["batch_size"], shuffle=shuffle)
    model = MiCuNet(MiCuNetConfig(ptm_dim=fc["ptm_dim"], num_classes=n_classes,
                                  **{**cfg["model"], "fusion": args.model}))
    fit(model, loader(tr, True), loader(va), epochs=tc["epochs"], lr=tc["lr"], patience=tc.get("patience", 10),
        device=args.device, verbose=False)

    results = {}
    code = {l[0].upper() for l in [args.train_language]}.pop()
    for test_lang in tests:
        te = pick(test_lang, "test", is_unseen if unseen else np.ones(len(df), bool))
        name = f"{code}->{test_lang[0].upper()}" + (" (unseen sources)" if unseen else "")
        results[name] = {}
        for task in TASKS:
            if unseen and task == "source":
                continue                      # unseen sources have no training class
            y, p = predict(model, loader(te), args.device, label_key=task, logits_key=f"logits_{task}")
            results[name][task] = classification_scores(y, p)
    for name, res in results.items():
        print(name + ": " + "   ".join(f"{t.upper()} acc {r['acc']:.2f} f1 {r['f1']:.2f} eer {r['eer']:.2f}"
                                       for t, r in res.items()))
    out = args.out or ("results/micunet.json" if args.model == "micunet" else f"results/micunet_{args.model}.json")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps({"model": args.model, "config": cfg, "results": results}, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
