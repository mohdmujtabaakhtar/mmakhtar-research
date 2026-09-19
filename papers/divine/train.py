"""Train and evaluate DIVINE (or a baseline) with subject-wise 5-fold cross-validation.

Examples:

    # smoke test on generated data, no downloads needed
    python -m papers.divine.train --synthetic

    # real run on pre-extracted Toronto NeuroFace features
    python -m papers.divine.train --manifest data/tnf/manifest.csv

    # concatenation baseline with the same splits
    python -m papers.divine.train --manifest data/tnf/manifest.csv --model concat_cnn

Every run is evaluated in the paper's three test conditions: audio + video,
video only and audio only. Results (mean and std over folds and seeds) are
printed and written to ``--out`` as JSON.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from common import metrics
from common.baselines import MultiTaskProbe
from common.data import FeatureDataset, make_synthetic_manifest, subject_kfold
from common.training import fit, move, set_seed
from papers.divine.model import Divine, DivineConfig

HERE = Path(__file__).parent
CONDITIONS = {"audio+video": ("video", "audio"), "video_only": ("video",), "audio_only": ("audio",)}


@torch.no_grad()
def predict(model, loader, device, present: tuple[str, ...]):
    """Predictions with absent modalities zeroed out and masked."""
    model.eval()
    ys = {"diagnosis": [], "severity": []}
    ps = {"diagnosis": [], "severity": [], "severity_expected": []}
    for batch in loader:
        batch = move(batch, device)
        bsz = batch["video"].shape[0]
        for m in ("video", "audio"):
            keep = torch.full((bsz,), float(m in present), device=device)
            batch[f"mask_{m}"] = keep
            batch[m] = batch[m] * keep[:, None, None]
        out = model(batch)
        for task in ("diagnosis", "severity"):
            ys[task].append(batch[task].cpu())
            ps[task].append(out[f"logits_{task}"].argmax(-1).cpu())
        probs = out["logits_severity"].softmax(-1)
        levels = torch.arange(probs.shape[-1], device=device, dtype=probs.dtype)
        ps["severity_expected"].append((probs * levels).sum(-1).cpu())
    y = {k: torch.cat(v).numpy() for k, v in ys.items()}
    p = {k: torch.cat(v).numpy() for k, v in ps.items()}
    return {
        "acc": metrics.accuracy(y["diagnosis"], p["diagnosis"]),
        "f1": metrics.macro_f1(y["diagnosis"], p["diagnosis"]),
        "sev_acc": metrics.accuracy(y["severity"], p["severity"]),
        "sev_mae": metrics.mae(y["severity"], p["severity_expected"]),
        "sev_rmse": metrics.rmse(y["severity"], p["severity_expected"]),
    }


def build_model(name: str, cfg: dict, dims: dict, frames: dict):
    if name == "divine":
        return Divine(DivineConfig(input_dims=dims, **cfg["model"]))
    head = name.split("_")[1]  # concat_cnn / concat_fcn
    modalities = {m: (frames[m], dims[m]) for m in dims}
    tasks = {"diagnosis": cfg["model"]["num_diagnosis"], "severity": cfg["model"]["num_severity"]}
    return MultiTaskProbe(modalities, tasks, head=head,
                          task_weights={"diagnosis": 1.0, "severity": cfg["model"]["alpha"]})


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=HERE / "configs" / "divine.yaml")
    ap.add_argument("--manifest", help="CSV manifest of pre-extracted features")
    ap.add_argument("--synthetic", action="store_true", help="generate and use a small synthetic dataset")
    ap.add_argument("--model", default="divine", choices=["divine", "concat_cnn", "concat_fcn"])
    ap.add_argument("--epochs", type=int, help="override the config")
    ap.add_argument("--out", default="results/divine.json")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    train_cfg = cfg["training"]
    if args.epochs:
        train_cfg["epochs"] = args.epochs

    frames = {m: spec["frames"] for m, spec in cfg["features"].items()}
    dims = {m: spec["dim"] for m, spec in cfg["features"].items()}
    if args.synthetic:
        manifest = make_synthetic_manifest("data/synthetic_divine",
                                           {m: (frames[m], dims[m]) for m in frames})
    elif args.manifest:
        manifest = Path(args.manifest)
    else:
        ap.error("pass --manifest or --synthetic")

    import pandas as pd
    subjects = pd.read_csv(manifest)["subject_id"].to_numpy()
    labels = ["diagnosis", "severity"]
    results = {c: [] for c in CONDITIONS}

    for seed in train_cfg["seeds"]:
        splits = subject_kfold(subjects, k=train_cfg["folds"], seed=seed)
        for fold, (tr, va, te) in enumerate(splits):
            print(f"seed {seed}  fold {fold + 1}/{train_cfg['folds']}  "
                  f"train {len(tr)}  val {len(va)}  test {len(te)}")
            set_seed(seed * 100 + fold)
            loaders = [
                DataLoader(FeatureDataset(manifest, frames, labels, idx),
                           batch_size=train_cfg["batch_size"], shuffle=(i == 0))
                for i, idx in enumerate((tr, va, te))
            ]
            model = build_model(args.model, cfg, dims, frames)
            fit(model, loaders[0], loaders[1], epochs=train_cfg["epochs"], lr=train_cfg["lr"],
                patience=train_cfg["patience"], device=args.device, verbose=False)
            for cond, present in CONDITIONS.items():
                results[cond].append(predict(model, loaders[2], args.device, present))

    summary = {}
    print(f"\n{args.model} - mean ± std over {len(results['audio+video'])} runs")
    print(f"{'condition':<13}{'Acc':>14}{'F1':>14}{'Sev MAE':>14}{'Sev RMSE':>14}")
    for cond, runs in results.items():
        summary[cond] = {k: metrics.summarise([r[k] for r in runs]) for k in runs[0]}
        s = summary[cond]
        print(f"{cond:<13}" + "".join(f"{s[k][0]:>8.2f} ±{s[k][1]:>5.2f}"
                                       for k in ("acc", "f1", "sev_mae", "sev_rmse")))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"model": args.model, "config": cfg, "summary": summary,
                                          "runs": results}, indent=2, default=float))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
