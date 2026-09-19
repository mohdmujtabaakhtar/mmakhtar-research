"""Train and evaluate GARUDA / GARUDA-FT for codec-fake speech detection.

Examples:

    python -m papers.garuda.train --synthetic                        # tiny offline LM, generated data
    python -m papers.garuda.train --synthetic --model concat         # concatenation without JS
    python -m papers.garuda.train --synthetic --model kl             # KL instead of JS alignment
    python -m papers.garuda.train --synthetic --model single --stream whisper   # "Only Wh"
    python -m papers.garuda.train --manifest data/sea_cf/manifest.csv                        # GARUDA
    python -m papers.garuda.train --manifest data/sea_cf/manifest.csv --config papers/garuda/configs/garuda_ft.yaml

Protocol (paper): trained on the combined training splits of CodecFake (English,
Chinese) and SEA-CF (Tamil, Hindi, Thai, Indonesian, Malay, Vietnamese); model
selection on the validation splits; results on the test splits. GARUDA trains only
the projection module (5 epochs, batch 32, lr 1e-4); GARUDA-FT also trains LoRA
adapters (rank 8, alpha 32, query / value projections; 3 epochs, lr 1e-5). AdamW,
dropout. Metrics: accuracy and EER (score = log p("fake") - log p("real")).

Manifest columns: ``subject_id`` (speaker), ``whisper`` and ``xvector`` (pooled
embeddings), ``label`` (0 = real, 1 = codec fake) and ``split``.
"""
from __future__ import annotations

from pathlib import Path

from common.runner import fusion_cli
from papers.garuda.model import Garuda, GarudaConfig

HERE = Path(__file__).parent
VARIANTS = {"garuda": {}, "concat": {"fusion": "concat"}, "kl": {"align": "kl"}, "single": {}}


def build(name: str, cfg: dict, stream: str):
    dims = {s: v["dim"] for s, v in cfg["features"].items()}
    if name == "single":
        dims = {stream if stream in dims else next(iter(dims)): dims.get(stream, next(iter(dims.values())))}
    kw = {**cfg["model"], **VARIANTS[name], "lm": cfg["lm"]}
    return (lambda: Garuda(GarudaConfig(input_dims=dims, **kw))), list(dims)


if __name__ == "__main__":
    fusion_cli(__doc__, "garuda", HERE / "configs" / "garuda.yaml", build, list(VARIANTS), "garuda")
