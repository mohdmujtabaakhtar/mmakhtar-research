"""Train and evaluate SATYAM for codec-fake speech detection.

Examples:

    python -m papers.satyam.train --synthetic                    # tiny offline LM, generated data
    python -m papers.satyam.train --synthetic --model concat     # (C)  Euclidean concatenation
    python -m papers.satyam.train --synthetic --model ma         # (MA) hyperbolic Mobius fusion, no BD
    python -m papers.satyam.train --synthetic --model e_bd       # (E-BD) BD alignment in Euclidean space
    python -m papers.satyam.train --synthetic --model h_bd_ss    # hyperbolic BD on speech-speech only
    python -m papers.satyam.train --synthetic --model h_bd_st    # hyperbolic BD on speech-prompt only
    python -m papers.satyam.train --manifest data/icf/manifest.csv                        # Qwen2-7B
    python -m papers.satyam.train --manifest data/icf/manifest.csv --set lm.backbone=Qwen/Qwen-1_8B

Protocol (paper): Indic-CodecFake (ICF) official train / validation / test splits;
only the CNN, projection, gating and hyperbolic alignment modules are trained
(the Qwen2 decoder stays frozen). AdamW, lr 1e-4, batch 32, 5 epochs. Metrics:
accuracy and EER (score = log p("Fake") - log p("Real")).

Manifest columns: ``subject_id`` (speaker), ``whisper`` and ``trillsson`` (pooled
embeddings), ``label`` (0 = real, 1 = codec fake) and ``split``. For seen / unseen
codec evaluation, put the held-out codecs only in the test split.
"""
from __future__ import annotations

from pathlib import Path

from common.runner import fusion_cli
from papers.satyam.model import Satyam, SatyamConfig

HERE = Path(__file__).parent
VARIANTS = {
    "satyam": {},
    "concat": {"fusion": "concat", "bd_ss": False, "bd_st": False},
    "ma": {"bd_ss": False, "bd_st": False},
    "e_bd": {"geometry": "euclidean"},
    "h_bd_ss": {"bd_st": False},
    "h_bd_st": {"bd_ss": False},
}


def build(name: str, cfg: dict, stream: str):
    dims = {s: v["dim"] for s, v in cfg["features"].items()}
    kw = {**cfg["model"], **VARIANTS[name], "lm": cfg["lm"]}
    return (lambda: Satyam(SatyamConfig(input_dims=dims, **kw))), list(dims)


if __name__ == "__main__":
    fusion_cli(__doc__, "satyam", HERE / "configs" / "satyam.yaml", build, list(VARIANTS), "satyam")
