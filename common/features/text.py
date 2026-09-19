"""Token-level feature extraction from frozen multilingual text encoders.

Usage (one .npy per transcript, shape (tokens, dim)):

    python -m common.features.text --model mbert --inputs transcripts.tsv --out-dir data/text/mbert

``--inputs`` is a TSV file with ``id<TAB>transcript`` per line. The output keeps
the per-token hidden states (padding and special tokens removed), so downstream
models can pool them however they need (ORBIT uses attention pooling).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

HF_MODELS = {
    "mbert": "bert-base-multilingual-cased",
    "xlmr": "xlm-roberta-large",
    "e5": "intfloat/multilingual-e5-large",
    "qwen3": "Qwen/Qwen3-Embedding-0.6B",
}


class TextExtractor:
    def __init__(self, name: str, device: str = "cpu", max_length: int = 512):
        if name not in HF_MODELS:
            raise ValueError(f"Unknown model '{name}'. Choose from {sorted(HF_MODELS)}")
        from transformers import AutoModel, AutoTokenizer
        self.name, self.device, self.max_length = name, torch.device(device), max_length
        self.tokenizer = AutoTokenizer.from_pretrained(HF_MODELS[name])
        self.model = AutoModel.from_pretrained(HF_MODELS[name]).to(self.device).eval()

    @torch.no_grad()
    def __call__(self, text: str) -> np.ndarray:
        if self.name == "e5":
            text = "query: " + text  # E5 expects a prefix
        enc = self.tokenizer(text, return_tensors="pt", truncation=True,
                             max_length=self.max_length, return_special_tokens_mask=True)
        special = enc.pop("special_tokens_mask")[0].bool()
        hidden = self.model(**{k: v.to(self.device) for k, v in enc.items()}).last_hidden_state[0]
        keep = ~special.to(self.device)
        return hidden[keep if keep.any() else slice(None)].cpu().numpy()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, choices=sorted(HF_MODELS))
    ap.add_argument("--inputs", required=True, help="TSV file: id<TAB>transcript")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    extractor = TextExtractor(args.model, args.device)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for line in Path(args.inputs).read_text(encoding="utf-8").splitlines():
        if "\t" in line:
            uid, text = line.split("\t", 1)
            np.save(out / f"{uid}.npy", extractor(text))


if __name__ == "__main__":
    main()
