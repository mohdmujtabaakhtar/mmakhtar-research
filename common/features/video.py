"""Clip-level feature extraction from frozen video foundation models.

Usage (one .npy per input video, shape (clips, dim)):

    python -m common.features.video --model videomae \
        --inputs data/videos.txt --out-dir data/features/videomae

Each video is split into consecutive 16-frame clips resized to 224x224; each
clip gives one embedding (mean of the encoder's patch tokens), so the output
is a sequence over clips. Needs ``decord`` (or ``torchvision``) to read video.

DeepSeek-VL2, OpenFace kinematics and the ResNet18+TANN temporal features used
in DIVINE depend on their own toolkits and are not wrapped here; save their
outputs as (frames, dim) ``.npy`` arrays and they plug into the same manifest.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

HF_MODELS = {
    "videomae": "MCG-NJU/videomae-base",
    "videomae_v2": "OpenGVLab/VideoMAEv2-Base",
    "vivit": "google/vivit-b-16x2-kinetics400",
}
CLIP_LEN = {"videomae": 16, "videomae_v2": 16, "vivit": 32}


def load_frames(path: str | Path) -> np.ndarray:
    """Return all frames as a (T, H, W, 3) uint8 array."""
    try:
        from decord import VideoReader
        vr = VideoReader(str(path))
        return vr.get_batch(range(len(vr))).asnumpy()
    except ImportError:
        from torchvision.io import read_video
        frames, _, _ = read_video(str(path), pts_unit="sec", output_format="THWC")
        return frames.numpy()


class VideoExtractor:
    def __init__(self, name: str, device: str = "cpu"):
        if name not in HF_MODELS:
            raise ValueError(f"Unknown model '{name}'. Choose from {sorted(HF_MODELS)}")
        from transformers import AutoImageProcessor, AutoModel
        self.name, self.device = name, torch.device(device)
        repo = HF_MODELS[name]
        remote = name == "videomae_v2"  # VideoMAE V2 ships custom modelling code
        self.processor = AutoImageProcessor.from_pretrained(repo, trust_remote_code=remote)
        self.model = AutoModel.from_pretrained(repo, trust_remote_code=remote).to(self.device).eval()

    @torch.no_grad()
    def __call__(self, frames: np.ndarray) -> np.ndarray:
        n = CLIP_LEN[self.name]
        if len(frames) < n:  # repeat the last frame for very short videos
            frames = np.concatenate([frames, np.repeat(frames[-1:], n - len(frames), 0)])
        feats = []
        for start in range(0, len(frames) - n + 1, n):
            inputs = self.processor(list(frames[start:start + n]), return_tensors="pt")
            out = self.model(**{k: v.to(self.device) for k, v in inputs.items()})
            hidden = out.last_hidden_state if hasattr(out, "last_hidden_state") else out
            feats.append(hidden.mean(dim=1).squeeze(0).cpu().numpy())
        return np.stack(feats)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, choices=sorted(HF_MODELS))
    ap.add_argument("--inputs", required=True, help="text file with one video path per line")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    extractor = VideoExtractor(args.model, args.device)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for line in Path(args.inputs).read_text().splitlines():
        if line.strip():
            np.save(out / f"{Path(line).stem}.npy", extractor(load_frames(line.strip())))


if __name__ == "__main__":
    main()
