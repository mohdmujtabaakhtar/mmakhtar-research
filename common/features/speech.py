"""Frame-level feature extraction from frozen speech foundation models.

Usage (one .npy per input file, shape (frames, dim)):

    python -m common.features.speech --model wavlm \
        --inputs data/wavs.txt --out-dir data/features/wavlm

``--inputs`` is a text file with one audio path per line. Audio is loaded as
mono and resampled to 16 kHz. Models load from the Hugging Face Hub on first
use; x-vector needs ``speechbrain`` and TRILLsson needs ``tensorflow`` and
``tensorflow_hub`` (both optional dependencies).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

SAMPLE_RATE = 16_000

HF_MODELS = {
    "wavlm": "microsoft/wavlm-base",
    "wav2vec2": "facebook/wav2vec2-base",
    "hubert": "facebook/hubert-base-ls960",
    "whisper": "openai/whisper-base",
}
TRILLSSON_URL = "https://tfhub.dev/google/trillsson3/1"
XVECTOR_SOURCE = "speechbrain/spkrec-xvect-voxceleb"


def load_audio(path: str | Path) -> np.ndarray:
    import librosa
    wav, _ = librosa.load(str(path), sr=SAMPLE_RATE, mono=True)
    if np.abs(wav).max() > 0:  # amplitude normalisation
        wav = wav / np.abs(wav).max()
    return wav.astype(np.float32)


class SpeechExtractor:
    """Wraps one frozen model and returns (frames, dim) features for a waveform."""

    def __init__(self, name: str, device: str = "cpu"):
        self.name, self.device = name, torch.device(device)
        if name in HF_MODELS:
            from transformers import AutoFeatureExtractor, AutoModel
            repo = HF_MODELS[name]
            self.processor = AutoFeatureExtractor.from_pretrained(repo)
            model = AutoModel.from_pretrained(repo)
            self.model = (model.encoder if name == "whisper" else model).to(self.device).eval()
        elif name == "xvector":
            from speechbrain.inference.speaker import EncoderClassifier
            self.model = EncoderClassifier.from_hparams(source=XVECTOR_SOURCE,
                                                        run_opts={"device": str(self.device)})
        elif name == "trillsson":
            import tensorflow_hub as hub
            self.model = hub.KerasLayer(TRILLSSON_URL)
        else:
            raise ValueError(f"Unknown model '{name}'. Choose from "
                             f"{sorted([*HF_MODELS, 'xvector', 'trillsson'])}")

    @torch.no_grad()
    def __call__(self, wav: np.ndarray) -> np.ndarray:
        if self.name == "trillsson":  # utterance-level embedding
            return np.asarray(self.model(wav[None, :])["embedding"])
        if self.name == "xvector":  # utterance-level embedding
            emb = self.model.encode_batch(torch.from_numpy(wav)[None].to(self.device))
            return emb.squeeze(0).cpu().numpy()
        inputs = self.processor(wav, sampling_rate=SAMPLE_RATE, return_tensors="pt")
        key = "input_features" if self.name == "whisper" else "input_values"
        hidden = self.model(inputs[key].to(self.device)).last_hidden_state
        if self.name == "whisper":  # Whisper pads to 30 s; keep only the real frames
            hidden = hidden[:, : max(1, int(len(wav) / SAMPLE_RATE * 50))]
        return hidden.squeeze(0).cpu().numpy()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--inputs", required=True, help="text file with one audio path per line")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    extractor = SpeechExtractor(args.model, args.device)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for line in Path(args.inputs).read_text().splitlines():
        if line.strip():
            np.save(out / f"{Path(line).stem}.npy", extractor(load_audio(line.strip())))


if __name__ == "__main__":
    main()
