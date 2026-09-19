"""Feature extraction from frozen speech and audio foundation models.

Usage (one .npy per input file, shape (frames, dim)):

    python -m common.features.speech --model wavlm \
        --inputs data/wavs.txt --out-dir data/features/wavlm

    # utterance-level vector (average pooling over frames), shape (dim,)
    python -m common.features.speech --model whisper --pool --inputs ... --out-dir ...

``--inputs`` is a text file with one audio path per line. Audio is loaded as
mono and resampled to 16 kHz. Models load from the Hugging Face Hub on first
use; x-vector and ECAPA need ``speechbrain``, TRILLsson needs ``tensorflow`` and
``tensorflow_hub``, and PaSST needs ``hear21passt`` (all optional dependencies).
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
    "voc2vec": "alkiskoudounas/voc2vec",          # non-verbal vocalisations (NOVA-ARC)
    "mhubert": "utter-project/mHuBERT-147",       # multilingual HuBERT (ORBIT)
    "mms": "facebook/mms-1b",                     # (NOVA-ARC, ORBIT)
    "xlsr": "facebook/wav2vec2-xls-r-1b",         # (ORBIT)
    "xlsr_300m": "facebook/wav2vec2-xls-r-300m",
    "unispeech_sat": "microsoft/unispeech-sat-base",
    "ast": "MIT/ast-finetuned-audioset-14-14-0.443",   # Audio Spectrogram Transformer (SNIFR)
}
TRILLSSON_URL = "https://tfhub.dev/google/trillsson3/1"
SPEECHBRAIN = {"xvector": "speechbrain/spkrec-xvect-voxceleb", "ecapa": "speechbrain/spkrec-ecapa-voxceleb"}


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
        elif name in SPEECHBRAIN:
            from speechbrain.inference.speaker import EncoderClassifier
            self.model = EncoderClassifier.from_hparams(source=SPEECHBRAIN[name],
                                                        run_opts={"device": str(self.device)})
        elif name == "passt":
            from hear21passt.base import get_basic_model
            self.model = get_basic_model(mode="embed_only").to(self.device).eval()
        elif name == "trillsson":
            import tensorflow_hub as hub
            self.model = hub.KerasLayer(TRILLSSON_URL)
        else:
            raise ValueError(f"Unknown model '{name}'. Choose from "
                             f"{sorted([*HF_MODELS, *SPEECHBRAIN, 'trillsson', 'passt'])}")

    @torch.no_grad()
    def __call__(self, wav: np.ndarray) -> np.ndarray:
        if self.name == "trillsson":  # utterance-level embedding
            return np.asarray(self.model(wav[None, :])["embedding"])
        if self.name == "passt":  # utterance-level embedding; PaSST expects 32 kHz audio
            import librosa
            wav32 = librosa.resample(wav, orig_sr=SAMPLE_RATE, target_sr=32_000)
            emb = self.model(torch.from_numpy(wav32)[None].to(self.device))
            return emb.reshape(1, -1).cpu().numpy()
        if self.name in SPEECHBRAIN:  # utterance-level embedding
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
    ap.add_argument("--pool", action="store_true", help="save the average over frames, shape (dim,)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    extractor = SpeechExtractor(args.model, args.device)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for line in Path(args.inputs).read_text().splitlines():
        if line.strip():
            feats = extractor(load_audio(line.strip()))
            np.save(out / f"{Path(line).stem}.npy", feats.mean(0) if args.pool else feats)


if __name__ == "__main__":
    main()
