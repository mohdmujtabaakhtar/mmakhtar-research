"""Hand-crafted cepstral features: MFCC and LFCC frame sequences.

Usage (one .npy per input file, shape (frames, n_ceps)):

    python -m common.features.spectral --kind mfcc --inputs coughs.txt --out-dir data/mfcc

Both use 40 cepstral coefficients over 25 ms windows with a 10 ms hop at 16 kHz.
MFCCs use a mel filterbank (librosa); LFCCs use a linearly spaced triangular
filterbank followed by the same log + DCT steps. For utterance-level baselines,
``stats_pool`` gives the mean and standard deviation over time.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.fft import dct

SR, N_FFT, HOP, N_CEPS, N_FILTERS = 16_000, 400, 160, 40, 70


def mfcc(wav: np.ndarray, sr: int = SR, n_ceps: int = N_CEPS) -> np.ndarray:
    import librosa
    feats = librosa.feature.mfcc(y=wav, sr=sr, n_mfcc=n_ceps, n_fft=N_FFT, hop_length=HOP)
    return feats.T.astype(np.float32)


def linear_filterbank(sr: int = SR, n_fft: int = N_FFT, n_filters: int = N_FILTERS) -> np.ndarray:
    """Triangular filters with centres spaced linearly in Hz, shape (n_filters, n_fft // 2 + 1)."""
    freqs = np.linspace(0, sr / 2, n_fft // 2 + 1)
    edges = np.linspace(0, sr / 2, n_filters + 2)
    bank = np.zeros((n_filters, len(freqs)))
    for i in range(n_filters):
        lo, mid, hi = edges[i], edges[i + 1], edges[i + 2]
        bank[i] = np.clip(np.minimum((freqs - lo) / (mid - lo), (hi - freqs) / (hi - mid)), 0, None)
    return bank


def lfcc(wav: np.ndarray, sr: int = SR, n_ceps: int = N_CEPS) -> np.ndarray:
    import librosa
    power = np.abs(librosa.stft(wav, n_fft=N_FFT, hop_length=HOP)) ** 2   # (freq, frames)
    energies = linear_filterbank(sr) @ power
    ceps = dct(np.log(energies + 1e-10), type=2, axis=0, norm="ortho")[:n_ceps]
    return ceps.T.astype(np.float32)


def stats_pool(frames: np.ndarray) -> np.ndarray:
    """Mean and standard deviation over time: (frames, d) -> (2d,)."""
    return np.concatenate([frames.mean(0), frames.std(0)]).astype(np.float32)


def main() -> None:
    from common.features.speech import load_audio
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kind", required=True, choices=["mfcc", "lfcc"])
    ap.add_argument("--inputs", required=True, help="text file with one audio path per line")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--pooled", action="store_true", help="save mean+std vectors instead of frames")
    args = ap.parse_args()

    fn = mfcc if args.kind == "mfcc" else lfcc
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for line in Path(args.inputs).read_text().splitlines():
        if line.strip():
            feats = fn(load_audio(line.strip()))
            np.save(out / f"{Path(line).stem}.npy", stats_pool(feats) if args.pooled else feats)


if __name__ == "__main__":
    main()
