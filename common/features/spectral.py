"""Hand-crafted features: MFCC / LFCC frame sequences and filterbank spectrograms.

Usage (one .npy per input file):

    # cepstra, shape (frames, n_ceps)
    python -m common.features.spectral --kind mfcc --inputs coughs.txt --out-dir data/mfcc

    # log filterbank spectrograms, shape (frames, n_filters): transform x filterbank
    python -m common.features.spectral --kind spectrogram --transform stft --filterbank mel \
        --inputs wavs.txt --out-dir data/s_mel

Spectrograms combine a time-frequency transform (``stft``, ``cqt`` or a Morlet
``cwt``) with an auditory filterbank (``mel``, ``gammatone`` or ``linear``) applied to
the transform's bin frequencies, as in MiCuNet (S-ME = STFT + mel, C-GT = CQT +
gammatone, W-LI = wavelet + linear, ...).

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


# ----------------------------------------------------------------- filterbank spectrograms

def _hz_to_mel(f):
    return 2595.0 * np.log10(1.0 + np.asarray(f, float) / 700.0)


def _mel_to_hz(m):
    return 700.0 * (10 ** (np.asarray(m, float) / 2595.0) - 1.0)


def _erb(f):
    return 24.7 * (4.37e-3 * np.asarray(f, float) + 1.0)


def _erb_space(lo, hi, n):
    """n centre frequencies equally spaced on the ERB-rate scale."""
    rate = lambda f: 21.4 * np.log10(4.37e-3 * f + 1.0)      # noqa: E731
    inv = lambda r: (10 ** (r / 21.4) - 1.0) / 4.37e-3       # noqa: E731
    return inv(np.linspace(rate(lo), rate(hi), n))


def filterbank_on_bins(kind: str, bin_freqs: np.ndarray, n_filters: int = 64,
                       fmin: float = 20.0, fmax: float = SR / 2) -> np.ndarray:
    """Filter weights (n_filters, n_bins) evaluated at arbitrary bin centre frequencies.

    ``mel`` and ``linear`` are triangular filters with centres equally spaced on the
    mel / Hz scale; ``gammatone`` uses 4th-order gammatone magnitude responses
    |H(f)| = (1 + ((f - fc) / b)^2)^-2 with b = 1.019 ERB(fc), centres ERB-spaced.
    """
    f = np.asarray(bin_freqs, float)
    if kind == "gammatone":
        fc = _erb_space(fmin, fmax, n_filters)
        b = 1.019 * _erb(fc)
        bank = (1.0 + ((f[None] - fc[:, None]) / b[:, None]) ** 2) ** -2.0
        return bank / bank.sum(1, keepdims=True).clip(1e-8)
    if kind == "mel":
        edges = _mel_to_hz(np.linspace(_hz_to_mel(fmin), _hz_to_mel(fmax), n_filters + 2))
    elif kind == "linear":
        edges = np.linspace(fmin, fmax, n_filters + 2)
    else:
        raise ValueError(f"unknown filterbank '{kind}'")
    lo, mid, hi = edges[:-2, None], edges[1:-1, None], edges[2:, None]
    bank = np.clip(np.minimum((f[None] - lo) / (mid - lo), (hi - f[None]) / (hi - mid)), 0, None)
    empty = bank.sum(1) == 0                  # filters narrower than the bin spacing: nearest bin
    for i in np.where(empty)[0]:
        bank[i, np.argmin(np.abs(f - mid[i, 0]))] = 1.0
    return bank


def morlet_cwt(wav: np.ndarray, freqs: np.ndarray, sr: int = SR, hop: int = HOP,
               cycles: float = 6.0) -> np.ndarray:
    """Magnitude of a complex Morlet continuous wavelet transform, (n_freqs, frames)."""
    n = len(wav)
    spec = np.fft.rfft(wav, 2 * n)
    omega = np.fft.rfftfreq(2 * n, 1.0 / sr)
    out = []
    for fc in freqs:
        sigma_f = fc / cycles                                    # frequency-domain width
        kernel = np.exp(-0.5 * ((omega - fc) / sigma_f) ** 2)    # analytic Morlet in frequency
        coeff = np.fft.irfft(spec * kernel, 2 * n)[:n]
        out.append(np.abs(coeff)[::hop])
    return np.stack(out)


def spectrogram(wav: np.ndarray, transform: str = "stft", filterbank: str = "mel",
                n_filters: int = 64, sr: int = SR) -> np.ndarray:
    """Log filterbank spectrogram, shape (frames, n_filters)."""
    import librosa
    if transform == "stft":
        power = np.abs(librosa.stft(wav, n_fft=N_FFT, hop_length=HOP)) ** 2
        freqs = librosa.fft_frequencies(sr=sr, n_fft=N_FFT)
    elif transform == "cqt":
        n_bins = 84
        power = np.abs(librosa.cqt(wav, sr=sr, hop_length=HOP, fmin=32.7, n_bins=n_bins,
                                   bins_per_octave=12)) ** 2
        freqs = librosa.cqt_frequencies(n_bins=n_bins, fmin=32.7, bins_per_octave=12)
    elif transform == "cwt":
        freqs = np.geomspace(40.0, sr / 2 * 0.95, 96)
        power = morlet_cwt(wav, freqs, sr) ** 2
    else:
        raise ValueError(f"unknown transform '{transform}'")
    bank = filterbank_on_bins(filterbank, freqs, n_filters, fmin=float(freqs[freqs > 0].min()),
                              fmax=float(freqs.max()))
    return np.log(bank @ power + 1e-10).T.astype(np.float32)


def stats_pool(frames: np.ndarray) -> np.ndarray:
    """Mean and standard deviation over time: (frames, d) -> (2d,)."""
    return np.concatenate([frames.mean(0), frames.std(0)]).astype(np.float32)


def main() -> None:
    from common.features.speech import load_audio
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kind", required=True, choices=["mfcc", "lfcc", "spectrogram"])
    ap.add_argument("--transform", default="stft", choices=["stft", "cqt", "cwt"])
    ap.add_argument("--filterbank", default="mel", choices=["mel", "gammatone", "linear"])
    ap.add_argument("--n-filters", type=int, default=64)
    ap.add_argument("--inputs", required=True, help="text file with one audio path per line")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--pooled", action="store_true", help="save mean+std vectors instead of frames")
    args = ap.parse_args()

    if args.kind == "spectrogram":
        fn = lambda w: spectrogram(w, args.transform, args.filterbank, args.n_filters)   # noqa: E731
    else:
        fn = mfcc if args.kind == "mfcc" else lfcc
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for line in Path(args.inputs).read_text().splitlines():
        if line.strip():
            feats = fn(load_audio(line.strip()))
            np.save(out / f"{Path(line).stem}.npy", stats_pool(feats) if args.pooled else feats)


if __name__ == "__main__":
    main()
