"""Feature spec v1 (frozen). Spec section 3. Nothing else may redefine these constants."""
from __future__ import annotations

import numpy as np

FEATURE_SPEC_VERSION = 1
SR = 16000
WIN = 16000
STREAM_HOP = 8000
N_FFT = 512
HOP = 256
N_FRAMES = 1 + (WIN - N_FFT) // HOP  # 61
N_BINS = N_FFT // 2 + 1  # 257
N_MELS = 30
FMIN = 40.0
FMAX = 6000.0
LOG_EPS = 1e-10
NORM_DIV = 40.0
FEATURE_DIM = N_FRAMES * N_MELS


def hann_periodic(n: int = N_FFT) -> np.ndarray:
    k = np.arange(n, dtype=np.float64)
    return (0.5 - 0.5 * np.cos(2.0 * np.pi * k / n)).astype(np.float32)


def _hz_to_mel(f):
    return 2595.0 * np.log10(1.0 + np.asarray(f, dtype=np.float64) / 700.0)


def _mel_to_hz(m):
    return 700.0 * (10.0 ** (np.asarray(m, dtype=np.float64) / 2595.0) - 1.0)


def mel_filterbank() -> np.ndarray:
    """(30, 257) triangular HTK-mel filters with unit peak, evaluated at bin centres."""
    edges = _mel_to_hz(np.linspace(_hz_to_mel(FMIN), _hz_to_mel(FMAX), N_MELS + 2))
    fk = np.arange(N_BINS, dtype=np.float64) * SR / N_FFT
    fb = np.zeros((N_MELS, N_BINS), dtype=np.float32)
    for i in range(N_MELS):
        lo, c, hi = edges[i], edges[i + 1], edges[i + 2]
        up = (fk - lo) / (c - lo)
        down = (hi - fk) / (hi - c)
        fb[i] = np.maximum(0.0, np.minimum(up, down)).astype(np.float32)
    return fb


HANN = hann_periodic()
FB = mel_filterbank()


def sparse_filterbank() -> list[tuple[int, np.ndarray]]:
    out = []
    for i in range(N_MELS):
        nz = np.nonzero(FB[i] > 0)[0]
        out.append((int(nz[0]), FB[i, nz[0]:nz[-1] + 1].copy()))
    return out


def frames_of(x) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.shape != (WIN,):
        raise ValueError(f"expected {WIN} samples, got {x.shape}")
    return np.lib.stride_tricks.sliding_window_view(x, N_FFT)[::HOP][:N_FRAMES]


def log_mel(x) -> np.ndarray:
    """(61, 30) log-mel power in dB, float64 reference."""
    fr = frames_of(x).astype(np.float64) * HANN.astype(np.float64)
    power = np.abs(np.fft.rfft(fr, axis=1)) ** 2
    mel = power @ FB.astype(np.float64).T
    return 10.0 * np.log10(mel + LOG_EPS)


def extract(x) -> np.ndarray:
    L = log_mel(x)
    return np.clip((L - L.mean()) / NORM_DIV, -1.0, 1.0).astype(np.float32)


def int16_to_float(x_i16) -> np.ndarray:
    return np.asarray(x_i16, dtype=np.float32) / 32768.0


def float_to_int16(x) -> np.ndarray:
    return np.clip(np.round(np.asarray(x, dtype=np.float64) * 32767.0), -32768, 32767).astype(np.int16)


def extract_int16(x_i16) -> np.ndarray:
    return extract(int16_to_float(x_i16))


def quantize(X, scale: float, zero_point: int) -> np.ndarray:
    q = np.round(np.asarray(X, dtype=np.float64) / scale) + zero_point
    return np.clip(q, -128, 127).astype(np.int8)


def feature_spec_dict() -> dict:
    return {
        "version": FEATURE_SPEC_VERSION,
        "sample_rate": SR,
        "window": WIN,
        "stream_hop": STREAM_HOP,
        "n_fft": N_FFT,
        "hop": HOP,
        "n_frames": N_FRAMES,
        "n_bins": N_BINS,
        "n_mels": N_MELS,
        "fmin": FMIN,
        "fmax": FMAX,
        "mel_scale": "htk",
        "log_eps": LOG_EPS,
        "norm_div": NORM_DIV,
        "window_type": "hann_periodic",
        "spectrum": "power",
        "normalisation": "clip((10log10(mel+eps) - mean)/40, -1, 1)",
    }
