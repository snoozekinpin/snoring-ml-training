"""Waveform augmentation (spec section 5)."""
from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np
from scipy.signal import fftconvolve, lfilter

from v5 import features as F


def rms(x) -> float:
    return float(np.sqrt(np.mean(np.asarray(x, dtype=np.float64) ** 2)))


ISM_ORDER = 10  # spec section 5


def normalise_direct_path(h: np.ndarray, distance_m: float, fs: int = F.SR, c: float = 343.0) -> np.ndarray:
    """Scale so the direct-path peak (within 2.5 ms after the geometric arrival) is 1."""
    arrival = int(round(distance_m / c * fs))
    lo, hi = max(0, arrival - 10), min(len(h), arrival + 40)
    peak = np.abs(h[lo:hi]).max() if hi > lo else 0.0
    if peak <= 0:
        peak = np.abs(h).max() + 1e-12
    return (h / peak).astype(np.float32)


def make_rir(rng, fs: int = F.SR, max_len: int = 8000) -> np.ndarray:
    import pyroomacoustics as pra

    dims = [float(rng.uniform(3.0, 5.0)), float(rng.uniform(3.0, 5.0)), float(rng.uniform(2.4, 3.0))]
    rt60 = float(rng.uniform(0.2, 0.6))
    e_abs, _ = pra.inverse_sabine(rt60, dims)
    room = pra.ShoeBox(dims, fs=fs, materials=pra.Material(e_abs), max_order=ISM_ORDER)
    mic = np.array([rng.uniform(0.3, dims[0] - 0.3), rng.uniform(0.3, dims[1] - 0.3), rng.uniform(0.5, 0.9)])
    src = None
    for _ in range(50):
        d, az = rng.uniform(0.4, 1.8), rng.uniform(0.0, 2 * np.pi)
        cand = np.array([mic[0] + d * np.cos(az), mic[1] + d * np.sin(az), rng.uniform(0.4, 0.7)])
        if np.all(cand > 0.2) and np.all(cand < np.array(dims) - 0.2):
            src = cand
            break
    if src is None:
        src = np.array([dims[0] / 2, dims[1] / 2, 0.6])
    room.add_source(src.tolist())
    room.add_microphone(mic.tolist())
    room.compute_rir()
    h = np.asarray(room.rir[0][0], dtype=np.float32)
    h = h[:max_len] if len(h) >= max_len else np.pad(h, (0, max_len - len(h)))
    return normalise_direct_path(h, float(np.linalg.norm(src - mic)), fs)


class RirBank:
    def __init__(self, rirs: list[np.ndarray]):
        self.rirs = [np.asarray(h, dtype=np.float32) for h in rirs]

    def __len__(self) -> int:
        return len(self.rirs)

    @classmethod
    def generate(cls, n: int, seed: int = 42, max_len: int = 8000) -> "RirBank":
        rng = np.random.default_rng(seed)
        return cls([make_rir(rng, max_len=max_len) for _ in range(n)])

    def save(self, path) -> None:
        np.savez_compressed(path, rirs=np.stack(self.rirs))

    @classmethod
    def load_or_generate(cls, path, n: int, seed: int = 42) -> "RirBank":
        path = Path(path)
        if path.exists():
            return cls(list(np.load(path)["rirs"]))
        bank = cls.generate(n, seed)
        path.parent.mkdir(parents=True, exist_ok=True)
        bank.save(path)
        return bank

    def random(self, rng) -> np.ndarray:
        return self.rirs[int(rng.integers(0, len(self.rirs)))]


def apply_rir(x, h) -> np.ndarray:
    return fftconvolve(np.asarray(x, dtype=np.float32), h)[: len(x)].astype(np.float32)


def mix_noise(x, noise, snr_db: float) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    pn = rms(noise)
    if pn <= 0:
        return x
    scale = rms(x) / (pn * 10 ** (snr_db / 20.0))
    return (x + scale * np.asarray(noise, dtype=np.float32)).astype(np.float32)


def spectral_tilt(x, a: float, fc: float = 1000.0) -> np.ndarray:
    alpha = float(np.exp(-2 * np.pi * fc / F.SR))
    lp = lfilter([1 - alpha], [1, -alpha], np.asarray(x, dtype=np.float64))
    return (x + a * (x - lp)).astype(np.float32)


def time_shift(x, shift_samples: int) -> np.ndarray:
    y = np.zeros_like(x, dtype=np.float32)
    s = int(shift_samples)
    if s > 0:
        y[s:] = x[:-s]
    elif s < 0:
        y[:s] = x[-s:]
    else:
        y[:] = x
    return y


def gain_clip(x, gain_db: float) -> np.ndarray:
    return np.clip(np.asarray(x, dtype=np.float32) * 10 ** (gain_db / 20.0), -1.0, 1.0).astype(np.float32)


def spec_augment(X, rng, max_f: int = 4, max_t: int = 8) -> np.ndarray:
    X = np.array(X, dtype=np.float32, copy=True)
    fw, tw = int(rng.integers(1, max_f + 1)), int(rng.integers(1, max_t + 1))
    f0, t0 = int(rng.integers(0, X.shape[1] - fw + 1)), int(rng.integers(0, X.shape[0] - tw + 1))
    X[:, f0: f0 + fw] = 0.0
    X[t0: t0 + tw, :] = 0.0
    return X


@dataclass
class AugmentConfig:
    p_rir: float = 0.6
    p_noise_pos: float = 0.7
    p_noise_neg: float = 0.4
    snr_range: tuple = (-5.0, 20.0)
    p_tilt: float = 0.3
    tilt_range: tuple = (-0.5, 0.5)
    p_shift: float = 0.5
    shift_ms: float = 100.0
    p_gain: float = 0.3
    gain_range: tuple = (-12.0, 6.0)
    p_specaug: float = 0.5

    @classmethod
    def from_dict(cls, d: dict) -> "AugmentConfig":
        names = {f.name for f in fields(cls)}
        kw = {}
        for k, v in d.items():
            if k in names:
                kw[k] = tuple(float(u) for u in v) if isinstance(v, (list, tuple)) else v
        return cls(**kw)


class Augmenter:
    def __init__(self, cfg: AugmentConfig, noise_bank, rir_bank: RirBank | None, rng):
        self.cfg, self.rir_bank, self.rng = cfg, rir_bank, rng
        # int16 banks are kept as-is and converted per pick (a float copy of 8k windows is 500 MB)
        self.noise_bank = np.asarray(noise_bank) if noise_bank is not None and len(noise_bank) else None

    def waveform(self, x, label: int, rng=None) -> np.ndarray:
        """Augment one waveform. Pass an explicit generator for reproducible per-sample draws."""
        c, rng = self.cfg, (rng if rng is not None else self.rng)
        y = np.asarray(x, dtype=np.float32)
        if self.rir_bank is not None and len(self.rir_bank) and rng.random() < c.p_rir:
            y = apply_rir(y, self.rir_bank.random(rng))
        p_noise = c.p_noise_pos if label == 1 else c.p_noise_neg
        if self.noise_bank is not None and rng.random() < p_noise:
            n = self.noise_bank[int(rng.integers(0, len(self.noise_bank)))]
            if n.dtype == np.int16:
                n = F.int16_to_float(n)
            y = mix_noise(y, n, float(rng.uniform(*c.snr_range)))
        if rng.random() < c.p_tilt:
            y = spectral_tilt(y, float(rng.uniform(*c.tilt_range)))
        if rng.random() < c.p_shift:
            y = time_shift(y, int(rng.integers(-int(c.shift_ms * F.SR / 1000), int(c.shift_ms * F.SR / 1000) + 1)))
        if rng.random() < c.p_gain:
            y = gain_clip(y, float(rng.uniform(*c.gain_range)))
        return y.astype(np.float32)

    def features(self, X, rng=None) -> np.ndarray:
        rng = rng if rng is not None else self.rng
        if rng.random() < self.cfg.p_specaug:
            return spec_augment(X, rng)
        return np.asarray(X, dtype=np.float32)
