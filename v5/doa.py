"""Direction of arrival by GCC-PHAT on two microphones (spec section 9)."""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from v5 import features as F


@dataclass(frozen=True)
class DoaParams:
    spacing_m: float = 0.06
    fs: int = 16000
    c: float = 343.0
    band: tuple = (60.0, 3000.0)
    n_fft: int = 512
    deadzone: float = 0.2
    min_ratio: float = 1.5
    floor_margin_db: float = 6.0

    @property
    def max_lag(self) -> int:
        return int(math.ceil(self.spacing_m / self.c * self.fs)) + 1


BAND_MIN_FRACTION = 0.05  # below this share of the L channel's power inside the band there is no usable estimate


def gcc_phat(l, r, max_lag: int, fs: int = 16000, band=(60.0, 3000.0), n_fft: int = 512):
    w = F.hann_periodic(n_fft).astype(np.float64)
    L = np.fft.rfft(np.asarray(l, np.float64)[:n_fft] * w, n=n_fft)
    R = np.fft.rfft(np.asarray(r, np.float64)[:n_fft] * w, n=n_fft)
    freqs = np.fft.rfftfreq(n_fft, 1.0 / fs)
    in_band = (freqs >= band[0]) & (freqs <= band[1])
    power = np.abs(L) ** 2
    total = power.sum()
    if total <= 1e-18 or power[in_band].sum() < BAND_MIN_FRACTION * total:
        return 0.0, 0.0  # nothing usable inside the band (silence or out-of-band signal): lag 0, never valid
    G = R * np.conj(L)
    G = G / (np.abs(G) + 1e-12)
    G[~in_band] = 0.0
    cc = np.fft.irfft(G, n=n_fft)
    lags = np.concatenate([cc[-max_lag:], cc[: max_lag + 1]])  # index m <-> lag m - max_lag
    k = int(np.argmax(lags))
    lag = float(k - max_lag)
    if 0 < k < len(lags) - 1:
        y0, y1, y2 = lags[k - 1], lags[k], lags[k + 1]
        denom = y0 - 2 * y1 + y2
        if denom < 0:
            lag += 0.5 * (y0 - y2) / denom
    # peak ratio against the highest correlation anywhere else (all n_fft lags except the peak and its
    # neighbours): a true delay dominates the whole correlation, uncorrelated frames do not
    idx = (k - max_lag) % n_fft
    mask = np.ones(n_fft, bool)
    mask[[(idx - 1) % n_fft, idx, (idx + 1) % n_fft]] = False
    second = cc[mask].max()
    if lags[k] <= 1e-9 or second <= 1e-9:  # degenerate correlation: never valid
        return 0.0, 0.0
    return lag, float(lags[k] / second)


def frame_level_db(x) -> float:
    return float(20.0 * np.log10(np.sqrt(np.mean(np.asarray(x, np.float64) ** 2)) + 1e-9))


class NoiseFloor:
    """Tracks the ambient level: fast when the input is near or below the floor, slow upward for
    loud incoherent frames (a fan eventually stops counting as signal), and never raised by loud
    coherent frames (a snoring source must not erode its own validity)."""

    def __init__(self, init_db: float = -60.0, alpha: float = 0.02, alpha_up: float = 0.002):
        self.db, self.alpha, self.alpha_up = init_db, alpha, alpha_up

    def update(self, level_db: float, coherent: bool = False) -> float:
        if level_db < self.db + 5.0:
            self.db += self.alpha * (level_db - self.db)
        elif not coherent:
            self.db += self.alpha_up * (level_db - self.db)
        return self.db


def side_from_lag(lag: float, max_lag: int, deadzone: float = 0.2) -> str:
    thr = deadzone * max_lag
    if lag > thr:
        return "left"
    if lag < -thr:
        return "right"
    return "unknown"


LAG_BIN = 0.01  # samples; histogram resolution of the episode aggregation (parity tolerance is 0.05)


class DoaTracker:
    """Bounded-memory episode aggregation: a lag histogram plus per-frame side counts (mirrored in C)."""

    def __init__(self, params: DoaParams):
        self.p = params
        self.floor = NoiseFloor()
        self.n_bins = int(round(2 * params.max_lag / LAG_BIN)) + 1
        self.reset()

    def reset(self) -> None:
        self.hist = np.zeros(self.n_bins, dtype=np.int64)
        self.n_valid = 0
        self.side_counts = {"left": 0, "right": 0, "unknown": 0}

    def _bin(self, lag: float) -> int:
        return int(min(max(math.floor((lag + self.p.max_lag) / LAG_BIN + 0.5), 0), self.n_bins - 1))

    def frame(self, l, r):
        level = frame_level_db(l)
        lag, ratio = gcc_phat(l, r, self.p.max_lag, self.p.fs, self.p.band, self.p.n_fft)
        coherent = ratio >= self.p.min_ratio
        valid = bool(level >= self.floor.db + self.p.floor_margin_db and coherent)  # judged against the floor before this frame
        self.floor.update(level, coherent)
        if valid:
            self.hist[self._bin(lag)] += 1
            self.n_valid += 1
            self.side_counts[side_from_lag(lag, self.p.max_lag, self.p.deadzone)] += 1
        return lag, valid

    def episode(self) -> dict:
        if self.n_valid == 0:
            return {"side": "unknown", "lag_samples": 0.0, "lag_ms": 0.0, "conf": 0.0, "n_valid": 0}
        target = (self.n_valid - 1) // 2 + 1  # lower median
        b = int(np.searchsorted(np.cumsum(self.hist), target))
        med = b * LAG_BIN - self.p.max_lag
        side = side_from_lag(med, self.p.max_lag, self.p.deadzone)
        return {"side": side, "lag_samples": float(med), "lag_ms": float(med) / self.p.fs * 1000.0, "conf": self.side_counts[side] / self.n_valid, "n_valid": int(self.n_valid)}


def both_sides_rule(episodes, min_conf: float = 0.7, min_share: float = 0.25) -> bool:
    conf = [e for e in episodes if e.get("conf", 0.0) >= min_conf and e.get("side") in ("left", "right")]
    if len(conf) < 2:
        return False
    left = sum(1 for e in conf if e["side"] == "left") / len(conf)
    return min(left, 1.0 - left) >= min_share
