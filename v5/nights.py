"""Synthetic nights for the streaming benchmark (spec section 7)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from v5 import features as F
from v5.data.augment import apply_rir, rms


@dataclass
class NightSpec:
    duration_s: float = 3600.0
    snr_db: float = 10.0
    bed_dbfs: float = -40.0
    n_episodes: tuple = (6, 12)
    episode_len_s: tuple = (20.0, 120.0)
    period_s: tuple = (2.5, 5.0)
    n_distractors: tuple = (20, 40)
    distractor_snr_db: tuple = (0.0, 20.0)
    gap_s: float = 60.0


def _scale_to_rms(x, target: float) -> np.ndarray:
    r = rms(x)
    return (x * (target / r)).astype(np.float32) if r > 0 else np.asarray(x, np.float32)


def make_bed(rng, beds, n_samples: int, bed_dbfs: float) -> np.ndarray:
    bed = np.asarray(beds[int(rng.integers(0, len(beds)))], dtype=np.float32)
    start = int(rng.integers(0, len(bed)))
    reps = int(np.ceil((n_samples + start) / len(bed)))
    out = np.tile(bed, reps)[start: start + n_samples]
    return _scale_to_rms(out, 10 ** (bed_dbfs / 20.0))


def place_episodes(rng, spec: NightSpec) -> list[tuple[float, float]]:
    n = int(rng.integers(spec.n_episodes[0], spec.n_episodes[1] + 1))
    t, out = spec.gap_s, []
    for _ in range(n):
        length = float(rng.uniform(*spec.episode_len_s))
        if t + length + spec.gap_s > spec.duration_s:
            break
        out.append((t, t + length))
        t += length + float(rng.uniform(spec.gap_s, 2 * spec.gap_s))
    return out


def _add(audio, seg, start: int) -> None:
    end = min(len(audio), start + len(seg))
    if end > start:
        audio[start:end] += seg[: end - start]


def generate_night(rng, beds, snore_windows, distractor_windows, spec: NightSpec, rir=None):
    n = int(spec.duration_s * F.SR)
    audio = make_bed(rng, beds, n, spec.bed_dbfs)
    bed_rms = rms(audio)
    episodes = place_episodes(rng, spec)
    target = bed_rms * 10 ** (spec.snr_db / 20.0)
    for s, e in episodes:
        period, t = float(rng.uniform(*spec.period_s)), s
        while t + 1.0 <= e:
            w = np.asarray(snore_windows[int(rng.integers(0, len(snore_windows)))], np.float32)
            if rir is not None:
                w = apply_rir(w, rir)
            _add(audio, _scale_to_rms(w, target), int(t * F.SR))
            t += period * float(rng.uniform(0.9, 1.1))
    n_d = int(rng.integers(spec.n_distractors[0], spec.n_distractors[1] + 1)) if spec.n_distractors[1] > 0 else 0
    placed = tries = 0
    while placed < n_d and tries < 10 * n_d:
        tries += 1
        t = float(rng.uniform(0, spec.duration_s - 1.0))
        if any(s - 2.0 <= t <= e + 2.0 for s, e in episodes):
            continue
        w = np.asarray(distractor_windows[int(rng.integers(0, len(distractor_windows)))], np.float32)
        if rir is not None:
            w = apply_rir(w, rir)
        _add(audio, _scale_to_rms(w, bed_rms * 10 ** (float(rng.uniform(*spec.distractor_snr_db)) / 20.0)), int(t * F.SR))
        placed += 1
    return np.clip(audio, -1.0, 1.0).astype(np.float32), episodes
