"""Golden vectors for host-side C parity tests (spec sections 3 and 11)."""
from __future__ import annotations

import numpy as np
from scipy.signal import chirp

from v5 import features as F

SYNTH_NAMES = ["sine100", "white40", "chirp", "silence"]
GOLDEN_NAMES = SYNTH_NAMES + [f"snore{i}" for i in range(1, 5)] + [f"noise{i}" for i in range(1, 5)]


def synth_golden_signals(seed: int = 1) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    t = np.arange(F.WIN) / F.SR
    floor = 3e-4 * rng.standard_normal(F.WIN)  # about -70 dBFS noise floor
    sine = 0.1 * np.sin(2 * np.pi * 100.0 * t) + floor
    white = 0.01 * rng.standard_normal(F.WIN)
    sweep = 0.1 * chirp(t, f0=1000.0, t1=1.0, f1=100.0) + floor
    return {
        "sine100": F.float_to_int16(sine),
        "white40": F.float_to_int16(white),
        "chirp": F.float_to_int16(sweep),
        "silence": np.zeros(F.WIN, dtype=np.int16),
    }


def make_golden(snore, noise, seed: int = 1, scale: float = 1.0 / 127.0, zero_point: int = 0):
    if len(snore) < 4 or len(noise) < 4:
        raise ValueError("need at least 4 snore and 4 noise clips")
    synth = synth_golden_signals(seed)
    x = [synth[n] for n in SYNTH_NAMES] + [np.asarray(c, np.int16) for c in snore[:4]] + [np.asarray(c, np.int16) for c in noise[:4]]
    x = np.stack(x)
    X = np.stack([F.extract_int16(xi) for xi in x]).astype(np.float32)
    q = np.stack([F.quantize(Xi, scale, zero_point) for Xi in X])
    return list(GOLDEN_NAMES), x, X, q


def write_golden_npz(path, names, x, X, q, scale: float, zero_point: int) -> None:
    np.savez(path, names=np.array(names), x=x, X=X, q=q, scale=np.float32(scale), zero_point=np.int32(zero_point))


def read_golden_npz(path):
    d = np.load(path)
    return [str(n) for n in d["names"]], d["x"], d["X"], d["q"], float(d["scale"]), int(d["zero_point"])


def write_golden_bin(path, x, X, q, scale: float, zero_point: int) -> None:
    x = np.asarray(x, np.int16)
    X = np.asarray(X, np.float32).reshape(len(x), -1)
    q = np.asarray(q, np.int8).reshape(len(x), -1)
    with open(path, "wb") as fh:
        fh.write(np.array([len(x), x.shape[1], X.shape[1]], dtype="<i4").tobytes())
        fh.write(np.array([scale], dtype="<f4").tobytes())
        fh.write(np.array([zero_point], dtype="<i4").tobytes())
        for xi, Xi, qi in zip(x, X, q):
            fh.write(xi.astype("<i2").tobytes())
            fh.write(Xi.astype("<f4").tobytes())
            fh.write(qi.tobytes())


def read_golden_bin(path):
    with open(path, "rb") as fh:
        count, win, dim = np.frombuffer(fh.read(12), dtype="<i4")
        scale = float(np.frombuffer(fh.read(4), dtype="<f4")[0])
        zero_point = int(np.frombuffer(fh.read(4), dtype="<i4")[0])
        xs, Xs, qs = [], [], []
        for _ in range(count):
            xs.append(np.frombuffer(fh.read(win * 2), dtype="<i2"))
            Xs.append(np.frombuffer(fh.read(dim * 4), dtype="<f4").reshape(F.N_FRAMES, F.N_MELS))
            qs.append(np.frombuffer(fh.read(dim), dtype=np.int8).reshape(F.N_FRAMES, F.N_MELS))
    return np.stack(xs), np.stack(Xs), np.stack(qs), scale, zero_point
