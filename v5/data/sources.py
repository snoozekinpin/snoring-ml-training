"""Dataset sources: decoding, slicing into one-second windows, group ids (spec 4.1-4.2)."""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import soundfile as sf
import soxr

from v5 import features as F

SOURCE_IDS = ["whl_s", "whl_e", "esc50", "mssnsd", "kaggle_adria", "kaggle_jibran", "wild"]
TEST_SOURCES = frozenset({"kaggle_adria", "kaggle_jibran"})
FRAME = 512


@dataclass(frozen=True)
class Window:
    source: str
    path: str      # relative to data_dir
    offset: int    # sample offset into the decoded 16 kHz mono file
    label: int     # 1 snore, 0 noise
    group: str
    category: str


def decode(path) -> np.ndarray:
    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    mono = data.mean(axis=1)
    if sr != F.SR:
        mono = soxr.resample(mono, sr, F.SR, quality="HQ")
    return np.ascontiguousarray(mono, dtype=np.float32)


def frame_rms_db(x, frame: int = FRAME) -> np.ndarray:
    n = len(x) // frame
    if n == 0:
        return np.full(1, -120.0)
    fr = np.asarray(x[: n * frame], dtype=np.float64).reshape(n, frame)
    return 20.0 * np.log10(np.sqrt((fr ** 2).mean(axis=1)) + 1e-9)


def fit_window(x, offset: int) -> np.ndarray:
    seg = np.asarray(x[offset: offset + F.WIN], dtype=np.float32)
    if len(seg) < F.WIN:
        seg = np.pad(seg, (0, F.WIN - len(seg)))
    return seg


def centre_offset(x) -> int:
    return max(0, (len(x) - F.WIN) // 2)


def _clip_offset(centre: int, n: int) -> int:
    return int(np.clip(centre - F.WIN // 2, 0, max(0, n - F.WIN)))


def max_rms_offset(x) -> int:
    rms = frame_rms_db(x)
    i = int(np.argmax(rms))
    return _clip_offset(i * FRAME + FRAME // 2, len(x))


def peak_offsets(x, n_max: int, min_sep_s: float, pct: float = 60.0) -> list[int]:
    rms = frame_rms_db(x)
    if len(rms) < 3 or n_max <= 0:
        return []
    thr = np.percentile(rms, pct)
    cand = [i for i in range(1, len(rms) - 1) if rms[i] >= thr and rms[i] > rms[i - 1] and rms[i] >= rms[i + 1]]
    cand.sort(key=lambda i: -rms[i])
    min_sep = int(min_sep_s * F.SR)
    centres: list[int] = []
    for i in cand:
        c = i * FRAME + FRAME // 2
        if all(abs(c - other) >= min_sep for other in centres):
            centres.append(c)
        if len(centres) >= n_max:
            break
    return [_clip_offset(c, len(x)) for c in centres]


def random_offset(x, rng) -> int:
    return int(rng.integers(0, max(0, len(x) - F.WIN) + 1))


def is_digital_silence(w) -> bool:
    return 20.0 * np.log10(np.sqrt(np.mean(np.asarray(w, dtype=np.float64) ** 2)) + 1e-9) < -80.0


def _rel(p: Path, data_dir: Path) -> str:
    return str(p.relative_to(data_dir))


def _safe_decode(f: Path, errors):
    try:
        return decode(f)
    except Exception as exc:  # unreadable file: record and move on (spec section 13)
        if errors is not None:
            errors.append({"path": str(f), "error": f"{type(exc).__name__}: {exc}"})
        return None


def _emit(source, f, data_dir, offsets, label, group, category, x):
    seen = set()
    for o in offsets:
        if o in seen:
            continue
        seen.add(o)
        w = fit_window(x, o)
        if is_digital_silence(w):
            continue
        yield Window(source, _rel(f, data_dir), int(o), label, group, category), w


def iter_windows(data_dir, source: str, rng, cfg: dict | None = None, errors: list | None = None) -> Iterator[tuple[Window, np.ndarray]]:
    data_dir = Path(data_dir)
    cfg = cfg or {}
    if source in ("whl_s", "whl_e"):
        prefix, label = ("s", 1) if source == "whl_s" else ("e", 0)
        base = data_dir / "whltalent"
        for d in sorted(p for p in base.iterdir() if p.is_dir() and p.name.startswith(prefix)):
            for f in sorted(d.glob("*.wav")):
                x = _safe_decode(f, errors)
                if x is None:
                    continue
                if label:
                    offs = peak_offsets(x, cfg.get("whl_snore_max_windows", 3), 1.5)
                    if not offs and errors is not None:
                        errors.append({"path": str(f), "error": "no_qualifying_peak"})  # spec 4.2: skipped files are logged, never silent
                else:
                    offs = [max_rms_offset(x), random_offset(x, rng)][: cfg.get("whl_env_max_windows", 2)]
                batch = f.stem.split("-")[0]  # WHLTalent file-name prefix = batch id (used for the split)
                yield from _emit(source, f, data_dir, offs, label, f"whl_{batch}_{f.stem}", batch, x)
    elif source == "esc50":
        meta = {}
        with open(data_dir / "esc50" / "meta" / "esc50.csv", newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                meta[row["filename"]] = row
        for f in sorted((data_dir / "esc50" / "audio").glob("*.wav")):
            row = meta.get(f.name)
            if row is None:
                continue
            label = 1 if row["category"] == "snoring" else 0
            n_max = cfg.get("esc50_snore_max_windows", 3) if label else cfg.get("esc50_max_windows", 2)
            x = _safe_decode(f, errors)
            if x is None:
                continue
            yield from _emit(source, f, data_dir, peak_offsets(x, n_max, 1.0), label, f"esc50_fold{row['fold']}", row["category"], x)
    elif source == "mssnsd":
        stride = int(cfg.get("mssnsd_stride_s", 5) * F.SR)
        n_max = cfg.get("mssnsd_max_windows", 40)
        for f in sorted((data_dir / "RAW" / "MS-SNSD" / "noise_train").glob("*.wav")):
            x = _safe_decode(f, errors)
            if x is None:
                continue
            offs = list(range(0, max(1, len(x) - F.WIN + 1), stride))[:n_max]
            yield from _emit(source, f, data_dir, offs, 0, f"mssnsd_{f.stem}", f.stem.split("_")[0], x)
    elif source == "kaggle_adria":
        for label, sub in ((1, "snore"), (0, "noise")):
            for f in sorted((data_dir / "adrianagaler" / sub).glob("*.wav")):
                x = _safe_decode(f, errors)
                if x is None:
                    continue
                yield from _emit(source, f, data_dir, [centre_offset(x)], label, f"adria_{f.stem}", sub, x)
    elif source == "kaggle_jibran":
        for f in sorted((data_dir / "snoring_extra" / "jibran").glob("*.wav")):
            label = 1 if "_s_" in f.name else 0
            x = _safe_decode(f, errors)
            if x is None:
                continue
            yield from _emit(source, f, data_dir, [centre_offset(x)], label, f"jibran_{f.stem}", "snore" if label else "noise", x)
    elif source == "wild":
        base = data_dir / "RAW" / "Snore_Detection_Project" / "Snore_Detection" / "inference_audios"
        for f in sorted(base.glob("snore*.wav")):
            x = _safe_decode(f, errors)
            if x is None:
                continue
            yield from _emit(source, f, data_dir, [centre_offset(x)], 1, f"wild_{f.stem}", "snore", x)
    else:
        raise ValueError(f"unknown source {source}")


def load_window(data_dir, w: Window) -> np.ndarray:
    return fit_window(decode(Path(data_dir) / w.path), w.offset)
