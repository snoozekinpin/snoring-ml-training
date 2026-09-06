"""Optional YAMNet teacher: soft labels, Platt calibration, label audit (spec 4.5 and 6)."""
from __future__ import annotations

import argparse
import csv
import io
import json
from collections import Counter
from pathlib import Path

import numpy as np

from v5 import features as F
from v5.config import load_config, resolve

YAMNET_URL = "https://tfhub.dev/google/yamnet/1"
TARGET_CLASSES = ("Snoring", "Snort")


def class_names(model) -> list[str]:
    import tensorflow as tf

    path = model.class_map_path().numpy().decode("utf-8")
    text = tf.io.read_file(path).numpy().decode("utf-8")
    return [r["display_name"] for r in csv.DictReader(io.StringIO(text))]


def load_yamnet():
    try:
        import tensorflow_hub as hub

        model = hub.load(YAMNET_URL)
        idx = [i for i, n in enumerate(class_names(model)) if n in TARGET_CLASSES]
        if not idx:
            raise RuntimeError("Snoring class not found in the YAMNet class map")
        return model, idx
    except Exception as exc:  # spec section 13: degrade to hard labels
        print(f"[teacher] unavailable ({type(exc).__name__}: {exc}); KD and audit disabled")
        return None, []


def logit(p) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def score_windows(model, idx, audio_i16) -> np.ndarray:
    z = np.empty(len(audio_i16), np.float64)
    for i, a in enumerate(audio_i16):
        scores, _, _ = model(F.int16_to_float(a))
        s = scores.numpy().mean(axis=0)
        z[i] = logit(float(s[idx].sum()))
    return z


def platt_fit(z, y) -> tuple[float, float]:
    from sklearn.linear_model import LogisticRegression

    lr = LogisticRegression(C=1e4, max_iter=1000).fit(np.asarray(z, dtype=np.float64).reshape(-1, 1), np.asarray(y).astype(int))
    return float(lr.coef_[0, 0]), float(lr.intercept_[0])


def platt_apply(z, a: float, b: float) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-(a * np.asarray(z, dtype=np.float64) + b)))


AUDIT_POS_THRESHOLD = 0.1   # same rule as v5/data/manifest.py apply_audit
AUDIT_NEG_THRESHOLD = 0.5


def audit_flags(z, y, weak: float = AUDIT_POS_THRESHOLD, contaminated: float = AUDIT_NEG_THRESHOLD) -> list[str]:
    p = 1.0 / (1.0 + np.exp(-np.asarray(z, dtype=np.float64)))
    out = []
    for pp, yy in zip(p, np.asarray(y)):  # thresholds identical to manifest.apply_audit
        if yy == 1 and pp < weak:
            out.append("weak_positive")
        elif yy == 0 and pp > contaminated:
            out.append("snore_in_negative")
        else:
            out.append("")
    return out


def write_scores(path, ids, z, flags, md5s=None, meta: dict | None = None) -> None:
    """teacher.csv rows carry the window md5 so a score can never be applied to different audio;
    teacher_meta.json (next to it) records the cache fingerprint, the teacher and the thresholds."""
    md5s = list(md5s) if md5s is not None else [""] * len(list(ids))
    with open(path, "w", newline="", encoding="utf-8") as fh:
        wr = csv.writer(fh)
        wr.writerow(["id", "md5", "z", "flag"])
        for i, h, zz, fl in zip(ids, md5s, z, flags):
            wr.writerow([int(i), h, f"{float(zz):.6f}", fl])
    if meta is not None:
        Path(path).with_name("teacher_meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")


def read_scores(path) -> dict[int, tuple[float, str]]:
    with open(path, newline="", encoding="utf-8") as fh:
        return {int(r["id"]): (float(r["z"]), r["flag"]) for r in csv.DictReader(fh)}


def read_score_md5s(path) -> dict[int, str]:
    with open(path, newline="", encoding="utf-8") as fh:
        return {int(r["id"]): r.get("md5", "") for r in csv.DictReader(fh)}


def read_meta(path) -> dict:
    p = Path(path).with_name("teacher_meta.json")
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def main(argv=None) -> None:
    from v5.data.manifest import load_cache, read_manifest, split_indices

    ap = argparse.ArgumentParser(description="Score training windows with YAMNet (never the test split)")
    ap.add_argument("--config", default=None)
    args = ap.parse_args(argv)
    cfg = resolve(load_config(args.config))
    out = Path(cfg["paths"]["out_dir"])
    model, idx = load_yamnet()
    if model is None:
        return
    pre_audit = out / "manifest_unaudited.csv"  # all windows the audit judges, including ones an earlier audit dropped
    rows = read_manifest(pre_audit if pre_audit.exists() else out / "manifest.csv")
    _, audio, _ = load_cache(out)
    ids = np.concatenate([split_indices(rows, s) for s in ("train", "val", "calib", "bench")])  # never the test split
    z = score_windows(model, idx, audio[ids])
    dcfg = cfg.get("data", {})
    pos_thr, neg_thr = float(dcfg.get("audit_pos_threshold", AUDIT_POS_THRESHOLD)), float(dcfg.get("audit_neg_threshold", AUDIT_NEG_THRESHOLD))
    flags = audit_flags(z, [rows[i]["label"] for i in ids], pos_thr, neg_thr)
    cache_meta = json.loads((out / "cache" / "meta.json").read_text(encoding="utf-8"))
    write_scores(out / "teacher.csv", ids, z, flags, [rows[i]["md5"] for i in ids],
                 {"teacher": YAMNET_URL, "classes": list(TARGET_CLASSES), "cache_fingerprint": cache_meta.get("fingerprint"),
                  "pos_threshold": pos_thr, "neg_threshold": neg_thr, "n_scored": int(len(ids))})
    print(f"[teacher] scored {len(ids)} non-test windows -> {out / 'teacher.csv'}; flags: {dict(Counter(f for f in flags if f))}")


if __name__ == "__main__":
    main()
