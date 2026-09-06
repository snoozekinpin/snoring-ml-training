"""Domain adaptation on labelled self-recordings (spec section 12)."""
from __future__ import annotations

import argparse
import csv
import json
from datetime import date
from pathlib import Path

import keras
import numpy as np

from v5 import features as F
from v5.config import ROOT, load_config, resolve
from v5.data.augment import AugmentConfig, Augmenter
from v5.data.dataset import TrainDataset, precompute_features
from v5.data.sources import decode
from v5.evaluate import clip_metrics, predict_probs
from v5.events import check_model_version
from v5.train import kd_loss

LABEL_TO_Y = {"snore": 1, "breathing": 0, "speech": 0, "tv": 0, "fan": 0, "other": 0}


def load_labels(session_dir) -> list[dict]:
    path = Path(session_dir) / "labels.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing: copy labels_template.csv to labels.csv and fill it in")
    with open(path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    bad = [r["label"] for r in rows if r["label"] not in LABEL_TO_Y]
    if bad:
        raise ValueError(f"unknown labels {sorted(set(bad))}; allowed: {sorted(LABEL_TO_Y)}")
    by_chunk: dict[str, list] = {}
    for r in rows:
        by_chunk.setdefault(r["chunk"], []).append((float(r["start_s"]), float(r["end_s"]), r["label"]))
    for chunk, spans in by_chunk.items():
        spans.sort()
        for (s1, e1, l1), (s2, e2, l2) in zip(spans, spans[1:]):
            if s2 < e1:
                raise ValueError(f"overlapping spans in {chunk}: [{s1}, {e1}] {l1} and [{s2}, {e2}] {l2}")
    return rows


def slice_session(session_dir, hop_s: float = 1.0):
    session_dir = Path(session_dir)
    audio, y = [], []
    cache = {}
    for r in load_labels(session_dir):
        if r["chunk"] not in cache:
            cache[r["chunk"]] = decode(session_dir / r["chunk"])
        x = cache[r["chunk"]]
        t = float(r["start_s"])
        while t + 1.0 <= float(r["end_s"]) + 1e-9:
            w = x[int(t * F.SR): int(t * F.SR) + F.WIN]
            if len(w) == F.WIN:
                audio.append(F.float_to_int16(w))
                y.append(LABEL_TO_Y[r["label"]])
            t += hop_s
    if not audio:
        raise ValueError(f"no labelled windows in {session_dir}")
    return np.stack(audio), np.array(y, np.float32)


def finetune(model_path, sessions, holdout, out_dir, epochs: int = 10, lr: float = 1e-4, tau: float = 0.65, seed: int = 42) -> dict:
    if any(Path(s).resolve() == Path(holdout).resolve() for s in sessions):
        raise ValueError("the holdout session must not be one of the training sessions")
    parts = [slice_session(s) for s in sessions]
    audio, y = np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts])
    h_audio, h_y = slice_session(holdout)
    if y.min() == y.max():
        raise ValueError("training sessions need both snore and non-snore spans")
    model = keras.models.load_model(model_path, compile=False)
    for name in ("conv1", "bn1"):
        model.get_layer(name).trainable = False
    Xh = precompute_features(h_audio)
    before = clip_metrics(h_y, predict_probs(model, Xh), tau)
    rng = np.random.default_rng(seed)
    aug = Augmenter(AugmentConfig(p_rir=0.0, p_noise_pos=0.3, p_noise_neg=0.3, snr_range=(5.0, 20.0), p_gain=0.0), audio[y == 0], None, rng)
    ds = TrainDataset(audio, y, y, aug, batch=32, seed=seed)
    keras.utils.set_random_seed(seed)
    model.compile(optimizer=keras.optimizers.Adam(lr), loss=kd_loss)
    model.fit(ds, epochs=epochs, verbose=2)
    after = clip_metrics(h_y, predict_probs(model, Xh), tau)
    version = check_model_version(f"cnn_v5_ft_candidate_{date.today():%Y%m%d}")  # a candidate, not a deployable model
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save(out_dir / "model.keras")
    metrics = {"model_version": version, "deployable": False, "n_train": int(len(y)), "n_holdout": int(len(h_y)), "tau": tau, "epochs": epochs, "before": before, "after": after,
               "to_deploy": "python -m v5.train import-candidate --model <this model.keras> --name ft_<date> ; python -m v5.train final --run ft_<date> (verifies the run, re-calibrates on the calib split, binds the threshold to the model) ; python -m v5.export model (release gates)"}
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Fine-tune the deployed model on self-recordings")
    ap.add_argument("--sessions", nargs="+", required=True)
    ap.add_argument("--holdout", required=True)
    ap.add_argument("--model", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--config", default=None)
    args = ap.parse_args(argv)
    cfg = resolve(load_config(args.config))
    out_root = Path(cfg["paths"]["out_dir"])
    thr = out_root / "deployed" / "threshold.json"
    tau = json.loads(thr.read_text())["tau"] if thr.exists() else 0.65  # no calibration yet: development default
    m = finetune(args.model or out_root / "deployed" / "model.keras", args.sessions, args.holdout,
                 args.out or out_root / "finetune" / date.today().strftime("%Y%m%d"), args.epochs, tau=tau, seed=cfg["seed"])
    print(json.dumps({k: m[k] for k in ("model_version", "n_train", "n_holdout")}), "\nbefore:", json.dumps(m["before"]), "\nafter:", json.dumps(m["after"]))


if __name__ == "__main__":
    main()
