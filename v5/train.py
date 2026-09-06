"""Training runs, validation-based selection and fail-closed threshold calibration (spec section 6)."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import time
from pathlib import Path

import keras
import numpy as np
from sklearn.metrics import roc_auc_score

from v5 import features as F
from v5 import teacher as T
from v5.config import load_config, resolve
from v5.data import manifest as M
from v5.data.augment import AugmentConfig, Augmenter, RirBank
from v5.data.dataset import TrainDataset, precompute_features
from v5.evaluate import clip_metrics, predict_probs, sigmoid
from v5.model import build_model
from v5.versioning import check_model_version

_BCE = keras.losses.BinaryCrossentropy(from_logits=True)


class CalibrationError(RuntimeError):
    """Threshold calibration cannot meet the FPR target; deployment must not proceed."""


def make_kd_loss(alpha: float):
    """alpha * BCE(hard) + (1 - alpha) * BCE(soft). Column 0 of y_true is the hard label, column 1 the soft target."""

    def kd_loss(y_true, logits):
        return alpha * _BCE(y_true[:, 0:1], logits) + (1.0 - alpha) * _BCE(y_true[:, 1:2], logits)

    kd_loss.__name__ = f"kd_loss_a{alpha:g}".replace(".", "p")
    return kd_loss


kd_loss = make_kd_loss(0.5)


class ValAuc(keras.callbacks.Callback):
    def __init__(self, Xv, yv, patience: int):
        super().__init__()
        self.Xv, self.yv, self.patience = Xv, np.asarray(yv).astype(int), patience
        self.best, self.best_epoch, self.best_weights, self.wait, self.history = -1.0, -1, None, 0, []

    def on_epoch_end(self, epoch, logs=None):
        p = sigmoid(self.model.predict(self.Xv, batch_size=256, verbose=0).ravel())
        auc = float(roc_auc_score(self.yv, p)) if len(np.unique(self.yv)) == 2 else 0.5
        self.history.append(auc)
        if logs is not None:
            logs["val_auc"] = auc
        if auc > self.best + 1e-4:
            self.best, self.best_epoch, self.best_weights, self.wait = auc, epoch, self.model.get_weights(), 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.model.stop_training = True

    def on_train_end(self, logs=None):
        if self.best_weights is not None:
            self.model.set_weights(self.best_weights)


def build_soft_targets(rows, teacher_csv, train_idx) -> np.ndarray:
    y = np.array([r["label"] for r in rows], np.float64)
    scores = T.read_scores(teacher_csv)
    z = np.array([scores.get(r["id"], (np.nan, ""))[0] for r in rows], np.float64)
    have = ~np.isnan(z)
    tr = np.array([i for i in train_idx if have[i]], dtype=int)
    if len(tr) < 50:
        raise ValueError("too few teacher scores on the training split")
    a, b = T.platt_fit(z[tr], y[tr])
    soft = y.copy()
    soft[have] = T.platt_apply(z[have], a, b)
    return soft.astype(np.float32)


def noise_bank_ids(rows, train_ids) -> np.ndarray:
    """Training negatives from MS-SNSD and ESC-50 only (spec section 5); WHLTalent environment windows are not mixed in as noise."""
    allowed = {"mssnsd", "esc50"}
    return np.array([i for i in train_ids if rows[i]["label"] == 0 and rows[i]["source"] in allowed], dtype=int)


def _teacher_excluded(rows, teacher_csv) -> set[int]:
    if not Path(teacher_csv).exists():
        return set()
    return {i for i, (_, flag) in T.read_scores(teacher_csv).items() if flag == "snore_in_negative"}


class AuditRequired(RuntimeError):
    """Training needs a manifest that went through the teacher audit (spec 4.5)."""


class TeacherMismatch(RuntimeError):
    """teacher.csv no longer matches the audit that shaped the manifest, or scores audio other than the training windows."""


def verify_teacher_binding(out: Path, rows, audit_status: dict, active_ids) -> None:
    """KD targets and the negative exclusion list must come from the very file the audit used (spec 4.5)."""
    teacher_csv, meta = out / "teacher.csv", out / "teacher_meta.json"
    if audit_status.get("applied"):  # an audited manifest is bound to its teacher files: they must exist and be unchanged
        for path, key in ((teacher_csv, "teacher_csv_sha256"), (meta, "teacher_meta_sha256")):
            if not path.exists() or file_sha256(path) != audit_status.get(key):
                raise TeacherMismatch(f"{path.name} is missing or differs from the file the audit used; re-run python -m v5.teacher and python -m v5.data.manifest --apply-audit")
    if not teacher_csv.exists():
        return  # explicitly unaudited hard-label run without a teacher
    md5_by_id = {r["id"]: r["md5"] for r in rows}
    md5s = T.read_score_md5s(teacher_csv)
    stale = [int(i) for i in active_ids if int(i) in md5s and md5s[int(i)] != md5_by_id.get(int(i))]
    if stale:
        raise TeacherMismatch(f"teacher.csv scores {len(stale)} training windows whose audio differs from the manifest; re-run python -m v5.teacher")


def require_audited_manifest(out: Path, cfg: dict) -> dict:
    status_path = out / "audit_status.json"
    status = json.loads(status_path.read_text(encoding="utf-8")) if status_path.exists() else {"applied": False}
    if not status.get("applied") and not cfg.get("train", {}).get("allow_unaudited", False):
        raise AuditRequired("manifest was not audited: run `python -m v5.teacher` then `python -m v5.data.manifest --apply-audit` "
                            "(or set train.allow_unaudited: true to record a degraded, unverified run)")
    return status


def train_one(cfg: dict, use_kd: bool, width: float, epochs: int, name: str, seed: int = 42) -> dict:
    out = Path(cfg["paths"]["out_dir"])
    audit_status = require_audited_manifest(out, cfg)
    rows = M.read_manifest(out / "manifest.csv")
    manifest_sha = file_sha256(out / "manifest.csv")
    _, audio, _ = M.load_cache(out)
    verify_teacher_binding(out, rows, audit_status, M.split_indices(rows, "train"))
    excluded = _teacher_excluded(rows, out / "teacher.csv")
    tr = np.array([i for i in M.split_indices(rows, "train") if i not in excluded], dtype=int)
    va = M.split_indices(rows, "val")
    y = np.array([r["label"] for r in rows], np.float32)
    kd_note = ""
    if use_kd and not (out / "teacher.csv").exists():
        use_kd, kd_note = False, "teacher.csv missing: trained with hard labels only"
        print(f"[train] {kd_note}")
    tcfg = cfg["train"]
    alpha = float(tcfg.get("kd_alpha", 0.5)) if use_kd else 1.0
    soft = build_soft_targets(rows, out / "teacher.csv", tr) if use_kd else y
    rng = np.random.default_rng(seed)
    rir_bank = RirBank.load_or_generate(out / "rir_bank.npz", int(cfg["augment"]["rir_bank_size"]), seed)
    aug = Augmenter(AugmentConfig.from_dict(cfg["augment"]), audio[noise_bank_ids(rows, tr)], rir_bank, rng)
    ds = TrainDataset(audio[tr], y[tr], soft[tr], aug, batch=int(tcfg["batch"]), seed=seed, workers=int(tcfg.get("workers", 8)))
    Xv, yv = precompute_features(audio[va]), y[va]
    keras.utils.set_random_seed(seed)
    model = build_model(width)
    schedule = keras.optimizers.schedules.CosineDecay(float(tcfg["lr"]), max(1, len(ds) * epochs), alpha=float(tcfg["lr_min"]) / float(tcfg["lr"]))
    model.compile(optimizer=keras.optimizers.Adam(schedule), loss=make_kd_loss(alpha))
    cb = ValAuc(Xv, yv, int(tcfg["patience"]))
    t0 = time.time()
    hist = model.fit(ds, epochs=epochs, callbacks=[cb], verbose=2)
    run_dir = out / "runs" / name
    run_dir.mkdir(parents=True, exist_ok=True)
    model.save(run_dir / "model.keras")
    metrics = {
        "name": name, "kd": bool(use_kd), "kd_alpha": alpha, "kd_note": kd_note, "width": width, "params": int(model.count_params()),
        "manifest_sha256": manifest_sha, "val_ids_sha256": hashlib.sha256(np.asarray(va, dtype=np.int64).tobytes()).hexdigest(),
        "model_sha256": file_sha256(run_dir / "model.keras"), "audit": audit_status,
        "epochs_run": len(cb.history), "best_epoch": cb.best_epoch, "train_seconds": round(time.time() - t0, 1),
        "n_train": int(len(tr)), "n_val": int(len(va)), "val": clip_metrics(yv, predict_probs(model, Xv), 0.5), "val_auc_history": cb.history,
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (run_dir / "history.json").write_text(json.dumps({k: [float(v) for v in vals] for k, vals in hist.history.items()}), encoding="utf-8")
    return metrics


def file_sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fpr_upper_bound(fp: int, n: int, conf: float = 0.95) -> float:
    """One-sided Clopper-Pearson upper bound on the false-positive rate."""
    from scipy.stats import beta

    if n <= 0:
        return 1.0
    return float(beta.ppf(conf, fp + 1, max(n - fp, 1))) if fp < n else 1.0


def choose_threshold(y_calib, p_calib, max_fpr: float = 0.01, min_neg: int = 300):
    y, p = np.asarray(y_calib).astype(int), np.asarray(p_calib, dtype=np.float64)
    neg, pos = np.sort(p[y == 0]), p[y == 1]
    n_neg = int(len(neg))
    if n_neg < min_neg:
        raise CalibrationError(f"need at least {min_neg} calibration negatives, have {n_neg}")
    if len(pos) == 0:
        raise CalibrationError("no calibration positives")
    k = int(math.floor(max_fpr * n_neg))  # negatives allowed at or above tau
    permitted = float(neg[n_neg - k - 1])  # the largest negative that may stay below tau
    if permitted >= 1.0:
        raise CalibrationError("negatives saturate at probability 1.0; no threshold meets the FPR target")
    # the firmware compares a float32 probability with a float32 threshold: use the smallest float32 strictly above
    # the permitted negative, so every consumer (metrics, parity, benchmark, JSON, model_meta.h) shares one exact value
    tau32 = np.float32(permitted)
    while float(tau32) <= permitted:
        tau32 = np.nextafter(tau32, np.float32(np.inf))
    tau = float(tau32)
    if tau > 1.0:
        raise CalibrationError("negatives saturate at probability 1.0; no threshold meets the FPR target")
    fp = int((neg >= tau).sum())
    recall = float((pos >= tau).mean())
    if recall == 0.0:
        raise CalibrationError("no calibration positive passes the FPR-constrained threshold")
    info = {"tau": tau, "n_neg": n_neg, "n_pos": int(len(pos)), "fp": fp, "fpr": fp / n_neg, "fpr_upper95": fpr_upper_bound(fp, n_neg), "recall": recall, "max_fpr": float(max_fpr)}
    assert info["fpr"] <= max_fpr
    return tau, info


def deploy_generation(out_dir, run_dir, metrics: dict, threshold: dict) -> Path:
    """Write model.keras, metrics.json and threshold.json as one generation and swap it into out_dir/deployed."""
    out_dir, run_dir = Path(out_dir), Path(run_dir)
    live, new, bak = out_dir / "deployed", out_dir / "deployed.new", out_dir / "deployed.bak"
    for d in (new, bak):
        if d.exists():
            shutil.rmtree(d)
    new.mkdir(parents=True)
    shutil.copy(run_dir / "model.keras", new / "model.keras")
    threshold = {**threshold, "model_sha256": file_sha256(new / "model.keras")}
    (new / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (new / "threshold.json").write_text(json.dumps(threshold, indent=2), encoding="utf-8")
    had_live = live.exists()
    try:
        if had_live:
            os.rename(live, bak)
        os.rename(new, live)
    except Exception:
        if had_live and bak.exists() and not live.exists():
            os.rename(bak, live)
        shutil.rmtree(new, ignore_errors=True)
        raise
    shutil.rmtree(bak, ignore_errors=True)
    return live


REQUIRED_RUNS = {"hard_w1": (False, 1.0), "kd_w1": (True, 1.0), "hard_w2": (False, 2.0), "kd_w2": (True, 2.0)}  # name -> (kd, width)


def current_runs(out: Path, manifest_sha: str) -> list[dict]:
    """Runs whose metrics were produced on the current manifest and whose saved model still matches its hash."""
    runs = []
    for mp in sorted(out.glob("runs/*/metrics.json")):
        m = json.loads(mp.read_text(encoding="utf-8"))
        model_path = mp.parent / "model.keras"
        if m.get("manifest_sha256") != manifest_sha or not model_path.exists() or file_sha256(model_path) != m.get("model_sha256"):
            continue
        runs.append(m)
    return runs


def select_config(results: list[dict]) -> dict:
    """Spec 6: the smallest run within 0.005 AUC and 2 points recall@2%FPR of the best; when no run
    satisfies both, the AUC-near-best run with the highest recall (parameter count, then hard labels
    over KD, only break exact recall ties)."""
    if not results:
        raise ValueError("no runs to select from")
    best_auc = max(r["val"]["auc"] for r in results)
    best_rec = max(r["val"]["recall_at_fpr2"] for r in results)
    near_auc = [r for r in results if r["val"]["auc"] >= best_auc - 0.005]
    cands = [r for r in near_auc if r["val"]["recall_at_fpr2"] >= best_rec - 0.02]
    if cands:
        return min(cands, key=lambda r: (r["params"], 1 if r["kd"] else 0))
    return max(near_auc, key=lambda r: (r["val"]["recall_at_fpr2"], -r["params"], 0 if r["kd"] else 1))


def run_report(out_dir) -> str:
    out = Path(out_dir)
    lines = ["# v5 training runs (selection on the validation split; the test split is untouched until export)", "",
             "| run | kd | width | params | epochs | val AUC | val R@FPR2% |", "|---|---|---|---|---|---|---|"]
    for mp in sorted(out.glob("runs/*/metrics.json")):
        m = json.loads(mp.read_text())
        lines.append(f"| {m['name']} | {m['kd']} | {m['width']} | {m['params']} | {m['epochs_run']} | {m['val']['auc']:.4f} | {m['val']['recall_at_fpr2']:.3f} |")
    sel = out / "selection.json"
    if sel.exists():
        lines += ["", f"selected: `{json.loads(sel.read_text())['name']}`"]
    thr = out / "deployed" / "threshold.json"
    if thr.exists():
        lines += ["", "deployed/threshold.json (calibrated on the calib split):", "```json", thr.read_text().strip(), "```"]
    return "\n".join(lines) + "\n"


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="v5 training")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--kd", action="store_true")
    r.add_argument("--width", type=float, default=1.0)
    r.add_argument("--epochs", type=int, default=None)
    r.add_argument("--name", required=True)
    sub.add_parser("select")
    fin = sub.add_parser("final")
    fin.add_argument("--run", default=None, help="deploy this run instead of the selected one (for imported fine-tuned candidates); still verified and calibrated")
    sub.add_parser("report")
    imp = sub.add_parser("import-candidate", help="evaluate a fine-tuned candidate on the current validation split and register it as a run")
    imp.add_argument("--model", required=True)
    imp.add_argument("--name", required=True)
    for p in sub.choices.values():
        p.add_argument("--config", default=None)
    args = ap.parse_args(argv)
    cfg = resolve(load_config(args.config))
    out = Path(cfg["paths"]["out_dir"])
    if args.cmd == "run":
        m = train_one(cfg, args.kd, args.width, args.epochs or int(cfg["train"]["epochs"]), args.name, cfg["seed"])
        print(json.dumps({k: m[k] for k in ("name", "params", "epochs_run", "val")}, indent=1))
    elif args.cmd == "select":
        manifest_sha = file_sha256(out / "manifest.csv")
        results = {r["name"]: r for r in current_runs(out, manifest_sha)}
        problems = []
        for name, (kd, width) in REQUIRED_RUNS.items():
            r = results.get(name)
            if r is None:
                problems.append(f"{name}: missing or stale")
            elif (bool(r["kd"]), float(r["width"])) != (kd, width) or (kd and not r.get("kd_alpha", 0) < 1.0):
                problems.append(f"{name}: recorded kd={r['kd']} width={r['width']} but the matrix expects kd={kd} width={width}")
        val_hashes = {results[n]["val_ids_sha256"] for n in REQUIRED_RUNS if n in results}
        if len(val_hashes) > 1:
            problems.append("runs were validated on different validation splits")
        if problems:
            raise SystemExit("selection needs the exact four-run matrix on the current manifest: " + "; ".join(problems))
        chosen = select_config([results[n] for n in REQUIRED_RUNS])
        (out / "selection.json").write_text(json.dumps({"name": chosen["name"], "kd": chosen["kd"], "width": chosen["width"], "manifest_sha256": manifest_sha, "model_sha256": chosen["model_sha256"]}, indent=2))
        print("selected:", chosen["name"])
    elif args.cmd == "import-candidate":
        rows = M.read_manifest(out / "manifest.csv")
        audit_status = require_audited_manifest(out, cfg)
        _, audio, _ = M.load_cache(out)
        va = M.split_indices(rows, "val")
        model = keras.models.load_model(args.model, compile=False)
        run_dir = out / "runs" / args.name
        run_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(args.model, run_dir / "model.keras")
        yv = np.array([rows[i]["label"] for i in va])
        metrics = {"name": args.name, "kd": False, "kd_alpha": 1.0, "kd_note": "imported candidate", "width": float("nan"), "params": int(model.count_params()),
                   "manifest_sha256": file_sha256(out / "manifest.csv"), "val_ids_sha256": hashlib.sha256(np.asarray(va, dtype=np.int64).tobytes()).hexdigest(),
                   "model_sha256": file_sha256(run_dir / "model.keras"), "audit": audit_status, "imported_from": str(args.model),
                   "n_train": 0, "n_val": int(len(va)), "val": clip_metrics(yv, predict_probs(model, precompute_features(audio[va])), 0.5)}
        (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        print(f"imported {args.model} as run {args.name}: val AUC {metrics['val']['auc']:.4f}; deploy with `python -m v5.train final --run {args.name}`")
    elif args.cmd == "final":
        check_model_version(cfg["model_version"])  # never deploy a version the cloud would treat as simulated
        rows = M.read_manifest(out / "manifest.csv")
        manifest_sha = file_sha256(out / "manifest.csv")
        if args.run:
            run_dir = out / "runs" / args.run
            run_metrics = json.loads((run_dir / "metrics.json").read_text())
            sel = {"name": args.run, "manifest_sha256": manifest_sha, "model_sha256": run_metrics.get("model_sha256"), "manual": True}
            (out / "selection.json").write_text(json.dumps(sel, indent=2))
        else:
            sel = json.loads((out / "selection.json").read_text())
            run_dir = out / "runs" / sel["name"]
            run_metrics = json.loads((run_dir / "metrics.json").read_text())
        if sel.get("manifest_sha256") != manifest_sha or run_metrics.get("manifest_sha256") != manifest_sha:
            raise SystemExit("selection or run metrics belong to another manifest; re-run the training matrix and `select`")
        if file_sha256(run_dir / "model.keras") != sel.get("model_sha256"):
            raise SystemExit(f"{run_dir / 'model.keras'} does not match the hash recorded at selection; re-run `select`")
        _, audio, _ = M.load_cache(out)
        model = keras.models.load_model(run_dir / "model.keras", compile=False)
        va = M.split_indices(rows, "val")
        recomputed = clip_metrics(np.array([rows[i]["label"] for i in va]), predict_probs(model, precompute_features(audio[va])), 0.5)
        if abs(recomputed["auc"] - run_metrics["val"]["auc"]) > 1e-3:
            raise SystemExit(f"validation AUC recomputed as {recomputed['auc']:.4f} but {run_metrics['val']['auc']:.4f} was recorded; the model is not the one that was selected")
        ca = M.split_indices(rows, "calib")
        pc = predict_probs(model, precompute_features(audio[ca]))
        yc = np.array([rows[i]["label"] for i in ca])
        tau, info = choose_threshold(yc, pc, float(cfg["threshold"]["max_fpr"]), int(cfg["threshold"]["min_calib_neg"]))
        if info["fpr"] > float(cfg["threshold"]["max_fpr"]):
            raise CalibrationError(f"measured calibration FPR {info['fpr']:.4f} exceeds {cfg['threshold']['max_fpr']}")
        run_metrics["calib"] = info
        deploy_generation(out, run_dir, run_metrics, {"tau": tau, "model_version": cfg["model_version"], "max_fpr": cfg["threshold"]["max_fpr"], "manifest_sha256": manifest_sha,
                                                      "val_ids_sha256": run_metrics.get("val_ids_sha256"), "run": sel["name"], "calib": info, "fsm": cfg["fsm"]})
        print(f"deployed {sel['name']} -> {out / 'deployed'} with tau={tau:.4f} calib_fpr={info['fpr']:.4f} (95% upper {info['fpr_upper95']:.4f}) calib_recall={info['recall']:.3f}")
    elif args.cmd == "report":
        (out / "deliverables").mkdir(parents=True, exist_ok=True)
        text = run_report(out)
        (out / "deliverables" / "experiments.md").write_text(text, encoding="utf-8")
        print(text)


if __name__ == "__main__":
    main()
