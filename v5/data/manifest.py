"""Manifest: decode every window once, dedup by content, immutable isolated split (spec 4.3-4.4)."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from v5 import features as F
from v5.config import load_config, resolve
from v5.data import sources as S

COLUMNS = ["id", "source", "path", "offset", "label", "group", "category", "md5", "dup_cluster", "split"]
INT_COLS = {"id", "offset", "label", "dup_cluster"}
SOURCE_PRIORITY = ["whl_s", "whl_e", "esc50", "mssnsd", "kaggle_adria", "kaggle_jibran", "wild"]
REQUIRED_SOURCES = ("whl_s", "whl_e", "esc50", "mssnsd")
PARTITIONS = ("test", "train", "val", "calib", "bench")  # priority order for isolation conflicts
EVAL_SPLITS = ("val", "calib", "test", "bench")
SPLITS = ("train",) + EVAL_SPLITS + ("sanity", "drop")
LEAKAGE_STATEMENT = (
    "Kaggle (adrianagaler + jibran, near-duplicate twins) is split by near-duplicate cluster into a training half and an immutable test half; "
    "the test half is read only by the exporter. WHLTalent batches 000000/100000/100001 are training material for negatives only "
    "(the teacher hears snoring in under 10 % of their 'snore' windows, see the audit section); batches 000002/100002 are the "
    "cross-collection validation and bench material (recording-disjoint between val and bench). ESC-50 uses its official folds. "
    "No subject metadata exists for any source, so splits are collection/batch/recording-disjoint, not proven subject-disjoint."
)
AUDIT_POS_THRESHOLD = 0.1   # teacher P(snoring+snort) below this -> positive is unverified and dropped
AUDIT_NEG_THRESHOLD = 0.5   # teacher P above this -> negative is contaminated and dropped


class DatasetMissing(RuntimeError):
    """The dataset root, a required source, or a minimum window count is missing (spec section 13)."""


class TestSplitAccess(RuntimeError):
    """The test split may only be read by v5/export.py."""


GENERATION_ITEMS = ("cache", "manifest.csv", "manifest_unaudited.csv", "manifest_report.md", "manifest_errors.csv", "exact_dups.json", "exact_conflicts.json", "drop_provenance.json", "audit_status.json")


def promote_generation(out_dir, build_dir) -> None:
    """Swap the validated items of build_dir into out_dir; on failure the previous generation is restored."""
    out_dir, build_dir = Path(out_dir), Path(build_dir)
    bak = out_dir / "manifest_prev"
    if bak.exists():
        shutil.rmtree(bak)
    bak.mkdir(parents=True)
    installed = []  # (name, had_backup): backup-less installs are removed on rollback too
    try:
        for name in GENERATION_ITEMS:
            new, live = build_dir / name, out_dir / name
            if not new.exists():
                continue  # e.g. a reused cache stays live
            had_backup = live.exists()
            if had_backup:
                os.rename(live, bak / name)
            installed.append((name, had_backup))
            os.rename(new, live)
    except Exception:
        for name, had_backup in reversed(installed):
            live = out_dir / name
            if live.exists():
                shutil.rmtree(live) if live.is_dir() else live.unlink()
            if had_backup and (bak / name).exists():
                os.rename(bak / name, live)
        raise
    shutil.rmtree(bak, ignore_errors=True)
    shutil.rmtree(build_dir, ignore_errors=True)


def _required_paths(data_dir: Path) -> dict:
    return {
        "whltalent/s* (whl_s)": sorted(p for p in (data_dir / "whltalent").glob("s*") if p.is_dir()),
        "whltalent/e* (whl_e)": sorted(p for p in (data_dir / "whltalent").glob("e*") if p.is_dir()),
        "esc50/audio": [data_dir / "esc50" / "audio"] if (data_dir / "esc50" / "audio").is_dir() else [],
        "esc50/meta/esc50.csv": [data_dir / "esc50" / "meta" / "esc50.csv"] if (data_dir / "esc50" / "meta" / "esc50.csv").exists() else [],
        "RAW/MS-SNSD/noise_train/*.wav": sorted((data_dir / "RAW" / "MS-SNSD" / "noise_train").glob("*.wav")),
    }


def check_dataset_root(data_dir) -> None:
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise DatasetMissing(f"dataset root {data_dir} does not exist")
    missing = [k for k, v in _required_paths(data_dir).items() if not v]
    if missing:
        raise DatasetMissing(f"dataset root {data_dir} is incomplete; missing: {', '.join(missing)}")


SOURCE_DIRS = ("whltalent", "esc50/audio", "RAW/MS-SNSD/noise_train", "adrianagaler", "snoring_extra/jibran", "RAW/Snore_Detection_Project/Snore_Detection/inference_audios")


def _file_sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def cache_fingerprint(data_dir, data_cfg: dict) -> str:
    """Changes whenever any source file (path, size, mtime), the ESC-50 metadata or the slicing config changes."""
    data_dir = Path(data_dir).resolve()
    h = hashlib.sha256(json.dumps({"data_dir": str(data_dir), "feature_spec": F.FEATURE_SPEC_VERSION,
                                   "slicing": {k: v for k, v in data_cfg.items() if k.endswith("_windows") or k.endswith("_stride_s")}}, sort_keys=True).encode("utf-8"))
    for sub_dir in SOURCE_DIRS:
        for p in sorted((data_dir / sub_dir).rglob("*.wav")):
            st = p.stat()
            h.update(f"{p.relative_to(data_dir)}|{st.st_size}|{st.st_mtime_ns}\n".encode("utf-8"))
    meta = data_dir / "esc50" / "meta" / "esc50.csv"
    if meta.exists():
        h.update(meta.read_bytes())
    return h.hexdigest()


def build_cache(data_dir, out_dir, cfg: dict, seed: int = 42):
    check_dataset_root(data_dir)
    out_dir = Path(out_dir)
    (out_dir / "cache").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    rows, audio, feats, errors = [], [], [], []
    per_source = Counter()
    for src in SOURCE_PRIORITY:  # exact duplicates are kept here and resolved after split assignment
        try:
            for w, x in S.iter_windows(data_dir, src, rng, cfg.get("data", {}), errors):
                xi = F.float_to_int16(x)
                h = hashlib.md5(xi.tobytes()).hexdigest()
                per_source[src] += 1
                rows.append({"id": len(rows), "source": w.source, "path": w.path, "offset": w.offset, "label": w.label,
                             "group": w.group, "category": w.category, "md5": h, "dup_cluster": -1, "split": ""})
                audio.append(xi)
                feats.append(F.extract_int16(xi).ravel())
        except FileNotFoundError as exc:
            if src in REQUIRED_SOURCES:
                raise DatasetMissing(f"required source {src} unreadable: {exc}") from exc
            errors.append({"path": src, "error": f"optional source missing: {exc}"})
    empty = [src for src in REQUIRED_SOURCES if per_source[src] == 0]
    if empty:
        raise DatasetMissing(f"required sources yielded no decoded windows: {empty} (see manifest_errors.csv)")
    audio = np.stack(audio)
    feats = np.stack(feats).astype(np.float32)
    np.save(out_dir / "cache" / "audio_i16.npy", audio)
    np.save(out_dir / "cache" / "feats.npy", feats)
    (out_dir / "cache" / "rows.json").write_text(json.dumps(rows), encoding="utf-8")
    (out_dir / "cache" / "meta.json").write_text(json.dumps({"fingerprint": cache_fingerprint(data_dir, cfg.get("data", {})), "windows": len(rows), "per_source": dict(per_source)}), encoding="utf-8")
    with open(out_dir / "manifest_errors.csv", "w", newline="", encoding="utf-8") as fh:  # out_dir is the build directory here
        wr = csv.DictWriter(fh, fieldnames=["path", "error"])
        wr.writeheader()
        wr.writerows(errors)
    return rows, audio, feats


def load_cache(out_dir):
    out_dir = Path(out_dir)
    rows = json.loads((out_dir / "cache" / "rows.json").read_text(encoding="utf-8"))
    return rows, np.load(out_dir / "cache" / "audio_i16.npy"), np.load(out_dir / "cache" / "feats.npy")


def load_exact_dups(out_dir) -> list[dict]:
    p = Path(out_dir) / "exact_dups.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else []


def load_exact_conflicts(out_dir) -> list[str]:
    p = Path(out_dir) / "exact_conflicts.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else []


def load_drop_provenance(out_dir) -> list[dict]:
    p = Path(out_dir) / "drop_provenance.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else []


def near_dup_clusters(feats, thr: float = 0.98, block: int = 1024) -> np.ndarray:
    fm = np.asarray(feats, dtype=np.float32)
    fm = fm - fm.mean(axis=1, keepdims=True)
    fm = fm / (np.linalg.norm(fm, axis=1, keepdims=True) + 1e-9)
    n = len(fm)
    parent = np.arange(n)

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i0 in range(0, n, block):
        sim = fm[i0: i0 + block] @ fm.T
        for r in range(sim.shape[0]):
            i = i0 + r
            for j in np.nonzero(sim[r, i + 1:] >= thr)[0] + i + 1:
                ra, rb = find(i), find(int(j))
                if ra != rb:
                    parent[max(ra, rb)] = min(ra, rb)
    roots = np.array([find(i) for i in range(n)])
    return np.unique(roots, return_inverse=True)[1].astype(int)


def conflicting_clusters(rows) -> set[int]:
    labels = defaultdict(set)
    for r in rows:
        labels[r["dup_cluster"]].add(r["label"])
    return {c for c, s in labels.items() if len(s) > 1}


def assign_splits(rows, d: dict, seed: int = 42) -> list[str]:
    rng = np.random.default_rng(seed)
    whl_val = {str(b) for b in d.get("whl_val_batches", ["000002", "100002"])}
    esc_val, esc_calib = int(d.get("esc50_val_fold", 4)), int(d.get("esc50_calib_fold", 5))
    # MS-SNSD files: bench, then val, calib, train
    ms_groups = sorted({r["group"] for r in rows if r["source"] == "mssnsd"})
    perm = [ms_groups[i] for i in rng.permutation(len(ms_groups))]
    n_bench = int(round(float(d.get("bench_frac", 0.2)) * len(perm)))
    rest = perm[n_bench:]
    n_val = int(math.ceil(float(d.get("mssnsd_val_frac", 0.15)) * len(rest)))
    n_calib = int(math.ceil(float(d.get("mssnsd_calib_frac", 0.15)) * len(rest)))
    ms_split = {g: "bench" for g in perm[:n_bench]}
    ms_split.update({g: "val" for g in rest[:n_val]})
    ms_split.update({g: "calib" for g in rest[n_val: n_val + n_calib]})
    ms_split.update({g: "train" for g in rest[n_val + n_calib:]})
    # WHLTalent validation batches: recordings split between bench and val
    whl_split = {}
    for batch in sorted(whl_val):
        recs = sorted({r["group"] for r in rows if r["source"] in ("whl_s", "whl_e") and r["category"] == batch})
        order = [recs[i] for i in rng.permutation(len(recs))]
        n_b = int(math.ceil(float(d.get("whl_bench_frac", 0.5)) * len(order))) if order else 0
        whl_split.update({g: "bench" for g in order[:n_b]})
        whl_split.update({g: "val" for g in order[n_b:]})
    # Kaggle: near-duplicate clusters (twins across adrianagaler/jibran) are split as units into test and train
    kg_clusters = sorted({r["dup_cluster"] for r in rows if r["source"] in S.TEST_SOURCES})
    kg_order = [kg_clusters[i] for i in rng.permutation(len(kg_clusters))]
    n_kg_train = int(round(float(d.get("kaggle_train_frac", 0.5)) * len(kg_order)))
    kg_train = set(kg_order[len(kg_order) - n_kg_train:]) if n_kg_train else set()
    split = []
    for r in rows:
        src = r["source"]
        if src == "wild":
            split.append("sanity")
        elif src in S.TEST_SOURCES:
            split.append("train" if r["dup_cluster"] in kg_train else "test")
        elif src in ("whl_s", "whl_e"):
            split.append(whl_split.get(r["group"], "train") if r["category"] in whl_val else "train")
        elif src == "esc50":
            fold = int(r["group"].rsplit("fold", 1)[1])
            split.append("val" if fold == esc_val else "calib" if fold == esc_calib else "train")
        elif src == "mssnsd":
            split.append(ms_split[r["group"]])
        else:
            raise ValueError(f"unknown source {src}")
    return split


def _priority(partitions) -> str:
    return next(p for p in PARTITIONS if p in partitions)


def resolve_partitions(rows, split):
    """Enforce isolation after split assignment (spec 4.3-4.4):
    1. exact duplicates (same md5): conflicting labels -> all copies dropped; otherwise one copy survives
       in the highest-priority partition, the others are dropped;
    2. near-duplicate clusters with conflicting labels are dropped everywhere;
    3. a group or cluster spanning several partitions keeps only its highest-priority partition."""
    out, dropped, prov = list(split), Counter(), []

    def drop(i, reason):
        out[i] = "drop"
        dropped[(reason, rows[i]["source"])] += 1
        prov.append({"id": rows[i]["id"], "source": rows[i]["source"], "path": rows[i]["path"], "reason": reason, "md5": rows[i]["md5"], "dup_cluster": rows[i]["dup_cluster"]})

    by_md5 = defaultdict(list)
    for i, r in enumerate(rows):
        if out[i] in PARTITIONS:
            by_md5[r["md5"]].append(i)
    for idxs in by_md5.values():
        if len(idxs) < 2:
            continue
        if len({rows[i]["label"] for i in idxs}) > 1:
            for i in idxs:
                drop(i, "exact_conflict")
            continue
        winner = _priority({out[i] for i in idxs})
        keep = next(i for i in idxs if out[i] == winner)  # first in source-priority order within the winning partition
        for i in idxs:
            if i != keep:
                drop(i, "exact_dup")
    conflict = conflicting_clusters([r for r, s in zip(rows, out) if s in PARTITIONS])
    for i, r in enumerate(rows):
        if out[i] in PARTITIONS and r["dup_cluster"] in conflict:
            drop(i, "conflict")
    for key in ("group", "dup_cluster"):
        present = defaultdict(set)
        for r, s in zip(rows, out):
            if s in PARTITIONS:
                present[r[key]].add(s)
        winner = {k: _priority(s) for k, s in present.items() if len(s) > 1}
        for i, r in enumerate(rows):
            if out[i] in PARTITIONS and r[key] in winner and out[i] != winner[r[key]]:
                drop(i, "partition")
    return out, dict(dropped), prov


def check_invariants(rows) -> None:
    kept = [r for r in rows if r["split"] in PARTITIONS]
    assert len({r["md5"] for r in kept}) == len(kept), "exact duplicates survive in the partitions"
    for key in ("group", "dup_cluster"):
        present = defaultdict(set)
        for r in rows:
            if r["split"] in PARTITIONS:
                present[r[key]].add(r["split"])
        bad = [k for k, s in present.items() if len(s) > 1]
        assert not bad, f"{key}s present in more than one partition: {bad[:5]}"
    conflict = conflicting_clusters([r for r in rows if r["split"] in PARTITIONS])
    assert not conflict, f"clusters with conflicting labels survive: {sorted(conflict)[:5]}"


def apply_audit(rows, scores: dict, pos_threshold: float = AUDIT_POS_THRESHOLD, neg_threshold: float = AUDIT_NEG_THRESHOLD):
    """Teacher verification (spec 4.5): outside the test split, positives the teacher does not hear as snoring
    and negatives it hears as snoring are dropped. Returns (n_dropped_by_reason, provenance)."""
    dropped, prov = Counter(), []
    for r in rows:
        if r["split"] not in ("train", "val", "calib", "bench") or r["id"] not in scores:
            continue
        z = scores[r["id"]][0]
        p = 1.0 / (1.0 + math.exp(-z))
        reason = None
        if r["label"] == 1 and p < pos_threshold:
            reason = "unverified_positive"
        elif r["label"] == 0 and p > neg_threshold:
            reason = "snore_in_negative"
        if reason:
            r["split"] = "drop"
            dropped[(reason, r["source"], r["category"] if r["source"].startswith("whl") else "")] += 1
            prov.append({"id": r["id"], "source": r["source"], "path": r["path"], "reason": reason, "md5": r["md5"], "dup_cluster": r["dup_cluster"], "teacher_p": round(p, 4)})
    return dict(dropped), prov


def check_min_counts(rows, min_counts: dict) -> None:
    cnt = Counter((r["split"], r["label"]) for r in rows)
    actual = {f"{split}_{'pos' if lab else 'neg'}": cnt[(split, lab)] for split in PARTITIONS for lab in (1, 0)}
    short = {k: (actual.get(k, 0), v) for k, v in min_counts.items() if actual.get(k, 0) < v}
    if short:
        raise DatasetMissing(f"window counts below minimum (actual, required): {short}")


def split_indices(rows, split: str, *, allow_test: bool = False) -> np.ndarray:
    if split == "test" and not allow_test:
        raise TestSplitAccess("the test split is read only by v5/export.py, which passes the allow_test flag")
    return np.array([r["id"] for r in rows if r["split"] == split], dtype=int)


def write_manifest(rows, path) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=COLUMNS)
        wr.writeheader()
        for r in rows:
            wr.writerow({k: r[k] for k in COLUMNS})


def read_manifest(path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as fh:
        return [{k: (int(v) if k in INT_COLS else v) for k, v in row.items()} for row in csv.DictReader(fh)]


def _report(rows, exact_dups, exact_conflicts, dropped, members, path, audit_dropped=None) -> None:
    exact_pairs = Counter(tuple(sorted((d["kept_source"], d["source"]))) for d in exact_dups)
    near_pairs = Counter()
    for srcs in members.values():
        u = sorted(set(srcs))
        for i in range(len(u)):
            for j in range(i + 1, len(u)):
                near_pairs[(u[i], u[j])] += 1
    n_kept = sum(1 for r in rows if r["split"] in PARTITIONS)
    lines = ["# Manifest report", "", f"windows decoded: {len(rows)}; kept in partitions: {n_kept}; exact duplicates dropped: {len(exact_dups)}; exact-duplicate waveforms with conflicting labels: {len(exact_conflicts)}",
             f"near-duplicate clusters with more than one member: {sum(1 for m in members.values() if len(m) > 1)}", "", LEAKAGE_STATEMENT, "",
             "exact duplicates across sources (kept ~ discarded):"]
    lines += [f"- {a} ~ {b}: {n}" for (a, b), n in sorted(exact_pairs.items())] or ["- none"]
    lines += ["", "near-duplicate clusters spanning two sources:"]
    lines += [f"- {a} ~ {b}: {n}" for (a, b), n in sorted(near_pairs.items())] or ["- none"]
    lines += ["", "| source | label | " + " | ".join(SPLITS) + " |", "|---|---|" + "---|" * len(SPLITS)]
    cnt = Counter((r["source"], r["label"], r["split"]) for r in rows)
    for src in SOURCE_PRIORITY:
        for lab in (1, 0):
            vals = [cnt[(src, lab, s)] for s in SPLITS]
            if sum(vals):
                lines.append(f"| {src} | {lab} | " + " | ".join(str(v) for v in vals) + " |")
    lines += ["", f"dropped (reason, source): {dropped}", ""]
    if audit_dropped is not None:
        lines += ["## Teacher audit (YAMNet, applied to train/val/calib/bench only)", "",
                  f"positives dropped as unverified (P(snoring+snort) < {AUDIT_POS_THRESHOLD}) and negatives dropped as contaminated (P > {AUDIT_NEG_THRESHOLD}), by (reason, source, batch):"]
        lines += [f"- {k}: {v}" for k, v in sorted(audit_dropped.items(), key=lambda kv: -kv[1])] or ["- none"]
        lines.append("")
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def _exact_dup_provenance(rows) -> tuple[list[dict], list[str]]:
    by_md5 = defaultdict(list)
    for r in rows:
        if r["split"] != "sanity":
            by_md5[r["md5"]].append(r)
    dups, conflicts = [], []
    for h, group in by_md5.items():
        if len(group) < 2:
            continue
        if len({r["label"] for r in group}) > 1:
            conflicts.append(h)
            continue
        kept = next((r for r in group if r["split"] != "drop"), group[0])
        dups += [{"kept_id": kept["id"], "kept_source": kept["source"], "id": r["id"], "source": r["source"], "path": r["path"], "label": r["label"]} for r in group if r is not kept]
    return dups, sorted(conflicts)


def build_manifest(data_dir, out_dir, cfg: dict, seed: int = 42, reuse_cache: bool = False, apply_teacher_audit: bool = False) -> list[dict]:
    out_dir = Path(out_dir)
    d = cfg.get("data", {})
    check_dataset_root(data_dir)
    build = out_dir / "build"
    if build.exists():
        shutil.rmtree(build)
    build.mkdir(parents=True)
    meta = out_dir / "cache" / "meta.json"
    fresh = reuse_cache and meta.exists() and json.loads(meta.read_text(encoding="utf-8")).get("fingerprint") == cache_fingerprint(data_dir, d)
    if reuse_cache and not fresh:
        print("[manifest] cache fingerprint mismatch or missing: rebuilding")
    try:
        rows, audio, feats = load_cache(out_dir) if fresh else build_cache(data_dir, build, cfg, seed)
        clusters = near_dup_clusters(feats, float(d.get("near_dup_threshold", 0.98)))
        for r, c in zip(rows, clusters):
            r["dup_cluster"] = int(c)
        split, dropped, provenance = resolve_partitions(rows, assign_splits(rows, d, seed))
        for r, s in zip(rows, split):
            r["split"] = s
        # the pre-audit assignment: the teacher scores exactly these non-test windows, so re-running it on an
        # audited generation still covers the windows the audit has to judge
        write_manifest(rows, build / "manifest_unaudited.csv")
        audit_dropped = {}
        teacher_binding = {}
        if apply_teacher_audit:
            teacher_csv = out_dir / "teacher.csv"
            if not teacher_csv.exists():
                raise DatasetMissing(f"{teacher_csv} missing: run python -m v5.teacher before --apply-audit")
            from v5.teacher import read_meta, read_score_md5s, read_scores

            scores = read_scores(teacher_csv)
            meta_fp = json.loads((out_dir / "cache" / "meta.json").read_text(encoding="utf-8")).get("fingerprint") if fresh else cache_fingerprint(data_dir, d)
            tmeta = read_meta(teacher_csv)
            if tmeta.get("cache_fingerprint") != meta_fp:
                raise DatasetMissing("teacher.csv was produced for a different cache (fingerprint mismatch); re-run python -m v5.teacher")
            from v5.teacher import TARGET_CLASSES, YAMNET_URL

            expected = {"teacher": YAMNET_URL, "classes": list(TARGET_CLASSES), "n_scored": len(scores),
                        "pos_threshold": float(d.get("audit_pos_threshold", AUDIT_POS_THRESHOLD)), "neg_threshold": float(d.get("audit_neg_threshold", AUDIT_NEG_THRESHOLD))}
            wrong = {k: (tmeta.get(k), v) for k, v in expected.items() if tmeta.get(k) != v}
            if wrong:
                raise DatasetMissing(f"teacher_meta.json does not describe the audit in force (recorded, expected): {wrong}; re-run python -m v5.teacher")
            teacher_binding = {"teacher_csv_sha256": _file_sha256(teacher_csv), "teacher_meta_sha256": _file_sha256(teacher_csv.with_name("teacher_meta.json")),
                               "teacher_id": YAMNET_URL, "classes": list(TARGET_CLASSES)}
            md5s = read_score_md5s(teacher_csv)
            stale = [r["id"] for r in rows if r["id"] in md5s and md5s[r["id"]] != r["md5"]]
            if stale:
                raise DatasetMissing(f"teacher.csv scores {len(stale)} windows whose audio changed; re-run python -m v5.teacher")
            unscored = [r["id"] for r in rows if r["split"] in ("train", "val", "calib", "bench") and r["id"] not in scores]
            if unscored:
                raise DatasetMissing(f"{len(unscored)} non-test windows have no teacher score; re-run python -m v5.teacher")
            audit_dropped, audit_prov = apply_audit(rows, scores, float(d.get("audit_pos_threshold", AUDIT_POS_THRESHOLD)), float(d.get("audit_neg_threshold", AUDIT_NEG_THRESHOLD)))
            provenance += audit_prov
            dropped.update({(k[0], k[1]): dropped.get((k[0], k[1]), 0) + v for k, v in audit_dropped.items()})
        check_invariants(rows)
        check_min_counts(rows, d.get("min_counts", {}))
        exact_dups, exact_conflicts = _exact_dup_provenance(rows)
        (build / "exact_dups.json").write_text(json.dumps(exact_dups), encoding="utf-8")
        (build / "exact_conflicts.json").write_text(json.dumps(exact_conflicts), encoding="utf-8")
        (build / "drop_provenance.json").write_text(json.dumps(provenance), encoding="utf-8")
        (build / "audit_status.json").write_text(json.dumps({"applied": bool(apply_teacher_audit), "teacher": "yamnet" if apply_teacher_audit else None,
                                                            "pos_threshold": float(d.get("audit_pos_threshold", AUDIT_POS_THRESHOLD)) if apply_teacher_audit else None,
                                                            "neg_threshold": float(d.get("audit_neg_threshold", AUDIT_NEG_THRESHOLD)) if apply_teacher_audit else None,
                                                            "dropped": {f"{k[0]}|{k[1]}|{k[2]}": v for k, v in audit_dropped.items()}, **teacher_binding}, indent=1), encoding="utf-8")
        members = defaultdict(list)
        for r in rows:
            if r["split"] in PARTITIONS:
                members[r["dup_cluster"]].append(r["source"])
        write_manifest(rows, build / "manifest.csv")
        _report(rows, exact_dups, exact_conflicts, dropped, members, build / "manifest_report.md", audit_dropped if apply_teacher_audit else None)
    except Exception:
        shutil.rmtree(build, ignore_errors=True)  # the previous generation in out_dir is untouched
        raise
    promote_generation(out_dir, build)
    return rows


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Build the v5 manifest")
    ap.add_argument("--config", default=None)
    ap.add_argument("--reuse-cache", action="store_true")
    ap.add_argument("--apply-audit", action="store_true", help="drop teacher-unverified positives and contaminated negatives (needs teacher.csv)")
    args = ap.parse_args(argv)
    cfg = resolve(load_config(args.config))
    rows = build_manifest(cfg["paths"]["data_dir"], cfg["paths"]["out_dir"], cfg, cfg["seed"], args.reuse_cache or args.apply_audit, args.apply_audit)
    print(f"manifest rows: {len(rows)} -> {cfg['paths']['out_dir']}/manifest.csv")


if __name__ == "__main__":
    main()
