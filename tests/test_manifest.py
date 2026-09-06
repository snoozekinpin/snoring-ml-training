import shutil

import numpy as np
import pytest
import soundfile as sf

from v5 import features as F
from v5.config import ROOT, load_config
from v5.data import manifest as M

TINY_MIN = {"train_pos": 1, "train_neg": 1, "val_pos": 0, "val_neg": 0, "calib_neg": 0, "test_pos": 1, "test_neg": 1, "bench_pos": 0, "bench_neg": 0}


def _rows(n, source="whl_s", cluster=None, split=None, label=None, group=None):
    return [{"id": i, "source": source, "path": f"p{i}", "offset": 0, "label": (label[i] if label else i % 2), "group": (group[i] if group else f"g{i}"), "category": "c",
             "md5": str(i), "dup_cluster": (cluster[i] if cluster else i), "split": (split[i] if split else "train")} for i in range(n)]


def _cfg(threshold=0.999):
    cfg = load_config()
    cfg["data"]["near_dup_threshold"] = threshold  # synthetic clips are similar by construction
    cfg["data"]["min_counts"] = dict(TINY_MIN)
    return cfg


def test_check_dataset_root_requires_every_source(tmp_path, mini_dataset):
    with pytest.raises(M.DatasetMissing):
        M.check_dataset_root(tmp_path / "nowhere")
    M.check_dataset_root(mini_dataset)
    shutil.rmtree(mini_dataset / "whltalent" / "s1.1")
    shutil.rmtree(mini_dataset / "whltalent" / "s3")
    with pytest.raises(M.DatasetMissing, match="whltalent/s"):
        M.check_dataset_root(mini_dataset)


def test_near_dup_clusters_joins_close_pairs():
    rng = np.random.default_rng(0)
    a = rng.standard_normal(F.FEATURE_DIM).astype(np.float32)
    b = a + 1e-3 * rng.standard_normal(F.FEATURE_DIM).astype(np.float32)
    c = rng.standard_normal(F.FEATURE_DIM).astype(np.float32)
    ids = M.near_dup_clusters(np.stack([a, b, c]), thr=0.98, block=2)
    assert ids[0] == ids[1] and ids[0] != ids[2]


def test_resolve_partitions_priority_exact_dups_and_conflicts():
    rows = _rows(8, cluster=[0, 0, 1, 2, 3, 3, 4, 4], label=[1, 1, 0, 1, 1, 0, 0, 0], group=["g0", "g1", "g2", "g3", "g4", "g5", "g6", "g6"])
    split = ["train", "val", "val", "test", "train", "calib", "val", "calib"]
    out, dropped, prov = M.resolve_partitions(rows, split)
    # cluster 0: train beats val; cluster 3: conflicting labels -> both dropped; group g6 spans val and calib -> val wins
    assert out == ["train", "drop", "val", "test", "drop", "drop", "val", "drop"]
    assert dropped == {("partition", "whl_s"): 2, ("conflict", "whl_s"): 2}
    assert {(e["id"], e["reason"]) for e in prov} == {(1, "partition"), (4, "conflict"), (5, "conflict"), (7, "partition")}
    # exact duplicates: same md5 in train and test -> the test copy survives; same md5 twice in train -> one survives
    rows2 = _rows(5, cluster=[0, 1, 2, 3, 4], label=[1, 1, 0, 0, 1])
    for i, h in enumerate(["a", "a", "b", "b", "c"]):
        rows2[i]["md5"] = h
    out2, dropped2, prov2 = M.resolve_partitions(rows2, ["train", "test", "train", "train", "val"])
    assert out2 == ["drop", "test", "train", "drop", "val"] and dropped2 == {("exact_dup", "whl_s"): 2}
    assert {e["id"] for e in prov2} == {0, 3} and all(e["reason"] == "exact_dup" for e in prov2)
    # exact duplicates with conflicting labels are dropped everywhere
    rows3 = _rows(2, cluster=[0, 1], label=[1, 0])
    rows3[0]["md5"] = rows3[1]["md5"] = "z"
    out3, _, prov3 = M.resolve_partitions(rows3, ["train", "test"])
    assert out3 == ["drop", "drop"] and all(e["reason"] == "exact_conflict" for e in prov3)


def test_check_invariants_rejects_any_shared_partition():
    for a, b in (("train", "val"), ("val", "calib"), ("calib", "test"), ("train", "bench"), ("bench", "test")):
        rows = _rows(2, split=[a, b])
        rows[1]["group"] = rows[0]["group"]
        with pytest.raises(AssertionError):
            M.check_invariants(rows)
        rows = _rows(2, split=[a, b], cluster=[7, 7])
        with pytest.raises(AssertionError):
            M.check_invariants(rows)
    M.check_invariants(_rows(2, split=["val", "sanity"], cluster=[7, 7]))  # sanity is outside the partitions


def test_check_min_counts():
    rows = _rows(4, split=["train", "train", "test", "test"], label=[1, 0, 1, 0])
    M.check_min_counts(rows, TINY_MIN)
    with pytest.raises(M.DatasetMissing):
        M.check_min_counts(rows, {**TINY_MIN, "train_pos": 2})


def test_split_indices_guards_the_test_split():
    rows = _rows(2, split=["train", "test"])
    assert list(M.split_indices(rows, "train")) == [0]
    with pytest.raises(M.TestSplitAccess):
        M.split_indices(rows, "test")
    assert list(M.split_indices(rows, "test", allow_test=True)) == [1]


def test_only_the_exporter_reads_the_test_split():
    users = sorted(p.name for p in (ROOT / "v5").rglob("*.py") if "allow_test=True" in p.read_text(encoding="utf-8"))
    assert users == ["export.py"]  # the exporter is the sole reader of the test split


def test_build_manifest_end_to_end(mini_dataset, tmp_path):
    rows = M.build_manifest(mini_dataset, tmp_path / "out", _cfg(), seed=42)
    out = tmp_path / "out"
    assert (out / "manifest.csv").exists() and (out / "manifest_report.md").exists() and (out / "cache" / "meta.json").exists()
    kept = [r for r in rows if r["split"] != "drop"]
    assert len({r["md5"] for r in kept}) == len(kept)  # exactly one copy of every waveform survives
    dups = M.load_exact_dups(out)
    assert len(dups) == 1 and dups[0]["source"] == "kaggle_jibran" and dups[0]["kept_source"] == "kaggle_adria"
    assert "kaggle_adria ~ kaggle_jibran" in (out / "manifest_report.md").read_text()
    kaggle = [r for r in rows if r["source"].startswith("kaggle")]
    assert kaggle and all(r["split"] in ("test", "train", "drop") for r in kaggle)
    prov = {e["id"]: e for e in M.load_drop_provenance(out)}
    assert all(r["id"] in prov and prov[r["id"]]["reason"] in ("exact_dup", "exact_conflict", "conflict") for r in kaggle if r["split"] == "drop")
    for h in {r["md5"] for r in kaggle if r["md5"] not in set(M.load_exact_conflicts(out))}:
        assert sum(1 for r in kaggle if r["md5"] == h and r["split"] in ("test", "train")) == 1  # one surviving copy per waveform
    assert all(r["split"] == "sanity" for r in rows if r["source"] == "wild")
    by_batch = {}
    for r in rows:
        if r["source"] in ("whl_s", "whl_e"):
            by_batch.setdefault(r["category"], set()).add(r["split"])
    assert by_batch["000000"] == {"train"} and by_batch["000002"] == {"val", "bench"} and by_batch["100002"] <= {"val", "bench"}
    esc = {r["group"]: r["split"] for r in rows if r["source"] == "esc50"}
    assert esc["esc50_fold5"] == "calib" and esc["esc50_fold4"] == "val" and esc["esc50_fold1"] == "train"
    bench_files = {r["group"] for r in rows if r["split"] == "bench" and r["source"] == "mssnsd"}
    assert len(bench_files) == 1
    M.check_invariants(rows)
    assert M.read_manifest(out / "manifest.csv") == rows
    audio, feats = M.load_cache(out)[1:]
    assert audio.shape == (len(rows), F.WIN) and feats.shape == (len(rows), F.FEATURE_DIM)
    tr = M.split_indices(rows, "train")
    assert len(tr) > 0 and all(rows[i]["split"] == "train" for i in tr)


def test_kaggle_clusters_split_as_units(mini_dataset, tmp_path):
    rows = M.build_manifest(mini_dataset, tmp_path / "out", _cfg())
    kaggle = [r for r in rows if r["source"].startswith("kaggle") and r["split"] != "drop"]
    assert {r["split"] for r in kaggle} == {"test", "train"}  # half of the clusters train, half immutable test
    by_cluster = {}
    for r in kaggle:
        by_cluster.setdefault(r["dup_cluster"], set()).add(r["split"])
    assert all(len(v) == 1 for v in by_cluster.values())  # a near-duplicate cluster never straddles the two halves


def test_apply_audit_rejects_stale_teacher_scores(mini_dataset, tmp_path):
    import json
    from v5 import teacher as T

    out = tmp_path / "out"
    rows = M.build_manifest(mini_dataset, out, _cfg())
    ids = [r["id"] for r in rows if r["split"] in ("train", "val", "calib", "bench")]
    z = np.full(len(ids), -1.0)  # P=0.27: keeps positives (>=0.1) and negatives (<=0.5)
    T.write_scores(out / "teacher.csv", ids, z, [""] * len(ids), [rows[i]["md5"] for i in ids], {"cache_fingerprint": "not-this-cache"})
    with pytest.raises(M.DatasetMissing, match="fingerprint"):
        M.build_manifest(mini_dataset, out, _cfg(), apply_teacher_audit=True)
    fp = json.loads((out / "cache" / "meta.json").read_text())["fingerprint"]
    meta = {"cache_fingerprint": fp, "teacher": T.YAMNET_URL, "classes": list(T.TARGET_CLASSES), "pos_threshold": 0.1, "neg_threshold": 0.5, "n_scored": len(ids)}
    T.write_scores(out / "teacher.csv", ids, z, [""] * len(ids), ["wrong"] * len(ids), meta)
    with pytest.raises(M.DatasetMissing, match="audio changed"):
        M.build_manifest(mini_dataset, out, _cfg(), apply_teacher_audit=True)
    T.write_scores(out / "teacher.csv", ids, z, [""] * len(ids), [rows[i]["md5"] for i in ids], {**meta, "pos_threshold": 0.3})
    with pytest.raises(M.DatasetMissing, match="audit in force"):
        M.build_manifest(mini_dataset, out, _cfg(), apply_teacher_audit=True)  # scored under other thresholds
    T.write_scores(out / "teacher.csv", ids, z, [""] * len(ids), [rows[i]["md5"] for i in ids], meta)
    rows2 = M.build_manifest(mini_dataset, out, _cfg(), apply_teacher_audit=True)  # consistent scores: audit applies
    status = json.loads((out / "audit_status.json").read_text())
    assert status["applied"] is True and len(rows2) == len(rows)
    assert status["teacher_csv_sha256"] == M._file_sha256(out / "teacher.csv") and status["teacher_meta_sha256"] == M._file_sha256(out / "teacher_meta.json")
    pre = M.read_manifest(out / "manifest_unaudited.csv")  # the pre-audit assignment is kept for re-scoring
    assert [r["id"] for r in pre] == [r["id"] for r in rows] and [r["split"] for r in pre] == [r["split"] for r in rows]


def test_apply_audit_drops_unverified_positives_and_contaminated_negatives():
    rows = _rows(6, split=["train", "val", "train", "bench", "test", "train"], label=[1, 1, 0, 0, 1, 1])
    z = lambda p: float(np.log(p / (1 - p)))
    scores = {0: (z(0.01), ""), 1: (z(0.6), ""), 2: (z(0.9), ""), 3: (z(0.01), ""), 4: (z(0.01), ""), 5: (z(0.2), "")}
    dropped, prov = M.apply_audit(rows, scores)
    assert [r["split"] for r in rows] == ["drop", "val", "drop", "bench", "test", "train"]  # test rows are never audited
    assert {(e["id"], e["reason"]) for e in prov} == {(0, "unverified_positive"), (2, "snore_in_negative")}
    assert sum(dropped.values()) == 2


def test_exact_duplicate_keeps_the_test_copy_over_a_training_copy(mini_dataset, tmp_path):
    clip = sf.read(mini_dataset / "adrianagaler" / "noise" / "adria_n_0000.wav", dtype="float32")[0]
    tail = 0.05 * np.random.default_rng(11).standard_normal(19 * F.SR).astype(np.float32)
    sf.write(mini_dataset / "RAW" / "MS-SNSD" / "noise_train" / "Typing_9.wav", np.concatenate([clip, tail]), F.SR, subtype="PCM_16")  # first second == the Kaggle clip
    cfg = _cfg()
    cfg["data"]["kaggle_train_frac"] = 0.0  # every Kaggle cluster on the test side for this check
    rows = M.build_manifest(mini_dataset, tmp_path / "out", cfg)
    twins = [r for r in rows if r["path"].endswith("adria_n_0000.wav") or (r["path"].endswith("Typing_9.wav") and r["offset"] == 0)]
    assert len(twins) == 2 and len({r["md5"] for r in twins}) == 1
    assert {r["split"] for r in twins if r["source"] == "kaggle_adria"} == {"test"}
    assert {r["split"] for r in twins if r["source"] == "mssnsd"} == {"drop"}
    assert any(d["source"] == "mssnsd" and d["kept_source"] == "kaggle_adria" for d in M.load_exact_dups(tmp_path / "out"))


def test_exact_duplicate_with_conflicting_label_is_dropped(mini_dataset, tmp_path):
    src = sf.read(mini_dataset / "adrianagaler" / "snore" / "adria_s_0001.wav", dtype="float32")[0]
    sf.write(mini_dataset / "snoring_extra" / "jibran" / "jibran_n_0001.wav", src, F.SR, subtype="PCM_16")  # same waveform, labelled noise
    rows = M.build_manifest(mini_dataset, tmp_path / "out", _cfg())
    assert {r["split"] for r in rows if r["path"].endswith(("adria_s_0001.wav", "jibran_n_0001.wav"))} == {"drop"}
    assert M.load_exact_conflicts(tmp_path / "out")
    assert "exact_conflict" in (tmp_path / "out" / "manifest_report.md").read_text()


def test_cache_reuse_validates_fingerprint(mini_dataset, tmp_path):
    rows = M.build_manifest(mini_dataset, tmp_path / "out", _cfg())
    assert M.build_manifest(mini_dataset, tmp_path / "out", _cfg(), reuse_cache=True) == rows
    shutil.rmtree(mini_dataset / "whltalent" / "s3")
    (mini_dataset / "whltalent" / "s9").mkdir()
    sf.write(mini_dataset / "whltalent" / "s9" / "000009-A-0-001.wav", 0.05 * np.random.default_rng(5).standard_normal(10 * F.SR).astype(np.float32), F.SR, subtype="PCM_16")
    rows2 = M.build_manifest(mini_dataset, tmp_path / "out", _cfg(), reuse_cache=True)  # fingerprint changed -> rebuilt
    assert any(r["category"] == "000009" for r in rows2) and not any(r["category"] == "000002" for r in rows2)


def test_cache_reuse_detects_content_and_metadata_changes(mini_dataset, tmp_path):
    fp0 = M.cache_fingerprint(mini_dataset, load_config()["data"])
    target = mini_dataset / "adrianagaler" / "noise" / "adria_n_0000.wav"
    sf.write(target, 0.03 * np.random.default_rng(8).standard_normal(F.SR).astype(np.float32), F.SR, subtype="PCM_16")  # same count, new content
    fp1 = M.cache_fingerprint(mini_dataset, load_config()["data"])
    assert fp1 != fp0
    meta = mini_dataset / "esc50" / "meta" / "esc50.csv"
    meta.write_text(meta.read_text().replace("rain,False", "rain,True"))  # metadata-only change
    assert M.cache_fingerprint(mini_dataset, load_config()["data"]) != fp1


def test_failed_rebuild_keeps_previous_generation(mini_dataset, tmp_path):
    out = tmp_path / "out"
    rows = M.build_manifest(mini_dataset, out, _cfg())
    snapshot = {p.relative_to(out): p.read_bytes() for p in out.rglob("*") if p.is_file()}
    cfg = _cfg()
    cfg["data"]["min_counts"]["train_pos"] = 10_000
    with pytest.raises(M.DatasetMissing):
        M.build_manifest(mini_dataset, out, cfg)  # full rebuild that fails validation
    assert {p.relative_to(out): p.read_bytes() for p in out.rglob("*") if p.is_file() and "build" not in p.parts} == snapshot
    assert M.read_manifest(out / "manifest.csv") == rows and not (out / "build").exists()


def test_required_source_failures(mini_dataset, tmp_path):
    for f in (mini_dataset / "esc50" / "audio").glob("*.wav"):
        f.write_bytes(b"corrupt")
    with pytest.raises(M.DatasetMissing, match="esc50"):
        M.build_manifest(mini_dataset, tmp_path / "out", _cfg())
    for f in (mini_dataset / "RAW" / "MS-SNSD" / "noise_train").glob("*.wav"):
        f.unlink()
    with pytest.raises(M.DatasetMissing, match="MS-SNSD"):
        M.check_dataset_root(mini_dataset)


def test_build_manifest_fails_on_min_counts(mini_dataset, tmp_path):
    cfg = _cfg()
    cfg["data"]["min_counts"]["train_pos"] = 10_000
    with pytest.raises(M.DatasetMissing):
        M.build_manifest(mini_dataset, tmp_path / "out", cfg)
