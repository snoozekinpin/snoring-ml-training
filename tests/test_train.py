import json
from pathlib import Path

import numpy as np
import pytest

from v5 import train as TR
from v5.config import load_config, resolve
from v5.data import manifest as M

TINY_MIN = {"train_pos": 1, "train_neg": 1, "val_pos": 0, "val_neg": 0, "calib_neg": 0, "test_pos": 1, "test_neg": 1}


def test_choose_threshold_meets_fpr_and_reports_bound():
    rng = np.random.default_rng(0)
    y = np.array([1] * 400 + [0] * 400)
    p = np.where(y == 1, rng.uniform(0.4, 1.0, 800), rng.uniform(0.0, 0.6, 800))
    tau, info = TR.choose_threshold(y, p, max_fpr=0.01)
    assert 0.5 <= tau <= 0.62 and info["fp"] <= 4 and info["fpr"] <= 0.01
    assert float(np.float32(tau)) == tau  # exactly representable: the firmware compares float32 values
    neg = np.sort(p[y == 0])
    assert tau > neg[len(neg) - info["fp"] - 1] and (neg >= tau).sum() == info["fp"]  # strictly above the permitted negative, in float32 too
    assert (p[y == 0] >= tau).mean() <= 0.01 and info["fpr_upper95"] > info["fpr"] and 0 < info["recall"] <= 1


def test_choose_threshold_never_clips_to_a_cap():
    y = np.array([1] * 300 + [0] * 300)
    p = np.r_[np.full(300, 0.995), np.full(300, 0.98)]
    tau, info = TR.choose_threshold(y, p, max_fpr=0.01)
    assert 0.98 < tau <= 0.995 and info["fp"] == 0 and info["recall"] == 1.0


def test_choose_threshold_fails_closed():
    y = np.array([1] * 300 + [0] * 300)
    p_ok = np.r_[np.full(300, 0.9), np.full(300, 0.1)]
    with pytest.raises(TR.CalibrationError):  # too few negatives
        TR.choose_threshold(y[:400], p_ok[:400], max_fpr=0.01, min_neg=300)
    with pytest.raises(TR.CalibrationError):  # every negative outscores every positive
        TR.choose_threshold(y, np.r_[np.full(300, 0.2), np.full(300, 0.9)], max_fpr=0.01)
    with pytest.raises(TR.CalibrationError):  # negatives saturate at 1.0
        TR.choose_threshold(y, np.r_[np.full(300, 1.0), np.full(300, 1.0)], max_fpr=0.01)


def test_noise_bank_uses_mssnsd_and_esc50_negatives_only():
    rows = [{"id": i, "label": lab, "source": src} for i, (lab, src) in enumerate([(0, "mssnsd"), (0, "esc50"), (0, "whl_e"), (1, "esc50"), (0, "mssnsd")])]
    assert list(TR.noise_bank_ids(rows, [0, 1, 2, 3, 4])) == [0, 1, 4]


def test_deploy_generation_is_atomic(tmp_path, monkeypatch):
    from v5.model import build_model

    run = tmp_path / "runs" / "r1"
    run.mkdir(parents=True)
    build_model(0.5).save(run / "model.keras")
    live = TR.deploy_generation(tmp_path, run, {"name": "r1"}, {"tau": 0.6, "model_version": "cnn_v5_int8"})
    thr = json.loads((live / "threshold.json").read_text())
    assert thr["model_sha256"] == TR.file_sha256(live / "model.keras") and (live / "metrics.json").exists()
    before = {p.name: p.read_bytes() for p in live.iterdir()}
    run2 = tmp_path / "runs" / "r2"
    run2.mkdir()
    build_model(0.5).save(run2 / "model.keras")
    real_rename, calls = TR.os.rename, {"n": 0}

    def failing_rename(src, dst):
        calls["n"] += 1
        if calls["n"] == 2:  # after the live generation was moved aside
            raise OSError("injected")
        return real_rename(src, dst)

    monkeypatch.setattr(TR.os, "rename", failing_rename)
    with pytest.raises(OSError):
        TR.deploy_generation(tmp_path, run2, {"name": "r2"}, {"tau": 0.7, "model_version": "cnn_v5_int8"})
    monkeypatch.setattr(TR.os, "rename", real_rename)
    assert {p.name: p.read_bytes() for p in live.iterdir()} == before and not (tmp_path / "deployed.new").exists()


def test_file_sha256(tmp_path):
    p = tmp_path / "f.bin"
    p.write_bytes(b"abc")
    assert TR.file_sha256(p) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"


def test_fpr_upper_bound_is_conservative():
    assert TR.fpr_upper_bound(0, 300) > 0 and TR.fpr_upper_bound(0, 300) < 0.011
    assert TR.fpr_upper_bound(3, 300) > 0.01


def test_select_config_prefers_smallest_near_best_on_validation():
    results = [
        {"name": "kd_w2", "params": 100_000, "kd": True, "val": {"auc": 0.990, "recall_at_fpr2": 0.95}},
        {"name": "hard_w1", "params": 25_000, "kd": False, "val": {"auc": 0.987, "recall_at_fpr2": 0.94}},
        {"name": "kd_w1", "params": 25_000, "kd": True, "val": {"auc": 0.988, "recall_at_fpr2": 0.945}},
        {"name": "hard_w0", "params": 8_000, "kd": False, "val": {"auc": 0.970, "recall_at_fpr2": 0.80}},
    ]
    assert TR.select_config(results)["name"] == "hard_w1"
    disagree = [  # best AUC and best recall on different runs: AUC-near-best set wins, ranked by recall
        {"name": "a", "params": 25_000, "kd": False, "val": {"auc": 0.907, "recall_at_fpr2": 0.40}},
        {"name": "b", "params": 100_000, "kd": False, "val": {"auc": 0.895, "recall_at_fpr2": 0.45}},
    ]
    assert TR.select_config(disagree)["name"] == "a"
    fallback = [  # no run meets both criteria: the AUC-near-best run with the highest recall wins, not a smaller one within 2 points
        {"name": "small", "params": 25_000, "kd": False, "val": {"auc": 0.907, "recall_at_fpr2": 0.40}},
        {"name": "big", "params": 100_000, "kd": True, "val": {"auc": 0.905, "recall_at_fpr2": 0.415}},
        {"name": "off", "params": 8_000, "kd": False, "val": {"auc": 0.880, "recall_at_fpr2": 0.60}},
    ]
    assert TR.select_config(fallback)["name"] == "big"
    tie = [{"name": "t_big", "params": 100_000, "kd": True, "val": {"auc": 0.907, "recall_at_fpr2": 0.40}},
           {"name": "t_small", "params": 25_000, "kd": False, "val": {"auc": 0.905, "recall_at_fpr2": 0.40}},
           {"name": "t_off", "params": 8_000, "kd": False, "val": {"auc": 0.880, "recall_at_fpr2": 0.60}}]
    assert TR.select_config(tie)["name"] == "t_small"  # exact recall tie: the smaller model


def test_kd_loss_factory():
    import keras

    y = np.array([[1.0, 1.0], [0.0, 0.0]], np.float32)
    soft = np.array([[1.0, 0.7], [0.0, 0.2]], np.float32)
    logits = np.array([[2.0], [-1.0]], np.float32)
    ref = keras.losses.BinaryCrossentropy(from_logits=True)(y[:, :1], logits)
    assert np.isclose(float(TR.kd_loss(y, logits)), float(ref), atol=1e-6)
    assert np.isclose(float(TR.make_kd_loss(1.0)(soft, logits)), float(ref), atol=1e-6)  # alpha 1 ignores soft targets
    assert not np.isclose(float(TR.make_kd_loss(0.5)(soft, logits)), float(ref), atol=1e-3)


def test_teacher_binding_is_verified_before_training(tmp_path):
    from v5 import teacher as T

    rows = [{"id": 0, "md5": "a0"}, {"id": 1, "md5": "a1"}, {"id": 2, "md5": "a2"}]
    T.write_scores(tmp_path / "teacher.csv", [0, 1], np.array([1.0, -1.0]), ["", ""], ["a0", "a1"], {"cache_fingerprint": "fp"})
    status = {"applied": True, "teacher_csv_sha256": TR.file_sha256(tmp_path / "teacher.csv"), "teacher_meta_sha256": TR.file_sha256(tmp_path / "teacher_meta.json")}
    TR.verify_teacher_binding(tmp_path, rows, status, [0, 1, 2])  # consistent
    TR.verify_teacher_binding(tmp_path, rows, {"applied": False}, [0, 1])  # unaudited runs still need matching audio
    with pytest.raises(TR.TeacherMismatch, match="audio differs"):
        TR.verify_teacher_binding(tmp_path, [{"id": 0, "md5": "changed"}, rows[1]], status, [0, 1])
    T.write_scores(tmp_path / "teacher.csv", [0, 1], np.array([5.0, -5.0]), ["", ""], ["a0", "a1"], {"cache_fingerprint": "fp"})  # regenerated after the audit
    with pytest.raises(TR.TeacherMismatch, match="differs from the file the audit used"):
        TR.verify_teacher_binding(tmp_path, rows, status, [0, 1])
    (tmp_path / "teacher.csv").unlink()
    with pytest.raises(TR.TeacherMismatch, match="missing"):
        TR.verify_teacher_binding(tmp_path, rows, status, [0, 1])  # an audited manifest needs its teacher file
    TR.verify_teacher_binding(tmp_path, rows, {"applied": False}, [0, 1])  # unaudited hard-label run without a teacher: nothing to bind


def test_train_one_smoke(mini_dataset, tmp_path):
    cfg = resolve(load_config())
    cfg["paths"]["data_dir"], cfg["paths"]["out_dir"] = str(mini_dataset), str(tmp_path / "out")
    cfg["data"]["near_dup_threshold"] = 0.999
    cfg["data"]["min_counts"] = dict(TINY_MIN)
    cfg["augment"]["rir_bank_size"] = 2
    cfg["train"]["workers"] = 1
    M.build_manifest(cfg["paths"]["data_dir"], cfg["paths"]["out_dir"], cfg)
    with pytest.raises(TR.AuditRequired):  # an unaudited manifest cannot train unless the degraded mode is explicit
        TR.train_one(cfg, use_kd=False, width=0.5, epochs=1, name="smoke")
    cfg["train"]["allow_unaudited"] = True
    metrics = TR.train_one(cfg, use_kd=False, width=0.5, epochs=1, name="smoke")
    assert metrics["audit"]["applied"] is False and len(metrics["model_sha256"]) == 64 and len(metrics["manifest_sha256"]) == 64
    assert [m["name"] for m in TR.current_runs(Path(cfg["paths"]["out_dir"]), TR.file_sha256(Path(cfg["paths"]["out_dir"]) / "manifest.csv"))] == ["smoke"]
    assert TR.current_runs(Path(cfg["paths"]["out_dir"]), "other-manifest") == []
    assert (tmp_path / "out" / "runs" / "smoke" / "model.keras").exists()
    assert "val" in metrics and "test" not in metrics and metrics["kd"] is False
    saved = json.loads((tmp_path / "out" / "runs" / "smoke" / "metrics.json").read_text())
    assert saved["name"] == "smoke" and saved["n_train"] > 0
