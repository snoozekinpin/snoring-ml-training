import hashlib
import json

import numpy as np
import pytest

from v5 import export as X
from v5 import features as F
from v5.doa import DoaParams
from v5.model import build_model
from v5.streaming import FsmParams


def test_headers_only_writes_all_files(tmp_path):
    X.write_headers_only(tmp_path / "gen", tmp_path / "golden", FsmParams(), DoaParams())
    spec = (tmp_path / "gen" / "feature_spec.h").read_text()
    assert "#define SF_N_FRAMES 61" in spec and "#define SF_NORM_DIV 40.0f" in spec
    fb = (tmp_path / "gen" / "mel_filterbank.h").read_text()
    assert "sf_mel_start[30]" in fb and "sf_mel_w[" in fb
    spec_json = json.loads((tmp_path / "gen" / "feature_spec.json").read_text())
    assert spec_json["n_frames"] == 61 and spec_json["norm_div"] == 40.0 and f"#define SF_N_MELS {spec_json['n_mels']}" in spec
    assert (tmp_path / "golden" / "doa_cases.bin").stat().st_size == 12 + 11 * (512 * 2 * 2 + 8)
    trace = (tmp_path / "golden" / "fsm_trace.txt").read_text().splitlines()
    assert trace[0].split()[0] == "0.65" and len(trace) > 300
    rows = [line.split() for line in trace[1:]]
    assert all(len(r) == 9 for r in rows)
    ends = [r for r in rows if r[3] == "2"]
    assert ends and float(ends[0][4]) > 0 and int(ends[0][7]) > 0
    assert (tmp_path / "golden" / "doa_tracker.bin").stat().st_size == 12 + 40 * (512 * 2 * 2 + 8) + 20


def test_fsm_trace_avoids_exact_threshold():
    assert all(abs(v - 0.65) > 1e-9 for v in X.fsm_trace_sequence())


def test_tflite_conversion_validation_and_headers(tmp_path):
    model = build_model(0.5)
    rng = np.random.default_rng(0)
    rep = rng.uniform(-1, 1, (64, F.N_FRAMES, F.N_MELS, 1)).astype(np.float32)
    tflite = X.to_tflite_int8(model, rep)
    info = X.validate_tflite(tflite)
    assert info["input_shape"] == [1, 61, 30, 1] and info["input_scale"] > 0
    if info["ops_checked"]:
        assert set(info["ops"]) <= X.ALLOWED_TFLITE_OPS
    from v5.evaluate import int8_probs, predict_probs

    Xt = rng.uniform(-1, 1, (16, F.N_FRAMES, F.N_MELS, 1)).astype(np.float32)
    assert np.abs(predict_probs(model, Xt) - int8_probs(tflite, Xt)).max() < 0.1
    X.write_c_array(tflite, "snore_v5_int8_tflite", tmp_path / "model_data.h", tmp_path / "model_data.c")
    assert "extern const unsigned char snore_v5_int8_tflite[]" in (tmp_path / "model_data.h").read_text()
    assert f"snore_v5_int8_tflite_len = {len(tflite)}" in (tmp_path / "model_data.c").read_text()
    X.write_model_meta_h(tmp_path / "model_meta.h", info, 0.7, FsmParams(tau=0.7), "cnn_v5_int8")
    meta = (tmp_path / "model_meta.h").read_text()
    assert "#define SNORE_THRESHOLD 0.699999988f" in meta and "#define FSM_CONFIRM_TICKS 20" in meta and '"cnn_v5_int8"' in meta  # float32(0.7) printed exactly
    printed = [ln for ln in meta.splitlines() if "SNORE_THRESHOLD" in ln][0].split()[-1].rstrip("f")
    assert np.float32(printed) == np.float32(0.7)  # the header round-trips the float32 threshold
    est = X.arena_estimate(model, len(tflite))
    assert est["activation_bytes_estimate"] > 10_000 and est["flash_bytes"] == len(tflite)


def test_gates_raise_export_error(monkeypatch):
    with pytest.raises(X.ExportError):
        X.check_parity({"passed": False, "delta_auc": 0.1, "agreement": 0.5, "max_abs_diff": 0.4})
    with pytest.raises(X.ExportError, match="gate failed"):
        X.check_parity({"passed": True, "delta_auc": 0.1, "agreement": 0.5, "max_abs_diff": 0.4})  # a stored flag is never trusted
    with pytest.raises(X.ExportError, match="incomplete"):
        X.check_parity({"passed": True})
    X.check_parity({"delta_auc": 0.001, "agreement": 0.995, "max_abs_diff": 0.03})
    with pytest.raises(X.ExportError):
        X.validate_tflite(b"not a model")
    model = build_model(0.5)
    tflite = X.to_tflite_int8(model, np.random.default_rng(0).uniform(-1, 1, (8, F.N_FRAMES, F.N_MELS, 1)).astype(np.float32))
    monkeypatch.setattr(X, "tflite_ops", lambda _b: None)  # operator enumeration failure must fail closed
    with pytest.raises(X.ExportError, match="operator set"):
        X.validate_tflite(tflite)
    with pytest.raises(X.ExportError, match="capacit"):
        X.check_fsm_capacity(FsmParams(hold_ticks=64))
    with pytest.raises(X.ExportError, match="capacit"):
        X.check_fsm_capacity(FsmParams(confirm_ticks=60))


def test_model_and_threshold_must_be_bound_to_the_current_manifest():
    X.check_manifest_binding({"manifest_sha256": "m1"}, {"manifest_sha256": "m1"}, "m1")
    with pytest.raises(X.ExportError, match="threshold.json"):
        X.check_manifest_binding({"manifest_sha256": "m0"}, {"manifest_sha256": "m1"}, "m1")  # calibrated on an older manifest
    with pytest.raises(X.ExportError, match="metrics.json"):
        X.check_manifest_binding({"manifest_sha256": "m1"}, {}, "m1")  # model of unknown provenance


def test_bound_report_compares_structured_parameters(tmp_path):
    (tmp_path / "deliverables").mkdir()
    fsm = {"hold_ticks": 12, "confirm_ticks": 20}
    render = lambda r: f"value {r['value']}\n"  # the card re-renders the markdown from the JSON it accepts
    (tmp_path / "r.json").write_text(json.dumps({"value": 1, "bound_to": {"manifest_sha256": "m1", "fsm": fsm, "trials": 50}}))
    (tmp_path / "deliverables" / "r.md").write_text("value 1\n")
    keys = ("manifest_sha256", "fsm", "trials")
    assert X._require_bound_report(tmp_path, "r.json", "r.md", {"manifest_sha256": "m1", "fsm": dict(fsm), "trials": 50}, keys, render)["value"] == 1
    (tmp_path / "r.json").write_text(json.dumps({"value": 2, "bound_to": {"manifest_sha256": "m1", "fsm": fsm, "trials": 50}}))  # results changed, markdown not
    with pytest.raises(X.ExportError, match="not the markdown rendered"):
        X._require_bound_report(tmp_path, "r.json", "r.md", {"manifest_sha256": "m1", "fsm": dict(fsm), "trials": 50}, keys, render)
    (tmp_path / "r.json").write_text(json.dumps({"value": 1, "bound_to": {"manifest_sha256": "m1", "fsm": fsm, "trials": 50}}))
    (tmp_path / "deliverables" / "r.md").write_text("value 1 (edited)\n")
    with pytest.raises(X.ExportError, match="not the markdown rendered"):
        X._require_bound_report(tmp_path, "r.json", "r.md", {"manifest_sha256": "m1", "fsm": dict(fsm), "trials": 50}, keys, render)
    (tmp_path / "deliverables" / "r.md").write_text("value 1\n")
    with pytest.raises(X.ExportError, match="different release"):
        X._require_bound_report(tmp_path, "r.json", "r.md", {"manifest_sha256": "m1", "fsm": {**fsm, "hold_ticks": 8}, "trials": 50}, keys, render)
    with pytest.raises(X.ExportError, match="different release"):
        X._require_bound_report(tmp_path, "r.json", "r.md", {"manifest_sha256": "m1", "fsm": fsm, "trials": 1}, keys, render)


def test_benchmark_coverage_is_validated():
    import copy

    from v5.benchmark_nights import _aggregate

    bench = {"n_nights": 4, "night_s": 3600.0, "snrs_db": [0.0, 10.0], "seed": 0}
    nights = [{"night": k, "snr_db": [0, 10][k % 2], "detection_rate": 0.25 * k, "confirm_latency_mean_s": 10.0 + k, "false_confirms_per_hour": 0.1 * k,
               "stop_latency_mean_s": float("nan") if k == 3 else 2.0} for k in range(4)]
    ok = {"by_snr": {p: _aggregate(nights, [0, 10]) for p in ("float", "int8")}, "nights": {p: [dict(n) for n in nights] for p in ("float", "int8")},
          "agreement": {"float~int8": {"tick_decision_agreement": 1.0}}}
    ok = json.loads(json.dumps(ok))  # exactly what the card reads back (NaN survives the round trip)
    X.check_benchmark_coverage(ok, bench)
    for field in ("detection_rate", "confirm_latency_mean_s", "false_confirms_per_hour", "stop_latency_mean_s"):
        forged = copy.deepcopy(ok)
        forged["by_snr"]["int8"]["10"][field] = 0.999  # aggregate edited, per-night records untouched
        with pytest.raises(X.ExportError, match=f"aggregate {field}"):
            X.check_benchmark_coverage(forged, bench)
    only_float = copy.deepcopy(ok); del only_float["by_snr"]["int8"]; del only_float["nights"]["int8"]
    with pytest.raises(X.ExportError, match="exactly the predictors"):
        X.check_benchmark_coverage(only_float, bench)  # the deployed int8 path is missing
    no_agree = copy.deepcopy(ok); del no_agree["agreement"]
    with pytest.raises(X.ExportError, match="agreement"):
        X.check_benchmark_coverage(no_agree, bench)
    dup = copy.deepcopy(ok); dup["nights"]["int8"][3] = dict(dup["nights"]["int8"][1])  # a night counted twice
    with pytest.raises(X.ExportError, match="distinct nights"):
        X.check_benchmark_coverage(dup, bench)
    wrong_snr = copy.deepcopy(ok); wrong_snr["nights"]["float"][0]["snr_db"] = 5
    with pytest.raises(X.ExportError, match="distinct nights"):
        X.check_benchmark_coverage(wrong_snr, bench)  # a night outside the approved SNRs
    bad_agg = copy.deepcopy(ok); bad_agg["by_snr"]["float"]["0"]["nights"] = 3
    with pytest.raises(X.ExportError, match="do not match the per-night"):
        X.check_benchmark_coverage(bad_agg, bench)
    short = copy.deepcopy(ok); short["nights"]["float"] = nights[:2]
    with pytest.raises(X.ExportError, match="distinct nights"):
        X.check_benchmark_coverage(short, bench)


def test_doa_coverage_is_validated():
    import copy

    sweep = {"trials": 2, "seed": 0, "spacings_m": [0.04, 0.06], "distances_m": [0.5], "snrs_db": [0, 10], "n_snore_windows": 3}
    rows = [{"spacing_m": s, "distance_m": 0.5, "snr_db": n, "trials": 2, "accuracy": 1.0, "unknown_rate": 0.0} for s in (0.04, 0.06) for n in (0, 10)]
    X.check_doa_coverage({"rows": rows}, sweep)
    with pytest.raises(X.ExportError, match="cells"):
        X.check_doa_coverage({"rows": rows[:-1]}, sweep)  # a cell missing
    with pytest.raises(X.ExportError, match="cells"):
        X.check_doa_coverage({"rows": rows + [dict(rows[0])]}, sweep)  # a cell twice
    fewer = copy.deepcopy(rows); fewer[1]["trials"] = 1
    with pytest.raises(X.ExportError, match="trials"):
        X.check_doa_coverage({"rows": fewer}, sweep)
    bad = copy.deepcopy(rows); bad[2]["accuracy"] = float("nan")
    with pytest.raises(X.ExportError, match="invalid accuracy"):
        X.check_doa_coverage({"rows": bad}, sweep)
    with pytest.raises(X.ExportError, match="cells"):
        X.check_doa_coverage({"rows": []}, sweep)


def test_threshold_must_be_bound_to_the_model(tmp_path):
    from v5.train import file_sha256

    a, b = build_model(0.5), build_model(0.5)
    a.save(tmp_path / "a.keras")
    b.save(tmp_path / "b.keras")
    thr = {"tau": 0.6, "model_sha256": file_sha256(tmp_path / "a.keras")}
    assert X.check_threshold_binding(tmp_path / "a.keras", thr) == thr["model_sha256"]
    with pytest.raises(X.ExportError):
        X.check_threshold_binding(tmp_path / "b.keras", thr)
    with pytest.raises(X.ExportError):
        X.check_threshold_binding(tmp_path / "a.keras", {"tau": 0.6})
    for bad in ("mock_v5", "cnn_demo", "simulator"):
        with pytest.raises(X.ExportError):
            X.precheck_release(tmp_path / "a.keras", {**thr, "model_version": bad})
    assert X.precheck_release(tmp_path / "a.keras", {**thr, "model_version": "cnn_v5_int8"}) == thr["model_sha256"]


def _stage(tmp_path, tag):
    stage = tmp_path / f"stage_{tag}"
    (stage / "golden").mkdir(parents=True)
    for name in X.PROMOTED_FILES:
        (stage / name).write_text(f"{name}:{tag}")  # includes export_status.json, which travels inside the swap
    (stage / "golden" / "features.bin").write_bytes(tag.encode())
    return stage


def _tree(d):
    return {str(p.relative_to(d)): p.read_bytes() for p in sorted(d.rglob("*")) if p.is_file()}


def test_test_split_is_consumed_once_per_manifest(tmp_path):
    assert X.claim_test_split(tmp_path, "m1", "modelA", 0.8) is None  # fresh claim
    with pytest.raises(X.ExportError):
        X.claim_test_split(tmp_path, "m1", "modelA", 0.8)  # interrupted attempt: record exists but no bundle -> locked
    bundle = X.persist_test_results(tmp_path, {"tflite_sha256": "t", "parity": {"passed": True}, "test_metrics": {}, "robustness": {}, "n_test": 8}, b"tflite-bytes", {"features.bin": b"g"})
    assert (bundle / "snore_v5_int8.tflite").read_bytes() == b"tflite-bytes" and (bundle / "golden" / "features.bin").read_bytes() == b"g"
    record = json.loads((tmp_path / "test_consumption.json").read_text())
    assert record["tau"] == 0.8 and set(record["bundle"]) == {"snore_v5_int8.tflite", "results.json", "golden/features.bin"}
    again = X.claim_test_split(tmp_path, "m1", "modelA", 0.8)  # same model, same tau: the completed bundle, no test access needed
    assert again["parity"]["passed"] is True and again["bundle_dir"] == str(bundle) and again["tflite_sha256"] == hashlib.sha256(b"tflite-bytes").hexdigest() and again["tau"] == 0.8
    with pytest.raises(X.ExportError, match="tau"):
        X.claim_test_split(tmp_path, "m1", "modelA", 0.75)  # re-calibrated threshold: the bundle's metrics no longer apply
    with pytest.raises(X.ExportError):
        X.persist_test_results(tmp_path, {"parity": {"passed": True}}, b"other", {})  # a completed bundle is immutable
    results_path = bundle / "results.json"
    original = results_path.read_text()
    results_path.write_text(original.replace('"passed": true', '"passed": true, "note": "edited"'))
    with pytest.raises(X.ExportError, match="does not match the hashes"):
        X.claim_test_split(tmp_path, "m1", "modelA", 0.8)  # edited results are refused
    results_path.write_text(original)
    (bundle / "golden" / "features.bin").write_bytes(b"h")
    with pytest.raises(X.ExportError, match="does not match the hashes"):
        X.claim_test_split(tmp_path, "m1", "modelA", 0.8)  # edited golden vectors are refused
    (bundle / "golden" / "features.bin").write_bytes(b"g")
    (bundle / "snore_v5_int8.tflite").write_bytes(b"tampered-but-structurally-plausible")
    with pytest.raises(X.ExportError):
        X.claim_test_split(tmp_path, "m1", "modelA", 0.8)  # modified bundle: stored results are not trusted
    with pytest.raises(X.ExportError):
        X.claim_test_split(tmp_path, "m1", "modelB", 0.8)  # another model on the same manifest
    with pytest.raises(X.ExportError):
        X.claim_test_split(tmp_path, "m2", "modelA", 0.8)  # a rebuilt manifest does not grant a fresh evaluation either
    (tmp_path / "fresh").mkdir()
    assert X.claim_test_split(tmp_path / "fresh", "m2", "modelB", 0.8) is None  # only a deliberately removed record allows a new one


def test_fresh_evaluation_needs_a_documented_reset(tmp_path):
    (tmp_path / "deliverables").mkdir()
    (tmp_path / "deliverables" / "export_info.json").write_text(json.dumps({"exported_at": "2026-09-05T10:00:00"}))  # an earlier release
    with pytest.raises(X.ExportError, match="no reset entry"):
        X.claim_test_split(tmp_path, "m1", "modelB", 0.8)  # record deleted by hand, nothing documented
    with pytest.raises(X.ExportError, match="reason"):
        X.reset_test_record(tmp_path, "oops")
    (tmp_path / "test_evaluation" / "old").mkdir(parents=True)
    ledger = X.reset_test_record(tmp_path, "model retrained after an augmentation fix; first evaluation superseded")
    assert " reset: model retrained" in ledger.read_text() and not (tmp_path / "test_evaluation").exists()
    assert X.claim_test_split(tmp_path, "m1", "modelB", 0.8) is None  # documented: a fresh evaluation may start
    (tmp_path / "test_consumption.json").unlink()
    (tmp_path / "test_evaluation" / "orphan").mkdir(parents=True)
    with pytest.raises(X.ExportError, match="without a record"):
        X.claim_test_split(tmp_path, "m1", "modelB", 0.8)  # bundles left behind are evidence too


def test_promote_drops_reports_bound_to_the_previous_release(tmp_path):
    live, new, stage = tmp_path / "deliverables", tmp_path / "deliverables.new", tmp_path / "stage"
    live.mkdir(); stage.mkdir(); (stage / "golden").mkdir()
    (live / "model_card.md").write_text("old card"); (live / "benchmark_nights.md").write_text("old bench"); (live / "manifest_report.md").write_text("kept")
    (stage / "export_info.json").write_text("{}")
    X._build_new_tree(live, new, stage, ["export_info.json"], golden=True, drop=X.RELEASE_BOUND_REPORTS)
    assert not (new / "model_card.md").exists() and not (new / "benchmark_nights.md").exists()
    assert (new / "manifest_report.md").read_text() == "kept" and (new / "export_info.json").exists()


def test_promote_keeps_other_files_and_is_transactional(tmp_path, monkeypatch):
    deliv, gen = tmp_path / "deliv", tmp_path / "gen"
    deliv.mkdir()
    (deliv / "experiments.md").write_text("keep me")
    X.promote(_stage(tmp_path, "v1"), deliv, gen)
    assert (deliv / "experiments.md").read_text() == "keep me"
    assert (deliv / "snore_v5_int8.tflite").read_text() == "snore_v5_int8.tflite:v1" and (gen / "model_meta.h").read_text() == "model_meta.h:v1"
    assert (deliv / "golden" / "features.bin").read_bytes() == b"v1" and not (tmp_path / "deliv.new").exists() and not (tmp_path / "deliv.bak").exists()
    before_deliv, before_gen = _tree(deliv), _tree(gen)
    import os as _os

    real_rename, calls = _os.rename, {"n": 0}

    def failing_rename(src, dst):
        calls["n"] += 1
        if calls["n"] == 4:  # fails while swapping the second (generated) directory, after the first swap succeeded
            raise OSError("injected failure")
        return real_rename(src, dst)

    monkeypatch.setattr(X.os, "rename", failing_rename)
    with pytest.raises(OSError):
        X.promote(_stage(tmp_path, "v2"), deliv, gen)
    monkeypatch.setattr(X.os, "rename", real_rename)
    assert _tree(deliv) == before_deliv and _tree(gen) == before_gen  # previous release fully restored
    assert not (tmp_path / "deliv.new").exists() and not (tmp_path / "gen.new").exists() and not (tmp_path / "deliv.bak").exists()


def test_promote_survives_backup_cleanup_failure(tmp_path, monkeypatch):
    deliv, gen = tmp_path / "deliv", tmp_path / "gen"
    X.promote(_stage(tmp_path, "v1"), deliv, gen)
    real_rmtree = X.shutil.rmtree

    def flaky_rmtree(path, *args, **kwargs):
        if str(path).endswith(".bak"):
            raise OSError("injected cleanup failure")
        return real_rmtree(path, *args, **kwargs)

    monkeypatch.setattr(X.shutil, "rmtree", flaky_rmtree)
    X.promote(_stage(tmp_path, "v2"), deliv, gen)  # must not raise: the release is live
    assert (deliv / "snore_v5_int8.tflite").read_text() == "snore_v5_int8.tflite:v2" and (gen / "model_meta.h").read_text() == "model_meta.h:v2"
