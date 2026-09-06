import numpy as np

from v5 import benchmark_nights as B
from v5 import features as F
from v5.config import load_config, resolve
from v5.data import manifest as M
from v5.streaming import FsmParams

TINY_MIN = {"train_pos": 1, "train_neg": 1, "val_pos": 0, "val_neg": 0, "calib_neg": 0, "test_pos": 1, "test_neg": 1}


def test_window_features_count():
    audio = np.zeros(F.SR * 10, np.float32)
    X = B.window_features(audio)
    assert X.shape == (19, 61, 30, 1)  # 1 + (160000 - 16000) / 8000


def _oracle_probs(episodes, n_ticks, tick_s=0.5):
    p = np.full(n_ticks, 0.05)
    for s, e in episodes:
        t = s
        while t + 1.0 <= e:  # bursts of 1 s every 4 s
            k = int(t / tick_s)
            p[k: k + 2] = 0.95
            t += 4.0
    return p


def test_run_fsm_and_score_on_oracle_probabilities():
    episodes = [(60.0, 100.0), (200.0, 260.0)]
    events, actives = B.run_fsm(_oracle_probs(episodes, 800), FsmParams())
    m = B.score_night(events, actives, episodes, tick_s=0.5, duration_s=400.0)
    assert m["n_episodes"] == 2 and m["detected"] == 2 and m["detection_rate"] == 1.0
    assert 9.0 <= m["confirm_latency_mean_s"] <= 14.0
    assert m["false_confirms"] == 0 and m["false_confirms_per_hour"] == 0.0
    assert 3.0 <= m["stop_latency_mean_s"] <= 8.0  # last burst ends 3 s before the labelled end, hold is 6 s
    assert m["end_event_delay_mean_s"] > m["stop_latency_mean_s"]


def test_match_events_is_one_to_one_with_one_tolerance():
    episodes = [(120.0, 150.0)]
    early = {"type": "episode_start", "episode_id": 3, "tick": 228, "t_end": 116.0}   # inside s - tol
    second = {"type": "episode_start", "episode_id": 4, "tick": 276, "t_end": 140.0}  # same episode, already matched
    far = {"type": "episode_start", "episode_id": 1, "tick": 98, "t_end": 51.0}
    matches, unmatched = B.match_events([far, second, early], episodes, tol_s=5.0)
    assert [j for j, _ in matches] == [0] and matches[0][1] is early
    assert unmatched == [far, second]
    m = B.score_night([far, second, early], [False] * 400, episodes, duration_s=100.0)
    assert m["detected"] == 1 and m["false_confirms"] == 2 and m["false_confirms_per_hour"] == 72.0
    assert m["confirm_latency_mean_s"] == 0.0  # early event clamps to zero latency
    assert m["stop_latency_mean_s"] == 0.0  # active already low: clamped, not negative
    assert np.isnan(m["end_event_delay_mean_s"])  # no episode_end with id 3


def test_run_benchmark_with_fake_predictors(mini_dataset, tmp_path):
    cfg = resolve(load_config())
    cfg["paths"]["data_dir"], cfg["paths"]["out_dir"] = str(mini_dataset), str(tmp_path / "out")
    cfg["data"]["near_dup_threshold"] = 0.999
    cfg["data"]["min_counts"] = dict(TINY_MIN)
    cfg["benchmark"] = {"n_nights": 2, "night_s": 90, "snrs": [10]}
    rows = M.build_manifest(cfg["paths"]["data_dir"], cfg["paths"]["out_dir"], cfg)
    assert any(r["split"] == "bench" and r["label"] == 1 for r in rows)
    beds = B.bench_bed_paths(rows)
    assert beds and all("MS-SNSD" in p for p in beds)
    assert all(r["source"] == "mssnsd" and r["label"] == 0 for r in rows if r["path"] in beds)
    zeros = lambda X: np.zeros(len(X))
    ones = lambda X: np.full(len(X), 0.9)
    result = B.run_benchmark({"float": zeros, "int8": ones}, FsmParams(), cfg, seed=0)
    assert set(result["by_snr"]) == {"float", "int8"} and result["by_snr"]["float"]["10"]["nights"] == 2
    assert result["by_snr"]["float"]["10"]["false_confirms_per_hour"] == 0.0
    assert result["agreement"]["float~int8"]["tick_decision_agreement"] == 0.0
    assert "| 10 |" in B.to_markdown(result)
