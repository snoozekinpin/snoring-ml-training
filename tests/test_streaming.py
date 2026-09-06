import numpy as np

from v5 import streaming as S


def periodic_pattern(n_cycles=6, burst=3, gap=5, p_hit=0.9, p_miss=0.1):
    seq = []
    for _ in range(n_cycles):
        seq += [p_hit] * burst + [p_miss] * gap
    return seq


def test_params_from_config_rounds_seconds_to_ticks():
    p = S.FsmParams.from_config(0.7, {"tick_ms": 500, "hold_s": 6, "confirm_s": 10, "verify_s": 15, "min_bursts": 3, "period_min_s": 1.5, "period_max_s": 7.0})
    assert p == S.FsmParams(tau=0.7, tick_ms=500, hold_ticks=12, confirm_ticks=20, verify_ticks=30, min_bursts=3, period_min_ticks=3, period_max_ticks=14)


def test_periodic_snoring_confirms_at_tick_19_with_three_bursts():
    events, states, actives = S.run_sequence(periodic_pattern(), S.FsmParams())
    starts = [e for e in events if e["type"] == "episode_start"]
    assert len(starts) == 1 and starts[0]["tick"] == 19 and starts[0]["n_bursts"] == 3 and starts[0]["episode_id"] == 1
    assert states[18] == S.ACTIVE and states[19] == S.CONFIRMED


def test_episode_end_active_flag_and_statistics():
    seq = periodic_pattern(5) + [0.1] * 80  # last hit at tick 34 (burst starting at 32)
    levels = [-30.0] * len(seq)
    events, states, actives = S.run_sequence(seq, S.FsmParams(), levels)
    assert actives[34] and actives[46] and not actives[47]  # hold = 12 ticks after the last hit
    ends = [e for e in events if e["type"] == "episode_end"]
    assert len(ends) == 1 and ends[0]["tick"] == 76 and ends[0]["episode_id"] == 1  # 47 + 30 - 1
    assert ends[0]["duration_s"] == (34 - 0 + 2) * 0.5 and ends[0]["n_bursts"] == 5
    assert ends[0]["n_hits"] == 15  # 5 bursts x 3 hits, including the hits before confirmation
    assert abs(ends[0]["mean_p"] - 0.9) < 1e-9 and ends[0]["level_dbfs"] == -30.0 and states[76] == S.IDLE


def test_pre_confirmation_hits_shape_the_mean():
    seq = [0.7] * 3 + [0.1] * 5 + [0.7] * 3 + [0.1] * 5 + [0.95] * 3 + [0.1] * 5 + [0.95] * 3 + [0.1] * 80
    events, _, _ = S.run_sequence(seq, S.FsmParams())
    end = [e for e in events if e["type"] == "episode_end"][0]
    assert end["n_hits"] == 12 and abs(end["mean_p"] - (6 * 0.7 + 6 * 0.95) / 12) < 1e-9


def test_continuous_sound_never_confirms_and_returns_to_idle():
    seq = [0.9] * 60 + [0.1] * 200
    events, states, _ = S.run_sequence(seq, S.FsmParams())
    assert events == [] and S.CONFIRMED not in states
    assert states[59] == S.ACTIVE and states[-1] == S.IDLE


def test_too_slow_or_too_fast_periodicity_is_rejected():
    slow = []
    for _ in range(4):
        slow += [0.9] * 2 + [0.1] * 30  # 16 s between bursts: gap breaks the hold, no confirm
    fast = []
    for _ in range(30):
        fast += [0.9] * 1 + [0.1] * 1  # 1 s period, below period_min
    for seq in (slow, fast):
        events, _, _ = S.run_sequence(seq, S.FsmParams())
        assert events == []


def test_reset_clears_state():
    f = S.EpisodeFsm(S.FsmParams())
    for p in periodic_pattern():
        f.tick(p)
    f.reset()
    assert f.state == S.IDLE and f.tick_i == 0 and not f.active
