import numpy as np
from scipy.signal import butter, sosfilt

from v5 import doa as D


def _burst(seed=0, n=512):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(n + 64)
    sos = butter(4, [100, 2000], btype="band", fs=16000, output="sos")
    return sosfilt(sos, x)[64:].astype(np.float32) * 0.1


def _delayed(x, d):
    """Return x delayed by d samples (fractional via FFT phase)."""
    n = len(x)
    f = np.fft.rfftfreq(n, 1 / 16000)
    return np.fft.irfft(np.fft.rfft(x) * np.exp(-2j * np.pi * f * d / 16000), n=n).astype(np.float32)


def test_gcc_phat_sign_and_integer_lag():
    l = _burst()
    for d in (2, -2):
        lag, ratio = D.gcc_phat(l, _delayed(l, d), max_lag=4)
        assert abs(lag - d) < 0.3 and ratio > 1.5


def test_gcc_phat_fractional_lag():
    l = _burst(1)
    lag, _ = D.gcc_phat(l, _delayed(l, 1.5), max_lag=4)
    assert abs(lag - 1.5) < 0.35


def test_quiet_frames_are_invalid():
    t = D.DoaTracker(D.DoaParams())
    quiet = 1e-5 * np.random.default_rng(0).standard_normal(512).astype(np.float32)
    assert not t.frame(quiet, quiet)[1]  # below the noise floor margin


def test_uncorrelated_frames_are_mostly_invalid():
    t = D.DoaTracker(D.DoaParams())
    valid = [t.frame(_burst(2 * k + 100), _burst(2 * k + 101))[1] for k in range(30)]
    assert np.mean(valid) <= 0.5


def test_tracker_episode_side_and_confidence():
    p = D.DoaParams(spacing_m=0.06)
    assert p.max_lag == 4
    t = D.DoaTracker(p)
    l = _burst(4)
    for k in range(10):
        t.frame(l, _delayed(l, 2.0))
    ep = t.episode()
    assert ep["side"] == "left" and ep["conf"] == 1.0 and ep["n_valid"] == 10 and abs(ep["lag_samples"] - 2.0) < 0.3
    t.reset()
    for k in range(10):
        t.frame(_delayed(l, 2.0), l)
    assert t.episode()["side"] == "right"


def test_tracker_reset_clears_votes_but_keeps_the_noise_floor():
    t = D.DoaTracker(D.DoaParams(spacing_m=0.06))
    l = _burst(4)
    for _ in range(10):
        t.frame(l, _delayed(l, 2.0))
    t.floor.db = floor = -45.0  # an adapted floor (coherent frames never raise it, so set one explicitly)
    assert t.n_valid == 10
    t.reset()
    assert t.n_valid == 0 and int(t.hist.sum()) == 0 and sum(t.side_counts.values()) == 0 and t.floor.db == floor  # same rule as doa_tracker_reset_episode


def test_noise_floor_adapts_up_slowly_and_down_fast_but_not_for_coherent_frames():
    nf = D.NoiseFloor()
    for _ in range(2000):
        nf.update(-40.0, coherent=True)
    assert nf.db == -60.0  # a loud coherent source never raises the floor
    for _ in range(2000):
        nf.update(-40.0)
    assert nf.db > -41.0  # loud incoherent background converges up to the steady level
    for _ in range(200):
        nf.update(-70.0)
    assert nf.db < -68.0  # fast decay back down


def test_tracker_handles_long_episodes_with_a_side_change():
    t = D.DoaTracker(D.DoaParams(spacing_m=0.06))
    l = _burst(5)
    left, right = (l, _delayed(l, 2.0)), (_delayed(l, 2.0), l)
    for _ in range(3000):
        t.frame(*left)
    for _ in range(2500):
        t.frame(*right)
    ep = t.episode()
    assert ep["n_valid"] == 5500 and ep["side"] == "left" and abs(ep["lag_samples"] - 2.0) < 0.3
    assert abs(ep["conf"] - 3000 / 5500) < 1e-9


def test_out_of_band_signal_is_never_valid():
    t = np.arange(512) / 16000
    tone = (0.3 * np.sin(2 * np.pi * 5000.0 * t)).astype(np.float32)  # loud, but entirely above the 60-3000 Hz band
    lag, ratio = D.gcc_phat(tone, np.roll(tone, 2), max_lag=4)
    assert ratio == 0.0
    tr = D.DoaTracker(D.DoaParams())
    assert not tr.frame(tone, np.roll(tone, 2))[1]


def test_side_from_lag_deadzone():
    assert D.side_from_lag(0.5, 4) == "unknown" and D.side_from_lag(1.0, 4) == "left" and D.side_from_lag(-1.0, 4) == "right"


def test_both_sides_rule():
    left = {"side": "left", "conf": 0.9}
    right = {"side": "right", "conf": 0.9}
    weak = {"side": "right", "conf": 0.3}
    assert D.both_sides_rule([left, left, right, right])
    assert D.both_sides_rule([left, left, left, right])  # right share exactly 0.25 meets min_share
    assert not D.both_sides_rule([left] * 4 + [right])  # share 0.2 is below min_share
    assert not D.both_sides_rule([left, left, weak, weak])
    assert not D.both_sides_rule([left])
