import numpy as np
import pytest

from v5 import features as F


def _sine(freq=100.0, amp=0.1, seed=0, floor=1e-3):
    """Tone plus a -60 dBFS noise floor: realistic bedroom levels, well above the 1e-10 power floor."""
    rng = np.random.default_rng(seed)
    t = np.arange(F.WIN) / F.SR
    return (amp * np.sin(2 * np.pi * freq * t) + floor * rng.standard_normal(F.WIN)).astype(np.float32)


def test_constants_match_spec():
    assert (F.SR, F.WIN, F.STREAM_HOP, F.N_FFT, F.HOP) == (16000, 16000, 8000, 512, 256)
    assert F.N_FRAMES == 61 and F.N_MELS == 30 and F.N_BINS == 257
    assert F.FMIN == 40.0 and F.FMAX == 6000.0 and F.NORM_DIV == 40.0 and F.LOG_EPS == 1e-10


def test_hann_is_periodic():
    w = F.hann_periodic(8)
    assert w[0] == 0.0 and np.isclose(w[4], 1.0)
    assert not np.isclose(w[-1], 0.0)  # periodic form does not end at zero


def test_filterbank_shape_and_coverage():
    fb = F.mel_filterbank()
    assert fb.shape == (30, 257)
    assert (fb.max(axis=1) > 0.5).all()  # every filter has real support
    assert fb[:, :1].sum() == 0  # DC bin unused (fmin 40 Hz)
    sparse = F.sparse_filterbank()
    dense = np.zeros_like(fb)
    for i, (start, w) in enumerate(sparse):
        dense[i, start:start + len(w)] = w
    assert np.allclose(dense, fb)


def test_extract_shape_range_and_gain_invariance():
    x = _sine()
    X = F.extract(x)
    assert X.shape == (61, 30) and X.dtype == np.float32
    assert X.min() >= -1.0 and X.max() <= 1.0
    assert np.allclose(F.extract(0.5 * x), X, atol=5e-3)  # gain-invariant for audio well above the log floor
    assert np.allclose(F.extract(2.0 * x), X, atol=5e-3)


def test_silence_is_all_zero():
    assert np.all(F.extract(np.zeros(F.WIN, np.float32)) == 0)


def test_sine_energy_lands_in_low_mels():
    X = F.extract(_sine(100.0))
    assert X.mean(axis=0)[:3].mean() > X.mean(axis=0)[15:].mean()


def test_int16_roundtrip_and_quantize():
    x = _sine()
    xi = F.float_to_int16(x)
    assert xi.dtype == np.int16
    X = F.extract_int16(xi)
    assert np.allclose(X, F.extract(x), atol=1e-2)  # int16 rounding perturbs only the quietest cells
    q = F.quantize(X, scale=1 / 127, zero_point=0)
    assert q.dtype == np.int8 and q.shape == (61, 30)
    assert q.max() <= 127 and q.min() >= -128


def test_feature_spec_dict_is_complete():
    d = F.feature_spec_dict()
    for k in ["version", "sample_rate", "window", "n_fft", "hop", "n_frames", "n_mels", "fmin", "fmax", "log_eps", "norm_div", "window_type", "normalisation"]:
        assert k in d


def test_extract_rejects_wrong_length():
    with pytest.raises(ValueError):
        F.extract(np.zeros(100, np.float32))
