import numpy as np
import pytest

from v5 import features as F
from v5.data import augment as A


def _sig(seed=0, level=0.1):
    rng = np.random.default_rng(seed)
    return (level * rng.standard_normal(F.WIN)).astype(np.float32)


def test_mix_noise_hits_requested_snr():
    x, n = _sig(0, 0.1), _sig(1, 0.02)
    for snr in (-5.0, 0.0, 10.0):
        y = A.mix_noise(x, n, snr)
        added = y - x
        got = 20 * np.log10(A.rms(x) / A.rms(added))
        assert abs(got - snr) < 0.05


def test_mix_noise_with_silent_noise_is_identity():
    x = _sig()
    assert np.array_equal(A.mix_noise(x, np.zeros(F.WIN, np.float32), 10.0), x)


@pytest.mark.slow
def test_rir_bank_generation_and_apply(tmp_path):
    bank = A.RirBank.generate(3, seed=0)
    assert len(bank) == 3
    for h in bank.rirs:
        assert h.ndim == 1 and len(h) == 8000 and np.abs(h).max() >= 1.0 - 1e-6  # direct-path peak normalised to 1
    y = A.apply_rir(_sig(), bank.random(np.random.default_rng(0)))
    assert y.shape == (F.WIN,) and y.dtype == np.float32
    bank.save(tmp_path / "rir.npz")
    bank2 = A.RirBank.load_or_generate(tmp_path / "rir.npz", 3, seed=0)
    assert np.array_equal(bank2.rirs[0], bank.rirs[0])


def test_tilt_shift_gain_specaug():
    x = _sig()
    assert np.allclose(A.spectral_tilt(x, 0.0), x, atol=1e-6)
    t = A.spectral_tilt(x, 0.5)
    assert t.shape == x.shape and not np.allclose(t, x)
    s = A.time_shift(x, 100)
    assert np.all(s[:100] == 0) and np.array_equal(s[100:], x[:-100])
    s2 = A.time_shift(x, -100)
    assert np.all(s2[-100:] == 0) and np.array_equal(s2[:-100], x[100:])
    g = A.gain_clip(np.full(F.WIN, 0.9, np.float32), 6.0)
    assert g.max() == 1.0
    X = np.ones((F.N_FRAMES, F.N_MELS), np.float32)
    Xa = A.spec_augment(X, np.random.default_rng(0))
    assert Xa.shape == X.shape and (Xa == 0).any() and (Xa == 1).any()


def test_augmenter_is_seed_deterministic_and_keeps_shape():
    cfg = A.AugmentConfig(p_rir=0.0)
    bank = np.stack([_sig(5, 0.05), _sig(6, 0.05)])
    a1 = A.Augmenter(cfg, bank, None, np.random.default_rng(3))
    a2 = A.Augmenter(cfg, bank, None, np.random.default_rng(3))
    x = _sig()
    y1, y2 = a1.waveform(x, 1), a2.waveform(x, 1)
    assert y1.shape == (F.WIN,) and y1.dtype == np.float32 and np.array_equal(y1, y2)
    X = F.extract(y1)
    assert a1.features(X).shape == X.shape
    e1, e2 = a1.waveform(x, 1, np.random.default_rng([7, 1])), a2.waveform(x, 1, np.random.default_rng([7, 1]))
    assert np.array_equal(e1, e2)  # explicit generators make single draws reproducible


def test_augment_config_from_dict_roundtrip():
    cfg = A.AugmentConfig.from_dict({"p_rir": 0.1, "snr_range": [0, 5], "gain_range": [-3, 3]})
    assert cfg.p_rir == 0.1 and cfg.snr_range == (0.0, 5.0) and cfg.p_noise_pos == 0.7
