import numpy as np

from v5 import features as F
from v5 import golden as G


def _clips(seed, n, level):
    rng = np.random.default_rng(seed)
    return [F.float_to_int16(level * rng.standard_normal(F.WIN)) for _ in range(n)]


def test_synth_signals_have_noise_floor_and_silence():
    s = G.synth_golden_signals()
    assert set(s) == {"sine100", "white40", "chirp", "silence"}
    assert np.all(s["silence"] == 0)
    assert np.abs(s["sine100"]).max() > 1000  # roughly -20 dBFS
    assert np.count_nonzero(s["sine100"] == 0) < F.WIN // 100  # dithered, not pure


def test_make_golden_is_deterministic_and_roundtrips(tmp_path):
    names, x, X, q = G.make_golden(_clips(1, 4, 0.1), _clips(2, 4, 0.05), scale=1 / 127, zero_point=0)
    names2, x2, X2, q2 = G.make_golden(_clips(1, 4, 0.1), _clips(2, 4, 0.05), scale=1 / 127, zero_point=0)
    assert names == G.GOLDEN_NAMES and x.shape == (12, F.WIN) and X.shape == (12, 61, 30) and q.shape == (12, 61, 30) and q.dtype == np.int8
    assert np.array_equal(x, x2) and np.array_equal(X, X2) and np.array_equal(q, q2)
    assert np.allclose(X[4], F.extract_int16(x[4])) and np.array_equal(q[4], F.quantize(X[4], 1 / 127, 0))
    G.write_golden_npz(tmp_path / "f.npz", names, x, X, q, 1 / 127, 0)
    n3, x3, X3, q3, sc, zp = G.read_golden_npz(tmp_path / "f.npz")
    assert n3 == names and np.array_equal(x3, x) and np.array_equal(X3, X) and np.array_equal(q3, q) and sc == np.float32(1 / 127) and zp == 0
    G.write_golden_bin(tmp_path / "f.bin", x, X, q, 1 / 127, 0)
    x4, X4, q4, sc4, zp4 = G.read_golden_bin(tmp_path / "f.bin")
    assert np.array_equal(x4, x) and np.array_equal(X4, X) and np.array_equal(q4, q) and zp4 == 0 and abs(sc4 - 1 / 127) < 1e-9
    assert (tmp_path / "f.bin").stat().st_size == 20 + 12 * (F.WIN * 2 + F.FEATURE_DIM * 4 + F.FEATURE_DIM)
