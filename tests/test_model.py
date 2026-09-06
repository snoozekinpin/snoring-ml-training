import keras
import numpy as np

from v5.model import build_model, check_ops


def test_width_one_is_small_and_tflm_safe():
    m = build_model(1.0)
    assert m.input_shape == (None, 61, 30, 1) and m.output_shape == (None, 1)
    assert 20_000 < m.count_params() < 40_000
    check_ops(m)
    assert m.get_layer("conv1").kernel.shape[-1] == 16 and m.get_layer("fc").units == 32


def test_width_two_is_larger_and_forward_works():
    m = build_model(2.0)
    assert m.count_params() > 80_000
    out = m(np.zeros((3, 61, 30, 1), np.float32), training=False)
    assert out.shape == (3, 1)


def test_save_and_reload(tmp_path):
    m = build_model(0.5)
    m.save(tmp_path / "m.keras")
    m2 = keras.models.load_model(tmp_path / "m.keras")
    x = np.random.default_rng(0).standard_normal((2, 61, 30, 1)).astype(np.float32)
    assert np.allclose(m(x, training=False), m2(x, training=False))
