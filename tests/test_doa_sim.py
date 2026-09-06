import numpy as np
import pytest

from v5 import doa_sim as DS
from v5.doa import DoaParams


def _snore(seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(16000) / 16000
    env = 0.5 * (1 + np.sin(2 * np.pi * 0.5 * t))
    return (0.1 * env * sum(np.sin(2 * np.pi * 110 * k * t) / k for k in range(1, 12)) + 1e-3 * rng.standard_normal(16000)).astype(np.float32)


@pytest.mark.slow
def test_simulated_sides_are_recovered():
    rng = np.random.default_rng(0)
    for az, expected in ((45.0, "right"), (-45.0, "left")):
        l, r, exp = DS.simulate_stereo(rng, _snore(), 0.06, 1.0, az, 20.0)
        assert exp == expected and l.shape == r.shape == (16000,)
        assert DS.classify(l, r, DoaParams(spacing_m=0.06))["side"] == expected


def test_markdown_table():
    rows = [{"spacing_m": 0.06, "distance_m": 1.0, "snr_db": 5, "trials": 2, "accuracy": 1.0, "unknown_rate": 0.0}]
    md = DS.to_markdown(rows)
    assert "| 0.06 | 1.0 | 5 | 1.000 | 0.000 |" in md
