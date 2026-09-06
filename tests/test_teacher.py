import os

import numpy as np
import pytest

from v5 import teacher as T


def test_platt_calibration_is_monotone_and_fits_separable_data():
    rng = np.random.default_rng(0)
    y = np.array([1] * 200 + [0] * 200)
    z = np.where(y == 1, rng.normal(1.0, 0.5, 400), rng.normal(-1.0, 0.5, 400))
    a, b = T.platt_fit(z, y)
    assert a > 0
    t = T.platt_apply(np.array([-3.0, 0.0, 3.0]), a, b)
    assert t[0] < t[1] < t[2] and t[0] < 0.1 and t[2] > 0.9


def test_audit_flags():
    z = T.logit(np.array([0.001, 0.9, 0.001, 0.9]))
    y = np.array([1, 1, 0, 0])
    assert T.audit_flags(z, y) == ["weak_positive", "", "", "snore_in_negative"]


def test_scores_roundtrip(tmp_path):
    T.write_scores(tmp_path / "t.csv", [3, 5], np.array([0.25, -1.5]), ["", "weak_positive"], ["aa", "bb"], {"cache_fingerprint": "fp", "teacher": "yamnet"})
    d = T.read_scores(tmp_path / "t.csv")
    assert d[3] == (0.25, "") and d[5] == (-1.5, "weak_positive")
    assert T.read_score_md5s(tmp_path / "t.csv") == {3: "aa", 5: "bb"} and T.read_meta(tmp_path / "t.csv")["cache_fingerprint"] == "fp"


@pytest.mark.network
@pytest.mark.skipif(not os.environ.get("V5_NETWORK_TESTS"), reason="set V5_NETWORK_TESTS=1 to download YAMNet")
def test_yamnet_loads_and_scores():
    model, idx = T.load_yamnet()
    assert model is not None and idx
    z = T.score_windows(model, idx, np.zeros((1, 16000), np.int16))
    assert z.shape == (1,)
