import numpy as np
import pytest

from v5 import evaluate as E


def test_clip_metrics_perfect_and_threshold():
    y = np.array([1, 1, 0, 0])
    p = np.array([0.9, 0.8, 0.2, 0.1])
    m = E.clip_metrics(y, p, 0.5)
    assert m["auc"] == 1.0 and m["recall"] == 1.0 and m["fpr"] == 0.0 and m["cm"] == [[2, 0], [0, 2]]
    m2 = E.clip_metrics(y, p, 0.85)
    assert m2["recall"] == 0.5 and m2["fnr"] == 0.5 and m2["precision"] == 1.0


def test_recall_at_fpr():
    y = np.array([1] * 5 + [0] * 5)
    p = np.array([0.9, 0.8, 0.7, 0.3, 0.2, 0.6, 0.1, 0.1, 0.1, 0.1])
    assert E.recall_at_fpr(y, p, 0.0) == pytest.approx(0.6)
    assert E.recall_at_fpr(y, p, 0.1) == pytest.approx(0.6)
    assert E.recall_at_fpr(y, p, 0.2) == pytest.approx(1.0)  # at threshold 0.2 all positives pass with one false positive
    assert E.recall_at_fpr(y, p, 1.0) == 1.0


def test_int8_parity_pass_and_fail():
    y = np.array([1, 1, 0, 0, 1, 0])
    p = np.array([0.9, 0.8, 0.2, 0.1, 0.7, 0.3])
    ok = E.int8_parity(p, p + 0.001, y, 0.5)
    assert ok["passed"] and ok["agreement"] == 1.0 and ok["max_abs_diff"] < 0.002
    bad = E.int8_parity(p, 1 - p, y, 0.5)
    assert not bad["passed"]
    drift = E.int8_parity(p, np.clip(p + 0.06, 0, 1), y, 0.5)  # decisions unchanged but probabilities drift too far
    assert not drift["passed"] and drift["agreement"] == 1.0


@pytest.mark.slow
def test_make_distance_rirs():
    rirs = E.make_distance_rirs(1.0, n=2, seed=0)
    assert len(rirs) == 2 and all(len(h) == 8000 and np.abs(h).max() >= 1.0 - 1e-6 for h in rirs)
