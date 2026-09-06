import numpy as np
import pytest

from v5 import features as F
from v5.data import sources as S


def _buzz(seconds=10.0, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * F.SR)) / F.SR
    env = 0.5 * (1 + np.sin(2 * np.pi * 0.4 * t))
    return (0.1 * np.sin(2 * np.pi * 120 * t) * env + 1e-3 * rng.standard_normal(t.size)).astype(np.float32)


def test_decode_resamples_to_16k(mini_dataset):
    x = S.decode(mini_dataset / "esc50" / "audio" / "1-100000-A-0.wav")
    assert x.dtype == np.float32 and x.ndim == 1
    assert abs(len(x) - 5 * F.SR) <= 2


def test_peak_offsets_are_separated_and_in_range():
    x = _buzz()
    offs = S.peak_offsets(x, n_max=3, min_sep_s=1.5)
    assert len(offs) == 3
    assert all(0 <= o <= len(x) - F.WIN for o in offs)
    centres = sorted(o + F.WIN // 2 for o in offs)
    assert min(np.diff(centres)) >= 1.5 * F.SR
    assert S.peak_offsets(np.zeros(F.SR * 3, np.float32), 3, 1.5) == []  # digital silence has no local maxima


def test_fit_window_pads_short_input():
    w = S.fit_window(np.ones(1000, np.float32), 0)
    assert w.shape == (F.WIN,) and w[999] == 1.0 and w[1000] == 0.0


def test_silence_detection():
    assert S.is_digital_silence(np.zeros(F.WIN, np.float32))
    assert not S.is_digital_silence(0.01 * np.ones(F.WIN, np.float32))


@pytest.mark.parametrize("source,n_expected,label", [
    ("whl_s", 15, 1), ("whl_e", 6, 0), ("mssnsd", 20, 0), ("kaggle_adria", 3, None), ("kaggle_jibran", 2, None), ("wild", 1, 1),
])
def test_iter_windows_counts(mini_dataset, source, n_expected, label):
    rng = np.random.default_rng(0)
    items = list(S.iter_windows(mini_dataset, source, rng))
    assert len(items) == n_expected
    for w, x in items:
        assert x.shape == (F.WIN,) and x.dtype == np.float32
        assert w.source == source and not w.path.startswith("/")
        if label is not None:
            assert w.label == label


def test_esc50_labels_and_groups(mini_dataset):
    items = list(S.iter_windows(mini_dataset, "esc50", np.random.default_rng(0)))
    by_cat = {w.category: w for w, _ in items}
    assert by_cat["snoring"].label == 1 and by_cat["snoring"].group == "esc50_fold5"
    assert by_cat["dog"].label == 0 and by_cat["dog"].group == "esc50_fold1"


def test_whl_groups_are_per_recording_and_category_is_batch(mini_dataset):
    items = list(S.iter_windows(mini_dataset, "whl_s", np.random.default_rng(0)))
    groups = {w.group for w, _ in items}
    assert len(groups) == 5 and all(g.startswith("whl_00000") for g in groups)
    assert {w.category for w, _ in items} == {"000000", "000002"}


def test_whl_snore_file_without_a_peak_is_logged(mini_dataset):
    import soundfile as sf

    quiet = mini_dataset / "whltalent" / "s9.9"
    quiet.mkdir()
    sf.write(quiet / "000009-A-0-001.wav", np.zeros(4000, dtype=np.float32), 16000)  # too short for any peak frame
    errors = []
    items = list(S.iter_windows(mini_dataset, "whl_s", np.random.default_rng(0), errors=errors))
    assert all("000009" not in w.path for w, _ in items) and any(e["error"] == "no_qualifying_peak" and "000009" in e["path"] for e in errors)


def test_errors_are_collected_not_raised(mini_dataset):
    bad = mini_dataset / "whltalent" / "s1.1" / "broken.wav"
    bad.write_bytes(b"not a wav")
    errors = []
    items = list(S.iter_windows(mini_dataset, "whl_s", np.random.default_rng(0), errors=errors))
    assert len(items) == 15 and len(errors) == 1 and "broken.wav" in errors[0]["path"]


def test_load_window_matches_iterated_audio(mini_dataset):
    w, x = next(iter(S.iter_windows(mini_dataset, "kaggle_adria", np.random.default_rng(0))))
    assert np.allclose(S.load_window(mini_dataset, w), x)
