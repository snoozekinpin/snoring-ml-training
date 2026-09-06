import numpy as np

from v5 import features as F
from v5 import nights as N
from v5.data.augment import rms


def _windows(n, seed, level):
    rng = np.random.default_rng(seed)
    return (level * rng.standard_normal((n, F.WIN))).astype(np.float32)


def test_place_episodes_respects_gaps_and_duration():
    spec = N.NightSpec(duration_s=600, n_episodes=(3, 5), episode_len_s=(20, 40), gap_s=30)
    eps = N.place_episodes(np.random.default_rng(0), spec)
    assert 1 <= len(eps) <= 5
    for (s, e), (s2, e2) in zip(eps, eps[1:]):
        assert e - s >= 20 and s2 - e >= 30
    assert eps[-1][1] + 30 <= 600


def test_generate_night_shape_and_snr():
    rng = np.random.default_rng(1)
    beds = [_windows(1, 5, 1.0)[0].repeat(2)]  # 2 s bed, tiled by make_bed
    spec = N.NightSpec(duration_s=120, snr_db=10.0, bed_dbfs=-40.0, n_episodes=(1, 1), episode_len_s=(30, 30), period_s=(3.0, 3.0), n_distractors=(0, 0), gap_s=20)
    audio, eps = N.generate_night(rng, beds, _windows(3, 2, 0.5), _windows(2, 3, 0.5), spec)
    assert audio.shape == (120 * F.SR,) and audio.dtype == np.float32 and len(eps) == 1
    s, e = eps[0]
    bed_only = audio[: int(10 * F.SR)]
    assert abs(20 * np.log10(rms(bed_only)) - (-40.0)) < 0.5
    burst = audio[int(s * F.SR): int(s * F.SR) + F.WIN]
    snr = 20 * np.log10(rms(burst) / rms(bed_only))
    assert 8.0 < snr < 12.0
