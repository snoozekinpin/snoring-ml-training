import numpy as np

from v5 import features as F
from v5.data import augment as A
from v5.data import dataset as D


def _audio(n, seed=0):
    rng = np.random.default_rng(seed)
    return F.float_to_int16(0.05 * rng.standard_normal((n, F.WIN)))


def _aug(seed=0):
    return A.Augmenter(A.AugmentConfig(p_rir=0.0), _audio(3, 9), None, np.random.default_rng(seed))


def test_batches_are_balanced_and_shaped():
    audio = _audio(90)
    y = np.array([1] * 30 + [0] * 60, np.float32)
    soft = np.linspace(0, 1, 90).astype(np.float32)
    ds = D.TrainDataset(audio, y, soft, _aug(), batch=30, pos_frac=1 / 3, seed=0)
    assert len(ds) == 3  # ceil(60 negatives / 20 per batch)
    X, T = ds[0]
    assert X.shape == (30, F.N_FRAMES, F.N_MELS, 1) and X.dtype == np.float32
    assert T.shape == (30, 2) and T[:, 0].sum() == 10
    assert np.all((T[:, 1] >= 0) & (T[:, 1] <= 1))


def test_batches_are_identical_across_instances_and_worker_counts():
    audio = _audio(60)
    y = np.array([1] * 20 + [0] * 40, np.float32)
    a = D.TrainDataset(audio, y, y, _aug(1), batch=30, seed=3, workers=1)
    b = D.TrainDataset(audio, y, y, _aug(2), batch=30, seed=3, workers=4)
    for i in range(len(a)):
        assert np.array_equal(a[i][0], b[i][0]) and np.array_equal(a[i][1], b[i][1])
    assert np.array_equal(a[1][0], a[1][0])  # fetching the same batch twice gives the same augmentation


def test_epoch_reshuffle_changes_batches_but_not_content():
    audio = _audio(30)
    y = np.array([1] * 10 + [0] * 20, np.float32)
    quiet = A.Augmenter(A.AugmentConfig(p_rir=0.0, p_noise_pos=0.0, p_noise_neg=0.0, p_tilt=0.0, p_shift=0.0, p_gain=0.0, p_specaug=0.0), None, None, np.random.default_rng(0))
    ds = D.TrainDataset(audio, y, y, quiet, batch=30, seed=1)
    X1, _ = ds[0]
    ds.on_epoch_end()
    X2, _ = ds[0]
    assert ds.epoch == 1 and not np.array_equal(X1, X2)
    assert np.allclose(np.sort(X1.sum(axis=(1, 2, 3))), np.sort(X2.sum(axis=(1, 2, 3))), atol=1e-3)


def test_precompute_features_shape():
    Xv = D.precompute_features(_audio(5))
    assert Xv.shape == (5, F.N_FRAMES, F.N_MELS, 1) and Xv.dtype == np.float32
