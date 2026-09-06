"""Keras data pipeline: reproducible on-the-fly augmentation, 1:2 balanced batches (spec 4.4, 5)."""
from __future__ import annotations

import math

import keras
import numpy as np

from v5 import features as F


def precompute_features(audio_i16) -> np.ndarray:
    X = np.stack([F.extract_int16(a) for a in audio_i16]).astype(np.float32)
    return X[..., None]


class TrainDataset(keras.utils.PyDataset):
    def __init__(self, audio_i16, y, soft, augmenter, batch: int = 64, pos_frac: float = 1 / 3, seed: int = 42, workers: int = 1, **kw):
        super().__init__(workers=workers, use_multiprocessing=False, **kw)
        self.audio = np.asarray(audio_i16)
        self.y = np.asarray(y, dtype=np.float32)
        self.soft = np.asarray(soft if soft is not None else y, dtype=np.float32)
        self.aug = augmenter
        self.batch = batch
        self.n_pos_b = max(1, int(round(batch * pos_frac)))
        self.n_neg_b = batch - self.n_pos_b
        self.seed = int(seed)
        self.epoch = 0
        self.shuffle_rng = np.random.default_rng([self.seed, 0xB00])
        self.pos = np.nonzero(self.y == 1)[0]
        self.neg = np.nonzero(self.y == 0)[0]
        if len(self.pos) == 0 or len(self.neg) == 0:
            raise ValueError("need both classes")
        self._shuffle()

    def _shuffle(self) -> None:
        self.pos_order = self.shuffle_rng.permutation(self.pos)
        self.neg_order = self.shuffle_rng.permutation(self.neg)

    def __len__(self) -> int:
        return int(math.ceil(len(self.neg) / self.n_neg_b))

    def _take(self, order, start, n):
        idx = np.arange(start, start + n) % len(order)
        return order[idx]

    def __getitem__(self, i):
        idx = np.concatenate([self._take(self.pos_order, i * self.n_pos_b, self.n_pos_b), self._take(self.neg_order, i * self.n_neg_b, self.n_neg_b)])
        X = np.empty((len(idx), F.N_FRAMES, F.N_MELS, 1), np.float32)
        for k, j in enumerate(idx):
            rng = np.random.default_rng([self.seed, self.epoch, int(i), int(k)])
            x = self.aug.waveform(F.int16_to_float(self.audio[j]), int(self.y[j]), rng)
            X[k, :, :, 0] = self.aug.features(F.extract(x), rng)
        T = np.stack([self.y[idx], self.soft[idx]], axis=1).astype(np.float32)
        return X, T

    def on_epoch_end(self) -> None:
        self.epoch += 1
        self._shuffle()
