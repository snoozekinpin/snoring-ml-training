"""Student CNN (spec section 6): Conv/BN/ReLU/MaxPool x3, GAP, Dense, logit."""
from __future__ import annotations

import keras
from keras import layers

from v5 import features as F

ALLOWED_LAYER_TYPES = {"InputLayer", "Conv2D", "BatchNormalization", "ReLU", "MaxPooling2D", "GlobalAveragePooling2D", "Dense", "Dropout"}


def build_model(width: float = 1.0, dropout: float = 0.3) -> keras.Model:
    def c(n: int) -> int:
        return max(8, int(round(n * width)))

    inp = keras.Input((F.N_FRAMES, F.N_MELS, 1), name="features")
    x = inp
    for i, ch in enumerate((16, 32, 64), start=1):
        x = layers.Conv2D(c(ch), 3, padding="same", use_bias=False, name=f"conv{i}")(x)
        x = layers.BatchNormalization(name=f"bn{i}")(x)
        x = layers.ReLU(name=f"relu{i}")(x)
        x = layers.MaxPooling2D(2, name=f"pool{i}")(x)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dense(c(32), activation="relu", name="fc")(x)
    x = layers.Dropout(dropout, name="drop")(x)
    out = layers.Dense(1, name="logit")(x)
    return keras.Model(inp, out, name=f"snore_v5_w{width:g}".replace(".", "p"))


def check_ops(model: keras.Model) -> None:
    bad = [type(l).__name__ for l in model.layers if type(l).__name__ not in ALLOWED_LAYER_TYPES]
    assert not bad, f"layers outside the TFLM-safe set: {bad}"
