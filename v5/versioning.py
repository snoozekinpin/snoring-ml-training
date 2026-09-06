"""Model version rule shared by training, export and cloud events (spec section 6)."""
from __future__ import annotations

MODEL_VERSION = "cnn_v5_int8"
FORBIDDEN_VERSION_SUBSTRINGS = ("simulator", "demo", "mock")


def check_model_version(v: str) -> str:
    low = str(v).lower()
    if any(s in low for s in FORBIDDEN_VERSION_SUBSTRINGS):
        raise ValueError(f"model_version {v!r} would be treated as simulated data by the cloud")
    return str(v)
