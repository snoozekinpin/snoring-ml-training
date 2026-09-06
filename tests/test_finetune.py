import csv
import json

import numpy as np
import soundfile as sf

from v5 import features as F
from v5 import finetune as FT
from v5.model import build_model


def _session(dir_, seed):
    rng = np.random.default_rng(seed)
    t = np.arange(20 * F.SR) / F.SR
    snore = (0.1 * np.sin(2 * np.pi * 110 * t) * (0.5 + 0.5 * np.sin(2 * np.pi * 0.4 * t)) + 1e-3 * rng.standard_normal(t.size)).astype(np.float32)
    noise = (0.02 * rng.standard_normal(t.size)).astype(np.float32)
    dir_.mkdir(parents=True)
    sf.write(dir_ / "chunk_0000.wav", snore, F.SR, subtype="PCM_16")
    sf.write(dir_ / "chunk_0001.wav", noise, F.SR, subtype="PCM_16")
    with open(dir_ / "labels.csv", "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["chunk", "start_s", "end_s", "label", "side", "posture", "distance_m", "pillow"])
        wr.writerow(["chunk_0000.wav", "0", "20", "snore", "left", "supine", "0.8", "thin"])
        wr.writerow(["chunk_0001.wav", "0", "20", "fan", "", "", "0.8", "thin"])
    return dir_


def test_slice_session_counts_and_labels(tmp_path):
    audio, y = FT.slice_session(_session(tmp_path / "a", 0))
    assert audio.shape == (40, F.WIN) and audio.dtype == np.int16 and y.sum() == 20


def test_holdout_and_overlap_are_rejected(tmp_path):
    import pytest

    a = _session(tmp_path / "a", 3)
    model = build_model(0.5)
    model.save(tmp_path / "m.keras")
    with pytest.raises(ValueError):
        FT.finetune(tmp_path / "m.keras", [a], a, tmp_path / "out", epochs=1)
    with open(a / "labels.csv", "a", newline="") as fh:
        csv.writer(fh).writerow(["chunk_0000.wav", "5", "8", "speech", "", "", "0.8", "thin"])  # overlaps the snore span
    with pytest.raises(ValueError):
        FT.load_labels(a)


def test_finetune_smoke(tmp_path):
    model = build_model(0.5)
    model.save(tmp_path / "m.keras")
    a, b = _session(tmp_path / "a", 1), _session(tmp_path / "b", 2)
    m = FT.finetune(tmp_path / "m.keras", [a], b, tmp_path / "out", epochs=1, tau=0.5)
    assert m["model_version"].startswith("cnn_v5_ft_candidate_") and m["deployable"] is False and m["n_train"] == 40 and m["n_holdout"] == 40
    assert "auc" in m["before"] and "auc" in m["after"]
    assert (tmp_path / "out" / "model.keras").exists()
    assert json.loads((tmp_path / "out" / "metrics.json").read_text())["n_train"] == 40
