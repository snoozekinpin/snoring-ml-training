import os
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from v5.config import ROOT

DATASET_DIR = ROOT / "dataset"


def pytest_collection_modifyitems(config, items):
    skip_ds = pytest.mark.skip(reason="dataset directory not present")
    for item in items:
        if "dataset" in item.keywords and not DATASET_DIR.exists():
            item.add_marker(skip_ds)


def _buzz(rng, seconds, f0=120.0, level=0.1, sr=16000):
    """Snore-like periodic buzz: harmonic-rich tone with ~0.4 Hz amplitude modulation.

    Harmonic count, modulation rate and noise level are randomised so that synthetic
    clips are not near-duplicates of each other."""
    t = np.arange(int(seconds * sr)) / sr
    n_harm = int(rng.integers(4, 12))
    x = sum(np.sin(2 * np.pi * f0 * k * t + rng.uniform(0, 2 * np.pi)) / k for k in range(1, n_harm))
    env = 0.5 * (1 + np.sin(2 * np.pi * rng.uniform(0.3, 0.5) * t))
    return (level * x * env / 4 + rng.uniform(1e-3, 5e-3) * rng.standard_normal(t.size)).astype(np.float32)


def _noise(rng, seconds, level=0.05, sr=16000):
    return (level * rng.standard_normal(int(seconds * sr))).astype(np.float32)


@pytest.fixture
def mini_dataset(tmp_path):
    """A tiny dataset tree with the real folder layout, all synthetic audio."""
    rng = np.random.default_rng(0)
    d = tmp_path / "dataset"
    w = lambda p, x, sr=16000: (p.parent.mkdir(parents=True, exist_ok=True), sf.write(p, x, sr, subtype="PCM_16"))
    # WHLTalent raw 10 s recordings; the file-name prefix is the batch id used for the split
    for i in range(3):
        w(d / "whltalent" / "s1.1" / f"000000-A-0-00{i + 1}.wav", _buzz(rng, 10, f0=100 + 10 * i))
    w(d / "whltalent" / "s3" / "000002-A-2-001.wav", _buzz(rng, 10, f0=140))
    w(d / "whltalent" / "s3" / "000002-A-2-002.wav", _buzz(rng, 10, f0=150))
    for sub_dir, batch in (("e1", "100000"), ("e2.1", "100001"), ("e3.1", "100002")):
        w(d / "whltalent" / sub_dir / f"{batch}-B-0-001.wav", _noise(rng, 10))
    # ESC-50 at 44.1 kHz with meta (folds 1 train, 4 val, 5 calib)
    meta = ["filename,fold,target,category,esc10,src_file,take"]
    for i, (cat, tgt, fold) in enumerate([("dog", 0, 1), ("rain", 10, 4), ("snoring", 28, 5), ("vacuum_cleaner", 36, 5)]):
        name = f"{fold}-{100000 + i}-A-{tgt}.wav"
        x = _buzz(rng, 5, sr=44100) if cat == "snoring" else _noise(rng, 5, sr=44100)
        w(d / "esc50" / "audio" / name, x, 44100)
        meta.append(f"{name},{fold},{tgt},{cat},False,{100000 + i},A")
    (d / "esc50" / "meta").mkdir(parents=True, exist_ok=True)
    (d / "esc50" / "meta" / "esc50.csv").write_text("\n".join(meta) + "\n")
    # MS-SNSD
    for name in ["AirConditioner_1", "Babble_2", "Typing_3", "VacuumCleaner_4", "ShuttingDoor_5"]:
        w(d / "RAW" / "MS-SNSD" / "noise_train" / f"{name}.wav", _noise(rng, 20))
    (d / "RAW" / "MS-SNSD" / "noise_train" / "Readme.md").write_text("noise\n")
    # Kaggle twins: jibran_s_0000 is an exact copy of adria_s_0000
    adria_s = [_buzz(rng, 1, f0=90), _buzz(rng, 1, f0=130)]
    for i, x in enumerate(adria_s):
        w(d / "adrianagaler" / "snore" / f"adria_s_{i:04d}.wav", x)
    w(d / "adrianagaler" / "noise" / "adria_n_0000.wav", _noise(rng, 1))
    w(d / "snoring_extra" / "jibran" / "jibran_s_0000.wav", adria_s[0])
    w(d / "snoring_extra" / "jibran" / "jibran_n_0000.wav", _noise(rng, 1))
    # wild
    w(d / "RAW" / "Snore_Detection_Project" / "Snore_Detection" / "inference_audios" / "snore1.wav", _buzz(rng, 1, f0=110))
    return d
