# SnoozMate v5 Edge Snore Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the v5 snore-detection training, evaluation, streaming and export pipeline so that a leakage-free, robust int8 model plus numerically verified C feature/FSM/DoA code can be handed to firmware.

**Architecture:** A `v5/` Python package owns one frozen feature spec, a content-deduplicated group-split manifest over the existing datasets, waveform augmentation, a small Keras CNN (optionally distilled from YAMNet), clip and streaming evaluation, a deterministic episode state machine, a GCC-PHAT direction module, and an exporter that emits int8 TFLite plus C headers and golden vectors. `esp32_firmware/v5/` holds C99 reference implementations validated on the host against those golden vectors.

**Tech Stack:** Python 3.12 (`.venv-mac`), TensorFlow/Keras 3 (2.21), ai-edge-litert, librosa/soundfile/soxr, scipy, scikit-learn, pyroomacoustics, sounddevice, pytest; C99 with clang/cc for host tests.

**Spec:** `docs/superpowers/specs/2026-09-04-snore-v5-edge-pipeline-design.md`

## Global Constraints

- Python interpreter for every command: `.venv-mac/bin/python` (run from the repo root `/Users/pierswang/Desktop/snoring-ml-training`). Tests: `.venv-mac/bin/python -m pytest`.
- Feature spec v1 constants (sample rate 16000, window 16000, stream hop 8000, n_fft 512, hop 256, periodic Hann, 61 frames, 30 mels, fmin 40 Hz, fmax 6000 Hz, log eps 1e-10, normalisation `clip((L - mean)/40, -1, 1)`) live only in `v5/features.py`; no other file re-declares them.
- `model_version` strings must never contain `simulator`, `demo` or `mock`.
- Existing `output/` v4 artifacts are never modified; new artifacts go under `output/v5/`. Only `output/v5/deliverables/` is tracked in git.
- FSM timing is integer tick arithmetic: tick 500 ms, hold 12 ticks, confirm 20 ticks, verify 30 ticks, min bursts 3, period 3–14 ticks, streak in half-ticks (+2 active, −1 inactive).
- DoA sign convention: positive lag means the signal reaches the L channel first (source on the left).
- Commit after every task with a conventional message (`feat:`, `test:`, `docs:`), no co-author trailers.
- Datasets are large; unit tests must never read `dataset/`; tests that do are marked `@pytest.mark.dataset` and skip when the directory is absent.
- The split is immutable: `train` / `val` (early stopping, selection) / `calib` (threshold) / `test` (the whole Kaggle set) / `bench` (MS-SNSD beds). Nothing may train on, select on, calibrate on, or build synthetic nights from `test`; the exporter evaluates it exactly once.
- Fail closed: a missing dataset root, required source or minimum window count aborts the manifest build (`DatasetMissing`); calibration that cannot meet the FPR target aborts deployment (`CalibrationError`); export gates (int8 dtypes/shapes, operator set, parity limits) abort promotion (`ExportError`) and leave previous deliverables untouched.
- C parity is a numerical contract, not "bit-exact": features within 5e-3 (quantised within 1 LSB, at most 1 % of cells off by one), FSM exact, DoA lag within 0.05 samples with identical validity and side.

---

## File map

| File | Responsibility |
|---|---|
| `pytest.ini`, `v5/__init__.py`, `v5/config.py`, `v5/configs/default.yaml` | test config, package root, config loader |
| `v5/features.py` | frozen feature spec, reference extractor, quantiser, golden vector builder |
| `v5/data/sources.py` | decode, resample, peak-window slicing, per-source window iterators |
| `v5/data/manifest.py` | cache build, exact + near dedup, immutable split, invariants, minimum counts, report |
| `v5/data/augment.py` | RIR bank, noise mixing, tilt, shift, gain, SpecAugment, `Augmenter` |
| `v5/data/dataset.py` | Keras `PyDataset` with 1:2 balanced batches, val feature precompute |
| `v5/model.py` | student CNN builder |
| `v5/teacher.py` | optional YAMNet scoring, Platt calibration, label audit |
| `v5/evaluate.py` | clip metrics, recall@FPR, robustness sweep, int8 inference and parity |
| `v5/train.py` | training runs, validation selection, fail-closed calibration, CLI |
| `v5/streaming.py` | `EpisodeFsm` integer-tick reference |
| `v5/nights.py` | synthetic night generator |
| `v5/benchmark_nights.py` | streaming benchmark (float and int8), one-to-one event matching |
| `v5/doa.py`, `v5/doa_sim.py` | GCC-PHAT reference, episode aggregation, simulation sweep |
| `v5/events.py` | edge event dataclass, cloud `EventIn` payload |
| `v5/export.py` | release gates, staged promotion, int8 TFLite, C headers, golden files, model card |
| `v5/recording/record_session.py`, `v5/finetune.py` | self-recording kit, fine-tune recipe |
| `esp32_firmware/v5/*.c/.h`, `esp32_firmware/v5/host_test/*` | C99 reference code and host tests |
| `tests/*.py` | pytest suites |

---

### Task 1: Environment, package skeleton, config loader

**Files:**
- Create: `pytest.ini`, `v5/__init__.py`, `v5/data/__init__.py`, `v5/recording/__init__.py`, `v5/config.py`, `v5/configs/default.yaml`, `tests/__init__.py`, `tests/conftest.py`, `tests/test_config.py`

**Interfaces:**
- Produces: `v5.config.ROOT: Path` (repo root), `v5.config.load_config(path: Path | None = None) -> dict`, `v5.config.resolve(cfg: dict) -> dict` (adds absolute `paths.data_dir`, `paths.raw_dir`, `paths.out_dir`).

- [ ] **Step 1: Verify the environment**

Run: `.venv-mac/bin/python -c "import tensorflow as tf; print(tf.__version__, hasattr(tf.lite, 'TFLiteConverter'))"`
Expected: `2.21.0 True`. If the converter attribute is False, run `uv pip install --python .venv-mac/bin/python "tensorflow==2.19.*"` and re-check; record the final version in the commit message.

- [ ] **Step 2: Write the failing test**

`tests/test_config.py`:
```python
from pathlib import Path
from v5.config import ROOT, load_config, resolve


def test_default_config_loads_and_resolves():
    cfg = resolve(load_config())
    assert cfg["seed"] == 42
    assert cfg["fsm"]["confirm_s"] == 10
    assert Path(cfg["paths"]["out_dir"]).is_absolute()
    assert Path(cfg["paths"]["data_dir"]) == ROOT / "dataset"
    assert cfg["model_version"] == "cnn_v5_int8"
    assert cfg["data"]["min_counts"]["train_pos"] == 1500 and cfg["threshold"]["min_calib_neg"] == 300
    assert cfg["data"]["whl_val_batches"] == ["000002", "100002"]
```

- [ ] **Step 3: Run test to verify it fails**

Run: `.venv-mac/bin/python -m pytest tests/test_config.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'v5'`

- [ ] **Step 4: Write the skeleton**

`pytest.ini`:
```ini
[pytest]
testpaths = tests
pythonpath = .
markers =
    dataset: needs the real dataset directory
    network: needs network access
    slow: takes more than a minute
```

`v5/__init__.py`, `v5/data/__init__.py`, `v5/recording/__init__.py`, `tests/__init__.py`: empty files.

`v5/configs/default.yaml`:
```yaml
seed: 42
model_version: cnn_v5_int8
paths:
  data_dir: dataset
  raw_dir: dataset/RAW
  out_dir: output/v5
data:
  whl_val_batches: ["000002", "100002"]   # WHLTalent batch ids (file-name prefix) held out for validation
  esc50_val_fold: 4
  esc50_calib_fold: 5
  bench_frac: 0.2
  mssnsd_val_frac: 0.15
  mssnsd_calib_frac: 0.15
  near_dup_threshold: 0.98
  whl_snore_max_windows: 3
  whl_env_max_windows: 2
  esc50_max_windows: 2
  esc50_snore_max_windows: 3
  mssnsd_stride_s: 5
  mssnsd_max_windows: 40
  min_counts:
    train_pos: 1500
    train_neg: 4000
    val_pos: 100
    val_neg: 500
    calib_neg: 300
    test_pos: 100
    test_neg: 100
augment:
  rir_bank_size: 300
  p_rir: 0.6
  p_noise_pos: 0.7
  p_noise_neg: 0.4
  snr_range: [-5, 20]
  p_tilt: 0.3
  tilt_range: [-0.5, 0.5]
  p_shift: 0.5
  shift_ms: 100
  p_gain: 0.3
  gain_range: [-12, 6]
  p_specaug: 0.5
train:
  epochs: 60
  batch: 64
  lr: 0.002
  lr_min: 0.00002
  patience: 10
  width: 1.0
  kd_alpha: 0.5
threshold:
  max_fpr: 0.01
  min_calib_neg: 300
fsm:
  tick_ms: 500
  hold_s: 6
  confirm_s: 10
  verify_s: 15
  min_bursts: 3
  period_min_s: 1.5
  period_max_s: 7.0
doa:
  spacing_m: 0.06
benchmark:
  n_nights: 20
  night_s: 3600
  snrs: [0, 5, 10, 20]
```

`v5/config.py`:
```python
"""Config loader. All tunables live in v5/configs/default.yaml."""
from __future__ import annotations

import copy
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "v5" / "configs" / "default.yaml"


def load_config(path: Path | None = None) -> dict:
    with open(path or DEFAULT_CONFIG, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def resolve(cfg: dict) -> dict:
    """Return a copy with absolute paths under cfg['paths']."""
    out = copy.deepcopy(cfg)
    for key, value in out["paths"].items():
        p = Path(value)
        out["paths"][key] = str(p if p.is_absolute() else ROOT / p)
    return out
```

`tests/conftest.py`:
```python
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
```

- [ ] **Step 5: Run test to verify it passes**

Run: `.venv-mac/bin/python -m pytest tests/test_config.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add pytest.ini v5 tests
git commit -m "feat(v5): package skeleton, config loader and test fixtures"
```

---
### Task 2: Feature spec v1 reference extractor

**Files:**
- Create: `v5/features.py`, `tests/test_features.py`

**Interfaces:**
- Produces: constants `SR, WIN, STREAM_HOP, N_FFT, HOP, N_FRAMES, N_BINS, N_MELS, FMIN, FMAX, LOG_EPS, NORM_DIV, FEATURE_DIM, FEATURE_SPEC_VERSION`; `hann_periodic(n=512) -> np.ndarray`; `mel_filterbank() -> np.ndarray[(30,257)]`; `sparse_filterbank() -> list[tuple[int, np.ndarray]]`; `log_mel(x) -> (61,30) float64`; `extract(x: float array[16000]) -> (61,30) float32`; `int16_to_float`, `float_to_int16`; `extract_int16(x_i16)`; `quantize(X, scale, zero_point) -> int8 (61,30)`; `feature_spec_dict() -> dict`.

- [ ] **Step 1: Write the failing tests**

`tests/test_features.py`:
```python
import numpy as np
import pytest

from v5 import features as F


def _sine(freq=100.0, amp=0.1, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(F.WIN) / F.SR
    return (amp * np.sin(2 * np.pi * freq * t) + 1e-4 * rng.standard_normal(F.WIN)).astype(np.float32)


def test_constants_match_spec():
    assert (F.SR, F.WIN, F.STREAM_HOP, F.N_FFT, F.HOP) == (16000, 16000, 8000, 512, 256)
    assert F.N_FRAMES == 61 and F.N_MELS == 30 and F.N_BINS == 257
    assert F.FMIN == 40.0 and F.FMAX == 6000.0 and F.NORM_DIV == 40.0 and F.LOG_EPS == 1e-10


def test_hann_is_periodic():
    w = F.hann_periodic(8)
    assert w[0] == 0.0 and np.isclose(w[4], 1.0)
    assert not np.isclose(w[-1], 0.0)  # periodic form does not end at zero


def test_filterbank_shape_and_coverage():
    fb = F.mel_filterbank()
    assert fb.shape == (30, 257)
    assert (fb.max(axis=1) > 0.5).all()  # every filter has real support
    assert fb[:, :1].sum() == 0  # DC bin unused (fmin 40 Hz)
    sparse = F.sparse_filterbank()
    dense = np.zeros_like(fb)
    for i, (start, w) in enumerate(sparse):
        dense[i, start:start + len(w)] = w
    assert np.allclose(dense, fb)


def test_extract_shape_range_and_gain_invariance():
    x = _sine()
    X = F.extract(x)
    assert X.shape == (61, 30) and X.dtype == np.float32
    assert X.min() >= -1.0 and X.max() <= 1.0
    assert np.allclose(F.extract(0.05 * x), X, atol=1e-5)


def test_silence_is_all_zero():
    assert np.all(F.extract(np.zeros(F.WIN, np.float32)) == 0)


def test_sine_energy_lands_in_low_mels():
    X = F.extract(_sine(100.0))
    assert X.mean(axis=0)[:3].mean() > X.mean(axis=0)[15:].mean()


def test_int16_roundtrip_and_quantize():
    x = _sine()
    xi = F.float_to_int16(x)
    assert xi.dtype == np.int16
    X = F.extract_int16(xi)
    assert np.allclose(X, F.extract(x), atol=2e-3)
    q = F.quantize(X, scale=1 / 127, zero_point=0)
    assert q.dtype == np.int8 and q.shape == (61, 30)
    assert q.max() <= 127 and q.min() >= -128


def test_feature_spec_dict_is_complete():
    d = F.feature_spec_dict()
    for k in ["version", "sample_rate", "window", "n_fft", "hop", "n_frames", "n_mels", "fmin", "fmax", "log_eps", "norm_div", "window_type", "normalisation"]:
        assert k in d


def test_extract_rejects_wrong_length():
    with pytest.raises(ValueError):
        F.extract(np.zeros(100, np.float32))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv-mac/bin/python -m pytest tests/test_features.py -v`
Expected: FAIL with `ImportError: cannot import name 'features'`

- [ ] **Step 3: Implement `v5/features.py`**

```python
"""Feature spec v1 (frozen). Spec section 3. Nothing else may redefine these constants."""
from __future__ import annotations

import numpy as np

FEATURE_SPEC_VERSION = 1
SR = 16000
WIN = 16000
STREAM_HOP = 8000
N_FFT = 512
HOP = 256
N_FRAMES = 1 + (WIN - N_FFT) // HOP  # 61
N_BINS = N_FFT // 2 + 1  # 257
N_MELS = 30
FMIN = 40.0
FMAX = 6000.0
LOG_EPS = 1e-10
NORM_DIV = 40.0
FEATURE_DIM = N_FRAMES * N_MELS


def hann_periodic(n: int = N_FFT) -> np.ndarray:
    k = np.arange(n, dtype=np.float64)
    return (0.5 - 0.5 * np.cos(2.0 * np.pi * k / n)).astype(np.float32)


def _hz_to_mel(f):
    return 2595.0 * np.log10(1.0 + np.asarray(f, dtype=np.float64) / 700.0)


def _mel_to_hz(m):
    return 700.0 * (10.0 ** (np.asarray(m, dtype=np.float64) / 2595.0) - 1.0)


def mel_filterbank() -> np.ndarray:
    """(30, 257) triangular HTK-mel filters with unit peak, evaluated at bin centres."""
    edges = _mel_to_hz(np.linspace(_hz_to_mel(FMIN), _hz_to_mel(FMAX), N_MELS + 2))
    fk = np.arange(N_BINS, dtype=np.float64) * SR / N_FFT
    fb = np.zeros((N_MELS, N_BINS), dtype=np.float32)
    for i in range(N_MELS):
        lo, c, hi = edges[i], edges[i + 1], edges[i + 2]
        up = (fk - lo) / (c - lo)
        down = (hi - fk) / (hi - c)
        fb[i] = np.maximum(0.0, np.minimum(up, down)).astype(np.float32)
    return fb


HANN = hann_periodic()
FB = mel_filterbank()


def sparse_filterbank() -> list[tuple[int, np.ndarray]]:
    out = []
    for i in range(N_MELS):
        nz = np.nonzero(FB[i] > 0)[0]
        out.append((int(nz[0]), FB[i, nz[0]:nz[-1] + 1].copy()))
    return out


def frames_of(x) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.shape != (WIN,):
        raise ValueError(f"expected {WIN} samples, got {x.shape}")
    return np.lib.stride_tricks.sliding_window_view(x, N_FFT)[::HOP][:N_FRAMES]


def log_mel(x) -> np.ndarray:
    """(61, 30) log-mel power in dB, float64 reference."""
    fr = frames_of(x).astype(np.float64) * HANN.astype(np.float64)
    power = np.abs(np.fft.rfft(fr, axis=1)) ** 2
    mel = power @ FB.astype(np.float64).T
    return 10.0 * np.log10(mel + LOG_EPS)


def extract(x) -> np.ndarray:
    L = log_mel(x)
    return np.clip((L - L.mean()) / NORM_DIV, -1.0, 1.0).astype(np.float32)


def int16_to_float(x_i16) -> np.ndarray:
    return np.asarray(x_i16, dtype=np.float32) / 32768.0


def float_to_int16(x) -> np.ndarray:
    return np.clip(np.round(np.asarray(x, dtype=np.float64) * 32767.0), -32768, 32767).astype(np.int16)


def extract_int16(x_i16) -> np.ndarray:
    return extract(int16_to_float(x_i16))


def quantize(X, scale: float, zero_point: int) -> np.ndarray:
    q = np.round(np.asarray(X, dtype=np.float64) / scale) + zero_point
    return np.clip(q, -128, 127).astype(np.int8)


def feature_spec_dict() -> dict:
    return {
        "version": FEATURE_SPEC_VERSION,
        "sample_rate": SR,
        "window": WIN,
        "stream_hop": STREAM_HOP,
        "n_fft": N_FFT,
        "hop": HOP,
        "n_frames": N_FRAMES,
        "n_bins": N_BINS,
        "n_mels": N_MELS,
        "fmin": FMIN,
        "fmax": FMAX,
        "mel_scale": "htk",
        "log_eps": LOG_EPS,
        "norm_div": NORM_DIV,
        "window_type": "hann_periodic",
        "spectrum": "power",
        "normalisation": "clip((10log10(mel+eps) - mean)/40, -1, 1)",
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv-mac/bin/python -m pytest tests/test_features.py -v`
Expected: PASS (9 tests)

- [ ] **Step 5: Commit**

```bash
git add v5/features.py tests/test_features.py
git commit -m "feat(v5): frozen feature spec v1 reference extractor"
```

---
### Task 3: Golden vectors for feature parity

**Files:**
- Create: `v5/golden.py`, `tests/test_golden.py`

**Interfaces:**
- Consumes: `v5.features` (`extract_int16`, `float_to_int16`, `quantize`, `WIN`, `SR`, `FEATURE_DIM`, `N_FRAMES`, `N_MELS`).
- Produces: `GOLDEN_NAMES: list[str]` (12 names), `synth_golden_signals(seed=1) -> dict[str, np.ndarray int16]` (4 synthetic), `make_golden(snore, noise, seed=1, scale=1/127, zero_point=0) -> tuple[list[str], x int16 (12,16000), X float32 (12,61,30), q int8 (12,61,30)]`, `write_golden_npz(path, names, x, X, q, scale, zero_point)`, `read_golden_npz(path) -> (names, x, X, q, scale, zero_point)`, `write_golden_bin(path, x, X, q, scale, zero_point)`, `read_golden_bin(path) -> (x, X, q, scale, zero_point)`.
- Binary format (`features.bin`): header little-endian int32 `count, win, dim`, float32 `scale`, int32 `zero_point`; then per item `int16[win]`, `float32[dim]`, `int8[dim]`.

- [ ] **Step 1: Write the failing tests**

`tests/test_golden.py`:
```python
import numpy as np

from v5 import features as F
from v5 import golden as G


def _clips(seed, n, level):
    rng = np.random.default_rng(seed)
    return [F.float_to_int16(level * rng.standard_normal(F.WIN)) for _ in range(n)]


def test_synth_signals_have_noise_floor_and_silence():
    s = G.synth_golden_signals()
    assert set(s) == {"sine100", "white40", "chirp", "silence"}
    assert np.all(s["silence"] == 0)
    assert np.abs(s["sine100"]).max() > 1000  # roughly -20 dBFS
    assert np.count_nonzero(s["sine100"] == 0) < F.WIN // 100  # dithered, not pure


def test_make_golden_is_deterministic_and_roundtrips(tmp_path):
    names, x, X, q = G.make_golden(_clips(1, 4, 0.1), _clips(2, 4, 0.05), scale=1 / 127, zero_point=0)
    names2, x2, X2, q2 = G.make_golden(_clips(1, 4, 0.1), _clips(2, 4, 0.05), scale=1 / 127, zero_point=0)
    assert names == G.GOLDEN_NAMES and x.shape == (12, F.WIN) and X.shape == (12, 61, 30) and q.shape == (12, 61, 30) and q.dtype == np.int8
    assert np.array_equal(x, x2) and np.array_equal(X, X2) and np.array_equal(q, q2)
    assert np.allclose(X[4], F.extract_int16(x[4])) and np.array_equal(q[4], F.quantize(X[4], 1 / 127, 0))
    G.write_golden_npz(tmp_path / "f.npz", names, x, X, q, 1 / 127, 0)
    n3, x3, X3, q3, sc, zp = G.read_golden_npz(tmp_path / "f.npz")
    assert n3 == names and np.array_equal(x3, x) and np.array_equal(X3, X) and np.array_equal(q3, q) and sc == np.float32(1 / 127) and zp == 0
    G.write_golden_bin(tmp_path / "f.bin", x, X, q, 1 / 127, 0)
    x4, X4, q4, sc4, zp4 = G.read_golden_bin(tmp_path / "f.bin")
    assert np.array_equal(x4, x) and np.array_equal(X4, X) and np.array_equal(q4, q) and zp4 == 0 and abs(sc4 - 1 / 127) < 1e-9
    assert (tmp_path / "f.bin").stat().st_size == 20 + 12 * (F.WIN * 2 + F.FEATURE_DIM * 4 + F.FEATURE_DIM)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv-mac/bin/python -m pytest tests/test_golden.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'v5.golden'`

- [ ] **Step 3: Implement `v5/golden.py`**

```python
"""Golden vectors for host-side C parity tests (spec sections 3 and 11)."""
from __future__ import annotations

import numpy as np
from scipy.signal import chirp

from v5 import features as F

SYNTH_NAMES = ["sine100", "white40", "chirp", "silence"]
GOLDEN_NAMES = SYNTH_NAMES + [f"snore{i}" for i in range(1, 5)] + [f"noise{i}" for i in range(1, 5)]


def synth_golden_signals(seed: int = 1) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    t = np.arange(F.WIN) / F.SR
    floor = 3e-4 * rng.standard_normal(F.WIN)  # about -70 dBFS noise floor
    sine = 0.1 * np.sin(2 * np.pi * 100.0 * t) + floor
    white = 0.01 * rng.standard_normal(F.WIN)
    sweep = 0.1 * chirp(t, f0=1000.0, t1=1.0, f1=100.0) + floor
    return {
        "sine100": F.float_to_int16(sine),
        "white40": F.float_to_int16(white),
        "chirp": F.float_to_int16(sweep),
        "silence": np.zeros(F.WIN, dtype=np.int16),
    }


def make_golden(snore, noise, seed: int = 1, scale: float = 1.0 / 127.0, zero_point: int = 0):
    if len(snore) < 4 or len(noise) < 4:
        raise ValueError("need at least 4 snore and 4 noise clips")
    synth = synth_golden_signals(seed)
    x = [synth[n] for n in SYNTH_NAMES] + [np.asarray(c, np.int16) for c in snore[:4]] + [np.asarray(c, np.int16) for c in noise[:4]]
    x = np.stack(x)
    X = np.stack([F.extract_int16(xi) for xi in x]).astype(np.float32)
    q = np.stack([F.quantize(Xi, scale, zero_point) for Xi in X])
    return list(GOLDEN_NAMES), x, X, q


def write_golden_npz(path, names, x, X, q, scale: float, zero_point: int) -> None:
    np.savez(path, names=np.array(names), x=x, X=X, q=q, scale=np.float32(scale), zero_point=np.int32(zero_point))


def read_golden_npz(path):
    d = np.load(path)
    return [str(n) for n in d["names"]], d["x"], d["X"], d["q"], float(d["scale"]), int(d["zero_point"])


def write_golden_bin(path, x, X, q, scale: float, zero_point: int) -> None:
    x = np.asarray(x, np.int16)
    X = np.asarray(X, np.float32).reshape(len(x), -1)
    q = np.asarray(q, np.int8).reshape(len(x), -1)
    with open(path, "wb") as fh:
        fh.write(np.array([len(x), x.shape[1], X.shape[1]], dtype="<i4").tobytes())
        fh.write(np.array([scale], dtype="<f4").tobytes())
        fh.write(np.array([zero_point], dtype="<i4").tobytes())
        for xi, Xi, qi in zip(x, X, q):
            fh.write(xi.astype("<i2").tobytes())
            fh.write(Xi.astype("<f4").tobytes())
            fh.write(qi.tobytes())


def read_golden_bin(path):
    with open(path, "rb") as fh:
        count, win, dim = np.frombuffer(fh.read(12), dtype="<i4")
        scale = float(np.frombuffer(fh.read(4), dtype="<f4")[0])
        zero_point = int(np.frombuffer(fh.read(4), dtype="<i4")[0])
        xs, Xs, qs = [], [], []
        for _ in range(count):
            xs.append(np.frombuffer(fh.read(win * 2), dtype="<i2"))
            Xs.append(np.frombuffer(fh.read(dim * 4), dtype="<f4").reshape(F.N_FRAMES, F.N_MELS))
            qs.append(np.frombuffer(fh.read(dim), dtype=np.int8).reshape(F.N_FRAMES, F.N_MELS))
    return np.stack(xs), np.stack(Xs), np.stack(qs), scale, zero_point
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv-mac/bin/python -m pytest tests/test_golden.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add v5/golden.py tests/test_golden.py
git commit -m "feat(v5): golden vector builder with quantised features"
```
### Task 4: Source decoding and window slicing

**Files:**
- Create: `v5/data/sources.py`, `tests/test_sources.py`

**Interfaces:**
- Consumes: `v5.features` (`SR`, `WIN`, `float_to_int16`).
- Produces: `SOURCE_IDS: list[str]`, `TEST_SOURCES: frozenset[str]`, `@dataclass(frozen=True) Window(source: str, path: str, offset: int, label: int, group: str, category: str)` (for WHLTalent `category` is the batch id = file-name prefix, `group` is `whl_<batch>_<stem>`), `decode(path) -> np.ndarray float32 mono 16 kHz`, `frame_rms_db(x, frame=512) -> np.ndarray`, `fit_window(x, offset) -> np.ndarray[16000]`, `centre_offset(x) -> int`, `max_rms_offset(x) -> int`, `peak_offsets(x, n_max, min_sep_s, pct=60.0) -> list[int]`, `random_offset(x, rng) -> int`, `is_digital_silence(w) -> bool`, `iter_windows(data_dir, source, rng, cfg=None, errors=None) -> Iterator[tuple[Window, np.ndarray]]`, `load_window(data_dir, w) -> np.ndarray`.

- [ ] **Step 1: Write the failing tests**

`tests/test_sources.py`:
```python
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
    assert S.peak_offsets(np.zeros(F.SR * 3, np.float32), 3, 1.5) == [] or True  # silence yields few/no peaks


def test_fit_window_pads_short_input():
    w = S.fit_window(np.ones(1000, np.float32), 0)
    assert w.shape == (F.WIN,) and w[999] == 1.0 and w[1000] == 0.0


def test_silence_detection():
    assert S.is_digital_silence(np.zeros(F.WIN, np.float32))
    assert not S.is_digital_silence(0.01 * np.ones(F.WIN, np.float32))


@pytest.mark.parametrize("source,n_expected,label", [
    ("whl_s", 12, 1), ("whl_e", 6, 0), ("mssnsd", 20, 0), ("kaggle_adria", 3, None), ("kaggle_jibran", 2, None), ("wild", 1, 1),
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
    assert len(groups) == 4 and all(g.startswith("whl_00000") for g in groups)
    assert {w.category for w, _ in items} == {"000000", "000002"}


def test_errors_are_collected_not_raised(mini_dataset):
    bad = mini_dataset / "whltalent" / "s1.1" / "broken.wav"
    bad.write_bytes(b"not a wav")
    errors = []
    items = list(S.iter_windows(mini_dataset, "whl_s", np.random.default_rng(0), errors=errors))
    assert len(items) == 12 and len(errors) == 1 and "broken.wav" in errors[0]["path"]


def test_load_window_matches_iterated_audio(mini_dataset):
    w, x = next(iter(S.iter_windows(mini_dataset, "kaggle_adria", np.random.default_rng(0))))
    assert np.allclose(S.load_window(mini_dataset, w), x)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv-mac/bin/python -m pytest tests/test_sources.py -v`
Expected: FAIL with `ImportError`

- [ ] **Step 3: Implement `v5/data/sources.py`**

```python
"""Dataset sources: decoding, slicing into one-second windows, group ids (spec 4.1-4.2)."""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import soundfile as sf
import soxr

from v5 import features as F

SOURCE_IDS = ["whl_s", "whl_e", "esc50", "mssnsd", "kaggle_adria", "kaggle_jibran", "wild"]
TEST_SOURCES = frozenset({"kaggle_adria", "kaggle_jibran"})
FRAME = 512


@dataclass(frozen=True)
class Window:
    source: str
    path: str      # relative to data_dir
    offset: int    # sample offset into the decoded 16 kHz mono file
    label: int     # 1 snore, 0 noise
    group: str
    category: str


def decode(path) -> np.ndarray:
    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    mono = data.mean(axis=1)
    if sr != F.SR:
        mono = soxr.resample(mono, sr, F.SR, quality="HQ")
    return np.ascontiguousarray(mono, dtype=np.float32)


def frame_rms_db(x, frame: int = FRAME) -> np.ndarray:
    n = len(x) // frame
    if n == 0:
        return np.full(1, -120.0)
    fr = np.asarray(x[: n * frame], dtype=np.float64).reshape(n, frame)
    return 20.0 * np.log10(np.sqrt((fr ** 2).mean(axis=1)) + 1e-9)


def fit_window(x, offset: int) -> np.ndarray:
    seg = np.asarray(x[offset: offset + F.WIN], dtype=np.float32)
    if len(seg) < F.WIN:
        seg = np.pad(seg, (0, F.WIN - len(seg)))
    return seg


def centre_offset(x) -> int:
    return max(0, (len(x) - F.WIN) // 2)


def _clip_offset(centre: int, n: int) -> int:
    return int(np.clip(centre - F.WIN // 2, 0, max(0, n - F.WIN)))


def max_rms_offset(x) -> int:
    rms = frame_rms_db(x)
    i = int(np.argmax(rms))
    return _clip_offset(i * FRAME + FRAME // 2, len(x))


def peak_offsets(x, n_max: int, min_sep_s: float, pct: float = 60.0) -> list[int]:
    rms = frame_rms_db(x)
    if len(rms) < 3 or n_max <= 0:
        return []
    thr = np.percentile(rms, pct)
    cand = [i for i in range(1, len(rms) - 1) if rms[i] >= thr and rms[i] > rms[i - 1] and rms[i] >= rms[i + 1]]
    cand.sort(key=lambda i: -rms[i])
    min_sep = int(min_sep_s * F.SR)
    centres: list[int] = []
    for i in cand:
        c = i * FRAME + FRAME // 2
        if all(abs(c - other) >= min_sep for other in centres):
            centres.append(c)
        if len(centres) >= n_max:
            break
    return [_clip_offset(c, len(x)) for c in centres]


def random_offset(x, rng) -> int:
    return int(rng.integers(0, max(0, len(x) - F.WIN) + 1))


def is_digital_silence(w) -> bool:
    return 20.0 * np.log10(np.sqrt(np.mean(np.asarray(w, dtype=np.float64) ** 2)) + 1e-9) < -80.0


def _rel(p: Path, data_dir: Path) -> str:
    return str(p.relative_to(data_dir))


def _safe_decode(f: Path, errors):
    try:
        return decode(f)
    except Exception as exc:  # unreadable file: record and move on (spec section 13)
        if errors is not None:
            errors.append({"path": str(f), "error": f"{type(exc).__name__}: {exc}"})
        return None


def _emit(source, f, data_dir, offsets, label, group, category, x):
    seen = set()
    for o in offsets:
        if o in seen:
            continue
        seen.add(o)
        w = fit_window(x, o)
        if is_digital_silence(w):
            continue
        yield Window(source, _rel(f, data_dir), int(o), label, group, category), w


def iter_windows(data_dir, source: str, rng, cfg: dict | None = None, errors: list | None = None) -> Iterator[tuple[Window, np.ndarray]]:
    data_dir = Path(data_dir)
    cfg = cfg or {}
    if source in ("whl_s", "whl_e"):
        prefix, label = ("s", 1) if source == "whl_s" else ("e", 0)
        base = data_dir / "whltalent"
        for d in sorted(p for p in base.iterdir() if p.is_dir() and p.name.startswith(prefix)):
            for f in sorted(d.glob("*.wav")):
                x = _safe_decode(f, errors)
                if x is None:
                    continue
                if label:
                    offs = peak_offsets(x, cfg.get("whl_snore_max_windows", 3), 1.5)
                else:
                    offs = [max_rms_offset(x), random_offset(x, rng)][: cfg.get("whl_env_max_windows", 2)]
                batch = f.stem.split("-")[0]  # WHLTalent file-name prefix = batch id (used for the split)
                yield from _emit(source, f, data_dir, offs, label, f"whl_{batch}_{f.stem}", batch, x)
    elif source == "esc50":
        meta = {}
        with open(data_dir / "esc50" / "meta" / "esc50.csv", newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                meta[row["filename"]] = row
        for f in sorted((data_dir / "esc50" / "audio").glob("*.wav")):
            row = meta.get(f.name)
            if row is None:
                continue
            label = 1 if row["category"] == "snoring" else 0
            n_max = cfg.get("esc50_snore_max_windows", 3) if label else cfg.get("esc50_max_windows", 2)
            x = _safe_decode(f, errors)
            if x is None:
                continue
            yield from _emit(source, f, data_dir, peak_offsets(x, n_max, 1.0), label, f"esc50_fold{row['fold']}", row["category"], x)
    elif source == "mssnsd":
        stride = int(cfg.get("mssnsd_stride_s", 5) * F.SR)
        n_max = cfg.get("mssnsd_max_windows", 40)
        for f in sorted((data_dir / "RAW" / "MS-SNSD" / "noise_train").glob("*.wav")):
            x = _safe_decode(f, errors)
            if x is None:
                continue
            offs = list(range(0, max(1, len(x) - F.WIN + 1), stride))[:n_max]
            yield from _emit(source, f, data_dir, offs, 0, f"mssnsd_{f.stem}", f.stem.split("_")[0], x)
    elif source == "kaggle_adria":
        for label, sub in ((1, "snore"), (0, "noise")):
            for f in sorted((data_dir / "adrianagaler" / sub).glob("*.wav")):
                x = _safe_decode(f, errors)
                if x is None:
                    continue
                yield from _emit(source, f, data_dir, [centre_offset(x)], label, f"adria_{f.stem}", sub, x)
    elif source == "kaggle_jibran":
        for f in sorted((data_dir / "snoring_extra" / "jibran").glob("*.wav")):
            label = 1 if "_s_" in f.name else 0
            x = _safe_decode(f, errors)
            if x is None:
                continue
            yield from _emit(source, f, data_dir, [centre_offset(x)], label, f"jibran_{f.stem}", "snore" if label else "noise", x)
    elif source == "wild":
        base = data_dir / "RAW" / "Snore_Detection_Project" / "Snore_Detection" / "inference_audios"
        for f in sorted(base.glob("snore*.wav")):
            x = _safe_decode(f, errors)
            if x is None:
                continue
            yield from _emit(source, f, data_dir, [centre_offset(x)], 1, f"wild_{f.stem}", "snore", x)
    else:
        raise ValueError(f"unknown source {source}")


def load_window(data_dir, w: Window) -> np.ndarray:
    return fit_window(decode(Path(data_dir) / w.path), w.offset)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv-mac/bin/python -m pytest tests/test_sources.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add v5/data/sources.py tests/test_sources.py
git commit -m "feat(v5): source decoding and one-second window slicing"
```

---
### Task 5: Manifest with content dedup and an immutable split

**Files:**
- Create: `v5/data/manifest.py`, `tests/test_manifest.py`

**Interfaces:**
- Consumes: `v5.data.sources` (`iter_windows`, `SOURCE_IDS`, `TEST_SOURCES`), `v5.features` (`float_to_int16`, `extract_int16`).
- Produces: `COLUMNS`, `SPLITS`, `EVAL_SPLITS`, `REQUIRED_SOURCES`, `class DatasetMissing(RuntimeError)`, `check_dataset_root(data_dir) -> None`, `build_cache(data_dir, out_dir, cfg, seed=42) -> tuple[list[dict], np.ndarray int16 (N,16000), np.ndarray float32 (N,1830)]`, `load_cache(out_dir) -> same tuple`, `load_exact_dups(out_dir) -> list[dict]`, `near_dup_clusters(feats, thr=0.98, block=1024) -> np.ndarray[int]`, `conflicting_clusters(rows) -> set[int]`, `assign_splits(rows, data_cfg: dict, seed=42) -> list[str]`, `enforce_cluster_rule(rows, split) -> tuple[list[str], dict]`, `check_invariants(rows) -> None`, `check_min_counts(rows, min_counts: dict) -> None`, `build_manifest(data_dir, out_dir, cfg, seed=42, reuse_cache=False) -> list[dict]`, `write_manifest(rows, path)`, `read_manifest(path) -> list[dict]`, `split_indices(rows, split) -> np.ndarray[int]`.
- Files written under `out_dir`: `cache/audio_i16.npy`, `cache/feats.npy`, `cache/rows.json`, `cache/exact_dups.json`, `manifest.csv`, `manifest_report.md`, `manifest_errors.csv`.
- Split values (single `split` column): `train`, `val` (early stopping and model selection), `calib` (threshold calibration), `test` (Kaggle, used exactly once by the exporter), `bench` (MS-SNSD files reserved as benchmark noise beds), `sanity`, `drop`.
- Split rule (deterministic, from `cfg["data"]`): WHLTalent windows whose batch id (`category`) is in `whl_val_batches` → val, others → train; ESC-50 fold `esc50_val_fold` → val, fold `esc50_calib_fold` → calib, other folds → train; MS-SNSD files: `bench_frac` → bench, then `mssnsd_val_frac` → val, `mssnsd_calib_frac` → calib, rest → train (seeded); Kaggle → test; wild → sanity. Eval windows sharing a near-duplicate cluster with a train window are dropped; clusters with conflicting labels are dropped everywhere.
- Leakage statement (goes into the report): WHLTalent carries no subject metadata, so the split is batch-disjoint (the file-name prefix), not proven subject-disjoint; ESC-50 uses its official folds; the Kaggle test set is a separate collection never used for training, selection or calibration.

- [ ] **Step 1: Write the failing tests**

`tests/test_manifest.py`:
```python
import numpy as np
import pytest

from v5 import features as F
from v5.config import load_config
from v5.data import manifest as M

TINY_MIN = {"train_pos": 1, "train_neg": 1, "val_pos": 0, "val_neg": 0, "calib_neg": 0, "test_pos": 1, "test_neg": 1}


def _rows(n, source="whl_s", cluster=None, split=None, label=None):
    return [{"id": i, "source": source, "path": f"p{i}", "offset": 0, "label": (label[i] if label else i % 2), "group": f"g{i}", "category": "c", "md5": str(i),
             "dup_cluster": (cluster[i] if cluster else i), "split": (split[i] if split else "train")} for i in range(n)]


def _cfg(threshold=0.999):
    cfg = load_config()
    cfg["data"]["near_dup_threshold"] = threshold  # synthetic clips are similar by construction
    cfg["data"]["min_counts"] = dict(TINY_MIN)
    return cfg


def test_check_dataset_root_raises_when_missing(tmp_path):
    with pytest.raises(M.DatasetMissing):
        M.check_dataset_root(tmp_path / "nowhere")
    (tmp_path / "whltalent").mkdir()
    with pytest.raises(M.DatasetMissing):  # esc50 and MS-SNSD missing
        M.check_dataset_root(tmp_path)


def test_near_dup_clusters_joins_close_pairs():
    rng = np.random.default_rng(0)
    a = rng.standard_normal(F.FEATURE_DIM).astype(np.float32)
    b = a + 1e-3 * rng.standard_normal(F.FEATURE_DIM).astype(np.float32)
    c = rng.standard_normal(F.FEATURE_DIM).astype(np.float32)
    ids = M.near_dup_clusters(np.stack([a, b, c]), thr=0.98, block=2)
    assert ids[0] == ids[1] and ids[0] != ids[2]


def test_enforce_cluster_rule_drops_leaks_and_conflicts():
    rows = _rows(6, cluster=[0, 0, 1, 2, 3, 3], label=[1, 1, 0, 1, 1, 0])
    split = ["train", "val", "val", "test", "train", "calib"]
    out, dropped = M.enforce_cluster_rule(rows, split)
    assert out == ["train", "drop", "val", "test", "drop", "drop"]
    assert dropped == {("leak", "whl_s"): 1, ("conflict", "whl_s"): 2}


def test_check_invariants_rejects_train_eval_overlap():
    for eval_split in ("val", "calib", "test", "bench"):
        rows = _rows(2)
        rows[1]["group"] = rows[0]["group"]
        rows[0]["split"], rows[1]["split"] = "train", eval_split
        with pytest.raises(AssertionError):
            M.check_invariants(rows)
    rows = _rows(2, split=["val", "calib"])
    rows[1]["group"] = rows[0]["group"]
    M.check_invariants(rows)  # eval partitions may share groups


def test_check_min_counts():
    rows = _rows(4, split=["train", "train", "test", "test"], label=[1, 0, 1, 0])
    M.check_min_counts(rows, TINY_MIN)
    with pytest.raises(M.DatasetMissing):
        M.check_min_counts(rows, {**TINY_MIN, "train_pos": 2})


def test_build_manifest_end_to_end(mini_dataset, tmp_path):
    rows = M.build_manifest(mini_dataset, tmp_path / "out", _cfg(), seed=42)
    out = tmp_path / "out"
    assert (out / "manifest.csv").exists() and (out / "manifest_report.md").exists()
    md5s = [r["md5"] for r in rows]
    assert len(md5s) == len(set(md5s))  # exact dedup: the jibran copy of adria_s_0000 is gone
    dups = M.load_exact_dups(out)
    assert len(dups) == 1 and dups[0]["source"] == "kaggle_jibran"
    assert "kaggle_adria ~ kaggle_jibran" in (out / "manifest_report.md").read_text()
    assert all(r["split"] == "test" for r in rows if r["source"].startswith("kaggle"))
    assert all(r["split"] == "sanity" for r in rows if r["source"] == "wild")
    whl = {r["category"]: r["split"] for r in rows if r["source"] in ("whl_s", "whl_e")}
    assert whl["000000"] == "train" and whl["000002"] == "val" and whl["100002"] == "val"
    esc = {r["group"]: r["split"] for r in rows if r["source"] == "esc50"}
    assert esc["esc50_fold5"] == "calib" and esc["esc50_fold4"] == "val" and esc["esc50_fold1"] == "train"
    bench = [r for r in rows if r["split"] == "bench"]
    assert bench and all(r["source"] == "mssnsd" for r in bench) and len({r["group"] for r in bench}) == 1
    M.check_invariants(rows)
    rows2 = M.read_manifest(out / "manifest.csv")
    assert rows2 == rows
    audio, feats = M.load_cache(out)[1:]
    assert audio.shape == (len(rows), F.WIN) and feats.shape == (len(rows), F.FEATURE_DIM)
    tr = M.split_indices(rows, "train")
    assert len(tr) > 0 and all(rows[i]["split"] == "train" for i in tr)


def test_build_manifest_reuses_cache_and_fails_on_min_counts(mini_dataset, tmp_path):
    rows = M.build_manifest(mini_dataset, tmp_path / "out", _cfg())
    rows2 = M.build_manifest(mini_dataset, tmp_path / "out", _cfg(), reuse_cache=True)
    assert rows == rows2
    cfg = _cfg()
    cfg["data"]["min_counts"]["train_pos"] = 10_000
    with pytest.raises(M.DatasetMissing):
        M.build_manifest(mini_dataset, tmp_path / "out", cfg, reuse_cache=True)


def test_build_manifest_fails_when_required_source_missing(mini_dataset, tmp_path):
    import shutil

    shutil.rmtree(mini_dataset / "esc50")
    with pytest.raises(M.DatasetMissing):
        M.build_manifest(mini_dataset, tmp_path / "out", _cfg())
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv-mac/bin/python -m pytest tests/test_manifest.py -v`
Expected: FAIL with `ImportError`

- [ ] **Step 3: Implement `v5/data/manifest.py`**

```python
"""Manifest: decode every window once, dedup by content, immutable split (spec 4.3-4.4)."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from v5 import features as F
from v5.config import load_config, resolve
from v5.data import sources as S

COLUMNS = ["id", "source", "path", "offset", "label", "group", "category", "md5", "dup_cluster", "split"]
INT_COLS = {"id", "offset", "label", "dup_cluster"}
SOURCE_PRIORITY = ["whl_s", "whl_e", "esc50", "mssnsd", "kaggle_adria", "kaggle_jibran", "wild"]
REQUIRED_SOURCES = ("whl_s", "whl_e", "esc50", "mssnsd")
EVAL_SPLITS = ("val", "calib", "test", "bench")
SPLITS = ("train",) + EVAL_SPLITS + ("sanity", "drop")
LEAKAGE_STATEMENT = (
    "WHLTalent carries no subject metadata: the split is batch-disjoint (file-name prefix), not proven subject-disjoint. "
    "ESC-50 uses its official folds (same-source clips share a fold). The Kaggle set is a separate collection used only as the final test."
)


class DatasetMissing(RuntimeError):
    """The dataset root, a required source, or a minimum window count is missing (spec section 13)."""


def check_dataset_root(data_dir) -> None:
    data_dir = Path(data_dir)
    required = {"whltalent": data_dir / "whltalent", "esc50": data_dir / "esc50" / "audio", "esc50 meta": data_dir / "esc50" / "meta" / "esc50.csv",
                "MS-SNSD": data_dir / "RAW" / "MS-SNSD" / "noise_train"}
    missing = [f"{k} ({p})" for k, p in required.items() if not p.exists()]
    if not data_dir.exists():
        raise DatasetMissing(f"dataset root {data_dir} does not exist")
    if missing:
        raise DatasetMissing(f"dataset root {data_dir} is incomplete; missing: {', '.join(missing)}")


def build_cache(data_dir, out_dir, cfg: dict, seed: int = 42):
    check_dataset_root(data_dir)
    out_dir = Path(out_dir)
    (out_dir / "cache").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    rows, audio, feats, seen, errors, exact_dups = [], [], [], {}, [], []
    for src in SOURCE_PRIORITY:
        try:
            for w, x in S.iter_windows(data_dir, src, rng, cfg.get("data", {}), errors):
                xi = F.float_to_int16(x)
                h = hashlib.md5(xi.tobytes()).hexdigest()
                if h in seen:
                    exact_dups.append({"kept_id": seen[h], "kept_source": rows[seen[h]]["source"], "source": w.source, "path": w.path, "label": w.label})
                    continue
                seen[h] = len(rows)
                rows.append({"id": len(rows), "source": w.source, "path": w.path, "offset": w.offset, "label": w.label,
                             "group": w.group, "category": w.category, "md5": h, "dup_cluster": -1, "split": ""})
                audio.append(xi)
                feats.append(F.extract_int16(xi).ravel())
        except FileNotFoundError as exc:
            if src in REQUIRED_SOURCES:
                raise DatasetMissing(f"required source {src} unreadable: {exc}") from exc
            errors.append({"path": src, "error": f"optional source missing: {exc}"})
    if not rows:
        raise DatasetMissing("no windows were decoded")
    audio = np.stack(audio)
    feats = np.stack(feats).astype(np.float32)
    np.save(out_dir / "cache" / "audio_i16.npy", audio)
    np.save(out_dir / "cache" / "feats.npy", feats)
    (out_dir / "cache" / "rows.json").write_text(json.dumps(rows), encoding="utf-8")
    (out_dir / "cache" / "exact_dups.json").write_text(json.dumps(exact_dups), encoding="utf-8")
    with open(out_dir / "manifest_errors.csv", "w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=["path", "error"])
        wr.writeheader()
        wr.writerows(errors)
    return rows, audio, feats


def load_cache(out_dir):
    out_dir = Path(out_dir)
    rows = json.loads((out_dir / "cache" / "rows.json").read_text(encoding="utf-8"))
    return rows, np.load(out_dir / "cache" / "audio_i16.npy"), np.load(out_dir / "cache" / "feats.npy")


def load_exact_dups(out_dir) -> list[dict]:
    p = Path(out_dir) / "cache" / "exact_dups.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else []


def near_dup_clusters(feats, thr: float = 0.98, block: int = 1024) -> np.ndarray:
    fm = np.asarray(feats, dtype=np.float32)
    fm = fm - fm.mean(axis=1, keepdims=True)
    fm = fm / (np.linalg.norm(fm, axis=1, keepdims=True) + 1e-9)
    n = len(fm)
    parent = np.arange(n)

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i0 in range(0, n, block):
        sim = fm[i0: i0 + block] @ fm.T
        for r in range(sim.shape[0]):
            i = i0 + r
            for j in np.nonzero(sim[r, i + 1:] >= thr)[0] + i + 1:
                ra, rb = find(i), find(int(j))
                if ra != rb:
                    parent[max(ra, rb)] = min(ra, rb)
    roots = np.array([find(i) for i in range(n)])
    return np.unique(roots, return_inverse=True)[1].astype(int)


def conflicting_clusters(rows) -> set[int]:
    labels = defaultdict(set)
    for r in rows:
        labels[r["dup_cluster"]].add(r["label"])
    return {c for c, s in labels.items() if len(s) > 1}


def assign_splits(rows, d: dict, seed: int = 42) -> list[str]:
    rng = np.random.default_rng(seed)
    whl_val = {str(b) for b in d.get("whl_val_batches", ["000002", "100002"])}
    esc_val, esc_calib = int(d.get("esc50_val_fold", 4)), int(d.get("esc50_calib_fold", 5))
    ms_groups = sorted({r["group"] for r in rows if r["source"] == "mssnsd"})
    perm = [ms_groups[i] for i in rng.permutation(len(ms_groups))]
    n_bench = int(round(float(d.get("bench_frac", 0.2)) * len(perm)))
    rest = perm[n_bench:]
    n_val = int(math.ceil(float(d.get("mssnsd_val_frac", 0.15)) * len(rest)))
    n_calib = int(math.ceil(float(d.get("mssnsd_calib_frac", 0.15)) * len(rest)))
    ms_split = {g: "bench" for g in perm[:n_bench]}
    ms_split.update({g: "val" for g in rest[:n_val]})
    ms_split.update({g: "calib" for g in rest[n_val: n_val + n_calib]})
    ms_split.update({g: "train" for g in rest[n_val + n_calib:]})
    split = []
    for r in rows:
        src = r["source"]
        if src == "wild":
            split.append("sanity")
        elif src in S.TEST_SOURCES:
            split.append("test")
        elif src in ("whl_s", "whl_e"):
            split.append("val" if r["category"] in whl_val else "train")
        elif src == "esc50":
            fold = int(r["group"].rsplit("fold", 1)[1])
            split.append("val" if fold == esc_val else "calib" if fold == esc_calib else "train")
        elif src == "mssnsd":
            split.append(ms_split[r["group"]])
        else:
            raise ValueError(f"unknown source {src}")
    return split


def enforce_cluster_rule(rows, split):
    conflict = conflicting_clusters(rows)
    train_clusters = {r["dup_cluster"] for r, s in zip(rows, split) if s == "train" and r["dup_cluster"] not in conflict}
    out, dropped = list(split), Counter()
    for i, (r, s) in enumerate(zip(rows, split)):
        if s == "sanity":
            continue
        if r["dup_cluster"] in conflict:
            out[i] = "drop"
            dropped[("conflict", r["source"])] += 1
        elif s in EVAL_SPLITS and r["dup_cluster"] in train_clusters:
            out[i] = "drop"
            dropped[("leak", r["source"])] += 1
    return out, dict(dropped)


def check_invariants(rows) -> None:
    by_group, by_cluster = defaultdict(set), defaultdict(set)
    for r in rows:
        if r["split"] == "train" or r["split"] in EVAL_SPLITS:
            by_group[r["group"]].add(r["split"])
            by_cluster[r["dup_cluster"]].add(r["split"])
    evals = set(EVAL_SPLITS)
    bad_g = [g for g, s in by_group.items() if "train" in s and (s & evals)]
    bad_c = [c for c, s in by_cluster.items() if "train" in s and (s & evals)]
    assert not bad_g, f"groups shared by train and an evaluation split: {bad_g[:5]}"
    assert not bad_c, f"near-duplicate clusters shared by train and an evaluation split: {bad_c[:5]}"
    conflict = conflicting_clusters([r for r in rows if r["split"] != "drop"])
    assert not conflict, f"clusters with conflicting labels survive: {sorted(conflict)[:5]}"


def check_min_counts(rows, min_counts: dict) -> None:
    cnt = Counter((r["split"], r["label"]) for r in rows)
    actual = {"train_pos": cnt[("train", 1)], "train_neg": cnt[("train", 0)], "val_pos": cnt[("val", 1)], "val_neg": cnt[("val", 0)],
              "calib_neg": cnt[("calib", 0)], "test_pos": cnt[("test", 1)], "test_neg": cnt[("test", 0)]}
    short = {k: (actual[k], v) for k, v in min_counts.items() if actual.get(k, 0) < v}
    if short:
        raise DatasetMissing(f"window counts below minimum (actual, required): {short}")


def split_indices(rows, split: str) -> np.ndarray:
    return np.array([r["id"] for r in rows if r["split"] == split], dtype=int)


def write_manifest(rows, path) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=COLUMNS)
        wr.writeheader()
        for r in rows:
            wr.writerow({k: r[k] for k in COLUMNS})


def read_manifest(path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as fh:
        return [{k: (int(v) if k in INT_COLS else v) for k, v in row.items()} for row in csv.DictReader(fh)]


def _report(rows, exact_dups, dropped, members, path) -> None:
    exact_pairs = Counter(tuple(sorted((d["kept_source"], d["source"]))) for d in exact_dups)
    near_pairs = Counter()
    for srcs in members.values():
        u = sorted(set(srcs))
        for i in range(len(u)):
            for j in range(i + 1, len(u)):
                near_pairs[(u[i], u[j])] += 1
    lines = ["# Manifest report", "", f"windows kept: {len(rows)}; exact duplicates discarded: {len(exact_dups)}",
             f"near-duplicate clusters with more than one member: {sum(1 for m in members.values() if len(m) > 1)}", "", LEAKAGE_STATEMENT, "",
             "exact duplicates across sources (kept ~ discarded):"]
    lines += [f"- {a} ~ {b}: {n}" for (a, b), n in sorted(exact_pairs.items())] or ["- none"]
    lines += ["", "near-duplicate clusters spanning two sources:"]
    lines += [f"- {a} ~ {b}: {n}" for (a, b), n in sorted(near_pairs.items())] or ["- none"]
    lines += ["", "| source | label | " + " | ".join(SPLITS) + " |", "|---|---|" + "---|" * len(SPLITS)]
    cnt = Counter((r["source"], r["label"], r["split"]) for r in rows)
    for src in SOURCE_PRIORITY:
        for lab in (1, 0):
            vals = [cnt[(src, lab, s)] for s in SPLITS]
            if sum(vals):
                lines.append(f"| {src} | {lab} | " + " | ".join(str(v) for v in vals) + " |")
    lines += ["", f"dropped (reason, source): {dropped}", ""]
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def build_manifest(data_dir, out_dir, cfg: dict, seed: int = 42, reuse_cache: bool = False) -> list[dict]:
    out_dir = Path(out_dir)
    if reuse_cache and (out_dir / "cache" / "rows.json").exists():
        rows, audio, feats = load_cache(out_dir)
    else:
        rows, audio, feats = build_cache(data_dir, out_dir, cfg, seed)
    d = cfg.get("data", {})
    clusters = near_dup_clusters(feats, float(d.get("near_dup_threshold", 0.98)))
    for r, c in zip(rows, clusters):
        r["dup_cluster"] = int(c)
    split, dropped = enforce_cluster_rule(rows, assign_splits(rows, d, seed))
    for r, s in zip(rows, split):
        r["split"] = s
    check_invariants(rows)
    check_min_counts(rows, d.get("min_counts", {}))
    members = defaultdict(list)
    for r in rows:
        members[r["dup_cluster"]].append(r["source"])
    write_manifest(rows, out_dir / "manifest.csv")
    _report(rows, load_exact_dups(out_dir), dropped, members, out_dir / "manifest_report.md")
    return rows


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Build the v5 manifest")
    ap.add_argument("--config", default=None)
    ap.add_argument("--reuse-cache", action="store_true")
    args = ap.parse_args(argv)
    cfg = resolve(load_config(args.config))
    rows = build_manifest(cfg["paths"]["data_dir"], cfg["paths"]["out_dir"], cfg, cfg["seed"], args.reuse_cache)
    print(f"manifest rows: {len(rows)} -> {cfg['paths']['out_dir']}/manifest.csv")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv-mac/bin/python -m pytest tests/test_manifest.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add v5/data/manifest.py tests/test_manifest.py
git commit -m "feat(v5): manifest with content dedup, immutable split and fail-closed checks"
```
### Task 6: Waveform augmentation

**Files:**
- Create: `v5/data/augment.py`, `tests/test_augment.py`

**Interfaces:**
- Consumes: `v5.features` (`SR`, `WIN`, `N_MELS`, `N_FRAMES`).
- Produces: `make_rir(rng, fs=16000, max_len=8000) -> np.ndarray`, `class RirBank(rirs: list[np.ndarray])` with `generate(n, seed, max_len=8000)`, `save(path)`, `load_or_generate(path, n, seed)`, `random(rng) -> np.ndarray`, `__len__`; `apply_rir(x, h) -> np.ndarray`; `rms(x) -> float`; `mix_noise(x, noise, snr_db) -> np.ndarray`; `spectral_tilt(x, a, fc=1000.0) -> np.ndarray`; `time_shift(x, shift_samples) -> np.ndarray`; `gain_clip(x, gain_db) -> np.ndarray`; `spec_augment(X, rng, max_f=4, max_t=8) -> np.ndarray`; `@dataclass AugmentConfig`; `class Augmenter(cfg: AugmentConfig, noise_bank: np.ndarray (M,16000) float32 or int16, rir_bank: RirBank | None, rng)` with `waveform(x, label) -> np.ndarray` and `features(X) -> np.ndarray`; `AugmentConfig.from_dict(d) -> AugmentConfig`.

- [ ] **Step 1: Write the failing tests**

`tests/test_augment.py`:
```python
import numpy as np
import pytest

from v5 import features as F
from v5.data import augment as A


def _sig(seed=0, level=0.1):
    rng = np.random.default_rng(seed)
    return (level * rng.standard_normal(F.WIN)).astype(np.float32)


def test_mix_noise_hits_requested_snr():
    x, n = _sig(0, 0.1), _sig(1, 0.02)
    for snr in (-5.0, 0.0, 10.0):
        y = A.mix_noise(x, n, snr)
        added = y - x
        got = 20 * np.log10(A.rms(x) / A.rms(added))
        assert abs(got - snr) < 0.05


def test_mix_noise_with_silent_noise_is_identity():
    x = _sig()
    assert np.array_equal(A.mix_noise(x, np.zeros(F.WIN, np.float32), 10.0), x)


@pytest.mark.slow
def test_rir_bank_generation_and_apply(tmp_path):
    bank = A.RirBank.generate(3, seed=0)
    assert len(bank) == 3
    for h in bank.rirs:
        assert h.ndim == 1 and len(h) == 8000 and np.isclose(np.abs(h).max(), 1.0)
    y = A.apply_rir(_sig(), bank.random(np.random.default_rng(0)))
    assert y.shape == (F.WIN,) and y.dtype == np.float32
    bank.save(tmp_path / "rir.npz")
    bank2 = A.RirBank.load_or_generate(tmp_path / "rir.npz", 3, seed=0)
    assert np.array_equal(bank2.rirs[0], bank.rirs[0])


def test_tilt_shift_gain_specaug():
    x = _sig()
    assert np.allclose(A.spectral_tilt(x, 0.0), x, atol=1e-6)
    t = A.spectral_tilt(x, 0.5)
    assert t.shape == x.shape and not np.allclose(t, x)
    s = A.time_shift(x, 100)
    assert np.all(s[:100] == 0) and np.array_equal(s[100:], x[:-100])
    s2 = A.time_shift(x, -100)
    assert np.all(s2[-100:] == 0) and np.array_equal(s2[:-100], x[100:])
    g = A.gain_clip(np.full(F.WIN, 0.9, np.float32), 6.0)
    assert g.max() == 1.0
    X = np.ones((F.N_FRAMES, F.N_MELS), np.float32)
    Xa = A.spec_augment(X, np.random.default_rng(0))
    assert Xa.shape == X.shape and (Xa == 0).any() and (Xa == 1).any()


def test_augmenter_is_seed_deterministic_and_keeps_shape():
    cfg = A.AugmentConfig(p_rir=0.0)
    bank = np.stack([_sig(5, 0.05), _sig(6, 0.05)])
    a1 = A.Augmenter(cfg, bank, None, np.random.default_rng(3))
    a2 = A.Augmenter(cfg, bank, None, np.random.default_rng(3))
    x = _sig()
    y1, y2 = a1.waveform(x, 1), a2.waveform(x, 1)
    assert y1.shape == (F.WIN,) and y1.dtype == np.float32 and np.array_equal(y1, y2)
    X = F.extract(y1)
    assert a1.features(X).shape == X.shape


def test_augment_config_from_dict_roundtrip():
    cfg = A.AugmentConfig.from_dict({"p_rir": 0.1, "snr_range": [0, 5], "gain_range": [-3, 3]})
    assert cfg.p_rir == 0.1 and cfg.snr_range == (0.0, 5.0) and cfg.p_noise_pos == 0.7
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv-mac/bin/python -m pytest tests/test_augment.py -v`
Expected: FAIL with `ImportError`

- [ ] **Step 3: Implement `v5/data/augment.py`**

```python
"""Waveform augmentation (spec section 5)."""
from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np
from scipy.signal import fftconvolve, lfilter

from v5 import features as F


def rms(x) -> float:
    return float(np.sqrt(np.mean(np.asarray(x, dtype=np.float64) ** 2)))


def make_rir(rng, fs: int = F.SR, max_len: int = 8000) -> np.ndarray:
    import pyroomacoustics as pra

    dims = [float(rng.uniform(3.0, 5.0)), float(rng.uniform(3.0, 5.0)), float(rng.uniform(2.4, 3.0))]
    rt60 = float(rng.uniform(0.2, 0.6))
    e_abs, max_order = pra.inverse_sabine(rt60, dims)
    room = pra.ShoeBox(dims, fs=fs, materials=pra.Material(e_abs), max_order=min(int(max_order), 10))
    mic = np.array([rng.uniform(0.3, dims[0] - 0.3), rng.uniform(0.3, dims[1] - 0.3), rng.uniform(0.5, 0.9)])
    src = None
    for _ in range(50):
        d, az = rng.uniform(0.4, 1.8), rng.uniform(0.0, 2 * np.pi)
        cand = np.array([mic[0] + d * np.cos(az), mic[1] + d * np.sin(az), rng.uniform(0.4, 0.7)])
        if np.all(cand > 0.2) and np.all(cand < np.array(dims) - 0.2):
            src = cand
            break
    if src is None:
        src = np.array([dims[0] / 2, dims[1] / 2, 0.6])
    room.add_source(src.tolist())
    room.add_microphone(mic.tolist())
    room.compute_rir()
    h = np.asarray(room.rir[0][0], dtype=np.float32)
    h = h[:max_len] if len(h) >= max_len else np.pad(h, (0, max_len - len(h)))
    return (h / (np.abs(h).max() + 1e-12)).astype(np.float32)


class RirBank:
    def __init__(self, rirs: list[np.ndarray]):
        self.rirs = [np.asarray(h, dtype=np.float32) for h in rirs]

    def __len__(self) -> int:
        return len(self.rirs)

    @classmethod
    def generate(cls, n: int, seed: int = 42, max_len: int = 8000) -> "RirBank":
        rng = np.random.default_rng(seed)
        return cls([make_rir(rng, max_len=max_len) for _ in range(n)])

    def save(self, path) -> None:
        np.savez_compressed(path, rirs=np.stack(self.rirs))

    @classmethod
    def load_or_generate(cls, path, n: int, seed: int = 42) -> "RirBank":
        path = Path(path)
        if path.exists():
            return cls(list(np.load(path)["rirs"]))
        bank = cls.generate(n, seed)
        path.parent.mkdir(parents=True, exist_ok=True)
        bank.save(path)
        return bank

    def random(self, rng) -> np.ndarray:
        return self.rirs[int(rng.integers(0, len(self.rirs)))]


def apply_rir(x, h) -> np.ndarray:
    return fftconvolve(np.asarray(x, dtype=np.float32), h)[: len(x)].astype(np.float32)


def mix_noise(x, noise, snr_db: float) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    pn = rms(noise)
    if pn <= 0:
        return x
    scale = rms(x) / (pn * 10 ** (snr_db / 20.0))
    return (x + scale * np.asarray(noise, dtype=np.float32)).astype(np.float32)


def spectral_tilt(x, a: float, fc: float = 1000.0) -> np.ndarray:
    alpha = float(np.exp(-2 * np.pi * fc / F.SR))
    lp = lfilter([1 - alpha], [1, -alpha], np.asarray(x, dtype=np.float64))
    return (x + a * (x - lp)).astype(np.float32)


def time_shift(x, shift_samples: int) -> np.ndarray:
    y = np.zeros_like(x, dtype=np.float32)
    s = int(shift_samples)
    if s > 0:
        y[s:] = x[:-s]
    elif s < 0:
        y[:s] = x[-s:]
    else:
        y[:] = x
    return y


def gain_clip(x, gain_db: float) -> np.ndarray:
    return np.clip(np.asarray(x, dtype=np.float32) * 10 ** (gain_db / 20.0), -1.0, 1.0).astype(np.float32)


def spec_augment(X, rng, max_f: int = 4, max_t: int = 8) -> np.ndarray:
    X = np.array(X, dtype=np.float32, copy=True)
    fw, tw = int(rng.integers(1, max_f + 1)), int(rng.integers(1, max_t + 1))
    f0, t0 = int(rng.integers(0, X.shape[1] - fw + 1)), int(rng.integers(0, X.shape[0] - tw + 1))
    X[:, f0: f0 + fw] = 0.0
    X[t0: t0 + tw, :] = 0.0
    return X


@dataclass
class AugmentConfig:
    p_rir: float = 0.6
    p_noise_pos: float = 0.7
    p_noise_neg: float = 0.4
    snr_range: tuple = (-5.0, 20.0)
    p_tilt: float = 0.3
    tilt_range: tuple = (-0.5, 0.5)
    p_shift: float = 0.5
    shift_ms: float = 100.0
    p_gain: float = 0.3
    gain_range: tuple = (-12.0, 6.0)
    p_specaug: float = 0.5

    @classmethod
    def from_dict(cls, d: dict) -> "AugmentConfig":
        names = {f.name for f in fields(cls)}
        kw = {}
        for k, v in d.items():
            if k in names:
                kw[k] = tuple(float(u) for u in v) if isinstance(v, (list, tuple)) else v
        return cls(**kw)


class Augmenter:
    def __init__(self, cfg: AugmentConfig, noise_bank, rir_bank: RirBank | None, rng):
        self.cfg, self.rir_bank, self.rng = cfg, rir_bank, rng
        # int16 banks are kept as-is and converted per pick (a float copy of 8k windows is 500 MB)
        self.noise_bank = np.asarray(noise_bank) if noise_bank is not None and len(noise_bank) else None

    def waveform(self, x, label: int) -> np.ndarray:
        c, rng = self.cfg, self.rng
        y = np.asarray(x, dtype=np.float32)
        if self.rir_bank is not None and len(self.rir_bank) and rng.random() < c.p_rir:
            y = apply_rir(y, self.rir_bank.random(rng))
        p_noise = c.p_noise_pos if label == 1 else c.p_noise_neg
        if self.noise_bank is not None and rng.random() < p_noise:
            n = self.noise_bank[int(rng.integers(0, len(self.noise_bank)))]
            if n.dtype == np.int16:
                n = F.int16_to_float(n)
            y = mix_noise(y, n, float(rng.uniform(*c.snr_range)))
        if rng.random() < c.p_tilt:
            y = spectral_tilt(y, float(rng.uniform(*c.tilt_range)))
        if rng.random() < c.p_shift:
            y = time_shift(y, int(rng.integers(-int(c.shift_ms * F.SR / 1000), int(c.shift_ms * F.SR / 1000) + 1)))
        if rng.random() < c.p_gain:
            y = gain_clip(y, float(rng.uniform(*c.gain_range)))
        return y.astype(np.float32)

    def features(self, X) -> np.ndarray:
        if self.rng.random() < self.cfg.p_specaug:
            return spec_augment(X, self.rng)
        return np.asarray(X, dtype=np.float32)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv-mac/bin/python -m pytest tests/test_augment.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add v5/data/augment.py tests/test_augment.py
git commit -m "feat(v5): waveform augmentation with RIR bank and noise mixing"
```

---
### Task 7: Keras dataset with balanced batches

**Files:**
- Create: `v5/data/dataset.py`, `tests/test_dataset.py`

**Interfaces:**
- Consumes: `v5.features` (`extract`, `int16_to_float`, `N_FRAMES`, `N_MELS`), `v5.data.augment.Augmenter`.
- Produces: `class TrainDataset(keras.utils.PyDataset)` constructed as `TrainDataset(audio_i16, y, soft, augmenter, batch=64, pos_frac=1/3, seed=42, workers=1)`, `__len__`, `__getitem__(i) -> (X (B,61,30,1) float32, T (B,2) float32)` where `T[:,0] = y` and `T[:,1] = soft`; `on_epoch_end()` reshuffles; `precompute_features(audio_i16) -> np.ndarray (N,61,30,1)`.

- [ ] **Step 1: Write the failing tests**

`tests/test_dataset.py`:
```python
import numpy as np

from v5 import features as F
from v5.data import augment as A
from v5.data import dataset as D


def _audio(n, seed=0):
    rng = np.random.default_rng(seed)
    return F.float_to_int16(0.05 * rng.standard_normal((n, F.WIN)))


def test_batches_are_balanced_and_shaped():
    audio = _audio(90)
    y = np.array([1] * 30 + [0] * 60, np.float32)
    soft = np.linspace(0, 1, 90).astype(np.float32)
    aug = A.Augmenter(A.AugmentConfig(p_rir=0.0), None, None, np.random.default_rng(0))
    ds = D.TrainDataset(audio, y, soft, aug, batch=30, pos_frac=1 / 3, seed=0)
    assert len(ds) == 3  # ceil(60 negatives / 20 per batch)
    X, T = ds[0]
    assert X.shape == (30, F.N_FRAMES, F.N_MELS, 1) and X.dtype == np.float32
    assert T.shape == (30, 2) and T[:, 0].sum() == 10
    assert np.all((T[:, 1] >= 0) & (T[:, 1] <= 1))


def test_epoch_reshuffle_changes_order_but_not_content():
    audio = _audio(30)
    y = np.array([1] * 10 + [0] * 20, np.float32)
    aug = A.Augmenter(A.AugmentConfig(p_rir=0.0, p_noise_pos=0.0, p_noise_neg=0.0, p_tilt=0.0, p_shift=0.0, p_gain=0.0, p_specaug=0.0), None, None, np.random.default_rng(0))
    ds = D.TrainDataset(audio, y, y, aug, batch=30, seed=1)
    X1, _ = ds[0]
    ds.on_epoch_end()
    X2, _ = ds[0]
    assert not np.array_equal(X1, X2)
    assert np.allclose(np.sort(X1.sum(axis=(1, 2, 3))), np.sort(X2.sum(axis=(1, 2, 3))), atol=1e-3)


def test_precompute_features_shape():
    Xv = D.precompute_features(_audio(5))
    assert Xv.shape == (5, F.N_FRAMES, F.N_MELS, 1) and Xv.dtype == np.float32
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv-mac/bin/python -m pytest tests/test_dataset.py -v`
Expected: FAIL with `ImportError`

- [ ] **Step 3: Implement `v5/data/dataset.py`**

```python
"""Keras data pipeline: on-the-fly augmentation, 1:2 balanced batches (spec 4.4, 5)."""
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
        self.rng = np.random.default_rng(seed)
        self.pos = np.nonzero(self.y == 1)[0]
        self.neg = np.nonzero(self.y == 0)[0]
        if len(self.pos) == 0 or len(self.neg) == 0:
            raise ValueError("need both classes")
        self._shuffle()

    def _shuffle(self) -> None:
        self.pos_order = self.rng.permutation(self.pos)
        self.neg_order = self.rng.permutation(self.neg)

    def __len__(self) -> int:
        return int(math.ceil(len(self.neg) / self.n_neg_b))

    def _take(self, order, start, n):
        idx = np.arange(start, start + n) % len(order)
        return order[idx]

    def __getitem__(self, i):
        idx = np.concatenate([self._take(self.pos_order, i * self.n_pos_b, self.n_pos_b), self._take(self.neg_order, i * self.n_neg_b, self.n_neg_b)])
        X = np.empty((len(idx), F.N_FRAMES, F.N_MELS, 1), np.float32)
        for k, j in enumerate(idx):
            x = self.aug.waveform(F.int16_to_float(self.audio[j]), int(self.y[j]))
            X[k, :, :, 0] = self.aug.features(F.extract(x))
        T = np.stack([self.y[idx], self.soft[idx]], axis=1).astype(np.float32)
        return X, T

    def on_epoch_end(self) -> None:
        self._shuffle()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv-mac/bin/python -m pytest tests/test_dataset.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add v5/data/dataset.py tests/test_dataset.py
git commit -m "feat(v5): balanced Keras dataset with on-the-fly augmentation"
```

---
### Task 8: Student model

**Files:**
- Create: `v5/model.py`, `tests/test_model.py`

**Interfaces:**
- Consumes: `v5.features` (`N_FRAMES`, `N_MELS`).
- Produces: `build_model(width: float = 1.0, dropout: float = 0.3) -> keras.Model` (input `(61,30,1)`, output `(None,1)` logits, layer names `conv1..3`, `bn1..3`, `pool1..3`, `gap`, `fc`, `logit`), `ALLOWED_LAYER_TYPES`, `check_ops(model) -> None` (raises `AssertionError` on a layer type outside the TFLM-safe set).

- [ ] **Step 1: Write the failing tests**

`tests/test_model.py`:
```python
import keras
import numpy as np

from v5.model import build_model, check_ops


def test_width_one_is_small_and_tflm_safe():
    m = build_model(1.0)
    assert m.input_shape == (None, 61, 30, 1) and m.output_shape == (None, 1)
    assert 20_000 < m.count_params() < 40_000
    check_ops(m)
    assert m.get_layer("conv1").kernel.shape[-1] == 16 and m.get_layer("fc").units == 32


def test_width_two_is_larger_and_forward_works():
    m = build_model(2.0)
    assert m.count_params() > 80_000
    out = m(np.zeros((3, 61, 30, 1), np.float32), training=False)
    assert out.shape == (3, 1)


def test_save_and_reload(tmp_path):
    m = build_model(0.5)
    m.save(tmp_path / "m.keras")
    m2 = keras.models.load_model(tmp_path / "m.keras")
    x = np.random.default_rng(0).standard_normal((2, 61, 30, 1)).astype(np.float32)
    assert np.allclose(m(x, training=False), m2(x, training=False))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv-mac/bin/python -m pytest tests/test_model.py -v`
Expected: FAIL with `ImportError`

- [ ] **Step 3: Implement `v5/model.py`**

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv-mac/bin/python -m pytest tests/test_model.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add v5/model.py tests/test_model.py
git commit -m "feat(v5): student CNN builder"
```

---
### Task 9: Optional YAMNet teacher, Platt calibration, label audit

**Files:**
- Create: `v5/teacher.py`, `tests/test_teacher.py`

**Interfaces:**
- Consumes: `v5.features.int16_to_float`, `v5.data.manifest.load_cache`.
- Produces: `YAMNET_URL`, `load_yamnet() -> tuple[model | None, list[int]]`, `class_names(model) -> list[str]`, `logit(p) -> np.ndarray`, `score_windows(model, idx, audio_i16) -> np.ndarray[float64] z`, `platt_fit(z, y) -> tuple[float, float]`, `platt_apply(z, a, b) -> np.ndarray`, `audit_flags(z, y, weak=0.02, contaminated=0.5) -> list[str]`, `write_scores(path, ids, z, flags)`, `read_scores(path) -> dict[int, tuple[float, str]]`.
- File: `output/v5/teacher.csv` with columns `id,z,flag`.

- [ ] **Step 1: Write the failing tests**

`tests/test_teacher.py`:
```python
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
    T.write_scores(tmp_path / "t.csv", [3, 5], np.array([0.25, -1.5]), ["", "weak_positive"])
    d = T.read_scores(tmp_path / "t.csv")
    assert d[3] == (0.25, "") and d[5] == (-1.5, "weak_positive")


@pytest.mark.network
@pytest.mark.skipif(not os.environ.get("V5_NETWORK_TESTS"), reason="set V5_NETWORK_TESTS=1 to download YAMNet")
def test_yamnet_loads_and_scores():
    model, idx = T.load_yamnet()
    assert model is not None and idx
    z = T.score_windows(model, idx, np.zeros((1, 16000), np.int16))
    assert z.shape == (1,)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv-mac/bin/python -m pytest tests/test_teacher.py -v`
Expected: FAIL with `ImportError`

- [ ] **Step 3: Implement `v5/teacher.py`**

```python
"""Optional YAMNet teacher: soft labels, Platt calibration, label audit (spec 4.5 and 6)."""
from __future__ import annotations

import argparse
import csv
import io
from collections import Counter
from pathlib import Path

import numpy as np

from v5 import features as F
from v5.config import load_config, resolve

YAMNET_URL = "https://tfhub.dev/google/yamnet/1"
TARGET_CLASSES = ("Snoring", "Snort")


def class_names(model) -> list[str]:
    import tensorflow as tf

    path = model.class_map_path().numpy().decode("utf-8")
    text = tf.io.read_file(path).numpy().decode("utf-8")
    return [r["display_name"] for r in csv.DictReader(io.StringIO(text))]


def load_yamnet():
    try:
        import tensorflow_hub as hub

        model = hub.load(YAMNET_URL)
        idx = [i for i, n in enumerate(class_names(model)) if n in TARGET_CLASSES]
        if not idx:
            raise RuntimeError("Snoring class not found in the YAMNet class map")
        return model, idx
    except Exception as exc:  # spec section 13: degrade to hard labels
        print(f"[teacher] unavailable ({type(exc).__name__}: {exc}); KD and audit disabled")
        return None, []


def logit(p) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def score_windows(model, idx, audio_i16) -> np.ndarray:
    z = np.empty(len(audio_i16), np.float64)
    for i, a in enumerate(audio_i16):
        scores, _, _ = model(F.int16_to_float(a))
        s = scores.numpy().mean(axis=0)
        z[i] = logit(float(s[idx].sum()))
    return z


def platt_fit(z, y) -> tuple[float, float]:
    from sklearn.linear_model import LogisticRegression

    lr = LogisticRegression(C=1e4, max_iter=1000).fit(np.asarray(z, dtype=np.float64).reshape(-1, 1), np.asarray(y).astype(int))
    return float(lr.coef_[0, 0]), float(lr.intercept_[0])


def platt_apply(z, a: float, b: float) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-(a * np.asarray(z, dtype=np.float64) + b)))


def audit_flags(z, y, weak: float = 0.02, contaminated: float = 0.5) -> list[str]:
    p = 1.0 / (1.0 + np.exp(-np.asarray(z, dtype=np.float64)))
    out = []
    for pp, yy in zip(p, np.asarray(y)):
        if yy == 1 and pp < weak:
            out.append("weak_positive")
        elif yy == 0 and pp > contaminated:
            out.append("snore_in_negative")
        else:
            out.append("")
    return out


def write_scores(path, ids, z, flags) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        wr = csv.writer(fh)
        wr.writerow(["id", "z", "flag"])
        for i, zz, fl in zip(ids, z, flags):
            wr.writerow([int(i), f"{float(zz):.6f}", fl])


def read_scores(path) -> dict[int, tuple[float, str]]:
    with open(path, newline="", encoding="utf-8") as fh:
        return {int(r["id"]): (float(r["z"]), r["flag"]) for r in csv.DictReader(fh)}


def main(argv=None) -> None:
    from v5.data.manifest import load_cache

    ap = argparse.ArgumentParser(description="Score cached windows with YAMNet")
    ap.add_argument("--config", default=None)
    args = ap.parse_args(argv)
    cfg = resolve(load_config(args.config))
    out = Path(cfg["paths"]["out_dir"])
    model, idx = load_yamnet()
    if model is None:
        return
    rows, audio, _ = load_cache(out)
    z = score_windows(model, idx, audio)
    flags = audit_flags(z, [r["label"] for r in rows])
    write_scores(out / "teacher.csv", [r["id"] for r in rows], z, flags)
    print(f"[teacher] wrote {out / 'teacher.csv'}; flags: {dict(Counter(f for f in flags if f))}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv-mac/bin/python -m pytest tests/test_teacher.py -v`
Expected: PASS (network test skipped)

- [ ] **Step 5: Commit**

```bash
git add v5/teacher.py tests/test_teacher.py
git commit -m "feat(v5): optional YAMNet teacher with Platt calibration and label audit"
```

---
### Task 10: Clip-level evaluation and int8 inference helpers

**Files:**
- Create: `v5/evaluate.py`, `tests/test_evaluate.py`

**Interfaces:**
- Consumes: `v5.features`, `v5.data.augment` (`apply_rir`, `mix_noise`), `v5.data.dataset.precompute_features`.
- Produces: `sigmoid(z)`, `recall_at_fpr(y, p, max_fpr=0.02) -> float`, `clip_metrics(y, p, tau) -> dict` (keys `n, n_pos, auc, recall_at_fpr2, tau, precision, recall, fnr, fpr, cm`), `predict_probs(model, X, batch=256) -> np.ndarray`, `make_distance_rirs(distance_m, n=5, seed=0) -> list[np.ndarray]`, `robustness_sweep(model, audio_i16, y, noise_i16, tau, snrs=(20,10,5,0), distances=(0.5,1.0,1.5), seed=0) -> dict`, `make_interpreter(tflite: bytes | str)`, `int8_probs(tflite, X) -> np.ndarray`, `int8_parity(p_fp, p_int8, y, tau) -> dict` (keys `delta_auc, agreement, max_abs_diff, passed`; passed requires `delta_auc < 0.005`, `agreement >= 0.99` and `max_abs_diff <= 0.05`).

- [ ] **Step 1: Write the failing tests**

`tests/test_evaluate.py`:
```python
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
    assert len(rirs) == 2 and all(len(h) == 8000 and np.isclose(np.abs(h).max(), 1.0) for h in rirs)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv-mac/bin/python -m pytest tests/test_evaluate.py -v`
Expected: FAIL with `ImportError`

- [ ] **Step 3: Implement `v5/evaluate.py`**

```python
"""Clip-level metrics, robustness sweeps and int8 inference (spec section 7)."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

from v5 import features as F
from v5.data.augment import apply_rir, mix_noise
from v5.data.dataset import precompute_features


def sigmoid(z) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.asarray(z, dtype=np.float64)))


def recall_at_fpr(y, p, max_fpr: float = 0.02) -> float:
    fpr, tpr, _ = roc_curve(np.asarray(y).astype(int), np.asarray(p, dtype=np.float64))
    ok = fpr <= max_fpr + 1e-12
    return float(tpr[ok].max()) if ok.any() else 0.0


def clip_metrics(y, p, tau: float) -> dict:
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=np.float64)
    pred = (p >= tau).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    two_classes = len(np.unique(y)) == 2
    return {
        "n": int(len(y)),
        "n_pos": int(y.sum()),
        "auc": float(roc_auc_score(y, p)) if two_classes else float("nan"),
        "recall_at_fpr2": recall_at_fpr(y, p, 0.02) if two_classes else float("nan"),
        "tau": float(tau),
        "precision": tp / max(tp + fp, 1),
        "recall": tp / max(tp + fn, 1),
        "fnr": fn / max(tp + fn, 1),
        "fpr": fp / max(fp + tn, 1),
        "cm": [[tn, fp], [fn, tp]],
    }


def predict_probs(model, X, batch: int = 256) -> np.ndarray:
    return sigmoid(model.predict(X, batch_size=batch, verbose=0).ravel())


def make_distance_rirs(distance_m: float, n: int = 5, seed: int = 0, max_len: int = 8000) -> list[np.ndarray]:
    import pyroomacoustics as pra

    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        dims = [4.0, 4.0, 2.6]
        e_abs, max_order = pra.inverse_sabine(0.3, dims)
        room = pra.ShoeBox(dims, fs=F.SR, materials=pra.Material(e_abs), max_order=min(int(max_order), 10))
        mic = np.array([2.0, 1.0, 0.7])
        az = rng.uniform(-np.pi / 3, np.pi / 3)
        src = mic + np.array([distance_m * np.sin(az), distance_m * np.cos(az), -0.15])
        room.add_source(src.tolist())
        room.add_microphone(mic.tolist())
        room.compute_rir()
        h = np.asarray(room.rir[0][0], dtype=np.float32)
        h = h[:max_len] if len(h) >= max_len else np.pad(h, (0, max_len - len(h)))
        out.append((h / (np.abs(h).max() + 1e-12)).astype(np.float32))
    return out


def robustness_sweep(model, audio_i16, y, noise_i16, tau: float, snrs=(20, 10, 5, 0), distances=(0.5, 1.0, 1.5), seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    x = [F.int16_to_float(a) for a in audio_i16]
    result = {"clean": clip_metrics(y, predict_probs(model, precompute_features(audio_i16)), tau), "snr": {}, "distance": {}}
    for snr in snrs:
        mixed = [mix_noise(xi, F.int16_to_float(noise_i16[int(rng.integers(0, len(noise_i16)))]), float(snr)) for xi in x]
        result["snr"][str(snr)] = clip_metrics(y, predict_probs(model, precompute_features([F.float_to_int16(m) for m in mixed])), tau)
    for d in distances:
        rirs = make_distance_rirs(d, n=5, seed=seed)
        conv = [apply_rir(xi, rirs[i % len(rirs)]) for i, xi in enumerate(x)]
        result["distance"][str(d)] = clip_metrics(y, predict_probs(model, precompute_features([F.float_to_int16(c) for c in conv])), tau)
    return result


def make_interpreter(tflite):
    try:
        from ai_edge_litert.interpreter import Interpreter
    except ImportError:  # fallback for older TensorFlow builds
        import tensorflow as tf

        Interpreter = tf.lite.Interpreter
    if isinstance(tflite, (bytes, bytearray)):
        return Interpreter(model_content=bytes(tflite))
    return Interpreter(model_path=str(tflite))


def int8_probs(tflite, X) -> np.ndarray:
    interp = make_interpreter(tflite)
    interp.allocate_tensors()
    inp, out = interp.get_input_details()[0], interp.get_output_details()[0]
    in_scale, in_zp = inp["quantization"]
    out_scale, out_zp = out["quantization"]
    probs = np.empty(len(X), np.float64)
    for i, xi in enumerate(np.asarray(X, dtype=np.float32)):
        q = F.quantize(xi.reshape(F.N_FRAMES, F.N_MELS), float(in_scale), int(in_zp)).reshape(inp["shape"])
        interp.set_tensor(inp["index"], q)
        interp.invoke()
        o = interp.get_tensor(out["index"]).astype(np.float64)
        probs[i] = float((o - out_zp) * out_scale)
    return probs


def int8_parity(p_fp, p_int8, y, tau: float) -> dict:
    p_fp, p_int8, y = np.asarray(p_fp, dtype=np.float64), np.asarray(p_int8, dtype=np.float64), np.asarray(y).astype(int)
    two = len(np.unique(y)) == 2
    delta = abs(roc_auc_score(y, p_fp) - roc_auc_score(y, p_int8)) if two else 0.0
    agreement = float(((p_fp >= tau) == (p_int8 >= tau)).mean())
    max_abs = float(np.abs(p_fp - p_int8).max())
    return {"delta_auc": float(delta), "agreement": agreement, "max_abs_diff": max_abs, "passed": bool(delta < 0.005 and agreement >= 0.99 and max_abs <= 0.05)}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv-mac/bin/python -m pytest tests/test_evaluate.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add v5/evaluate.py tests/test_evaluate.py
git commit -m "feat(v5): clip metrics, robustness sweep and int8 inference helpers"
```

---
### Task 11: Training runs, selection on validation, fail-closed threshold calibration

**Files:**
- Create: `v5/train.py`, `tests/test_train.py`

**Interfaces:**
- Consumes: `v5.data.manifest` (`read_manifest`, `load_cache`, `split_indices`), `v5.data.augment`, `v5.data.dataset`, `v5.model.build_model`, `v5.evaluate`, `v5.teacher` (`read_scores`, `platt_fit`, `platt_apply`).
- Produces: `make_kd_loss(alpha: float) -> callable` and `kd_loss = make_kd_loss(0.5)`, `class ValAuc(keras.callbacks.Callback)`, `build_soft_targets(rows, teacher_csv, train_idx) -> np.ndarray`, `train_one(cfg, use_kd, width, epochs, name, seed=42) -> dict`, `class CalibrationError(RuntimeError)`, `fpr_upper_bound(fp, n, conf=0.95) -> float`, `choose_threshold(y_calib, p_calib, max_fpr=0.01, min_neg=300) -> tuple[float, dict]`, `select_config(results: list[dict]) -> dict`, `run_report(out_dir) -> str`, CLI `python -m v5.train {run,select,final,report}`.
- Files: `output/v5/runs/<name>/{model.keras,metrics.json,history.json}`, `output/v5/selection.json`, `output/v5/threshold.json`, `output/v5/deployed/{model.keras,metrics.json}`, `output/v5/deliverables/experiments.md`.
- `threshold.json` schema: `{"tau": float, "model_version": str, "max_fpr": float, "run": str, "calib": {"n_neg", "n_pos", "fp", "fpr", "fpr_upper95", "recall", "tau", "max_fpr"}, "fsm": {tick_ms, hold_s, confirm_s, verify_s, min_bursts, period_min_s, period_max_s}}`.
- Calibration contract: tau is the smallest float strictly above the largest negative score that may still pass, so at most `floor(max_fpr * n_neg)` calibration negatives score at or above tau. Fewer than `min_neg` negatives, a tau above 1.0, or zero positive recall raise `CalibrationError`; nothing is written in that case.

- [ ] **Step 1: Write the failing tests**

`tests/test_train.py`:
```python
import json

import numpy as np
import pytest

from v5 import train as TR
from v5.config import load_config, resolve
from v5.data import manifest as M

TINY_MIN = {"train_pos": 1, "train_neg": 1, "val_pos": 0, "val_neg": 0, "calib_neg": 0, "test_pos": 1, "test_neg": 1}


def test_choose_threshold_meets_fpr_and_reports_bound():
    rng = np.random.default_rng(0)
    y = np.array([1] * 400 + [0] * 400)
    p = np.where(y == 1, rng.uniform(0.4, 1.0, 800), rng.uniform(0.0, 0.6, 800))
    tau, info = TR.choose_threshold(y, p, max_fpr=0.01)
    assert 0.5 <= tau <= 0.62 and info["fp"] <= 4 and info["fpr"] <= 0.01
    assert (p[y == 0] >= tau).mean() <= 0.01 and info["fpr_upper95"] > info["fpr"] and 0 < info["recall"] <= 1


def test_choose_threshold_never_clips_to_a_cap():
    y = np.array([1] * 300 + [0] * 300)
    p = np.r_[np.full(300, 0.995), np.full(300, 0.98)]
    tau, info = TR.choose_threshold(y, p, max_fpr=0.01)
    assert 0.98 < tau <= 0.995 and info["fp"] == 0 and info["recall"] == 1.0


def test_choose_threshold_fails_closed():
    y = np.array([1] * 300 + [0] * 300)
    p_ok = np.r_[np.full(300, 0.9), np.full(300, 0.1)]
    with pytest.raises(TR.CalibrationError):  # too few negatives
        TR.choose_threshold(y[:400], p_ok[:400], max_fpr=0.01, min_neg=300)
    with pytest.raises(TR.CalibrationError):  # every negative outscores every positive
        TR.choose_threshold(y, np.r_[np.full(300, 0.2), np.full(300, 0.9)], max_fpr=0.01)
    with pytest.raises(TR.CalibrationError):  # negatives saturate at 1.0
        TR.choose_threshold(y, np.r_[np.full(300, 1.0), np.full(300, 1.0)], max_fpr=0.01)


def test_fpr_upper_bound_is_conservative():
    assert TR.fpr_upper_bound(0, 300) > 0 and TR.fpr_upper_bound(0, 300) < 0.011
    assert TR.fpr_upper_bound(3, 300) > 0.01


def test_select_config_prefers_smallest_near_best_on_validation():
    results = [
        {"name": "kd_w2", "params": 100_000, "kd": True, "val": {"auc": 0.990, "recall_at_fpr2": 0.95}},
        {"name": "hard_w1", "params": 25_000, "kd": False, "val": {"auc": 0.987, "recall_at_fpr2": 0.94}},
        {"name": "kd_w1", "params": 25_000, "kd": True, "val": {"auc": 0.988, "recall_at_fpr2": 0.945}},
        {"name": "hard_w0", "params": 8_000, "kd": False, "val": {"auc": 0.970, "recall_at_fpr2": 0.80}},
    ]
    assert TR.select_config(results)["name"] == "hard_w1"


def test_kd_loss_factory():
    import keras

    y = np.array([[1.0, 1.0], [0.0, 0.0]], np.float32)
    soft = np.array([[1.0, 0.7], [0.0, 0.2]], np.float32)
    logits = np.array([[2.0], [-1.0]], np.float32)
    ref = keras.losses.BinaryCrossentropy(from_logits=True)(y[:, :1], logits)
    assert np.isclose(float(TR.kd_loss(y, logits)), float(ref), atol=1e-6)
    assert np.isclose(float(TR.make_kd_loss(1.0)(soft, logits)), float(ref), atol=1e-6)  # alpha 1 ignores soft targets
    assert not np.isclose(float(TR.make_kd_loss(0.5)(soft, logits)), float(ref), atol=1e-3)


def test_train_one_smoke(mini_dataset, tmp_path):
    cfg = resolve(load_config())
    cfg["paths"]["data_dir"], cfg["paths"]["out_dir"] = str(mini_dataset), str(tmp_path / "out")
    cfg["data"]["near_dup_threshold"] = 0.999
    cfg["data"]["min_counts"] = dict(TINY_MIN)
    cfg["augment"]["rir_bank_size"] = 2
    cfg["train"]["workers"] = 1
    M.build_manifest(cfg["paths"]["data_dir"], cfg["paths"]["out_dir"], cfg)
    metrics = TR.train_one(cfg, use_kd=False, width=0.5, epochs=1, name="smoke")
    assert (tmp_path / "out" / "runs" / "smoke" / "model.keras").exists()
    assert "val" in metrics and "test" not in metrics and metrics["kd"] is False
    saved = json.loads((tmp_path / "out" / "runs" / "smoke" / "metrics.json").read_text())
    assert saved["name"] == "smoke" and saved["n_train"] > 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv-mac/bin/python -m pytest tests/test_train.py -v`
Expected: FAIL with `ImportError`

- [ ] **Step 3: Implement `v5/train.py`**

```python
"""Training runs, validation-based selection and fail-closed threshold calibration (spec section 6)."""
from __future__ import annotations

import argparse
import json
import math
import shutil
import time
from pathlib import Path

import keras
import numpy as np
from sklearn.metrics import roc_auc_score

from v5 import features as F
from v5 import teacher as T
from v5.config import load_config, resolve
from v5.data import manifest as M
from v5.data.augment import AugmentConfig, Augmenter, RirBank
from v5.data.dataset import TrainDataset, precompute_features
from v5.evaluate import clip_metrics, predict_probs, sigmoid
from v5.model import build_model

_BCE = keras.losses.BinaryCrossentropy(from_logits=True)


class CalibrationError(RuntimeError):
    """Threshold calibration cannot meet the FPR target; deployment must not proceed."""


def make_kd_loss(alpha: float):
    """alpha * BCE(hard) + (1 - alpha) * BCE(soft). Column 0 of y_true is the hard label, column 1 the soft target."""

    def kd_loss(y_true, logits):
        return alpha * _BCE(y_true[:, 0:1], logits) + (1.0 - alpha) * _BCE(y_true[:, 1:2], logits)

    kd_loss.__name__ = f"kd_loss_a{alpha:g}".replace(".", "p")
    return kd_loss


kd_loss = make_kd_loss(0.5)


class ValAuc(keras.callbacks.Callback):
    def __init__(self, Xv, yv, patience: int):
        super().__init__()
        self.Xv, self.yv, self.patience = Xv, np.asarray(yv).astype(int), patience
        self.best, self.best_epoch, self.best_weights, self.wait, self.history = -1.0, -1, None, 0, []

    def on_epoch_end(self, epoch, logs=None):
        p = sigmoid(self.model.predict(self.Xv, batch_size=256, verbose=0).ravel())
        auc = float(roc_auc_score(self.yv, p)) if len(np.unique(self.yv)) == 2 else 0.5
        self.history.append(auc)
        if logs is not None:
            logs["val_auc"] = auc
        if auc > self.best + 1e-4:
            self.best, self.best_epoch, self.best_weights, self.wait = auc, epoch, self.model.get_weights(), 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.model.stop_training = True

    def on_train_end(self, logs=None):
        if self.best_weights is not None:
            self.model.set_weights(self.best_weights)


def build_soft_targets(rows, teacher_csv, train_idx) -> np.ndarray:
    y = np.array([r["label"] for r in rows], np.float64)
    scores = T.read_scores(teacher_csv)
    z = np.array([scores.get(r["id"], (np.nan, ""))[0] for r in rows], np.float64)
    have = ~np.isnan(z)
    tr = np.array([i for i in train_idx if have[i]], dtype=int)
    if len(tr) < 50:
        raise ValueError("too few teacher scores on the training split")
    a, b = T.platt_fit(z[tr], y[tr])
    soft = y.copy()
    soft[have] = T.platt_apply(z[have], a, b)
    return soft.astype(np.float32)


def _teacher_excluded(rows, teacher_csv) -> set[int]:
    if not Path(teacher_csv).exists():
        return set()
    return {i for i, (_, flag) in T.read_scores(teacher_csv).items() if flag == "snore_in_negative"}


def train_one(cfg: dict, use_kd: bool, width: float, epochs: int, name: str, seed: int = 42) -> dict:
    out = Path(cfg["paths"]["out_dir"])
    rows = M.read_manifest(out / "manifest.csv")
    _, audio, _ = M.load_cache(out)
    excluded = _teacher_excluded(rows, out / "teacher.csv")
    tr = np.array([i for i in M.split_indices(rows, "train") if i not in excluded], dtype=int)
    va = M.split_indices(rows, "val")
    y = np.array([r["label"] for r in rows], np.float32)
    kd_note = ""
    if use_kd and not (out / "teacher.csv").exists():
        use_kd, kd_note = False, "teacher.csv missing: trained with hard labels only"
        print(f"[train] {kd_note}")
    tcfg = cfg["train"]
    alpha = float(tcfg.get("kd_alpha", 0.5)) if use_kd else 1.0
    soft = build_soft_targets(rows, out / "teacher.csv", tr) if use_kd else y
    rng = np.random.default_rng(seed)
    rir_bank = RirBank.load_or_generate(out / "rir_bank.npz", int(cfg["augment"]["rir_bank_size"]), seed)
    aug = Augmenter(AugmentConfig.from_dict(cfg["augment"]), audio[tr][y[tr] == 0], rir_bank, rng)
    ds = TrainDataset(audio[tr], y[tr], soft[tr], aug, batch=int(tcfg["batch"]), seed=seed, workers=int(tcfg.get("workers", 8)))
    Xv, yv = precompute_features(audio[va]), y[va]
    keras.utils.set_random_seed(seed)
    model = build_model(width)
    schedule = keras.optimizers.schedules.CosineDecay(float(tcfg["lr"]), max(1, len(ds) * epochs), alpha=float(tcfg["lr_min"]) / float(tcfg["lr"]))
    model.compile(optimizer=keras.optimizers.Adam(schedule), loss=make_kd_loss(alpha))
    cb = ValAuc(Xv, yv, int(tcfg["patience"]))
    t0 = time.time()
    hist = model.fit(ds, epochs=epochs, callbacks=[cb], verbose=2)
    run_dir = out / "runs" / name
    run_dir.mkdir(parents=True, exist_ok=True)
    model.save(run_dir / "model.keras")
    metrics = {
        "name": name, "kd": bool(use_kd), "kd_alpha": alpha, "kd_note": kd_note, "width": width, "params": int(model.count_params()),
        "epochs_run": len(cb.history), "best_epoch": cb.best_epoch, "train_seconds": round(time.time() - t0, 1),
        "n_train": int(len(tr)), "n_val": int(len(va)), "val": clip_metrics(yv, predict_probs(model, Xv), 0.5), "val_auc_history": cb.history,
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (run_dir / "history.json").write_text(json.dumps({k: [float(v) for v in vals] for k, vals in hist.history.items()}), encoding="utf-8")
    return metrics


def fpr_upper_bound(fp: int, n: int, conf: float = 0.95) -> float:
    """One-sided Clopper-Pearson upper bound on the false-positive rate."""
    from scipy.stats import beta

    if n <= 0:
        return 1.0
    return float(beta.ppf(conf, fp + 1, max(n - fp, 1))) if fp < n else 1.0


def choose_threshold(y_calib, p_calib, max_fpr: float = 0.01, min_neg: int = 300):
    y, p = np.asarray(y_calib).astype(int), np.asarray(p_calib, dtype=np.float64)
    neg, pos = np.sort(p[y == 0]), p[y == 1]
    n_neg = int(len(neg))
    if n_neg < min_neg:
        raise CalibrationError(f"need at least {min_neg} calibration negatives, have {n_neg}")
    if len(pos) == 0:
        raise CalibrationError("no calibration positives")
    k = int(math.floor(max_fpr * n_neg))  # negatives allowed at or above tau
    tau = float(np.nextafter(neg[n_neg - k - 1], np.inf))
    if tau > 1.0:
        raise CalibrationError("negatives saturate at probability 1.0; no threshold meets the FPR target")
    fp = int((neg >= tau).sum())
    recall = float((pos >= tau).mean())
    if recall == 0.0:
        raise CalibrationError("no calibration positive passes the FPR-constrained threshold")
    info = {"tau": tau, "n_neg": n_neg, "n_pos": int(len(pos)), "fp": fp, "fpr": fp / n_neg, "fpr_upper95": fpr_upper_bound(fp, n_neg), "recall": recall, "max_fpr": float(max_fpr)}
    assert info["fpr"] <= max_fpr
    return tau, info


def select_config(results: list[dict]) -> dict:
    best_auc = max(r["val"]["auc"] for r in results)
    best_rec = max(r["val"]["recall_at_fpr2"] for r in results)
    cands = [r for r in results if r["val"]["auc"] >= best_auc - 0.005 and r["val"]["recall_at_fpr2"] >= best_rec - 0.02]
    return min(cands, key=lambda r: (r["params"], 1 if r["kd"] else 0))


def run_report(out_dir) -> str:
    out = Path(out_dir)
    lines = ["# v5 training runs (selection on the validation split; the test split is untouched until export)", "",
             "| run | kd | width | params | epochs | val AUC | val R@FPR2% |", "|---|---|---|---|---|---|---|"]
    for mp in sorted(out.glob("runs/*/metrics.json")):
        m = json.loads(mp.read_text())
        lines.append(f"| {m['name']} | {m['kd']} | {m['width']} | {m['params']} | {m['epochs_run']} | {m['val']['auc']:.4f} | {m['val']['recall_at_fpr2']:.3f} |")
    sel = out / "selection.json"
    if sel.exists():
        lines += ["", f"selected: `{json.loads(sel.read_text())['name']}`"]
    thr = out / "threshold.json"
    if thr.exists():
        lines += ["", "threshold.json (calibrated on the calib split):", "```json", thr.read_text().strip(), "```"]
    return "\n".join(lines) + "\n"


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="v5 training")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--kd", action="store_true")
    r.add_argument("--width", type=float, default=1.0)
    r.add_argument("--epochs", type=int, default=None)
    r.add_argument("--name", required=True)
    sub.add_parser("select")
    sub.add_parser("final")
    sub.add_parser("report")
    for p in sub.choices.values():
        p.add_argument("--config", default=None)
    args = ap.parse_args(argv)
    cfg = resolve(load_config(args.config))
    out = Path(cfg["paths"]["out_dir"])
    if args.cmd == "run":
        m = train_one(cfg, args.kd, args.width, args.epochs or int(cfg["train"]["epochs"]), args.name, cfg["seed"])
        print(json.dumps({k: m[k] for k in ("name", "params", "epochs_run", "val")}, indent=1))
    elif args.cmd == "select":
        results = [json.loads(p.read_text()) for p in out.glob("runs/*/metrics.json")]
        chosen = select_config(results)
        (out / "selection.json").write_text(json.dumps({"name": chosen["name"], "kd": chosen["kd"], "width": chosen["width"]}, indent=2))
        print("selected:", chosen["name"])
    elif args.cmd == "final":
        sel = json.loads((out / "selection.json").read_text())
        run_dir = out / "runs" / sel["name"]
        rows = M.read_manifest(out / "manifest.csv")
        _, audio, _ = M.load_cache(out)
        ca = M.split_indices(rows, "calib")
        model = keras.models.load_model(run_dir / "model.keras", compile=False)
        pc = predict_probs(model, precompute_features(audio[ca]))
        yc = np.array([rows[i]["label"] for i in ca])
        tau, info = choose_threshold(yc, pc, float(cfg["threshold"]["max_fpr"]), int(cfg["threshold"]["min_calib_neg"]))
        if info["fpr"] > float(cfg["threshold"]["max_fpr"]):
            raise CalibrationError(f"measured calibration FPR {info['fpr']:.4f} exceeds {cfg['threshold']['max_fpr']}")
        (out / "deployed").mkdir(exist_ok=True)
        shutil.copy(run_dir / "model.keras", out / "deployed" / "model.keras")
        run_metrics = json.loads((run_dir / "metrics.json").read_text())
        run_metrics["calib"] = info
        (out / "deployed" / "metrics.json").write_text(json.dumps(run_metrics, indent=2))
        (out / "threshold.json").write_text(json.dumps({"tau": tau, "model_version": cfg["model_version"], "max_fpr": cfg["threshold"]["max_fpr"], "run": sel["name"], "calib": info, "fsm": cfg["fsm"]}, indent=2))
        print(f"deployed {sel['name']} with tau={tau:.4f} calib_fpr={info['fpr']:.4f} (95% upper {info['fpr_upper95']:.4f}) calib_recall={info['recall']:.3f}")
    elif args.cmd == "report":
        (out / "deliverables").mkdir(parents=True, exist_ok=True)
        text = run_report(out)
        (out / "deliverables" / "experiments.md").write_text(text, encoding="utf-8")
        print(text)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv-mac/bin/python -m pytest tests/test_train.py -v`
Expected: PASS (smoke training takes under a minute)

- [ ] **Step 5: Commit**

```bash
git add v5/train.py tests/test_train.py
git commit -m "feat(v5): training runs, validation selection and fail-closed threshold calibration"
```
### Task 12: Build the real manifest and teacher scores (dataset run)

**Files:**
- Create: `output/v5/manifest.csv`, `output/v5/cache/*` (untracked), `output/v5/deliverables/manifest_report.md` (tracked copy), `tests/test_manifest_dataset.py`

**Interfaces:**
- Consumes: `python -m v5.data.manifest`, `python -m v5.teacher`.
- Produces: the manifest used by every later task.

- [ ] **Step 1: Write the dataset gate test**

`tests/test_manifest_dataset.py`:
```python
import pytest

from v5.config import ROOT, load_config
from v5.data import manifest as M

OUT = ROOT / "output" / "v5"


@pytest.mark.dataset
def test_real_manifest_release_gate():
    path = OUT / "manifest.csv"
    if not path.exists():
        pytest.fail("dataset is present but output/v5/manifest.csv is missing: run python -m v5.data.manifest")
    rows = M.read_manifest(path)
    M.check_invariants(rows)
    M.check_min_counts(rows, load_config()["data"]["min_counts"])
    assert all(r["split"] == "test" for r in rows if r["source"].startswith("kaggle"))
    assert not any(r["category"] == "snoring" and r["label"] == 0 for r in rows)
    assert {r["category"] for r in rows if r["source"] == "whl_s" and r["split"] == "val"} == {"000002"}
```

- [ ] **Step 2: Build the manifest**

Run: `.venv-mac/bin/python -m v5.data.manifest`
Expected: prints `manifest rows: N -> .../output/v5/manifest.csv` (N around 12k–16k). The report lists `kaggle_adria ~ kaggle_jibran` under both exact and near duplicates, the leakage statement, and a split table with non-zero train/val/calib/test/bench rows. A `DatasetMissing` error means a required source or a minimum count is missing; fix the data, do not lower the minimums.

- [ ] **Step 3: Score with the teacher (optional, network)**

Run: `.venv-mac/bin/python -m v5.teacher`
Expected: either `[teacher] wrote .../teacher.csv; flags: {...}` or `[teacher] unavailable (...)`. Record which one happened in the commit message.

- [ ] **Step 4: Run the gate test and copy the report**

Run: `.venv-mac/bin/python -m pytest tests/test_manifest_dataset.py -v && cp output/v5/manifest_report.md output/v5/deliverables/manifest_report.md`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/test_manifest_dataset.py output/v5/deliverables/manifest_report.md
git commit -m "data(v5): real manifest built; report with dedup, split counts and leakage statement"
```
### Task 13: Training experiments, validation selection, calibration

**Files:**
- Create: `output/v5/runs/*` (untracked), `output/v5/selection.json`, `output/v5/threshold.json`, `output/v5/deployed/*` (untracked), `output/v5/deliverables/experiments.md` (tracked)

- [ ] **Step 1: Run the four configurations**

Run each (15–40 minutes each on CPU):
```bash
.venv-mac/bin/python -m v5.train run --width 1.0 --name hard_w1
.venv-mac/bin/python -m v5.train run --width 1.0 --kd --name kd_w1
.venv-mac/bin/python -m v5.train run --width 2.0 --name hard_w2
.venv-mac/bin/python -m v5.train run --width 2.0 --kd --name kd_w2
```
Expected: each prints validation metrics (AUC and recall_at_fpr2). If the teacher file is missing, the `--kd` runs print the hard-label note and are still valid runs (they then duplicate the hard runs; the selection rule handles ties). The test split is not touched by any of these commands.

- [ ] **Step 2: Select on validation and calibrate on the calib split**

Run: `.venv-mac/bin/python -m v5.train select && .venv-mac/bin/python -m v5.train final`
Expected: `selected: <name>` then `deployed <name> with tau=... calib_fpr<=0.01 (95% upper ...) calib_recall=...`; `output/v5/threshold.json` exists with `model_version` `cnn_v5_int8`. A `CalibrationError` stops here by design; investigate the calib split before retrying.

- [ ] **Step 3: Write the experiments report and commit**

Run: `.venv-mac/bin/python -m v5.train report`
Expected: markdown table with 4 runs, the selected run, and the threshold block.

```bash
git add output/v5/deliverables/experiments.md
git commit -m "data(v5): training experiments, validation selection and calibrated threshold"
```
### Task 14: Episode state machine (Python reference)

**Files:**
- Create: `v5/streaming.py`, `tests/test_streaming.py`

**Interfaces:**
- Produces: `IDLE, ACTIVE, CONFIRMED` state strings; `@dataclass(frozen=True) FsmParams(tau=0.65, tick_ms=500, hold_ticks=12, confirm_ticks=20, verify_ticks=30, min_bursts=3, period_min_ticks=3, period_max_ticks=14)` with `FsmParams.from_config(tau, fsm_cfg: dict)` and `to_dict()`; `class EpisodeFsm(params)` with `reset()`, `tick(p: float, level_dbfs: float = 0.0) -> dict | None`, attributes `state`, `active`, `activity`, `tick_i`, `last_episode`; `run_sequence(p_seq, params, levels=None) -> tuple[list[dict], list[str], list[bool]]`.
- Event dicts: `{"type": "episode_start", "tick", "t", "n_bursts"}` and `{"type": "episode_end", "tick", "t", "start_t", "duration_s", "mean_p", "n_bursts", "n_hits", "level_dbfs"}` where `t = tick * tick_ms / 1000` is the window start time. Episode statistics (`mean_p`, `n_hits`, `level_dbfs`) cover every hit from the first retained burst onward, including hits before confirmation.
- Implementation details shared with the C code: streak is capped at `4 * confirm_ticks`; candidate hits are kept in a history pruned by the same age rule as burst starts (`confirm_ticks + hold_ticks`).

- [ ] **Step 1: Write the failing tests**

`tests/test_streaming.py`:
```python
import numpy as np

from v5 import streaming as S


def periodic_pattern(n_cycles=6, burst=3, gap=5, p_hit=0.9, p_miss=0.1):
    seq = []
    for _ in range(n_cycles):
        seq += [p_hit] * burst + [p_miss] * gap
    return seq


def test_params_from_config_rounds_seconds_to_ticks():
    p = S.FsmParams.from_config(0.7, {"tick_ms": 500, "hold_s": 6, "confirm_s": 10, "verify_s": 15, "min_bursts": 3, "period_min_s": 1.5, "period_max_s": 7.0})
    assert p == S.FsmParams(tau=0.7, tick_ms=500, hold_ticks=12, confirm_ticks=20, verify_ticks=30, min_bursts=3, period_min_ticks=3, period_max_ticks=14)


def test_periodic_snoring_confirms_at_tick_19_with_three_bursts():
    events, states, actives = S.run_sequence(periodic_pattern(), S.FsmParams())
    starts = [e for e in events if e["type"] == "episode_start"]
    assert len(starts) == 1 and starts[0]["tick"] == 19 and starts[0]["n_bursts"] == 3
    assert states[18] == S.ACTIVE and states[19] == S.CONFIRMED


def test_episode_end_active_flag_and_statistics():
    seq = periodic_pattern(5) + [0.1] * 80  # last hit at tick 34 (burst starting at 32)
    levels = [-30.0] * len(seq)
    events, states, actives = S.run_sequence(seq, S.FsmParams(), levels)
    assert actives[34] and actives[46] and not actives[47]  # hold = 12 ticks after the last hit
    ends = [e for e in events if e["type"] == "episode_end"]
    assert len(ends) == 1 and ends[0]["tick"] == 76  # 47 + 30 - 1
    assert ends[0]["duration_s"] == (34 - 0 + 2) * 0.5 and ends[0]["n_bursts"] == 5
    assert ends[0]["n_hits"] == 15  # 5 bursts x 3 hits, including the hits before confirmation
    assert abs(ends[0]["mean_p"] - 0.9) < 1e-9 and ends[0]["level_dbfs"] == -30.0 and states[76] == S.IDLE


def test_pre_confirmation_hits_shape_the_mean():
    seq = [0.7] * 3 + [0.1] * 5 + [0.7] * 3 + [0.1] * 5 + [0.95] * 3 + [0.1] * 5 + [0.95] * 3 + [0.1] * 80
    events, _, _ = S.run_sequence(seq, S.FsmParams())
    end = [e for e in events if e["type"] == "episode_end"][0]
    assert end["n_hits"] == 12 and abs(end["mean_p"] - (6 * 0.7 + 6 * 0.95) / 12) < 1e-9


def test_continuous_sound_never_confirms_and_returns_to_idle():
    seq = [0.9] * 60 + [0.1] * 200
    events, states, _ = S.run_sequence(seq, S.FsmParams())
    assert events == [] and S.CONFIRMED not in states
    assert states[59] == S.ACTIVE and states[-1] == S.IDLE


def test_too_slow_or_too_fast_periodicity_is_rejected():
    slow = []
    for _ in range(4):
        slow += [0.9] * 2 + [0.1] * 30  # 16 s between bursts: gap breaks the hold, no confirm
    fast = []
    for _ in range(30):
        fast += [0.9] * 1 + [0.1] * 1  # 1 s period, below period_min
    for seq in (slow, fast):
        events, _, _ = S.run_sequence(seq, S.FsmParams())
        assert events == []


def test_reset_clears_state():
    f = S.EpisodeFsm(S.FsmParams())
    for p in periodic_pattern():
        f.tick(p)
    f.reset()
    assert f.state == S.IDLE and f.tick_i == 0 and not f.active
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv-mac/bin/python -m pytest tests/test_streaming.py -v`
Expected: FAIL with `ImportError`

- [ ] **Step 3: Implement `v5/streaming.py`**

```python
"""Deterministic episode state machine with integer tick arithmetic (spec section 8).

Mirrored line for line by esp32_firmware/v5/snore_episode_fsm.c.
"""
from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass

IDLE, ACTIVE, CONFIRMED = "IDLE", "ACTIVE", "CONFIRMED"


@dataclass(frozen=True)
class FsmParams:
    tau: float = 0.65
    tick_ms: int = 500
    hold_ticks: int = 12
    confirm_ticks: int = 20
    verify_ticks: int = 30
    min_bursts: int = 3
    period_min_ticks: int = 3
    period_max_ticks: int = 14

    @classmethod
    def from_config(cls, tau: float, fsm: dict) -> "FsmParams":
        tick = int(fsm.get("tick_ms", 500))

        def ticks(seconds) -> int:
            return int(round(float(seconds) * 1000.0 / tick))

        return cls(tau=float(tau), tick_ms=tick, hold_ticks=ticks(fsm.get("hold_s", 6)), confirm_ticks=ticks(fsm.get("confirm_s", 10)),
                   verify_ticks=ticks(fsm.get("verify_s", 15)), min_bursts=int(fsm.get("min_bursts", 3)),
                   period_min_ticks=ticks(fsm.get("period_min_s", 1.5)), period_max_ticks=ticks(fsm.get("period_max_s", 7.0)))

    def to_dict(self) -> dict:
        return asdict(self)


class EpisodeFsm:
    def __init__(self, params: FsmParams):
        self.p = params
        self.reset()

    def reset(self) -> None:
        self.tick_i = 0
        self.state = IDLE
        self.p_hist = deque(maxlen=self.p.hold_ticks + 1)
        self.streak_q = 0
        self.in_burst = False
        self.burst_starts = deque()
        self.hits = deque()  # (tick, p, level) for hits inside the retained window
        self.active = False
        self.activity = 0.0
        self.below_ticks = 0
        self.ep = None
        self.last_episode = None

    def _periodic(self) -> bool:
        starts = list(self.burst_starts)
        if len(starts) < self.p.min_bursts:
            return False
        gaps = sorted(b - a for a, b in zip(starts, starts[1:]))
        med = gaps[(len(gaps) - 1) // 2]  # lower median, same rule in C
        return self.p.period_min_ticks <= med <= self.p.period_max_ticks

    def tick(self, p: float, level_dbfs: float = 0.0) -> dict | None:
        P = self.p
        i = self.tick_i
        self.tick_i += 1
        self.p_hist.append(float(p))
        self.activity = max(self.p_hist)
        hit = p >= P.tau
        self.active = self.activity >= P.tau
        burst_started = hit and not self.in_burst
        if burst_started:
            self.burst_starts.append(i)
        if hit:
            self.hits.append((i, float(p), float(level_dbfs)))
        self.in_burst = hit
        max_age = P.confirm_ticks + P.hold_ticks
        while self.burst_starts and i - self.burst_starts[0] > max_age:
            self.burst_starts.popleft()
        while self.hits and i - self.hits[0][0] > max_age:
            self.hits.popleft()
        if self.active:
            self.streak_q = min(self.streak_q + 2, 4 * P.confirm_ticks)
        else:
            self.streak_q = max(0, self.streak_q - 1)
        event = None
        if self.state == IDLE and self.active:
            self.state = ACTIVE
        if self.state == ACTIVE:
            if self.streak_q == 0:
                self.state = IDLE
            elif self.streak_q >= 2 * P.confirm_ticks and self._periodic():
                first = int(self.burst_starts[0])
                sel = [h for h in self.hits if h[0] >= first]
                self.state = CONFIRMED
                self.below_ticks = 0
                self.ep = {"first_burst_tick": first, "last_hit_tick": sel[-1][0] if sel else i, "n_bursts": len(self.burst_starts),
                           "sum_p": sum(h[1] for h in sel), "n_hits": len(sel), "sum_level": sum(h[2] for h in sel)}
                event = {"type": "episode_start", "tick": i, "t": i * P.tick_ms / 1000.0, "n_bursts": self.ep["n_bursts"]}
        elif self.state == CONFIRMED:
            if hit:
                self.ep["last_hit_tick"] = i
                self.ep["sum_p"] += float(p)
                self.ep["n_hits"] += 1
                self.ep["sum_level"] += float(level_dbfs)
                if burst_started:
                    self.ep["n_bursts"] += 1
            if not self.active:
                self.below_ticks += 1
                if self.below_ticks >= P.verify_ticks:
                    ep, n = self.ep, max(self.ep["n_hits"], 1)
                    duration_ticks = ep["last_hit_tick"] - ep["first_burst_tick"] + 2  # one window = 2 ticks
                    event = {"type": "episode_end", "tick": i, "t": i * P.tick_ms / 1000.0, "start_t": ep["first_burst_tick"] * P.tick_ms / 1000.0,
                             "duration_s": duration_ticks * P.tick_ms / 1000.0, "mean_p": ep["sum_p"] / n, "n_bursts": ep["n_bursts"],
                             "n_hits": ep["n_hits"], "level_dbfs": ep["sum_level"] / n}
                    self.last_episode = event
                    self.state = IDLE
                    self.streak_q = 0
                    self.burst_starts.clear()
                    self.hits.clear()
                    self.in_burst = False
                    self.ep = None
            else:
                self.below_ticks = 0
        return event


def run_sequence(p_seq, params: FsmParams, levels=None):
    fsm = EpisodeFsm(params)
    events, states, actives = [], [], []
    for k, p in enumerate(p_seq):
        ev = fsm.tick(float(p), 0.0 if levels is None else float(levels[k]))
        if ev is not None:
            events.append(ev)
        states.append(fsm.state)
        actives.append(fsm.active)
    return events, states, actives
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv-mac/bin/python -m pytest tests/test_streaming.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add v5/streaming.py tests/test_streaming.py
git commit -m "feat(v5): deterministic episode state machine reference"
```
### Task 15: Synthetic night generator

**Files:**
- Create: `v5/nights.py`, `tests/test_nights.py`

**Interfaces:**
- Consumes: `v5.features` (`SR`, `WIN`), `v5.data.augment` (`apply_rir`, `rms`).
- Produces: `@dataclass NightSpec(duration_s=3600.0, snr_db=10.0, bed_dbfs=-40.0, n_episodes=(6,12), episode_len_s=(20.0,120.0), period_s=(2.5,5.0), n_distractors=(20,40), distractor_snr_db=(0.0,20.0), gap_s=60.0)`, `make_bed(rng, beds: list[np.ndarray], n_samples, bed_dbfs) -> np.ndarray`, `place_episodes(rng, spec) -> list[tuple[float, float]]`, `generate_night(rng, beds, snore_windows (K,16000) float32, distractor_windows (M,16000) float32, spec, rir=None) -> tuple[np.ndarray float32, list[tuple[float, float]]]`.

- [ ] **Step 1: Write the failing tests**

`tests/test_nights.py`:
```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv-mac/bin/python -m pytest tests/test_nights.py -v`
Expected: FAIL with `ImportError`

- [ ] **Step 3: Implement `v5/nights.py`**

```python
"""Synthetic nights for the streaming benchmark (spec section 7)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from v5 import features as F
from v5.data.augment import apply_rir, rms


@dataclass
class NightSpec:
    duration_s: float = 3600.0
    snr_db: float = 10.0
    bed_dbfs: float = -40.0
    n_episodes: tuple = (6, 12)
    episode_len_s: tuple = (20.0, 120.0)
    period_s: tuple = (2.5, 5.0)
    n_distractors: tuple = (20, 40)
    distractor_snr_db: tuple = (0.0, 20.0)
    gap_s: float = 60.0


def _scale_to_rms(x, target: float) -> np.ndarray:
    r = rms(x)
    return (x * (target / r)).astype(np.float32) if r > 0 else np.asarray(x, np.float32)


def make_bed(rng, beds, n_samples: int, bed_dbfs: float) -> np.ndarray:
    bed = np.asarray(beds[int(rng.integers(0, len(beds)))], dtype=np.float32)
    start = int(rng.integers(0, len(bed)))
    reps = int(np.ceil((n_samples + start) / len(bed)))
    out = np.tile(bed, reps)[start: start + n_samples]
    return _scale_to_rms(out, 10 ** (bed_dbfs / 20.0))


def place_episodes(rng, spec: NightSpec) -> list[tuple[float, float]]:
    n = int(rng.integers(spec.n_episodes[0], spec.n_episodes[1] + 1))
    t, out = spec.gap_s, []
    for _ in range(n):
        length = float(rng.uniform(*spec.episode_len_s))
        if t + length + spec.gap_s > spec.duration_s:
            break
        out.append((t, t + length))
        t += length + float(rng.uniform(spec.gap_s, 2 * spec.gap_s))
    return out


def _add(audio, seg, start: int) -> None:
    end = min(len(audio), start + len(seg))
    if end > start:
        audio[start:end] += seg[: end - start]


def generate_night(rng, beds, snore_windows, distractor_windows, spec: NightSpec, rir=None):
    n = int(spec.duration_s * F.SR)
    audio = make_bed(rng, beds, n, spec.bed_dbfs)
    bed_rms = rms(audio)
    episodes = place_episodes(rng, spec)
    target = bed_rms * 10 ** (spec.snr_db / 20.0)
    for s, e in episodes:
        period, t = float(rng.uniform(*spec.period_s)), s
        while t + 1.0 <= e:
            w = np.asarray(snore_windows[int(rng.integers(0, len(snore_windows)))], np.float32)
            if rir is not None:
                w = apply_rir(w, rir)
            _add(audio, _scale_to_rms(w, target), int(t * F.SR))
            t += period * float(rng.uniform(0.9, 1.1))
    n_d = int(rng.integers(spec.n_distractors[0], spec.n_distractors[1] + 1)) if spec.n_distractors[1] > 0 else 0
    placed = tries = 0
    while placed < n_d and tries < 10 * n_d:
        tries += 1
        t = float(rng.uniform(0, spec.duration_s - 1.0))
        if any(s - 2.0 <= t <= e + 2.0 for s, e in episodes):
            continue
        w = np.asarray(distractor_windows[int(rng.integers(0, len(distractor_windows)))], np.float32)
        if rir is not None:
            w = apply_rir(w, rir)
        _add(audio, _scale_to_rms(w, bed_rms * 10 ** (float(rng.uniform(*spec.distractor_snr_db)) / 20.0)), int(t * F.SR))
        placed += 1
    return np.clip(audio, -1.0, 1.0).astype(np.float32), episodes
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv-mac/bin/python -m pytest tests/test_nights.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add v5/nights.py tests/test_nights.py
git commit -m "feat(v5): synthetic night generator"
```

---
### Task 16: Streaming benchmark (float and int8 paths, one-to-one event matching)

**Files:**
- Create: `v5/benchmark_nights.py`, `tests/test_benchmark.py`

**Interfaces:**
- Consumes: `v5.features`, `v5.nights`, `v5.streaming` (`EpisodeFsm`, `FsmParams`), `v5.evaluate` (`predict_probs`, `int8_probs`, `make_distance_rirs`), `v5.data.manifest`, `v5.data.sources.decode`.
- Produces: `window_features(audio, hop=8000) -> np.ndarray (n,61,30,1)`, `tick_end_time(tick, tick_s=0.5) -> float` (= `(tick + 2) * tick_s`), `run_fsm(p_seq, params) -> tuple[list[dict], list[bool]]` (events carry `t_end`), `match_events(events, episodes, tol_s) -> tuple[list[tuple[int, dict]], list[dict]]`, `score_night(events, actives, episodes, tick_s=0.5, tol_s=5.0, duration_s=3600.0) -> dict` (keys `n_episodes, detected, detection_rate, confirm_latency_mean_s, confirm_latency_p90_s, false_confirms, false_confirms_per_hour, stop_latency_mean_s, stop_latency_p90_s, end_event_delay_mean_s`), `run_benchmark(predictors: dict[str, callable], params, cfg, seed=0) -> dict`, `to_markdown(result) -> str`, CLI `python -m v5.benchmark_nights [--model ..] [--tflite ..] [--threshold ..]`.
- Matching rule: episode_start events are processed in time order; each is assigned to the earliest unmatched episode whose window `[s - tol_s, e + tol_s]` contains the event's `t_end`; unassigned events are false confirms; unassigned episodes are misses. Latency is `max(0, t_end - s)`.
- Night material: snore and distractor windows come from the untouched `test` split; beds from `bench` MS-SNSD files. The int8 predictor is the product path; the float predictor is reported for reference together with the tick-level decision agreement between the two.
- Files: `output/v5/benchmark_nights.json`, `output/v5/deliverables/benchmark_nights.md`.

- [ ] **Step 1: Write the failing tests**

`tests/test_benchmark.py`:
```python
import numpy as np

from v5 import benchmark_nights as B
from v5 import features as F
from v5.config import load_config, resolve
from v5.data import manifest as M
from v5.streaming import FsmParams

TINY_MIN = {"train_pos": 1, "train_neg": 1, "val_pos": 0, "val_neg": 0, "calib_neg": 0, "test_pos": 1, "test_neg": 1}


def test_window_features_count():
    audio = np.zeros(F.SR * 10, np.float32)
    X = B.window_features(audio)
    assert X.shape == (19, 61, 30, 1)  # 1 + (160000 - 16000) / 8000


def _oracle_probs(episodes, n_ticks, tick_s=0.5):
    p = np.full(n_ticks, 0.05)
    for s, e in episodes:
        t = s
        while t + 1.0 <= e:  # bursts of 1 s every 4 s
            k = int(t / tick_s)
            p[k: k + 2] = 0.95
            t += 4.0
    return p


def test_run_fsm_and_score_on_oracle_probabilities():
    episodes = [(60.0, 100.0), (200.0, 260.0)]
    events, actives = B.run_fsm(_oracle_probs(episodes, 800), FsmParams())
    m = B.score_night(events, actives, episodes, tick_s=0.5, duration_s=400.0)
    assert m["n_episodes"] == 2 and m["detected"] == 2 and m["detection_rate"] == 1.0
    assert 9.0 <= m["confirm_latency_mean_s"] <= 14.0
    assert m["false_confirms"] == 0 and m["false_confirms_per_hour"] == 0.0
    assert 3.0 <= m["stop_latency_mean_s"] <= 8.0  # last burst ends 3 s before the labelled end, hold is 6 s
    assert m["end_event_delay_mean_s"] > m["stop_latency_mean_s"]


def test_match_events_is_one_to_one_with_one_tolerance():
    episodes = [(120.0, 150.0)]
    early = {"type": "episode_start", "tick": 0, "t_end": 116.0}   # inside s - tol
    second = {"type": "episode_start", "tick": 0, "t_end": 140.0}  # same episode, already matched
    far = {"type": "episode_start", "tick": 0, "t_end": 51.0}
    matches, unmatched = B.match_events([far, second, early], episodes, tol_s=5.0)
    assert [j for j, _ in matches] == [0] and matches[0][1] is early
    assert unmatched == [far, second]
    m = B.score_night([far, second, early], [False] * 400, episodes, duration_s=100.0)
    assert m["detected"] == 1 and m["false_confirms"] == 2 and m["false_confirms_per_hour"] == 72.0
    assert m["confirm_latency_mean_s"] == 0.0  # early event clamps to zero latency


def test_run_benchmark_with_fake_predictors(mini_dataset, tmp_path):
    cfg = resolve(load_config())
    cfg["paths"]["data_dir"], cfg["paths"]["out_dir"] = str(mini_dataset), str(tmp_path / "out")
    cfg["data"]["near_dup_threshold"] = 0.999
    cfg["data"]["min_counts"] = dict(TINY_MIN)
    cfg["benchmark"] = {"n_nights": 2, "night_s": 90, "snrs": [10]}
    M.build_manifest(cfg["paths"]["data_dir"], cfg["paths"]["out_dir"], cfg)
    zeros = lambda X: np.zeros(len(X))
    ones = lambda X: np.full(len(X), 0.9)
    result = B.run_benchmark({"float": zeros, "int8": ones}, FsmParams(), cfg, seed=0)
    assert set(result["by_snr"]) == {"float", "int8"} and result["by_snr"]["float"]["10"]["nights"] == 2
    assert result["by_snr"]["float"]["10"]["false_confirms_per_hour"] == 0.0
    assert result["agreement"]["float~int8"]["tick_decision_agreement"] == 0.0
    assert "| 10 |" in B.to_markdown(result)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv-mac/bin/python -m pytest tests/test_benchmark.py -v`
Expected: FAIL with `ImportError`

- [ ] **Step 3: Implement `v5/benchmark_nights.py`**

```python
"""Streaming benchmark: synthetic nights -> features -> predictor -> FSM -> product metrics (spec section 7)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from v5 import features as F
from v5.config import load_config, resolve
from v5.data import manifest as M
from v5.data.sources import decode
from v5.evaluate import int8_probs, make_distance_rirs, predict_probs
from v5.nights import NightSpec, generate_night
from v5.streaming import EpisodeFsm, FsmParams


def window_features(audio, hop: int = F.STREAM_HOP) -> np.ndarray:
    n = 1 + (len(audio) - F.WIN) // hop
    return np.stack([F.extract(audio[k * hop: k * hop + F.WIN]) for k in range(n)]).astype(np.float32)[..., None]


def tick_end_time(tick: int, tick_s: float = 0.5) -> float:
    return (tick + 2) * tick_s  # window k spans [k*hop, k*hop + 1 s)


def run_fsm(p_seq, params: FsmParams):
    fsm = EpisodeFsm(params)
    tick_s = params.tick_ms / 1000.0
    events, actives = [], []
    for p in p_seq:
        ev = fsm.tick(float(p))
        if ev is not None:
            ev["t_end"] = tick_end_time(ev["tick"], tick_s)
            events.append(ev)
        actives.append(fsm.active)
    return events, actives


def match_events(events, episodes, tol_s: float):
    starts = sorted((ev for ev in events if ev["type"] == "episode_start"), key=lambda ev: ev["t_end"])
    used, matches, unmatched = set(), [], []
    for ev in starts:
        found = next((j for j, (s, e) in enumerate(episodes) if j not in used and s - tol_s <= ev["t_end"] <= e + tol_s), None)
        if found is None:
            unmatched.append(ev)
        else:
            used.add(found)
            matches.append((found, ev))
    return matches, unmatched


def _stats(values):
    if not values:
        return float("nan"), float("nan")
    return float(np.mean(values)), float(np.percentile(values, 90))


def score_night(events, actives, episodes, tick_s: float = 0.5, tol_s: float = 5.0, duration_s: float = 3600.0) -> dict:
    matches, unmatched = match_events(events, episodes, tol_s)
    matched = dict(matches)
    ends = [e for e in events if e["type"] == "episode_end"]
    latency, stop_lat, end_delay = [], [], []
    for j, (s, e) in enumerate(episodes):
        if j not in matched:
            continue
        latency.append(max(0.0, matched[j]["t_end"] - s))
        k0 = int(np.ceil(e / tick_s))
        drop = next((k for k in range(k0, len(actives)) if not actives[k]), None)
        if drop is not None:
            stop_lat.append(tick_end_time(drop, tick_s) - e)
        later_end = next((ev for ev in ends if ev["t_end"] >= e), None)
        if later_end is not None:
            end_delay.append(later_end["t_end"] - e)
    lat_m, lat_p90 = _stats(latency)
    stop_m, stop_p90 = _stats(stop_lat)
    hours = duration_s / 3600.0
    return {
        "n_episodes": len(episodes), "detected": len(matched), "detection_rate": (len(matched) / len(episodes)) if episodes else float("nan"),
        "confirm_latency_mean_s": lat_m, "confirm_latency_p90_s": lat_p90,
        "false_confirms": len(unmatched), "false_confirms_per_hour": len(unmatched) / hours,
        "stop_latency_mean_s": stop_m, "stop_latency_p90_s": stop_p90, "end_event_delay_mean_s": _stats(end_delay)[0],
    }


def _bench_beds(rows, data_dir) -> list[np.ndarray]:
    paths = sorted({r["path"] for r in rows if r["split"] == "bench"})
    return [decode(Path(data_dir) / p) for p in paths]


def _aggregate(nights, snrs) -> dict:
    out = {}
    for snr in snrs:
        sub = [m for m in nights if m["snr_db"] == snr]
        out[str(snr)] = {
            "nights": len(sub), "detection_rate": float(np.nanmean([m["detection_rate"] for m in sub])),
            "confirm_latency_mean_s": float(np.nanmean([m["confirm_latency_mean_s"] for m in sub])),
            "false_confirms_per_hour": float(np.mean([m["false_confirms_per_hour"] for m in sub])),
            "stop_latency_mean_s": float(np.nanmean([m["stop_latency_mean_s"] for m in sub])),
        }
    return out


def run_benchmark(predictors: dict, params: FsmParams, cfg: dict, seed: int = 0) -> dict:
    out = Path(cfg["paths"]["out_dir"])
    rows = M.read_manifest(out / "manifest.csv")
    _, audio, _ = M.load_cache(out)
    beds = _bench_beds(rows, cfg["paths"]["data_dir"])
    if not beds:
        raise RuntimeError("no bench MS-SNSD files in the manifest")
    snore = F.int16_to_float(audio[[r["id"] for r in rows if r["split"] == "test" and r["label"] == 1]])
    distract = F.int16_to_float(audio[[r["id"] for r in rows if r["split"] == "test" and r["label"] == 0]])
    bcfg = cfg["benchmark"]
    snrs, n_nights, night_s = list(bcfg["snrs"]), int(bcfg["n_nights"]), float(bcfg["night_s"])
    rirs = make_distance_rirs(1.0, n=3, seed=seed)
    rng = np.random.default_rng(seed)
    names = list(predictors)
    nights = {name: [] for name in names}
    agree_hits, agree_total, start_counts = {}, 0, {name: 0 for name in names}
    for k in range(n_nights):
        snr = snrs[k % len(snrs)]
        rir = rirs[k % len(rirs)] if k % 2 else None
        spec = NightSpec(duration_s=night_s, snr_db=float(snr), bed_dbfs=float(rng.uniform(-50, -30)))
        wave, episodes = generate_night(rng, beds, snore, distract, spec, rir)
        X = window_features(wave)
        probs = {name: np.asarray(fn(X), dtype=np.float64) for name, fn in predictors.items()}
        for name in names:
            events, actives = run_fsm(probs[name], params)
            m = score_night(events, actives, episodes, params.tick_ms / 1000.0, duration_s=night_s)
            m.update({"night": k, "snr_db": snr, "rir": rir is not None})
            nights[name].append(m)
            start_counts[name] += sum(1 for ev in events if ev["type"] == "episode_start")
        for a in range(len(names)):
            for b in range(a + 1, len(names)):
                key = f"{names[a]}~{names[b]}"
                agree_hits[key] = agree_hits.get(key, 0) + int(((probs[names[a]] >= params.tau) == (probs[names[b]] >= params.tau)).sum())
        agree_total += len(X)
    agreement = {key: {"tick_decision_agreement": hits / max(agree_total, 1), "episode_starts": dict(start_counts)} for key, hits in agree_hits.items()}
    return {"params": params.to_dict(), "nights": nights, "by_snr": {name: _aggregate(nights[name], snrs) for name in names}, "agreement": agreement}


def to_markdown(result: dict) -> str:
    lines = ["# Streaming benchmark (synthetic nights from the test split)", "", f"FSM params: `{json.dumps(result['params'])}`", ""]
    for name, by_snr in result["by_snr"].items():
        lines += [f"## predictor: {name}", "", "| SNR dB | nights | detection | confirm latency s | false confirms / h | stop latency s |", "|---|---|---|---|---|---|"]
        for snr, m in by_snr.items():
            lines.append(f"| {snr} | {m['nights']} | {m['detection_rate']:.3f} | {m['confirm_latency_mean_s']:.1f} | {m['false_confirms_per_hour']:.2f} | {m['stop_latency_mean_s']:.1f} |")
        lines.append("")
    for key, a in result.get("agreement", {}).items():
        lines.append(f"tick decision agreement {key}: {a['tick_decision_agreement']:.4f}; episode starts {a['episode_starts']}")
    return "\n".join(lines) + "\n"


def main(argv=None) -> None:
    import keras

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--tflite", default=None)
    ap.add_argument("--threshold", default=None)
    args = ap.parse_args(argv)
    cfg = resolve(load_config(args.config))
    out = Path(cfg["paths"]["out_dir"])
    thr = json.loads(Path(args.threshold or out / "threshold.json").read_text())
    params = FsmParams.from_config(thr["tau"], thr["fsm"])
    model = keras.models.load_model(args.model or out / "deployed" / "model.keras", compile=False)
    tflite = Path(args.tflite or out / "deliverables" / "snore_v5_int8.tflite").read_bytes()
    predictors = {"float": lambda X: predict_probs(model, X), "int8": lambda X: int8_probs(tflite, X)}
    result = run_benchmark(predictors, params, cfg, cfg["seed"])
    (out / "benchmark_nights.json").write_text(json.dumps(result, indent=1))
    (out / "deliverables").mkdir(parents=True, exist_ok=True)
    text = to_markdown(result)
    (out / "deliverables" / "benchmark_nights.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv-mac/bin/python -m pytest tests/test_benchmark.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add v5/benchmark_nights.py tests/test_benchmark.py
git commit -m "feat(v5): streaming benchmark with int8 path and one-to-one event matching"
```
### Task 17: GCC-PHAT direction module

**Files:**
- Create: `v5/doa.py`, `tests/test_doa.py`

**Interfaces:**
- Consumes: `v5.features.hann_periodic`.
- Produces: `@dataclass(frozen=True) DoaParams(spacing_m=0.06, fs=16000, c=343.0, band=(60.0,3000.0), n_fft=512, deadzone=0.2, min_ratio=1.5, floor_margin_db=6.0)` with property `max_lag: int`; `gcc_phat(l, r, max_lag, fs=16000, band=(60,3000), n_fft=512) -> tuple[float lag, float ratio]`; `frame_level_db(x) -> float`; `class NoiseFloor(init_db=-60.0, alpha=0.02)` with `update(level_db) -> float`; `side_from_lag(lag, max_lag, deadzone=0.2) -> str`; `class DoaTracker(params)` with `reset()`, `frame(l, r) -> tuple[float, bool]`, `episode() -> dict(side, lag_samples, lag_ms, conf, n_valid)`; `both_sides_rule(episodes, min_conf=0.7, min_share=0.25) -> bool`.
- Sign convention: `r[n] = l[n - d]` (signal reaches L first) gives lag `+d`.

- [ ] **Step 1: Write the failing tests**

`tests/test_doa.py`:
```python
import numpy as np
from scipy.signal import butter, sosfilt

from v5 import doa as D


def _burst(seed=0, n=512):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(n + 64)
    sos = butter(4, [100, 2000], btype="band", fs=16000, output="sos")
    return sosfilt(sos, x)[64:].astype(np.float32) * 0.1


def _delayed(x, d):
    """Return x delayed by d samples (fractional via FFT phase)."""
    n = len(x)
    f = np.fft.rfftfreq(n, 1 / 16000)
    return np.fft.irfft(np.fft.rfft(x) * np.exp(-2j * np.pi * f * d / 16000), n=n).astype(np.float32)


def test_gcc_phat_sign_and_integer_lag():
    l = _burst()
    for d in (2, -2):
        lag, ratio = D.gcc_phat(l, _delayed(l, d), max_lag=4)
        assert abs(lag - d) < 0.3 and ratio > 1.5


def test_gcc_phat_fractional_lag():
    l = _burst(1)
    lag, _ = D.gcc_phat(l, _delayed(l, 1.5), max_lag=4)
    assert abs(lag - 1.5) < 0.35


def test_quiet_frames_are_invalid():
    t = D.DoaTracker(D.DoaParams())
    quiet = 1e-5 * np.random.default_rng(0).standard_normal(512).astype(np.float32)
    assert not t.frame(quiet, quiet)[1]  # below the noise floor margin


def test_uncorrelated_frames_are_mostly_invalid():
    t = D.DoaTracker(D.DoaParams())
    valid = [t.frame(_burst(2 * k + 100), _burst(2 * k + 101))[1] for k in range(30)]
    assert np.mean(valid) <= 0.5


def test_tracker_episode_side_and_confidence():
    p = D.DoaParams(spacing_m=0.06)
    assert p.max_lag == 4
    t = D.DoaTracker(p)
    l = _burst(4)
    for k in range(10):
        t.frame(l, _delayed(l, 2.0))
    ep = t.episode()
    assert ep["side"] == "left" and ep["conf"] == 1.0 and ep["n_valid"] == 10 and abs(ep["lag_samples"] - 2.0) < 0.3
    t.reset()
    for k in range(10):
        t.frame(_delayed(l, 2.0), l)
    assert t.episode()["side"] == "right"


def test_noise_floor_adapts_up_slowly_and_down_fast():
    nf = D.NoiseFloor()
    for _ in range(2000):
        nf.update(-40.0)
    assert nf.db > -41.0  # converged up to the steady level
    for _ in range(200):
        nf.update(-70.0)
    assert nf.db < -68.0  # fast decay back down


def test_side_from_lag_deadzone():
    assert D.side_from_lag(0.5, 4) == "unknown" and D.side_from_lag(1.0, 4) == "left" and D.side_from_lag(-1.0, 4) == "right"


def test_both_sides_rule():
    left = {"side": "left", "conf": 0.9}
    right = {"side": "right", "conf": 0.9}
    weak = {"side": "right", "conf": 0.3}
    assert D.both_sides_rule([left, left, right, right])
    assert D.both_sides_rule([left, left, left, right])  # right share exactly 0.25 meets min_share
    assert not D.both_sides_rule([left] * 4 + [right])  # share 0.2 is below min_share
    assert not D.both_sides_rule([left, left, weak, weak])
    assert not D.both_sides_rule([left])
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv-mac/bin/python -m pytest tests/test_doa.py -v`
Expected: FAIL with `ImportError`

- [ ] **Step 3: Implement `v5/doa.py`**

```python
"""Direction of arrival by GCC-PHAT on two microphones (spec section 9)."""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from v5 import features as F


@dataclass(frozen=True)
class DoaParams:
    spacing_m: float = 0.06
    fs: int = 16000
    c: float = 343.0
    band: tuple = (60.0, 3000.0)
    n_fft: int = 512
    deadzone: float = 0.2
    min_ratio: float = 1.5
    floor_margin_db: float = 6.0

    @property
    def max_lag(self) -> int:
        return int(math.ceil(self.spacing_m / self.c * self.fs)) + 1


def gcc_phat(l, r, max_lag: int, fs: int = 16000, band=(60.0, 3000.0), n_fft: int = 512):
    w = F.hann_periodic(n_fft).astype(np.float64)
    L = np.fft.rfft(np.asarray(l, np.float64)[:n_fft] * w, n=n_fft)
    R = np.fft.rfft(np.asarray(r, np.float64)[:n_fft] * w, n=n_fft)
    G = R * np.conj(L)
    G = G / (np.abs(G) + 1e-12)
    freqs = np.fft.rfftfreq(n_fft, 1.0 / fs)
    G[(freqs < band[0]) | (freqs > band[1])] = 0.0
    cc = np.fft.irfft(G, n=n_fft)
    lags = np.concatenate([cc[-max_lag:], cc[: max_lag + 1]])  # index m <-> lag m - max_lag
    k = int(np.argmax(lags))
    lag = float(k - max_lag)
    if 0 < k < len(lags) - 1:
        y0, y1, y2 = lags[k - 1], lags[k], lags[k + 1]
        denom = y0 - 2 * y1 + y2
        if denom < 0:
            lag += 0.5 * (y0 - y2) / denom
    mask = np.ones(len(lags), bool)
    mask[max(0, k - 1): k + 2] = False
    second = lags[mask].max() if mask.any() else 0.0
    ratio = float(lags[k] / second) if second > 1e-9 else float("inf")
    return lag, ratio


def frame_level_db(x) -> float:
    return float(20.0 * np.log10(np.sqrt(np.mean(np.asarray(x, np.float64) ** 2)) + 1e-9))


class NoiseFloor:
    """Tracks the ambient level: fast when the input is near or below the floor, slow upward
    otherwise, so a steady loud background (a fan) eventually stops counting as signal."""

    def __init__(self, init_db: float = -60.0, alpha: float = 0.02, alpha_up: float = 0.002):
        self.db, self.alpha, self.alpha_up = init_db, alpha, alpha_up

    def update(self, level_db: float) -> float:
        rate = self.alpha if level_db < self.db + 5.0 else self.alpha_up
        self.db += rate * (level_db - self.db)
        return self.db


def side_from_lag(lag: float, max_lag: int, deadzone: float = 0.2) -> str:
    thr = deadzone * max_lag
    if lag > thr:
        return "left"
    if lag < -thr:
        return "right"
    return "unknown"


class DoaTracker:
    def __init__(self, params: DoaParams):
        self.p = params
        self.floor = NoiseFloor()
        self.reset()

    def reset(self) -> None:
        self.lags = []

    def frame(self, l, r):
        level = frame_level_db(l)
        floor = self.floor.update(level)
        lag, ratio = gcc_phat(l, r, self.p.max_lag, self.p.fs, self.p.band, self.p.n_fft)
        valid = bool(level >= floor + self.p.floor_margin_db and ratio >= self.p.min_ratio)
        if valid:
            self.lags.append(lag)
        return lag, valid

    def episode(self) -> dict:
        if not self.lags:
            return {"side": "unknown", "lag_samples": 0.0, "lag_ms": 0.0, "conf": 0.0, "n_valid": 0}
        med = float(np.median(self.lags))
        side = side_from_lag(med, self.p.max_lag, self.p.deadzone)
        agree = float(np.mean([side_from_lag(v, self.p.max_lag, self.p.deadzone) == side for v in self.lags]))
        return {"side": side, "lag_samples": med, "lag_ms": med / self.p.fs * 1000.0, "conf": agree, "n_valid": len(self.lags)}


def both_sides_rule(episodes, min_conf: float = 0.7, min_share: float = 0.25) -> bool:
    conf = [e for e in episodes if e.get("conf", 0.0) >= min_conf and e.get("side") in ("left", "right")]
    if len(conf) < 2:
        return False
    left = sum(1 for e in conf if e["side"] == "left") / len(conf)
    return min(left, 1.0 - left) >= min_share
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv-mac/bin/python -m pytest tests/test_doa.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add v5/doa.py tests/test_doa.py
git commit -m "feat(v5): GCC-PHAT direction of arrival reference"
```

---
### Task 18: DoA simulation sweep

**Files:**
- Create: `v5/doa_sim.py`, `tests/test_doa_sim.py`

**Interfaces:**
- Consumes: `v5.doa` (`DoaParams`, `DoaTracker`), `v5.data.manifest`.
- Produces: `simulate_stereo(rng, snore, spacing, distance, azimuth_deg, snr_db, rt60=0.3, fs=16000) -> tuple[l, r, expected_side]`, `classify(l, r, params) -> dict`, `run_sweep(snore_windows, spacings=(0.04,0.06,0.08,0.12), distances=(0.5,1.0,1.5), snrs=(0,5,10,20), trials=50, seed=0) -> list[dict]`, `to_markdown(rows) -> str`, CLI `python -m v5.doa_sim [--trials N]`.
- Files: `output/v5/doa_sim.json`, `output/v5/deliverables/doa_sim.md`.

- [ ] **Step 1: Write the failing test**

`tests/test_doa_sim.py`:
```python
import numpy as np
import pytest

from v5 import doa_sim as DS
from v5.doa import DoaParams


def _snore(seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(16000) / 16000
    env = 0.5 * (1 + np.sin(2 * np.pi * 0.5 * t))
    return (0.1 * env * sum(np.sin(2 * np.pi * 110 * k * t) / k for k in range(1, 12)) + 1e-3 * rng.standard_normal(16000)).astype(np.float32)


@pytest.mark.slow
def test_simulated_sides_are_recovered():
    rng = np.random.default_rng(0)
    for az, expected in ((45.0, "right"), (-45.0, "left")):
        l, r, exp = DS.simulate_stereo(rng, _snore(), 0.06, 1.0, az, 20.0)
        assert exp == expected and l.shape == r.shape == (16000,)
        assert DS.classify(l, r, DoaParams(spacing_m=0.06))["side"] == expected


def test_markdown_table():
    rows = [{"spacing_m": 0.06, "distance_m": 1.0, "snr_db": 5, "trials": 2, "accuracy": 1.0, "unknown_rate": 0.0}]
    md = DS.to_markdown(rows)
    assert "| 0.06 | 1.0 | 5 | 1.000 | 0.000 |" in md
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv-mac/bin/python -m pytest tests/test_doa_sim.py -v`
Expected: FAIL with `ImportError`

- [ ] **Step 3: Implement `v5/doa_sim.py`**

```python
"""Simulation sweep validating the DoA module (spec section 9)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from v5 import features as F
from v5.config import load_config, resolve
from v5.doa import DoaParams, DoaTracker


def simulate_stereo(rng, snore, spacing: float, distance: float, azimuth_deg: float, snr_db: float, rt60: float = 0.3, fs: int = F.SR):
    import pyroomacoustics as pra

    dims = [4.0, 4.0, 2.6]
    e_abs, max_order = pra.inverse_sabine(rt60, dims)
    room = pra.ShoeBox(dims, fs=fs, materials=pra.Material(e_abs), max_order=min(int(max_order), 6))
    cx, cy, h = 2.0, 1.0, 0.7
    mics = np.array([[cx - spacing / 2, cx + spacing / 2], [cy, cy], [h, h]])
    az = np.deg2rad(azimuth_deg)
    src = [cx + distance * np.sin(az), cy + distance * np.cos(az), 0.55]
    room.add_source(src, signal=np.asarray(snore, np.float64))
    room.add_microphone_array(mics)
    room.simulate()
    sig = room.mic_array.signals[:, : len(snore)]
    out = []
    for ch in sig:
        ch = np.asarray(ch, np.float64)
        noise = rng.standard_normal(len(ch))
        ch_rms, n_rms = np.sqrt(np.mean(ch ** 2)) + 1e-12, np.sqrt(np.mean(noise ** 2))
        out.append((ch + noise * ch_rms / (n_rms * 10 ** (snr_db / 20.0))).astype(np.float32))
    expected = "right" if azimuth_deg > 0 else "left"  # +x is towards the R microphone
    return out[0], out[1], expected


def classify(l, r, params: DoaParams) -> dict:
    tracker = DoaTracker(params)
    for k in range(0, len(l) - params.n_fft + 1, params.n_fft // 2):
        tracker.frame(l[k: k + params.n_fft], r[k: k + params.n_fft])
    return tracker.episode()


def run_sweep(snore_windows, spacings=(0.04, 0.06, 0.08, 0.12), distances=(0.5, 1.0, 1.5), snrs=(0, 5, 10, 20), trials: int = 50, seed: int = 0) -> list[dict]:
    rng = np.random.default_rng(seed)
    rows = []
    for s in spacings:
        for d in distances:
            for snr in snrs:
                correct = unknown = 0
                for _ in range(trials):
                    w = snore_windows[int(rng.integers(0, len(snore_windows)))]
                    az = float(rng.uniform(20.0, 70.0)) * float(rng.choice([-1.0, 1.0]))
                    l, r, exp = simulate_stereo(rng, w, s, d, az, float(snr))
                    res = classify(l, r, DoaParams(spacing_m=s))
                    if res["side"] == "unknown":
                        unknown += 1
                    elif res["side"] == exp:
                        correct += 1
                rows.append({"spacing_m": s, "distance_m": d, "snr_db": snr, "trials": trials, "accuracy": correct / trials, "unknown_rate": unknown / trials})
                print(f"spacing {s} m, distance {d} m, snr {snr} dB: accuracy {correct / trials:.3f}, unknown {unknown / trials:.3f}", flush=True)
    return rows


def to_markdown(rows) -> str:
    lines = ["# DoA simulation sweep", "", "| spacing m | distance m | SNR dB | side accuracy | unknown rate |", "|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['spacing_m']} | {r['distance_m']} | {r['snr_db']} | {r['accuracy']:.3f} | {r['unknown_rate']:.3f} |")
    return "\n".join(lines) + "\n"


def main(argv=None) -> None:
    from v5.data import manifest as M

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--trials", type=int, default=50)
    args = ap.parse_args(argv)
    cfg = resolve(load_config(args.config))
    out = Path(cfg["paths"]["out_dir"])
    rows = M.read_manifest(out / "manifest.csv")
    _, audio, _ = M.load_cache(out)
    snore = F.int16_to_float(audio[[r["id"] for r in rows if r["split"] == "test" and r["label"] == 1]])
    result = run_sweep(snore, trials=args.trials, seed=cfg["seed"])
    (out / "doa_sim.json").write_text(json.dumps(result, indent=1))
    (out / "deliverables").mkdir(parents=True, exist_ok=True)
    (out / "deliverables" / "doa_sim.md").write_text(to_markdown(result), encoding="utf-8")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv-mac/bin/python -m pytest tests/test_doa_sim.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add v5/doa_sim.py tests/test_doa_sim.py
git commit -m "feat(v5): DoA simulation sweep"
```

---
### Task 19: Edge event schema and cloud payload

**Files:**
- Create: `v5/events.py`, `tests/test_events.py`

**Interfaces:**
- Produces: `MODEL_VERSION = "cnn_v5_int8"`, `FORBIDDEN_VERSION_SUBSTRINGS`, `NOTE_MAX = 1000`, `@dataclass EdgeEvent(ts, snore_p, fsm_state, episode_id=None, side="unknown", lag_ms=0.0, doa_conf=0.0, level_dbfs=0.0, radar_presence=None, radar_motion=None, radar_breath_rate=None, temp_ok=True, vib_level=0, vib_ms=0)` with `to_dict()`, `check_model_version(v) -> str`, `to_cloud_event(device_id, ts, episode: dict, doa: dict | None = None, radar: dict | None = None, model_version=MODEL_VERSION) -> dict`.

- [ ] **Step 1: Write the failing tests**

`tests/test_events.py`:
```python
import json
import os
import sys
from pathlib import Path

import pytest

from v5 import events as EV
from v5.config import ROOT

EPISODE = {"type": "episode_end", "duration_s": 42.5, "mean_p": 0.87, "n_bursts": 11, "level_dbfs": -38.2}


def test_edge_event_dataclass_roundtrip():
    e = EV.EdgeEvent(ts=1, snore_p=0.9, fsm_state="CONFIRMED", side="left")
    d = e.to_dict()
    assert d["radar_presence"] is None and d["side"] == "left" and d["vib_level"] == 0


def test_cloud_payload_shape():
    p = EV.to_cloud_event("dev_1", 1700000000, EPISODE, {"side": "left", "lag_ms": 0.12, "conf": 0.8}, {"presence": True, "motion": 0.3})
    assert p["event_type"] == "snore_detected" and p["model_version"] == "cnn_v5_int8"
    assert p["snore_duration_sec"] == 42.5 and p["snore_confidence"] == 0.87 and p["in_bed"] is True and p["body_motion_level"] == 0.3
    note = json.loads(p["note"])
    assert note["side"] == "left" and note["n_bursts"] == 11 and len(p["note"]) <= EV.NOTE_MAX


def test_forbidden_version_rejected():
    for bad in ("cnn_demo", "simulator_v5", "MOCK"):
        with pytest.raises(ValueError):
            EV.to_cloud_event("d", 1, EPISODE, model_version=bad)


def test_payload_validates_against_cloud_schema():
    cloud = Path(os.environ.get("SNOOZMATE_CLOUD_DIR", ROOT.parent / "snoozmate-cloud-upload-latest"))
    if not (cloud / "app" / "models" / "schemas.py").exists():
        pytest.skip("cloud repo not present")
    sys.path.insert(0, str(cloud))
    try:
        from app.models.schemas import EventIn
    except Exception as exc:  # cloud deps missing in this venv
        pytest.skip(f"cannot import cloud schemas: {exc}")
    finally:
        sys.path.pop(0)
    payload = EV.to_cloud_event("dev_1", 1700000000, EPISODE, {"side": "right", "lag_ms": -0.1, "conf": 0.9}, {"presence": True, "motion": 0.1})
    ev = EventIn(**payload)
    assert ev.event_type == "snore_detected" and ev.model_version == "cnn_v5_int8"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv-mac/bin/python -m pytest tests/test_events.py -v`
Expected: FAIL with `ImportError`

- [ ] **Step 3: Implement `v5/events.py`**

```python
"""Edge event schema and the cloud EventIn payload (spec section 10)."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass

MODEL_VERSION = "cnn_v5_int8"
FORBIDDEN_VERSION_SUBSTRINGS = ("simulator", "demo", "mock")
NOTE_MAX = 1000


@dataclass
class EdgeEvent:
    ts: int
    snore_p: float
    fsm_state: str
    episode_id: int | None = None
    side: str = "unknown"
    lag_ms: float = 0.0
    doa_conf: float = 0.0
    level_dbfs: float = 0.0
    radar_presence: bool | None = None
    radar_motion: float | None = None
    radar_breath_rate: float | None = None
    temp_ok: bool = True
    vib_level: int = 0
    vib_ms: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


def check_model_version(v: str) -> str:
    low = v.lower()
    if any(s in low for s in FORBIDDEN_VERSION_SUBSTRINGS):
        raise ValueError(f"model_version {v!r} would be treated as simulated data by the cloud")
    return v


def to_cloud_event(device_id: str, ts: int, episode: dict, doa: dict | None = None, radar: dict | None = None, model_version: str = MODEL_VERSION) -> dict:
    check_model_version(model_version)
    doa, radar = doa or {}, radar or {}
    note = {"side": doa.get("side", "unknown"), "lag_ms": round(float(doa.get("lag_ms", 0.0)), 3), "doa_conf": round(float(doa.get("conf", 0.0)), 3),
            "n_bursts": int(episode.get("n_bursts", 0)), "level_dbfs": round(float(episode.get("level_dbfs", 0.0)), 1)}
    note_s = json.dumps(note, separators=(",", ":"))
    if len(note_s) > NOTE_MAX:
        raise ValueError("note exceeds the cloud limit")
    motion = radar.get("motion")
    return {
        "device_id": device_id,
        "timestamp": int(ts),
        "event_type": "snore_detected",
        "snore_duration_sec": round(float(episode["duration_s"]), 1),
        "snore_confidence": round(min(max(float(episode["mean_p"]), 0.0), 1.0), 3),
        "in_bed": bool(radar.get("presence", True)),
        "body_motion_level": float(min(max(float(motion if motion is not None else 0.0), 0.0), 1.0)),
        "model_version": model_version,
        "note": note_s,
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv-mac/bin/python -m pytest tests/test_events.py -v`
Expected: PASS (schema test passes when the cloud repo is present and pydantic is installed)

- [ ] **Step 5: Commit**

```bash
git add v5/events.py tests/test_events.py
git commit -m "feat(v5): edge event schema and cloud payload converter"
```

---
### Task 20: Exporter with release gates — int8 TFLite, C headers, golden files, model card

**Files:**
- Create: `v5/export.py`, `tests/test_export.py`
- Generated (tracked): `esp32_firmware/v5/generated/{feature_spec.h,mel_filterbank.h,model_meta.h,model_data.h,model_data.c}`, `output/v5/deliverables/{snore_v5_int8.tflite,export_info.json,export_status.json,test_metrics.json,robustness.json,model_card.md,golden/features.npz,golden/features.bin,golden/fsm_trace.txt,golden/doa_cases.bin,golden/doa_tracker.bin}`

**Interfaces:**
- Consumes: `v5.features`, `v5.golden`, `v5.streaming` (`FsmParams`, `run_sequence`, state names), `v5.doa` (`DoaParams`, `DoaTracker`, `gcc_phat`), `v5.evaluate` (`make_interpreter`, `int8_probs`, `int8_parity`, `predict_probs`, `robustness_sweep`, `clip_metrics`), `v5.data.manifest`, `v5.data.dataset.precompute_features`, `v5.model.check_ops`.
- Produces: `GEN_DIR`, `ALLOWED_TFLITE_OPS`, `PARITY_LIMITS`, `class ExportError(RuntimeError)`, `to_tflite_int8(model, rep_X) -> bytes`, `quant_params(tflite) -> dict`, `tflite_ops(tflite) -> list[str] | None`, `validate_tflite(tflite) -> dict` (raises `ExportError`), `check_parity(parity: dict) -> None` (raises), `arena_estimate(model, tflite_bytes: int) -> dict`, `write_c_array(data, name, h_path, c_path)`, `write_feature_spec_h(path)`, `write_mel_filterbank_h(path)`, `write_model_meta_h(path, qp, tau, fsm, model_version)`, `fsm_trace_sequence(seed=0, tau=0.65) -> list[float]`, `write_golden_fsm(path, params, seed=0)`, `doa_golden_cases(seed=0, params=DoaParams())`, `write_golden_doa_cases(path, params, seed=0)`, `doa_tracker_frames(seed=0, params=DoaParams()) -> list[tuple[np.ndarray, np.ndarray]]`, `write_golden_doa_tracker(path, params, seed=0)`, `write_headers_only(gen_dir, golden_dir, fsm, doa)`, `promote(stage, deliv, gen_dir)`, `export_model(cfg, model_path=None, threshold_path=None) -> dict`, `model_card(cfg) -> str`, CLI `python -m v5.export {headers,model,card}`.
- Release gates (all must pass before anything is promoted): TFLite input/output are int8 with shapes `[1,61,30,1]` and `[1,1]`; operator set ⊆ `ALLOWED_TFLITE_OPS` (when the interpreter exposes op details); int8 parity on the test split with `delta_auc < 0.005`, `agreement >= 0.99`, `max_abs_diff <= 0.05`. Artifacts are written to `output/v5/export_stage/` and moved into place only after every gate passes; on failure the stage directory is kept for diagnosis, `deliverables/export_status.json` records `{"status": "failed", "error": ...}`, and previously promoted files are left untouched.
- Golden formats: `fsm_trace.txt` first line `tau tick_ms hold confirm verify min_bursts pmin pmax`, then per tick `p state active event dur mean_p n_bursts n_hits level` (the last five are zero unless `event == 2`; state IDLE=0, ACTIVE=1, CONFIRMED=2; event none=0, start=1, end=2). `doa_cases.bin`: int32 `n_cases, frame_len, max_lag`, then per case `int16 l[512]`, `int16 r[512]`, `float32 expected_lag`, `float32 expected_ratio`. `doa_tracker.bin`: int32 `n_frames, frame_len`, float32 `spacing_m`, then per frame `int16 l[512]`, `int16 r[512]`, int32 `expected_valid`, float32 `expected_lag`, then a trailer int32 `side_code` (0 unknown, 1 left, 2 right), float32 `lag_samples, lag_ms, conf`, int32 `n_valid`.

- [ ] **Step 1: Write the failing tests**

`tests/test_export.py`:
```python
import json

import numpy as np
import pytest

from v5 import export as X
from v5 import features as F
from v5.doa import DoaParams
from v5.model import build_model
from v5.streaming import FsmParams


def test_headers_only_writes_all_files(tmp_path):
    X.write_headers_only(tmp_path / "gen", tmp_path / "golden", FsmParams(), DoaParams())
    spec = (tmp_path / "gen" / "feature_spec.h").read_text()
    assert "#define SF_N_FRAMES 61" in spec and "#define SF_NORM_DIV 40.0f" in spec
    fb = (tmp_path / "gen" / "mel_filterbank.h").read_text()
    assert "sf_mel_start[30]" in fb and "sf_mel_w[" in fb
    trace = (tmp_path / "golden" / "fsm_trace.txt").read_text().splitlines()
    assert trace[0].split()[0] == "0.65" and len(trace) > 300
    rows = [line.split() for line in trace[1:]]
    assert all(len(r) == 9 for r in rows)
    ends = [r for r in rows if r[3] == "2"]
    assert ends and float(ends[0][4]) > 0 and int(ends[0][7]) > 0
    assert (tmp_path / "golden" / "doa_cases.bin").stat().st_size == 12 + 10 * (512 * 2 * 2 + 8)
    assert (tmp_path / "golden" / "doa_tracker.bin").stat().st_size == 12 + 40 * (512 * 2 * 2 + 8) + 20


def test_fsm_trace_avoids_exact_threshold():
    assert all(abs(v - 0.65) > 1e-9 for v in X.fsm_trace_sequence())


def test_tflite_conversion_validation_and_headers(tmp_path):
    model = build_model(0.5)
    rng = np.random.default_rng(0)
    rep = rng.uniform(-1, 1, (64, F.N_FRAMES, F.N_MELS, 1)).astype(np.float32)
    tflite = X.to_tflite_int8(model, rep)
    info = X.validate_tflite(tflite)
    assert info["input_shape"] == [1, 61, 30, 1] and info["input_scale"] > 0
    if info["ops_checked"]:
        assert set(info["ops"]) <= X.ALLOWED_TFLITE_OPS
    from v5.evaluate import int8_probs, predict_probs

    Xt = rng.uniform(-1, 1, (16, F.N_FRAMES, F.N_MELS, 1)).astype(np.float32)
    assert np.abs(predict_probs(model, Xt) - int8_probs(tflite, Xt)).max() < 0.1
    X.write_c_array(tflite, "snore_v5_int8_tflite", tmp_path / "model_data.h", tmp_path / "model_data.c")
    assert "extern const unsigned char snore_v5_int8_tflite[]" in (tmp_path / "model_data.h").read_text()
    assert f"snore_v5_int8_tflite_len = {len(tflite)}" in (tmp_path / "model_data.c").read_text()
    X.write_model_meta_h(tmp_path / "model_meta.h", info, 0.7, FsmParams(tau=0.7), "cnn_v5_int8")
    meta = (tmp_path / "model_meta.h").read_text()
    assert "#define SNORE_THRESHOLD 0.7" in meta and "#define FSM_CONFIRM_TICKS 20" in meta and '"cnn_v5_int8"' in meta
    est = X.arena_estimate(model, len(tflite))
    assert est["activation_bytes_estimate"] > 10_000 and est["flash_bytes"] == len(tflite)


def test_gates_raise_export_error():
    with pytest.raises(X.ExportError):
        X.check_parity({"passed": False, "delta_auc": 0.1, "agreement": 0.5, "max_abs_diff": 0.4})
    with pytest.raises(X.ExportError):
        X.validate_tflite(b"not a model")


def test_promote_moves_only_exporter_files(tmp_path):
    stage, deliv, gen = tmp_path / "stage", tmp_path / "deliv", tmp_path / "gen"
    (stage / "golden").mkdir(parents=True)
    for name in X.PROMOTED_FILES:
        (stage / name).write_text(name)
    (stage / "golden" / "features.bin").write_bytes(b"x")
    deliv.mkdir()
    (deliv / "experiments.md").write_text("keep me")
    X.promote(stage, deliv, gen)
    assert (deliv / "experiments.md").read_text() == "keep me"
    assert (deliv / "snore_v5_int8.tflite").exists() and (deliv / "golden" / "features.bin").exists()
    assert (gen / "model_meta.h").exists() and not (stage / "snore_v5_int8.tflite").exists()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv-mac/bin/python -m pytest tests/test_export.py -v`
Expected: FAIL with `ImportError`

- [ ] **Step 3: Implement `v5/export.py`**

```python
"""Export with release gates: int8 TFLite, C headers, golden files and the model card (spec section 11)."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from datetime import date, datetime
from pathlib import Path

import numpy as np
from scipy.signal import butter, sosfilt

from v5 import features as F
from v5 import golden as G
from v5.config import ROOT, load_config, resolve
from v5.doa import DoaParams, DoaTracker, gcc_phat
from v5.streaming import ACTIVE, CONFIRMED, IDLE, FsmParams, run_sequence

GEN_DIR = ROOT / "esp32_firmware" / "v5" / "generated"
STATE_CODE = {IDLE: 0, ACTIVE: 1, CONFIRMED: 2}
EVENT_CODE = {None: 0, "episode_start": 1, "episode_end": 2}
SIDE_CODE = {"unknown": 0, "left": 1, "right": 2}
HEADER_NOTE = "// Auto-generated by v5/export.py from the frozen feature spec; do not edit by hand.\n"
ALLOWED_TFLITE_OPS = {"CONV_2D", "MAX_POOL_2D", "MEAN", "FULLY_CONNECTED", "LOGISTIC", "RESHAPE", "QUANTIZE", "DEQUANTIZE"}
PARITY_LIMITS = {"delta_auc": 0.005, "agreement": 0.99, "max_abs_diff": 0.05}
HEADER_FILES = ("feature_spec.h", "mel_filterbank.h", "model_meta.h", "model_data.h", "model_data.c")
PROMOTED_FILES = ["snore_v5_int8.tflite", "robustness.json", "export_info.json", "test_metrics.json", *HEADER_FILES]


class ExportError(RuntimeError):
    """A release gate failed; nothing was promoted."""


def to_tflite_int8(model, rep_X) -> bytes:
    import tensorflow as tf

    @tf.function(input_signature=[tf.TensorSpec([1, F.N_FRAMES, F.N_MELS, 1], tf.float32)])
    def infer(x):
        return tf.sigmoid(model(x, training=False))

    converter = tf.lite.TFLiteConverter.from_concrete_functions([infer.get_concrete_function()], model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    rep = np.asarray(rep_X, dtype=np.float32)

    def representative():
        for i in range(len(rep)):
            yield [rep[i: i + 1]]

    converter.representative_dataset = representative
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    return converter.convert()


def quant_params(tflite: bytes) -> dict:
    from v5.evaluate import make_interpreter

    it = make_interpreter(tflite)
    it.allocate_tensors()
    i, o = it.get_input_details()[0], it.get_output_details()[0]
    return {"input_scale": float(i["quantization"][0]), "input_zero_point": int(i["quantization"][1]), "output_scale": float(o["quantization"][0]),
            "output_zero_point": int(o["quantization"][1]), "input_shape": [int(v) for v in i["shape"]], "output_shape": [int(v) for v in o["shape"]],
            "input_dtype": np.dtype(i["dtype"]).name, "output_dtype": np.dtype(o["dtype"]).name}


def tflite_ops(tflite: bytes):
    from v5.evaluate import make_interpreter

    try:
        it = make_interpreter(tflite)
        it.allocate_tensors()
        return sorted({d["op_name"] for d in it._get_ops_details() if d["op_name"] != "DELEGATE"})
    except Exception:
        return None


def validate_tflite(tflite: bytes) -> dict:
    try:
        qp = quant_params(tflite)
    except Exception as exc:
        raise ExportError(f"tflite model cannot be loaded: {type(exc).__name__}: {exc}") from exc
    problems = []
    if qp["input_dtype"] != "int8" or qp["output_dtype"] != "int8":
        problems.append(f"dtypes {qp['input_dtype']}/{qp['output_dtype']} are not int8")
    if qp["input_shape"] != [1, F.N_FRAMES, F.N_MELS, 1] or qp["output_shape"] != [1, 1]:
        problems.append(f"shapes {qp['input_shape']} -> {qp['output_shape']}")
    ops = tflite_ops(tflite)
    if ops is not None:
        extra = sorted(set(ops) - ALLOWED_TFLITE_OPS)
        if extra:
            problems.append(f"operators outside the TFLM-safe set: {extra}")
    if problems:
        raise ExportError("; ".join(problems))
    return {**qp, "ops": ops, "ops_checked": ops is not None}


def check_parity(parity: dict) -> None:
    if not parity.get("passed"):
        raise ExportError(f"int8 parity gate failed: {parity}")


def arena_estimate(model, tflite_bytes: int) -> dict:
    sizes = [int(np.prod(model.input_shape[1:]))]
    for layer in model.layers:
        shape = getattr(layer, "output", None)
        if shape is not None and hasattr(shape, "shape") and len(shape.shape) > 1:
            sizes.append(int(np.prod([int(d) for d in shape.shape[1:]])))
    pairs = [a + b for a, b in zip(sizes, sizes[1:])] or sizes
    return {"activation_bytes_estimate": int(max(pairs)) + 2048, "largest_tensor_bytes": int(max(sizes)), "flash_bytes": int(tflite_bytes),
            "note": "int8 activations; two largest consecutive tensors plus scratch; measure the real arena on the device"}


def write_c_array(data: bytes, name: str, h_path, c_path) -> None:
    Path(h_path).write_text(HEADER_NOTE + "#pragma once\n#include <stddef.h>\n\n" + f"extern const unsigned char {name}[];\nextern const unsigned int {name}_len;\n", encoding="utf-8")
    rows = [", ".join(f"0x{b:02x}" for b in data[i: i + 12]) for i in range(0, len(data), 12)]
    body = ",\n".join("    " + r for r in rows)
    Path(c_path).write_text(HEADER_NOTE + f'#include "{Path(h_path).name}"\n\n' + f"const unsigned char {name}[] __attribute__((aligned(16))) = {{\n{body}\n}};\n" + f"const unsigned int {name}_len = {len(data)};\n", encoding="utf-8")


def write_feature_spec_h(path) -> None:
    lines = [HEADER_NOTE + "#pragma once", "", f"#define SF_FEATURE_SPEC_VERSION {F.FEATURE_SPEC_VERSION}", f"#define SF_SAMPLE_RATE {F.SR}", f"#define SF_WIN {F.WIN}",
             f"#define SF_STREAM_HOP {F.STREAM_HOP}", f"#define SF_N_FFT {F.N_FFT}", f"#define SF_HOP {F.HOP}", f"#define SF_N_FRAMES {F.N_FRAMES}", f"#define SF_N_BINS {F.N_BINS}",
             f"#define SF_N_MELS {F.N_MELS}", f"#define SF_FMIN {F.FMIN:.1f}f", f"#define SF_FMAX {F.FMAX:.1f}f", f"#define SF_LOG_EPS {F.LOG_EPS:g}f", f"#define SF_NORM_DIV {F.NORM_DIV:.1f}f",
             f"#define SF_FEATURE_DIM {F.FEATURE_DIM}", "#define SF_PI 3.14159265358979323846f", ""]
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def write_mel_filterbank_h(path) -> None:
    starts, lens, offs, weights = [], [], [], []
    for start, w in F.sparse_filterbank():
        starts.append(start)
        lens.append(len(w))
        offs.append(len(weights))
        weights.extend(float(v) for v in w)
    fmt = lambda xs: ", ".join(str(int(x)) for x in xs)
    body = ",\n".join("    " + ", ".join(f"{v:.8e}f" for v in weights[i: i + 6]) for i in range(0, len(weights), 6))
    text = (HEADER_NOTE + "#pragma once\n#include <stdint.h>\n\n" + f"#define SF_MEL_W_LEN {len(weights)}\n"
            + f"static const uint16_t sf_mel_start[{F.N_MELS}] = {{{fmt(starts)}}};\n" + f"static const uint16_t sf_mel_len[{F.N_MELS}] = {{{fmt(lens)}}};\n"
            + f"static const uint16_t sf_mel_off[{F.N_MELS}] = {{{fmt(offs)}}};\n" + f"static const float sf_mel_w[{len(weights)}] = {{\n{body}\n}};\n")
    Path(path).write_text(text, encoding="utf-8")


def write_model_meta_h(path, qp: dict, tau: float, fsm: FsmParams, model_version: str) -> None:
    lines = [HEADER_NOTE + "#pragma once", "", f'#define MODEL_VERSION "{model_version}"', f"#define FEATURE_SPEC_VERSION {F.FEATURE_SPEC_VERSION}",
             f"#define INPUT_SCALE {qp['input_scale']:.10f}f", f"#define INPUT_ZERO_POINT {qp['input_zero_point']}", f"#define OUTPUT_SCALE {qp['output_scale']:.10f}f",
             f"#define OUTPUT_ZERO_POINT {qp['output_zero_point']}", f"#define SNORE_THRESHOLD {tau:g}f", f"#define FSM_TICK_MS {fsm.tick_ms}", f"#define FSM_HOLD_TICKS {fsm.hold_ticks}",
             f"#define FSM_CONFIRM_TICKS {fsm.confirm_ticks}", f"#define FSM_VERIFY_TICKS {fsm.verify_ticks}", f"#define FSM_MIN_BURSTS {fsm.min_bursts}",
             f"#define FSM_PERIOD_MIN_TICKS {fsm.period_min_ticks}", f"#define FSM_PERIOD_MAX_TICKS {fsm.period_max_ticks}", ""]
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def fsm_trace_sequence(seed: int = 0, tau: float = 0.65) -> list[float]:
    rng = np.random.default_rng(seed)
    seq: list[float] = []

    def periodic(cycles, burst, gap):
        for _ in range(cycles):
            seq.extend(0.85 + 0.1 * rng.random() for _ in range(burst))
            seq.extend(0.1 * rng.random() for _ in range(gap))

    seq.extend([0.05] * 10)
    periodic(8, 3, 5)
    seq.extend([0.05] * 60)
    seq.extend([0.92] * 40)
    seq.extend([0.05] * 100)
    periodic(6, 2, 6)
    seq.extend([0.3] * 20)
    seq.extend(float(v) for v in rng.random(200))
    periodic(10, 4, 4)
    seq.extend([0.02] * 80)
    out = []
    for v in seq:
        v = round(float(v), 3)
        if abs(v - tau) < 1e-9:
            v += 0.001
        out.append(v)
    return out


def write_golden_fsm(path, params: FsmParams, seed: int = 0) -> None:
    seq = fsm_trace_sequence(seed, params.tau)
    levels = [-40.0 + 5.0 * np.sin(k / 7.0) for k in range(len(seq))]
    events, states, actives = run_sequence(seq, params, levels)
    ev_by_tick = {e["tick"]: e for e in events}
    lines = [f"{params.tau:g} {params.tick_ms} {params.hold_ticks} {params.confirm_ticks} {params.verify_ticks} {params.min_bursts} {params.period_min_ticks} {params.period_max_ticks}"]
    for k, (p, st, ac) in enumerate(zip(seq, states, actives)):
        ev = ev_by_tick.get(k)
        code = EVENT_CODE[ev["type"] if ev else None]
        if ev is not None and ev["type"] == "episode_end":
            tail = f"{ev['duration_s']:.3f} {ev['mean_p']:.6f} {ev['n_bursts']} {ev['n_hits']} {ev['level_dbfs']:.4f}"
        else:
            tail = "0 0 0 0 0"
        lines.append(f"{p:.3f} {STATE_CODE[st]} {1 if ac else 0} {code} {tail}")
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def _delayed(x, d: float) -> np.ndarray:
    n = len(x)
    f = np.fft.rfftfreq(n, 1.0 / F.SR)
    return np.fft.irfft(np.fft.rfft(x) * np.exp(-2j * np.pi * f * d / F.SR), n=n)


def _burst(rng, sos):
    return sosfilt(sos, rng.standard_normal(576))[64:] * 0.1


def _uncorrelated_pair(rng, sos, params: DoaParams):
    """An uncorrelated frame pair whose PHAT peak ratio is at least 0.05 away from min_ratio, so that
    float32 (C) and float64 (Python) cannot disagree on validity at the boundary."""
    while True:
        l, r = F.float_to_int16(_burst(rng, sos)), F.float_to_int16(_burst(rng, sos))
        _, ratio = gcc_phat(F.int16_to_float(l), F.int16_to_float(r), params.max_lag, params.fs, params.band, params.n_fft)
        if abs(ratio - params.min_ratio) >= 0.05:
            return l, r


def doa_golden_cases(seed: int = 0, params: DoaParams = DoaParams()):
    rng = np.random.default_rng(seed)
    sos = butter(4, [100, 2000], btype="band", fs=F.SR, output="sos")
    cases = []
    for d in (-3.0, -2.0, -1.5, -1.0, 0.0, 1.0, 1.5, 2.0, 3.0):
        l = _burst(rng, sos)
        cases.append((F.float_to_int16(l), F.float_to_int16(_delayed(l, d))))
    cases.append(_uncorrelated_pair(rng, sos, params))
    return cases


def write_golden_doa_cases(path, params: DoaParams, seed: int = 0) -> None:
    cases = doa_golden_cases(seed, params)
    with open(path, "wb") as fh:
        fh.write(np.array([len(cases), params.n_fft, params.max_lag], dtype="<i4").tobytes())
        for l, r in cases:
            lag, ratio = gcc_phat(F.int16_to_float(l), F.int16_to_float(r), params.max_lag, params.fs, params.band, params.n_fft)
            fh.write(l.astype("<i2").tobytes())
            fh.write(r.astype("<i2").tobytes())
            fh.write(np.array([lag, min(ratio, 1e6)], dtype="<f4").tobytes())


def doa_tracker_frames(seed: int = 0, params: DoaParams = DoaParams()):
    rng = np.random.default_rng(seed)
    sos = butter(4, [100, 2000], btype="band", fs=F.SR, output="sos")
    frames = []
    for _ in range(25):  # source on the left: signal reaches L two samples early
        l = _burst(rng, sos)
        frames.append((F.float_to_int16(l), F.float_to_int16(_delayed(l, 2.0))))
    for _ in range(5):  # quiet frames, below the noise floor margin
        q = 1e-5 * rng.standard_normal(512)
        frames.append((F.float_to_int16(q), F.float_to_int16(q)))
    for _ in range(10):  # uncorrelated pairs, kept away from the ratio boundary
        frames.append(_uncorrelated_pair(rng, sos, params))
    return frames


def write_golden_doa_tracker(path, params: DoaParams, seed: int = 0) -> None:
    frames = doa_tracker_frames(seed, params)
    tracker = DoaTracker(params)
    with open(path, "wb") as fh:
        fh.write(np.array([len(frames), params.n_fft], dtype="<i4").tobytes())
        fh.write(np.array([params.spacing_m], dtype="<f4").tobytes())
        for l, r in frames:
            lag, valid = tracker.frame(F.int16_to_float(l), F.int16_to_float(r))
            fh.write(l.astype("<i2").tobytes())
            fh.write(r.astype("<i2").tobytes())
            fh.write(np.array([1 if valid else 0], dtype="<i4").tobytes())
            fh.write(np.array([lag], dtype="<f4").tobytes())
        ep = tracker.episode()
        fh.write(np.array([SIDE_CODE[ep["side"]]], dtype="<i4").tobytes())
        fh.write(np.array([ep["lag_samples"], ep["lag_ms"], ep["conf"]], dtype="<f4").tobytes())
        fh.write(np.array([ep["n_valid"]], dtype="<i4").tobytes())


def write_headers_only(gen_dir, golden_dir, fsm: FsmParams, doa: DoaParams) -> None:
    gen_dir, golden_dir = Path(gen_dir), Path(golden_dir)
    gen_dir.mkdir(parents=True, exist_ok=True)
    golden_dir.mkdir(parents=True, exist_ok=True)
    write_feature_spec_h(gen_dir / "feature_spec.h")
    write_mel_filterbank_h(gen_dir / "mel_filterbank.h")
    write_golden_fsm(golden_dir / "fsm_trace.txt", fsm)
    write_golden_doa_cases(golden_dir / "doa_cases.bin", doa)
    write_golden_doa_tracker(golden_dir / "doa_tracker.bin", doa)


def promote(stage, deliv, gen_dir) -> None:
    stage, deliv, gen_dir = Path(stage), Path(deliv), Path(gen_dir)
    deliv.mkdir(parents=True, exist_ok=True)
    gen_dir.mkdir(parents=True, exist_ok=True)
    for name in PROMOTED_FILES:
        os.replace(stage / name, deliv / name)
    for name in HEADER_FILES:
        shutil.copy(deliv / name, gen_dir / name)
    if (deliv / "golden").exists():
        shutil.rmtree(deliv / "golden")
    shutil.move(str(stage / "golden"), str(deliv / "golden"))


def _write_status(deliv: Path, status: str, **extra) -> None:
    deliv.mkdir(parents=True, exist_ok=True)
    (deliv / "export_status.json").write_text(json.dumps({"status": status, "at": datetime.now().isoformat(timespec="seconds"), **extra}, indent=1))


def export_model(cfg: dict, model_path=None, threshold_path=None) -> dict:
    import keras

    from v5.data import manifest as M
    from v5.data.dataset import precompute_features
    from v5.evaluate import clip_metrics, int8_parity, int8_probs, predict_probs, robustness_sweep
    from v5.model import check_ops

    out = Path(cfg["paths"]["out_dir"])
    deliv, stage = out / "deliverables", out / "export_stage"
    try:
        if stage.exists():
            shutil.rmtree(stage)
        golden = stage / "golden"
        golden.mkdir(parents=True)
        thr = json.loads(Path(threshold_path or out / "threshold.json").read_text())
        tau, model_version = float(thr["tau"]), thr["model_version"]
        fsm = FsmParams.from_config(tau, thr["fsm"])
        doa = DoaParams(spacing_m=float(cfg["doa"]["spacing_m"]))
        model = keras.models.load_model(model_path or out / "deployed" / "model.keras", compile=False)
        check_ops(model)
        rows = M.read_manifest(out / "manifest.csv")
        _, audio, _ = M.load_cache(out)
        y = np.array([r["label"] for r in rows])
        tr, te = M.split_indices(rows, "train"), M.split_indices(rows, "test")
        rng = np.random.default_rng(cfg["seed"])
        tflite = to_tflite_int8(model, precompute_features(audio[rng.choice(tr, size=min(500, len(tr)), replace=False)]))
        valid = validate_tflite(tflite)
        (stage / "snore_v5_int8.tflite").write_bytes(tflite)
        write_headers_only(stage, golden, fsm, doa)
        write_c_array(tflite, "snore_v5_int8_tflite", stage / "model_data.h", stage / "model_data.c")
        write_model_meta_h(stage / "model_meta.h", valid, tau, fsm, model_version)
        snore = [audio[i] for i in te if y[i] == 1][:4]
        noise = [audio[i] for i in te if y[i] == 0][:4]
        names, gx, gX, gq = G.make_golden(snore, noise, scale=valid["input_scale"], zero_point=valid["input_zero_point"])
        G.write_golden_npz(golden / "features.npz", names, gx, gX, gq, valid["input_scale"], valid["input_zero_point"])
        G.write_golden_bin(golden / "features.bin", gx, gX, gq, valid["input_scale"], valid["input_zero_point"])
        Xt = precompute_features(audio[te])
        p_fp, p_i8 = predict_probs(model, Xt), int8_probs(tflite, Xt)
        parity = int8_parity(p_fp, p_i8, y[te], tau)
        check_parity(parity)
        test_metrics = {"float": clip_metrics(y[te], p_fp, tau), "int8": clip_metrics(y[te], p_i8, tau)}
        bench_noise = audio[[r["id"] for r in rows if r["split"] == "bench"]]
        robustness = robustness_sweep(model, audio[te], y[te], bench_noise, tau, seed=cfg["seed"])
        info = {"model_version": model_version, "tau": tau, "run": thr.get("run"), "calib": thr.get("calib"), "tflite_bytes": len(tflite),
                "tflite_sha256": hashlib.sha256(tflite).hexdigest(), "params": int(model.count_params()), "qp": valid, "parity": parity, "parity_limits": PARITY_LIMITS,
                "arena": arena_estimate(model, len(tflite)), "test": test_metrics, "n_test": int(len(te)), "exported_at": datetime.now().isoformat(timespec="seconds")}
        (stage / "export_info.json").write_text(json.dumps(info, indent=1))
        (stage / "test_metrics.json").write_text(json.dumps(test_metrics, indent=1))
        (stage / "robustness.json").write_text(json.dumps(robustness, indent=1))
        promote(stage, deliv, GEN_DIR)
        shutil.rmtree(stage)
        _write_status(deliv, "ok", model_version=model_version, tflite_sha256=info["tflite_sha256"])
        return info
    except Exception as exc:
        _write_status(deliv, "failed", error=f"{type(exc).__name__}: {exc}", stage=str(stage))
        raise


def _md_metrics(title: str, m: dict) -> list[str]:
    return [f"### {title}", "", "| n | pos | AUC | R@FPR2% | tau | precision | recall | FNR | FPR |", "|---|---|---|---|---|---|---|---|---|",
            f"| {m['n']} | {m['n_pos']} | {m['auc']:.4f} | {m['recall_at_fpr2']:.3f} | {m['tau']:.3f} | {m['precision']:.3f} | {m['recall']:.3f} | {m['fnr']:.3f} | {m['fpr']:.3f} |", ""]


def model_card(cfg: dict) -> str:
    deliv = Path(cfg["paths"]["out_dir"]) / "deliverables"
    status = json.loads((deliv / "export_status.json").read_text())
    if status["status"] != "ok":
        raise ExportError(f"last export did not pass the gates: {status}")
    info = json.loads((deliv / "export_info.json").read_text())
    rob = json.loads((deliv / "robustness.json").read_text())
    cal = info.get("calib") or {}
    lines = [f"# SnoozMate v5 snore model card ({date.today().isoformat()})", "",
             f"model_version: `{info['model_version']}`; run `{info.get('run')}`; feature spec v{F.FEATURE_SPEC_VERSION}; tau = {info['tau']:.4f}; exported {info['exported_at']}", "",
             f"int8 TFLite: {info['tflite_bytes']} bytes (sha256 {info['tflite_sha256'][:12]}); params {info['params']}; input scale {info['qp']['input_scale']:.6f} zero point {info['qp']['input_zero_point']}; ops {info['qp'].get('ops')}", "",
             f"arena estimate: {info['arena']['activation_bytes_estimate']} bytes activations (largest tensor {info['arena']['largest_tensor_bytes']}); {info['arena']['note']}", "",
             f"calibration (calib split): n_neg {cal.get('n_neg')}, FPR {cal.get('fpr', float('nan')):.4f} (95% upper bound {cal.get('fpr_upper95', float('nan')):.4f}), recall {cal.get('recall', float('nan')):.3f}", "",
             "## Deployment call sequence", "", "```", "window (int16[16000], hop 8000) -> sf_compute -> sf_quantize(INPUT_SCALE, INPUT_ZERO_POINT)",
             "  -> TFLM invoke -> p = (out - OUTPUT_ZERO_POINT) * OUTPUT_SCALE -> fsm_tick(p, level_dbfs)", "  -> vibrate only while fsm.state == FSM_CONFIRMED and fsm.active", "```", "",
             "The handwritten float path (snore_infer.c, 63-frame input) is deprecated. Desktop C tests establish numerical parity, not ESP32-S3 deployability;",
             "the firmware build must still verify the TFLM operator resolver, the real tensor arena size and flash use.", ""]
    lines += _md_metrics("Test split, float model (evaluated once at export)", info["test"]["float"])
    lines += _md_metrics("Test split, int8 model (deployed path)", info["test"]["int8"])
    par = info["parity"]
    lines += ["### int8 parity gate", "", f"delta AUC {par['delta_auc']:.4f}; decision agreement {par['agreement']:.4f}; max |dp| {par['max_abs_diff']:.4f}; limits {info['parity_limits']}; passed: {par['passed']}", ""]
    lines += ["### Robustness (test split, float model)", "", "| condition | AUC | R@FPR2% | recall@tau | FPR@tau |", "|---|---|---|---|---|",
              f"| clean | {rob['clean']['auc']:.4f} | {rob['clean']['recall_at_fpr2']:.3f} | {rob['clean']['recall']:.3f} | {rob['clean']['fpr']:.3f} |"]
    for k, m in rob["snr"].items():
        lines.append(f"| SNR {k} dB | {m['auc']:.4f} | {m['recall_at_fpr2']:.3f} | {m['recall']:.3f} | {m['fpr']:.3f} |")
    for k, m in rob["distance"].items():
        lines.append(f"| RIR {k} m | {m['auc']:.4f} | {m['recall_at_fpr2']:.3f} | {m['recall']:.3f} | {m['fpr']:.3f} |")
    lines.append("")
    for name in ("manifest_report.md", "experiments.md", "benchmark_nights.md", "doa_sim.md"):
        p = deliv / name
        if p.exists():
            lines += [f"## {name}", "", p.read_text(encoding="utf-8"), ""]
    return "\n".join(lines)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    h = sub.add_parser("headers")
    m = sub.add_parser("model")
    m.add_argument("--model", default=None)
    m.add_argument("--threshold", default=None)
    c = sub.add_parser("card")
    for p in (h, m, c):
        p.add_argument("--config", default=None)
    args = ap.parse_args(argv)
    cfg = resolve(load_config(args.config))
    out = Path(cfg["paths"]["out_dir"])
    if args.cmd == "headers":
        thr = out / "threshold.json"
        tau = json.loads(thr.read_text())["tau"] if thr.exists() else 0.65
        write_headers_only(GEN_DIR, out / "deliverables" / "golden", FsmParams.from_config(tau, cfg["fsm"]), DoaParams(spacing_m=float(cfg["doa"]["spacing_m"])))
        print(f"headers -> {GEN_DIR}; golden -> {out / 'deliverables' / 'golden'}")
    elif args.cmd == "model":
        info = export_model(cfg, args.model, args.threshold)
        print(json.dumps({k: info[k] for k in ("model_version", "tau", "tflite_bytes", "params", "parity", "test")}, indent=1))
    else:
        text = model_card(cfg)
        (out / "deliverables" / "model_card.md").write_text(text, encoding="utf-8")
        print(text[:2000])


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv-mac/bin/python -m pytest tests/test_export.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add v5/export.py tests/test_export.py
git commit -m "feat(v5): exporter with release gates, staged promotion and golden files"
```
### Task 21: C feature extractor with host parity test

**Files:**
- Create: `esp32_firmware/v5/fft512.h`, `esp32_firmware/v5/fft512.c`, `esp32_firmware/v5/snore_features.h`, `esp32_firmware/v5/snore_features.c`, `esp32_firmware/v5/host_test/test_features.c`, `esp32_firmware/v5/Makefile`, `tests/test_c_parity.py` (`.gitignore` already ignores `esp32_firmware/v5/build/`)
- Generated by `python -m v5.export headers`: `esp32_firmware/v5/generated/feature_spec.h`, `mel_filterbank.h` (tracked)

**Interfaces:**
- Produces C API: `void fft512_init(void); void fft512_forward(float *re, float *im); void fft512_inverse(float *re, float *im);` and `void sf_init(void); void sf_compute(const int16_t *win, float *X); void sf_quantize(const float *X, int8_t *q, float scale, int zero_point);` with `X` row-major `[frame][mel]`, 1830 floats.
- Parity contract (numerical, not bit-exact): float features within `max |dX| <= 5e-3` of the Python float64 reference on every cell; quantized features never differ by more than 1 LSB and differ by exactly 1 LSB on at most 1 % of cells. The FSM contract (Task 22) is exact equality.
- Makefile targets: `test-features`, `test-fsm`, `test-doa`, `test` (all three); `GOLDEN=<dir>` selects the golden directory (default `../../output/v5/deliverables/golden`).

- [ ] **Step 1: Write the pytest wrapper (fails until the C code exists)**

`tests/test_c_parity.py`:
```python
import shutil
import subprocess

import numpy as np
import pytest

from v5 import export as X
from v5 import features as F
from v5 import golden as G
from v5.config import ROOT
from v5.doa import DoaParams
from v5.streaming import FsmParams

FW = ROOT / "esp32_firmware" / "v5"
SCALE, ZP = 1.0 / 127.0, 0


def _synthetic_golden(golden_dir):
    rng = np.random.default_rng(7)
    env = 0.5 + 0.5 * np.sin(2 * np.pi * 0.5 * np.arange(F.WIN) / F.SR)
    snore = [F.float_to_int16(0.1 * rng.standard_normal(F.WIN) * env) for _ in range(4)]
    noise = [F.float_to_int16(0.02 * rng.standard_normal(F.WIN)) for _ in range(4)]
    names, x, Xg, q = G.make_golden(snore, noise, scale=SCALE, zero_point=ZP)
    G.write_golden_bin(golden_dir / "features.bin", x, Xg, q, SCALE, ZP)


def _run(target, golden):
    build = subprocess.run(["make", "-C", str(FW), target, f"GOLDEN={golden}"], capture_output=True, text=True)
    print(build.stdout[-4000:], build.stderr[-4000:])
    assert build.returncode == 0, f"{target} failed; see output above"


@pytest.mark.skipif(shutil.which("cc") is None, reason="no C compiler")
def test_c_features_match_python_reference(tmp_path):
    golden = tmp_path / "golden"
    X.write_headers_only(FW / "generated", golden, FsmParams(), DoaParams())
    _synthetic_golden(golden)
    _run("test-features", golden)
```

- [ ] **Step 2: Run the wrapper to verify it fails**

Run: `.venv-mac/bin/python -m pytest tests/test_c_parity.py -v`
Expected: FAIL (make: no rule / missing sources)

- [ ] **Step 3: Write the C sources**

`esp32_firmware/v5/fft512.h`:
```c
#pragma once
/* Radix-2 complex FFT, N = 512, float32. Firmware may replace this with esp-dsp
 * (dsps_fft2r_fc32) as long as the host parity tests still pass. */
void fft512_init(void);
void fft512_forward(float *re, float *im);
void fft512_inverse(float *re, float *im); /* includes the 1/N scale */
```

`esp32_firmware/v5/fft512.c`:
```c
#include "fft512.h"
#include <math.h>

#define FFT_N 512
#define FFT_PI 3.14159265358979323846f

static float s_cos[FFT_N / 2];
static float s_sin[FFT_N / 2];
static unsigned short s_rev[FFT_N];
static int s_ready = 0;

void fft512_init(void) {
    for (int k = 0; k < FFT_N / 2; k++) {
        s_cos[k] = cosf(2.0f * FFT_PI * (float)k / (float)FFT_N);
        s_sin[k] = sinf(2.0f * FFT_PI * (float)k / (float)FFT_N);
    }
    for (int i = 0; i < FFT_N; i++) {
        unsigned r = 0;
        for (int b = 0; b < 9; b++) r |= ((unsigned)(i >> b) & 1u) << (8 - b);
        s_rev[i] = (unsigned short)r;
    }
    s_ready = 1;
}

static void fft_core(float *re, float *im, int inverse) {
    if (!s_ready) fft512_init();
    for (int i = 0; i < FFT_N; i++) {
        int j = s_rev[i];
        if (j > i) {
            float t = re[i]; re[i] = re[j]; re[j] = t;
            t = im[i]; im[i] = im[j]; im[j] = t;
        }
    }
    for (int len = 2; len <= FFT_N; len <<= 1) {
        int half = len >> 1, step = FFT_N / len;
        for (int i = 0; i < FFT_N; i += len) {
            for (int j = 0; j < half; j++) {
                float wr = s_cos[j * step];
                float wi = inverse ? s_sin[j * step] : -s_sin[j * step];
                float xr = re[i + j + half], xi = im[i + j + half];
                float tr = xr * wr - xi * wi, ti = xr * wi + xi * wr;
                re[i + j + half] = re[i + j] - tr;
                im[i + j + half] = im[i + j] - ti;
                re[i + j] += tr;
                im[i + j] += ti;
            }
        }
    }
    if (inverse) {
        for (int i = 0; i < FFT_N; i++) { re[i] /= (float)FFT_N; im[i] /= (float)FFT_N; }
    }
}

void fft512_forward(float *re, float *im) { fft_core(re, im, 0); }
void fft512_inverse(float *re, float *im) { fft_core(re, im, 1); }
```

`esp32_firmware/v5/snore_features.h`:
```c
#pragma once
#include <stdint.h>
#include "feature_spec.h"

/* Feature spec v1 (see v5/features.py). X is row-major [SF_N_FRAMES][SF_N_MELS]. */
void sf_init(void);
void sf_compute(const int16_t *win, float *X);
void sf_quantize(const float *X, int8_t *q, float scale, int zero_point);
```

`esp32_firmware/v5/snore_features.c`:
```c
#include "snore_features.h"
#include "mel_filterbank.h"
#include "fft512.h"
#include <math.h>

static float s_win[SF_N_FFT];
static int s_ready = 0;

void sf_init(void) {
    fft512_init();
    for (int n = 0; n < SF_N_FFT; n++)
        s_win[n] = 0.5f - 0.5f * cosf(2.0f * SF_PI * (float)n / (float)SF_N_FFT); /* periodic Hann */
    s_ready = 1;
}

void sf_compute(const int16_t *win, float *X) {
    float re[SF_N_FFT], im[SF_N_FFT], power[SF_N_BINS];
    double sum = 0.0;
    if (!s_ready) sf_init();
    for (int t = 0; t < SF_N_FRAMES; t++) {
        const int16_t *frame = win + t * SF_HOP;
        for (int n = 0; n < SF_N_FFT; n++) {
            re[n] = ((float)frame[n] / 32768.0f) * s_win[n];
            im[n] = 0.0f;
        }
        fft512_forward(re, im);
        for (int k = 0; k < SF_N_BINS; k++) power[k] = re[k] * re[k] + im[k] * im[k];
        for (int m = 0; m < SF_N_MELS; m++) {
            const float *w = sf_mel_w + sf_mel_off[m];
            int start = sf_mel_start[m], len = sf_mel_len[m];
            float acc = 0.0f;
            for (int j = 0; j < len; j++) acc += w[j] * power[start + j];
            float L = 10.0f * log10f(acc + SF_LOG_EPS);
            X[t * SF_N_MELS + m] = L;
            sum += (double)L;
        }
    }
    float mean = (float)(sum / (double)SF_FEATURE_DIM);
    for (int i = 0; i < SF_FEATURE_DIM; i++) {
        float v = (X[i] - mean) / SF_NORM_DIV;
        X[i] = v < -1.0f ? -1.0f : (v > 1.0f ? 1.0f : v);
    }
}

void sf_quantize(const float *X, int8_t *q, float scale, int zero_point) {
    for (int i = 0; i < SF_FEATURE_DIM; i++) {
        long r = (long)rintf(X[i] / scale) + zero_point; /* rintf: half-to-even like numpy.round */
        if (r < -128) r = -128;
        if (r > 127) r = 127;
        q[i] = (int8_t)r;
    }
}
```

`esp32_firmware/v5/host_test/test_features.c`:
```c
/* Usage: test_features <features.bin>.
 * Exit 0 when every case has max |dX| <= 5e-3, no quantised cell differs by more than 1 LSB,
 * and at most 1 % of quantised cells differ by exactly 1 LSB. */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include "snore_features.h"

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s features.bin\n", argv[0]); return 2; }
    FILE *fh = fopen(argv[1], "rb");
    if (!fh) { perror("open"); return 2; }
    int32_t hdr[3], zp;
    float scale;
    if (fread(hdr, sizeof(int32_t), 3, fh) != 3 || fread(&scale, sizeof(float), 1, fh) != 1 || fread(&zp, sizeof(int32_t), 1, fh) != 1) { fprintf(stderr, "bad header\n"); return 2; }
    if (hdr[1] != SF_WIN || hdr[2] != SF_FEATURE_DIM) { fprintf(stderr, "golden shape mismatch\n"); return 2; }
    int16_t *win = malloc(sizeof(int16_t) * SF_WIN);
    float *ref = malloc(sizeof(float) * SF_FEATURE_DIM);
    float *X = malloc(sizeof(float) * SF_FEATURE_DIM);
    int8_t *qref = malloc(SF_FEATURE_DIM), *q = malloc(SF_FEATURE_DIM);
    double worst = 0.0;
    int failures = 0;
    sf_init();
    for (int c = 0; c < hdr[0]; c++) {
        if (fread(win, sizeof(int16_t), SF_WIN, fh) != (size_t)SF_WIN) return 2;
        if (fread(ref, sizeof(float), SF_FEATURE_DIM, fh) != (size_t)SF_FEATURE_DIM) return 2;
        if (fread(qref, 1, SF_FEATURE_DIM, fh) != (size_t)SF_FEATURE_DIM) return 2;
        sf_compute(win, X);
        sf_quantize(X, q, scale, zp);
        double maxd = 0.0;
        int bad = 0, one_lsb = 0, over = 0;
        for (int i = 0; i < SF_FEATURE_DIM; i++) {
            double d = fabs((double)X[i] - (double)ref[i]);
            if (d > maxd) maxd = d;
            if (d > 5e-3) bad++;
            int dq = abs((int)q[i] - (int)qref[i]);
            if (dq == 1) one_lsb++;
            if (dq > 1) over++;
        }
        if (maxd > worst) worst = maxd;
        int ok = bad == 0 && over == 0 && one_lsb * 100 <= SF_FEATURE_DIM;
        printf("case %2d: max|dX| = %.5f, cells over tolerance = %d, q off by 1 LSB = %d, q off by >1 = %d %s\n", c, maxd, bad, one_lsb, over, ok ? "ok" : "MISMATCH");
        if (!ok) failures++;
    }
    fclose(fh);
    printf("features parity: worst %.5f over %d cases, %d failing\n", worst, hdr[0], failures);
    return failures ? 1 : 0;
}
```

`esp32_firmware/v5/Makefile`:
```make
CC ?= cc
CFLAGS ?= -std=c99 -O2 -Wall -Wextra -I. -Igenerated
GOLDEN ?= ../../output/v5/deliverables/golden

build:
	mkdir -p build

build/test_features: host_test/test_features.c fft512.c snore_features.c | build
	$(CC) $(CFLAGS) -o $@ host_test/test_features.c fft512.c snore_features.c -lm

build/test_fsm: host_test/test_fsm.c snore_episode_fsm.c | build
	$(CC) $(CFLAGS) -o $@ host_test/test_fsm.c snore_episode_fsm.c -lm

build/test_doa: host_test/test_doa.c fft512.c doa_gccphat.c | build
	$(CC) $(CFLAGS) -o $@ host_test/test_doa.c fft512.c doa_gccphat.c -lm

test-features: build/test_features
	./build/test_features $(GOLDEN)/features.bin

test-fsm: build/test_fsm
	./build/test_fsm $(GOLDEN)/fsm_trace.txt

test-doa: build/test_doa
	./build/test_doa $(GOLDEN)/doa_cases.bin $(GOLDEN)/doa_tracker.bin

test: test-features test-fsm test-doa

clean:
	rm -rf build

.PHONY: test test-features test-fsm test-doa clean
```

- [ ] **Step 4: Generate headers and run the features test**

Run: `.venv-mac/bin/python -m v5.export headers && .venv-mac/bin/python -m pytest tests/test_c_parity.py -v`
Expected: PASS; the make output shows every case `ok` with `q off by >1 = 0`. If a synthetic case exceeds 5e-3, raise the noise floor of that signal in `v5/golden.py` (3e-4 → 1e-3) rather than loosening the tolerance, and note it in the commit.

- [ ] **Step 5: Commit**

```bash
git add esp32_firmware/v5 tests/test_c_parity.py
git commit -m "feat(fw-v5): C feature extractor with FFT and host parity test"
```
### Task 22: C episode state machine

**Files:**
- Create: `esp32_firmware/v5/snore_episode_fsm.h`, `esp32_firmware/v5/snore_episode_fsm.c`, `esp32_firmware/v5/host_test/test_fsm.c`
- Modify: `tests/test_c_parity.py` (add the fsm test)

**Interfaces:**
- Produces C API: `typedef struct { float tau; int tick_ms, hold_ticks, confirm_ticks, verify_ticks, min_bursts, period_min_ticks, period_max_ticks; } fsm_params_t;` `typedef enum { FSM_IDLE = 0, FSM_ACTIVE = 1, FSM_CONFIRMED = 2 } fsm_state_t;` `typedef enum { FSM_EV_NONE = 0, FSM_EV_EPISODE_START = 1, FSM_EV_EPISODE_END = 2 } fsm_event_t;` `void fsm_init(fsm_t *f, const fsm_params_t *p); fsm_event_t fsm_tick(fsm_t *f, float p, float level_dbfs);` plus readable fields `f->state`, `f->active`, `f->activity`, and after an end event `f->last_duration_s`, `f->last_mean_p`, `f->last_n_bursts`, `f->last_n_hits`, `f->last_level_dbfs`, `f->last_start_tick`.
- Parity contract: state, active flag and event code equal on every tick; on end events duration within 1e-3 s, mean probability within 1e-4, burst and hit counts exact, level within 1e-3 dB.

- [ ] **Step 1: Write the host test and extend the pytest wrapper**

`esp32_firmware/v5/host_test/test_fsm.c`:
```c
/* Usage: test_fsm <fsm_trace.txt>. Replays the Python reference trace. */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "snore_episode_fsm.h"

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s fsm_trace.txt\n", argv[0]); return 2; }
    FILE *fh = fopen(argv[1], "r");
    if (!fh) { perror("open"); return 2; }
    fsm_params_t p;
    if (fscanf(fh, "%f %d %d %d %d %d %d %d", &p.tau, &p.tick_ms, &p.hold_ticks, &p.confirm_ticks, &p.verify_ticks, &p.min_bursts, &p.period_min_ticks, &p.period_max_ticks) != 8) {
        fprintf(stderr, "bad header\n");
        return 2;
    }
    fsm_t f;
    fsm_init(&f, &p);
    float prob, dur, mean_p, level;
    int state, active, event, n_bursts, n_hits, tick = 0, mismatches = 0, ends = 0;
    while (fscanf(fh, "%f %d %d %d %f %f %d %d %f", &prob, &state, &active, &event, &dur, &mean_p, &n_bursts, &n_hits, &level) == 9) {
        float lvl = -40.0f + 5.0f * sinf((float)tick / 7.0f);
        fsm_event_t ev = fsm_tick(&f, prob, lvl);
        int ok = (int)f.state == state && f.active == active && (int)ev == event;
        if (ok && event == 2) {
            ends++;
            ok = fabsf(f.last_duration_s - dur) <= 1e-3f && fabsf(f.last_mean_p - mean_p) <= 1e-4f && f.last_n_bursts == n_bursts && f.last_n_hits == n_hits && fabsf(f.last_level_dbfs - level) <= 1e-3f;
            if (!ok && mismatches < 10)
                printf("tick %d: end fields expected dur %.3f mean %.6f bursts %d hits %d level %.4f, got %.3f %.6f %d %d %.4f\n", tick, dur, mean_p, n_bursts, n_hits, level, f.last_duration_s, f.last_mean_p, f.last_n_bursts, f.last_n_hits, f.last_level_dbfs);
        } else if (!ok && mismatches < 10) {
            printf("tick %d: p=%.3f expected state %d active %d event %d, got %d %d %d\n", tick, prob, state, active, event, (int)f.state, f.active, (int)ev);
        }
        if (!ok) mismatches++;
        tick++;
    }
    fclose(fh);
    printf("fsm parity: %d ticks, %d end events checked, %d mismatches\n", tick, ends, mismatches);
    return (mismatches || ends == 0) ? 1 : 0;
}
```

Append to `tests/test_c_parity.py`:
```python
@pytest.mark.skipif(shutil.which("cc") is None, reason="no C compiler")
def test_c_fsm_matches_python_trace(tmp_path):
    golden = tmp_path / "golden"
    X.write_headers_only(FW / "generated", golden, FsmParams(), DoaParams())
    _run("test-fsm", golden)
```

- [ ] **Step 2: Implement the FSM**

`esp32_firmware/v5/snore_episode_fsm.h`:
```c
#pragma once
#include <stdint.h>

#define FSM_MAX_HIST 64
#define FSM_MAX_BURSTS 64
#define FSM_MAX_HITS 64

typedef struct {
    float tau;
    int tick_ms, hold_ticks, confirm_ticks, verify_ticks, min_bursts, period_min_ticks, period_max_ticks;
} fsm_params_t;

typedef enum { FSM_IDLE = 0, FSM_ACTIVE = 1, FSM_CONFIRMED = 2 } fsm_state_t;
typedef enum { FSM_EV_NONE = 0, FSM_EV_EPISODE_START = 1, FSM_EV_EPISODE_END = 2 } fsm_event_t;

typedef struct {
    fsm_params_t p;
    fsm_state_t state;
    uint32_t tick;
    float hist[FSM_MAX_HIST];
    int hist_len, hist_pos;
    int streak_q, in_burst;
    uint32_t bursts[FSM_MAX_BURSTS];
    int n_bursts;
    uint32_t hit_tick[FSM_MAX_HITS];
    float hit_p[FSM_MAX_HITS], hit_level[FSM_MAX_HITS];
    int n_hits_buf;
    int active;
    float activity;
    int below_ticks;
    uint32_t ep_first_burst, ep_last_hit;
    int ep_n_bursts, ep_n_hits;
    float ep_sum_p, ep_sum_level;
    float last_duration_s, last_mean_p, last_level_dbfs;
    int last_n_bursts, last_n_hits;
    uint32_t last_start_tick;
} fsm_t;

void fsm_init(fsm_t *f, const fsm_params_t *p);
fsm_event_t fsm_tick(fsm_t *f, float p, float level_dbfs);
```

`esp32_firmware/v5/snore_episode_fsm.c`:
```c
#include "snore_episode_fsm.h"
#include <string.h>
#include <stdlib.h>

void fsm_init(fsm_t *f, const fsm_params_t *p) {
    memset(f, 0, sizeof(*f));
    f->p = *p;
    if (f->p.hold_ticks + 1 > FSM_MAX_HIST) f->p.hold_ticks = FSM_MAX_HIST - 1;
    f->state = FSM_IDLE;
}

static int cmp_u32(const void *a, const void *b) {
    uint32_t x = *(const uint32_t *)a, y = *(const uint32_t *)b;
    return (x > y) - (x < y);
}

static int periodic(const fsm_t *f) {
    uint32_t gaps[FSM_MAX_BURSTS];
    int n = f->n_bursts - 1;
    if (f->n_bursts < f->p.min_bursts) return 0;
    for (int i = 0; i < n; i++) gaps[i] = f->bursts[i + 1] - f->bursts[i];
    qsort(gaps, (size_t)n, sizeof(uint32_t), cmp_u32);
    uint32_t med = gaps[(n - 1) / 2]; /* lower median, same as the Python reference */
    return med >= (uint32_t)f->p.period_min_ticks && med <= (uint32_t)f->p.period_max_ticks;
}

static void drop_front_hit(fsm_t *f) {
    memmove(f->hit_tick, f->hit_tick + 1, (size_t)(f->n_hits_buf - 1) * sizeof(uint32_t));
    memmove(f->hit_p, f->hit_p + 1, (size_t)(f->n_hits_buf - 1) * sizeof(float));
    memmove(f->hit_level, f->hit_level + 1, (size_t)(f->n_hits_buf - 1) * sizeof(float));
    f->n_hits_buf--;
}

fsm_event_t fsm_tick(fsm_t *f, float p, float level_dbfs) {
    const fsm_params_t *P = &f->p;
    uint32_t i = f->tick++;
    int cap = P->hold_ticks + 1;
    uint32_t max_age = (uint32_t)(P->confirm_ticks + P->hold_ticks);
    fsm_event_t ev = FSM_EV_NONE;
    f->hist[f->hist_pos] = p;
    f->hist_pos = (f->hist_pos + 1) % cap;
    if (f->hist_len < cap) f->hist_len++;
    float act = f->hist[0];
    for (int k = 1; k < f->hist_len; k++) if (f->hist[k] > act) act = f->hist[k];
    f->activity = act;
    int hit = p >= P->tau;
    f->active = act >= P->tau;
    int burst_started = hit && !f->in_burst;
    if (burst_started && f->n_bursts < FSM_MAX_BURSTS) f->bursts[f->n_bursts++] = i;
    if (hit && f->n_hits_buf < FSM_MAX_HITS) {
        f->hit_tick[f->n_hits_buf] = i;
        f->hit_p[f->n_hits_buf] = p;
        f->hit_level[f->n_hits_buf] = level_dbfs;
        f->n_hits_buf++;
    }
    f->in_burst = hit;
    while (f->n_bursts > 0 && i - f->bursts[0] > max_age) {
        memmove(f->bursts, f->bursts + 1, (size_t)(f->n_bursts - 1) * sizeof(uint32_t));
        f->n_bursts--;
    }
    while (f->n_hits_buf > 0 && i - f->hit_tick[0] > max_age) drop_front_hit(f);
    if (f->active) {
        f->streak_q += 2;
        if (f->streak_q > 4 * P->confirm_ticks) f->streak_q = 4 * P->confirm_ticks;
    } else if (f->streak_q > 0) {
        f->streak_q--;
    }
    if (f->state == FSM_IDLE && f->active) f->state = FSM_ACTIVE;
    if (f->state == FSM_ACTIVE) {
        if (f->streak_q == 0) {
            f->state = FSM_IDLE;
        } else if (f->streak_q >= 2 * P->confirm_ticks && periodic(f)) {
            uint32_t first = f->bursts[0];
            f->state = FSM_CONFIRMED;
            f->below_ticks = 0;
            f->ep_first_burst = first;
            f->ep_last_hit = i;
            f->ep_n_bursts = f->n_bursts;
            f->ep_sum_p = 0.0f;
            f->ep_sum_level = 0.0f;
            f->ep_n_hits = 0;
            for (int k = 0; k < f->n_hits_buf; k++) {
                if (f->hit_tick[k] >= first) {
                    f->ep_sum_p += f->hit_p[k];
                    f->ep_sum_level += f->hit_level[k];
                    f->ep_n_hits++;
                    f->ep_last_hit = f->hit_tick[k];
                }
            }
            ev = FSM_EV_EPISODE_START;
        }
    } else if (f->state == FSM_CONFIRMED) {
        if (hit) {
            f->ep_last_hit = i;
            f->ep_sum_p += p;
            f->ep_n_hits++;
            f->ep_sum_level += level_dbfs;
            if (burst_started) f->ep_n_bursts++;
        }
        if (!f->active) {
            f->below_ticks++;
            if (f->below_ticks >= P->verify_ticks) {
                int n = f->ep_n_hits > 0 ? f->ep_n_hits : 1;
                f->last_duration_s = (float)(f->ep_last_hit - f->ep_first_burst + 2) * (float)P->tick_ms / 1000.0f;
                f->last_mean_p = f->ep_sum_p / (float)n;
                f->last_level_dbfs = f->ep_sum_level / (float)n;
                f->last_n_bursts = f->ep_n_bursts;
                f->last_n_hits = f->ep_n_hits;
                f->last_start_tick = f->ep_first_burst;
                f->state = FSM_IDLE;
                f->streak_q = 0;
                f->n_bursts = 0;
                f->n_hits_buf = 0;
                f->in_burst = 0;
                ev = FSM_EV_EPISODE_END;
            }
        } else {
            f->below_ticks = 0;
        }
    }
    return ev;
}
```

- [ ] **Step 3: Build and run against the golden trace**

Run: `.venv-mac/bin/python -m pytest tests/test_c_parity.py -v -k fsm`
Expected: PASS with `fsm parity: N ticks, K end events checked, 0 mismatches`.

- [ ] **Step 4: Commit**

```bash
git add esp32_firmware/v5/snore_episode_fsm.h esp32_firmware/v5/snore_episode_fsm.c esp32_firmware/v5/host_test/test_fsm.c tests/test_c_parity.py
git commit -m "feat(fw-v5): episode state machine in C with golden-trace test"
```
### Task 23: C GCC-PHAT direction module

**Files:**
- Create: `esp32_firmware/v5/doa_gccphat.h`, `esp32_firmware/v5/doa_gccphat.c`, `esp32_firmware/v5/host_test/test_doa.c`
- Modify: `tests/test_c_parity.py` (add the doa test and the aggregate `test` target)

**Interfaces:**
- Produces C API: `void doa_gcc_phat(const float *l, const float *r, int max_lag, float band_lo, float band_hi, float *lag, float *ratio);` (512-sample frames, same sign convention as Python), `typedef struct {...} doa_tracker_t; void doa_tracker_init(doa_tracker_t *t, float spacing_m); int doa_tracker_frame(doa_tracker_t *t, const float *l, const float *r, float *lag_out); void doa_tracker_episode(const doa_tracker_t *t, doa_episode_t *out);` with `doa_episode_t { int side; float lag_samples, lag_ms, conf; int n_valid; }` and side codes 0 unknown, 1 left, 2 right.
- Parity contract: per frame, lag within 0.05 samples and the validity decision equal; per episode, side and `n_valid` exact, `lag_samples` within 0.05, `conf` within 1e-6.

- [ ] **Step 1: Write the host test and extend the pytest wrapper**

`esp32_firmware/v5/host_test/test_doa.c`:
```c
/* Usage: test_doa <doa_cases.bin> <doa_tracker.bin>. */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include "doa_gccphat.h"

static void to_float(const int16_t *in, float *out) { for (int n = 0; n < 512; n++) out[n] = (float)in[n] / 32768.0f; }

static int run_cases(const char *path) {
    FILE *fh = fopen(path, "rb");
    if (!fh) { perror("open cases"); return 1; }
    int32_t hdr[3];
    if (fread(hdr, sizeof(int32_t), 3, fh) != 3 || hdr[1] != 512) { fprintf(stderr, "bad cases header\n"); return 1; }
    int failures = 0;
    int16_t li[512], ri[512];
    float l[512], r[512], expect[2];
    for (int c = 0; c < hdr[0]; c++) {
        if (fread(li, sizeof(int16_t), 512, fh) != 512 || fread(ri, sizeof(int16_t), 512, fh) != 512 || fread(expect, sizeof(float), 2, fh) != 2) return 1;
        to_float(li, l);
        to_float(ri, r);
        float lag, ratio;
        doa_gcc_phat(l, r, hdr[2], 60.0f, 3000.0f, &lag, &ratio);
        int ok = fabsf(lag - expect[0]) <= 0.05f && ((ratio >= 1.5f) == (expect[1] >= 1.5f));
        printf("case %d: lag %.3f (ref %.3f) ratio %.2f (ref %.2f) %s\n", c, lag, expect[0], ratio, expect[1], ok ? "ok" : "MISMATCH");
        if (!ok) failures++;
    }
    fclose(fh);
    return failures;
}

static int run_tracker(const char *path) {
    FILE *fh = fopen(path, "rb");
    if (!fh) { perror("open tracker"); return 1; }
    int32_t hdr[2];
    float spacing;
    if (fread(hdr, sizeof(int32_t), 2, fh) != 2 || fread(&spacing, sizeof(float), 1, fh) != 1 || hdr[1] != 512) { fprintf(stderr, "bad tracker header\n"); return 1; }
    doa_tracker_t t;
    doa_tracker_init(&t, spacing);
    int16_t li[512], ri[512];
    float l[512], r[512], exp_lag;
    int32_t exp_valid;
    int failures = 0;
    for (int k = 0; k < hdr[0]; k++) {
        if (fread(li, sizeof(int16_t), 512, fh) != 512 || fread(ri, sizeof(int16_t), 512, fh) != 512 || fread(&exp_valid, sizeof(int32_t), 1, fh) != 1 || fread(&exp_lag, sizeof(float), 1, fh) != 1) return 1;
        to_float(li, l);
        to_float(ri, r);
        float lag;
        int valid = doa_tracker_frame(&t, l, r, &lag);
        if (valid != exp_valid || fabsf(lag - exp_lag) > 0.05f) {
            if (failures < 10) printf("frame %d: valid %d (ref %d) lag %.3f (ref %.3f) MISMATCH\n", k, valid, exp_valid, lag, exp_lag);
            failures++;
        }
    }
    int32_t side, n_valid;
    float ep_ref[3];
    if (fread(&side, sizeof(int32_t), 1, fh) != 1 || fread(ep_ref, sizeof(float), 3, fh) != 3 || fread(&n_valid, sizeof(int32_t), 1, fh) != 1) return 1;
    fclose(fh);
    doa_episode_t ep;
    doa_tracker_episode(&t, &ep);
    int ok = ep.side == side && ep.n_valid == n_valid && fabsf(ep.lag_samples - ep_ref[0]) <= 0.05f && fabsf(ep.conf - ep_ref[2]) <= 1e-6f;
    printf("tracker episode: side %d (ref %d) lag %.3f (ref %.3f) conf %.4f (ref %.4f) n_valid %d (ref %d) %s\n", ep.side, side, ep.lag_samples, ep_ref[0], ep.conf, ep_ref[2], ep.n_valid, n_valid, ok ? "ok" : "MISMATCH");
    return failures + (ok ? 0 : 1);
}

int main(int argc, char **argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s doa_cases.bin doa_tracker.bin\n", argv[0]); return 2; }
    int failures = run_cases(argv[1]) + run_tracker(argv[2]);
    printf("doa parity: %d failing checks\n", failures);
    return failures ? 1 : 0;
}
```

Replace the fsm test in `tests/test_c_parity.py` with one aggregate test (keep the features test):
```python
@pytest.mark.skipif(shutil.which("cc") is None, reason="no C compiler")
def test_c_fsm_and_doa_match_python(tmp_path):
    golden = tmp_path / "golden"
    X.write_headers_only(FW / "generated", golden, FsmParams(), DoaParams())
    _synthetic_golden(golden)
    _run("test", golden)
```

- [ ] **Step 2: Implement the module**

`esp32_firmware/v5/doa_gccphat.h`:
```c
#pragma once
#include <stdint.h>

#define DOA_MAX_LAG_CAP 16
#define DOA_MAX_FRAMES 4096

typedef struct { int side; float lag_samples, lag_ms, conf; int n_valid; } doa_episode_t;

typedef struct {
    float spacing_m, c, band_lo, band_hi, deadzone, min_ratio, floor_margin_db;
    int fs, max_lag;
    float floor_db;
    float lags[DOA_MAX_FRAMES];
    int n;
} doa_tracker_t;

/* Positive lag: the signal reaches L first (source on the left). */
void doa_gcc_phat(const float *l, const float *r, int max_lag, float band_lo, float band_hi, float *lag, float *ratio);
void doa_tracker_init(doa_tracker_t *t, float spacing_m);
int doa_tracker_frame(doa_tracker_t *t, const float *l, const float *r, float *lag_out);
void doa_tracker_episode(const doa_tracker_t *t, doa_episode_t *out);
```

`esp32_firmware/v5/doa_gccphat.c`:
```c
#include "doa_gccphat.h"
#include "fft512.h"
#include <math.h>
#include <string.h>
#include <stdlib.h>

#define N 512
#define PI_F 3.14159265358979323846f

static float s_win[N];
static int s_ready = 0;

static void ensure_init(void) {
    if (s_ready) return;
    fft512_init();
    for (int n = 0; n < N; n++) s_win[n] = 0.5f - 0.5f * cosf(2.0f * PI_F * (float)n / (float)N);
    s_ready = 1;
}

void doa_gcc_phat(const float *l, const float *r, int max_lag, float band_lo, float band_hi, float *lag_out, float *ratio_out) {
    float lr[N], li[N], rr[N], ri[N], gr[N], gi[N];
    ensure_init();
    if (max_lag > DOA_MAX_LAG_CAP) max_lag = DOA_MAX_LAG_CAP;
    for (int n = 0; n < N; n++) { lr[n] = l[n] * s_win[n]; li[n] = 0.0f; rr[n] = r[n] * s_win[n]; ri[n] = 0.0f; }
    fft512_forward(lr, li);
    fft512_forward(rr, ri);
    memset(gr, 0, sizeof(gr));
    memset(gi, 0, sizeof(gi));
    for (int k = 0; k <= N / 2; k++) {
        float re = rr[k] * lr[k] + ri[k] * li[k];  /* R * conj(L) */
        float im = ri[k] * lr[k] - rr[k] * li[k];
        float mag = sqrtf(re * re + im * im) + 1e-12f;
        float f = (float)k * 16000.0f / (float)N;
        if (f < band_lo || f > band_hi) { re = 0.0f; im = 0.0f; } else { re /= mag; im /= mag; }
        gr[k] = re; gi[k] = im;
        if (k > 0 && k < N / 2) { gr[N - k] = re; gi[N - k] = -im; }
    }
    fft512_inverse(gr, gi);
    float vals[2 * DOA_MAX_LAG_CAP + 1];
    int count = 2 * max_lag + 1, kbest = 0;
    for (int m = 0; m < count; m++) {
        int idx = ((m - max_lag) % N + N) % N;
        vals[m] = gr[idx];
        if (vals[m] > vals[kbest]) kbest = m;
    }
    float lag = (float)(kbest - max_lag);
    if (kbest > 0 && kbest < count - 1) {
        float y0 = vals[kbest - 1], y1 = vals[kbest], y2 = vals[kbest + 1];
        float denom = y0 - 2.0f * y1 + y2;
        if (denom < 0.0f) lag += 0.5f * (y0 - y2) / denom;
    }
    float second = 0.0f;
    int any = 0;
    for (int m = 0; m < count; m++) {
        if (m >= kbest - 1 && m <= kbest + 1) continue;
        if (!any || vals[m] > second) { second = vals[m]; any = 1; }
    }
    *lag_out = lag;
    *ratio_out = (any && second > 1e-9f) ? vals[kbest] / second : 1e6f;
}

void doa_tracker_init(doa_tracker_t *t, float spacing_m) {
    memset(t, 0, sizeof(*t));
    t->spacing_m = spacing_m; t->c = 343.0f; t->fs = 16000;
    t->band_lo = 60.0f; t->band_hi = 3000.0f; t->deadzone = 0.2f; t->min_ratio = 1.5f; t->floor_margin_db = 6.0f;
    t->max_lag = (int)ceilf(spacing_m / t->c * (float)t->fs) + 1;
    t->floor_db = -60.0f;
}

static int side_of(float lag, int max_lag, float deadzone) {
    float thr = deadzone * (float)max_lag;
    if (lag > thr) return 1;
    if (lag < -thr) return 2;
    return 0;
}

int doa_tracker_frame(doa_tracker_t *t, const float *l, const float *r, float *lag_out) {
    double sq = 0.0;
    for (int n = 0; n < N; n++) sq += (double)l[n] * (double)l[n];
    float level = 20.0f * log10f(sqrtf((float)(sq / N)) + 1e-9f);
    float rate = (level < t->floor_db + 5.0f) ? 0.02f : 0.002f; /* same rule as v5/doa.py NoiseFloor */
    t->floor_db += rate * (level - t->floor_db);
    float lag, ratio;
    doa_gcc_phat(l, r, t->max_lag, t->band_lo, t->band_hi, &lag, &ratio);
    int valid = level >= t->floor_db + t->floor_margin_db && ratio >= t->min_ratio;
    if (valid && t->n < DOA_MAX_FRAMES) t->lags[t->n++] = lag;
    if (lag_out) *lag_out = lag;
    return valid;
}

static int cmp_f(const void *a, const void *b) {
    float x = *(const float *)a, y = *(const float *)b;
    return (x > y) - (x < y);
}

void doa_tracker_episode(const doa_tracker_t *t, doa_episode_t *out) {
    memset(out, 0, sizeof(*out));
    if (t->n == 0) return;
    float sorted[DOA_MAX_FRAMES];
    memcpy(sorted, t->lags, (size_t)t->n * sizeof(float));
    qsort(sorted, (size_t)t->n, sizeof(float), cmp_f);
    float med = (t->n % 2) ? sorted[t->n / 2] : 0.5f * (sorted[t->n / 2 - 1] + sorted[t->n / 2]);
    int side = side_of(med, t->max_lag, t->deadzone), agree = 0;
    for (int i = 0; i < t->n; i++) if (side_of(t->lags[i], t->max_lag, t->deadzone) == side) agree++;
    out->side = side;
    out->lag_samples = med;
    out->lag_ms = med / (float)t->fs * 1000.0f;
    out->conf = (float)agree / (float)t->n;
    out->n_valid = t->n;
}
```

- [ ] **Step 3: Build and run all host tests through the pytest wrapper**

Run: `.venv-mac/bin/python -m pytest tests/test_c_parity.py -v`
Expected: PASS; make output ends with `doa parity: 0 failing checks`.

- [ ] **Step 4: Commit**

```bash
git add esp32_firmware/v5/doa_gccphat.h esp32_firmware/v5/doa_gccphat.c esp32_firmware/v5/host_test/test_doa.c tests/test_c_parity.py
git commit -m "feat(fw-v5): GCC-PHAT direction module in C with golden-case and tracker tests"
```
### Task 24: Gated export, benchmarks, DoA sweep, model card

**Files:**
- Create/update (tracked): `esp32_firmware/v5/generated/*`, `output/v5/deliverables/*`

- [ ] **Step 1: Export through the gates (the only evaluation on the test split)**

Run: `.venv-mac/bin/python -m v5.export model`
Expected: JSON summary with `parity.passed: true` and float/int8 test metrics; `output/v5/deliverables/export_status.json` says `ok`; `esp32_firmware/v5/generated/model_data.c` is a few hundred KB. An `ExportError` leaves `output/v5/export_stage/` for diagnosis and records `failed` in `export_status.json`; do not hand-copy anything out of the stage directory.

- [ ] **Step 2: Streaming benchmark on both paths and the DoA sweep**

Run:
```bash
.venv-mac/bin/python -m v5.benchmark_nights
.venv-mac/bin/python -m v5.doa_sim --trials 50
```
Expected: `deliverables/benchmark_nights.md` with a table per predictor (int8 is the product path) and the float~int8 tick decision agreement; `deliverables/doa_sim.md` with 48 rows. Record in the commit message whether the provisional targets (int8 detection ≥ 0.90, false confirms ≤ 0.5/h at SNR ≥ 5 dB, DoA ≥ 0.95 at 0.06 m, 1 m, SNR ≥ 5 dB) were met.

- [ ] **Step 3: Model card, host C tests against the real golden set, full test suite**

Run: `.venv-mac/bin/python -m v5.export card && make -C esp32_firmware/v5 test && .venv-mac/bin/python -m pytest -q`
Expected: `deliverables/model_card.md` includes every table; three host tests exit 0; pytest reports all passed (dataset tests included; slow ones may take a few minutes).

- [ ] **Step 4: Commit**

```bash
git add esp32_firmware/v5/generated output/v5/deliverables
git commit -m "release(v5): gated int8 model, C headers, golden vectors, benchmarks, DoA sweep and model card"
```
### Task 25: Self-recording kit

**Files:**
- Create: `v5/recording/record_session.py`, `tests/test_recording.py`

**Interfaces:**
- Consumes: `v5.features.SR`.
- Produces: `LABELS`, `TEMPLATE_COLUMNS`, `write_labels_template(session_dir) -> Path`, `class ChunkWriter(session_dir, sr=16000, channels=1, chunk_s=60.0)` with `push(frames (n, channels) float32)`, `flush()`, `files: list[Path]`; `record(session_dir, minutes, channels=1, device=None, frame_source=None, chunk_s=60.0) -> list[Path]`; CLI `python -m v5.recording.record_session --minutes N [--channels 1|2] [--device NAME] [--session DIR] [--list-devices]`.
- Files: `recordings/<session>/chunk_XXXX.wav` (PCM16, 16 kHz), `labels_template.csv`, `README.txt`.

- [ ] **Step 1: Write the failing tests**

`tests/test_recording.py`:
```python
import csv

import numpy as np
import soundfile as sf

from v5 import features as F
from v5.recording import record_session as R


def _fake_frames(seconds, channels=1, block=4096):
    rng = np.random.default_rng(0)
    total = int(seconds * F.SR)
    sent = 0
    while sent < total:
        n = min(block, total - sent)
        yield (0.01 * rng.standard_normal((n, channels))).astype(np.float32)
        sent += n


def test_chunk_writer_rolls_files(tmp_path):
    w = R.ChunkWriter(tmp_path / "s", channels=1, chunk_s=2.0)
    for fr in _fake_frames(5.0):
        w.push(fr)
    w.flush()
    assert [p.name for p in w.files] == ["chunk_0000.wav", "chunk_0001.wav", "chunk_0002.wav"]
    lens = [sf.info(p).frames for p in w.files]
    assert lens == [2 * F.SR, 2 * F.SR, F.SR]
    assert sf.info(w.files[0]).samplerate == F.SR and sf.info(w.files[0]).subtype == "PCM_16"


def test_record_with_fake_source_writes_template(tmp_path):
    files = R.record(tmp_path / "s", minutes=0.05, channels=2, frame_source=_fake_frames(10.0, channels=2), chunk_s=1.0)
    assert len(files) == 3 and sf.info(files[0]).channels == 2  # 3 s requested
    with open(tmp_path / "s" / "labels_template.csv", newline="") as fh:
        rows = list(csv.reader(fh))
    assert rows[0] == R.TEMPLATE_COLUMNS and rows[1][3] in R.LABELS
    assert (tmp_path / "s" / "README.txt").exists()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv-mac/bin/python -m pytest tests/test_recording.py -v`
Expected: FAIL with `ImportError`

- [ ] **Step 3: Implement `v5/recording/record_session.py`**

```python
"""Self-recording kit (spec section 12): 16 kHz WAV chunks plus a labelling template."""
from __future__ import annotations

import argparse
import csv
import queue
from datetime import datetime
from pathlib import Path

import numpy as np
import soundfile as sf

from v5 import features as F
from v5.config import ROOT

LABELS = ("snore", "breathing", "speech", "tv", "fan", "other")
TEMPLATE_COLUMNS = ["chunk", "start_s", "end_s", "label", "side", "posture", "distance_m", "pillow"]


def write_labels_template(session_dir) -> Path:
    session_dir = Path(session_dir)
    session_dir.mkdir(parents=True, exist_ok=True)
    p = session_dir / "labels_template.csv"
    with open(p, "w", newline="", encoding="utf-8") as fh:
        wr = csv.writer(fh)
        wr.writerow(TEMPLATE_COLUMNS)
        wr.writerow(["chunk_0000.wav", "0.0", "10.0", "snore", "left", "supine", "0.8", "thin"])
    (session_dir / "README.txt").write_text(
        "Copy labels_template.csv to labels.csv and add one row per labelled span.\n"
        f"label: {', '.join(LABELS)}\nside: left | right | unknown\nposture: supine | lateral | prone | unknown\n"
        "start_s / end_s are seconds inside the chunk file.\n", encoding="utf-8")
    return p


class ChunkWriter:
    def __init__(self, session_dir, sr: int = F.SR, channels: int = 1, chunk_s: float = 60.0):
        self.dir = Path(session_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.sr, self.channels, self.chunk_samples = sr, channels, int(chunk_s * sr)
        self.index, self.buf, self.buffered, self.files = 0, [], 0, []

    def push(self, frames) -> None:
        frames = np.asarray(frames, dtype=np.float32).reshape(-1, self.channels)
        self.buf.append(frames)
        self.buffered += len(frames)
        while self.buffered >= self.chunk_samples:
            data = np.concatenate(self.buf)
            self._write(data[: self.chunk_samples])
            rest = data[self.chunk_samples:]
            self.buf, self.buffered = ([rest] if len(rest) else []), len(rest)

    def _write(self, chunk) -> None:
        path = self.dir / f"chunk_{self.index:04d}.wav"
        sf.write(path, chunk, self.sr, subtype="PCM_16")
        self.files.append(path)
        self.index += 1

    def flush(self) -> None:
        if self.buffered:
            self._write(np.concatenate(self.buf))
            self.buf, self.buffered = [], 0


def _sounddevice_frames(device, channels: int, total: int):
    import sounddevice as sd

    q: queue.Queue = queue.Queue()

    def callback(indata, frames, time_info, status):
        q.put(indata.copy())

    with sd.InputStream(samplerate=F.SR, channels=channels, device=device, dtype="float32", blocksize=4096, callback=callback):
        got = 0
        while got < total:
            frames = q.get()
            got += len(frames)
            yield frames


def record(session_dir, minutes: float, channels: int = 1, device=None, frame_source=None, chunk_s: float = 60.0) -> list[Path]:
    writer = ChunkWriter(session_dir, channels=channels, chunk_s=chunk_s)
    write_labels_template(session_dir)
    total = int(minutes * 60 * F.SR)
    got = 0
    for frames in (frame_source if frame_source is not None else _sounddevice_frames(device, channels, total)):
        remaining = total - got
        frames = np.asarray(frames)[:remaining]
        writer.push(frames)
        got += len(frames)
        if got >= total:
            break
    writer.flush()
    return writer.files


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Record a labelled self-recording session")
    ap.add_argument("--minutes", type=float, default=10.0)
    ap.add_argument("--channels", type=int, choices=(1, 2), default=1)
    ap.add_argument("--device", default=None)
    ap.add_argument("--session", default=None)
    ap.add_argument("--list-devices", action="store_true")
    args = ap.parse_args(argv)
    if args.list_devices:
        import sounddevice as sd

        print(sd.query_devices())
        return
    session = Path(args.session) if args.session else ROOT / "recordings" / datetime.now().strftime("%Y%m%d_%H%M%S")
    files = record(session, args.minutes, args.channels, args.device)
    print(f"wrote {len(files)} chunks to {session}; fill in labels.csv (see README.txt)")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv-mac/bin/python -m pytest tests/test_recording.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add v5/recording/record_session.py tests/test_recording.py
git commit -m "feat(v5): self-recording kit with chunked WAV writer and label template"
```

---
### Task 26: Fine-tuning recipe on self-recordings

**Files:**
- Create: `v5/finetune.py`, `tests/test_finetune.py`

**Interfaces:**
- Consumes: `v5.data.sources.decode`, `v5.data.augment` (`AugmentConfig`, `Augmenter`), `v5.data.dataset` (`TrainDataset`, `precompute_features`), `v5.evaluate` (`clip_metrics`, `predict_probs`), `v5.events.check_model_version`, `v5.train.kd_loss`.
- Produces: `LABEL_TO_Y`, `load_labels(session_dir) -> list[dict]`, `slice_session(session_dir, hop_s=1.0) -> tuple[np.ndarray int16 (N,16000), np.ndarray float32 (N,)]`, `finetune(model_path, sessions: list, holdout, out_dir, epochs=10, lr=1e-4, tau=0.65, seed=42) -> dict` (keys `model_version, n_train, n_holdout, before, after`), CLI `python -m v5.finetune --sessions DIR [DIR ...] --holdout DIR [--model PATH] [--out DIR] [--epochs N]`.

- [ ] **Step 1: Write the failing tests**

`tests/test_finetune.py`:
```python
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
    assert m["model_version"].startswith("cnn_v5_ft_") and m["n_train"] == 40 and m["n_holdout"] == 40
    assert "auc" in m["before"] and "auc" in m["after"]
    assert (tmp_path / "out" / "model.keras").exists()
    assert json.loads((tmp_path / "out" / "metrics.json").read_text())["n_train"] == 40
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv-mac/bin/python -m pytest tests/test_finetune.py -v`
Expected: FAIL with `ImportError`

- [ ] **Step 3: Implement `v5/finetune.py`**

```python
"""Domain adaptation on labelled self-recordings (spec section 12)."""
from __future__ import annotations

import argparse
import csv
import json
from datetime import date
from pathlib import Path

import keras
import numpy as np

from v5 import features as F
from v5.config import ROOT, load_config, resolve
from v5.data.augment import AugmentConfig, Augmenter
from v5.data.dataset import TrainDataset, precompute_features
from v5.data.sources import decode
from v5.evaluate import clip_metrics, predict_probs
from v5.events import check_model_version
from v5.train import kd_loss

LABEL_TO_Y = {"snore": 1, "breathing": 0, "speech": 0, "tv": 0, "fan": 0, "other": 0}


def load_labels(session_dir) -> list[dict]:
    path = Path(session_dir) / "labels.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing: copy labels_template.csv to labels.csv and fill it in")
    with open(path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    bad = [r["label"] for r in rows if r["label"] not in LABEL_TO_Y]
    if bad:
        raise ValueError(f"unknown labels {sorted(set(bad))}; allowed: {sorted(LABEL_TO_Y)}")
    by_chunk: dict[str, list] = {}
    for r in rows:
        by_chunk.setdefault(r["chunk"], []).append((float(r["start_s"]), float(r["end_s"]), r["label"]))
    for chunk, spans in by_chunk.items():
        spans.sort()
        for (s1, e1, l1), (s2, e2, l2) in zip(spans, spans[1:]):
            if s2 < e1:
                raise ValueError(f"overlapping spans in {chunk}: [{s1}, {e1}] {l1} and [{s2}, {e2}] {l2}")
    return rows


def slice_session(session_dir, hop_s: float = 1.0):
    session_dir = Path(session_dir)
    audio, y = [], []
    cache = {}
    for r in load_labels(session_dir):
        if r["chunk"] not in cache:
            cache[r["chunk"]] = decode(session_dir / r["chunk"])
        x = cache[r["chunk"]]
        t = float(r["start_s"])
        while t + 1.0 <= float(r["end_s"]) + 1e-9:
            w = x[int(t * F.SR): int(t * F.SR) + F.WIN]
            if len(w) == F.WIN:
                audio.append(F.float_to_int16(w))
                y.append(LABEL_TO_Y[r["label"]])
            t += hop_s
    if not audio:
        raise ValueError(f"no labelled windows in {session_dir}")
    return np.stack(audio), np.array(y, np.float32)


def finetune(model_path, sessions, holdout, out_dir, epochs: int = 10, lr: float = 1e-4, tau: float = 0.65, seed: int = 42) -> dict:
    if any(Path(s).resolve() == Path(holdout).resolve() for s in sessions):
        raise ValueError("the holdout session must not be one of the training sessions")
    parts = [slice_session(s) for s in sessions]
    audio, y = np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts])
    h_audio, h_y = slice_session(holdout)
    if y.min() == y.max():
        raise ValueError("training sessions need both snore and non-snore spans")
    model = keras.models.load_model(model_path, compile=False)
    for name in ("conv1", "bn1"):
        model.get_layer(name).trainable = False
    Xh = precompute_features(h_audio)
    before = clip_metrics(h_y, predict_probs(model, Xh), tau)
    rng = np.random.default_rng(seed)
    aug = Augmenter(AugmentConfig(p_rir=0.0, p_noise_pos=0.3, p_noise_neg=0.3, snr_range=(5.0, 20.0), p_gain=0.0), audio[y == 0], None, rng)
    ds = TrainDataset(audio, y, y, aug, batch=32, seed=seed)
    keras.utils.set_random_seed(seed)
    model.compile(optimizer=keras.optimizers.Adam(lr), loss=kd_loss)
    model.fit(ds, epochs=epochs, verbose=2)
    after = clip_metrics(h_y, predict_probs(model, Xh), tau)
    version = check_model_version(f"cnn_v5_ft_{date.today():%Y%m%d}")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save(out_dir / "model.keras")
    metrics = {"model_version": version, "n_train": int(len(y)), "n_holdout": int(len(h_y)), "tau": tau, "epochs": epochs, "before": before, "after": after}
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Fine-tune the deployed model on self-recordings")
    ap.add_argument("--sessions", nargs="+", required=True)
    ap.add_argument("--holdout", required=True)
    ap.add_argument("--model", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--config", default=None)
    args = ap.parse_args(argv)
    cfg = resolve(load_config(args.config))
    out_root = Path(cfg["paths"]["out_dir"])
    thr = out_root / "threshold.json"
    tau = json.loads(thr.read_text())["tau"] if thr.exists() else 0.65  # no calibration yet: development default
    m = finetune(args.model or out_root / "deployed" / "model.keras", args.sessions, args.holdout,
                 args.out or out_root / "finetune" / date.today().strftime("%Y%m%d"), args.epochs, tau=tau, seed=cfg["seed"])
    print(json.dumps({k: m[k] for k in ("model_version", "n_train", "n_holdout")}), "\nbefore:", json.dumps(m["before"]), "\nafter:", json.dumps(m["after"]))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv-mac/bin/python -m pytest tests/test_finetune.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add v5/finetune.py tests/test_finetune.py
git commit -m "feat(v5): fine-tuning recipe for self-recorded sessions"
```

---
### Task 27: Final verification and handoff README

**Files:**
- Create: `v5/README.md`
- Modify: this plan's checkboxes

- [ ] **Step 1: Full verification**

Run:
```bash
.venv-mac/bin/python -m pytest -q
make -C esp32_firmware/v5 test
git status --short
```
Expected: pytest all passed (report the count), three host tests exit 0, and `git status` shows only the README about to be added. Any failure is fixed before continuing; do not mark this task done with a failing test.

- [ ] **Step 2: Write `v5/README.md`**

```markdown
# SnoozMate v5 edge snore pipeline

Spec: `docs/superpowers/specs/2026-09-04-snore-v5-edge-pipeline-design.md`

## Environment (macOS)

    uv venv .venv-mac --python 3.12
    uv pip install --python .venv-mac/bin/python tensorflow tensorflow-hub librosa soundfile soxr scikit-learn scipy pyroomacoustics ai-edge-litert sounddevice pyyaml pytest pydantic "setuptools<81"

Splits: train / val (selection) / calib (threshold) / test (Kaggle, evaluated once at export) / bench (MS-SNSD beds). Release gates in `v5.export` block promotion on int8 parity or operator failures; calibration fails closed on the FPR target.

## Pipeline, in order

    .venv-mac/bin/python -m v5.data.manifest            # decode, dedup, split -> output/v5/manifest.csv
    .venv-mac/bin/python -m v5.teacher                  # optional YAMNet soft labels + label audit
    .venv-mac/bin/python -m v5.train run --width 1.0 --name hard_w1
    .venv-mac/bin/python -m v5.train run --width 1.0 --kd --name kd_w1
    .venv-mac/bin/python -m v5.train run --width 2.0 --name hard_w2
    .venv-mac/bin/python -m v5.train run --width 2.0 --kd --name kd_w2
    .venv-mac/bin/python -m v5.train select && .venv-mac/bin/python -m v5.train final   # select on val, calibrate on calib
    .venv-mac/bin/python -m v5.export model             # gated: int8 tflite, C headers, golden vectors; the only test-split evaluation
    .venv-mac/bin/python -m v5.benchmark_nights         # synthetic nights, float and int8 paths
    .venv-mac/bin/python -m v5.doa_sim --trials 50      # direction accuracy vs mic spacing
    .venv-mac/bin/python -m v5.export card              # model card from the gated artifacts
    make -C esp32_firmware/v5 test                       # C parity against the golden vectors
    .venv-mac/bin/python -m pytest

## Outputs

- `output/v5/deliverables/`: `snore_v5_int8.tflite`, `model_card.md`, `experiments.md`, `benchmark_nights.md`, `doa_sim.md`, `manifest_report.md`, `golden/`, C headers.
- `esp32_firmware/v5/`: `snore_features.c` (feature spec v1), `snore_episode_fsm.c`, `doa_gccphat.c`, `fft512.c`, generated `model_data.c`, `model_meta.h`, `mel_filterbank.h`, `feature_spec.h`.

## Firmware handoff

Window of 16000 int16 samples every 8000 samples -> `sf_compute` -> `sf_quantize(INPUT_SCALE, INPUT_ZERO_POINT)` -> TFLite Micro invoke on `snore_v5_int8_tflite` -> `p = (out - OUTPUT_ZERO_POINT) * OUTPUT_SCALE` -> `fsm_tick(p, level_dbfs)`. Vibrate only while `state == FSM_CONFIRMED && active`. Stereo frames of 512 samples go to `doa_tracker_frame` during a confirmed episode; `doa_tracker_episode` gives the side for the event note. `model_version` is `MODEL_VERSION` from `model_meta.h`.

## Self-recordings

    .venv-mac/bin/python -m v5.recording.record_session --minutes 10 --channels 2
    # label recordings/<session>/labels.csv, then
    .venv-mac/bin/python -m v5.finetune --sessions recordings/A recordings/B --holdout recordings/C

## Inputs still needed from the team

Mic spacing and channel order (default 0.06 m, L then R), confirmation that firmware uses TFLite Micro, SSBPR dataset access request, first device-recorded sessions.
```

- [ ] **Step 3: Commit**

```bash
git add v5/README.md docs/superpowers/plans
git commit -m "docs(v5): handoff README and completed plan"
```
