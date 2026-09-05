# SnoozMate v5 edge snore model and streaming pipeline — design

Date: 2026-09-04
Project: `snoring-ml-training` (sibling of `snoozmate-cloud-upload-latest`)
Status: approved in chat, spec written for implementation planning

## 1. Purpose and scope

The two product decks ("酣眠 产品形态与运营场景", "好眠 SnoozMate 产品介绍路演") describe one night loop:
continuous periodic snoring (连续鼾声) is cross-confirmed with in-bed state and target posture
(交叉确认), the pad on the snorer's side vibrates progressively (渐进提醒), stops as soon as the
snoring stops or the person turns (即时停止), and the result is recorded (记录结果). Safety limits
are deterministic and never learned. Seven-night learning tunes only intensity, upgrade speed and
cooldown inside those limits.

This spec covers the parts of that loop that can be built from the data on hand:

| Deck requirement | Covered by | Trainable now? |
|---|---|---|
| 鼾声初筛 at 0.5–1 m on ESP32-S3 (本地蒸馏模型) | v5 edge model, feature spec, export | yes |
| 连续周期性鼾声 / 鼾声停止 / 响应时间 | streaming episode state machine | deterministic |
| 声源方向 → 只提醒打鼾者一侧; 双方都打鼾则不干预 | GCC-PHAT direction module | DSP + simulation |
| 在床 / 体动 / 翻身 (radar) | edge event schema, fields logged for later training | no data yet |
| 仰卧 vs 侧卧 | deferred: SSBPR acoustic posture (needs dataset access) | no data yet |
| 七晚学习, 晨报, 伴侣净收益 | already in the cloud repo (bandit, LLM reports) | n/a |
| 自录数据 > 下载数据 (README) | self-recording kit + fine-tune recipe | after MVP |

Out of scope: firmware integration and flashing (Arthur), cloud repo changes, mini program, radar
models, SSBPR posture head, auxiliary sound-class head, safety state machine (must stay
deterministic and lives in firmware + `engine.py`).

Existing v4 artifacts under `output/` are left untouched. All new outputs go to `output/v5/`.

## 2. Repository layout

```
snoring-ml-training/
  v5/                         python package (new)
    features.py               frozen feature spec v1 + numpy reference extractor + golden vectors
    data/sources.py           per-source decode + slicing rules
    data/manifest.py          scan, dedup, group ids, splits -> output/v5/manifest.csv
    data/augment.py           waveform augmentation (RIR bank, noise mixing, tilt, shift, gain)
    data/dataset.py           keras PyDataset producing (features, label, soft_label)
    teacher.py                optional YAMNet wrapper: soft labels + label audit
    model.py                  student CNN builder
    train.py                  protocol A / protocol B training, selection, threshold.json
    evaluate.py               clip-level metrics, robustness sweeps, int8 parity
    streaming.py              episode FSM reference implementation (mirrors C)
    nights.py                 synthetic night generator
    benchmark_nights.py       streaming benchmark -> product metrics
    doa.py                    GCC-PHAT reference + episode aggregation
    doa_sim.py                pyroomacoustics validation sweep
    events.py                 edge_event_v1 dataclass + cloud EventIn payload converter
    export.py                 int8 TFLite, C headers, golden vectors, model card
    recording/record_session.py   capture script
    recording/labels_template.csv
    finetune.py               domain adaptation on self-recordings
    configs/default.yaml      every tunable in one place
  esp32_firmware/v5/          C sources + host tests (see section 11)
  tests/                      pytest
  output/v5/                  generated artifacts (git-ignored except deliverables/)
  docs/superpowers/specs/     this spec; plans under docs/superpowers/plans/
```

Environment: Python 3.12 venv `.venv-mac` created with uv (the existing `.venv` is a Windows venv).
Dependencies: tensorflow 2.21, tensorflow-hub, librosa, soundfile, soxr, scikit-learn, scipy,
pyroomacoustics, ai-edge-litert (interpreter), sounddevice, pyyaml, pytest. Training runs on CPU.

## 3. Feature spec v1 (frozen)

All constants live in `v5/features.py` and are emitted to `output/v5/feature_spec.json` and to
`esp32_firmware/v5/feature_spec.h`. Nothing else may redefine them.

| Constant | Value |
|---|---|
| sample rate | 16000 Hz, mono, int16 on device (float = int16 / 32768 in Python) |
| analysis window | 1.0 s = 16000 samples; streaming hop 0.5 s (2 inferences per second) |
| STFT | n_fft 512, hop 256, periodic Hann `w[n] = 0.5 - 0.5 cos(2πn/512)`, no padding, no centering |
| frames | 61 = 1 + (16000 - 512) // 256 |
| spectrum | power `P[k] = |X[k]|^2`, k = 0..256 |
| mel filterbank | 30 triangular filters, HTK mel scale, fmin 40 Hz, fmax 6000 Hz, peak gain 1.0, evaluated at bin centre frequencies `k * 16000 / 512` |
| log | `L = 10 log10(M + 1e-10)` |
| normalisation | `X = clip((L - mean(L)) / 40, -1, 1)`, mean over all 61×30 cells |
| output | float32 (61, 30), gain-invariant; digital silence maps to all zeros |

The filterbank is generated once in Python and written as a sparse C table
(`mel_filterbank.h`: per filter `start_bin`, `n_bins`, weights). C never recomputes it.

Int8 quantisation of the model input uses the scale and zero point that the TFLite converter
assigns; `export.py` writes them into `model_meta.h`, and the C feature code applies
`q = clamp(round(X / scale) + zero_point, -128, 127)`.

Golden vectors (`output/v5/golden/features.npz` + `.bin`): 12 windows — 100 Hz sine at −20 dBFS,
white noise at −40 dBFS, 1 kHz→100 Hz chirp, digital silence, four snore clips and four noise
clips from the Kaggle test set — with their expected X. Parity tolerance for the C implementation:
max |ΔX| ≤ 5e-3 before quantisation; after quantisation at most ±1 LSB on ≤ 1 % of cells.

## 4. Data

### 4.1 Sources

| id | path | label | group key | role |
|---|---|---|---|---|
| kaggle_adria | `dataset/adrianagaler/{snore,noise}` | folder | near-dup cluster id (protocol B); whole source (protocol A) | cross-domain test |
| kaggle_jibran | `dataset/snoring_extra/jibran/jibran_{s,n}_*` | filename | same as above | cross-domain test |
| whl_s | `dataset/whltalent/s*/` 834 raw 10 s recordings | snore | recording file | train/val |
| whl_e | `dataset/whltalent/e*/` 1520 raw 10 s recordings | noise | recording file | train/val |
| esc50 | `dataset/esc50/audio` 2000 × 5 s @ 44.1 kHz + `meta/esc50.csv` | category `snoring` → snore, all other categories → noise | fold 1–5 (fold 5 = val) | train/val |
| mssnsd | `dataset/RAW/MS-SNSD/noise_train` 128 files | noise | file; category from filename prefix | train/val by file; 20 % of files (seeded, 42) held out as benchmark noise beds and never used for training or augmentation |
| wild | `Snore_Detection_Project/.../inference_audios/snore{1..6}.wav` | snore | file | sanity listing only, no metrics |

Excluded on purpose: `dataset/snore/synth_*` (synthetic), the pre-sliced `dataset/snore`,
`dataset/noise`, `dataset/whltalent_extra`, `dataset/esc50_16k_1s` (superseded by re-slicing the raw
files with group ids), `Snore_Detection_Project/data/{0,1}` (raw of the same Kaggle set), the
stereo `sms.wav` (unknown label).

### 4.2 Decoding and slicing

Decode with soundfile, average channels to mono, resample to 16 kHz with soxr. Keep original
levels. Per-window RMS uses 32 ms frames.

- whl_s: frame-RMS peaks at least 1.5 s apart and above the 60th percentile of the file's frame
  RMS; take up to 3 one-second windows centred on the highest peaks. Files with no qualifying
  peak are skipped and logged.
- whl_e: up to 2 windows: one centred on the maximum-RMS frame, one uniformly random.
- esc50: up to 2 windows centred on RMS peaks ≥ 1 s apart (up to 3 for the snoring class).
  Windows with RMS < −80 dBFS (digital silence) are dropped.
- mssnsd: non-overlapping one-second windows every 5 s, at most 40 per file.
- kaggle: one-second clips used as is (centre crop or zero pad).

### 4.3 Deduplication

1. Exact: md5 of the decoded int16 mono 16 kHz window; duplicates dropped, keeping the first by
   source priority (whl, esc50, mssnsd, kaggle_adria, kaggle_jibran).
2. Near: fingerprint = feature X flattened and standardised; cosine ≥ 0.98 joins two windows in
   one cluster (union-find over a blocked similarity matrix). Every window carries its
   `dup_cluster` id.
3. Split rule: a cluster may live on one side of any split only. If a test or validation window
   shares a cluster with a training window, the test/validation window is dropped. Counts of
   dropped windows are written to `output/v5/manifest_report.md`.

### 4.4 Splits

- Protocol A (cross-domain): test = all Kaggle windows. Train/val from the other sources with a
  group-stratified 85/15 split (WHLTalent recording, ESC-50 fold 5 as val, MS-SNSD file).
- Protocol B (deployment): Kaggle joins train/val using `dup_cluster` as its group; same 85/15
  group split.
- Both protocols assert zero group overlap and zero cluster overlap between splits.
- Class balance: batches are sampled 1:2 positive:negative; loss is unweighted.

Expected volumes (protocol A): ≈ 2.6 k positive and ≈ 10 k negative windows for training,
≈ 1 k Kaggle test windows after dedup.

### 4.5 Label audit (only when the teacher is available)

YAMNet scores on clean windows: positives with `P(Snoring) + P(Snort) < 0.02` are flagged
`weak_positive` and kept; negatives with `> 0.5` are flagged `snore_in_negative` and excluded
from training. Flag counts go to `manifest_report.md`.

## 5. Augmentation (training only, waveform domain, independent per sample)

| Step | Parameters | Probability |
|---|---|---|
| room impulse response | bank of 300 pyroomacoustics ShoeBox RIRs: room 3–5 m × 3–5 m × 2.4–3.0 m, RT60 0.2–0.6 s, source–mic distance 0.4–1.8 m, mic height 0.5–0.9 m, source height 0.4–0.7 m, ISM order 10, direct-path peak normalised to 1 | 0.6 |
| additive noise | random one-second window from the training noise bank (MS-SNSD train files + ESC-50 negative windows), SNR uniform in [−5, 20] dB by RMS | 0.7 for positives, 0.4 for negatives |
| spectral tilt | `y = x + a (x − lowpass_1kHz(x))`, a uniform in [−0.5, 0.5] | 0.3 |
| time shift | uniform in [−100, 100] ms, zero fill | 0.5 |
| gain and clipping | gain uniform in [−12, +6] dB, then hard clip to [−1, 1] | 0.3 |
| SpecAugment (on X) | one frequency mask ≤ 4 mels, one time mask ≤ 8 frames, filled with 0 | 0.5 |

RNG is seeded per epoch for reproducibility. The RIR bank is generated once and cached under
`output/v5/rir_bank.npz`.

## 6. Model and training

Student (`snore_v5`, Keras):

```
Input (61, 30, 1)
Conv2D 16 3×3 same + BN + ReLU, MaxPool 2×2      -> (30, 15, 16)
Conv2D 32 3×3 same + BN + ReLU, MaxPool 2×2      -> (15, 7, 32)
Conv2D 64 3×3 same + BN + ReLU, MaxPool 2×2      -> (7, 3, 64)
GlobalAveragePooling2D                            -> 64
Dense 32 ReLU, Dropout 0.3
Dense 1 (logit)
```

≈ 25 k parameters, ≈ 4.3 M MACs per inference; ops limited to CONV_2D, MAX_POOL_2D, MEAN,
FULLY_CONNECTED so both TFLite Micro and a handwritten path can run it. A `width` multiplier
(default 1.0) exists for the size comparison experiment; the deployed width is chosen by the
selection rule below.

Training: Adam, learning rate 2e-3 with cosine decay to 2e-5 over 60 epochs, batch 64 sampled
1:2, BCE from logits, early stopping on validation AUC with patience 10 and best-weight restore,
seed 42.

Knowledge distillation (the deck's 本地蒸馏模型), optional: teacher = YAMNet via tensorflow-hub.
Teacher logit `z = logit(P(Snoring) + P(Snort))` on the clean window is Platt-calibrated on the
training split (`t = σ(a z + b)` fitted against folder labels), cached per window, and used as a
soft target for every augmented view of that window: `loss = 0.5·BCE(y) + 0.5·BCE(t)`.

Selection rule (protocol A, Kaggle test): compare {hard-label, KD} × {width 1.0, width 2.0} and
pick the smallest configuration whose AUC is within 0.005 of the best and whose recall at
2 % false-positive rate is within 2 points of the best. Retrain that configuration under
protocol B for deployment.

Threshold τ: on the protocol B validation split, the smallest τ with clip-level FPR ≤ 1 %;
fallback 0.65 if fewer than 200 negatives are available. Written to `output/v5/threshold.json`
together with the FSM parameters of section 8 and `model_version = "cnn_v5_int8"`.
The version string must never contain `simulator`, `demo` or `mock`; the cloud treats such
events as simulated data.

## 7. Evaluation and acceptance

Clip level (`evaluate.py`), reported for validation, Kaggle test, and Kaggle test under
robustness sweeps (SNR 20/10/5/0 dB with held-out MS-SNSD beds; RIR at 0.5/1.0/1.5 m):
AUC, recall at 2 % FPR, precision/recall/FNR/FPR at τ, confusion matrix.

Int8 parity: fp32 vs int8 probabilities on the test set; pass if ΔAUC < 0.005 and decision
agreement at τ ≥ 99 %.

Streaming benchmark (`benchmark_nights.py`): 20 synthetic nights of 1 h each, seeded.
Each night = held-out MS-SNSD noise bed at −50…−30 dBFS, optional RIR, 6–12 snore episodes of
20–120 s built from WHLTalent snore windows of the protocol B validation split (never trained
on) with burst period 2.5–5 s and breathing gaps, plus 20–40 distractor events (speech, cough,
door, typing, vacuum from validation-split negative windows). Ground truth = episode intervals. The pipeline runs features → model → FSM at 2 Hz.
Metrics: episode detection rate, confirm latency (episode start → CONFIRMED), false confirms per
hour (CONFIRMED outside any episode ± 5 s), snore-stop latency (true episode end → the FSM's
`active` flag drops, which is the signal the intervention loop uses; the bookkeeping
`episode_end` event follows verify_window_seconds later and is reported separately).

Provisional targets, reported not promised: detection ≥ 90 %, confirm latency ≤ confirm_seconds
+ 3 s, false confirms ≤ 0.5 per hour at SNR ≥ 5 dB. Results at all SNRs go into the model card
unfiltered.

## 8. Streaming episode state machine (deterministic)

Reference in `v5/streaming.py`, mirrored bit-for-bit in `snore_episode_fsm.c`. Tick every 0.5 s
with probability `p`.

- `hit = p ≥ τ`
- `activity = max(p over the last hold_seconds)`; `active = activity ≥ τ` (hold bridges breathing
  gaps; default hold 6 s)
- streak: `active` → `streak += 0.5`; else `streak = max(0, streak − 0.25)` (same 2:1 ratio as
  `engine.py`)
- bursts: a burst is a maximal run of consecutive hits; burst start times are kept for the last
  `confirm_seconds + hold_seconds`
- periodic: at least `min_bursts` (3) bursts in that window and the median inter-burst interval in
  [1.5, 7] s (the firmware's SD_MIN_PERIOD_MS/SD_MAX_PERIOD_MS)
- states: IDLE → ACTIVE when `active`; ACTIVE → CONFIRMED when `streak ≥ confirm_seconds` and
  `periodic` (emit `episode_start`); CONFIRMED → IDLE when `activity < τ` for
  `verify_window_seconds` (emit `episode_end` with duration, mean p over hits, burst count,
  level in dBFS); ACTIVE → IDLE when streak returns to 0. The FSM also exposes `active` every
  tick; the firmware stops vibrating as soon as `active` is false while CONFIRMED.
- all timing is integer tick arithmetic (tick = 0.5 s; streak counted in half-ticks: +2 per
  active tick, −1 per inactive tick) so the Python reference and the C code agree exactly.

Parameter mapping to the cloud `DeviceConfig`: τ ↔ `snore_confidence_threshold`,
confirm_seconds ↔ `snore_confirm_seconds`, verify_window_seconds ↔ `verify_window_seconds`.
hold_seconds, min_bursts and the period range are edge-only constants in `threshold.json`.
Intervention start/stop, cooldown, budget and temperature stay in the firmware state machine.

## 9. Direction of arrival (co-sleeping attribution)

`v5/doa.py` + `doa_gccphat.c`. Inputs: two-channel int16 frames of 512 samples, mic spacing d
(config, default 0.06 m, channel order L,R), c = 343 m/s, max lag = ceil(d/c · 16000) + 1 samples.

- Per frame: GCC-PHAT with bins outside 60–3000 Hz zeroed; lag search within ± max lag with
  parabolic sub-sample interpolation; valid if frame RMS is ≥ 6 dB above the running noise floor
  and the PHAT peak is ≥ 1.5 × the second-highest peak outside ± 1 sample.
- Per episode: lag = median of valid frame lags within the episode; side = left if
  lag > +0.2 · max lag, right if < −0.2 · max lag, else unknown; `doa_conf` = fraction of valid
  frames agreeing with the median side. Positive lag means the signal reaches L first.
- Both-sides rule per night: confident episodes (`doa_conf ≥ 0.7`) on both sides with each side
  holding ≥ 25 % of confident episodes → `both_sides_snoring = true`; the deck says the first
  version records but does not intervene in that case.
- Validation (`doa_sim.py`): pyroomacoustics stereo rooms, d ∈ {0.04, 0.06, 0.08, 0.12} m,
  distance {0.5, 1.0, 1.5} m, SNR {0, 5, 10, 20} dB, azimuth ± 20–70°, 50 trials per cell using
  Kaggle test snore windows; report side accuracy per cell. Target for d = 0.06 m at 1 m and
  SNR ≥ 5 dB: ≥ 95 %. The table is an input to the hardware interface freeze.

## 10. Edge event schema and cloud mapping

`v5/events.py` defines `edge_event_v1`: `ts, snore_p, fsm_state, episode_id, side, lag_ms,
doa_conf, level_dbfs, radar_presence, radar_motion, radar_breath_rate, temp_ok, vib_level,
vib_ms`. Radar fields are placeholders populated by firmware once the MR24BSD1 parser exists;
they are logged so a dataset accumulates.

`to_cloud_event(episode_end)` produces the cloud `EventIn` payload: `event_type =
"snore_detected"`, `snore_duration_sec`, `snore_confidence` = mean p over hits, `in_bed` =
radar_presence, `body_motion_level` = radar_motion, `model_version = "cnn_v5_int8"`, `note` =
JSON with side, lag_ms, doa_conf, n_bursts, level_dbfs. A test validates the payload against the
cloud repo's pydantic `EventIn` when that repo is present on disk (skipped otherwise).

## 11. Export and firmware contract

`export.py` produces under `output/v5/deliverables/`:

- `snore_v5_int8.tflite` (full-integer, int8 in/out, representative set of 500 training windows)
- `model_data.h` (`alignas(16) const unsigned char snore_v5_int8_tflite[]`, length macro)
- `model_meta.h` (INPUT_SCALE, INPUT_ZERO_POINT, OUTPUT_SCALE, OUTPUT_ZERO_POINT, THRESHOLD,
  MODEL_VERSION, FEATURE_SPEC_VERSION, FSM parameters)
- `mel_filterbank.h`, `feature_spec.h`, `golden/` (features, FSM traces, DoA stereo cases)
- `model_card.md` (data volumes, dedup counts, every table from section 7 and 9, int8 parity)

C sources under `esp32_firmware/v5/` (C99, no malloc, no dependencies beyond libm):

- `fft512.c/.h` radix-2 real FFT reference (firmware may substitute esp-dsp; both must pass the
  golden tolerance)
- `snore_features.c/.h`: `sf_compute(const int16_t win[16000], float X[61*30])` and
  `sf_quantize(const float X[], int8_t q[], float scale, int zp)`
- `snore_episode_fsm.c/.h`: `fsm_init(params)`, `fsm_tick(p) -> event`
- `doa_gccphat.c/.h`: `doa_frame(l[512], r[512]) -> lag, valid`, `doa_episode_*` aggregation
- `host_test/Makefile` + `test_features.c`, `test_fsm.c`, `test_doa.c` reading the golden files;
  `make test` exits non-zero on any mismatch. A pytest wrapper runs it when a C compiler exists.

The TFLite Micro invoke glue in the existing `snore_detector_ml.c` is Arthur's; the model card
documents the call sequence (window → `sf_compute` → `sf_quantize` → invoke → dequantise →
`fsm_tick`). The handwritten float path (`snore_infer.c`, 63-frame input) is documented as
deprecated.

## 12. Self-recording kit and fine-tuning

`record_session.py --device <name> --channels 1|2 --minutes N` records 16 kHz WAV chunks of 60 s
into `recordings/<session>/` with a `labels_template.csv` (`chunk, start_s, end_s, label
{snore, breathing, speech, tv, fan, other}, side, posture, distance_m, pillow`). Labelling is
manual.

`finetune.py --session <dir>` slices labelled spans into one-second windows, holds out one
session, fine-tunes the deployed model at learning rate 1e-4 for 10 epochs with the conv1 block
frozen, reports before/after metrics on the held-out session, and exports with
`model_version = "cnn_v5_ft_<date>"`.

## 13. Error handling and degraded modes

- Undecodable or empty audio: skipped, listed in `output/v5/manifest_errors.csv`.
- Teacher unavailable (no network, hub failure): KD and the label audit are disabled, a warning is
  printed, and the model card states that hard labels only were used.
- No C compiler: C parity tests are skipped with an explicit skip reason; never a silent pass.
- TFLite conversion failure: export fails loudly; there is no float fallback artifact.
- Dataset directory missing: dataset-marked tests skip; unit tests use generated audio.

## 14. Testing

Unit tests (no dataset needed, synthetic audio): feature shape and gain invariance, silence →
zeros, golden regeneration determinism, filterbank table round-trip, augmentation SNR accuracy,
RIR normalisation, manifest split invariants on a temp dataset, FSM confirms a periodic hit
pattern within the expected time and rejects a continuous one, FSM stop latency, DoA on simulated
stereo, event payload shape.

Integration tests (`@pytest.mark.dataset`, skipped without data): manifest build over the real
sources with the invariants of section 4.3–4.4, a 1-epoch smoke train, export + parity on the
resulting model, `make test` for the C sources.

## 15. Deferred

Auxiliary sound-class head (breathing/speech/cough/mechanical), SSBPR posture-from-snore head
(request access from the authors first), radar-based in-bed/motion models (need device logs),
learned DoA (the simulator in section 9 can produce training data if ever needed).

## 16. Inputs needed from the team

Mic spacing and channel order once hardware is frozen; confirmation that the firmware uses TFLite
Micro (this spec assumes it); the SSBPR access request; the first device-recorded sessions for
section 12.
