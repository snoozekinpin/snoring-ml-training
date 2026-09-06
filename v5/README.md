# SnoozMate v5 edge snore pipeline

Spec: `docs/superpowers/specs/2026-09-04-snore-v5-edge-pipeline-design.md`

## Environment (macOS)

    uv venv .venv-mac --python 3.12
    uv pip install --python .venv-mac/bin/python tensorflow tensorflow-hub librosa soundfile soxr scikit-learn scipy pyroomacoustics ai-edge-litert sounddevice pyyaml pytest pydantic "setuptools<81"

Splits: train / val (selection) / calib (threshold) / test (Kaggle, read only by the exporter) / bench (benchmark and DoA material, MS-SNSD beds). Fine-tuned models are candidates only until they pass calibration and the export gates. Release gates in `v5.export` block promotion on int8 parity or operator failures; calibration fails closed on the FPR target.

## Pipeline, in order

    .venv-mac/bin/python -m v5.data.manifest            # decode, dedup, split -> output/v5/manifest.csv
    .venv-mac/bin/python -m v5.teacher                  # YAMNet scores for every non-test window (mandatory); teacher.csv carries each window's md5 and teacher_meta.json the cache fingerprint, so --apply-audit refuses scores made for other audio
    .venv-mac/bin/python -m v5.data.manifest --apply-audit   # drop unverified positives / contaminated negatives; training refuses an unaudited manifest
    .venv-mac/bin/python -m v5.train run --width 1.0 --name hard_w1
    .venv-mac/bin/python -m v5.train run --width 1.0 --kd --name kd_w1
    .venv-mac/bin/python -m v5.train run --width 2.0 --name hard_w2
    .venv-mac/bin/python -m v5.train run --width 2.0 --kd --name kd_w2
    .venv-mac/bin/python -m v5.train select && .venv-mac/bin/python -m v5.train final   # select on val (exact four-run matrix hard_w1/kd_w1/hard_w2/kd_w2, same manifest and validation ids), calibrate on calib
    # `final` binds threshold.json to the manifest; `export model` refuses a model or threshold calibrated on another manifest, and `export card` refuses benchmark/DoA reports whose bindings (model, TFLite, manifest, tau, FSM parameters, DoA sweep parameters) differ from the release
    # the test split is read once per model: output/v5/test_consumption.json records model, manifest, tau and the hash of every file of the evaluation bundle; a re-export of the same model at the same tau reuses the bundle, anything else is refused; `python -m v5.export reset-test-record --reason '...'` is the only sanctioned reset (it writes the reason to output/v5/test_consumption_resets.log, printed in the model card, before removing the record and bundles)
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

Window of 16000 int16 samples every 8000 samples -> `sf_compute` -> `sf_quantize(INPUT_SCALE, INPUT_ZERO_POINT)` -> TFLite Micro invoke on `snore_v5_int8_tflite` -> `p = (out - OUTPUT_ZERO_POINT) * OUTPUT_SCALE` -> `fsm_tick(p, level_dbfs)`. Vibrate only while `state == FSM_CONFIRMED && active`. Stereo frames of 512 samples go to `doa_tracker_frame_i16` during a confirmed episode (`doa_tracker_reset_episode` at each episode start: it clears the votes and keeps the adapted noise floor); `doa_tracker_episode` gives the side for the event note. `model_version` is `MODEL_VERSION` from `model_meta.h`.

## Self-recordings

    .venv-mac/bin/python -m v5.recording.record_session --minutes 10 --channels 2
    # label recordings/<session>/labels.csv, then
    .venv-mac/bin/python -m v5.finetune --sessions recordings/A recordings/B --holdout recordings/C
    .venv-mac/bin/python -m v5.train import-candidate --model output/v5/finetune/<date>/model.keras --name ft_<date>   # registers the candidate as a run (val metrics + hash bindings)
    .venv-mac/bin/python -m v5.train final --run ft_<date>   # verifies, re-calibrates on calib; then `export model` applies the release gates

## Inputs still needed from the team

Mic spacing and channel order (default 0.06 m, L then R), confirmation that firmware uses TFLite Micro, SSBPR dataset access request, first device-recorded sessions.

## Results of the 2026-09-05 build (see `output/v5/deliverables/model_card.md`)

- Deployed run `hard_w1` (25,745 parameters, 34 KB int8 TFLite), threshold 0.806 calibrated to 1 % FPR on 1,013 calibration negatives (95 % upper bound 1.7 %).
- Held-out Kaggle test half (998 windows, evaluated once): AUC 0.998, recall 96 % at 0.4 % FPR; int8 parity within 0.03.
- Cross-collection material never trained on (WHLTalent batch 000002 snores vs its environment batch): AUC 0.80, recall at the deployed threshold 12–19 %. The streaming benchmark on that material confirms 4–29 % of synthetic episodes with 0–0.6 false confirms per hour.
- Read that gap as the real state of generalisation: the only sizeable audible-snore training data is the Kaggle half; the WHLTalent batch-1/2 "snore" recordings turned out to carry no audible snoring for 91 % of windows (teacher audit) and were excluded as positives.
- What moves the number: self-recordings on the actual hub (`v5.recording.record_session` + `v5.finetune`, then `train final` + `export model`), and more verified snore audio (e.g. AudioSet snoring clips) added to the training half.
- DoA simulation: side accuracy at 0.06 m spacing and 0.5–1.0 m is 86–100 % at SNR ≥ 5 dB; see `doa_sim.md` for all spacings.
