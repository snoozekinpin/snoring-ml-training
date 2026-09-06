# SnoozMate v5 snore model card (2026-09-05)

test split consumed once by model c0d3ac44ba0d on manifest e7dd580f6a0e at 2026-09-05T20:00:38 (tau 0.8060; 4 bundle files hash-bound)

test-consumption record history:
- 2026-09-05T18:56:57-07:00 test-consumption record reset: export code, RIR normalisation and DoA validity were fixed after the first evaluation; the first model is superseded by the retrained one. Recorded in model_card.md.
- 2026-09-05T19:20:53 record migrated to the hash-bound format: tau 0.8060 (the threshold in force at the evaluation, see release_log.txt) and the SHA-256 of every bundle file were added; no test evaluation was performed
- 2026-09-05T20:00:35 reset: threshold rule changed to the smallest float32 strictly above the permitted calibration negative (firmware compares float32 values); the deployed model is unchanged and is re-evaluated once at the re-calibrated tau

model_version: `cnn_v5_int8`; run `hard_w1`; feature spec v1; tau = 0.8060; exported 2026-09-05T20:00:42

int8 TFLite: 34336 bytes (sha256 754c8b0ad436); params 25745; input scale 0.007843 zero point -1; ops ['CONV_2D', 'FULLY_CONNECTED', 'LOGISTIC', 'MAX_POOL_2D', 'MEAN']

arena estimate: 60608 bytes activations (largest tensor 29280); int8 activations; two largest consecutive tensors plus scratch; measure the real arena on the device

calibration (calib split): n_neg 1013, FPR 0.0099 (95% upper bound 0.0167), recall 0.529

## Deployment call sequence

```
window (int16[16000], hop 8000) -> sf_compute -> sf_quantize(INPUT_SCALE, INPUT_ZERO_POINT)
  -> TFLM invoke -> p = (out - OUTPUT_ZERO_POINT) * OUTPUT_SCALE -> fsm_tick(p, level_dbfs)
  -> vibrate only while fsm.state == FSM_CONFIRMED and fsm.active
```

The handwritten float path (snore_infer.c, 63-frame input) is deprecated. Desktop C tests establish numerical parity, not ESP32-S3 deployability;
the firmware build must still verify the TFLM operator resolver, the real tensor arena size and flash use.

### Test split, float model (evaluated once at export)

| n | pos | AUC | R@FPR2% | tau | precision | recall | FNR | FPR |
|---|---|---|---|---|---|---|---|---|
| 998 | 514 | 0.9978 | 0.984 | 0.806 | 0.996 | 0.961 | 0.039 | 0.004 |

### Test split, int8 model (deployed path)

| n | pos | AUC | R@FPR2% | tau | precision | recall | FNR | FPR |
|---|---|---|---|---|---|---|---|---|
| 998 | 514 | 0.9977 | 0.984 | 0.806 | 0.996 | 0.963 | 0.037 | 0.004 |

### int8 parity gate

delta AUC 0.0001; decision agreement 0.9990; max |dp| 0.0279; limits {'delta_auc': 0.005, 'agreement': 0.99, 'max_abs_diff': 0.05}; passed: True

### Robustness (test split, float model)

| condition | AUC | R@FPR2% | recall@tau | FPR@tau |
|---|---|---|---|---|
| clean | 0.9978 | 0.984 | 0.961 | 0.004 |
| SNR 20 dB | 0.9993 | 0.992 | 0.977 | 0.004 |
| SNR 10 dB | 0.9976 | 0.963 | 0.920 | 0.006 |
| SNR 5 dB | 0.9931 | 0.918 | 0.792 | 0.002 |
| SNR 0 dB | 0.9663 | 0.700 | 0.554 | 0.002 |
| RIR 0.5 m | 0.9918 | 0.918 | 0.724 | 0.000 |
| RIR 1.0 m | 0.9772 | 0.733 | 0.706 | 0.019 |
| RIR 1.5 m | 0.9719 | 0.615 | 0.696 | 0.025 |

## cross_collection.md

# Deployed model on validation and bench material (never trained on)

| split / source | n | pos | AUC | recall@tau | FPR@tau |
|---|---|---|---|---|---|
| val all | 1739 | 126 | 0.907 | 0.262 | 0.0012 |
| val whl_s (batch 000002) vs whl_e | 678 | 107 | 0.805 | 0.187 | 0.0018 |
| val esc50 | 786 | 19 | 0.961 | 0.684 | 0.0013 |
| val mssnsd | 275 | 0 | - | - | 0.0000 |
| bench all | 1045 | 108 | 0.863 | 0.120 | 0.0043 |
| bench whl_s (batch 000002) vs whl_e | 672 | 108 | 0.793 | 0.120 | 0.0053 |
| bench mssnsd | 373 | 0 | - | - | 0.0027 |

The validation and bench positives are WHLTalent batch 000002 (a different collection than the Kaggle training half) plus ESC-50 snoring; recall at the calibrated threshold on that batch explains the episode detection rate in the streaming benchmark.


## manifest_report.md

# Manifest report

windows decoded: 12874; kept in partitions: 10466; exact duplicates dropped: 14; exact-duplicate waveforms with conflicting labels: 0
near-duplicate clusters with more than one member: 766

Kaggle (adrianagaler + jibran, near-duplicate twins) is split by near-duplicate cluster into a training half and an immutable test half; the test half is read only by the exporter. WHLTalent batches 000000/100000/100001 are training material for negatives only (the teacher hears snoring in under 10 % of their 'snore' windows, see the audit section); batches 000002/100002 are the cross-collection validation and bench material (recording-disjoint between val and bench). ESC-50 uses its official folds. No subject metadata exists for any source, so splits are collection/batch/recording-disjoint, not proven subject-disjoint.

exact duplicates across sources (kept ~ discarded):
- mssnsd ~ mssnsd: 14

near-duplicate clusters spanning two sources:
- kaggle_adria ~ kaggle_jibran: 745

| source | label | train | val | calib | test | bench | sanity | drop |
|---|---|---|---|---|---|---|---|---|
| whl_s | 1 | 184 | 107 | 0 | 0 | 108 | 0 | 2103 |
| whl_e | 0 | 1343 | 571 | 0 | 0 | 564 | 0 | 14 |
| esc50 | 1 | 44 | 19 | 17 | 0 | 0 | 0 | 35 |
| esc50 | 0 | 2268 | 767 | 751 | 0 | 0 | 0 | 26 |
| mssnsd | 0 | 996 | 275 | 262 | 0 | 373 | 0 | 61 |
| kaggle_adria | 1 | 159 | 0 | 0 | 257 | 0 | 0 | 84 |
| kaggle_adria | 0 | 252 | 0 | 0 | 238 | 0 | 0 | 0 |
| kaggle_jibran | 1 | 164 | 0 | 0 | 257 | 0 | 0 | 79 |
| kaggle_jibran | 0 | 244 | 0 | 0 | 246 | 0 | 0 | 0 |
| wild | 1 | 0 | 0 | 0 | 0 | 0 | 6 | 0 |

dropped (reason, source): {('exact_dup', 'mssnsd'): 14, ('partition', 'whl_e'): 4, ('partition', 'esc50'): 6, ('partition', 'mssnsd'): 47, ('unverified_positive', 'whl_s'): 259, ('snore_in_negative', 'whl_e'): 9, ('unverified_positive', 'esc50'): 35, ('snore_in_negative', 'esc50'): 20, ('unverified_positive', 'kaggle_adria'): 84, ('unverified_positive', 'kaggle_jibran'): 79}

## Teacher audit (YAMNet, applied to train/val/calib/bench only)

positives dropped as unverified (P(snoring+snort) < 0.1) and negatives dropped as contaminated (P > 0.5), by (reason, source, batch):
- ('unverified_positive', 'whl_s', '000000'): 1844
- ('unverified_positive', 'whl_s', '000002'): 259
- ('unverified_positive', 'kaggle_adria', ''): 84
- ('unverified_positive', 'kaggle_jibran', ''): 79
- ('unverified_positive', 'esc50', ''): 35
- ('snore_in_negative', 'esc50', ''): 20
- ('snore_in_negative', 'whl_e', '100002'): 9
- ('snore_in_negative', 'whl_e', '100000'): 1


## experiments.md

# v5 training runs (selection on the validation split; the test split is untouched until export)

| run | kd | width | params | epochs | val AUC | val R@FPR2% |
|---|---|---|---|---|---|---|
| hard_w1 | False | 1.0 | 25745 | 27 | 0.9073 | 0.397 |
| hard_w2 | False | 2.0 | 101665 | 16 | 0.8952 | 0.452 |
| kd_w1 | True | 1.0 | 25745 | 22 | 0.9011 | 0.444 |
| kd_w2 | True | 2.0 | 101665 | 29 | 0.8861 | 0.429 |

selected: `hard_w1`

deployed/threshold.json (calibrated on the calib split):
```json
{
  "tau": 0.8060498363150644,
  "model_version": "cnn_v5_int8",
  "max_fpr": 0.01,
  "run": "hard_w1",
  "calib": {
    "tau": 0.8060498363150644,
    "n_neg": 1013,
    "n_pos": 17,
    "fp": 10,
    "fpr": 0.009871668311944718,
    "fpr_upper95": 0.016687001783597985,
    "recall": 0.5294117647058824,
    "max_fpr": 0.01
  },
  "fsm": {
    "tick_ms": 500,
    "hold_s": 6,
    "confirm_s": 10,
    "verify_s": 15,
    "min_bursts": 3,
    "period_min_s": 1.5,
    "period_max_s": 7.0
  },
  "model_sha256": "c0d3ac44ba0d8ae4387734a25c445248a417ec6835a63e2f739b573550b3546f"
}
```


## benchmark_nights.md

# Streaming benchmark (synthetic nights from the bench partition)

FSM params: `{"tau": 0.8060498833656311, "tick_ms": 500, "hold_ticks": 12, "confirm_ticks": 20, "verify_ticks": 30, "min_bursts": 3, "period_min_ticks": 3, "period_max_ticks": 14}`

## predictor: float

| SNR dB | nights | detection | confirm latency s | false confirms / h | stop latency s |
|---|---|---|---|---|---|
| 0 | 5 | 0.044 | 65.4 | 0.60 | 5.3 |
| 5 | 5 | 0.057 | 14.8 | 0.00 | 0.0 |
| 10 | 5 | 0.158 | 43.1 | 0.00 | 0.0 |
| 20 | 5 | 0.271 | 50.6 | 0.20 | 0.0 |

## predictor: int8

| SNR dB | nights | detection | confirm latency s | false confirms / h | stop latency s |
|---|---|---|---|---|---|
| 0 | 5 | 0.044 | 52.6 | 0.60 | 2.3 |
| 5 | 5 | 0.057 | 14.8 | 0.00 | 0.0 |
| 10 | 5 | 0.183 | 41.8 | 0.00 | 0.0 |
| 20 | 5 | 0.293 | 50.1 | 0.20 | 0.2 |

tick decision agreement float~int8: 0.9996; episode starts {'float': 26, 'int8': 28}


## doa_sim.md

# DoA simulation sweep

| spacing m | distance m | SNR dB | side accuracy | unknown rate |
|---|---|---|---|---|
| 0.04 | 0.5 | 0 | 0.820 | 0.120 |
| 0.04 | 0.5 | 5 | 0.820 | 0.180 |
| 0.04 | 0.5 | 10 | 0.940 | 0.060 |
| 0.04 | 0.5 | 20 | 0.960 | 0.040 |
| 0.04 | 1.0 | 0 | 0.800 | 0.180 |
| 0.04 | 1.0 | 5 | 0.880 | 0.120 |
| 0.04 | 1.0 | 10 | 0.960 | 0.040 |
| 0.04 | 1.0 | 20 | 0.900 | 0.100 |
| 0.04 | 1.5 | 0 | 0.580 | 0.420 |
| 0.04 | 1.5 | 5 | 0.620 | 0.340 |
| 0.04 | 1.5 | 10 | 0.680 | 0.320 |
| 0.04 | 1.5 | 20 | 0.820 | 0.180 |
| 0.06 | 0.5 | 0 | 0.860 | 0.120 |
| 0.06 | 0.5 | 5 | 1.000 | 0.000 |
| 0.06 | 0.5 | 10 | 1.000 | 0.000 |
| 0.06 | 0.5 | 20 | 0.980 | 0.020 |
| 0.06 | 1.0 | 0 | 0.820 | 0.160 |
| 0.06 | 1.0 | 5 | 0.960 | 0.040 |
| 0.06 | 1.0 | 10 | 0.940 | 0.060 |
| 0.06 | 1.0 | 20 | 1.000 | 0.000 |
| 0.06 | 1.5 | 0 | 0.660 | 0.320 |
| 0.06 | 1.5 | 5 | 0.900 | 0.100 |
| 0.06 | 1.5 | 10 | 0.980 | 0.020 |
| 0.06 | 1.5 | 20 | 0.980 | 0.020 |
| 0.08 | 0.5 | 0 | 0.900 | 0.100 |
| 0.08 | 0.5 | 5 | 0.980 | 0.020 |
| 0.08 | 0.5 | 10 | 0.980 | 0.020 |
| 0.08 | 0.5 | 20 | 1.000 | 0.000 |
| 0.08 | 1.0 | 0 | 0.940 | 0.040 |
| 0.08 | 1.0 | 5 | 0.980 | 0.020 |
| 0.08 | 1.0 | 10 | 0.980 | 0.020 |
| 0.08 | 1.0 | 20 | 1.000 | 0.000 |
| 0.08 | 1.5 | 0 | 0.760 | 0.220 |
| 0.08 | 1.5 | 5 | 0.940 | 0.060 |
| 0.08 | 1.5 | 10 | 1.000 | 0.000 |
| 0.08 | 1.5 | 20 | 1.000 | 0.000 |
| 0.12 | 0.5 | 0 | 0.880 | 0.120 |
| 0.12 | 0.5 | 5 | 1.000 | 0.000 |
| 0.12 | 0.5 | 10 | 1.000 | 0.000 |
| 0.12 | 0.5 | 20 | 1.000 | 0.000 |
| 0.12 | 1.0 | 0 | 1.000 | 0.000 |
| 0.12 | 1.0 | 5 | 1.000 | 0.000 |
| 0.12 | 1.0 | 10 | 1.000 | 0.000 |
| 0.12 | 1.0 | 20 | 1.000 | 0.000 |
| 0.12 | 1.5 | 0 | 0.880 | 0.120 |
| 0.12 | 1.5 | 5 | 1.000 | 0.000 |
| 0.12 | 1.5 | 10 | 1.000 | 0.000 |
| 0.12 | 1.5 | 20 | 1.000 | 0.000 |

