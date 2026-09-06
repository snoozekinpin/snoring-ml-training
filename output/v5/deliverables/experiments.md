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
