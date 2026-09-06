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
