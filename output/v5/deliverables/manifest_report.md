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
