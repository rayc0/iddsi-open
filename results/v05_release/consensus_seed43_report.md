# IDDSI-Open evaluation — seed 43

> Research demonstration for culinary education. Estimates visual similarity to IDDSI descriptors; does not perform official IDDSI tests, assess swallowing, determine suitability for any person, or decide whether food is safe to consume.

## Evaluation scope

- Events: 142
- Confidence threshold calibrated on validation data: 0.2084
- Abstentions are represented as `unclear`; they are counted as misses in macro-F1.
- Weighted kappa and dangerous-direction FNR are computed on non-abstained (covered) events.
- Label set for every kappa in this report (H-YARDSTICK): v0.5 consensus labels. A weighted kappa is not
  interpretable without the label set it was measured against: the same checkpoints fail the
  wk1>=0.70 gate on the v0.5 consensus and pass it on the v0.7 full-coverage consensus
  (v0.7: 0.838-0.860). Population and coverage are unchanged by this note.
- Dangerous direction means predicting a lower numeric IDDSI level than the physical-test label.

## Metrics

| Metric | Estimate | Bootstrap 95% CI |
|---|---:|---:|
| macro_f1_with_abstentions | 0.2802 | [0.2130, 0.3444] |
| weighted_kappa_covered | 0.5485 | [0.3657, 0.6995] |
| dangerous_underclassification_fnr_covered | 0.1053 | [0.0531, 0.1624] |
| dangerous_underclassification_rate_all | 0.0845 | [0.0423, 0.1338] |
| coverage | 0.8028 | [0.7394, 0.8662] |
| abstention_rate | 0.1972 | [0.1338, 0.2606] |
| ece_all | 0.0527 | [0.0318, 0.1307] |
| ece_covered | 0.0810 | [0.0441, 0.1762] |

## Per-level confusion matrix

| Truth ↓ / prediction → | L3 | L4 | L5 | L6 | L7 | unclear |
|---|---:|---:|---:|---:|---:|---:|
| L3 | 0 | 5 | 1 | 0 | 2 | 0 |
| L4 | 2 | 5 | 4 | 1 | 5 | 4 |
| L5 | 0 | 0 | 5 | 1 | 8 | 5 |
| L6 | 0 | 1 | 0 | 1 | 8 | 5 |
| L7 | 0 | 1 | 5 | 3 | 56 | 14 |

## Validation calibration

- Temperature: 1.7411
- Threshold selection target met: `false`
- Selection: gate not met; least under-classification risk among candidates meeting minimum coverage

## Risk–coverage data

The machine-readable `metrics.json` contains the full risk–coverage curve. Threshold selection on a test set is prohibited.

## Interpretation boundary

Results apply only to the declared held-out events and data provenance. Photo appearance cannot measure flow, hardness, adhesiveness, cohesiveness, or swallowing suitability. Confirm with the relevant physical IDDSI test.

This project is not the official IDDSI website and is not endorsed by IDDSI.
