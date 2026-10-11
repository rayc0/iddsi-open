# IDDSI-Open evaluation — seed 17

> Research demonstration for culinary education. Estimates visual similarity to IDDSI descriptors; does not perform official IDDSI tests, assess swallowing, determine suitability for any person, or decide whether food is safe to consume.

## Evaluation scope

- Events: 142
- Confidence threshold calibrated on validation data: 0.2012
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
| macro_f1_with_abstentions | 0.3879 | [0.2930, 0.4769] |
| weighted_kappa_covered | 0.5516 | [0.3546, 0.7077] |
| dangerous_underclassification_fnr_covered | 0.1583 | [0.0948, 0.2232] |
| dangerous_underclassification_rate_all | 0.1338 | [0.0775, 0.1901] |
| coverage | 0.8451 | [0.7817, 0.9014] |
| abstention_rate | 0.1549 | [0.0986, 0.2183] |
| ece_all | 0.1180 | [0.0806, 0.1976] |
| ece_covered | 0.1084 | [0.0694, 0.1998] |

## Per-level confusion matrix

| Truth ↓ / prediction → | L3 | L4 | L5 | L6 | L7 | unclear |
|---|---:|---:|---:|---:|---:|---:|
| L3 | 2 | 3 | 0 | 0 | 2 | 1 |
| L4 | 4 | 5 | 2 | 1 | 6 | 3 |
| L5 | 0 | 1 | 5 | 0 | 5 | 8 |
| L6 | 1 | 0 | 1 | 4 | 8 | 1 |
| L7 | 1 | 1 | 5 | 5 | 58 | 9 |

## Validation calibration

- Temperature: 1.8025
- Threshold selection target met: `false`
- Selection: gate not met; least under-classification risk among candidates meeting minimum coverage

## Risk–coverage data

The machine-readable `metrics.json` contains the full risk–coverage curve. Threshold selection on a test set is prohibited.

## Interpretation boundary

Results apply only to the declared held-out events and data provenance. Photo appearance cannot measure flow, hardness, adhesiveness, cohesiveness, or swallowing suitability. Confirm with the relevant physical IDDSI test.

This project is not the official IDDSI website and is not endorsed by IDDSI.
