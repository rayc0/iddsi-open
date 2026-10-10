# IDDSI-Open evaluation — seed 71

> Research demonstration for culinary education. Estimates visual similarity to IDDSI descriptors; does not perform official IDDSI tests, assess swallowing, determine suitability for any person, or decide whether food is safe to consume.

## Evaluation scope

- Events: 142
- Confidence threshold calibrated on validation data: 0.2231
- Abstentions are represented as `unclear`; they are counted as misses in macro-F1.
- Weighted kappa and dangerous-direction FNR are computed on non-abstained (covered) events.
- Dangerous direction means predicting a lower numeric IDDSI level than the physical-test label.

## Metrics

| Metric | Estimate | Bootstrap 95% CI |
|---|---:|---:|
| macro_f1_with_abstentions | 0.3159 | [0.2121, 0.4177] |
| weighted_kappa_covered | 0.5842 | [0.3994, 0.7360] |
| dangerous_underclassification_fnr_covered | 0.1296 | [0.0695, 0.1923] |
| dangerous_underclassification_rate_all | 0.0986 | [0.0562, 0.1479] |
| coverage | 0.7606 | [0.6972, 0.8310] |
| abstention_rate | 0.2394 | [0.1690, 0.3028] |
| ece_all | 0.0536 | [0.0356, 0.1293] |
| ece_covered | 0.0683 | [0.0429, 0.1566] |

## Per-level confusion matrix

| Truth ↓ / prediction → | L3 | L4 | L5 | L6 | L7 | unclear |
|---|---:|---:|---:|---:|---:|---:|
| L3 | 3 | 1 | 2 | 0 | 0 | 2 |
| L4 | 1 | 1 | 2 | 3 | 2 | 5 |
| L5 | 0 | 0 | 2 | 1 | 3 | 1 |
| L6 | 0 | 0 | 5 | 2 | 14 | 15 |
| L7 | 2 | 0 | 4 | 2 | 58 | 11 |

## Validation calibration

- Temperature: 1.5692
- Threshold selection target met: `false`
- Selection: gate not met; least under-classification risk among candidates meeting minimum coverage

## Risk–coverage data

The machine-readable `metrics.json` contains the full risk–coverage curve. Threshold selection on a test set is prohibited.

## Interpretation boundary

Results apply only to the declared held-out events and data provenance. Photo appearance cannot measure flow, hardness, adhesiveness, cohesiveness, or swallowing suitability. Confirm with the relevant physical IDDSI test.

This project is not the official IDDSI website and is not endorsed by IDDSI.
