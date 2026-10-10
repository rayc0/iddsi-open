# IDDSI-Open evaluation — seed 17

> Research demonstration for culinary education. Estimates visual similarity to IDDSI descriptors; does not perform official IDDSI tests, assess swallowing, determine suitability for any person, or decide whether food is safe to consume.

## Evaluation scope

- Events: 142
- Confidence threshold calibrated on validation data: 0.2180
- Abstentions are represented as `unclear`; they are counted as misses in macro-F1.
- Weighted kappa and dangerous-direction FNR are computed on non-abstained (covered) events.
- Dangerous direction means predicting a lower numeric IDDSI level than the physical-test label.

## Metrics

| Metric | Estimate | Bootstrap 95% CI |
|---|---:|---:|
| macro_f1_with_abstentions | 0.2846 | [0.1861, 0.3868] |
| weighted_kappa_covered | 0.5958 | [0.3517, 0.7698] |
| dangerous_underclassification_fnr_covered | 0.0909 | [0.0341, 0.1495] |
| dangerous_underclassification_rate_all | 0.0563 | [0.0211, 0.0915] |
| coverage | 0.6197 | [0.5423, 0.6972] |
| abstention_rate | 0.3803 | [0.3028, 0.4577] |
| ece_all | 0.0395 | [0.0301, 0.1162] |
| ece_covered | 0.0565 | [0.0399, 0.1585] |

## Per-level confusion matrix

| Truth ↓ / prediction → | L3 | L4 | L5 | L6 | L7 | unclear |
|---|---:|---:|---:|---:|---:|---:|
| L3 | 2 | 2 | 0 | 0 | 0 | 4 |
| L4 | 0 | 2 | 3 | 1 | 2 | 6 |
| L5 | 0 | 0 | 1 | 0 | 3 | 3 |
| L6 | 0 | 0 | 2 | 1 | 14 | 19 |
| L7 | 2 | 0 | 1 | 3 | 49 | 22 |

## Validation calibration

- Temperature: 1.6245
- Threshold selection target met: `false`
- Selection: gate not met; least under-classification risk among candidates meeting minimum coverage

## Risk–coverage data

The machine-readable `metrics.json` contains the full risk–coverage curve. Threshold selection on a test set is prohibited.

## Interpretation boundary

Results apply only to the declared held-out events and data provenance. Photo appearance cannot measure flow, hardness, adhesiveness, cohesiveness, or swallowing suitability. Confirm with the relevant physical IDDSI test.

This project is not the official IDDSI website and is not endorsed by IDDSI.
