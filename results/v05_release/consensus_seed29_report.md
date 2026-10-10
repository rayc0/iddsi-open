# IDDSI-Open evaluation — seed 29

> Research demonstration for culinary education. Estimates visual similarity to IDDSI descriptors; does not perform official IDDSI tests, assess swallowing, determine suitability for any person, or decide whether food is safe to consume.

## Evaluation scope

- Events: 142
- Confidence threshold calibrated on validation data: 0.1912
- Abstentions are represented as `unclear`; they are counted as misses in macro-F1.
- Weighted kappa and dangerous-direction FNR are computed on non-abstained (covered) events.
- Dangerous direction means predicting a lower numeric IDDSI level than the physical-test label.

## Metrics

| Metric | Estimate | Bootstrap 95% CI |
|---|---:|---:|
| macro_f1_with_abstentions | 0.2961 | [0.2160, 0.3889] |
| weighted_kappa_covered | 0.5404 | [0.3508, 0.7022] |
| dangerous_underclassification_fnr_covered | 0.0893 | [0.0435, 0.1466] |
| dangerous_underclassification_rate_all | 0.0704 | [0.0352, 0.1197] |
| coverage | 0.7887 | [0.7254, 0.8521] |
| abstention_rate | 0.2113 | [0.1479, 0.2746] |
| ece_all | 0.0929 | [0.0688, 0.1684] |
| ece_covered | 0.1055 | [0.0816, 0.1961] |

## Per-level confusion matrix

| Truth ↓ / prediction → | L3 | L4 | L5 | L6 | L7 | unclear |
|---|---:|---:|---:|---:|---:|---:|
| L3 | 1 | 4 | 1 | 0 | 2 | 0 |
| L4 | 3 | 5 | 1 | 0 | 6 | 6 |
| L5 | 0 | 1 | 4 | 0 | 8 | 6 |
| L6 | 1 | 0 | 0 | 0 | 9 | 5 |
| L7 | 1 | 1 | 3 | 0 | 61 | 13 |

## Validation calibration

- Temperature: 1.9319
- Threshold selection target met: `false`
- Selection: gate not met; least under-classification risk among candidates meeting minimum coverage

## Risk–coverage data

The machine-readable `metrics.json` contains the full risk–coverage curve. Threshold selection on a test set is prohibited.

## Interpretation boundary

Results apply only to the declared held-out events and data provenance. Photo appearance cannot measure flow, hardness, adhesiveness, cohesiveness, or swallowing suitability. Confirm with the relevant physical IDDSI test.

This project is not the official IDDSI website and is not endorsed by IDDSI.
