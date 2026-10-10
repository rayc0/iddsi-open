# IDDSI-Open evaluation — seed 101

> Research demonstration for culinary education. Estimates visual similarity to IDDSI descriptors; does not perform official IDDSI tests, assess swallowing, determine suitability for any person, or decide whether food is safe to consume.

## Evaluation scope

- Events: 142
- Confidence threshold calibrated on validation data: 0.2175
- Abstentions are represented as `unclear`; they are counted as misses in macro-F1.
- Weighted kappa and dangerous-direction FNR are computed on non-abstained (covered) events.
- Dangerous direction means predicting a lower numeric IDDSI level than the physical-test label.

## Metrics

| Metric | Estimate | Bootstrap 95% CI |
|---|---:|---:|
| macro_f1_with_abstentions | 0.2528 | [0.1654, 0.3454] |
| weighted_kappa_covered | 0.5895 | [0.3640, 0.7667] |
| dangerous_underclassification_fnr_covered | 0.1471 | [0.0825, 0.2155] |
| dangerous_underclassification_rate_all | 0.1056 | [0.0563, 0.1549] |
| coverage | 0.7183 | [0.6479, 0.7887] |
| abstention_rate | 0.2817 | [0.2113, 0.3521] |
| ece_all | 0.0405 | [0.0314, 0.1203] |
| ece_covered | 0.0579 | [0.0457, 0.1493] |

## Per-level confusion matrix

| Truth ↓ / prediction → | L3 | L4 | L5 | L6 | L7 | unclear |
|---|---:|---:|---:|---:|---:|---:|
| L3 | 2 | 2 | 1 | 0 | 1 | 2 |
| L4 | 1 | 1 | 4 | 0 | 2 | 6 |
| L5 | 0 | 0 | 0 | 1 | 3 | 3 |
| L6 | 0 | 0 | 6 | 2 | 13 | 15 |
| L7 | 1 | 1 | 1 | 5 | 55 | 14 |

## Validation calibration

- Temperature: 1.6245
- Threshold selection target met: `false`
- Selection: gate not met; least under-classification risk among candidates meeting minimum coverage

## Risk–coverage data

The machine-readable `metrics.json` contains the full risk–coverage curve. Threshold selection on a test set is prohibited.

## Interpretation boundary

Results apply only to the declared held-out events and data provenance. Photo appearance cannot measure flow, hardness, adhesiveness, cohesiveness, or swallowing suitability. Confirm with the relevant physical IDDSI test.

This project is not the official IDDSI website and is not endorsed by IDDSI.
