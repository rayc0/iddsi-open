# IDDSI-Open evaluation — seed 29

> Research demonstration for culinary education. Estimates visual similarity to IDDSI descriptors; does not perform official IDDSI tests, assess swallowing, determine suitability for any person, or decide whether food is safe to consume.

## Evaluation scope

- Events: 142
- Confidence threshold calibrated on validation data: 0.2179
- Abstentions are represented as `unclear`; they are counted as misses in macro-F1.
- Weighted kappa and dangerous-direction FNR are computed on non-abstained (covered) events.
- Dangerous direction means predicting a lower numeric IDDSI level than the physical-test label.

## Metrics

| Metric | Estimate | Bootstrap 95% CI |
|---|---:|---:|
| macro_f1_with_abstentions | 0.3004 | [0.1964, 0.4050] |
| weighted_kappa_covered | 0.6199 | [0.3726, 0.7820] |
| dangerous_underclassification_fnr_covered | 0.0800 | [0.0309, 0.1383] |
| dangerous_underclassification_rate_all | 0.0563 | [0.0211, 0.0986] |
| coverage | 0.7042 | [0.6338, 0.7746] |
| abstention_rate | 0.2958 | [0.2254, 0.3662] |
| ece_all | 0.0604 | [0.0411, 0.1366] |
| ece_covered | 0.0659 | [0.0460, 0.1581] |

## Per-level confusion matrix

| Truth ↓ / prediction → | L3 | L4 | L5 | L6 | L7 | unclear |
|---|---:|---:|---:|---:|---:|---:|
| L3 | 2 | 2 | 0 | 0 | 0 | 4 |
| L4 | 1 | 1 | 2 | 1 | 2 | 7 |
| L5 | 0 | 0 | 1 | 0 | 3 | 3 |
| L6 | 0 | 0 | 1 | 3 | 16 | 16 |
| L7 | 1 | 1 | 1 | 3 | 59 | 12 |

## Validation calibration

- Temperature: 1.6245
- Threshold selection target met: `false`
- Selection: gate not met; least under-classification risk among candidates meeting minimum coverage

## Risk–coverage data

The machine-readable `metrics.json` contains the full risk–coverage curve. Threshold selection on a test set is prohibited.

## Interpretation boundary

Results apply only to the declared held-out events and data provenance. Photo appearance cannot measure flow, hardness, adhesiveness, cohesiveness, or swallowing suitability. Confirm with the relevant physical IDDSI test.

This project is not the official IDDSI website and is not endorsed by IDDSI.
