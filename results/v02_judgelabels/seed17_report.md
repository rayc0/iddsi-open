# IDDSI-Open evaluation — seed 17

> Research demonstration for culinary education. Estimates visual similarity to IDDSI descriptors; does not perform official IDDSI tests, assess swallowing, determine suitability for any person, or decide whether food is safe to consume.

## Evaluation scope

- Events: 102
- Confidence threshold calibrated on validation data: 0.2169
- Abstentions are represented as `unclear`; they are counted as misses in macro-F1.
- Weighted kappa and dangerous-direction FNR are computed on non-abstained (covered) events.
- Dangerous direction means predicting a lower numeric IDDSI level than the physical-test label.

## Metrics

| Metric | Estimate | Bootstrap 95% CI |
|---|---:|---:|
| macro_f1_with_abstentions | 0.2843 | [0.1525, 0.3759] |
| weighted_kappa_covered | 0.5053 | [0.1600, 0.7467] |
| dangerous_underclassification_fnr_covered | 0.0392 | [0.0000, 0.1000] |
| dangerous_underclassification_rate_all | 0.0196 | [0.0000, 0.0490] |
| coverage | 0.5000 | [0.4118, 0.5980] |
| abstention_rate | 0.5000 | [0.4020, 0.5882] |
| ece_all | 0.0522 | [0.0405, 0.1434] |
| ece_covered | 0.1024 | [0.0651, 0.2203] |

## Per-level confusion matrix

| Truth ↓ / prediction → | L3 | L4 | L5 | L6 | L7 | unclear |
|---|---:|---:|---:|---:|---:|---:|
| L3 | 2 | 0 | 0 | 0 | 0 | 2 |
| L4 | 0 | 1 | 0 | 0 | 3 | 5 |
| L5 | 0 | 0 | 0 | 0 | 3 | 4 |
| L6 | 0 | 1 | 0 | 0 | 10 | 15 |
| L7 | 0 | 1 | 0 | 0 | 30 | 25 |

## Validation calibration

- Temperature: 1.6245
- Threshold selection target met: `false`
- Selection: gate not met; least under-classification risk among candidates meeting minimum coverage

## Risk–coverage data

The machine-readable `metrics.json` contains the full risk–coverage curve. Threshold selection on a test set is prohibited.

## Interpretation boundary

Results apply only to the declared held-out events and data provenance. Photo appearance cannot measure flow, hardness, adhesiveness, cohesiveness, or swallowing suitability. Confirm with the relevant physical IDDSI test.

This project is not the official IDDSI website and is not endorsed by IDDSI.
