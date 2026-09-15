# IDDSI-Open evaluation — seed 17

> Research demonstration for culinary education. Estimates visual similarity to IDDSI descriptors; does not perform official IDDSI tests, assess swallowing, determine suitability for any person, or decide whether food is safe to consume.

## Evaluation scope

- Events: 109
- Confidence threshold calibrated on validation data: 0.3649
- Abstentions are represented as `unclear`; they are counted as misses in macro-F1.
- Weighted kappa and dangerous-direction FNR are computed on non-abstained (covered) events.
- Dangerous direction means predicting a lower numeric IDDSI level than the physical-test label.

## Metrics

| Metric | Estimate | Bootstrap 95% CI |
|---|---:|---:|
| macro_f1_with_abstentions | 0.3578 | [0.2615, 0.4581] |
| weighted_kappa_covered | 0.4227 | [0.2233, 0.5919] |
| dangerous_underclassification_fnr_covered | 0.3333 | [0.2550, 0.4167] |
| dangerous_underclassification_rate_all | 0.2936 | [0.2202, 0.3670] |
| coverage | 0.8807 | [0.8257, 0.9358] |
| abstention_rate | 0.1193 | [0.0642, 0.1743] |
| ece_all | 0.0906 | [0.0468, 0.1843] |
| ece_covered | 0.0856 | [0.0464, 0.1869] |

## Per-level confusion matrix

| Truth ↓ / prediction → | L3 | L4 | L5 | L6 | L7 | unclear |
|---|---:|---:|---:|---:|---:|---:|
| L3 | 0 | 0 | 0 | 0 | 0 | 0 |
| L4 | 2 | 6 | 13 | 6 | 1 | 2 |
| L5 | 0 | 2 | 13 | 7 | 0 | 3 |
| L6 | 0 | 0 | 13 | 10 | 1 | 6 |
| L7 | 0 | 1 | 6 | 8 | 7 | 2 |

## Validation calibration

- Temperature: 0.9330
- Threshold selection target met: `false`
- Selection: gate not met; least under-classification risk among candidates meeting minimum coverage

## Risk–coverage data

The machine-readable `metrics.json` contains the full risk–coverage curve. Threshold selection on a test set is prohibited.

## Interpretation boundary

Results apply only to the declared held-out events and data provenance. Photo appearance cannot measure flow, hardness, adhesiveness, cohesiveness, or swallowing suitability. Confirm with the relevant physical IDDSI test.

This project is not the official IDDSI website and is not endorsed by IDDSI.
