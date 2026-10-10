# IDDSI-Open evaluation — seed 101

> Research demonstration for culinary education. Estimates visual similarity to IDDSI descriptors; does not perform official IDDSI tests, assess swallowing, determine suitability for any person, or decide whether food is safe to consume.

## Evaluation scope

- Events: 102
- Confidence threshold calibrated on validation data: 0.2164
- Abstentions are represented as `unclear`; they are counted as misses in macro-F1.
- Weighted kappa and dangerous-direction FNR are computed on non-abstained (covered) events.
- Dangerous direction means predicting a lower numeric IDDSI level than the physical-test label.

## Metrics

| Metric | Estimate | Bootstrap 95% CI |
|---|---:|---:|
| macro_f1_with_abstentions | 0.3199 | [0.1644, 0.4232] |
| weighted_kappa_covered | 0.4797 | [0.2231, 0.7038] |
| dangerous_underclassification_fnr_covered | 0.0952 | [0.0317, 0.1725] |
| dangerous_underclassification_rate_all | 0.0588 | [0.0196, 0.1078] |
| coverage | 0.6176 | [0.5294, 0.7059] |
| abstention_rate | 0.3824 | [0.2941, 0.4706] |
| ece_all | 0.0742 | [0.0489, 0.1665] |
| ece_covered | 0.1207 | [0.0726, 0.2359] |

## Per-level confusion matrix

| Truth ↓ / prediction → | L3 | L4 | L5 | L6 | L7 | unclear |
|---|---:|---:|---:|---:|---:|---:|
| L3 | 2 | 0 | 0 | 1 | 0 | 1 |
| L4 | 0 | 0 | 2 | 1 | 3 | 3 |
| L5 | 0 | 0 | 1 | 0 | 3 | 3 |
| L6 | 0 | 1 | 0 | 2 | 10 | 13 |
| L7 | 0 | 1 | 0 | 4 | 32 | 19 |

## Validation calibration

- Temperature: 1.6245
- Threshold selection target met: `false`
- Selection: gate not met; least under-classification risk among candidates meeting minimum coverage

## Risk–coverage data

The machine-readable `metrics.json` contains the full risk–coverage curve. Threshold selection on a test set is prohibited.

## Interpretation boundary

Results apply only to the declared held-out events and data provenance. Photo appearance cannot measure flow, hardness, adhesiveness, cohesiveness, or swallowing suitability. Confirm with the relevant physical IDDSI test.

This project is not the official IDDSI website and is not endorsed by IDDSI.
