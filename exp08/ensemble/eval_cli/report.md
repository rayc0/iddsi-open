# IDDSI-Open held-out evaluation

> Research demonstration for culinary education. Estimates visual similarity to IDDSI descriptors; does not perform official IDDSI tests, assess swallowing, determine suitability for any person, or decide whether food is safe to consume.

## Evaluation scope

- Events: 108
- Confidence threshold calibrated on validation data: 0.5018
- Abstentions are represented as `unclear`; they are counted as misses in macro-F1.
- Weighted kappa and dangerous-direction FNR are computed on non-abstained (covered) events.
- Dangerous direction means predicting a lower numeric IDDSI level than the physical-test label.

## Metrics

| Metric | Estimate | Bootstrap 95% CI |
|---|---:|---:|
| macro_f1_with_abstentions | 0.3248 | [0.2851, 0.3605] |
| weighted_kappa_covered | 0.8440 | [0.7105, 0.9362] |
| dangerous_underclassification_fnr_covered | 0.0460 | [0.0116, 0.0824] |
| dangerous_underclassification_rate_all | 0.0370 | [0.0093, 0.0648] |
| coverage | 0.8056 | [0.7407, 0.8613] |
| abstention_rate | 0.1944 | [0.1387, 0.2593] |
| ece_all | 0.0584 | [0.0432, 0.1474] |
| ece_covered | 0.0345 | [0.0318, 0.1282] |

## Per-level confusion matrix

| Truth ↓ / prediction → | L3 | L4 | L5 | L6 | L7 | unclear |
|---|---:|---:|---:|---:|---:|---:|
| L3 | 8 | 0 | 0 | 0 | 1 | 1 |
| L4 | 1 | 0 | 0 | 0 | 1 | 9 |
| L5 | 3 | 0 | 0 | 0 | 1 | 3 |
| L6 | 0 | 0 | 0 | 0 | 7 | 3 |
| L7 | 0 | 0 | 0 | 0 | 65 | 5 |

## Risk–coverage data

The machine-readable `metrics.json` contains the full risk–coverage curve. Threshold selection on a test set is prohibited.

## Interpretation boundary

Results apply only to the declared held-out events and data provenance. Photo appearance cannot measure flow, hardness, adhesiveness, cohesiveness, or swallowing suitability. Confirm with the relevant physical IDDSI test.

This project is not the official IDDSI website and is not endorsed by IDDSI.
