# IDDSI-Open evaluation — seed 101

> Research demonstration for culinary education. Estimates visual similarity to IDDSI descriptors; does not perform official IDDSI tests, assess swallowing, determine suitability for any person, or decide whether food is safe to consume.

## Evaluation scope

- Events: 142
- Confidence threshold calibrated on validation data: 0.2090
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
| macro_f1_with_abstentions | 0.2937 | [0.2154, 0.3812] |
| weighted_kappa_covered | 0.5904 | [0.4022, 0.7419] |
| dangerous_underclassification_fnr_covered | 0.0943 | [0.0450, 0.1509] |
| dangerous_underclassification_rate_all | 0.0704 | [0.0352, 0.1127] |
| coverage | 0.7465 | [0.6761, 0.8099] |
| abstention_rate | 0.2535 | [0.1901, 0.3239] |
| ece_all | 0.0636 | [0.0332, 0.1486] |
| ece_covered | 0.0984 | [0.0547, 0.1991] |

## Per-level confusion matrix

| Truth ↓ / prediction → | L3 | L4 | L5 | L6 | L7 | unclear |
|---|---:|---:|---:|---:|---:|---:|
| L3 | 1 | 4 | 1 | 0 | 2 | 0 |
| L4 | 3 | 5 | 2 | 0 | 4 | 7 |
| L5 | 0 | 0 | 4 | 0 | 7 | 8 |
| L6 | 1 | 0 | 0 | 0 | 8 | 6 |
| L7 | 0 | 1 | 5 | 0 | 58 | 15 |

## Validation calibration

- Temperature: 1.7411
- Threshold selection target met: `false`
- Selection: gate not met; least under-classification risk among candidates meeting minimum coverage

## Risk–coverage data

The machine-readable `metrics.json` contains the full risk–coverage curve. Threshold selection on a test set is prohibited.

## Interpretation boundary

Results apply only to the declared held-out events and data provenance. Photo appearance cannot measure flow, hardness, adhesiveness, cohesiveness, or swallowing suitability. Confirm with the relevant physical IDDSI test.

This project is not the official IDDSI website and is not endorsed by IDDSI.
