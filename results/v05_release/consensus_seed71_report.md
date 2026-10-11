# IDDSI-Open evaluation — seed 71

> Research demonstration for culinary education. Estimates visual similarity to IDDSI descriptors; does not perform official IDDSI tests, assess swallowing, determine suitability for any person, or decide whether food is safe to consume.

## Evaluation scope

- Events: 142
- Confidence threshold calibrated on validation data: 0.2082
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
| macro_f1_with_abstentions | 0.2416 | [0.1861, 0.2999] |
| weighted_kappa_covered | 0.5995 | [0.4148, 0.7432] |
| dangerous_underclassification_fnr_covered | 0.1827 | [0.1143, 0.2600] |
| dangerous_underclassification_rate_all | 0.1338 | [0.0845, 0.1972] |
| coverage | 0.7324 | [0.6618, 0.8028] |
| abstention_rate | 0.2676 | [0.1972, 0.3382] |
| ece_all | 0.0297 | [0.0250, 0.1116] |
| ece_covered | 0.0581 | [0.0348, 0.1504] |

## Per-level confusion matrix

| Truth ↓ / prediction → | L3 | L4 | L5 | L6 | L7 | unclear |
|---|---:|---:|---:|---:|---:|---:|
| L3 | 0 | 4 | 1 | 0 | 2 | 1 |
| L4 | 3 | 7 | 4 | 1 | 4 | 2 |
| L5 | 0 | 3 | 3 | 2 | 3 | 8 |
| L6 | 0 | 1 | 1 | 0 | 8 | 5 |
| L7 | 0 | 1 | 5 | 5 | 46 | 22 |

## Validation calibration

- Temperature: 1.7411
- Threshold selection target met: `false`
- Selection: gate not met; least under-classification risk among candidates meeting minimum coverage

## Risk–coverage data

The machine-readable `metrics.json` contains the full risk–coverage curve. Threshold selection on a test set is prohibited.

## Interpretation boundary

Results apply only to the declared held-out events and data provenance. Photo appearance cannot measure flow, hardness, adhesiveness, cohesiveness, or swallowing suitability. Confirm with the relevant physical IDDSI test.

This project is not the official IDDSI website and is not endorsed by IDDSI.
