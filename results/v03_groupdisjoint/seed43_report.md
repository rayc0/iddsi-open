# IDDSI-Open evaluation — seed 43

> Research demonstration for culinary education. Estimates visual similarity to IDDSI descriptors; does not perform official IDDSI tests, assess swallowing, determine suitability for any person, or decide whether food is safe to consume.

## Evaluation scope

- Events: 142
- Confidence threshold calibrated on validation data: 0.2110
- Abstentions are represented as `unclear`; they are counted as misses in macro-F1.
- Weighted kappa and dangerous-direction FNR are computed on non-abstained (covered) events.
- Label set for every kappa in this report (H-YARDSTICK): Qwen tier-2 judge labels (v0.3, group-disjoint split). A weighted kappa is not
  interpretable without the label set it was measured against: the same checkpoints fail the
  wk1>=0.70 gate on the v0.5 consensus and pass it on the v0.7 full-coverage consensus
  (v0.7: 0.838-0.860). Population and coverage are unchanged by this note.
- Dangerous direction means predicting a lower numeric IDDSI level than the physical-test label.

## Metrics

| Metric | Estimate | Bootstrap 95% CI |
|---|---:|---:|
| macro_f1_with_abstentions | 0.2875 | [0.1823, 0.3886] |
| weighted_kappa_covered | 0.5553 | [0.2862, 0.7486] |
| dangerous_underclassification_fnr_covered | 0.0606 | [0.0198, 0.1087] |
| dangerous_underclassification_rate_all | 0.0423 | [0.0141, 0.0775] |
| coverage | 0.6972 | [0.6268, 0.7678] |
| abstention_rate | 0.3028 | [0.2322, 0.3732] |
| ece_all | 0.0311 | [0.0307, 0.1154] |
| ece_covered | 0.0434 | [0.0415, 0.1454] |

## Per-level confusion matrix

| Truth ↓ / prediction → | L3 | L4 | L5 | L6 | L7 | unclear |
|---|---:|---:|---:|---:|---:|---:|
| L3 | 2 | 2 | 1 | 0 | 1 | 2 |
| L4 | 0 | 2 | 1 | 0 | 2 | 9 |
| L5 | 0 | 0 | 1 | 0 | 3 | 3 |
| L6 | 0 | 0 | 2 | 0 | 20 | 14 |
| L7 | 1 | 1 | 2 | 0 | 58 | 15 |

## Validation calibration

- Temperature: 1.6818
- Threshold selection target met: `false`
- Selection: gate not met; least under-classification risk among candidates meeting minimum coverage

## Risk–coverage data

The machine-readable `metrics.json` contains the full risk–coverage curve. Threshold selection on a test set is prohibited.

## Interpretation boundary

Results apply only to the declared held-out events and data provenance. Photo appearance cannot measure flow, hardness, adhesiveness, cohesiveness, or swallowing suitability. Confirm with the relevant physical IDDSI test.

This project is not the official IDDSI website and is not endorsed by IDDSI.
