# IDDSI-Open evaluation — seed 43

> Research demonstration for culinary education. Estimates visual similarity to IDDSI descriptors; does not perform official IDDSI tests, assess swallowing, determine suitability for any person, or decide whether food is safe to consume.

## Evaluation scope

- Events: 102
- Confidence threshold calibrated on validation data: 0.2231
- Abstentions are represented as `unclear`; they are counted as misses in macro-F1.
- Weighted kappa and dangerous-direction FNR are computed on non-abstained (covered) events.
- Label set for every kappa in this report (H-YARDSTICK): Qwen tier-2 judge labels (v0.2, row-wise stratified split). A weighted kappa is not
  interpretable without the label set it was measured against: the same checkpoints fail the
  wk1>=0.70 gate on the v0.5 consensus and pass it on the v0.7 full-coverage consensus
  (v0.7: 0.838-0.860). Population and coverage are unchanged by this note.
- Dangerous direction means predicting a lower numeric IDDSI level than the physical-test label.

## Metrics

| Metric | Estimate | Bootstrap 95% CI |
|---|---:|---:|
| macro_f1_with_abstentions | 0.3160 | [0.1760, 0.4200] |
| weighted_kappa_covered | 0.4547 | [0.1140, 0.6725] |
| dangerous_underclassification_fnr_covered | 0.1333 | [0.0563, 0.2182] |
| dangerous_underclassification_rate_all | 0.0784 | [0.0294, 0.1275] |
| coverage | 0.5882 | [0.4902, 0.6765] |
| abstention_rate | 0.4118 | [0.3235, 0.5098] |
| ece_all | 0.0499 | [0.0383, 0.1384] |
| ece_covered | 0.0888 | [0.0565, 0.2038] |

## Per-level confusion matrix

| Truth ↓ / prediction → | L3 | L4 | L5 | L6 | L7 | unclear |
|---|---:|---:|---:|---:|---:|---:|
| L3 | 2 | 0 | 0 | 0 | 0 | 2 |
| L4 | 0 | 0 | 0 | 1 | 2 | 6 |
| L5 | 0 | 0 | 1 | 0 | 3 | 3 |
| L6 | 0 | 2 | 1 | 2 | 11 | 10 |
| L7 | 0 | 1 | 1 | 3 | 30 | 21 |

## Validation calibration

- Temperature: 1.5692
- Threshold selection target met: `false`
- Selection: gate not met; least under-classification risk among candidates meeting minimum coverage

## Risk–coverage data

The machine-readable `metrics.json` contains the full risk–coverage curve. Threshold selection on a test set is prohibited.

## Interpretation boundary

Results apply only to the declared held-out events and data provenance. Photo appearance cannot measure flow, hardness, adhesiveness, cohesiveness, or swallowing suitability. Confirm with the relevant physical IDDSI test.

This project is not the official IDDSI website and is not endorsed by IDDSI.
