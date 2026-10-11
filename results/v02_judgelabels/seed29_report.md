# IDDSI-Open evaluation — seed 29

> Research demonstration for culinary education. Estimates visual similarity to IDDSI descriptors; does not perform official IDDSI tests, assess swallowing, determine suitability for any person, or decide whether food is safe to consume.

## Evaluation scope

- Events: 102
- Confidence threshold calibrated on validation data: 0.2168
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
| macro_f1_with_abstentions | 0.2956 | [0.1552, 0.3872] |
| weighted_kappa_covered | 0.5149 | [0.2623, 0.7213] |
| dangerous_underclassification_fnr_covered | 0.0952 | [0.0317, 0.1765] |
| dangerous_underclassification_rate_all | 0.0588 | [0.0196, 0.1078] |
| coverage | 0.6176 | [0.5294, 0.7059] |
| abstention_rate | 0.3824 | [0.2941, 0.4706] |
| ece_all | 0.0631 | [0.0413, 0.1532] |
| ece_covered | 0.0966 | [0.0601, 0.2130] |

## Per-level confusion matrix

| Truth ↓ / prediction → | L3 | L4 | L5 | L6 | L7 | unclear |
|---|---:|---:|---:|---:|---:|---:|
| L3 | 2 | 0 | 1 | 0 | 0 | 1 |
| L4 | 0 | 1 | 1 | 0 | 3 | 4 |
| L5 | 0 | 0 | 0 | 0 | 3 | 4 |
| L6 | 0 | 0 | 1 | 0 | 12 | 13 |
| L7 | 0 | 1 | 1 | 3 | 34 | 17 |

## Validation calibration

- Temperature: 1.6245
- Threshold selection target met: `true`
- Selection: highest coverage satisfying the configured validation under-classification gate

## Risk–coverage data

The machine-readable `metrics.json` contains the full risk–coverage curve. Threshold selection on a test set is prohibited.

## Interpretation boundary

Results apply only to the declared held-out events and data provenance. Photo appearance cannot measure flow, hardness, adhesiveness, cohesiveness, or swallowing suitability. Confirm with the relevant physical IDDSI test.

This project is not the official IDDSI website and is not endorsed by IDDSI.
