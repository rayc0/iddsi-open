# H-YARDSTICK re-derivation

Independent re-derivation of the claim in `hf/HAZARD_LOG.md` section
H-YARDSTICK: *against the v0.5 consensus all three model versions fail wk1
(0/5 seeds each); against the v0.7 full-coverage consensus all three pass*.

## Verdict

**The published claim is CONFIRMED, and could not be refuted.** Both halves
reproduce from the per-event prediction files, and the quoted per-version
ranges match the published ones to the digit:

| yardstick | v0.4 wk1 range | v0.5 wk1 range | v0.7 wk1 range | published ranges |
| --- | --- | --- | --- | --- |
| v0.5 consensus | 0.5404 - 0.5995 (0/5) | 0.5371 - 0.6137 (0/5) | 0.4913 - 0.5547 (0/5) | all fail |
| v0.7 full-coverage consensus | 0.7364 - 0.8077 (5/5) | 0.7036 - 0.7953 (5/5) | 0.8375 - 0.8604 (5/5) | v0.4 0.736-0.808, v0.5 0.704-0.795, v0.7 0.838-0.860 |

Denominators: 5 seeds each (17, 29, 43, 71, 101). v0.4 is scored on its own
142-event held-out split for the v0.5 column (its published denominator) and
on the 108 events common to all three models for the v0.7 column; the
108-event-common rows below show the verdict is unchanged when all three
models share one denominator. Every number is quadratic weighted kappa on
the covered rows only -- the gate metric wk1.

One material discrepancy was found, and it is in the evidence file rather
than the claim: see 'What `bias_matrix.json` actually contains' below.

Quadratic weighted kappa recomputed from `test_predictions.npz` per run,
CORAL rank decode re-implemented, QWK implemented from its definition
(no sklearn, no `trainer/eval/metrics.py`).

## Control: does this script reproduce each run's own published wk1?

| version | seed | n | n_covered | published wk1 | recomputed wk1 | abs diff |
| --- | --- | --- | --- | --- | --- | --- |
| v0.4 | 17 | 142 | 120 | 0.551598 | 0.551598 | 2.22e-16 |
| v0.4 | 29 | 142 | 112 | 0.540356 | 0.540356 | 0.00e+00 |
| v0.4 | 43 | 142 | 114 | 0.548510 | 0.548510 | 0.00e+00 |
| v0.4 | 71 | 142 | 104 | 0.599469 | 0.599469 | 0.00e+00 |
| v0.4 | 101 | 142 | 106 | 0.590376 | 0.590376 | 0.00e+00 |
| v0.5 | 17 | 108 | 83 | 0.567338 | 0.567338 | 1.11e-16 |
| v0.5 | 29 | 108 | 84 | 0.569775 | 0.569775 | 0.00e+00 |
| v0.5 | 43 | 108 | 77 | 0.613718 | 0.613718 | 0.00e+00 |
| v0.5 | 71 | 108 | 75 | 0.553462 | 0.553462 | 0.00e+00 |
| v0.5 | 101 | 108 | 87 | 0.537097 | 0.537097 | 2.22e-16 |
| v0.7 | 17 | 108 | 89 | 0.849789 | 0.849789 | 0.00e+00 |
| v0.7 | 29 | 108 | 86 | 0.850360 | 0.850360 | 1.11e-16 |
| v0.7 | 43 | 108 | 92 | 0.850721 | 0.850721 | 0.00e+00 |
| v0.7 | 71 | 108 | 85 | 0.837545 | 0.837545 | 0.00e+00 |
| v0.7 | 101 | 108 | 85 | 0.860407 | 0.860407 | 0.00e+00 |

max abs diff across 15 runs = 2.220e-16

## Yardstick: v0.5 consensus

| version | seed | n events | n covered | coverage | wk1 (gate) | passes wk1 >= 0.70 |
| --- | --- | --- | --- | --- | --- | --- |
| v0.4 | 17 | 142 | 120 | 0.845 | 0.5516 | FAIL |
| v0.4 | 29 | 142 | 112 | 0.789 | 0.5404 | FAIL |
| v0.4 | 43 | 142 | 114 | 0.803 | 0.5485 | FAIL |
| v0.4 | 71 | 142 | 104 | 0.732 | 0.5995 | FAIL |
| v0.4 | 101 | 142 | 106 | 0.746 | 0.5904 | FAIL |
| **v0.4 range** | | | | | **0.5404 - 0.5995** | **0/5 seeds** |
| v0.5 | 17 | 108 | 83 | 0.769 | 0.5673 | FAIL |
| v0.5 | 29 | 108 | 84 | 0.778 | 0.5698 | FAIL |
| v0.5 | 43 | 108 | 77 | 0.713 | 0.6137 | FAIL |
| v0.5 | 71 | 108 | 75 | 0.694 | 0.5535 | FAIL |
| v0.5 | 101 | 108 | 87 | 0.806 | 0.5371 | FAIL |
| **v0.5 range** | | | | | **0.5371 - 0.6137** | **0/5 seeds** |
| v0.7 | 17 | 108 | 89 | 0.824 | 0.5389 | FAIL |
| v0.7 | 29 | 108 | 86 | 0.796 | 0.4969 | FAIL |
| v0.7 | 43 | 108 | 92 | 0.852 | 0.5077 | FAIL |
| v0.7 | 71 | 108 | 85 | 0.787 | 0.4913 | FAIL |
| v0.7 | 101 | 108 | 85 | 0.787 | 0.5547 | FAIL |
| **v0.7 range** | | | | | **0.4913 - 0.5547** | **0/5 seeds** |

## Yardstick: v0.7 full-coverage consensus

| version | seed | n events | n covered | coverage | wk1 (gate) | passes wk1 >= 0.70 |
| --- | --- | --- | --- | --- | --- | --- |
| v0.4 | 17 | 108 | 93 | 0.861 | 0.7364 | PASS |
| v0.4 | 29 | 108 | 86 | 0.796 | 0.7371 | PASS |
| v0.4 | 43 | 108 | 90 | 0.833 | 0.7736 | PASS |
| v0.4 | 71 | 108 | 84 | 0.778 | 0.7366 | PASS |
| v0.4 | 101 | 108 | 84 | 0.778 | 0.8077 | PASS |
| **v0.4 range** | | | | | **0.7364 - 0.8077** | **5/5 seeds** |
| v0.5 | 17 | 108 | 83 | 0.769 | 0.7588 | PASS |
| v0.5 | 29 | 108 | 84 | 0.778 | 0.7953 | PASS |
| v0.5 | 43 | 108 | 77 | 0.713 | 0.7171 | PASS |
| v0.5 | 71 | 108 | 75 | 0.694 | 0.7169 | PASS |
| v0.5 | 101 | 108 | 87 | 0.806 | 0.7036 | PASS |
| **v0.5 range** | | | | | **0.7036 - 0.7953** | **5/5 seeds** |
| v0.7 | 17 | 108 | 89 | 0.824 | 0.8498 | PASS |
| v0.7 | 29 | 108 | 86 | 0.796 | 0.8504 | PASS |
| v0.7 | 43 | 108 | 92 | 0.852 | 0.8507 | PASS |
| v0.7 | 71 | 108 | 85 | 0.787 | 0.8375 | PASS |
| v0.7 | 101 | 108 | 85 | 0.787 | 0.8604 | PASS |
| **v0.7 range** | | | | | **0.8375 - 0.8604** | **5/5 seeds** |

## Yardstick: v0.5 consensus (108-event common split)

| version | seed | n events | n covered | coverage | wk1 (gate) | passes wk1 >= 0.70 |
| --- | --- | --- | --- | --- | --- | --- |
| v0.4 | 17 | 108 | 93 | 0.861 | 0.5802 | FAIL |
| v0.4 | 29 | 108 | 86 | 0.796 | 0.5830 | FAIL |
| v0.4 | 43 | 108 | 90 | 0.833 | 0.5733 | FAIL |
| v0.4 | 71 | 108 | 84 | 0.778 | 0.6053 | FAIL |
| v0.4 | 101 | 108 | 84 | 0.778 | 0.5844 | FAIL |
| **v0.4 range** | | | | | **0.5733 - 0.6053** | **0/5 seeds** |
| v0.5 | 17 | 108 | 83 | 0.769 | 0.5673 | FAIL |
| v0.5 | 29 | 108 | 84 | 0.778 | 0.5698 | FAIL |
| v0.5 | 43 | 108 | 77 | 0.713 | 0.6137 | FAIL |
| v0.5 | 71 | 108 | 75 | 0.694 | 0.5535 | FAIL |
| v0.5 | 101 | 108 | 87 | 0.806 | 0.5371 | FAIL |
| **v0.5 range** | | | | | **0.5371 - 0.6137** | **0/5 seeds** |
| v0.7 | 17 | 108 | 89 | 0.824 | 0.5389 | FAIL |
| v0.7 | 29 | 108 | 86 | 0.796 | 0.4969 | FAIL |
| v0.7 | 43 | 108 | 92 | 0.852 | 0.5077 | FAIL |
| v0.7 | 71 | 108 | 85 | 0.787 | 0.4913 | FAIL |
| v0.7 | 101 | 108 | 85 | 0.787 | 0.5547 | FAIL |
| **v0.7 range** | | | | | **0.4913 - 0.5547** | **0/5 seeds** |

## Yardstick: v0.7 full-coverage consensus (108-event common split)

| version | seed | n events | n covered | coverage | wk1 (gate) | passes wk1 >= 0.70 |
| --- | --- | --- | --- | --- | --- | --- |
| v0.4 | 17 | 108 | 93 | 0.861 | 0.7364 | PASS |
| v0.4 | 29 | 108 | 86 | 0.796 | 0.7371 | PASS |
| v0.4 | 43 | 108 | 90 | 0.833 | 0.7736 | PASS |
| v0.4 | 71 | 108 | 84 | 0.778 | 0.7366 | PASS |
| v0.4 | 101 | 108 | 84 | 0.778 | 0.8077 | PASS |
| **v0.4 range** | | | | | **0.7364 - 0.8077** | **5/5 seeds** |
| v0.5 | 17 | 108 | 83 | 0.769 | 0.7588 | PASS |
| v0.5 | 29 | 108 | 84 | 0.778 | 0.7953 | PASS |
| v0.5 | 43 | 108 | 77 | 0.713 | 0.7171 | PASS |
| v0.5 | 71 | 108 | 75 | 0.694 | 0.7169 | PASS |
| v0.5 | 101 | 108 | 87 | 0.806 | 0.7036 | PASS |
| **v0.5 range** | | | | | **0.7036 - 0.7953** | **5/5 seeds** |
| v0.7 | 17 | 108 | 89 | 0.824 | 0.8498 | PASS |
| v0.7 | 29 | 108 | 86 | 0.796 | 0.8504 | PASS |
| v0.7 | 43 | 108 | 92 | 0.852 | 0.8507 | PASS |
| v0.7 | 71 | 108 | 85 | 0.787 | 0.8375 | PASS |
| v0.7 | 101 | 108 | 85 | 0.787 | 0.8604 | PASS |
| **v0.7 range** | | | | | **0.8375 - 0.8604** | **5/5 seeds** |

## What `results/v07_release/bias_matrix.json` actually contains

H-YARDSTICK cites `bias_matrix.json` as the evidence. Reproducing its six
cells from the per-seed recomputation identifies the quantity it holds:

| model | yardstick | published value | mean wk1 (gate) | mean kappa, all rows incl. abstentions | identified as | as the 0.70 gate it would |
| --- | --- | --- | --- | --- | --- | --- |
| v0.4 | v0.5 consensus | 0.553996 | 0.585230 | 0.553996 | mean kappa over ALL rows, abstentions included | FAIL |
| v0.4 | v0.7 consensus (full coverage) | 0.704578 | 0.758291 | 0.704578 | mean kappa over ALL rows, abstentions included | PASS |
| v0.5 | v0.5 consensus | 0.518891 | 0.568278 | 0.518891 | mean kappa over ALL rows, abstentions included | FAIL |
| v0.5 | v0.7 consensus (full coverage) | 0.671259 | 0.738350 | 0.671259 | mean kappa over ALL rows, abstentions included | FAIL |
| v0.7 | v0.5 consensus | 0.487183 | 0.517894 | 0.487183 | mean kappa over ALL rows, abstentions included | FAIL |
| v0.7 | v0.7 consensus (full coverage) | 0.812725 | 0.849764 | 0.812725 | mean kappa over ALL rows, abstentions included | PASS |

Read as the gate metric, `bias_matrix.json` does **not** support the sentence
it is attached to: its v0.7-labels column has v0.5 at 0.671, below the 0.70
gate. The sentence is supported by wk1 (kappa on covered rows), which this
audit reproduces above and which no committed JSON contains for v0.4 or v0.5.

## Decomposition: the two yardsticks disagree, so where does the flip live?

- common held-out events (all three models): **108**
- label identical in v0.5 and v0.7: **63** (58.3% exact agreement)
- label changed: **45** (41.7%)

| model | seed | wk1 on unchanged-label events | wk1 on changed-label events |
| --- | --- | --- | --- |
| v0.4 | 17 | 0.8149 (n=58) | 0.6260 (n=35) |
| v0.4 | 29 | 0.8333 (n=52) | 0.6264 (n=34) |
| v0.4 | 43 | 0.8266 (n=54) | 0.6988 (n=36) |
| v0.4 | 71 | 0.8144 (n=50) | 0.6272 (n=34) |
| v0.4 | 101 | 0.8482 (n=53) | 0.7427 (n=31) |
| v0.5 | 17 | 0.8667 (n=48) | 0.6410 (n=35) |
| v0.5 | 29 | 0.8686 (n=51) | 0.7193 (n=33) |
| v0.5 | 43 | 0.9121 (n=40) | 0.5428 (n=37) |
| v0.5 | 71 | 0.7494 (n=42) | 0.6504 (n=33) |
| v0.5 | 101 | 0.7143 (n=48) | 0.6430 (n=39) |
| v0.7 | 17 | 0.8737 (n=55) | 0.8145 (n=34) |
| v0.7 | 29 | 0.7857 (n=52) | 0.8194 (n=34) |
| v0.7 | 43 | 0.7917 (n=56) | 0.8307 (n=36) |
| v0.7 | 71 | 0.7857 (n=52) | 0.8024 (n=33) |
| v0.7 | 101 | 0.8437 (n=53) | 0.8144 (n=32) |

`wk1 on unchanged-label events` is the same quantity on both yardsticks,
because the two yardsticks agree there. It is the control for the flip.
Kappa is not a weighted average of subset kappas -- the pooled confusion
matrix mixes the two groups -- so read this as a diagnostic, not a
decomposition that sums.

### Direction of the label change (108 common held-out events)

| transition | events |
| --- | --- |
| L5 -> L7 | 10 |
| L4 -> L7 | 6 |
| L4 -> L3 | 5 |
| L6 -> L7 | 5 |
| L7 -> L6 | 3 |
| L3 -> L7 | 2 |
| L5 -> L4 | 2 |
| L3 -> L4 | 2 |
| L7 -> L5 | 2 |
| L6 -> L4 | 2 |
| L7 -> L4 | 2 |
| L5 -> L6 | 1 |
| L7 -> L3 | 1 |
| L4 -> L5 | 1 |
| L5 -> L3 | 1 |

### Labeller provenance on the same 108 events

- v0.5 labels produced by: mm31, q2b, q8b, step5
- v0.7 labels produced by: mm31, q2b, q8b, step5
- labellers in the v0.7 yardstick that never produced a v0.5 label: none

This is the circularity H-YARDSTICK already discloses: the two yardsticks
are not independent of the models, because the labeller families overlap.

## Which metric is the H-YARDSTICK residual-risk range built from?

H-YARDSTICK's residual risk says the three models agree more with the v0.7
labels (0.70-0.86) than with the v0.5 ones (0.48-0.57). Ranges of the three
models pooled, on the 108 events common to all three:

| metric | vs v0.5 labels | vs v0.7 labels |
| --- | --- | --- |
| wk1 (the gate metric, covered rows) | 0.4913 - 0.6137 | 0.7036 - 0.8604 |
| kappa over all rows, abstentions included | 0.4753 - 0.5662 | 0.6468 - 0.8253 |

`0.70-0.86` is wk1 against the v0.7 labels; `0.48-0.57` is kappa over all
rows against the v0.5 labels. The sentence therefore compares the gate
metric on one side with a non-gate metric on the other. The direction of
the finding survives on either metric alone (wk1: 0.70-0.86 vs 0.49-0.61;
all rows: 0.65-0.83 vs 0.48-0.57), but the two ranges are not the same
quantity and must not be quoted in one sentence.

