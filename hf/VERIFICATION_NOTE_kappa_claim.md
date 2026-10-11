# Verification note — `bias_matrix.json` does not contain the metric the claim rests on

**Status: CORRECTED 2026-10-12.** An earlier version of this note concluded that
the H-YARDSTICK claim "all three pass against the v0.7 consensus" was false. That
conclusion was **wrong** and is retracted below. Lane `an01_yardstick`
independently re-derived the numbers from the per-event prediction files and
reproduced the published ranges to 2.2e-16. What survives — and it is the more
useful finding — is a defect in the *evidence file*, not in the claim.

## The claim is CONFIRMED

Quadratic weighted kappa on **covered rows** (the gate metric, `wk1`), recomputed
from `test_predictions.npz` with the CORAL decode and QWK re-implemented from
definition (no sklearn, no `trainer/eval/metrics.py`), 5 seeds each:

| yardstick | v0.4 wk1 | v0.5 wk1 | v0.7 wk1 | verdict |
|---|---|---|---|---|
| v0.5 consensus | 0.5404–0.5995 (0/5) | 0.5371–0.6137 (0/5) | 0.4913–0.5547 (0/5) | all FAIL |
| v0.7 full-coverage consensus | 0.7364–0.8077 (5/5) | **0.7036–0.7953 (5/5)** | 0.8375–0.8604 (5/5) | all PASS |

These match the published ranges (v0.4 0.736–0.808, v0.5 0.704–0.795, v0.7
0.838–0.860) digit for digit. Control: the script reproduces each run's own
published wk1 with max abs diff 2.220e-16 over 15 runs. Denominators: 108 test
events common to all three models (v0.4's own v0.5-column denominator is its
142-event held-out split; the verdict is unchanged on the common 108).

## What was wrong with the first version of this note

It read `results/v07_release/bias_matrix.json`, took its v0.7-labels column at
face value as the gate metric, saw v0.5 at 0.671, and concluded the claim was
false. `bias_matrix.json` is **not** the gate metric. It holds the **mean kappa
over ALL rows, abstentions included**:

| model | yardstick | published | mean wk1 (gate) | mean kappa, all rows | identified as | as the 0.70 gate |
|---|---|---|---|---|---|---|
| v0.4 | v0.5 consensus | 0.553996 | 0.585230 | **0.553996** | all-rows kappa | FAIL |
| v0.4 | v0.7 consensus | 0.704578 | 0.758291 | **0.704578** | all-rows kappa | PASS |
| v0.5 | v0.5 consensus | 0.518891 | 0.568278 | **0.518891** | all-rows kappa | FAIL |
| v0.5 | v0.7 consensus | 0.671259 | 0.738350 | **0.671259** | all-rows kappa | FAIL |
| v0.7 | v0.5 consensus | 0.487183 | 0.517894 | **0.487183** | all-rows kappa | FAIL |
| v0.7 | v0.7 consensus | 0.812725 | 0.849764 | **0.812725** | all-rows kappa | PASS |

So the sentence attached to `bias_matrix.json` is **supported by a metric that no
committed JSON contains for v0.4 or v0.5**. The evidence file and the claim
disagree, and the claim is the one that is right.

## The three real defects that do remain

1. **`bias_matrix.json` is mislabelled as the evidence for a wk1 claim.** It is
   the all-rows kappa. Either the file is renamed/annotated, or the wk1 matrix is
   committed alongside it. `an01_yardstick`'s recomputation is the only place the
   wk1 numbers for v0.4/v0.5 currently exist.

2. **H-YARDSTICK's residual-risk sentence mixes two metrics in one breath.** Its
   "the three models agree more with the v0.7 labels (0.70-0.86) than with the
   v0.5 ones (0.48-0.57)" compares *wk1 against v0.7 labels* with *all-rows kappa
   against v0.5 labels*. On one metric alone the direction survives (wk1:
   0.70–0.86 vs 0.49–0.61; all rows: 0.65–0.83 vs 0.48–0.57), but the two ranges
   are not the same quantity and must not be quoted in one sentence.

3. **The two yardsticks are not independent of the models.** On the 108 common
   events, 63 labels (58.3%) are identical between v0.5 and v0.7 and 45 (41.7%)
   changed; the flip lives almost entirely in the changed-label subset (v0.5 seed
   43: wk1 0.9121 on unchanged labels vs 0.5428 on changed ones). The v0.5 and
   v0.7 labels were produced by the *same four labeller families* (mm31, q2b, q8b,
   step5), so the yardsticks share provenance with the thing being measured. This
   is the circularity H-YARDSTICK already discloses; it is not closed by this
   audit.

## Method note (why the first version was wrong)

The error was reading a JSON's values as the metric named in the prose that cites
it, without checking which quantity the file holds. The fix that caught it was a
full independent re-derivation from the per-event predictions — the same
discipline `hf/HAZARD_LOG.md` records for the three earlier validation errors in
this project. Any number quoted from an aggregate file should be re-derived from
the per-event data before it is used to overturn a claim.
