# Verification note — the "all three pass against v0.7" claim does not hold

Recorded 2026-10-12 by the base terminal (claude-stepstep131) during convo 0146,
while independently re-checking the project's central claim against source files.
This is a VERIFICATION finding, not a new experiment. The lane `an01_yardstick`
was dispatched to re-derive the same thing; treat its result as the cross-check.

## What the pack (convo 0146) claims

> Against the v0.5 consensus all three model versions FAIL the wk1 kappa>=0.70
> gate (0/5 seeds each). Against the v0.7 full-coverage consensus all three
> PASS — v0.4 kappa 0.736-0.808, v0.5 0.704-0.795, v0.7 0.838-0.860
> (`results/v07_release/bias_matrix.json`).

## What the source files actually say

`results/v07_release/bias_matrix.json` — one number per model x yardstick:

| model | vs v0.5 consensus | vs v0.7 full-coverage consensus |
|---|---|---|
| v0.3 (Qwen tier-2) | 0.360 | 0.173 |
| v0.4 (partial consensus) | 0.554 | **0.7046** |
| v0.5 (cleaned consensus) | 0.519 | **0.6713** |
| v0.7 (full-coverage consensus) | 0.487 | **0.8127** |

`results/v07_release/v07_seed*_metrics.json` — per-seed, n=108 test events, the
kappa field is named **`weighted_kappa_covered`** and `coverage` is 0.787-0.852:

| seed | coverage | weighted_kappa_covered |
|---|---|---|
| 17 | 0.824 | 0.8498 |
| 29 | 0.796 | 0.8504 |
| 43 | 0.852 | 0.8507 |
| 71 | 0.787 | 0.8375 |
| 101 | 0.787 | 0.8604 |

`results/v05_release/cleaned_seed*_metrics.json` — same fields, n=108:

| seed | coverage | weighted_kappa_covered |
|---|---|---|
| 17 | 0.769 | 0.5673 |
| 29 | 0.778 | 0.5698 |
| 43 | 0.713 | 0.6137 |
| 71 | 0.694 | 0.5535 |
| 101 | 0.806 | 0.5371 |

## The three discrepancies

1. **v0.5 does not pass on either yardstick.** Against the v0.7 labels the bias
   matrix gives 0.6713 (<0.70). Its own per-seed covered kappa is 0.537-0.614.
   The quoted "v0.5 0.704-0.795" matches neither file. The claim "all three
   PASS" is therefore wrong: only v0.4 (0.7046) and v0.7 (0.8127) clear 0.70 on
   the v0.7 yardstick.

2. **The quoted ranges are coverage-filtered, the gate is not.** The per-seed
   numbers are `weighted_kappa_covered`, computed on the ~80% of the 108 test
   events the model chose to answer. This is hazard **H-KAPPACOV** ("coverage-
   filtered kappa reported against an unconditional gate"), already recorded in
   `hf/HAZARD_LOG.md`. The v0.7 range 0.838-0.860 is the covered kappa; the same
   model's unconditional number in the bias matrix is 0.8127.

3. **n = 108, not the 532 cleaned events.** Every figure above is on the 108-event
   held-out split. Any statement that implies a larger denominator is wrong.

## Why it matters

The release card and the paper quote these numbers, and the decision about
whether anything can ship is conditional on the gate. A claim of "all three
pass" that is really "two pass, on a filtered subset" is the same failure mode
the project has already recorded twice (H-KAPPACOV, H-YARDSTICK).

## What is still true

The H-YARDSTICK *direction* is real and reproducible: the same predictions score
very differently against the two label sets (v0.7: 0.487 vs 0.813; v0.5: 0.519
vs 0.671; v0.4: 0.554 vs 0.705). The yardstick does change the outcome. What is
wrong is the specific "all three pass" summary and the unattributed ranges.

## Recommended corrections

- Every kappa in the card, the paper and the README must name the yardstick AND
  say whether it is covered or unconditional, with n.
- Restate the headline as: against the v0.7 full-coverage consensus, v0.4 and
  v0.7 clear wk1>=0.70 on the covered subset; v0.3 and v0.5 do not.
- Lane `rl03_yardstick_caveat` was dispatched to add the yardstick caveat; it
  must also carry this H-KAPPACOV correction.
