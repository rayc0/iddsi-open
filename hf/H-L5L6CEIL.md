# H-L5L6CEIL — the L5/L6 collapse is a label-and-support ceiling, not a bug

**Measured 2026-10-12 on the shipped v0.7 manifest and predictions.** This entry
records a diagnosis with its evidence, and what it rules out. No gate is changed,
no figure is restated, and the weights stay withheld.

## The symptom

Per-level recall on the v0.7 held-out split (n = 108) is **L3 0.90, L4 0.50,
L5 0.00, L6 0.00, L7 1.00** across all five seeds. Argmax never tops L4 or L6
either. Macro-F1 is capped by two zero-recall classes, so the published
`weighted_kappa_covered` of 0.838–0.860 is in practice a **three-class** result.

## What was ruled out, with measurements

**1. The decode rule.** Eight candidate rules compared on the validation split
only (110 events, 5 seeds, no abstention, `lane_runs/d01_decode/`):

| rule | L3 | L4 | L5 | L6 | L7 |
|---|---|---|---|---|---|
| shipped CORAL rank | 0.371 | 0.275 | **0.000** | 0.311 | 0.900 |
| per-level cut-points, out-of-fold | 0.543 | 0.425 | 0.167 | 0.133 | 0.882 |
| per-level cut-points, in-sample | 0.543 | 0.400 | 0.133 | 0.133 | 0.874 |
| argmax over 5 posteriors | 0.786 | 0.000 | **0.000** | **0.000** | 0.997 |
| top-2 band, lower edge | 0.786 | 0.000 | **0.000** | 0.778 | **0.005** |
| top-2 band, upper edge | 0.000 | 0.525 | **0.000** | **0.000** | 0.997 |

The shipped decode *does* emit L5 (4–5 times per seed) and L6 (11–15 times) — it
simply never gets an L5 right, and only ~31% of L6. The band rule buys L6 recall
0.778 by predicting L6 86 times out of 110 and dropping L7 recall to 0.005.
**No decode rule makes L5/L6 both reachable and correct.**

**2. The abstention threshold.** A full threshold sweep on validation
(`lane_runs/d02_threshold/`) found no setting that makes L5/L6 reachable without
destroying coverage or the dangerous-direction FNR.

**3. Class balance.** Equalising the training levels gave κ 0.766–0.781 at full
coverage — *worse* than the 0.838–0.860 baseline, and still failing calibration.

**4. Retraining.** A retrain with an alternative decode
(`lane_runs/d03_retrain/runs/v07_rankdecode/`) still produced L5 recall 0/7 and
L6 recall 0/10, with 8 of 10 true-L6 events routed to L7. A 2× cosine schedule
gave κ 0.830, below baseline.

## The cause

`data/manifest_v07_weighted.jsonl`, 532 rows (314 train / 110 val / 108 test):

| level | train examples | share of train | labels disputed (1/2) |
|---|---|---|---|
| L3 | 28 | 8.9% | 10% |
| L4 | 37 | 11.8% | 23% |
| **L5** | **13** | **4.1%** | **46%** |
| **L6** | 35 | 11.1% | **54%** |
| L7 | 201 | 64.0% | 8% |

The two levels the model never predicts are the two with the least training
support and the most contested labels. **L5 has 13 training examples, 12 of the
26 manifest-wide L5 rows disputed; L6 is 29 of 54 disputed.** A photo classifier
trained on 13 mostly-disputed L5 examples cannot learn L5, and it is not
obvious that it should try.

Corroboration from the agreement filter (`lane_runs/x09_labelnoise/`): scoring
only the 86 events where the labellers did not disagree lifts κ from 0.8498 to
0.8899 (mean over 5 seeds) — but **L5 and L6 recall stays 0.00 on every subset
and every seed**. Most of the remaining error is model error; the part that is
label noise is concentrated in exactly the middle levels the filter deletes.

## What this means

- The L5/L6 collapse is **not** a decode, threshold, class-balance or schedule
  defect. It is the honest signal of a label ceiling. Reporting it as a bug to be
  engineered away would be the same failure mode as H-KAPPACOV and H-YARDSTICK:
  optimising the artifact instead of the claim.
- **No architecture change can be evaluated against this test set for L5/L6**,
  because there are 7 L5 and 10 L6 test events with contested labels. Any "L5
  recall improved" claim from this split is noise.
- The only legitimate routes to real L5/L6 performance are (a) physically tested
  L5/L6 specimens — which is exactly what the sealed evaluation set is for — or
  (b) publishing the model honestly as what it is: a detector for the levels the
  data supports, with abstention everywhere else.
- κ improvements that leave L5/L6 at zero are improvements in predicting L7, and
  must not be quoted as model improvements.

## Reproduce

```bash
python3 ~/support.py                      # level support + agreement by split
python3 ~/summ_d01.py                     # decode-rule comparison, val split
cat lane_runs/x09_labelnoise/REPORT.md    # agreement-filtered ceiling
```
