# v0.7 per-level error analysis — lane an02_errors

**Model:** `google/siglip2-base-patch16-224` + monotone CORAL ordinal head, v0.7
consensus manifest, group-disjoint test split. **Scope:** visual compatibility with
IDDSI texture levels L3–L7 for culinary-education / QA use. This is **not** a
swallowing-safety classifier and nothing below should be read as a safety claim.

**Split and denominators.** Test split = **108 events** (10 L3, 11 L4, 7 L5, 10 L6,
70 L7) out of 532 manifest rows (314 train / 110 val / 108 test). Every rate below
states its denominator. Confidence intervals are 4000 non-stratified bootstrap
resamples of those 108 events (seed 20261012).

**Files used (all read-only inputs; no sealed file was opened):**

| file | role |
|---|---|
| `/work/lane_runs/an02_errors/artifacts/v07_seed{17,29,43,71,101}_test_predictions.npz` | per-event calibrated CORAL probabilities + labels + event_ids (copied from `/work/runs/v07/siglip2_v07_fullcoverage_seed*/test_predictions.npz`) |
| `/work/lane_runs/an02_errors/artifacts/v07_seed*_calibration.json` | validation-calibrated abstention threshold per seed |
| `/work/lane_runs/an02_errors/artifacts/v07_seed*_metrics.published.json` | the shipped metrics, used only to verify my recomputation |
| `/work/data/manifest_v07_weighted.jsonl` | levels, splits, group_id, source_corpus, weak_rule, label agreement |
| `/work/data/harvest_d1/events.jsonl`, `harvest_d1b/events.jsonl`, `realweak_pilot/events.jsonl` | provenance titles, for the food-family axis only |
| `/work/lanes/an02_errors/trainer/train/model.py` | `coral_decode` (ordinal rank rule, not argmax) |

Scripts: `analysis/errors_v07/analyze_errors.py`, `analysis/errors_v07/analyze_structure.py`.
Outputs: `/work/lane_runs/an02_errors/analysis/`.

---

## 0. Which seed, and does my recomputation match the shipped run?

Selection rule (stated so the choice is checkable): among the 5 v0.7 seeds whose
validation calibration gate (`under-FNR ≤ 0.05`, `abstention ≤ 0.30`) was met —
**all five met it** — take the highest `macro_f1_with_abstentions`.

| seed | macro F1 (w/ abst.) | wtd κ (covered) | coverage | under-FNR (covered) | ECE |
|---|---|---|---|---|---|
| **43** | **0.3949** | **0.8507** | **0.8519** | 0.0435 | **0.0301** |
| 17 | 0.3594 | 0.8498 | 0.8241 | 0.0562 | 0.0240 |
| 101 | 0.3399 | 0.8604 | 0.7870 | 0.0353 | 0.0644 |
| 29 | 0.3317 | 0.8504 | 0.7963 | 0.0349 | 0.0774 |
| 71 | 0.3193 | 0.8375 | 0.7870 | 0.0353 | 0.0447 |

→ **best seed = 43** (`seed_ranking.json`). It also leads on coverage, quadratic κ
and ECE, so the choice is not knife-edge.

Reproduction check (`reproduction_check.json`,
`headline_metric_reproduction_seed43` in `structural_seed43.json`):

* my recomputed 6×6 confusion matrix equals the shipped one **exactly, for all 5 seeds**;
* `macro_f1_with_abstentions` 0.394931 recomputed vs 0.394931 shipped;
* `weighted_kappa_covered` 0.850721 recomputed vs 0.850721 shipped.

So the numbers below are the shipped model's, re-derived from its own test NPZ.

---

## 1. Confusion matrices (seed 43, n = 108)

Decision path: CORAL rank decode, then abstain when decoded confidence <
threshold 0.239427. **92 of 108 events got a level decision, 16 abstained.**

5×5 over decided events (row-normalised rates in brackets):

| true \ pred | L3 | L4 | L5 | L6 | L7 | row n |
|---|---|---|---|---|---|---|
| L3 | 9 (0.90) | 0 | 0 | 0 | 1 (0.10) | 10 |
| L4 | 1 (0.25) | 2 (0.50) | 0 | 0 | 1 (0.25) | 4 |
| L5 | 3 (0.60) | 0 | **0 (0.00)** | 0 | 2 (0.40) | 5 |
| L6 | 0 | 0 | 0 | **0 (0.00)** | 8 (1.00) | 8 |
| L7 | 0 | 0 | 0 | 0 | 65 (1.00) | 65 |

6×6 including abstentions (counts):

| true \ pred | L3 | L4 | L5 | L6 | L7 | unclear | row n |
|---|---|---|---|---|---|---|---|
| L3 | 9 | 0 | 0 | 0 | 1 | 0 | 10 |
| L4 | 1 | 2 | 0 | 0 | 1 | **7** | 11 |
| L5 | 3 | 0 | 0 | 0 | 2 | 2 | 7 |
| L6 | 0 | 0 | 0 | 0 | 8 | 2 | 10 |
| L7 | 0 | 0 | 0 | 0 | 65 | 5 | 70 |
| **col n** | 13 | 2 | **0** | **0** | 77 | 16 | 108 |

Outcome tally (denominator = 108 test events): 76 correct level, **16 wrong level**,
16 abstained. Of the 16 wrong-level decisions, **12 over-predicted, 4 under-predicted**.

---

## 2. (a) Which adjacent pairs confuse most

`adjacent_pairs_seed43.json`. Adjacent confusions, by pair and direction
(denominator = all test events of that true level):

| pair | up (thin→thick) | down (thick→thin) | rate |
|---|---|---|---|
| **L6–L7** | **8** (true L6 → pred L7) | 0 | **0.80 of the 10 true-L6 events**; 8/8 = 1.00 of the *decided* L6 events |
| L3–L4 | 0 | 1 (true L4 → pred L3) | 0.09 of the 11 true-L4 events |
| L4–L5 | 0 | 0 | 0 of 11 / 0 of 7 |
| L5–L6 | 0 | 0 | 0 of 7 / 0 of 10 |

**L6→L7 is the only adjacent pair that confuses at all**, and it is entirely
one-directional. Bootstrap 95% CI for the L6→L7 rate: **0.028–0.130** per test event.

**But the zeros for L4–L5 and L5–L6 are degenerate, not skill.** The model never
emits L5 or L6 as a decision: column totals for L5 and L6 are **0** in the whole
108-event test set, and the same holds in all five seeds (`emitted_level_census`).
Decided predictions are only ever L3 (13), L4 (2) or L7 (77). Even the raw argmax of
the calibrated distribution never tops L4 or L6 in any seed. So "which adjacent pair
confuses most" is partly the wrong question — on this test set the model is a
**three-way classifier plus abstention**.

That collapse is a representation failure, not a threshold artifact
(`structural_seed43.json`). Mean calibrated probability mass by true level:

| true | n | L3 | L4 | L5 | L6 | L7 |
|---|---|---|---|---|---|---|
| L3 | 10 | 0.634 | 0.141 | 0.081 | 0.049 | 0.094 |
| L4 | 11 | 0.285 | 0.177 | 0.172 | 0.135 | 0.231 |
| L5 | 7 | 0.333 | 0.140 | 0.133 | 0.132 | 0.262 |
| L6 | 10 | 0.045 | 0.060 | 0.106 | **0.155** | **0.633** |
| L7 | 70 | 0.021 | 0.028 | 0.054 | 0.097 | 0.800 |

For genuinely-L6 events the model puts **63.3%** of its mass on L7 and only 15.5% on
L6; **0 of 10** true-L6 events have argmax = L6 and 9 of 10 have argmax = L7. True-L5
events are bimodal instead (0.333 on L3, 0.262 on L7), which is why every decided L5
event goes to a tail.

Per-level recall over decided events: L3 0.90, L4 0.50, **L5 0.00, L6 0.00**, L7 1.00.
Precision: L3 0.69, L4 1.00, L7 0.84; **L5/L6 precision undefined** (never emitted).

---

## 3. (b) Symmetric, or biased to over/under?

`bias_seed43.json`. Denominator = 92 decided events unless stated.

* **over-predicted 12, under-predicted 4** → errors are **biased toward
  over-prediction** (mean signed error **+0.130** levels, median 0).
* Binomial sign test on the 16 wrong-level decisions: **p = 0.077** — the asymmetry
  is suggestive at this sample size, not significant on its own.
* Every over-prediction lands on **L7** (+1: 8 events, +2: 2, +3: 1, +4: 1); every
  under-prediction lands on **L3** (−1: 1, −2: 3). Errors are toward the two tails the
  model can actually express.
* Under-classification (the direction the project's gate constrains) = **4 of 108
  test events = 0.037** [95% CI 0.009–0.074], i.e. 4 of 65 decided L4–L7 events
  = 0.062. The gate target is ≤ 0.05 on validation; this test figure sits above it.
* Per-level signed bias is **not** uniform: L3 +0.40, L4 +0.50, L5 −0.40, **L6 +1.00**
  (all 8 decided L6 events over-predicted), L7 0.00. So the bias is one-sided at the
  level of the L6 row and cancels out in the aggregate.

Abstentions are doing real work in the thin-middle
(`abstention_analysis_seed43`): abstention rate by true level is **L3 0.00, L4 0.64
(7 of 11), L5 0.29 (2 of 7), L6 0.20 (2 of 10), L7 0.07 (5 of 70)**. Forced to argmax,
7 of the 16 abstained events would have been under-predictions — the abstention
mechanism, not the level head, is what keeps under-classification low on L4/L5.

Cross-seed stability (`cross_seed_stability.json`): the structure is **not a seed
artifact**. In all 5 seeds L6→L7 is the top confusion (7–8 events), L5→L3 is second
(2–3), over-predictions outnumber under-predictions (10–12 vs 3–5), and mean signed
error is +0.112 to +0.130.

---

## 4. (c) Does a food group dominate the errors?

**No.** `group_errors_seed43.json`, `significance_seed43.json`.

* **Heuristic food family** (word-boundary keyword match on provenance titles; 46 of
  108 titles carry no food word and are kept in an explicit `uninformative_title`
  bucket rather than being forced into a family): the largest error count in any
  *named* family is **2 of 16** (starch_dish 2/12 events = 0.167; meat_fish_egg 1/8 =
  0.125; candy_baked_sweet 1/7 = 0.143; smooth_dessert_pudding 1/2 = 0.500). The
  biggest single bucket is `uninformative_title` at 7/16 = 44% of errors, which is a
  **metadata-coverage artifact, not a food group**. Worst-group Fisher exact p = 0.23
  (n = 2) — nothing.
* **Harvest corpus / provider** (the strongest structured "where did this image come
  from" axis): `harvest_d1` 8/68 = 0.118, `harvest_d1b` 6/29 = 0.207,
  `realweak_pilot` 2/11 = 0.182; by provider open-images 8/68 = 0.118, openverse 6/28
  = 0.214, nutrition5k 2/11 = 0.182, wikimedia 0/1. Worst group Fisher exact
  p = 0.36 — **no corpus dominates**.
* **Harvest visual bucket** (`weak_rule`): `visible_pieces_about_15mm_l6_candidate`
  8/29 = **0.276** vs 8/79 = 0.101 for everything else, Fisher exact **p = 0.033**.
  This is the only axis with a statistically distinguishable error concentration — but
  it is a *texture* bucket, not a food group, and it is the same fact as §2 (L6 is the
  problem level).
* **Label agreement** is the strongest signal of all: events with a **split** consensus
  (`1/2`) go wrong **9 of 22 = 0.409** vs 7/86 = 0.081 for unanimous-or-single events,
  Fisher exact **p = 0.0006**. Split-label events are 20% of the test set but carry
  **56%** of the wrong-level decisions. Read this as *label noise, not model skill*:
  the model is being scored against ground truth that two labeller families already
  disagreed about.

Caveat on the L6 bucket (`harvest_l6_bucket_seed43`): the harvest rule
`visible_pieces_about_15mm_l6_candidate` and the consensus labels disagree badly —
only **4 of its 29 events** (13.8%) actually carry a consensus L6 label; 18 are
labelled L7 and 6 L5. So that bucket's elevated error rate is partly a *harvest-rule
vs consensus-label* disagreement, which is a data-quality observation about the
corpus, not a finding about the model.

---

## 5. Headline numbers, one place

| quantity | value | denominator |
|---|---|---|
| test events | 108 | v0.7 group-disjoint test split |
| decided / abstained | 92 / 16 | 108 |
| correct level | 76 | 108 |
| wrong level | 16 | 108 (0.148 [0.083–0.213]) |
| over / under | 12 / 4 | 16 wrong-level decisions |
| mean signed error | +0.130 | 92 decided |
| adjacent slips | 9 | 16 wrong-level decisions |
| L6→L7 | 8 | 10 true-L6 events (8/8 of decided) |
| levels never emitted | L5, L6 | 92 decisions, seed 43; same in all 5 seeds |
| under-classification rate | 0.037 [0.009–0.074] | 108 test events |
| split-label error rate | 0.409 (9/22) | 22 `1/2` events; Fisher p = 0.0006 |

## 6. Limitations stated plainly

* 108 test events, 7 of them true-L5 and 10 true-L6. Per-level rates for L4/L5/L6 rest
  on 4–8 decisions each; the CIs above are wide for a reason.
* The food-family axis is a heuristic keyword map over third-party provenance titles,
  is my own construction, and leaves 46 of 108 events unclassified by design. Treat it
  as weak evidence; the corpus/provider and label-agreement axes are the structured ones.
* Abstention is a non-decision. Coverage here is 0.852, so ~15% of events produce no
  level at all, and that share is concentrated on L4 (0.64).
* All of this is the **held-out consensus-labelled** split. It says nothing about the
  sealed physical evaluation set, which was not opened.
