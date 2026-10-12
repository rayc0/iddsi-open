# Abstention-threshold sweep on the VALIDATION split (lane fix02_threshold)

> Research demonstration for culinary education. Estimates visual similarity to IDDSI
> descriptors; does not perform official IDDSI tests, assess swallowing, determine
> suitability for any person, or decide whether food is safe to consume. Never replaces
> clinician-prescribed texture or physical IDDSI testing. This project is not the
> official IDDSI website and is not endorsed by IDDSI.

## Question

Wave 1 measured that on the v0.7 held-out TEST split (n=108) no seed emits L5 or L6 as
a decision — per-level recall L5 0.00, L6 0.00
(`analysis/errors_v07/outputs/structural_seed43.json`). Two mechanisms produce that
symptom and the published confusion matrix cannot separate them:

1. the CORAL rank decode never **produces** L5/L6 (a representation failure), or
2. the decode produces them but every such decode falls below the confidence
   threshold (an abstention-gate failure).

This lane sweeps the confidence threshold over every value that is achievable on the
**validation** split and reports coverage, per-level recall, dangerous-direction FNR
and ECE at each point.

## Denominators and provenance

| item | value |
|---|---|
| split used for every number below | `val` (split field of `manifest_v07_weighted.jsonl`) |
| validation events | **110** (identical for all five seeds) |
| validation level support | L3 14, L4 8, **L5 6**, **L6 9**, L7 73 |
| held-out test events | **never loaded by any script in this lane** |
| decode | `train.model.coral_decode` — the shipped CORAL ordinal rank rule `#{k : P(y>k) > 0.5}` |
| probabilities | `coral_logits_to_probs(cutpoint_logits / temperature)` at the temperature shipped in `calibration.json` |
| threshold grid | exhaustive: `{0.0} ∪ {110 distinct confidences} ∪ {nextafter(max)} ∪ {shipped threshold}` = **113** candidates per seed |
| seeds | 17, 29, 43, 71, 101 (the five v0.7 release seeds) |

The grid is exhaustive rather than sampled: acceptance is `confidence >= threshold`, so
every threshold between two consecutive distinct confidence values yields an identical
acceptance set. A round-number 0.00–1.00 grid is also emitted in the CSV.

Code provenance (sha256, resolved through this lane's clone):

- `lane`: fix02_threshold
- `trainer_dir`: /work/lanes2/fix02_threshold/trainer
- `eval.metrics`: /work/lanes2/fix02_threshold/trainer/eval/metrics.py sha256=f5cf2934893a491e5336f8775b222a5f18e93fa7c9ae1a0b6ce097718e64c158
- `eval.calibration`: /work/lanes2/fix02_threshold/trainer/eval/calibration.py sha256=f5544f6ff7df28369e47e5f4b72d43264bee9a2f5904044a819cd2c0de092281
- `train.model`: /work/lanes2/fix02_threshold/trainer/train/model.py sha256=25e61bd3ba8e341d1ccd23cf28c875c048c9c4554672387622097758a616610b
- `manifest`: /work/data/manifest_v07_weighted.jsonl (split field == 'val' only)
- `note`: sys.path pinned to this lane's trainer clone; the container's editable install of another lane's clone is shadowed. TEST rows are never loaded.

## Answer, in one line

**Yes for reachability, no for recall.** On validation the CORAL rank decode *does*
produce L5 (4–5 events per seed) and L6 (11–15 events per seed) before abstention, and
lowering the threshold below the CORAL interior mass ceiling makes both levels appear as
decisions — at **no cost in coverage** (coverage rises, it does not fall). But **L5
recall stays 0.000 at every threshold for every seed**, because not one of the six
true-L5 validation events is ever decoded as L5 by the rank rule. L5's failure is a
representation failure, not a threshold failure. L6 is partly recoverable: recall
0.333 (3/9) at threshold 0 where the covered FNR is 0.109–0.136, and 0.143–0.250 at the
reachability boundary where it is 0.074–0.083 — so for four of five seeds the boundary
point does not satisfy the 0.05 gate at all.

## 1. Does the decode produce L5/L6 at all? (pre-abstention census, n=110 val)

| seed | L3 | L4 | **L5** | **L6** | L7 | interior decodes | interior mass ceiling | interior confidence range |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 17 | 7 | 12 | **5** | **15** | 71 | 32 | 0.2305 | 0.2206–0.2362 |
| 29 | 9 | 9 | **4** | **12** | 76 | 25 | 0.2229 | 0.2131–0.2264 |
| 43 | 8 | 11 | **5** | **12** | 74 | 28 | 0.2383 | 0.2293–0.2447 |
| 71 | 9 | 9 | **4** | **11** | 77 | 24 | 0.2228 | 0.2140–0.2290 |
| 101 | 10 | 10 | **4** | **12** | 74 | 26 | 0.2232 | 0.2160–0.2291 |

Every seed decodes interior levels on validation, so mechanism (2) is live: the
abstention gate is deleting interior levels the decode actually produced.

### Where the true-L5 and true-L6 validation events actually land

Pre-abstention decode of the validation rows, by true level (denominator = true-level
support on the val split):

| seed | true L3 (n=14) | true L4 (n=8) | **true L5 (n=6)** | **true L6 (n=9)** | true L7 (n=73) |
|---|---|---|---|---|---|
| 17 | L3 5, L4 6, L6 1, L7 2 | L3 1, L4 3, L5 4 | L3 1, L4 1, L6 2, L7 2 | L4 2, L6 3, L7 4 | L5 1, L6 9, L7 63 |
| 29 | L3 5, L4 5, L5 1, L6 1, L7 2 | L3 2, L4 2, L5 3, L6 1 | L3 2, L6 1, L7 3 | L4 2, L6 2, L7 5 | L6 7, L7 66 |
| 43 | L3 6, L4 5, L6 1, L7 2 | L3 1, L4 3, L5 4 | L3 1, L4 1, L6 1, L7 3 | L4 2, L6 3, L7 4 | L5 1, L6 7, L7 65 |
| 71 | L3 5, L4 5, L5 1, L6 1, L7 2 | L3 3, L4 1, L5 3, L6 1 | L3 1, L4 1, L6 1, L7 3 | L4 2, L6 3, L7 4 | L6 5, L7 68 |
| 101 | L3 5, L4 6, L6 2, L7 1 | L3 3, L4 2, L5 3 | L3 2, L6 1, L7 3 | L4 2, L6 3, L7 4 | L5 1, L6 6, L7 66 |

**No true-L5 validation event is decoded as L5 by any seed, at any threshold.** The six
true-L5 rows decode as L3/L4/L6/L7. That is the representation failure, and no
threshold can reach it.

## 2. The shipped operating point (validation)

| seed | shipped threshold | interior mass ceiling | threshold > ceiling? | coverage | accepted | interior decoded / accepted | levels emitted as decisions | fnr_covered | fnr (abstention detector) | ECE (all) |
|---|---:|---:|---|---:|---:|---:|---|---:|---:|---:|
| 17 | 0.2327 | 0.2305 | **yes** | 0.7727 | 85 | 32 / **7** | L3, L4, L7 | 0.0471 | 0.2667 | 0.0656 |
| 29 | 0.5092 | 0.2229 | **yes** | 0.7636 | 84 | 25 / **0** | L3, L7 | 0.0476 | 0.3077 | 0.0663 |
| 43 | 0.2394 | 0.2383 | **yes** | 0.7909 | 87 | 28 / **5** | L3, L4, L6, L7 | 0.0460 | 0.3077 | 0.0852 |
| 71 | 0.2286 | 0.2228 | **yes** | 0.8000 | 88 | 24 / **2** | L3, L4, L7 | 0.0455 | 0.3333 | 0.1005 |
| 101 | 0.5189 | 0.2232 | **yes** | 0.7545 | 83 | 26 / **0** | L3, L7 | 0.0482 | 0.2857 | 0.0684 |

In all five seeds the shipped threshold sits **above** the CORAL interior mass ceiling
`tanh(gap/4)`, so the gate has been 'satisfied' in a region where interior decodes are
structurally un-acceptable. For seeds 29 and 101 the shipped threshold (0.5092, 0.5189)
is more than twice the ceiling (0.2229, 0.2232) and **zero** of 25 and 26 interior
decodes survive. This is the abstention-gate twin of the H-CORAL argmax failure.

## 3. Coverage-targeted view: what does reachability cost in coverage?

For each target coverage the most conservative threshold that still covers at least
that fraction of the 110 validation events is selected, and the row is reported
unchanged. `L5/L6 emitted` is the count of decisions at that level.

### target coverage 70%

| seed | threshold | coverage | accepted | L5 emitted | L6 emitted | L5 recall (hits/covered) | L6 recall (hits/covered) | fnr_covered | fnr (detector) | rate_all | ECE all | ECE covered |
|---|---:|---:|---:|---:|---:|---|---|---:|---:|---:|---:|---:|
| 17 | 0.5142 | 0.7000 | 77 | 0 | 0 | 0.000 (0/3) | 0.000 (0/4) | 0.0260 | 0.1333 | 0.0182 | 0.0656 | 0.0829 |
| 29 | 0.5750 | 0.7000 | 77 | 0 | 0 | 0.000 (0/4) | 0.000 (0/4) | 0.0260 | 0.1538 | 0.0182 | 0.0663 | 0.0798 |
| 43 | 0.5585 | 0.7000 | 77 | 0 | 0 | 0.000 (0/4) | 0.000 (0/4) | 0.0260 | 0.1538 | 0.0182 | 0.0852 | 0.0954 |
| 71 | 0.5628 | 0.7000 | 77 | 0 | 0 | 0.000 (0/4) | 0.000 (0/4) | 0.0260 | 0.1667 | 0.0182 | 0.1005 | 0.1095 |
| 101 | 0.5623 | 0.7000 | 77 | 0 | 0 | 0.000 (0/5) | 0.000 (0/4) | 0.0519 | 0.2857 | 0.0364 | 0.0684 | 0.0709 |

### target coverage 80%

| seed | threshold | coverage | accepted | L5 emitted | L6 emitted | L5 recall (hits/covered) | L6 recall (hits/covered) | fnr_covered | fnr (detector) | rate_all | ECE all | ECE covered |
|---|---:|---:|---:|---:|---:|---|---|---:|---:|---:|---:|---:|
| 17 | 0.2313 | 0.8000 | 88 | 0 | 2 | 0.000 (0/4) | 0.000 (0/6) | 0.0795 | 0.4667 | 0.0636 | 0.0656 | 0.0707 |
| 29 | 0.2247 | 0.8000 | 88 | 0 | 0 | 0.000 (0/5) | 0.000 (0/6) | 0.0568 | 0.3846 | 0.0455 | 0.0663 | 0.0693 |
| 43 | 0.2394 | 0.8000 | 88 | 0 | 2 | 0.000 (0/4) | 0.000 (0/5) | 0.0455 | 0.3077 | 0.0364 | 0.0852 | 0.1043 |
| 71 | 0.2286 | 0.8000 | 88 | 0 | 0 | 0.000 (0/4) | 0.000 (0/4) | 0.0455 | 0.3333 | 0.0364 | 0.1005 | 0.1166 |
| 101 | 0.2255 | 0.8000 | 88 | 0 | 0 | 0.000 (0/5) | 0.000 (0/5) | 0.0682 | 0.4286 | 0.0545 | 0.0684 | 0.0777 |

### target coverage 90%

| seed | threshold | coverage | accepted | L5 emitted | L6 emitted | L5 recall (hits/covered) | L6 recall (hits/covered) | fnr_covered | fnr (detector) | rate_all | ECE all | ECE covered |
|---|---:|---:|---:|---:|---:|---|---|---:|---:|---:|---:|---:|
| 17 | 0.2280 | 0.9000 | 99 | 4 | 7 | 0.000 (0/5) | 0.250 (2/8) | 0.0909 | 0.6000 | 0.0818 | 0.0656 | 0.0607 |
| 29 | 0.2215 | 0.9000 | 99 | 2 | 5 | 0.000 (0/5) | 0.125 (1/8) | 0.0909 | 0.6923 | 0.0818 | 0.0663 | 0.0697 |
| 43 | 0.2372 | 0.9000 | 99 | 3 | 8 | 0.000 (0/5) | 0.286 (2/7) | 0.0707 | 0.5385 | 0.0636 | 0.0852 | 0.0990 |
| 71 | 0.2225 | 0.9000 | 99 | 1 | 6 | 0.000 (0/5) | 0.250 (2/8) | 0.0808 | 0.6667 | 0.0727 | 0.1005 | 0.0990 |
| 101 | 0.2219 | 0.9000 | 99 | 2 | 5 | 0.000 (0/6) | 0.143 (1/7) | 0.0909 | 0.6429 | 0.0818 | 0.0684 | 0.0718 |

### target coverage 100%

| seed | threshold | coverage | accepted | L5 emitted | L6 emitted | L5 recall (hits/covered) | L6 recall (hits/covered) | fnr_covered | fnr (detector) | rate_all | ECE all | ECE covered |
|---|---:|---:|---:|---:|---:|---|---|---:|---:|---:|---:|---:|
| 17 | 0.2206 | 1.0000 | 110 | 5 | 15 | 0.000 (0/6) | 0.333 (3/9) | 0.1364 | 1.0000 | 0.1364 | 0.0656 | 0.0656 |
| 29 | 0.2131 | 1.0000 | 110 | 4 | 12 | 0.000 (0/6) | 0.222 (2/9) | 0.1182 | 1.0000 | 0.1182 | 0.0663 | 0.0663 |
| 43 | 0.2293 | 1.0000 | 110 | 5 | 12 | 0.000 (0/6) | 0.333 (3/9) | 0.1182 | 1.0000 | 0.1182 | 0.0852 | 0.0852 |
| 71 | 0.2140 | 1.0000 | 110 | 4 | 11 | 0.000 (0/6) | 0.333 (3/9) | 0.1091 | 1.0000 | 0.1091 | 0.1005 | 0.1005 |
| 101 | 0.2160 | 1.0000 | 110 | 4 | 12 | 0.000 (0/6) | 0.333 (3/9) | 0.1273 | 1.0000 | 0.1273 | 0.0684 | 0.0684 |

**Coverage is not the cost.** Reachability requires a *lower* threshold, which raises
coverage. The binding cost is the dangerous-direction FNR: the cheapest covered FNR
among thresholds that emit L5 or L6 is
0.0581 / 0.0659 / 0.0455 / 0.0722 / 0.0833 for seeds 17 / 29 / 43 / 71 / 101, against
the 0.05 gate — so only seed 43 has any operating point that both emits an interior
level and meets the gate (3 such thresholds, all at or below its shipped 0.2394).
The abstention-detector FNR at those points runs 0.31–0.58, and rises to 1.00 at
threshold 0 because a threshold that accepts everything catches nothing.

## 4. The reachability boundary: highest threshold that still emits both L5 and L6

| seed | threshold | coverage | accepted | interior decoded / accepted | L5 emitted | L6 emitted | L5 recall | L6 recall | L3 recall | L4 recall | L7 recall | fnr_covered | fnr (detector) | rate_all | ECE all |
|---|---:|---:|---:|---:|---:|---:|---|---|---|---|---|---:|---:|---:|---:|
| 17 | 0.2305 | 0.8273 | 91 | 32 / 13 | 1 | 4 | 0.000 (0/5) | 0.143 (1/7) | 0.500 (5/10) | 0.500 (2/4) | 0.969 (63/65) | 0.0769 | 0.4667 | 0.0636 | 0.0656 |
| 29 | 0.2229 | 0.8455 | 93 | 25 / 8 | 1 | 2 | 0.000 (0/5) | 0.000 (0/6) | 0.500 (5/10) | 0.250 (1/4) | 0.971 (66/68) | 0.0753 | 0.5385 | 0.0636 | 0.0663 |
| 43 | 0.2383 | 0.8636 | 95 | 28 / 13 | 1 | 7 | 0.000 (0/5) | 0.167 (1/6) | 0.500 (6/12) | 0.333 (1/3) | 0.942 (65/69) | 0.0737 | 0.5385 | 0.0636 | 0.0852 |
| 71 | 0.2225 | 0.9000 | 99 | 24 / 13 | 1 | 6 | 0.000 (0/5) | 0.250 (2/8) | 0.455 (5/11) | 0.200 (1/5) | 0.971 (68/70) | 0.0808 | 0.6667 | 0.0727 | 0.1005 |
| 101 | 0.2231 | 0.8727 | 96 | 26 / 12 | 1 | 4 | 0.000 (0/6) | 0.143 (1/7) | 0.500 (5/10) | 0.333 (2/6) | 0.985 (66/67) | 0.0833 | 0.5714 | 0.0727 | 0.0684 |

Note what this table does **not** buy: L5 recall is 0.000 (0 hits) at every one of
these points, for every seed. The level is emitted and never right.

### Which direction do the newly-reachable decisions err in?

Levels are ordered soft -> thick (L3 softest, L7 thickest), so the dangerous
direction is `y_pred < y_true` — calling food SOFTER than it is. Composition of the
accepted L5 and L6 decisions at the reachability boundary and at threshold 0:

| seed | point | level | emitted | correct | dangerous under (`y_pred < y_true`) | over (`y_pred > y_true`) | true-level breakdown |
|---|---|---|---:|---:|---:|---:|---|
| 17 | boundary | L5 | 1 | 0 | 0 | 1 | L4 1 |
| 17 | boundary | L6 | 4 | 1 | 2 | 1 | L5 1, L6 1, L7 2 |
| 17 | threshold 0 | L5 | 5 | 0 | 1 | 4 | L4 4, L7 1 |
| 17 | threshold 0 | L6 | 15 | 3 | 9 | 3 | L3 1, L5 2, L6 3, L7 9 |
| 29 | boundary | L5 | 1 | 0 | 0 | 1 | L4 1 |
| 29 | boundary | L6 | 2 | 0 | 2 | 0 | L7 2 |
| 29 | threshold 0 | L5 | 4 | 0 | 0 | 4 | L3 1, L4 3 |
| 29 | threshold 0 | L6 | 12 | 2 | 7 | 3 | L3 1, L4 1, L5 1, L6 2, L7 7 |
| 43 | boundary | L5 | 1 | 0 | 0 | 1 | L4 1 |
| 43 | boundary | L6 | 7 | 1 | 4 | 2 | L3 1, L5 1, L6 1, L7 4 |
| 43 | threshold 0 | L5 | 5 | 0 | 1 | 4 | L4 4, L7 1 |
| 43 | threshold 0 | L6 | 12 | 3 | 7 | 2 | L3 1, L5 1, L6 3, L7 7 |
| 71 | boundary | L5 | 1 | 0 | 0 | 1 | L4 1 |
| 71 | boundary | L6 | 6 | 2 | 2 | 2 | L3 1, L5 1, L6 2, L7 2 |
| 71 | threshold 0 | L5 | 4 | 0 | 0 | 4 | L3 1, L4 3 |
| 71 | threshold 0 | L6 | 11 | 3 | 5 | 3 | L3 1, L4 1, L5 1, L6 3, L7 5 |
| 101 | boundary | L5 | 1 | 0 | 0 | 1 | L4 1 |
| 101 | boundary | L6 | 4 | 1 | 1 | 2 | L3 1, L5 1, L6 1, L7 1 |
| 101 | threshold 0 | L5 | 4 | 0 | 1 | 3 | L4 3, L7 1 |
| 101 | threshold 0 | L6 | 12 | 3 | 6 | 3 | L3 2, L5 1, L6 3, L7 6 |

At the reachability boundary each seed emits exactly **one** L5 decision and it is
always a true-L4 row called L5 — an over-prediction, so in the safe direction but
still wrong. The L6 decisions that appear at the boundary are dominated by true-L7
rows called L6, which IS the dangerous direction. Pooling the five seeds at threshold
0: **34 of 62 emitted L6 decisions (54.8%) are dangerous under-classifications**, and
19 of 22 emitted L5 decisions (86.4%) are over-predictions of true L3/L4 rows. So
lowering the threshold to reach the interior levels does not buy correctness; it buys
mostly L7 rows relabelled one level softer, which is precisely the error direction the
wk2 gate exists to catch.

### The gate-meeting interior-emitting points

| seed | thresholds emitting L5/L6 that also meet the 0.05 covered-FNR gate | most conservative such point |
|---|---|---|
| 17 | 0 | none |
| 29 | 0 | none |
| 43 | 3 | threshold 0.239426, coverage 0.8000, emitted L3 8, L4 4, L6 2, L7 74 |
| 71 | 0 | none |
| 101 | 0 | none |

For four of five seeds there is **no** threshold on the whole achievable grid that
both emits an interior level and satisfies the 0.05 under-classification gate. The one
exception is seed 43, whose shipped threshold already emits L6 — and note that its
three gate-meeting points sit within 2e-6 of each other (0.2394257 / 0.2394267 /
0.2394276), straddling a single event, so this exception is a knife-edge rather than
a robust operating point.

## 5. Threshold 0.0 (accept everything) — the ceiling on reachability

| seed | coverage | accepted | levels emitted | L5 recall | L6 recall | L3 recall | L4 recall | L7 recall | fnr_covered | fnr (detector) | ECE all |
|---|---:|---:|---|---|---|---|---|---|---:|---:|---:|
| 17 | 1.0000 | 110 | L3, L4, L5, L6, L7 | 0.000 (0/6) | 0.333 (3/9) | 0.357 (5/14) | 0.375 (3/8) | 0.863 (63/73) | 0.1364 | 1.0000 | 0.0656 |
| 29 | 1.0000 | 110 | L3, L4, L5, L6, L7 | 0.000 (0/6) | 0.222 (2/9) | 0.357 (5/14) | 0.250 (2/8) | 0.904 (66/73) | 0.1182 | 1.0000 | 0.0663 |
| 43 | 1.0000 | 110 | L3, L4, L5, L6, L7 | 0.000 (0/6) | 0.333 (3/9) | 0.429 (6/14) | 0.375 (3/8) | 0.890 (65/73) | 0.1182 | 1.0000 | 0.0852 |
| 71 | 1.0000 | 110 | L3, L4, L5, L6, L7 | 0.000 (0/6) | 0.333 (3/9) | 0.357 (5/14) | 0.125 (1/8) | 0.932 (68/73) | 0.1091 | 1.0000 | 0.1005 |
| 101 | 1.0000 | 110 | L3, L4, L5, L6, L7 | 0.000 (0/6) | 0.333 (3/9) | 0.357 (5/14) | 0.250 (2/8) | 0.904 (66/73) | 0.1273 | 1.0000 | 0.0684 |

## 6. Dangerous-direction FNR: three definitions, three denominators

`trainer/eval/abstention.py` defines the dangerous direction as `y_pred < y_true`
(predicting a SAFER texture than truth). Three quantities circulate in this repo and
they do not share a denominator:

| name | numerator | denominator | where it lives |
|---|---|---|---|
| `dangerous_fnr_abstention_detector` | dangerous rows accepted | all dangerous rows (pre-abstention) | `eval/abstention.py:risk_coverage_row` |
| `dangerous_underclassification_fnr_covered` | dangerous rows accepted | all covered rows | `eval/metrics.py:_point_metrics` (the shipped name) |
| `dangerous_underclassification_rate_all` | dangerous rows accepted | all N | `eval/metrics.py:_point_metrics` |

H-ABSTBLIND records that `dangerous_underclassification_rate_all` gates its numerator
on acceptance while dividing by all N, so it is the only one of the three that
improves when the model abstains. All three are reported at every point above.

Dangerous-direction samples on the validation split (pre-abstention, denominator of the detector FNR): seed 17 = 15/110, seed 29 = 13/110, seed 43 = 13/110, seed 71 = 12/110, seed 101 = 14/110.

## 7. Why no threshold recovers L5 (measured, not argued)

| seed | distinct L5 hit counts over all candidate thresholds | L5 hits at threshold 0 | true-L5 val rows | thresholds that emit L5 |
|---|---:|---:|---:|---:|
| 17 | [0] | 0 | 6 | 21 |
| 29 | [0] | 0 | 6 | 19 |
| 43 | [0] | 0 | 6 | 17 |
| 71 | [0] | 0 | 6 | 13 |
| 101 | [0] | 0 | 6 | 16 |

The set of distinct L5 hit counts is the single value `[0]` for every seed across the
whole candidate grid: L5 is emitted at 13–21 thresholds per seed and is never once
correct. Reachable and recalled are different claims, and only the first is true.

## 8. ECE across the sweep

ECE is `eval.metrics.expected_calibration_error` with 10 equal-width bins on the
CORAL rank-decode confidence (the probability mass of the decoded level), computed on
the 110 validation events. ECE (all) does not depend on the threshold — the bins and
the confidences are threshold-free — and it is repeated in every row so the table is
self-contained; ECE (covered) does move with the threshold because the mask changes.

| seed | ECE all (threshold-free) | ECE covered at shipped threshold | ECE covered at the reachability boundary | ECE covered at threshold 0 |
|---|---:|---:|---:|---:|
| 17 | 0.0656 | 0.0733 | 0.0649 | 0.0656 |
| 29 | 0.0663 | 0.0748 | 0.0706 | 0.0663 |
| 43 | 0.0852 | 0.1028 | 0.1037 | 0.0852 |
| 71 | 0.1005 | 0.1166 | 0.0990 | 0.1005 |
| 101 | 0.0684 | 0.0874 | 0.0734 | 0.0684 |

## 9. The decode is temperature-invariant, so this is not a calibration artifact

`P(y > k) > 0.5` is equivalent to `cutpoint_logit_k > 0`, and dividing the logits by a
positive temperature preserves the sign. The decoded level therefore cannot change
with temperature; only the confidence (class mass) can. Verified on the validation
rows of all five seeds:

| seed | decoded levels differing between T=1 and the shipped T |
|---|
| 17 | 0 of 110 |
| 29 | 0 of 110 |
| 43 | 0 of 110 |
| 71 | 0 of 110 |
| 101 | 0 of 110 |

The pre-abstention census in section 1 is therefore a property of the checkpoint, not
of the temperature that happened to be selected on the same 110 rows.

## 10. Reproduction fidelity: the CPU forward against the shipped GPU run

The container `iddsi-train` had lost CUDA when this lane ran (NVML `Unknown Error`,
`torch.cuda.device_count() == 0`), so the validation forward pass was re-run on CPU in
fp32. Two checks say that is safe here, and one says where it is not.

**Bit-identical logits.** Wave-1 lane `an03_calibration` dumped the same validation
forward pass from the same `best.pt` on the GPU
(`/work/lane_runs/an03_calibration/val_predictions_seed43.npz`, sha256
`6988394cea0f238b26e20ed9f7e685df6512e3cc98e5104b40f1e053de0f2764`). Comparing the two:

| quantity | value |
|---|---|
| max abs difference in the 110x4 cut-point logits | **0.0** (bit-identical) |
| labels and event_ids identical | yes (110/110) |
| decoded level disagreements | **0 of 110** |
| pre-abstention decode census (reference vs this lane) | L3 8, L4 11, L5 5, L6 12, L7 74 — identical |

**Where the CPU path does differ.** Applying the shipped threshold to the recomputed
confidences reproduces the shipped coverage for 3 of 5 seeds exactly and is one event
short for 2 of 5:

| seed | shipped threshold | coverage in shipped `calibration.json` | coverage recomputed here | accepted, shipped | accepted, recomputed |
|---|---:|---:|---:|---:|---:|
| 17 | 0.232672930 | 0.7727 | 0.7727 | 85 | 85 |
| 29 | 0.509160221 | 0.7727 | 0.7636 | 85 | 84 |
| 43 | 0.239426672 | 0.8000 | 0.7909 | 88 | 87 |
| 71 | 0.228564054 | 0.8000 | 0.8000 | 88 | 88 |
| 101 | 0.518900275 | 0.7545 | 0.7545 | 83 | 83 |

The shipped threshold is not one of the recomputed decoded confidences in any seed: it
is a knife-edge value that falls between two of them, and for seeds 29 and 43 one
recomputed confidence lands on the other side of it. The gap is ~1e-6 in the
confidence of a single event — the residual of CUDA kernel non-determinism between
the original training-time run and any later re-run, CPU or GPU. It moves coverage by
0.009 and nothing else: the decode census, the reachability boundary and the L5
zero-hit result are unchanged, because the decode itself has zero disagreements. The
shipped threshold is nevertheless carried as an explicit candidate in the sweep grid,
so the rows labelled 'shipped' are evaluated at the shipped value, not at the
nearest achievable one.

## 11. What this means for the threshold rule

1. **The shipped selector can pick a threshold above the CORAL interior mass ceiling,
   and did so in all five seeds.** `eval/calibration.py` in this clone already clips
   the candidate grid to `threshold <= ceiling`; the five shipped `calibration.json`
   files were produced before that clip. Re-running the shipped selector with the clip
   moves the threshold to 0.2302 / 0.2229 / 0.2372 / 0.2225 / 0.2231 and reports
   `target_met: false` in all five cases, because the gate can no longer be met inside
   the reachable region. That is an honest outcome and it is the one the clip was
   written to produce.
2. **L5 cannot be fixed by a threshold.** Zero of six true-L5 validation events decode
   as L5 in any seed at any threshold. The remedy is the representation, not the gate.
3. **L6 is partially recoverable but not for free.** L6 recall reaches 0.33 (3/9) at
   threshold 0 and 0.14–0.25 at the reachability boundary, where the covered FNR is
   0.0737–0.0833 against the 0.05 gate — so for four of five seeds the boundary point
   does not satisfy the gate at all.
4. **Do not report coverage as the price of reachability** — it is the wrong sign. The
   price is dangerous-direction FNR and gate compliance.

## Interpretation boundary

Research demonstration for culinary education. Estimates visual similarity to IDDSI
descriptors; does not perform official IDDSI tests, assess swallowing, determine
suitability for any person, or decide whether food is safe to consume. Never replaces
clinician-prescribed texture or physical IDDSI testing. This project is not the
official IDDSI website and is not endorsed by IDDSI. Training labels are VLM weak
labels, not physical IDDSI tests and not clinician-validated; every figure above is
measured against the inter-judge label noise recorded in `hf/HAZARD_LOG.md` (H-NOISE,
H-QWENBIAS) and means nothing about real-world accuracy.

