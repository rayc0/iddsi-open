# Benchmark framing — IDDSI-Open v0.7 against the EdUHK / Frontiers in Nutrition anchor

**Scope of this document.** It puts our v0.7 exact-match and MAE next to the one
external single-image number this repo cites, on the basis that external paper
uses, and then lists every reason the two numbers are not the same measurement.
Nothing here is a claim that our model is better or worse than theirs. Our
weights are withheld and photo-classification is killed as a product capability
under the wk1 kill rule; this document changes neither.

**Reproduce everything below from:**

```bash
docker exec -w /work/lanes/an05_external iddsi-train python -I tools/benchmark_framing.py
docker exec -w /work/lanes/an05_external iddsi-train python -I tools/label_noise_band.py
```

Outputs: `lane_runs/an05_external/benchmark_framing.json`,
`lane_runs/an05_external/label_noise_band.json`. CPU only, no GPU, no network,
no model weights, no reading of the sealed evaluation set.

---

## 1. The anchor, and what we can and cannot say about it

The repo cites the anchor in these places:

| file:line | what it says |
|---|---|
| `hf/model_README.md:149-155` | 61.74% exact match (chosen Qwen pipeline), 70.43% (GPT-4.1), FNR 9.2%, MAE 0.809, IDDSI levels 0–7, 115 images, "physically rheology-verified labels from certified clinicians", 14-model sweep |
| `paper/paper.md:76,134-136` and `paper/sections/related_work.md:50-52` | 61.74% exact match on CEIV-115; open VLMs missed every liquid hazard; CEIV publishes examples, not the full machine-readable corpus |

The citation is `@eduhk_frontiers` at `paper/references.bib:19`, whose own note
records that the authors, volume and pages are still `TODO`.

**What I could not verify.** I did not open the paper. No network access was
used and no copy of CEIV-115 exists in this workspace — the repo itself states
(`paper/sections/related_work.md:97`) that CEIV publishes examples rather than
the machine-readable corpus. So 61.74 / 70.43 / 0.809 / 9.2 / 115 / 8 / 14 are
**restated from this repo's own citation, not measured by me**. Every number in
sections 2–5 below *is* measured by me from the files named there.

One detail circulating with the anchor is **not present in any file in this
repo**: that 50 of the 115 images were web-sourced. A repo-wide grep for it
returns only `TASK.md`. It is therefore not asserted anywhere below; treat it as
unverified until someone reads the paper. (The "14-model sweep" *is* in the
release card, at `hf/model_README.md:149`.)

---

## 2. Our numbers on the same basis (exact level match, all covered test events)

Ground truth: `data/manifest_v07_weighted.jsonl`, `split == "test"` — **108
events** of 532 rows (314 train / 110 val / 108 test). Test level distribution:
L3 = 10, L4 = 11, L5 = 7, L6 = 10, **L7 = 70**. All 108 are web-harvested
(`harvest_d1` 68, `harvest_d1b` 29, `realweak_pilot` 11); 0 synthetic; 108
distinct groups, all of size 1.

Predictions: `runs/v07/siglip2_v07_fullcoverage_seed{17,29,43,71,101}/test_predictions.npz`
plus `test_structured_outputs.json`. "Covered" means
`confidence >= calibrated_threshold`; anything below is an abstention
(`decision: "unclear"`, no `level` key).

**Self-check before any of these numbers is trusted:** the script recomputes
quadratic weighted kappa and coverage for all five seeds from the raw `.npz`
and compares them to each run's published `metrics.json`. Maximum absolute
difference across all 5 seeds: **0.0** (`benchmark_framing.json → self_checks`).
The reading path is therefore the same one that produced the published v0.7
metrics. Note in passing that `test_predictions.npz` stores **0-indexed** class
labels (0–4) while the manifest stores real IDDSI levels (3–7); the script
asserts the off-by-three rather than assuming it.

### The same basis as the anchor

| seed | covered / 108 | **exact match** | 95% bootstrap CI | **MAE (levels)** |
|---|---:|---:|---:|---:|
| 17 | 89 | **80.90%** | [71.91, 88.76] | **0.292** |
| 29 | 86 | **84.88%** | [76.74, 91.86] | **0.244** |
| 43 | 92 | **82.61%** | [75.00, 90.22] | **0.283** |
| 71 | 85 | **84.71%** | [76.47, 91.76] | **0.247** |
| 101 | 85 | **84.71%** | [76.47, 91.76] | **0.247** |
| **pooled** | **437 covered events** | **83.52%** | [79.86, 86.96] | **0.263** |

Range across the 5 seeds: **exact match 80.90% – 84.88%**, **MAE 0.244 – 0.292
IDDSI levels**, at coverage 78.7% – 85.2%.

### The other denominators, because the choice of denominator is the whole argument

| basis | exact match | MAE |
|---|---:|---:|
| covered events only, CORAL decode (the anchor's basis) | 80.90 – 84.88% | 0.244 – 0.292 |
| all 108 events, abstentions scored as misses | 66.67 – 70.37% | 0.244 – 0.292 |
| all 108 events, CORAL decode, **abstention bypassed** | 69.44 – 74.07% | 0.380 – 0.444 |
| all 108 events, plain argmax, abstention bypassed | 71.30 – 72.22% | 0.454 – 0.509 |
| covered events, macro (unweighted per-class) exact match | 37.5 – 48.0% | — |

### Baselines on the identical 108 events — read the exact match against these

| predictor | exact match | MAE |
|---|---:|---:|
| always answer L7 (the majority test class, 70/108) | **64.81%** | **0.898** |
| always answer L6 / L5 / L4 / L3 | 9.26 / 6.48 / 10.19 / 9.26% | 1.194 / 1.676 / 2.287 / 3.102 |
| draw a label from the empirical test prior | 45.18% | — |

On the covered subset the same always-L7 baseline scores 70.65% – 76.47%, against
our 80.9% – 84.9% — a lift of **8.2 – 12.0 points**. On all 108 events with
abstentions scored as misses the lift over the same 64.81% baseline is only
**1.9 – 5.6 points**; with abstention bypassed, 4.6 – 9.3 points. On MAE the
picture is stronger and is the honest signal here: our covered MAE is **0.356 –
0.412 of** the matched constant predictor's (0.600 – 0.761 levels), i.e. about
60% lower, and the abstention-bypassed all-events MAE is 0.423 – 0.495 of its
0.898. **The lift over "always say L7" is 1.9 – 12.0 points depending on the
denominator, and the MAE is roughly 40–60% below the constant predictor's.** That
is the honest shape of the result: MAE carries real signal, exact match is
dominated by a 65%-L7 test set.

Per-class exact match on covered events, every seed: **L7 100%**, L3
77.8–90.0%, L4 0–50.0%, **L5 0.0%, L6 0.0% in all five seeds**. The model never
once correctly names an L5 or an L6. An exact-match number this high is an
L7-detector plus a partial L3 detector.

### Label-noise band on our own headline

`label_noise_band.json`. 22 of the 108 test events (20.4%) carry two
disagreeing labeller votes (`agreement: "1/2"`; the manifest records both votes
in `votes` but not the rule that resolved them). Re-scoring all five seeds with
those 22 labels flipped to the *other* labeller's vote:

| | as published | split labels flipped |
|---|---|---|
| exact match, covered | 80.90 – 84.88% | 79.78 – 87.06% |
| MAE, covered | 0.244 – 0.292 | 0.188 – 0.326 |

Roughly ±3 points of exact match and ±0.04 levels of MAE move with the yardstick
alone, on 20% of the test set. The EdUHK labels, being clinician-verified, do
not carry a band of this kind.

---

## 3. The number this release card printed against the anchor was on the wrong basis

The v0.7 results table printed `exact match 0.694 – 0.741` and
`MAE in IDDSI levels 0.380 – 0.444`, immediately followed by the EdUHK comparison
and the sentence "Our numbers are better than that on both." Those lines are at
321–322 and 325–331 of the pre-edit file, still recoverable with
`git show HEAD:hf/model_README.md`. The table has since been rewritten with every
basis labelled (`hf/model_README.md`, "v0.7 on its own labels, stated with the
caveat attached"); this section records what the old row was and why it was
wrong.

I tested every candidate rule against those two printed ranges
(`benchmark_framing.json → summary.decode_variant_audit`). Exactly one rule
reproduces **both**:

| candidate rule | exact match | MAE | reproduces the printed row? |
|---|---:|---:|---|
| CORAL decode over all 108 events, abstention bypassed | 69.44 – 74.07% | 0.380 – 0.444 | **yes, both** |
| argmax over all 108 events, abstention bypassed | 71.30 – 72.22% | 0.454 – 0.509 | no |
| CORAL over covered events only | 80.90 – 84.88% | 0.244 – 0.292 | no |
| all 108 events, abstentions scored as misses | 66.67 – 70.37% | 0.244 – 0.292 | no |

So the printed row **bypasses the abstention mechanism**: the 16–23 test events
each seed declared `unclear` are forced to a level and scored. That is a
different quantity from the covered-events kappa printed three rows above it in
the same table, and it is not the number the release card's own
"abstention is mandatory" design produces. It is also, notably, the *closer*
analogue to a VLM that always answers — which is an argument for printing it
with its basis stated, not for leaving it unlabelled next to 61.74%.

A second, weaker instance of the same problem sits in the v0.3 section
(`hf/model_README.md`, "Where this sits against published work"), which says "our
MAE (0.655–0.761) is lower than their 0.809". That MAE is the correct v0.3 figure
for its own section (per-seed values in `runs/v03/selective_metrics.json`,
held-out n=142), so the attribution is not wrong — but the caveat attached to it
lists **two** reasons the comparison flatters us (8 levels vs 5; 31%-consistent
labels) where ten apply, and it does not say that 0.655–0.761 is an
all-events, abstention-bypassed figure. Section 4 below is the full list; the v0.3
caveat now points at it rather than restating a subset.

### The mislabelled basis is systemic, not a one-off in the v0.7 row

The same audit runs on the v0.5 model trained on the cleaned 314-row corpus,
whose runs are still on disk
(`runs/v05/siglip2_v05_clean_seed{17,29,43,71,101}/test_predictions.npz`) and
whose numbers `hf/model_README.md` Finding 3 prints as
`exact 0.426 – 0.500 | MAE 0.778 – 0.870` on the identical 108-event test set:

| rule | exact match | MAE | reproduces the printed v0.5 row? |
|---|---:|---:|---|
| CORAL decode over all 108 events, abstention bypassed | 42.59 – 50.00% | 0.778 – 0.870 | **yes, both** |
| argmax over all 108 events, abstention bypassed | 51.85 – 54.63% | 0.852 – 0.944 | no |

So in at least two independently verifiable places the release card's
exact-match and MAE rows are **all-events, abstention-bypassed** figures, printed
without that basis, alongside kappa/F1/coverage rows that are **covered-events**
figures. Any reader who takes "exact match" at face value next to "weighted
kappa, covered" is comparing two different denominators. The fix is not to
change the numbers — it is to label them, which is what
`hf/model_README.md:317-361` now does, and to print both bases side by side.

---

## 4. Every reason these two numbers are not directly comparable

1. **Different label authority.** Theirs are clinician-verified physical tests.
   Ours are VLM consensus (`label_source: consensus`, 108/108 of the test
   split). Our MAE measures agreement with a machine-built yardstick, not error
   against physical truth. The repo's own inter-judge finding is that two Qwen
   judges agree on 31.0% of the same images (n_pairs = 575,
   `hf/HAZARD_LOG.md` H-NOISE); the v0.7 doubled-sample cross-family agreement
   is 74.5% (251/337, `results/v07_release/coverage_meta.json`). A yardstick
   that is itself ~75% consistent cannot certify an 83.5% exact match.
2. **Circularity.** 102 of the 108 test rows carry a judgement from `step5`,
   one of the labellers whose votes define the consensus label
   (`data/manifest_v07_weighted.jsonl`, field `labellers`). For the 22 split
   test events the consensus follows `mm31` 13 times and `step5` 9 times. Our
   model is trained on these labels and then scored against them. The share of
   the agreement that is self-agreement is unknown and unmeasured.
3. **Different number of classes.** Theirs spans IDDSI 0–7 (8 levels, liquids
   included); ours spans L3–L7 (5 levels, no liquids at all). Exact match gets
   harder as the class count grows and MAE's attainable range grows with it
   (max |error| 7 vs 4). Our covered MAE is 0.263/4 = 6.6% of our scale range;
   their 0.809/7 = 11.6% of theirs — a normalisation that flatters us less than
   the raw comparison does, and which is still not a like-for-like comparison
   because our range excludes the liquid levels entirely.
4. **Different denominator and different abstention semantics.** Their 115
   images are all scored. Our covered basis drops 16–23 of 108 events that the
   model itself selected as low-confidence, which is a self-chosen subset and
   biases the covered number upward relative to an always-answers baseline. Our
   all-events number (66.7–70.4% with abstentions as misses) is the one to read
   against an always-answers VLM, and it sits **below** their GPT-4.1 figure.
5. **Different test-set class balance.** 70/108 = 64.8% of our test split is L7.
   A constant "L7" predictor scores 64.81% exact match on it, which already
   exceeds their 61.74% Qwen figure. Their 115-image class distribution is not
   stated anywhere in this repo, so no baseline-normalised comparison is
   possible without inventing it.
6. **Different image provenance.** All 108 of our test events are web-harvested
   CC-licence candidates from three harvest sets. Theirs is a clinician-verified
   set. (The claim that 50 of their 115 were web-sourced appears only in
   `TASK.md` and in no repo source file; unverified here.)
7. **Different model class.** They swept 14 general-purpose VLMs prompted to
   name a level. Ours is a fine-tuned SigLIP-2 classifier with a CORAL ordinal
   head, trained and calibrated on this corpus, with a decision threshold tuned
   on our own validation split. Fine-tuning on a corpus drawn from the same
   distribution as the test set is not comparable to zero-shot VLM prompting.
8. **Different test-set size and uncertainty.** 115 vs 108 events, and our
   covered subset is only 85–92 events. Our pooled 83.52% carries a bootstrap
   95% CI of [79.86, 86.96] (`benchmark_framing.json`), which lies **entirely
   above** both of their figures — and that is exactly why the CI is the wrong
   thing to lean on. It quantifies sampling noise on *our* 108 events and says
   nothing about the ~±3-point label-noise band in section 2, the 64.8%-L7 class
   balance in point 5, or the circularity in point 2. A confidence interval
   around a number measured against the wrong yardstick is still a number
   measured against the wrong yardstick.
9. **Their numbers cannot be re-derived here.** No CEIV-115 images, no label
   file, no per-image outputs, no paper text in this workspace. Nothing about
   their 61.74 / 70.43 / 0.809 / 9.2 can be checked, re-scored, or
   denominator-audited from here.
10. **Our own metric provenance was wrong until this run.** See section 3: the
    row printed against the anchor used a decode rule that bypasses the
    abstention the release card calls mandatory, and the v0.3 paragraph beside it
    gave an unstated basis and only two of the ten reasons. Any comparison built
    on those rows inherits the defect.

---

## 5. What may be claimed, and what may not

**May be claimed, with the basis attached:**

- On 108 held-out web-harvested events, at 78.7–85.2% coverage, exact IDDSI
  level match is 80.90–84.88% and MAE is 0.244–0.292 levels (5 seeds,
  `benchmark_framing.json`). Pooled over 437 covered events: 83.52%,
  95% CI [79.86, 86.96].
- On all 108 events with abstentions scored as misses, exact match is
  66.67–70.37%.
- Against the always-L7 baseline on the same 108 events (64.81% exact,
  MAE 0.898), our exact match is only **1.9 – 12.0 points higher depending on the
  denominator** (8.2 – 12.0 points on the covered basis against the matched
  covered-subset baseline of 70.65 – 76.47%; 1.9 – 5.6 points on the all-events
  basis with abstentions scored as misses), while our covered MAE is **0.356 –
  0.412 of** the matched constant predictor's, i.e. about 60% lower.
- Per-class exact match is 100% on L7 and **0% on L5 and L6 in every seed**;
  macro (per-class) exact match is 37.5–48.0%.

**May not be claimed:**

- That we beat, match, or lose to the EdUHK anchor. The two numbers are not the
  same measurement, for the ten reasons above. The correct statement is that the
  comparison is uninformative, not that it favours either side.
- Any accuracy statement at all. Our labels are machine consensus, ~75%
  cross-family consistent, partly produced by a labeller family our model was
  trained on, and no physical test has ever been run on these images. The
  physically-tested protocol (300 specimens) remains unrun.
- Any use of the printed `0.694 – 0.741` / `0.380 – 0.444` row without stating
  that it bypasses abstention.

---

## 6. Changes made by this pass

- `hf/model_README.md` — the v0.7 results table now states the decode basis and
  denominator of every exact-match and MAE row, prints the covered-basis,
  all-events, abstain-as-miss, argmax and macro variants side by side, adds the
  always-L7 and prior-sampler baselines, and records the correction to the old
  unlabelled row. The "Where this sits against published work" paragraph is
  replaced by a ten-point non-comparability list and an explicit statement that
  no better/equal/worse claim is made. The v0.3 paragraph gains a pointer to the
  full list and disclosure of its own unstated basis.
- `hf/HAZARD_LOG.md` — new **H-BENCH** entry recording the mislabelled-basis
  defect, how it was found, the corrected figures, and the residual risk that the
  corrected number is *higher* than the old one and could be misread as an
  upgrade. Change-control line added for v0.7 (2026-10-12).
- `paper/sections/related_work.md` — the EdUHK bullet now carries a short
  "on comparing with that number" paragraph with our figures and the reasons they
  are not comparable.
- `paper/references.bib` — the `@eduhk_frontiers` note records that the author
  list, date and title come from the release card and were **not** verified
  against the paper, and points here.
- `tools/benchmark_framing.py`, `tools/label_noise_band.py` — the two
  measurement scripts; `tools/tests/test_benchmark_framing.py` — 11 regression
  tests pinning every published figure to the recomputation.
- `lane_runs/an05_external/benchmark_framing.json`,
  `lane_runs/an05_external/label_noise_band.json` — the machine-readable outputs.

No gate, hazard entry, disclaimer, abstention requirement, or the wk1 kill-rule
statement was weakened, removed, or silenced. Nothing was trained, no weights
were read, and the sealed evaluation set was not opened.
