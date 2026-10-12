---
language: en
license: apache-2.0
pipeline_tag: image-classification
library_name: transformers
base_model: google/siglip2-base-patch16-224
tags:
- image-classification
- food
- food-texture
- food-image-classification
- texture
- texture-analysis
- dysphagia
- iddsi
- iddsi-texture-levels
- dietary
- texture-modified-diet
- ordinal-classification
- coral
- siglip2
- calibrated-abstention
- selective-classification
- culinary-education
- research-preview
- not-for-clinical-use
---

# IDDSI-Open SigLIP2 v0 — IDDSI texture level (L3–L7) from a food photograph

**Repository:** [`lingualeap/iddsi-open-siglip2-v0`](https://huggingface.co/lingualeap/iddsi-open-siglip2-v0) ·
**Code licence:** Apache-2.0 · **Status:** research preview, release shape locked 2026-09-07 ·
**Base model:** [`google/siglip2-base-patch16-224`](https://huggingface.co/google/siglip2-base-patch16-224)

> ### ⚠️ This is NOT a swallowing-safety classifier
>
> It does not and cannot say whether a food is safe to eat, safe to swallow, or safe
> for any person. It never issues a safe/unsafe verdict, and it is not validated for
> assessing swallowing, aspiration risk, choking risk, or suitability for any
> individual. It is **not** a medical device and must not be used as one.
> **The model weights are deliberately WITHHELD** because the photo-classification
> head fails this project's own pre-registered safety gates (see
> [Why the weights are withheld](#why-the-weights-are-withheld)). What ships here is
> code, eval results, the flow-test grader, and a measured negative finding.
>
> **Who it is for:** researchers working on IDDSI texture-level vision, and culinary
> educators. **Culinary-education and research use only**, always confirmed by the
> applicable physical test.

**A food image classification model, described in one line:** a SigLIP-2 backbone with
a monotone CORAL ordinal head and calibrated abstention that reads an **IDDSI texture
level** (L3, L4, L5, L6 or L7) off a single food photograph — and says
`unclear — perform physical test` instead of guessing when it should. It is a
**SigLIP-2 food** classifier in the narrow sense that its whole training corpus is food
photography; it is not a general food recogniser and it has no ingredient, cuisine or
dish classes.

---

## Table of contents

1. [What this model does](#what-this-model-does)
2. [What this repository contains](#what-this-repository-contains)
3. [Why you may have found this card](#why-you-may-have-found-this-card)
4. [Model description](#model-description)
5. [Intended use and domain](#intended-use-and-domain)
6. [Out of scope — read before using anything here](#out-of-scope--read-before-using-anything-here)
7. [Evaluation](#evaluation)
8. [Bias, risks and limitations](#bias-risks-and-limitations)
9. [Why the weights are withheld](#why-the-weights-are-withheld)
10. [Why the dataset is withheld](#why-the-dataset-is-withheld)
11. [Training data](#training-data)
12. [The flow-test grader — the working deliverable](#the-flow-test-grader--the-working-deliverable)
13. [How to reproduce](#how-to-reproduce)
14. [Provenance of every number in this card](#provenance-of-every-number-in-this-card)
15. [Honest claims and never-claims](#honest-claims-and-never-claims)
16. [Citation, licence, contact](#citation-licence-contact)

---

## What this model does

IDDSI texture levels are the eight-level scale (L0–L7) used in **texture-modified diet**
practice for people who need food and drink altered in consistency. Levels L3–L7 cover
the solid end of the scale — from *liquidised/mashed* (L3) through *pureed* (L4),
*minced and moist* (L5) and *soft and bite-sized* (L6) to *regular* (L7). The L0–L4 end
of the scale is the **thickened liquid** end and is graded by the syringe flow test
rather than by appearance. The scale is defined by **physical** tests: fork pressure,
spoon tilt, spoon-tilt flow, finger pressure, and the 10 ml / 10 s syringe flow test.

This project asks a deliberately narrow question: **how much of an IDDSI texture level
is recoverable from an ordinary food photograph?** The answer, measured over five
release versions, is *less than the field assumes*, and this card is mostly about that
negative result. The question matters because food texture classification is where
texture-modified diet practice and computer vision meet, and the honest answer is that
the photograph is the bottleneck.

Concretely, the released head:

- takes one RGB food photograph (224×224 after the SigLIP-2 preprocessor);
- outputs a probability distribution over L3–L7 **and** an `unclear` option;
- decodes the level with the CORAL ordinal rank rule `level = #{k : P(y>k) > 0.5}`,
  not `argmax`;
- abstains — emitting `unclear — perform physical test` — when confidence falls below
  a calibrated threshold, and **never** substitutes an abstention for a verdict.

## What this repository contains

| ships | does not ship | why |
|---|---|---|
| Training + eval code (`trainer/`) | **Model weights** | fails the pre-registered safety gates |
| Recorded eval results (`results/`) | **The harvested image dataset** | unpublishable on licence grounds |
| The flow-test grader (`flowtest/`) | A clinician-validated label set | never existed |
| The negative finding, in full | Any safe/unsafe verdict capability | out of scope by design |

Withholding the weights and the dataset is a deliberate, stated decision. It is a
contribution, not an embarrassment — see
[Why the weights are withheld](#why-the-weights-are-withheld) and
[Why the dataset is withheld](#why-the-dataset-is-withheld).

## Why you may have found this card

This card is written for people who arrive from any of these searches, so that the
limitation is the first thing they read rather than the last:

> *"iddsi texture level"* · *"food texture classification"* ·
> *"food image classification model"* · *"texture modified diet"* ·
> *"dysphagia food texture"* · *"siglip2 food"* · *"ordinal classification"* ·
> *"thickened liquid"* · *"culinary education"* · *"iddsi level from a photograph"*

If you came here looking for a tool that tells you whether a plate is safe for a
patient: **there is no such tool here, and there should not be.** The project's own
conclusion is that a still photograph cannot carry the information the decision needs.
If you came here to reproduce the negative finding, or to study **calibrated
abstention** on an ordinal scale, you are in the right place.

## Model description

| | |
|---|---|
| Architecture | `OrdinalImageModel` — a frozen-then-unfrozen SigLIP-2 vision tower with a linear `OrderedCoralHead` on top |
| Backbone | `google/siglip2-base-patch16-224` (patch 16, 224×224 input) |
| Head | monotone CORAL: one shared score with strictly ordered learned cut-points, softplus gaps initialised at 1.5 so every interior level is reachable |
| Task | **ordinal classification** over 5 ordered levels (L3–L7) + an abstention option |
| Decode rule | `level = #{k : P(y>k) > 0.5}` (ordinal rank rule) |
| Calibration | temperature scaling on a held-out validation split, then a single confidence threshold chosen for the highest coverage that meets a 5% under-classification gate |
| Output | `{level: L3..L7 \| unclear, confidence: float}` |
| Abstention string | `unclear — perform physical test` (never a verdict) |

**Why CORAL and not 5-way softmax.** IDDSI levels are ordered: predicting L6 when the
truth is L7 is a *dangerous* error and predicting L7 when the truth is L6 is a *safe*
error, but a 5-way softmax treats both as equally wrong. CORAL's cumulative
formulation matches the structure of the scale.

**The CORAL decode bug, and why it is in this card.** An earlier build decoded the
CORAL head with `argmax` over the reconstructed categorical distribution. Because
`coral_logits_to_probs` builds interior classes as *differences of adjacent sigmoids*,
interior probability mass is capped near `tanh(gap/4) ≈ 0.17` while both tails reach
~0.26, so **`argmax` provably can never return an interior level for any input**. The
first trained run emitted only L3, L7 and abstain, on balanced L4–L7 data with no L3
at all. Proven numerically from the checkpoint, then fixed with the rank rule above
(`results/fix1_siglip2_coral_seed17/`) — hazard **H-CORAL**:

| condition | macro-F1 | coverage | ECE | temperature | note |
|---|---|---|---|---|---|
| before fix | 0.088 | 0.70 | 0.138 | 2.83 | interior levels unreachable |
| after fix (n = 109 held out) | 0.358 | 0.88 | 0.091 | 0.93 | all levels predicted |

Weighted kappa was 0.42 under both, measured against label noise. **If you reimplement
a CORAL head, this is the bug you will hit.**

## Intended use and domain

**Directly in scope**

- Reproducible methods research on IDDSI texture-level vision: ordinal heads,
  calibrated abstention, selective classification, and the AUGRC joint risk metric.
- Culinary-education settings — teaching kitchens, dietetic and catering courses — as a
  *discussion* artifact about what a photograph can and cannot show, with every output
  confirmed by the applicable physical test.
- Studying the negative finding: that web photographs without a physical scale fiducial
  cannot be weakly labelled for IDDSI texture levels.

**Domain vocabulary this model lives in:** IDDSI, IDDSI texture levels L3–L7,
IDDSI framework, food, food texture, texture analysis, texture-modified diet,
texture-modified foods, dietary, dysphagia, dysphagia food texture, minced and moist,
soft and bite-sized, pureed, thickened liquid, fork test, spoon-tilt test, syringe flow
test, food photograph, culinary education, food image classification.

IDDSI texture levels come from **dysphagia food-texture** practice: the scale exists
because some people need food and drink altered in consistency, and the levels are
defined by physical tests rather than by appearance. That is exactly why a photograph
is the wrong instrument for the middle of the scale — see
[Bias, risks and limitations](#bias-risks-and-limitations).

## Out of scope — read before using anything here

- **No safe or unsafe verdicts.** The release never issues a verdict. The abstention
  path exists precisely so that uncertainty is never resolved into a claim.
- **No dietary, clinical or patient claims.** Not for assessing swallowing, aspiration
  risk or suitability for any person. Not for prescribing or modifying a texture.
- **Not IDDSI-endorsed.** This project is independent of IDDSI and of every prior-art
  group named in [Honest claims and never-claims](#honest-claims-and-never-claims).
- **Regulatory posture.** An IDDSI-eating-advisor would be Software-as-a-Medical-Device
  (HK TR-007; NMPA class II–III; EU MDR Rule 11). This release is engineered to stay
  **out** of that class by never issuing a verdict. Do not wrap it in something that
  does issue verdicts.

## Evaluation

Every figure below is a **label-consistency measurement, not a real-world accuracy
measurement**. The test labels are machine-generated consensus labels, not
clinician-verified physical tests, and the model's own predictions correlate with
them by construction. Read the caveat in
[Bias, risks and limitations](#bias-risks-and-limitations) before quoting any number.

### The headline negative finding

Two VLM judges — Qwen3-VL-2B (tier-1) and Qwen3-VL-8B (tier-2), two sizes of one model
family — agree on **31.0%** of the same images (n_pairs = 575, full tier-2 sample).
That is only marginally above the ~20% chance level for 5 classes. Chance-corrected
agreement is very low: **κ_unweighted = 0.1062** and **κ_quadratic-weighted = 0.2766**
(Landis–Koch: *slight* / *fair* at best). Recomputed this run from the published
cross-tab: exact agreement 0.3096 over 575 pairs, κ_unweighted 0.1062,
κ_quadratic-weighted 0.2766. Source:
`results/regen_20260916/inter_judge_20260916.md`.

This is the finding the release is built around (hazard **H-NOISE**): **web photos
without a physical scale fiducial cannot be weakly labelled for IDDSI texture levels,
even by strong VLMs**, because the middle levels (L4/L5/L6) require absolute size and
flow information a photograph does not carry. Every label-derived metric in this card is
therefore measured against this noise and means nothing about real-world accuracy.

The disagreement is not random. Tier-1 **under-predicts** difficulty — the interior
classes drift toward the regular-food end, which is the *dangerous* direction
(predicting a SAFER texture than truth). This is hazard **H-UNDERPRED**, and it is the
same direction the wk2 gate is built to catch:

| tier-1 \ tier-2 | L3 | L4 | L5 | L6 | L7 | total |
|---|---:|---:|---:|---:|---:|---:|
| L3 | 11 | 1 | 0 | 0 | 0 | 12 |
| L4 | 4 | 42 | 8 | 28 | 69 | 151 |
| L5 | 5 | 3 | 27 | 42 | 54 | 131 |
| L6 | 3 | 8 | 3 | 30 | 111 | 155 |
| L7 | 4 | 1 | 6 | 47 | 68 | 126 |

L4 calls: 69 mis-predict to L7, 28 to L6, only 42 stay at L4. L6 calls: 111 collapse
into L7, only 30 stay. **A related defect (H-L7SINK) is now fixed:** unmeasurable rows
used to default to regular food — the least restrictive level — and now carry scale
anchors instead.

### The Qwen family was the outlier, not the photographs (H-QWENBIAS)

Every label before v0.4 came from Qwen3-VL-2B and Qwen3-VL-8B — two sizes of **one**
model family. Re-labelling the corpus blind with two architecturally unrelated families
(`step-5-preview` and `MiniMax-M3.1f`) gives:

| labeller pair | exact agreement | within ±1 level |
|---|---:|---:|
| Qwen-2B vs Qwen-8B (same family) | **30.9%** | 68.8% |
| Step-5 vs Qwen-8B (unrelated families) | **47.1%** | 73.2% |
| Step-5 vs MiniMax-M3.1f (both unrelated to Qwen) | **74.7%** | — |

Two unrelated families agree with each other at more than twice the rate either agrees
with Qwen, and Qwen-8B is *less* consistent with its own smaller sibling than a foreign
family is with it. Consensus labelling redistributed the classes substantially:
L6 172→87, L4 64→96, L5 47→75. Trained on 4-way consensus, this is worth about
**+0.12 weighted kappa and −0.19 levels of MAE** on identical ground truth
(`results/v07_release/head_to_head2.json`).

**And a warning we are publishing because we nearly got it wrong (H-LABELCEIL).** An
earlier 30-image pilot found a third labeller agreeing with Qwen-8B on 40.0% — no better
than the two Qwen sizes agreed with each other, 42.9% — and the conclusion drawn was
*"more labellers cannot help; the ceiling is physical."* **That conclusion was wrong,
and it was wrong because the sample was too small.** Re-run over 650 events the gap is
16 points the other way. A null result from an underpowered sample is not a result, and
this project nearly stopped spending on relabelling on the strength of 30 images.

Consensus is a **lower-variance** label set, not a **correct** one. It shows the Qwen
labels are *inconsistent* with independent judgements, not that they are *wrong*; two
correlated wrong families can agree with each other. Distinguishing the two needs a
physically-measured anchor, which does not exist.

### Released model: v0.7, 5 seeds, held-out test n = 108 each

Trained on 4-way consensus labels with full labeller coverage
(`data/manifest_v07_weighted.jsonl`), evaluated on the held-out test split. The test
split is 108 of 532 rows (314 train / 110 val / 108 test), all web-harvested, level
distribution **L3 10 / L4 11 / L5 7 / L6 10 / L7 70** — so 64.8% of it is L7.

**Every row below states its denominator and its decode rule.** Rows marked *covered*
are computed over the 85–92 events the model actually decided on; rows marked *all 108*
are computed over the whole test split.

| metric (5 seeds) | value | denominator / decode rule |
|---|---|---|
| weighted kappa, **covered** (the wk1 gate quantity) | **0.838 – 0.860** | covered, CORAL decode |
| weighted kappa, all rows | 0.793 – 0.825 | all 108, CORAL decode |
| **exact match, covered events, CORAL decode** | **80.90 – 84.88%** | covered, CORAL decode |
| **MAE in IDDSI levels, covered events, CORAL decode** | **0.244 – 0.292** | covered, CORAL decode |
| exact match, all 108 events, abstentions scored as misses | 0.667 – 0.704 | all 108, abstain = miss |
| exact match, all 108 events, CORAL decode, abstention bypassed | 0.694 – 0.741 | all 108, CORAL decode, abstention bypassed |
| MAE in IDDSI levels, all 108 events, abstention bypassed | 0.380 – 0.444 | all 108, CORAL decode, abstention bypassed |
| exact match, covered, macro (unweighted per-class) | 0.375 – 0.480 | covered, CORAL decode |
| under-classification, **all-N** (both sides of the fraction) | **0.093 – 0.130** | all 108, CORAL decode |
| dangerous under-classification, covered FNR | 0.035 – 0.056 | covered, CORAL decode |
| coverage | 0.787 – 0.852 | all 108 |
| abstention rate | 0.148 – 0.213 | all 108 |
| ECE (all rows) | 0.024 – 0.077 | all 108 |
| macro-F1, **published definition** | 0.319 – 0.395 | covered, published label set |
| AUGRC (joint selective risk; 0.5 = random) | 0.369 – 0.389 | all 108, **vs the v0.5 labels** |
| **baseline: always answer L7** (the majority test class, 70/108) | **64.81% exact, MAE 0.898** | all 108 |
| baseline: always answer L7, restricted to the covered subset | 70.65 – 76.47% exact, MAE 0.600 – 0.761 | covered |
| baseline: draw a label from the empirical test prior | 45.18% exact | all 108 |
| **wk1 gate (κ ≥ 0.70, covered)** | **5/5 seeds pass** | — |
| **wk2 gate (FNR ≤ 5% at ≤ 30% abstention)** | **not met** | still failing |

Sources: `results/v07_release/v07_seed{17,29,43,71,101}_metrics.json` (per-seed
metrics), `results/v07_release/head_to_head2.json` (AUGRC and the v0.5-label column),
`runs/v07/*/test_predictions.npz` (all-rows and covered-basis recomputation), and the
baselines and label-noise band in `lane_runs/an05_external/benchmark_framing.json` and
`lane_runs/an05_external/label_noise_band.json`. The independent re-derivation for this
card is `lane_runs/rel01_modelcard/card_claims_verification.json`.

**Two rows need reading against the baselines, not against each other.** Against
always-L7 on the same 108 events our covered exact match is only **8.2 – 12.0 points
higher** than the matched covered-subset baseline of 70.65 – 76.47%, and only
**1.9 – 5.6 points** higher on the all-events basis with abstentions scored as misses.
On MAE the signal is real: our covered MAE is **0.356 – 0.412 of** the matched constant
predictor's, about 60% lower. Per-level recall on covered events (seed 43, n = 92
decided) is **L3 0.90, L4 0.50, L5 0.00, L6 0.00, L7 1.00**, and across all five seeds
**L5 and L6 are never emitted as a decision at all** — the emitted census is L3, L4 and
L7 only. This is an L7 detector with a partial L3 detector, not a 5-level classifier.
Source: `analysis/errors_v07/outputs/confusion_seed43.json` and
`structural_seed43.json`.

**A correction to what this card printed before.** The v0.7 exact-match and MAE rows
carried in earlier revisions (0.694 – 0.741 and 0.380 – 0.444) were computed with the
abstention mechanism **bypassed**: all 108 events forced to a CORAL level, including the
16 – 23 the model itself called `unclear`. That is a different quantity from the
covered-events kappa printed three rows above it, and it was once placed next to an
external number without saying so. All four candidate rules were tested against the old
printed ranges; only "CORAL over all 108 events, abstention bypassed" reproduces both
(`benchmark_framing.json → summary.decode_variant_audit`). Both bases are now printed,
labelled. This is the defect recorded in `docs/BENCHMARK_FRAMING.md` §3.

**About ±3 points of our own headline is a property of the yardstick, not the model.**
Flipping the **22 of 108** test labels (20.4%) whose two labellers disagreed to the
other labeller's vote moves covered exact match to **79.78 – 87.06%** and covered MAE to
**0.188 – 0.326** (`lane_runs/an05_external/label_noise_band.json`). Pooled over all
five seeds the covered exact match is 83.52% over 437 covered events, bootstrap 95% CI
[79.86, 86.96] — an interval that says nothing about the label-noise band above.

**Three of those rows are flattered by known metric defects, and they are labelled
here rather than quietly quoted** (all three in `hf/HAZARD_LOG.md`):

- **H-KAPPACOV** — the wk1 gate is a single *unconditional* number (κ ≥ 0.70), but the
  kappa reported against it, `weighted_kappa_covered`, drops every abstention before the
  kappa is taken. Every weighted kappa in this card is a **selective, coverage-filtered
  kappa**; the gate threshold and the measured value describe two different populations.
  The all-rows kappa is published beside it for exactly this reason.
- **H-ABSTBLIND** — `dangerous_underclassification_rate_all` gates its numerator on
  acceptance while dividing by all N, so an abstained under-classification is never
  counted and abstaining more drives it toward zero. Read `fnr_covered`, or the honestly
  named all-rows rate recomputed from the stored predictions, never the `_all` variant.
- **H-MACROF1** — `macro_f1_with_abstentions` builds its label set from the true labels
  only, so the abstention class is never in it: an abstention is charged against its true
  class's recall and counted nowhere on the prediction side. The published macro-F1 is
  therefore **optimistic relative to its name**, and any external comparison against a
  macro-F1 that scores abstention as a class is not like-for-like until both sides are
  recomputed on the same label set. v0.3 has the corrected figure
  (0.211 – 0.263 vs 0.253 – 0.316 published, `results/v03_groupdisjoint/selective_metrics.json`);
  v0.7 does not, so the row above carries the uncorrected number and this warning.

### The most important result in the project: the gate outcome is a property of the yardstick, not the model

The same models that **failed** every published gate against the v0.5 consensus labels
**pass** against the v0.7 consensus labels. Nothing about the models changed between
those two columns — only the labels they were scored against. Both tables below are the
**same checkpoints, the same 108 test events**, scored against the two label sets.

**Basis A — scored against the v0.5 consensus labels**
(`results/v07_release/head_to_head2.json`, n = 108 per seed, 5 seeds each):

| checkpoint | κ (all rows) | exact | MAE | AUGRC | under-class. (all-N) |
|---|---:|---:|---:|---:|---:|
| v0.3 (tier-2 Qwen labels) | 0.315 – 0.399 | 0.370 – 0.454 | 0.907 – 1.019 | 0.335 – 0.407 | 0.167 – 0.287 |
| v0.4 (4-way consensus, 400 train rows) | 0.540 – 0.566 | 0.444 – 0.546 | 0.722 – 0.778 | 0.339 – 0.397 | 0.157 – 0.269 |
| v0.5 (consensus, cleaned 532 corpus, 314 train rows) | 0.498 – 0.547 | 0.426 – 0.500 | 0.778 – 0.870 | 0.387 – 0.461 | 0.185 – 0.333 |
| v0.7 (consensus, FULL labeller coverage) | 0.475 – 0.506 | 0.472 – 0.500 | 0.870 – 0.917 | 0.369 – 0.389 | 0.167 – 0.194 |

**Basis B — the same checkpoints scored against the v0.7 full-coverage consensus
labels** (recomputed this run from `runs/v0X/*/test_predictions.npz`, n = 108 per seed,
5 seeds each):

| checkpoint | κ **covered** (wk1) | κ all rows | exact | MAE | under-class. (all-N) | wk1 |
|---|---:|---:|---:|---:|---:|---|
| v0.4 | **0.736 – 0.808** | 0.687 – 0.720 | 0.509 – 0.620 | 0.528 – 0.639 | 0.185 – 0.306 | 5/5 pass |
| v0.5 | **0.704 – 0.795** | 0.647 – 0.726 | 0.472 – 0.602 | 0.565 – 0.750 | 0.185 – 0.361 | 5/5 pass |
| v0.7 | **0.838 – 0.860** | 0.793 – 0.825 | 0.694 – 0.741 | 0.380 – 0.444 | 0.093 – 0.130 | 5/5 pass |

Five versions of "we have not met the gate" were therefore partly an artefact of
measuring against a label set we ourselves had built and then improved. **A reviewer is
entitled to discount every performance claim in this card, including this one.** This is
hazard **H-YARDSTICK**.

How it was found: by scoring every model against every label set
(`results/v07_release/bias_matrix.json`) after noticing that each model scored best on
the labels it had trained on — the signature of circular measurement. There is no
software fix available. The gate quantity is well-defined; the label set it is measured
against is not independent of the models, and the only non-circular yardstick is
physically-measured ground truth, which does not exist.

**Read the evidence file for what it actually holds (H-YARDEVID).** H-YARDSTICK's own
text cites `results/v07_release/bias_matrix.json` as where it was found. That file does
**not** contain wk1: all six of its cells are quadratic weighted kappa over **all**
held-out rows with the model's raw pre-abstention CORAL prediction — abstentions
included in the denominator — not the coverage-filtered kappa the gate uses. Read as the
gate metric, it *refutes* the sentence it is attached to: its v0.7-labels column puts
v0.5 at 0.671, below the 0.70 gate, i.e. 0/5 rather than 5/5. Reproduced cell-for-cell
in `results/yardstick_audit/bias_matrix_identification.json`. The sentence itself is
correct on wk1, and the table above is that table with its denominators stated.

The one thing that is *not* circular: v0.4 and v0.5 never saw the v0.7 labels, and they
score 0.736–0.808 and 0.704–0.795 against them. Three models trained on three different
label sets all agree more with the v0.7 labels than with the v0.5 ones. That is evidence
the labels *improved*, not evidence that they are *correct*.

### Version history, in one table

| version | what changed | headline | source |
|---|---|---|---|
| v0.1 | CORAL argmax decode bug (H-CORAL) | interior levels unreachable; macro-F1 0.088, coverage 0.70, ECE 0.138, n = 109 | `results/fix1_siglip2_coral_seed17/` |
| v0.2 | corrected manifest (684 unique events, stratified 70/15/15); four metric defects disclosed | macro-F1 **regressed** 0.358 → 0.284–0.320; coverage **collapsed** 0.881 → 0.500–0.618; wk1 fires 5/5 | `results/v02_judgelabels/honest_metrics.json` |
| v0.3 | group-disjoint splits (H-LEAK: 55 near-dup pairs, 7 families, 4 leaked in v0.2); AUGRC added; test set 102 → 142 events | AUGRC **0.242 – 0.296** (0.5 = random); under-classification all-N **0.176 – 0.282**; macro-F1 corrected for abstention 0.211 – 0.263 (published 0.253 – 0.316); all on held-out n = 142 | `results/v03_groupdisjoint/` |
| v0.4 | 4-way consensus labels (H-QWENBIAS) | vs v0.5 labels κ_all 0.540 – 0.566, exact 0.444 – 0.546, MAE 0.722 – 0.778; vs v0.7 labels κ_covered 0.736 – 0.808 | `results/v07_release/head_to_head2.json`, `runs/v04/` |
| v0.5 | cleaned corpus (H-UNUSABLE: 152 of 684 removed) | vs v0.5 labels κ_all 0.498 – 0.547, exact 0.426 – 0.500, MAE 0.778 – 0.870; vs v0.7 labels κ_covered 0.704 – 0.795 | `results/v07_release/head_to_head2.json`, `runs/v05/` |
| v0.6 | *no model change* — only the scoring yardstick (H-YARDSTICK) | the wk1 gate flips 0/5 → 5/5 | `results/v07_release/bias_matrix.json` |
| v0.7 | full-coverage consensus labels (337 of 532 events doubled) | vs v0.5 labels κ_all 0.475 – 0.506; vs v0.7 labels exact 0.694 – 0.741, MAE 0.380 – 0.444, κ_covered 0.838 – 0.860 | `results/v07_release/`, `runs/v07/` |

Every row in that table except the last was, at the time it was published, a **failing**
result. That is the release.

**v0.2 does not classify better; it declines more.** Across its five seeds (held-out
n = 102 each) the model makes **28 – 36 dangerous under-calls**, but *answers* wrongly on
only **2 – 8** of them, because the calibrated abstention intercepts **75.0 – 93.9%** of
its own dangerous calls. That is why v0.2's covered under-classification falls while its
coverage collapses: read it as a property of the abstention policy, not of the
classifier. The metric cannot see an abstention at all (hazard **H-ABSTBLIND**), and
v0.1's 0.333 was measured at coverage 0.881 while v0.2's is measured at ~0.55, so **the
two figures are not the same measurement**. Source:
`results/v02_judgelabels/honest_metrics.json`.

### How this compares to published work — and why the comparison is uninformative

The strongest published single-image result uses **clinically verified** labels
(115 images, 8 levels, IDDSI 0–7) and reports 61.74% exact match (its chosen Qwen
pipeline), 70.43% (GPT-4.1), MAE 0.809, FNR 9.2% —
https://www.frontiersin.org/journals/nutrition/articles/10.3389/fnut.2026.1829703/full
(He, Mung, Kam & Chan, *Frontiers in Nutrition*, 2026-06-29). We did not open
that paper and cannot re-derive any of its numbers: no CEIV-115 images, labels,
or per-image outputs exist in this workspace, and the repo itself notes that CEIV
publishes examples rather than the machine-readable corpus
(`paper/sections/related_work.md:97`). Those figures are restated from our own
citation, not measured here.

**We make no claim of being better, equal, or worse than it.** On the covered
basis we print 80.90 – 84.88% exact and 0.244 – 0.292 MAE against their 61.74%
and 0.809; on the all-events basis with abstentions scored as misses we print
66.67 – 70.37%, which is *below* their GPT-4.1 figure. The full list of reasons
the two are not the same measurement is in `docs/BENCHMARK_FRAMING.md`; the
load-bearing ones:

1. **Label authority.** Theirs are clinician-verified physical tests; ours are
   VLM consensus (108/108 of the test split). Our MAE measures agreement with a
   machine-built yardstick, not error against physical truth. Two Qwen judges
   agree on 31.0% of the same images (n_pairs = 575, `hf/HAZARD_LOG.md` H-NOISE);
   v0.7 cross-family agreement on the doubled sample is 74.5% (251/337,
   `results/v07_release/coverage_meta.json`). A yardstick that is itself ~75%
   consistent cannot certify an 83.5% exact match.
2. **Circularity.** 102 of the 108 test rows carry a judgement from `step5`, one
   of the labellers whose votes define the consensus label, and our model is
   trained on those labels. The self-agreement share is unknown and unmeasured.
3. **Class count.** Theirs spans IDDSI 0–7 (8 levels, liquids included); ours
   L3–L7 (5 levels, no liquids). Exact match gets harder as the class count
   grows, and MAE's attainable range grows with it (max |error| 7 vs 4).
4. **Denominator and abstention semantics.** Their 115 images are all scored; our
   covered basis drops the 16 – 23 events the model itself selected as
   low-confidence, which biases it upward against an always-answers baseline.
5. **Class balance.** 70/108 = 64.8% of our test split is L7, so a constant "L7"
   predictor scores 64.81% — already above their 61.74%. Their class distribution
   is not stated anywhere in this repo, so no baseline-normalised comparison is
   possible without inventing it.
6. **Provenance.** All 108 of our test events are web-harvested CC-licence
   candidates; theirs is clinician-verified.
7. **Model class.** They swept 14 prompted general-purpose VLMs; ours is a
   SigLIP-2 CORAL head fine-tuned and threshold-calibrated on this corpus.
8. **Sample size.** 115 vs 108, and our covered subset is 85 – 92 events; the
   pooled 83.52% carries a bootstrap 95% CI of [79.86, 86.96], which says nothing
   about the label-noise band or the class-balance effect above.
9. **Unverifiable anchor.** None of their numbers can be checked or
   denominator-audited from here.
10. **Our own provenance was wrong until this run** (see the correction above).

The correct conclusion is that this comparison is **uninformative in both
directions**, not that it favours either side.

## Bias, risks and limitations

**1. The labels are the ceiling, and they are machine labels.** No number in this card
is a claim about real-world accuracy. Two independent model families agree on an IDDSI
level from a photograph about three times in four; the "is there food" question is
answered reliably (**96.7%** cross-family, 493/510) while the 5-level question never is
(**85.5%**, 436/510). Source: `results/v05_release/labelability_sweep.json`.

**2. The label noise is irreducible from photographs.** L3/L4 is a viscosity boundary
and L4/L5 is an absolute-particle-size boundary; neither is recoverable from a still
image. More machine labelling will not fix it. Corroborated by the literature: a panel
of 9 frontier LLMs from 7 families reaches an effective independence of only **24.2%**
(<https://ar5iv.arxiv.org/html/2605.29800>), and Dawid–Skene-family label models need
**three to five** annotators before their likelihood is identifiable at all
(<https://arxiv.org/pdf/1712.04577>). This project has two correlated labellers plus
later independent ones — below that threshold for the earliest versions.

**3. The dangerous error is systematic, not random.** Predicting a SAFER texture than
truth (under-classification) is the direction that matters, and the label noise is
biased in exactly that direction (see the cross-tab above). The first released run
measured **33%** dangerous under-classification on 109 held-out events
(`results/fix1_siglip2_coral_seed17/metrics.json`); the current all-rows figure is
**0.093 – 0.130**, but it is measured against the same noisy labels. An L7 catch-all
sink that defaulted unmeasurable rows to *regular food* — the least restrictive level —
has been fixed with scale anchors (hazard **H-L7SINK**), but the residual direction of
bias is the one to worry about.

**4. Abstention makes the headline metric lie, and we say so.** `dangerous_under-
classification_rate_all` gates its numerator on acceptance while dividing by all N, so
an abstained under-classification is never counted and abstaining more drives it toward
zero (hazard **H-ABSTBLIND**). v0.3 publishes the correctly-named all-N rate and adds
**AUGRC** — the joint unconditional selective risk integrated over thresholds (Traub et
al., NeurIPS 2024, <https://arxiv.org/abs/2407.01032>), which cannot be gamed by
abstaining because coverage sits inside the integrand (hazard **H-AUGRC**). All five v0.7
seeds land at 0.369–0.389 against the 0.5 random reference, so the model beats random at
*every* threshold rather than only at one chosen operating point. Note the honest
provenance: AUGRC is computed post hoc from the stored probabilities by a documented
second implementation, not by `trainer/eval/metrics.py`, so the library metric and the
published metric are still two code paths and can drift.

**5. A metric we chose after seeing results is still a metric we chose.** We are aware
the wk1 gate is contested for ordinal tasks (hazard **H-GATE**): Delgado & Tibau (2019),
"Why Cohen's Kappa should be avoided as a performance measure in ordinal classification",
*Pattern Recognition* 105:107192, argues precisely against it — the metric ranks ordinal
models wrongly under class imbalance, which is exactly our condition (L7 is 54% of the
v0.3 tier-2 label set, 373 of 684 rows in `data/manifest_v03.jsonl`). We are **not**
changing a pre-registered gate after seeing results, because doing so would make every
number in this release indefensible. We report the objection and leave the gate standing.

**6. Domain bias.** The corpus is web-harvested food photography: English-language
sources, Western plating conventions, restaurant and stock-style presentation, single
objects, no scale fiducial. Performance on home-cooked, shared-plate, non-Western, or
low-light food photography is unmeasured and expected to be worse.

**7. No physically-tested ground truth exists.** The remedy path — 300 desk specimens
(60 base foods × 5 levels), one operator, protocol-following, test-filmed — is fully
specified in `docs/CAPTURE_PROTOCOL.md` and **has never been run**.

**8. The split's leakage control is configured but close to vacuous (H-GROUP, H-LEAK).**
On the v0.7 manifest, 501 groups cover 532 rows and no `group_id` appears in more than
one split — but **495 of those 501 groups (98.8%) are singletons**, so the control still
separates almost nothing, and 6 food names sit on both sides of the split boundary. No
image-level near-duplicate (dHash ≤ 6 **and** pHash ≤ 8, recomputed from the JPEGs over
141,246 pairs), byte duplicate, source page, Nutrition5k dish id or recipe id crosses
the boundary. Read every gate result in this card as *optimistic*, not clean. Source:
`HGROUP_VERIFICATION.md`.

## Why the weights are withheld

The photo-classification head fails the project's own pre-registered safety gates.
Dangerous under-classification (predicting a SAFER texture than truth) was measured at
**33%** on 109 held-out events in the CORAL-fix run
(`results/fix1_siglip2_coral_seed17/metrics.json`), and the current all-rows figure of
**0.093 – 0.130** is measured against labels that are themselves only 31%-consistent.
A downloadable classifier would be used for verdicts whatever a card says, and a
disclaimer does not cure a claim. Withholding the weights is the mitigation.

**Pre-registered gates and which fired**

- **wk1** — weighted kappa ≥ 0.70, else kill photo-classification.
- **wk2** — external macro-F1 ≥ 0.75 **plus** dangerous under-classification ≤ 5% at
  ≤ 30% abstention.

Against the v0.5 consensus labels the wk1 kill rule fires on every seed (best κ 0.566).
Against the v0.7 consensus labels it passes 5/5 — and the difference between those two
statements is the label set, not the model (see
[The most important result](#the-most-important-result-in-the-project-the-gate-outcome-is-a-property-of-the-yardstick-not-the-model)).
The wk2 calibration gate is not met. **The weights stay withheld.** Nothing in any
version reopens that decision.

## Why the dataset is withheld

The harvested dataset is unpublishable on licence grounds. Of 587 unique events exactly
**2** would ship cleanly. **592** rows carry unverified open-images/openverse licences,
**7** are CC BY-SA against the BY-NC-SA main release, and **109** have unresolved
redistribution status. Fail-closed licence verification is the gate, with a regression
test under `ops/`. Source: `hf/HAZARD_LOG.md` (hazard H-LICENCE).

Note the disclosed inconsistency, which we are not hiding: those counts do not
reconcile with the 544 grouped events or the 704 labelled rows, because the grouped
population is not the labelled population and neither number certifies the other
(`hf/HAZARD_LOG.md`, hazard H-GROUP).

## Training data

**Not published.** The training corpus was web-harvested food photography under
CC-licence candidates, weak-labelled by two tiers of VLM judges and later re-labelled by
independent model families. The data engine is stated exactly in `hf/HAZARD_LOG.md`
(hazard H-PROV): **713 harvested real CC-licence-candidate images in 3 sets**
(realweak_pilot 117+12, harvest_d1 400, harvest_d1b 184), **701 raw plus 6 validated
cloud-generated**. The judged corpora denominators (400 / 184 / 126) are corroborated by
the tier-2 judging progress logs on the training box. After deduplication and cleaning
the supervised population was **684 unique events**, then **532**.

| artefact | rows | unique events | level mix (L3/L4/L5/L6/L7) | train rows | source |
|---|---:|---:|---|---:|---|
| tier-2 Qwen labels (v0.3) | 684 | 684 | 28 / 64 / 47 / 172 / 373 | 400 | `data/manifest_v03.jsonl` |
| 4-way consensus labels (v0.4) | 684 | 684 | 26 / 96 / 75 / 87 / 400 | 400 | `data/manifest_v04_consensus.jsonl` |
| cleaned corpus (v0.5) | 532 | 532 | 24 / 66 / 65 / 72 / 305 | 314 | `data/manifest_v05_clean.jsonl` |
| full-coverage consensus (v0.7) | 532 | 532 | 52 / 56 / 26 / 54 / 344 | 314 | `data/manifest_v07_weighted.jsonl` |

What the level-mix columns show: consensus labelling moved a great deal of mass out of
L6 (172 → 87) and into L4/L5 (64 → 96, 47 → 75), and the v0.7 re-labelling moved it
again (L3 24 → 52, L5 65 → 26). **A label set that changes that much under
re-labelling is not ground truth.**

Splits are group-disjoint: perceptual deduplication (dHash ≤ 6 **and** pHash ≤ 8) over
all 684 images found **55 near-duplicate pairs** that `event_id` never caught, forming
**7 genuine visual families**, of which **4 spanned train, val and test simultaneously**
in the v0.2 row-wise split (hazard **H-LEAK**). v0.3 onward assigns whole groups to
splits (**650 groups over 684 rows**) and asserts group-disjointness and per-level
coverage rather than assuming them. Source:
`results/v03_groupdisjoint/dedup_report.json`, and the group count in
`data/siglip2_v07.yaml`.

**152 of 684 events (22.2%) were unusable** — empty vessels, diagrams, no discernible
food mass — and every number published before v0.5 was measured with them in the test
set (`results/v05_release/unusable_event_ids.txt`, 152 lines; hazard **H-UNUSABLE**).

**Which of the two unusable percentages to quote.** The corpus-wide figure is the one
that survives contact with the data: all 152 ids are present in the 684-event pre-clean
manifests (`data/manifest_v04_consensus.jsonl`) and none survive in the 532-row cleaned
manifest (`data/manifest_v07_weighted.jsonl`), i.e. 684 − 152 = 532 rows were actually
removed. The other figure is a *conditional* rate over the both-judged subset, and it is
only valid if all 152 unusable events fall inside that subset — the aggregate sweep
output does not record which ones do, and its own counts
(683 + 512 − 510 = 685) cannot all be subsets of the same 684 events. So:

- **152 / 684 = 22.2%** — the share of the whole corpus. Use this for any claim about the
  corpus.
- **152 / 510 = 29.8%** — the share of the both-judged subset. Use this only for a claim
  explicitly conditioned on that subset, and say so.

Reproduce both with `python3 tools/denominator_audit.py`.

**v0.7 changed only the labels.** A coverage-closure run gave a second independent
judgement to the 314 events that previously carried only one, so **337 of the 532
cleaned events now carry two genuine independent judgements** (251 unanimous, 86 split)
instead of an unverifiable single call; 195 still have one and are kept at weight 1.
Cross-family agreement is stable and reproducible at **~74.5%** — 251/337 = 74.5% on the
doubled sample and 166/223 = 74.4% on the subset that was already doubled. Two
independent model families agree on an IDDSI level from a photograph about three times
in four. Sources: `results/v07_release/coverage_meta.json` and `data/siglip2_v07.yaml`.

Two attempts to improve training this way are recorded **because they failed**: dropping
every disputed event kept only 31% of the corpus and destroyed the middle classes
(L4 66→7, L5 65→4, L6 72→5); weighting by agreement produced a replication factor of 1
for every event, because agreement is all-or-nothing inside a level rather than graded.
v0.7's training recipe is byte-identical to every prior version.

## The flow-test grader — the working deliverable

This is the part that **works**, and it is the genuinely useful artifact of the release.
It is a deterministic CV pipeline for the 10 ml / 10 s IDDSI syringe flow test from
smartphone video, grading **L0–L4** — including the **thickened liquid** end of the
scale. **MAE 0.065 ml on a 100-video bench**, of which 16 were graded and 84 abstained.
Source: `flowtest/flowtest/bench/results_n100/bench_summary.json` — note its own scope
line: *"synthetic pipeline-development benchmark only; not real-video validation and not
a safety benchmark."*

This is **distinct from photo classification. A still photograph is never used to grade
liquid flow.** If you want the working piece, take this one.

## How to reproduce

```bash
# 1. build the corrected, group-disjoint manifest (recipe identical across versions)
docker exec -w /work/lanes2/rel01_modelcard/trainer iddsi-train \
  python -m train.cli --config configs/siglip2_v07.yaml

# 2. re-derive every number in this card from the raw artefacts
docker exec -w /work/lanes2/rel01_modelcard -e LANE_RUNS=/work/lane_runs/rel01_modelcard \
  iddsi-train python hf/verify_card_claims.py
```

Repo structure:

| path | contents |
|---|---|
| `trainer/` | SigLIP-2 CORAL head training, eval metrics, calibrated abstention |
| `results/` | per-version eval artefacts, including every per-seed metrics file |
| `flowtest/` | the working smartphone-video grader for the 10 ml / 10 s IDDSI Flow Test |
| `harvest/` | CC-licence-candidate image harvesting, licence tracking, two-tier VLM weak labelling |
| `dataschema/` | manifests, dataset card, datasheet wording |
| `synth/` | cloud-generated development images (never eval ground truth) |
| `crowd/` | capture app used for guided capture work |
| `ops/` | release checks and runbooks, including the fail-closed licence verification |
| `docs/` | capture protocol, shot list, base food list, regulatory posture, data governance, benchmark framing |
| `hf/` | this card, the metadata specification, the hazard log, and the claim verifier |
| `paper/` | write-up of the methods, the negative finding, and the remedy path |

## Provenance of every number in this card

House rule on this project: *every claim must be measured by the author and
reproducible from a named file.* `hf/verify_card_claims.py` re-derives each headline
quantity from the primary artefacts — the per-seed metrics JSON, the saved
`test_predictions.npz` per seed, the v0.7 consensus manifest, the inter-judge
cross-tab, and the flow-test bench summary — rather than trusting any prose. It also
audits the card for the target search phrases and for the presence of every disclaimer,
gate and hazard entry.

Run output: `lane_runs/rel01_modelcard/card_claims_verification.json` and
`card_claims_verification.md`.

Verified in that run, against the ranges quoted above:

| quantity | range | source artefact |
|---|---|---|
| v0.7 weighted kappa (covered) | 0.838 – 0.860 | `results/v07_release/v07_seed*_metrics.json` |
| v0.7 weighted kappa (all rows) | 0.793 – 0.825 | `runs/v07/*/test_predictions.npz` |
| v0.7 exact match, all 108, abstention bypassed | 0.694 – 0.741 | `runs/v07/*/test_predictions.npz` |
| v0.7 MAE in levels, all 108, abstention bypassed | 0.380 – 0.444 | `runs/v07/*/test_predictions.npz` |
| v0.7 under-classification, all rows | 0.093 – 0.130 | `runs/v07/*/test_predictions.npz` |
| v0.7 coverage / abstention rate / ECE (all) | 0.787 – 0.852 / 0.148 – 0.213 / 0.024 – 0.077 | `results/v07_release/v07_seed*_metrics.json` |
| v0.7 macro-F1 (published definition) / covered FNR | 0.319 – 0.395 / 0.035 – 0.056 | `results/v07_release/v07_seed*_metrics.json` |
| v0.4 / v0.5 kappa (covered) vs v0.7 labels | 0.736 – 0.808 / 0.704 – 0.795 | `runs/v04`, `runs/v05` |
| v0.4 / v0.5 / v0.7 kappa (all) vs v0.7 labels | 0.687 – 0.720 / 0.647 – 0.726 / 0.793 – 0.825 | `runs/v04`, `runs/v05`, `runs/v07` |
| v0.4 / v0.5 exact, MAE, under-classification vs v0.7 labels | 0.509 – 0.620, 0.528 – 0.639, 0.185 – 0.306 / 0.472 – 0.602, 0.565 – 0.750, 0.185 – 0.361 | `runs/v04`, `runs/v05` |
| v0.3 / v0.4 / v0.5 / v0.7 exact, MAE, κ, AUGRC, under-class. vs v0.5 labels | see the Basis A table above | `results/v07_release/head_to_head2.json` |
| inter-judge exact agreement, n = 575 | 0.3096 | `results/regen_20260916/inter_judge_20260916.md` cross-tab |
| κ unweighted / κ quadratic-weighted (recomputed) | 0.1062 / 0.2766 | same cross-tab |
| cross-family "is there food" agreement | 493/510 = 96.7% | `results/v05_release/labelability_sweep.json` |
| unusable events | 152 of 684 = 22.2% (152/510 = 29.8% conditional) | `results/v05_release/unusable_event_ids.txt` |
| coverage closure | 337 events ≥ 2 judgements, 251 unanimous (74.5%) | `results/v07_release/coverage_meta.json` |
| flow-test grader MAE | 0.0651 ml, n = 100 (16 graded) | `flowtest/flowtest/bench/results_n100/bench_summary.json` |
| CORAL decode bug, before / after | macro-F1 0.088 → 0.358, n = 109 | `results/fix1_siglip2_coral_seed17/metrics.json` |
| 33% dangerous under-classification | 0.3333 covered FNR, n = 109 | `results/fix1_siglip2_coral_seed17/metrics.json` |
| per-level recall on covered events (seed 43) | L3 0.90 / L4 0.50 / L5 0.00 / L6 0.00 / L7 1.00 | `analysis/errors_v07/outputs/confusion_seed43.json` |
| L5 and L6 never emitted as a decision | 5/5 seeds | `analysis/errors_v07/outputs/structural_seed43.json` |
| always-L7 baseline, all 108 / covered subset | 64.81% exact, MAE 0.898 / 70.65 – 76.47% exact, MAE 0.600 – 0.761 | `lane_runs/an05_external/benchmark_framing.json`, recomputed this run |
| label-noise band (22 of 108 flipped) | exact 79.78 – 87.06%, MAE 0.188 – 0.326 | `lane_runs/an05_external/label_noise_band.json` |
| step5 judgements among the 108 test rows | 102 of 108 | `data/manifest_v07_weighted.jsonl` |
| dataset licence counts (587 / 592 / 7 / 109) | — | `hf/HAZARD_LOG.md` (H-LICENCE) |

Three numbers that appeared in earlier revisions of this card are **not** in it because no
file in the repository backs them, and the house rule is that every claim must be
reproducible from a named file:

1. the per-corpus harvest tallies as a total of 713 (superseded by the
   manifest-verifiable counts in the table above);
2. the v0.1 manifest's 704 labelled rows / 20 duplicate `event_id` values (the v0.1
   manifest is not in the repository; the 544-grouped / 704-labelled /
   587-licence-accounted discrepancy they produce is disclosed under hazard H-GROUP in
   `hf/HAZARD_LOG.md`);
3. the 30-image pilot's *"100% of disagreements with Qwen-8B were exactly ±1 IDDSI level
   (19 of 19)"* and *"78.6% of items had at least two of three labellers agreeing"*.
   These appear in no artefact in this repository — not in
   `results/regen_20260916/`, not in `results/v05_release/`, and not in the H-LABELCEIL
   hazard entry. The pilot's 40.0% / 42.9% agreement figures **are** carried in
   H-LABELCEIL and in this card, because the hazard log states them; the two derived
   figures above are not, and are therefore not restated here.

Where a figure in this card is quoted from the hazard log rather than recomputed from a
data artefact, the hazard log entry is named as the source.

## Honest claims and never-claims

**Honest claims**

- The ambition was *"first publicly downloadable IDDSI-specific smartphone dataset +
  fine-tuned model under explicit reuse licences"*. **v0 does NOT meet it**: the dataset
  is withheld on licence grounds and the weights are withheld on safety gates.
- The flow-test grader claim **does** stand: *"first open-source smartphone grader for
  the 10 ml IDDSI Flow Test"*.

**Never-claims**

- **No novelty claim over prior IDDSI vision work.** Prior art: PolyU ViscoCam (2021),
  HKUST OptiTexture (91.96% with an optical sensor), EdUHK *Frontiers in Nutrition*
  (2026, 61.74% with clinician-verified labels), Safe Plate (AU).
- **No safe or unsafe verdicts**, no dietary, clinical or patient claims.
- **Not IDDSI-endorsed.** Research plus culinary-education use only.

## Citation, licence, contact

**Licence:** Apache-2.0 (code). No model weights are published, so no weights licence
is granted.

```bibtex
@misc{iddsiopen_v0,
  title  = {IDDSI-Open research preview: code, eval results and a negative finding
            on reading IDDSI texture levels from food photographs},
  author = {IDDSI-Open contributors},
  year   = {2026},
  note   = {Hugging Face: lingualeap/iddsi-open-siglip2-v0. Release shape decided
            2026-09-07. Model weights and harvested image dataset withheld for the
            safety and licence reasons stated in the model card.},
  url    = {https://huggingface.co/lingualeap/iddsi-open-siglip2-v0}
}
```

**Model card authors:** the IDDSI-Open contributors. **Contact:** open an issue or
discussion on the repository.

**If you use this work, please cite the negative finding with its denominator.** The
honest summary of this release is one sentence: *five versions of a SigLIP-2 CORAL head
trained on machine labels could not be reliably failed by its own pre-registered gates,
because the gates were measured against a yardstick the project built itself — and the
photographs, not the models, are the bottleneck.*
