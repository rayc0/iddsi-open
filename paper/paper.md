---
title: "IDDSI-Open v0: Code, Eval Results, and Smartphone Flow-Test Video Grader - Photo Classification Killed at Its Pre-registered Gates; Dataset and Weights Withheld"
subtitle: "Technical note - v0 research preview (release shape locked 2026-09-07)"
author:
  - "TODO: Author list (LinguaLeap Tech Limited and contributors)"
date: "7 September 2026"
abstract: |
  IDDSI-Open v0 ships code plus eval results plus a deterministic
  computer-vision grader for the 10 mL / 10 s IDDSI Flow Test, plus a measured
  negative finding on photo classification. The model weights and the harvested
  image dataset are WITHHELD - a deliberate, stated decision. The dataset is
  withheld on licence grounds: of 587 unique events exactly 2 would ship
  cleanly (592 rows carry unverified open-images/openverse licences, 7 are
  CC BY-SA against the BY-NC-SA main release, 109 have unresolved
  redistribution status). The weights are withheld because the
  photo-classification head failed its pre-registered gates - wk1 quadratic
  weighted kappa on covered rows >= 0.70 (measured 0.4227, held-out n = 109,
  96 of 109 events covered, label set: the Qwen tier-2 judge labels), wk2
  macro-F1 >= 0.75 (measured 0.358) with
  dangerous under-classification at 33% against a <= 5% gate - so the wk1 kill
  rule fired and photo-classification is dead as a product capability.
  Label set and population for that kappa (H-YARDSTICK, H-YARDEVID): the 0.4227
  was measured against the Qwen tier-2 judge labels over covered rows only, and
  the v0.4/v0.5 numbers were measured against the v0.5 consensus; on both of
  those label sets the models fail the wk1>=0.70 gate. The same three
  checkpoints measured against the v0.7 full-coverage consensus clear that same
  covered-rows metric in 5/5 seeds each (v0.4 0.736-0.808, v0.5 0.704-0.795,
  v0.7 0.838-0.860; held-out n = 108 events, coverage 78.7-85.2% so 85-92 of
  108 events carry a decision). The pass verdict is metric-dependent: on the
  all-rows kappa, which keeps abstentions in the denominator, the v0.5 model
  averages 0.671 and clears 0.70 on only 1 of 5 seeds. The
  headline negative finding: two VLM judges (Qwen3-VL-2B vs Qwen3-VL-8B) agree
  on only **31.0%** of the same images (n_pairs = 575, full tier-2 sample,
  regen 2026-09-16; Cohen's kappa unweighted 0.1062, quadratic-weighted 0.2766
  - still essentially chance for 5 classes, both computed over all 575 judged
  pairs with no coverage filter because neither judge abstains), with a
  systematic directional
  bias toward under-predicting difficulty (Section 3.3) - because middle
  IDDSI levels (L4/L5/L6) need absolute size/flow information a scale-free web
  photo does not carry; every label-derived metric is therefore measured
  against noise. Label set for those inter-judge kappas (H-YARDSTICK): they are
  agreement between the two VLM judges who produced the label set, so they
  characterise the yardstick itself rather than a model scored against one.
  A CORAL ordinal-decode bug that made interior levels
  unreachable was found and fixed (held-out n = 109: macro-F1 0.088 -> 0.358,
  coverage 0.70 -> 0.88, ECE 0.138 -> 0.091, temperature 2.83 -> 0.93;
  quadratic weighted kappa on covered rows flat at 0.4227). The v0.7
  evaluation (held-out n = 108 of the 532-row cleaned corpus, 5 seeds) reaches
  quadratic weighted kappa on covered rows 0.838-0.860 and exact level match on
  covered events 80.9-84.9%, but **L5 and L6 are never emitted as a decision in
  any of the five seeds**, so the head is an L7 detector with a partial L3
  detector rather than a five-level classifier (Sections 3.6 and 6). The
  flow-test grader works: MAE 0.065 ml on a
  100-video bench (L0-L4). Planned next step, not done: physically-tested desk
  specimens (60 base foods x 5 levels = 300, single-operator,
  protocol-following, test-filmed, with a filmed re-test subsample for
  intra-rater reliability). This is a research preview. Research demonstration
  for culinary education. Estimates visual similarity to IDDSI descriptors;
  does not perform official IDDSI tests, assess swallowing, determine
  suitability for any person, or decide whether food is safe to consume. The
  ambition was "the first publicly downloadable IDDSI-specific
  smartphone-food dataset and fine-tuned model released under explicit reuse
  licences, with physically tested Chinese/Cantonese soft-meal labels"; v0
  does NOT meet it, and the ONE claim that stands is "the first open-source
  smartphone grader for the 10 ml IDDSI Flow Test". We claim neither clinical
  validity nor any ability to determine swallowing safety from a photograph.
keywords:
  - IDDSI
  - dysphagia
  - texture-modified-food
  - food-texture-classification
  - computer-vision
  - open-dataset
  - flow-test
lang: en
bibliography: references.bib
link-citations: true
---

<!--
W22 methods note (v0.1 regen 2026-09-16; honest release
card hf/model_README.md is the authority): classifier and flow-grader result
cells now report the locked v0 numbers (CORAL decode fix, pre-registered
gates, 31.0% exact inter-judge agreement on n=575 with kappa unweighted
0.1062 / quadratic 0.2766, under-prediction finding, flow MAE 0.065 ml on the
100-video bench). Population for those inter-judge kappas: all 575 judged
pairs, no coverage filter, because neither judge abstains. Label set for
those inter-judge kappas (H-YARDSTICK): they are
agreement between the two VLM judges who produced the label set, so they
characterise the yardstick itself; the model kappas in this note were measured
against the Qwen tier-2 judge labels over covered rows (0.4227, held-out
n = 109) and the v0.4/v0.5 numbers against the v0.5
consensus (both fail wk1>=0.70), while the same models measured against the
v0.7 full-coverage consensus clear that same covered-rows metric in 5/5 seeds
each (v0.4 0.736-0.808, v0.5 0.704-0.795, v0.7 0.838-0.860, held-out n = 108)
- and on the all-rows kappa the v0.5 model does not clear it (mean 0.671,
1 of 5 seeds), so the pass verdict is metric-dependent as well as
yardstick-dependent (H-YARDEVID).
RD-SLP inter-rater baseline, subgroup analyses, and
bootstrap CIs were not measured in v0. The 0/24 generator-QA pilot is the
sole reported development-pilot outcome. Frozen boundary wording comes from
the project's frozen source files (r5_prior.md, r5_reg.md,
dataschema/README.md, dataschema/docs/CAPTURE_PROTOCOL.md, synth/README.md,
crowd/README.md, harvest/README.md, RUNBOARD.md).
-->

# 1. Introduction

Dysphagia-safe food and liquid preparation depends on the IDDSI framework,
whose levels are defined by physical tests (flow test, fork pressure, fork
drip, spoon tilt), not by appearance. Existing automated approaches (Section 2)
infer levels from images or video with accuracies ranging from 61.74% exact
match (EdUHK, CEIV-115) to 96.5% controlled accuracy (ViscoCam), but none
publishes a downloadable IDDSI-specific dataset plus fine-tuned weights under
an explicit reuse licence, and none pairs smartphone media with physically
tested labels for Chinese/Cantonese soft meals.

IDDSI-Open v0 contributes:

1. a frozen capture protocol and event schema pairing guided smartphone media
   (45° plate photo + 8-12 s fork-press / spoon-tilt / fork-drip clips; liquid
   side-view 10 mL / 10 s syringe flow videos) with contemporaneous physical
   tests performed independently by a registered dietitian (RD) and a
   speech-language pathologist (SLP), with third-person adjudication of
   disagreements - shipped as code and protocol, not as a downloadable dataset;
2. code and eval results for the photo-classification path (SigLIP-2 CORAL
   head with calibrated abstention) - the model weights are WITHHELD because
   the head failed its pre-registered gates, and the harvested dataset is
   WITHHELD on licence grounds; the result is a measured negative finding,
   not a product capability;
3. an open-source deterministic CV grader for the IDDSI Flow Test (10 mL /
   10 s), claiming "first open-source smartphone grader for the 10 ml/10-second
   IDDSI Flow Test" - explicitly **not** "first video grader";
4. mandatory abstention: the system must answer "unclear - perform physical
   test" rather than guess.

**Contribution claim (v0 truth; release shape locked 2026-09-07 - the honest
release card `hf/model_README.md` is the authority):**

> "The ambition was 'the first publicly downloadable IDDSI-specific
> smartphone-food dataset and fine-tuned model released under explicit reuse
> licences, with physically tested Chinese/Cantonese soft-meal labels'; v0
> does NOT meet it - the harvested dataset is withheld on licence grounds (of
> 587 unique events exactly 2 would ship cleanly; 592 rows carry unverified
> open-images/openverse licences, 7 are CC BY-SA against the BY-NC-SA main
> release, 109 have unresolved redistribution status) and the model weights
> are withheld because the photo-classification head failed its
> pre-registered gates (wk1 quadratic weighted kappa on covered rows >= 0.70
> measured 0.4227 on held-out n = 109 against the Qwen tier-2 judge labels;
> wk2
> macro-F1 >= 0.75 measured 0.358, with dangerous under-classification at 33%
> against a <= 5% gate; the wk1 kill rule fired and photo-classification is
> dead as a product capability). Label set and population for that kappa
> (H-YARDSTICK, H-YARDEVID): covered rows only, 96 of 109 events. The
> v0.4/v0.5 numbers were measured against the v0.5 consensus and also fail
> wk1>=0.70; the same three checkpoints measured against the v0.7
> full-coverage consensus clear that same covered-rows metric in 5/5 seeds each
> (v0.4 0.736-0.808, v0.5 0.704-0.795, v0.7 0.838-0.860, held-out n = 108),
> while on the all-rows kappa the v0.5 model averages 0.671 and clears 0.70 on
> only 1 of 5 seeds - the pass verdict is a property of the metric as well as
> of the yardstick. The ONE claim that stands is 'the first
> open-source smartphone grader for the 10 ml IDDSI Flow Test'."

We never claim: first IDDSI AI; first image classifier; first
dataset/benchmark; clinically validated; diagnostic; medical-grade; or able to
determine swallowing safety from a photograph (frozen never-claim list,
r5_prior.md).

# 2. Related work

Five direct prior works frame the landscape (as stated in r5_prior.md):

- **ViscoCam (PolyU, 2021).** Smartphone video plus motion sensing to classify
  three IDDSI liquid levels - 96.5% controlled accuracy, but only >81% under
  extreme conditions [@viscocam]. Exposes neither dataset nor weights.
- **OptiTexture (HKUST).** Combines RGB with a <\$20 light-scattering sensor:
  91.96% on 112 L3-L6 samples; its vision-only baseline was 69.64%
  [@optitexture]. Exposes neither dataset nor weights.
- **EdUHK (Frontiers in Nutrition, 2026).** Single-image Qwen pipeline
  achieved only 61.74% exact match on CEIV-115; open VLMs missed every liquid
  hazard [@eduhk_frontiers]. CEIV publishes examples, not the full
  machine-readable corpus.
- **Automated photogrammetric syringe test (2019).** Automated
  video/photogrammetric measurement of the official syringe test appeared in
  2019 [@syringe2019]. (This is why we claim "first open-source smartphone
  grader for the 10 ml/10-second IDDSI Flow Test," not "first video grader.")
- **Safe Plate (Australia, proprietary).** Advertises photo-and-video IDDSI
  verification [@safeplate]. Proprietary; no published dataset or weights.

Also relevant context from r5_prior.md: the official IDDSI iOS/Android app is
a descriptor/test-reference library, not a classifier; Viscgo hardware,
CF-200N texture analysers, and menu-software vendors (MealSuite/Dietech;
Kewpie/UDF, Pulmuone, Hyundai Green Food, apetito/winVitalis, Hormel/Lyons,
SimplyThick) classify or test known products rather than infer an unknown
plate. No independent mainland-Chinese plate-photo classifier was found. Open
FoodSense/PlateInsight resources concern generic perceived texture, not
IDDSI.

# 3. Methods

This section specifies the frozen study design and the implementation as
evaluated in v0. Classifier and flow-grader results are reported in Sections
3.3-3.5; v0 ships no downloadable dataset and no model weights.

## 3.1 Capture protocol and reference labels

The unit of observation is one independently prepared recipe batch or one
freshly prepared liquid test. Stills, clips, retakes, and extracted frames
from that preparation remain attached to one `event_id`; they are not treated
as independent samples. Kitchen staff or interns capture media with a native
phone camera, filters and portrait effects disabled where possible. The kit
includes a standard metal fork, spoon, timer, and a non-food-contact reference
card carrying a verified 15 mm feature and the event ID. Liquid events use the
study-approved 10 mL syringe with a 61.5 mm calibrated barrel. The food and
liquid procedures follow the current official physical-test instructions
[@iddsi_framework].

Food events at declared L3-L7 include a whole-plate still from approximately
45 degrees and continuous 8-12 s guided clips. L3 and L4 require spoon-tilt
and fork-drip views; L5 requires spoon-tilt and fork-press; L6 and L7 require
fork-press. The clip begins before utensil contact and continues through
release, drip, deformation, or rebound. Fork-press capture includes thumbnail
blanching as a repeatable visual endpoint, not as an inferred force
measurement. The entire utensil, sample, and scale reference remain visible.
Mixed-component meals require an explicit study decision and are not silently
assigned a level from a single component.

Liquid events at declared L0-L4 use one uninterrupted side-view Flow Test
video. The full barrel, outlet, markings, and collection vessel are shown
against a contrasting background; the syringe begins at exactly 10 mL at the
recorded serving temperature. Recording begins before release, retains a
visible or audible release cue, keeps the syringe vertical and the meniscus
visible, and continues for at least one second after the 10 s reading. Wrong
starting volume, uncertain release time, occlusion, camera movement, or an
unreadable meniscus triggers a documented retake rather than an inferred
result.

Each capture session records recipe, batch, kitchen, phone, operator,
declared level, cuisine, serving temperature, lighting, plate, consent, and
protocol-deviation metadata. Faces, patient identifiers, staff names,
medication, order slips, identifying speech, and geolocation metadata are
excluded. Recipe, batch, and kitchen groups are prospectively separated
across `train`, `val`, and locked `external_test` splits.

For release-eligible real events, an RD and an SLP independently perform and
record the applicable physical tests on the same batch at the same serving
condition without seeing one another's result. A qualified third reviewer
adjudicates disagreements. Only a final numeric adjudicated level enters the
physically tested release; unresolved, invalid, pending, or protocol-deviating
events remain quarantined or excluded. Tests were performed by LinguaLeap and
participating assessors; they are not IDDSI certification.

## 3.2 Data sources and provenance controls

The canonical interchange is `events.jsonl` plus a `media/` tree, with one
event per row and safe relative media paths. The schema records source,
split, recipe/batch/kitchen groups, capture conditions, consent and privacy
checks, media hashes and roles, declared/weak/user levels, independent
physical-test levels, adjudication, and release eligibility. Four source
classes are kept distinct:

1. **Synthetic (`source="synthetic"`).** Development-only photographs are
   prompted toward visible L3-L7 descriptors for pipeline development and
   representation-aware pretraining. The prompt target is stored only as
   `level_declared`; all physical-test fields are null. Rows may enter
   `train` or `val`, never `external_test`, and cannot be counted as physical
   ground truth, achieved coverage, validation evidence, or a product claim.
2. **Real-weak (`source="real-weak"`).** Licence-eligible third-party food
   images support pipeline development. Physical-test fields remain null;
   model-derived weak labels, when present, remain explicitly weak. These rows
   are restricted to `train` and retain a mandatory attribution ledger.
3. **Real-crowd (`source="real-crowd"`).** `level_user` records a contributor's
   statement, not a performed physical test. These events may support
   robustness or separately reported weak-supervision work, but they do not
   enter a physically tested benchmark. They remain release-ineligible while
   privacy screening is incomplete.
4. **Real-tested (`source="real"`).** Release eligibility requires the
   contemporaneous independent assessments and adjudication described in
   Section 3.1. Declared recipe levels, visual labels, synthetic targets,
   crowd statements, and model predictions are never substitutes.

Counts by source, level, and split are as recorded in the shipped manifests;
the harvested image dataset itself is WITHHELD in v0 on licence grounds and
no downloadable dataset ships. Release-licence audit (v0): of 587 unique
events exactly 2 would ship cleanly - 592 rows carry unverified
open-images/openverse licences, 7 are CC BY-SA against the BY-NC-SA main
release, and 109 have unresolved redistribution status. Code ships under
Apache-2.0; model weights are withheld on safety-gate grounds (Section 3.3).
The project is not the official IDDSI website and is not IDDSI-endorsed.

**Corpus denominators, stated separately because they are not interchangeable
(H-UNUSABLE).** A labelability sweep judged the corpus with two independent
model families that did **not** judge the same events: Step-5 returned a
verdict on 683 of the 684 events, MiniMax-M3.1f on 512, and **510** events
carry a verdict from both. **152 events fail at least one labelability test** -
empty vessels, diagrams, no discernible food mass. That single count has two
denominators. **152/684 = 22.2%** of the whole pre-clean corpus is the rate to
quote for any claim about the corpus: all 152 ids are present in the 684-event
pre-clean manifests and none survives in the cleaned manifest (684 - 152 =
532). **152/510 = 29.8%** is a conditional rate over the both-judged subset,
which the aggregate sweep output never enumerates, and the aggregate is not
self-consistent (683 + 512 - 510 = 685, against a corpus of 684), so 29.8%
must not be presented as a corpus rate. On the 510 both-judged events the two
families agree 96.7% (493/510) on "is there food" and 85.5% (436/510) on "is
it levellable" - the binary judgement is reliable across families, the
five-level judgement is not. Reproduce every figure in this paragraph with
`python3 tools/denominator_audit.py`.

The v0.7 held-out test split is **108 of the 532** cleaned rows (314 train /
110 val / 108 test), all web-harvested, with a test-level distribution of
L3 10 / L4 11 / L5 7 / L6 10 / **L7 70** - so the majority test class is L7
at 70/108 and any comparison against a baseline must say so.

## 3.3 Level-compatibility model

The primary still-image path uses
`google/siglip2-base-patch16-224` as the encoder and a five-class ordinal head
for L3-L7. The head implements monotone CORAL-style thresholds: a shared
scalar projection and ordered learned cut-points are converted into a
probability distribution over the five ordered levels. A `timm` ResNet-18
with the same ordinal head is the small-CNN baseline. The executable training
configuration uses 224-pixel inputs, class-balanced CORAL loss, dropout,
weight decay, mixed precision, and gradient clipping. It first trains the head
with the encoder frozen, then unfreezes the final encoder blocks; the frozen
configuration specifies five independent seeds. v0 ships the training and
evaluation code with its recorded results only; no model weights ship.

Model selection, temperature calibration, and the confidence threshold are
performed on validation data only. At inference, samples below the frozen
threshold return `unclear - perform physical test`; an abstention is never
silently converted to a level. Qwen3-VL-2B is reserved for grounded
multilingual explanations and guided multi-frame reasoning. It receives the
structured decision but cannot change it, and it is not the grading path.
Explanation-adapter weights are withheld with the model weights.

A CORAL ordinal-decode bug made the interior levels (L4/L5/L6) unreachable:
argmax over sigmoid-difference class masses could never fire for interior
classes (proven numerically from the checkpoint). It was fixed with the
ordinal rank rule `level = #{k : P(y>k) > 0.5}`. Held-out results (n = 109,
96 of 109 events covered, label set: the Qwen tier-2 judge labels of that
run): macro-F1 0.088 -> 0.358, coverage 0.70 -> 0.88, ECE 0.138 -> 0.091,
temperature 2.83 -> 0.93; quadratic weighted kappa on covered rows flat at
0.4227 before and after the fix. The pre-registered
gates all failed: wk1 quadratic weighted kappa on covered rows >= 0.70
(measured 0.4227 on held-out n = 109, Qwen tier-2 judge labels - the kill rule
FIRES, so photo-classification is dead as a product capability); wk2
macro-F1 >= 0.75 (measured 0.358) with dangerous under-classification <= 5%
(measured 33%, and 33% is the covered figure: the abstention-blind `_all`
variant reads 29.4% on the same run and is the flattering direction, see
H-ABSTBLIND). The ceiling finding (regen 2026-09-16 on n_pairs = 575, full
tier-2 sample): two VLM judges (Qwen3-VL-2B vs Qwen3-VL-8B) agree on only
**31.0%** of the same images (Cohen's kappa unweighted 0.1062, quadratic-weighted
0.2766, both over all 575 judged pairs with no coverage filter - still
essentially chance for 5 classes) - because middle IDDSI levels
(L4/L5/L6) need absolute size/flow information a scale-free web photo does not
carry. Every label-derived metric above is therefore measured against noise.

**Label set and population for every kappa above (H-YARDSTICK, H-YARDEVID).**
The 0.4227 is quadratic weighted kappa over covered rows only, measured
against the Qwen tier-2 judge labels on held-out n = 109; every abstention is
dropped from both numerator and denominator before the kappa is taken
(H-KAPPACOV), so it is a selective kappa and is not the unconditional quantity
the wk1 threshold is written against. The v0.4/v0.5 numbers were measured
against the v0.5 consensus and fail the wk1 >= 0.70 gate on it. The same three
checkpoints measured against the v0.7 full-coverage consensus clear that same
covered-rows metric in 5/5 seeds each - v0.4 0.7364-0.8077, v0.5
0.7036-0.7953, v0.7 0.8375-0.8604 on held-out n = 108 (v0.4's own v0.5-column
denominator is its 142-event split; its verdict is unchanged on the 108 events
common to all three models). **The pass verdict is metric-dependent, not only
yardstick-dependent**: on the all-rows kappa, which keeps abstentions in the
denominator, the same cells are v0.4 0.687-0.720 (3 of 5 seeds at or above
0.70), v0.5 0.647-0.726 (mean 0.671, 1 of 5 seeds) and v0.7 0.793-0.825
(5 of 5). `results/v07_release/bias_matrix.json` holds that all-rows quantity,
not the gate metric, and `head_to_head2.json` is scored against the v0.5
labels; the covered-rows table for v0.4 and v0.5 is
`results/yardstick_audit/per_seed.json`. The inter-judge kappas 0.1062 /
0.2766 are agreement between the two VLM judges who produced the label set,
over all 575 judged pairs with no coverage filter because neither judge
abstains; they characterise the yardstick itself rather than a model scored
against one. The wk2 33% failure does not depend on the yardstick and stands
on every reading.

**Under-prediction finding (new in v0.1, not in v0.0).** The regen cross-tab on
the full tier-2 sample (n = 575) reveals a systematic pattern the preliminary
22.6% figure did not state: tier-1 systematically *under-predicts* difficulty.
Only L3 is reliably classified (11/12 correct). The interior classes drag
toward the regular-food end - L4 calls (151) miss to L7 in 69 and to L6 in 28;
L5 calls (131) miss to L7 in 54 and to L6 in 42; L6 calls (155) collapse into
L7 in 111; L7 calls (126) over-pull to L6 in 47. This is the *dangerous
direction* (predicting a SAFER texture than truth), the same direction the wk2
gate catches at 33% dangerous under-classification. Judges do not just
disagree randomly; they share a directional bias toward the safe end. The
remedy path is unchanged: physically-tested desk specimens.

## 3.4 Deterministic Flow Test grader

The video grader is a deterministic OpenCV pipeline with no learned model,
network call, or GPU dependency. Its implementation-specific calibrated mode
places a high-contrast ID-1 rectangle (85.60 x 53.98 mm) in the syringe plane
to estimate pixels per millimetre and verify the 61.5 +/- 3.0 mm barrel. The
grader detects paired barrel edges, checks verticality, rectifies the barrel,
detects release from a nozzle change or the start of liquid motion, tracks the
meniscus, and takes a robust reading at 10 s after release. Phone rotation
metadata and non-ASCII input paths are handled during decoding.

Residual volume maps to L0 below 1 mL, L1 from 1 to below 4 mL, L2 from 4 to
below 8 mL, L3 at 8 mL or above after visible flow, and L4 when no detectable
flow occurs. A reading within 0.5 mL of the 1, 4, or 8 mL boundary abstains.
The grader also abstains for a missing or unverifiable scale/syringe, excessive
tilt, missing release, insufficient duration, low meniscus confidence,
occlusion, or detected bubbles/lumps. It retains the numeric estimate and all
quality checks for audit while suppressing the level. Fiducial-optional mode
is explicitly marked uncalibrated and is not used for the planned calibrated
evaluation.

This contribution is described narrowly as the "first open-source smartphone
grader for the 10 ml/10-second IDDSI Flow Test," not the first video grader,
because automated photogrammetric measurement of the syringe test appeared in
2019 [@syringe2019]. End-to-end performance (v0, the working deliverable):
MAE 0.065 ml on a 100-video bench, grading L0-L4. A still photograph is never
used to grade liquid flow.

## 3.5 Evaluation plan

Only `source="real"`, protocol-complete, adjudicated events in the locked
`external_test` split contribute to the primary benchmark. Synthetic,
real-weak, and real-crowd events are excluded or reported separately and are
never cited as ground truth. The split is enforced at recipe, batch, and
kitchen level. Calibration and abstention thresholds are frozen on
validation data before external-test evaluation.

For the L3-L7 model, the v0 report contains macro-F1 with abstentions counted
as misses; quadratic-weighted kappa on covered events, always with its label
set, its n, its coverage and the fact that it is coverage-filtered attached
(H-KAPPACOV, H-YARDSTICK and H-YARDEVID all apply to that number, and the
all-rows kappa on the same events is a different, lower quantity); coverage and
abstention rate; dangerous-direction under-classification (a prediction below
the physical-test level); expected calibration error and temperature; and the
CORAL-decode before/after comparison (Section 3.3). RD-versus-SLP raw
agreement before adjudication would form the human inter-rater baseline but
was not measured in v0; prespecified subgroup analyses and stratified
bootstrap 95% confidence intervals are likewise not reported in v0.

The Flow Test evaluation compares estimated residual volume with the recorded
physical reading, reports mL mean absolute error and L0-L4 agreement away from
the predefined 1/4/8 mL boundaries, and separately reports boundary cases,
abstentions, and technical failure reasons. Failure cases accompany held-out
results. The project gates are prespecified in the runboard; failing a gate
keeps the system at research-preview status rather than changing the metric or
test set. In v0 the wk1 gate failed and the kill rule fired (Section 3.3):
photo-classification is dead as a product capability, and only code, results,
and the flow-test grader ship. Section 3.6 reports the v0.7 evaluation on the
same terms.

| Observed metric (v0 / v0.1 regen) | Value |
|---|---|
| CORAL fix, held-out n = 109: macro-F1 | 0.088 -> 0.358 |
| Coverage | 0.70 -> 0.88 (96 of 109 events covered after the fix) |
| ECE | 0.138 -> 0.091 |
| Temperature | 2.83 -> 0.93 |
| Quadratic weighted kappa, covered rows (n = 109, 96 covered) | 0.4227 (flat before/after fix; label set: Qwen tier-2 judge labels - see H-YARDSTICK, H-KAPPACOV) |
| Dangerous under-classification, covered rows (n = 109) | 33% (gate: <= 5% - FAILED); the abstention-blind all-N variant reads 29.4% on the same run (H-ABSTBLIND) |
| wk1 gate (quadratic weighted kappa on covered rows >= 0.70) | FAILED on the Qwen tier-2 judge labels (0.4227, n = 109) and on the v0.5 consensus (v0.4 0.540-0.600, v0.5 0.537-0.614, v0.7 0.491-0.555; 0/5 seeds each); the same three checkpoints clear it on the v0.7 full-coverage consensus, 5/5 seeds each, on covered rows (v0.4 0.736-0.808, v0.5 0.704-0.795, v0.7 0.838-0.860; held-out n = 108) - but NOT on the all-rows kappa, where v0.5 averages 0.671 with 1 of 5 seeds at or above 0.70 (H-YARDEVID) |
| wk2 gate (macro-F1 >= 0.75) | FAILED (measured 0.358) |
| Inter-judge agreement (Qwen3-VL-2B vs Qwen3-VL-8B, all 575 judged pairs, no coverage filter, v0.1 regen) | 31.0% exact; kappa unweighted 0.1062, quadratic 0.2766 (agreement between the two judges who produced the label set - the yardstick itself, not a model scored against one) |
| Under-prediction bias (v0.1 regen) | tier-1 systematically under-predicts L4-L6 toward L7; dangerous direction |
| Flow residual-volume MAE (100-video bench, L0-L4) | 0.065 ml |
| RD-SLP inter-rater baseline | not measured in v0 |
| Stratified-bootstrap 95% CIs | not reported in v0 |
| Prespecified subgroups | not reported in v0 |

## 3.6 The v0.7 evaluation, stated with its denominators

The v0.7 model is the SigLIP-2 CORAL head trained on the four-family
full-coverage consensus labels over the cleaned 532-row corpus. Its held-out
test split is **108 events** of 532 rows (314 train / 110 val / 108 test), all
web-harvested, level distribution L3 10 / L4 11 / L5 7 / L6 10 / **L7 70**.
Five seeds; every figure below is the five-seed range. Label set for every
kappa below (H-YARDSTICK): **the v0.7 full-coverage consensus** - the circular
comparison, discounted accordingly. Population is stated per row because the
covered subset and all 108 events are different populations and the gap
between them is the model's own abstention (H-KAPPACOV, H-YARDEVID).
Reproduce with `python3 -P tools/verify_paper_claims.py`, which recomputes
every number in this section from `results/v07_release/v07_seed*_metrics.json`.

| metric (v0.7, 5 seeds, held-out n = 108, label set: v0.7 full-coverage consensus) | range |
|---|---|
| quadratic weighted kappa, **covered rows** (the wk1 gate quantity) | 0.838 - 0.860 |
| quadratic weighted kappa, **all 108 rows**, abstentions in the denominator | 0.793 - 0.825 |
| coverage (events carrying a decision) | 78.7 - 85.2% (85 - 92 of 108) |
| abstention rate | 14.8 - 21.3% (16 - 23 of 108) |
| exact level match, covered events | 80.9 - 84.9% |
| exact level match, all 108 events, abstentions scored as misses | 66.7 - 70.4% |
| macro-F1 with abstentions counted as misses | 0.319 - 0.395 |
| dangerous under-classification, covered rows | 3.5 - 5.6% |
| baseline: always answer L7 (the majority test class, 70/108) | 64.8% exact |

The two exact-match rows must be read against the baseline, not against each
other: on the covered basis the model beats always-L7 by 16.1 - 20.1 points,
and on the all-events basis with abstentions scored as misses by only
1.9 - 5.6 points. The 108-event test set is 64.8% L7, so a kappa or an exact
match on it is dominated by one class - and the label set for every kappa in
this section is the v0.7 full-coverage consensus (H-YARDSTICK), which is the
circular comparison.

**Decode collapse on the middle levels.** Per-level recall on decided events,
seed 43 (92 decisions of 108 events), is L3 0.90, L4 0.50, **L5 0.00,
L6 0.00**, L7 1.00; per-level precision on decided events is *undefined* for
L5 and L6 because neither level is ever predicted. **L5 and L6 are never
emitted as a decision in any of the five seeds**: seed 43 emits L3 13, L4 2
and L7 77 of its 92 decisions, and seeds 29, 71 and 101 emit no L4 either.
This is the L7 catch-all sink measured on the current head, and it is carried
as a limitation in Section 6.

# 4. Generator failure case study

The first overnight synthetic-pipeline pilot on 1 September 2026 accepted
**0 of 24** candidate images. This was a generator-quality-control outcome,
not a model or dataset evaluation result. Manual audit agreed with the local
Qwen3-VL-2B consistency judge and identified three recurring prompt-to-image
failures:

1. **Grain texture ignored.** Images prompted as smooth L3/L4 masses retained
   visible whole rice grains or granular kernels.
2. **Garnish added.** The generator introduced sesame seeds, scallions, herbs,
   or other toppings whose untested texture contradicted the requested food
   presentation.
3. **Camera-facet nouns rendered as objects.** Terms intended to describe the
   capture style were interpreted literally, including a smartphone/device
   rendered in or on the food scene.

The prompt-bank remedy was applied before further GPU work: per-level visual
guidance was strengthened with explicit smoothness, particle, garnish, and
separated-liquid constraints; the common negative prompt now excludes
phones/devices, whole or visible rice grains, garnish, and related artefacts;
and camera-profile facets explicitly state that no device is visible in the
scene. These constraints target all three audited failure modes while
preserving capture-style variation. No post-remedy acceptance rate or
downstream benefit is claimed here; both are **TBD** pending a controlled
rerun. The pilot reinforces the provenance rule that synthetic imagery is
development-only and must be judged before use.

# 5. Regulatory posture

From r5_reg.md (frozen): as originally proposed (Snap-to-IDDSI), the system
is probably Software as a Medical Device (SaMD), because it interprets an
individual meal and recommends modification to mitigate a swallowing
disorder; "Guidance only," "not a diagnosis," and "食品不能代替藥物" do not
neutralize that intended medical purpose. Hong Kong: standalone
medical-purpose software is SaMD under TR-007; MDACS listing generally
voluntary. Mainland China: plausibly Class II, potentially Class III -
obtain a formal classification determination before mainland release. EU:
MDSW, Rule 11, Class IIa minimum, IIb arguable given choking/aspiration
consequences. US: dysphagia-specific claims are not disease-unrelated
"general wellness."

This technical note therefore describes an **open research/quality-assurance
benchmark**, not a clinical or deployed safety classifier. Static RGB cannot
directly measure flow, hardness, adhesiveness or cohesiveness; mandatory
abstention and physical-test confirmation are essential (r5_prior.md, frozen
verdict).

**Frozen public-demo boundary wording (r5_reg.md):**

> "Research demonstration for culinary education. Estimates visual similarity
> to IDDSI descriptors; does not perform official IDDSI tests, assess
> swallowing, determine suitability for any person, or decide whether food is
> safe to consume."

A regulated version would require fixed intended use, formal classification,
ISO 13485/14971, IEC 62304/62366, locked/versioned model, analytical and
clinical validation, subgroup/external-site evidence, cybersecurity/privacy,
human-factors work, adverse-event/post-market processes, and jurisdictional
submission (r5_reg.md). This note does not describe such a version.

# 6. Limitations

- **Photographs are not physical tests.** Static RGB cannot directly measure
  flow, hardness, adhesiveness, or cohesiveness (frozen verdict, r5_prior.md).
  A "not diagnosis" disclaimer does not override intended purpose (r5_reg.md).
- **Labels are not clinical ground truth for safety.** Tests were performed
  by LinguaLeap staff and participating assessors, not clinicians
  unaffiliated with the project, and they are not IDDSI certification
  (frozen wording, Section 3.1).
- **Weak/synthetic/crowd provenance classes** support pipeline development
  only; they must not be cited as ground truth (frozen, Section 3.2).
- **Synthetic generation is not self-validating.** The audited 0/24 pilot and
  its three failure modes are documented in Section 4. The prompt remedy has
  not yet been evaluated; its effect is TBD.
- **Photo classification killed at its gates.** The head failed every
  pre-registered gate (wk1 quadratic weighted kappa on covered rows >= 0.70
  measured 0.4227 on held-out n = 109, 96 of 109 events covered, against the
  Qwen tier-2 judge labels - kill rule
  fired; wk2 macro-F1 >= 0.75 measured 0.358, dangerous under-classification
  33% against a <= 5% gate), and that kappa is measured against label noise:
  two VLM judges agree on only **31.0%** of the same images (all 575 judged
  pairs, kappa unweighted 0.1062, quadratic-weighted 0.2766 - still essentially
  chance for 5 classes), and tier-1 systematically under-predicts difficulty
  (Section 3.3).
  Weights are withheld; photo-classification is dead as a product capability.
  Label set and population for these kappas (H-YARDSTICK, H-YARDEVID): the
  0.4227 is a covered-rows kappa against the Qwen tier-2 judge labels, and the
  v0.4/v0.5 numbers are covered-rows kappas against the v0.5 consensus; the
  models fail the wk1>=0.70 gate on both of those label sets. The same three
  checkpoints measured against the v0.7 full-coverage consensus clear that same
  covered-rows metric in 5/5 seeds each (v0.4 0.736-0.808, v0.5 0.704-0.795,
  v0.7 0.838-0.860; held-out n = 108), but on the all-rows kappa the v0.5
  model averages 0.671 and clears 0.70 on only 1 of 5 seeds, so "all three
  pass" is true of the covered-rows metric and false of the all-rows metric.
  The 0.1062 / 0.2766 inter-judge kappas are agreement
  between the two judges who produced the label set, over all 575 judged pairs
  with no coverage filter, so they characterise the yardstick itself rather
  than a model scored against one.
- **Decode collapse: L5 and L6 are never emitted as a decision.** On the v0.7
  held-out split (108 events, 5 seeds), per-level recall on decided events is
  L3 0.90, L4 0.50, **L5 0.00, L6 0.00**, L7 1.00 at seed 43 (92 decisions of
  108 events), and L5 and L6 are never emitted as a decision in **any** of the
  five seeds - seeds 29, 71 and 101 emit no L4 either. Per-level precision is
  undefined for L5 and L6 because the level is never predicted. The head is
  therefore an L7 detector with a partial L3 detector, not a five-level
  classifier, and a reported kappa or exact match on this test set is dominated
  by the L7 class (70 of 108 events; the always-L7 baseline scores 64.8%
  exact, against 80.9-84.9% for the model on covered events and only
  66.7-70.4% on all 108 events with abstentions scored as misses). Section 3.6
  carries the numbers.
- **Every kappa is conditional on its yardstick and on its population.** The
  same three checkpoints fail wk1 against the v0.5 consensus and pass it
  against the v0.7 full-coverage consensus (H-YARDSTICK), and the same cells
  pass on the covered-rows metric while the v0.5 model fails on the all-rows
  metric (H-YARDEVID). `results/v07_release/bias_matrix.json` holds the
  all-rows kappa, not the gate metric. No non-circular yardstick exists: the
  label sets were produced by the same labeller families as the models, so a
  reviewer should discount every performance claim here, including this one.
- **Corpus denominator: 152 of 684 events were never labelable.** The
  labelability sweep removed 152 events - **152/684 = 22.2%** of the whole
  pre-clean corpus, which is the rate to quote for any claim about the corpus
  (all 152 ids are present in the 684-event pre-clean manifests and none
  survives in the cleaned 532-row manifest, 684 - 152 = 532). **152/510 =
  29.8%** is a conditional rate over the both-judged subset and must not be
  quoted as a corpus rate: the aggregate is not self-consistent (683 + 512 -
  510 = 685 against a corpus of 684). The removals fall on the hard middle
  levels, so the cleaned corpus is a harder problem, not an easier one.
- **Population and cuisine coverage:** unverified in v0. The planned remedy is
  physically-tested desk specimens (60 base foods x 5 levels = 300,
  single-operator, protocol-following, test-filmed, with a filmed re-test
  subsample for intra-rater reliability) - planned, not done.
- **Flow grader scope:** claims only the 10 mL / 10 s smartphone Flow Test
  grader (MAE 0.065 ml on the 100-video bench, L0-L4), not "first video
  grader" (2019 photogrammetric syringe measurement predates it).
- **What v0 reports and what it does not.** Classifier results (CORAL fix,
  gates, 31.0% ceiling finding + kappa + under-prediction, the v0.7 evaluation
  and the L5/L6 decode collapse) and the flow-grader
  bench are reported in Sections 3.3-3.6, each with its label set and its
  population named (H-YARDSTICK, H-YARDEVID). The RD-SLP inter-rater baseline,
  subgroup analyses, and bootstrap CIs were not measured in v0. The 0/24
  generator-QA pilot is not a model, dataset, or flow-grader evaluation.

# 7. Conclusion

IDDSI-Open v0 ships code, eval results, and an open-source smartphone grader
for the 10 mL / 10 s IDDSI Flow Test (MAE 0.065 ml on the 100-video bench,
L0-L4) - claiming only the narrow first stated in Section 1 ("the first
open-source smartphone grader for the 10 ml IDDSI Flow Test") and never
clinical validity - plus a measured negative finding: photo-classification
failed its pre-registered gates (quadratic weighted kappa on covered rows
0.4227 against a >= 0.70 gate; macro-F1 0.358 against a >= 0.75 gate;
dangerous under-classification 33% against a <= 5% gate) measured against
**31.0%** inter-judge label noise on all 575 judged
pairs (kappa unweighted 0.1062, quadratic-weighted 0.2766, with a
systematic tier-1 under-prediction bias - Section 3.3), so the model weights
and the harvested dataset are withheld and photo-classification is dead as a
product capability. Label set and population for these kappas (H-YARDSTICK,
H-YARDEVID): the 0.4227 is a covered-rows kappa against the Qwen tier-2 judge
labels (held-out n = 109), the v0.4/v0.5 numbers are covered-rows kappas
against the v0.5 consensus, and the models fail the wk1>=0.70 gate on both of
those label sets. The same three checkpoints measured against the v0.7
full-coverage consensus clear that same covered-rows metric in 5/5 seeds each
(v0.4 0.736-0.808, v0.5 0.704-0.795, v0.7 0.838-0.860, held-out n = 108); on
the all-rows kappa the v0.5 model averages 0.671 with 1 of 5 seeds at or above
0.70, so the pass verdict is a property of the metric as well as of the
yardstick. The v0.7 head also collapses its decode onto L7: L5 and L6 are
never emitted as a decision in any of the five seeds (Section 3.6, Section 6).
Held-out results, failure cases, and test videos
accompany the repository. Planned next step, not done: physically-tested
desk specimens (60 base foods x 5 levels = 300, single-operator,
protocol-following, test-filmed, with a filmed re-test subsample for
intra-rater reliability). Numeric conclusions: as reported in Sections 3.3-3.6.

# References

::: {#refs}
:::
