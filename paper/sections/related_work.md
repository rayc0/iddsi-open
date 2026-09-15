<!--
W36 section file. Standalone expansion of paper.md Section 2 ("Related work").
Every factual claim and number below traces to r5_prior.md / r5_sol.md /
r5_feas.md (frozen project sources). Citation keys refer to
../references.bib. W22 owns paper.md; this file is not yet included by it.
-->

# Related work

## The IDDSI framework and the limits of physical testing

The International Dysphagia Diet Standardisation Initiative (IDDSI) defines
eight food and liquid levels (L0-L7) whose compliance is determined by
physical tests - the 10 mL / 10 s flow test for liquids, and fork drip, spoon
tilt, and fork pressure for foods - not by appearance
[@iddsi_framework]. [CITATION TBD — see r5_prior.md: full bibliographic
reference for the original 2017 IDDSI framework publication is not stated in
the project sources and must be added before submission.]

Crucially for any automated approach, the physical tests themselves are not
perfectly reproducible across human raters: reported inter-rater agreement
for spoon-tilt and fork-pressure characteristics is weak, with kappa values
of only 0.23-0.36 [@iddsi_reliability]. This cuts both ways for machine
grading: it motivates objective measurement, but it also means that any
"ground truth" label collected from a single rater carries substantial noise,
and that human inter-rater agreement must be reported alongside model results
(r5_feas.md, evaluation plan).

## Automated measurement of the IDDSI syringe test (2019)

Automated video/photogrammetric measurement of the official IDDSI syringe
flow test already appeared in 2019 [@syringe2019]. For this reason IDDSI-Open
claims only the narrow, verifiable first of "first open-source smartphone
grader for the 10 ml/10-second IDDSI Flow Test" - explicitly **not** "first
video grader" (frozen wording, r5_prior.md).

## Learning-based level classification from images and video

Three direct automated predecessors frame the landscape (all facts below as
stated in r5_prior.md):

- **ViscoCam (PolyU, 2021)** used smartphone video plus motion sensing to
  classify three IDDSI liquid levels, reaching 96.5% controlled accuracy but
  only >81% under extreme conditions [@viscocam]. It exposes neither dataset
  nor weights.
- **OptiTexture (HKUST)** combines RGB with a <$20 light-scattering sensor,
  reaching 91.96% on 112 L3-L6 samples; its vision-only baseline was 69.64%
  [@optitexture]. It exposes neither dataset nor weights.
- **EdUHK (Frontiers in Nutrition, 2026)** evaluated a single-image Qwen
  pipeline that achieved only 61.74% exact match on CEIV-115, while open
  vision-language models missed every liquid hazard [@eduhk_frontiers]. CEIV
  publishes examples, not the full machine-readable corpus.

Taken together, these results bracket the realistic accuracy of image-only
IDDSI inference - from 61.74% exact match on HK-oriented images to 96.5% only
in controlled liquid-only settings - and none of the three releases a
downloadable IDDSI-specific dataset or trained weights under an explicit
reuse licence. The v0 negative finding reframes this bracket from below: two
VLM judges (Qwen3-VL-2B vs Qwen3-VL-8B) agree on only 22.6% of the same
images - chance level for 5 classes - because middle IDDSI levels (L4/L5/L6)
need absolute size/flow information a scale-free web photo does not carry, so
label-derived accuracy numbers on such photos are measured against noise.

## Meal-screening systems and product-side tools

Australia's proprietary **Safe Plate** advertises photo-and-video IDDSI
verification [@safeplate]; it is closed, with no published dataset or
weights. The official IDDSI iOS/Android app is a descriptor and
test-reference library, not a classifier [@iddsi_app]. Hardware testers and
menu-software ecosystems (Viscgo, CF-200N texture analysers,
MealSuite/Dietech) and manufacturer texture-modified ranges (Japan's
Kewpie/UDF, Korea's Pulmuone DesignMeal and Hyundai Green Food Greating,
Europe's apetito/winVitalis, US Hormel/Lyons and SimplyThick) classify or
test **known** products rather than inferring the level of an unknown plate
(r5_prior.md). No independent mainland-Chinese plate-photo classifier was
found (r5_prior.md).

## Openness gap and the position of IDDSI-Open (v0)

No downloadable IDDSI-specific trained weights plus a fully licensed
image/video dataset were found as of the sources' search (r5_prior.md).
CEIV publishes examples rather than the full machine-readable corpus;
OptiTexture and ViscoCam expose neither dataset nor weights; open resources
such as FoodSense/PlateInsight concern generic perceived texture, not IDDSI
[@foodsense]. That gap remains open after v0 (see the claim below): v0 ships
code, eval results, and the flow-test grader, while the harvested dataset and
the model weights are withheld.

IDDSI-Open v0 holds its position along four axes, while claiming none of the
forbidden firsts (r5_prior.md frozen never-claim list: not "first IDDSI AI",
not "first image classifier", not "first dataset/benchmark", not clinically
validated - prior art: PolyU ViscoCam 2021, HKUST OptiTexture 91.96%, EdUHK
Frontiers 2026 61.74%, Safe Plate AU):

1. **A licence-audited, deliberately withheld harvest.** The harvested
   dataset is unpublishable on licence grounds: of 587 unique events exactly
   2 would ship cleanly (592 rows carry unverified open-images/openverse
   licences, 7 are CC BY-SA against the BY-NC-SA main release, 109 have
   unresolved redistribution status). Withholding is the stated v0 decision.
2. **Synthetic-first pipeline development.** A four-provenance-class design
   (synthetic / real-weak / real-crowd / real-tested) that uses synthetic and
   weakly labelled media for pipeline development and pretraining while
   quarantining them from every physically tested benchmark split.
3. **Mandatory abstention.** The system must answer "unclear - perform
   physical test" rather than guess - directly addressing the
   dangerous-under-classification failure mode highlighted by the EdUHK
   liquid-hazard result [@eduhk_frontiers]. v0 measured dangerous
   under-classification at 33% against a <= 5% gate (FAILED).
4. **An open-source flow-test grader that works.** A deterministic CV grader
   for the 10 mL / 10 s IDDSI Flow Test, released as source: MAE 0.065 ml on
   a 100-video bench, grading L0-L4. This is the ONE claim that stands.

**v0 contribution claim (release shape locked 2026-09-07; the honest release
card `hf/model_README.md` is the authority):**

> "The ambition was 'the first publicly downloadable IDDSI-specific
> smartphone-food dataset and fine-tuned model released under explicit reuse
> licences, with physically tested Chinese/Cantonese soft-meal labels'; v0
> does NOT meet it - the harvested dataset is withheld on licence grounds (of
> 587 unique events exactly 2 would ship cleanly; 592 rows carry unverified
> open-images/openverse licences, 7 are CC BY-SA against the BY-NC-SA main
> release, 109 have unresolved redistribution status) and the model weights
> are withheld because the photo-classification head failed its
> pre-registered gates (wk1 weighted kappa >= 0.70 measured 0.42; wk2
> macro-F1 >= 0.75 measured 0.358, with dangerous under-classification at 33%
> against a <= 5% gate; the wk1 kill rule fired and photo-classification is
> dead as a product capability). The ONE claim that stands is 'the first
> open-source smartphone grader for the 10 ml IDDSI Flow Test'."
