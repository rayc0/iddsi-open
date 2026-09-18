<!--
W36 section file. Standalone expansion of paper.md Section 8 ("Limitations").
Sources: r5_feas.md, r5_reg.md, r5_prior.md, r5_sol.md (frozen project
sources) plus the audited 0/24 synthetic pilot (2026-09-01 Spark run;
synth/synth/judge_prompts.py and ops/tests/fixtures/synth_pilot_night.log).
Updated v0.1 (2026-09-16 regen; honest release card hf/model_README.md is the
authority): classifier results (CORAL fix, gates, 31.0% exact agreement on
n=575 with kappa unweighted 0.1062 / quadratic 0.2766, under-prediction
finding) and the flow-grader bench (MAE 0.065 ml, 100 videos) are reported;
RD-SLP baseline, subgroups, and CIs were not measured in v0. Citation keys
refer to ../references.bib. W22 owns paper.md; this file is not yet
included by it.
-->

# Limitations

## Photographs are not physical tests

Static RGB cannot directly measure flow, hardness, adhesiveness, or
cohesiveness (frozen verdict, r5_prior.md). The adjacent IDDSI levels are
separated precisely by properties that surface appearance does not reveal
(r5_feas.md): L3/L4 differ by flow; L4/L5 by granulation, cohesion, and
stickiness; L5/L6 by particle size plus softness; L6/L7 by size restrictions,
chewing demand, and deformation. Level 7 Regular versus Easy-to-Chew must
also be distinguished, and Regular itself has no texture restriction. Flow
tests therefore need video or syringe physics, not stills - and even the
physical tests have weak human inter-rater agreement (kappa 0.23-0.36 for
spoon-tilt/fork-pressure characteristics [@iddsi_reliability]), so
single-rater "ground truth" labels are themselves noisy.

## Known dangerous zone: L4/L5/L6 confusions

The L4/L5 and especially L5/L6 boundaries require spoon/fork tests and are
the known dangerous zone for image-only inference (r5_feas.md, r5_sol.md).
The project's evaluation plan therefore reports dangerous-direction FNR
(under-classification: predicting a softer level than physically tested) as
a first-class metric - v0 measured 33% against a <= 5% gate (FAILED) - and
the system abstains rather than guess. The v0.1 classifier numbers, measured
against label noise: a CORAL ordinal-decode bug made interior levels
unreachable and was fixed (held-out n = 109: macro-F1 0.088 -> 0.358,
coverage 0.70 -> 0.88, ECE 0.138 -> 0.091, temperature 2.83 -> 0.93;
weighted kappa flat at 0.42); wk1 weighted kappa >= 0.70 measured 0.42, so
the kill rule fired and photo-classification is dead as a product capability.
The ceiling finding (regen 2026-09-16, n_pairs = 575, full tier-2 sample):
two VLM judges (Qwen3-VL-2B vs Qwen3-VL-8B) agree on only **31.0%** of the
same images (Cohen's kappa unweighted 0.1062, quadratic-weighted 0.2766 -
still essentially chance for 5 classes), with a systematic tier-1
under-prediction bias dragging interior classes toward L7 (the dangerous
direction). Middle levels need absolute size/flow information a scale-free
web photo does not carry.

## Synthetic-first training has generator artefacts

The synthetic provenance class is development-only and its rows carry null
physical-test labels by construction, but its images also exhibit concrete
generator artefacts, documented by the audited pilot run of 2026-09-01 on
the DGX Spark (72 attempts, 0/24 candidates accepted, judged by
Qwen/Qwen3-VL-2B-Instruct; recorded in `synth/synth/judge_prompts.py` and
`ops/tests/fixtures/synth_pilot_night.log` - internal artefacts, not a
publication). The audited failure modes are:

- **grain texture ignored** - whole rice grains rendered inside L3/L4 masses;
- **garnish added on top** - sesame seeds, scallions, and herbs sprinkled
  onto texture-modified dishes;
- **camera-facet nouns rendered as objects** - prompt words describing the
  photo style (e.g. "phone-camera style", "noise") appearing as physical
  objects in the scene;
- plus separated liquid pooling around the mass and smartphones/devices
  rendered literally in or on the food.

These artefacts motivated the negative-signature judge prompts, but they
mean synthetic data can teach texture-violating visual patterns unless every
candidate is judged; the 0/24 acceptance rate is itself evidence that
unfiltered synthetic generation is unusable as-is. Synthetic rows can only
enter `train` or `val`, never `external_test`, and must never be counted in
a physically tested dataset, result, coverage statistic, or product claim.

## Weak and crowd labels are not ground truth

**Real-weak labels are deferred and unverified.** The harvest pipeline
deliberately defers GPU labelling (`--labeler defer`): the accepted pilot
manifests (100 events, 2026-08-31) carry no level labels at all, and weak
labels, once produced, are model guesses - "a photograph cannot establish
flow, softness, stickiness, cohesiveness, particle compliance, swallowing
suitability, or safety" (harvest/README.md, frozen). **Crowd labels are
unverified.** `level_user` records what the contributor says the recipe is;
an uploaded video is not proof that the official procedure was followed
(crowd/README.md, frozen). Neither class may enter a physically tested
benchmark split or be cited as ground truth.

## Tests performed by LinguaLeap staff, not clinicians

The physical tests were performed by LinguaLeap staff and participating
assessors, not by clinicians unaffiliated with the project, and they are not
IDDSI certification (frozen wording, dataschema/docs/DATASHEET_TEMPLATE.md:
"Tests were performed by LinguaLeap and participating assessors; they are not
IDDSI certification."). No IDDSI certification or endorsement is claimed or
implied, and the project follows IDDSI's labelling rules.

## Regulatory posture: research preview, not a medical device

As originally proposed (Snap-to-IDDSI), the system is probably Software as a
Medical Device: it interprets an individual meal and recommends modification
to mitigate a swallowing disorder, and "guidance only" / "not a diagnosis"
disclaimers do not neutralize that intended medical purpose (r5_reg.md,
frozen). Hong Kong treats standalone medical-purpose software as SaMD under
TR-007 [@hk_tr007]; mainland China classification is plausibly Class II and
potentially Class III [@nmpa_class]; under EU MDR Rule 11 it would be Class
IIa minimum, with IIb arguable given choking/aspiration consequences
[@mdcg2019_11]; and in the US, dysphagia-specific claims are not
disease-unrelated "general wellness" [@fda_cds]. This note therefore
describes an **open research/quality-assurance benchmark**, not a clinical
or deployed safety classifier, and the public demo holds the frozen boundary
wording (r5_reg.md):

> "Research demonstration for culinary education. Estimates visual
> similarity to IDDSI descriptors; does not perform official IDDSI tests,
> assess swallowing, determine suitability for any person, or decide whether
> food is safe to consume."

A regulated version would additionally require fixed intended use, formal
classification, ISO 13485/14971, IEC 62304/62366, locked/versioned models,
analytical and clinical validation, subgroup/external-site evidence,
cybersecurity/privacy, human-factors work, and post-market processes
(r5_reg.md). This project does not describe such a version.

## Coverage and flow-grader scope

The tested-label corpus is Hong Kong Chinese/Cantonese soft meals;
generalization to other cuisines, kitchens, phones, and populations is
unverified in v0. The planned remedy is physically-tested desk specimens (60
base foods x 5 levels = 300, single-operator, protocol-following,
test-filmed, with a filmed re-test subsample for intra-rater reliability) -
planned, not done. The flow grader claims only the 10 mL / 10 s smartphone
Flow Test - not "first video grader" (2019 photogrammetric syringe
measurement [@syringe2019] predates it) - and abstains near level boundaries
or when bubbles, lumps, occlusion, or an unverified syringe/scale are
detected. On the one real seed video processed early (an old 640x368 phone
recording, 2026-08-31), the grader correctly returned `abstain` on multiple
quality gates - abstention works - and on the v0 100-video bench the grader
achieves MAE 0.065 ml, grading L0-L4. A still photograph is never used to
grade liquid flow.

## What v0 reports and what it does not

This is a research-preview technical note. v0.1 reports the classifier
results (CORAL decode fix: macro-F1 0.088 -> 0.358, coverage 0.70 -> 0.88,
ECE 0.138 -> 0.091, temperature 2.83 -> 0.93, n = 109; weighted kappa 0.42;
dangerous under-classification 33%; wk1/wk2 gates failed, kill rule fired),
the **31.0%** inter-judge ceiling finding (n=575, kappa unweighted 0.1062,
quadratic-weighted 0.2766, plus the new under-prediction finding), and the
flow-grader bench (MAE 0.065 ml, 100 videos, L0-L4). The RD-SLP inter-rater
baseline, subgroup analyses, and stratified-bootstrap 95% CIs were not
measured in v0. v0 ships code, eval results, and the flow-test grader only:
the harvested dataset is withheld on licence grounds (of 587 unique events
exactly 2 would ship cleanly) and the model weights are withheld on
safety-gate grounds.
