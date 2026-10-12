<!--
W36 section file. Standalone expansion of paper.md Section 8 ("Limitations").
Sources: r5_feas.md, r5_reg.md, r5_prior.md, r5_sol.md (frozen project
sources) plus the audited 0/24 synthetic pilot (2026-09-01 Spark run;
synth/synth/judge_prompts.py and ops/tests/fixtures/synth_pilot_night.log).
Updated v0.1 (2026-09-16 regen; honest release card hf/model_README.md is the
authority): classifier results (CORAL fix, gates, 31.0% exact agreement on
n=575 with kappa unweighted 0.1062 / quadratic 0.2766, under-prediction
finding) and the flow-grader bench (MAE 0.065 ml, 100 videos) are reported;
RD-SLP baseline, subgroups, and CIs were not measured in v0. Population for
those inter-judge kappas: all 575 judged pairs, no coverage filter, because
neither judge abstains. Label set for
those kappas (H-YARDSTICK): 0.1062 / 0.2766 is agreement between the two VLM
judges who produced the label set, so it characterises the yardstick itself;
the model kappas in this file are quadratic weighted kappas over covered rows
measured against the Qwen tier-2 judge labels (0.4227, held-out n = 109, 96 of
109 covered), and the v0.4/v0.5 numbers are covered-rows kappas against the
v0.5 consensus (both fail wk1>=0.70). The same three checkpoints measured
against the v0.7 full-coverage consensus clear that same covered-rows metric
in 5/5 seeds each (v0.4 0.736-0.808, v0.5 0.704-0.795, v0.7 0.838-0.860;
held-out n = 108) - and on the all-rows kappa the v0.5 model averages 0.671
with 1 of 5 seeds at or above 0.70, so the pass verdict is metric-dependent as
well as yardstick-dependent (H-YARDEVID). Corpus denominators (H-UNUSABLE):
152/684 = 22.2% of the whole pre-clean corpus is the rate to quote; 152/510 =
29.8% is a conditional rate over the both-judged subset and must not be
quoted as a corpus rate. v0.7 decode collapse: L5 and L6 are never emitted as
a decision in any of the five seeds.
Citation keys
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
spoon-tilt/fork-pressure characteristics [@iddsi_reliability]; label set for
that kappa (H-YARDSTICK): human raters scoring the official IDDSI physical
tests, a different label set from any model label set in this project, and not
a model scored against one; population: every pair both raters scored, with no
coverage filter because a human rater does not abstain), so
single-rater "ground truth" labels are themselves noisy.

## Known dangerous zone: L4/L5/L6 confusions

The L4/L5 and especially L5/L6 boundaries require spoon/fork tests and are
the known dangerous zone for image-only inference (r5_feas.md, r5_sol.md).
The project's evaluation plan therefore reports dangerous-direction FNR
(under-classification: predicting a softer level than physically tested) as
a first-class metric - v0 measured 33% against a <= 5% gate (FAILED) - and
the system abstains rather than guess. The v0.1 classifier numbers, measured
against label noise: a CORAL ordinal-decode bug made interior levels
unreachable and was fixed (held-out n = 109, 96 of 109 covered: macro-F1
0.088 -> 0.358, coverage 0.70 -> 0.88, ECE 0.138 -> 0.091, temperature
2.83 -> 0.93; quadratic weighted kappa on covered rows flat at 0.4227); wk1
quadratic weighted kappa on covered rows >= 0.70 measured 0.4227, so
the kill rule fired and photo-classification is dead as a product capability.
Label set and population for these kappas (H-YARDSTICK, H-YARDEVID): the
0.4227 is a covered-rows kappa (96 of 109 events) against the Qwen tier-2
judge labels, and the v0.4/v0.5 numbers are covered-rows kappas against the
v0.5 consensus; the models fail the wk1>=0.70 gate on both of those label
sets. The same three checkpoints measured against the v0.7 full-coverage
consensus clear that same covered-rows metric in 5/5 seeds each (v0.4
0.736-0.808, v0.5 0.704-0.795, v0.7 0.838-0.860; held-out n = 108, coverage
78.7-85.2%), but on the all-rows kappa the v0.5 model averages 0.671 and
clears 0.70 on only 1 of 5 seeds, so the pass verdict is a property of the
metric as well as of the yardstick. The 0.1062 / 0.2766 figures below are
agreement between the two VLM judges who produced the label set, over all 575
judged pairs with no coverage filter, so they
characterise the yardstick itself rather than a model scored against one. The
0.23-0.36 figures in the previous paragraph are the published physical-test
literature's own inter-rater agreement on a different label set again, not this
project's model.
The ceiling finding (regen 2026-09-16, n_pairs = 575, full tier-2 sample):
two VLM judges (Qwen3-VL-2B vs Qwen3-VL-8B) agree on only **31.0%** of the
same images (Cohen's kappa unweighted 0.1062, quadratic-weighted 0.2766, both
over all 575 judged pairs with no coverage filter -
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
unverified in v0. The corpus itself is smaller than it looks: the labelability
sweep removed **152 of the 684 pre-clean events (152/684 = 22.2%** of the whole
corpus, the rate to quote for any claim about the corpus - all 152 ids are
present in the 684-event pre-clean manifests and none survives in the cleaned
532-row manifest, 684 - 152 = 532). The same count over the both-judged subset
is **152/510 = 29.8%**, a conditional rate over a subset the aggregate sweep
output never enumerates, and the aggregate is not self-consistent (683 + 512 -
510 = 685 against a corpus of 684), so 29.8% must not be quoted as a corpus
rate. The removals fall on the hard middle levels, so the cleaned corpus is a
harder problem rather than an easier one, and the v0.7 held-out test split is
**108 of the 532** cleaned rows with a level distribution of L3 10 / L4 11 /
L5 7 / L6 10 / **L7 70**.
The planned remedy is physically-tested desk specimens (60
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

## Decode collapse: L5 and L6 are never emitted as a decision

The CORAL decode bug that made the interior levels unreachable was fixed, and
the shipped model still does not use three of its five levels. On the v0.7
held-out split (**108 events, 5 seeds**, label set: the v0.7 full-coverage
consensus), per-level recall on decided events at seed 43 (92 decisions of
108 events) is L3 0.90, L4 0.50, **L5 0.00, L6 0.00**, L7 1.00, and per-level
precision on decided events is *undefined* for L5 and L6 because neither level
is ever predicted. **L5 and L6 are never emitted as a decision in any of the
five seeds**; seed 43 emits L3 13, L4 2 and L7 77 of its 92 decisions, and
seeds 29, 71 and 101 emit no L4 either. This is the L7 catch-all sink measured
on the current head: it is an L7 detector with a partial L3 detector, not a
five-level classifier, and it is the same dangerous direction the wk2 gate
catches - a middle level that cannot be named is a middle level that will be
reported as regular food. Because the test split is 64.8% L7 (70 of 108), a
headline kappa or exact match on it is dominated by one class: the always-L7
baseline scores 64.8% exact against the model's 80.9-84.9% on covered events,
and only 66.7-70.4% on all 108 events with abstentions scored as misses.
Reproduce with `python3 -P tools/verify_paper_claims.py`.

## Every kappa is conditional on its yardstick and on its population

Two independent conditionals apply to every kappa in this note, and they
compound. The **yardstick**: the same three checkpoints fail wk1 >= 0.70
against the v0.5 consensus (0/5 seeds each) and clear it against the v0.7
full-coverage consensus (5/5 seeds each) - nothing about the models changed
between those columns (H-YARDSTICK). The **population**: the covered-rows
metric and the all-rows metric give different verdicts on the same cells - on
the v0.7 labels the v0.5 model averages 0.671 over all rows and clears 0.70 on
only 1 of 5 seeds, while its covered-rows range is 0.704-0.795 (H-YARDEVID).
`results/v07_release/bias_matrix.json` holds the all-rows kappa, not the gate
metric; `results/yardstick_audit/per_seed.json` holds the covered-rows table.
No non-circular yardstick exists, because the label sets were produced by the
same labeller families as the models, so every performance claim here should
be discounted equally.

## What v0 reports and what it does not

This is a research-preview technical note. v0.1 reports the classifier
results (CORAL decode fix: macro-F1 0.088 -> 0.358, coverage 0.70 -> 0.88,
ECE 0.138 -> 0.091, temperature 2.83 -> 0.93, n = 109 with 96 of 109 covered;
quadratic weighted kappa on covered rows 0.4227;
dangerous under-classification 33% on covered rows; wk1/wk2 gates failed, kill rule fired),
the **31.0%** inter-judge ceiling finding (all 575 judged pairs, kappa unweighted 0.1062,
quadratic-weighted 0.2766, plus the new under-prediction finding), and the
flow-grader bench (MAE 0.065 ml, 100 videos, L0-L4). Label set and population
for those kappas (H-YARDSTICK, H-YARDEVID): the 0.4227 is a covered-rows kappa
against the Qwen tier-2 judge labels, and the v0.4/v0.5 numbers are
covered-rows kappas against the v0.5 consensus; the
models fail the wk1>=0.70 gate on both of those label sets, and the same three
checkpoints measured against the v0.7 full-coverage consensus clear that same
covered-rows metric in 5/5 seeds each (v0.4 0.736-0.808, v0.5 0.704-0.795,
v0.7 0.838-0.860; held-out n = 108) - while on the all-rows kappa the v0.5
model averages 0.671 with 1 of 5 seeds at or above 0.70, so "all three pass"
holds on the covered-rows metric and not on the all-rows metric.
The 0.1062 / 0.2766 figures are agreement between the two VLM judges who
produced the label set, over all 575 judged pairs with no coverage filter, so
they characterise the yardstick itself rather than a
model scored against one. The RD-SLP inter-rater
baseline, subgroup analyses, and stratified-bootstrap 95% CIs were not
measured in v0. v0 ships code, eval results, and the flow-test grader only:
the harvested dataset is withheld on licence grounds (of 587 unique events
exactly 2 would ship cleanly) and the model weights are withheld on
safety-gate grounds.
