---
language: en
pipeline_tag: image-classification
tags:
- iddsi
- food-texture
- research-preview
- siglip2
license: apache-2.0
---

# IDDSI-Open research preview — HONEST release card

> Research preview. Release shape decided 2026-09-07, locked. Published on Hugging Face under org `lingualeap` as the public face of this release.

## What this is

This release ships CODE plus EVAL RESULTS plus the flow-test grader plus a measured NEGATIVE FINDING. The model WEIGHTS and the harvested image DATASET are WITHHELD. Withholding is a deliberate, stated decision. It is a contribution, not an embarrassment.

- The photo-classification head fails the project's own pre-registered safety gates. Dangerous under-classification (predicting a SAFER texture than truth) is 33%. A downloadable classifier would be used for verdicts whatever the card says, and a disclaimer does not cure a claim. Withholding the weights is the mitigation.
- The harvested dataset is unpublishable on licence grounds. Of 587 unique events exactly 2 would ship cleanly. 592 rows carry unverified open-images/openverse licences, 7 are CC BY-SA vs the BY-NC-SA main release, 109 have unresolved redistribution status.

Code licence: apache-2.0.

## Pipeline

Harvest, then two-tier VLM weak labels, then a SigLIP-2 CORAL head, then calibrated abstention:

- Harvest: 713 harvested real CC-licence-candidate images in 3 sets (realweak_pilot 117+12, harvest_d1 400, harvest_d1b 184). 701 raw plus 6 validated cloud-generated.
- Weak labels: two VLM judges, Qwen3-VL-2B tier-1 vs Qwen3-VL-8B tier-2.
- Model: SigLIP-2 CORAL head with calibrated abstention. Abstention outputs `unclear — perform physical test` and never a verdict.

## Headline negative finding (regen 2026-09-16 — supersedes the preliminary 22.6%)

The two VLM judges (Qwen3-VL-2B tier-1 vs Qwen3-VL-8B tier-2) agree on **31.0%** of the same images (n_pairs = 575, full tier-2 sample). This is only marginally above the ~20% chance level for 5 classes. The chance-corrected agreement is also very low: Cohen's kappa is **κ_unweighted = 0.1062** and **κ_quadratic-weighted = 0.2766** (Landis-Koch: slight / fair at best). THEREFORE every label-derived metric below is measured against label noise and means nothing about real-world accuracy.

This inter-judge agreement IS the headline negative finding: web photos without a physical scale fiducial cannot be weakly labelled for IDDSI texture levels, even by strong VLMs, because middle levels (L4/L5/L6) require absolute size and flow information a photograph doesn't carry. The new kappa disclosure was not in v0.0 because the preliminary 22.6% was measured on a smaller subset; the regen on the full tier-2 sample (n=575) is the publication-grade figure. Full table: `results/regen_20260916/inter_judge_20260916.md`.

**Label set for this kappa (H-YARDSTICK).** This κ is agreement *between the two VLM judges who produced the label set*, not a model scored against a label set — it characterises the yardstick itself, so it has no "which label set" answer of its own. For contrast, every model kappa published in this card is measured against a label set we built: the v0.4/v0.5 numbers were measured against the v0.5 consensus and fail the wk1>=0.70 gate; the same models measured against the v0.7 full-coverage consensus pass (v0.7: 0.838-0.860).

### Under-prediction finding (new in v0.1, not in v0.0)

The cross-tab on the full sample reveals a systematic pattern the preliminary ceiling finding did not state: tier-1 **under-predicts** difficulty. Only L3 calls are reliable. The interior classes drift toward the regular-food end:

| t1 \ t2 | L3 | L4 | L5 | L6 | L7 | (total) |
|---|---:|---:|---:|---:|---:|---:|
| L3 | 11 | 1 | 0 | 0 | 0 | 12 |
| L4 | 4 | 42 | 8 | 28 | 69 | 151 |
| L5 | 5 | 3 | 27 | 42 | 54 | 131 |
| L6 | 3 | 8 | 3 | 30 | 111 | 155 |
| L7 | 4 | 1 | 6 | 47 | 68 | 126 |

- L4 calls (151): 69 mis-predict to L7, 28 mis-predict to L6 — only 42 stay at L4
- L5 calls (131): 54 mis-predict to L7, 42 mis-predict to L6 — only 27 stay at L5
- L6 calls (155): 111 mis-predict to L7 — only 30 stay at L6
- L7 calls (126): 47 over-pull to L6, 6 over-pull to L5 — 68 stay at L7

This is the **dangerous direction** — predicting a SAFER texture than truth (see H-L7SINK in `HAZARD_LOG.md`). The wk2 gate already catches it at 33% dangerous under-classification. The regen does not relax the headline: judges do not just disagree randomly — they share a directional bias toward the safe end.

## v0.2 — retrained on a corrected manifest (2026-10-10)

Same code, corrected data. The v0.1 label set contained 20 duplicate `event_id`
values across corpora and no working group key, so v0.2 trains on **684 unique
events** (704 minus the 20 duplicates), split **stratified on level 70/15/15,
seed 20260918**, so all five levels appear in every split. The `group_id` leak
control in the old manifest is vacuous — see H-GROUP.

Five seeds, held-out test n=102 each. Full per-seed files in
`results/v02_judgelabels/`; corrected metrics in `results/v02_judgelabels/honest_metrics.json`.

| metric | v0.1 | v0.2 (5 seeds, range) | direction |
|---|---|---|---|
| macro-F1 (as published) | 0.358 | 0.284 – 0.320 | **regressed** |
| macro-F1 (abstention scored as a class) | not computed | 0.237 – 0.267 | — |
| weighted kappa (covered) | 0.4227 | 0.455 – 0.539 | up, but see H-KAPPACOV |
| weighted kappa (all rows) | not computed | 0.405 – 0.457 | — |
| dangerous under-classification (covered) | 0.333 | 0.039 – 0.133 | down, but see below |
| coverage | 0.881 | 0.500 – 0.618 | **collapsed** |
| calibration gate (FNR <= 0.05) | FALSE | 1 of 5 seeds | still failing |

**Label set for this kappa (H-YARDSTICK).** The v0.1 and v0.2 kappas above were measured against the **Qwen tier-2 judge labels** — the two-size, one-family weak labels, not a consensus label set. On that yardstick both fail the pre-registered wk1 gate (weighted kappa >= 0.70); the same models measured against the **v0.7 full-coverage consensus** pass it (v0.7: 0.838-0.860). The gate outcome here is therefore a property of the label set as much as of the model.

**What actually happened.** v0.2 does not classify better; it declines more.
Each seed makes 28–36 dangerous under-calls across 102 test events but *answers*
wrongly on only 2–8 of them — the abstention intercepts roughly 85% of the
model's own dangerous calls. That is why covered under-classification falls.
Read that as a property of the abstention policy, not of the classifier: the
metric cannot see an abstention at all (H-ABSTBLIND), and v0.1's 0.333 was
measured at coverage 0.881 while v0.2's is measured at ~0.55, so **the two
figures are not the same measurement**.

**Gates.** wk1 (kappa >= 0.70) fires on all five seeds — best 0.539 (label set:
Qwen tier-2 judge labels; see the H-YARDSTICK note above). wk2
(FNR <= 0.05) is met on 1 of 5 (seed 29, 0.0488). The wk1 kill rule therefore
still fires and the weights remain withheld under the 2026-09-07 decision.
Nothing in v0.2 reopens it.

**What must not be read from this.** v0.2 is not clinically validated, not
deployable, and not a safety product. Every figure above is measured against
the 31.0% inter-judge label noise (H-NOISE), on a split with no working leakage
guarantee (H-GROUP). Coverage is the number that carries the safety line here:
it is computed on all N and is not acceptance-gated.

## v0.3 — leakage-fixed retrain (2026-10-11)

**What changed: the evaluation split was contaminated and is now not.** Nothing
else about the recipe moved, so every number difference is attributable to the
split fix alone.

Perceptual deduplication (dHash ≤ 6 **and** pHash ≤ 8) over all 684 images found
**55 near-duplicate pairs that `event_id` never caught**, forming **7 genuine
visual families**. In the v0.2 row-wise split, **4 of those families spanned
train, val and test simultaneously** — so v0.2's held-out set contained images
effectively present in training. Source: `results/v03_groupdisjoint/dedup_report.json`.

v0.3 assigns **whole groups** to splits (650 groups over 684 rows) and asserts
group-disjointness and per-level coverage rather than assuming them; both
assertions pass. The test set grew 102 → 142 events, so intervals tighten.

### Results — 5 seeds, held-out n=142 each

| metric | v0.2 (contaminated) | v0.3 (group-disjoint) |
|---|---|---|
| **AUGRC** (joint risk, lower better; 0.5 = random) | not computed | **0.242 – 0.296** |
| macro-F1 as published | 0.284 – 0.320 | 0.253 – 0.316 |
| macro-F1, abstention scored as a class | 0.237 – 0.267 | 0.211 – 0.263 |
| weighted kappa (covered) | 0.455 – 0.539 | **0.555 – 0.620** |
| weighted kappa (all rows) | 0.405 – 0.457 | **0.465 – 0.526** |
| under-classification rate, **all-N** (both sides of the fraction) | not computed | **0.176 – 0.282** |
| under-classification rate (covered only) | 0.039 – 0.133 | 0.061 – 0.147 |
| coverage | 0.500 – 0.618 | **0.620 – 0.761** |
| MAE in IDDSI levels | not computed | **0.655 – 0.761** |
| wk1 (κ ≥ 0.70) | 0/5 | **0/5** (best 0.620) |
| wk2 calibration gate | 1/5 | **0/5** |

Per-seed files in `results/v03_groupdisjoint/`; AUGRC and the corrected all-N
rate in `selective_metrics.json`.

**Label set for this kappa (H-YARDSTICK).** The v0.3 kappas above were measured against the **v0.3 Qwen tier-2 judge labels** (the split fix changed which rows are held out, not which label set the rows carry). On that yardstick they fail the pre-registered wk1 gate (weighted kappa >= 0.70); the same models measured against the **v0.7 full-coverage consensus** pass it (v0.7: 0.838-0.860).

### We fixed the metric we said was broken

v0.2 disclosed **H-ABSTBLIND**: `dangerous_underclassification_rate_all` gates its
numerator on acceptance while dividing by all N, so an abstained
under-classification is never counted and abstaining more drives it toward zero.
v0.3 publishes the correctly-named **all-N rate (0.176–0.282)**, and adds **AUGRC**
— the joint, unconditional selective risk integrated over thresholds
(Traub et al., NeurIPS 2024, https://arxiv.org/abs/2407.01032). AUGRC cannot be
gamed by abstaining, because coverage sits inside the integrand. All five seeds
land well below the 0.5 random reference, so the model beats random at *every*
threshold rather than only at one chosen operating point.

### Where this sits against published work

The strongest published single-image result uses **physically rheology-verified
labels from certified clinicians** and a 14-model sweep: **61.74% exact match**
for its chosen Qwen pipeline and **70.43%** for GPT4.1, with **FNR 9.2%** and
**MAE 0.809** over IDDSI levels 0–7 on 115 images —
https://www.frontiersin.org/journals/nutrition/articles/10.3389/fnut.2026.1829703/full
(He, Mung, Kam & Chan, *Frontiers in Nutrition*, 2026-06-29; title: "Multimodal
LLM-driven IoT digital healthcare platform for intelligent dysphagia dietary
monitoring").

Our covered FNR (0.061–0.147) is in the same range as their 9.2%, and our MAE
(0.655–0.761) is lower than their 0.809 — **but that comparison flatters us**, for
two reasons stated plainly: their task spans 8 levels and ours 5, and **our test
labels are themselves only 31%-consistent**, so a low MAE partly measures
agreement with noisy targets rather than accuracy. Treat it as context, not a win.

Two corrections to that paragraph, both from the 2026-10-12 benchmark-framing pass
(`docs/BENCHMARK_FRAMING.md`, `hf/HAZARD_LOG.md` H-BENCH). First, the MAE quoted
here (0.655–0.761, per-seed values in `runs/v03/selective_metrics.json`) is an
**all-142-events, abstention-bypassed** figure, as are the exact-match and MAE
rows in the v0.5 and v0.7 tables below — a basis that was never stated and that
differs from the covered-events basis of the kappa rows beside them. Second, two
reasons understate it: the full list of reasons the two benchmarks are not the
same measurement has **ten** entries, including clinician-verified versus VLM
consensus labels, circularity between our labeller and our training labels, an
abstention mechanism the external pipeline does not have, a test split that is
64.8% L7 (a constant "L7" predictor already scores 64.81% exact on it, above
their 61.74%), and 108 versus 115 events. On the all-events basis with
abstentions scored as misses we are at 66.67–70.37%, which is **below** their
GPT-4.1 figure. The correct conclusion is that this comparison is uninformative
in both directions, not that it favours either side.

### Gates, and a challenge we are disclosing rather than hiding

The wk1 gate stays **κ ≥ 0.70**, unchanged. It fires on every v0.3 seed.

**Label set for this gate result (H-YARDSTICK).** "Fires on every v0.3 seed" is measured against the **v0.3 Qwen tier-2 judge labels**. The v0.4/v0.5 numbers were measured against the **v0.5 consensus** and also fail the wk1>=0.70 gate on it; the same models measured against the **v0.7 full-coverage consensus** pass it (v0.7: 0.838-0.860). A kappa in this card is not interpretable without the label set it was measured against — see H-YARDSTICK in `HAZARD_LOG.md`.

We are aware this gate is contested for our task type: Delgado & Tibau (2019),
"Why Cohen's Kappa should be avoided as a performance measure in ordinal
classification", *Pattern Recognition* 105:107192
(https://www.sciencedirect.com/science/article/abs/pii/S0167865518313413), argues
precisely against using Cohen's kappa as an ordinal model-selection metric. We
are **not** changing a pre-registered gate after seeing results, because doing so
would make every number in this release indefensible. We report the objection and
leave the gate standing.

### Labels are still the ceiling, and more machine labelling will not fix it

A third labeller from an architecturally **independent** model family
(step-5-preview, unrelated to Qwen) was run blind over 32 stratified images. It
agreed with the Qwen-8B tier-2 labeller on **40.0%** — no better than the two
Qwen sizes agree with each other (**42.9%** on the same sample). But **100% of
its disagreements with Qwen-8B were exactly ±1 IDDSI level** (19 of 19), and
**78.6%** of items had at least two of three labellers agreeing.

This is corroborated by the literature: a panel of 9 frontier LLMs from 7 families
reaches an effective independence of only **24.2%** and its best single judge
matches or beats the full ensemble
(https://ar5iv.arxiv.org/html/2605.29800), and Dawid–Skene-family label models
require **three to five** annotators before their likelihood is identifiable at
all (https://arxiv.org/pdf/1712.04577). We have two correlated labellers, which
is below that threshold.

**Conclusion: the label noise is irreducible from photographs.** L3/L4 is a
viscosity boundary and L4/L5 is an absolute-particle-size boundary; neither is
recoverable from a still image. Only the physically-tested route (300 desk
specimens) produces ground truth, and it has never been run.

## v0.5 — consensus labels, measured on a clean test set (2026-10-11)

**Two findings, one of which finally justifies the multi-labeller work and one of
which is a cost we did not expect.**

### Finding 1 — the Qwen family was the outlier, not the photographs

The v0.2/v0.3 labels came from two sizes of one model family. A third labeller
from an **architecturally independent** family (`step-5-preview`, unrelated to
Qwen) and a fourth (`MiniMax-M3.1f`) re-labelled the corpus blind.

```
Qwen-2B vs Qwen-8B   30.9% exact    68.8% within ±1 level
Step-5  vs Qwen-8B   47.1% exact    73.2% within ±1 level
Step-5  vs MM-3.1f   74.7% exact
```

Two unrelated families agree with each other at 74.7% while each agrees with
Qwen-8B at only 31–47%. On the ±1 scale that matters for an ordinal texture
scale, Step-5 agrees with Qwen-8B (73.2%) *better* than Qwen-2B does with its own
sibling (68.8%). Consensus labelling redistributes the classes substantially:
L6 172→87, L4 64→96, L5 47→75.

**This is the release model.** Trained on 4-way consensus labels, evaluated on
the clean test set below:

| metric (5 seeds, held-out n=108) | value |
|---|---|
| exact match | **0.444 – 0.546** |
| weighted kappa (all rows) | **0.540 – 0.566** |
| AUGRC (joint risk; 0.5 = random) | 0.339 – 0.397 |
| MAE in IDDSI levels | **0.722 – 0.778** |
| coverage | 0.732 – 0.845 |
| under-classification, all-N | 0.162 – 0.296 |

**Label set for this kappa (H-YARDSTICK).** The kappa in this table was measured against the **v0.5 consensus** label set, and on that yardstick it fails the pre-registered wk1 gate (weighted kappa >= 0.70; best seed 0.566). The same models measured against the **v0.7 full-coverage consensus** pass it — v0.7 scores 0.838–0.860 (5 seeds, held-out n=108, `results/v07_release/v07_seed*_metrics.json`). Nothing about the models changed between those two columns; only the yardstick did.

Against the **identical 108 events**, the model trained on the old Qwen tier-2
labels scores exact 0.370–0.454, kappa 0.315–0.399, MAE 0.907–1.019. Consensus
labels are worth roughly **+0.12 kappa and −0.19 levels of MAE** on the same
ground truth. That is a real improvement, not the scoring artefact an earlier
comparison of ours suggested. (Label set for both kappas: the v0.5 consensus;
both fail wk1>=0.70 on it — see the H-YARDSTICK note above.)

### Finding 2 — 22.2% of the corpus was never usable (152 of 684 events)

A labelability sweep judged the corpus with two independent model families. They
did **not** judge the same events: Step-5 returned a verdict on 683 of the 684
events, MiniMax-M3.1f on 512, and **510** events carry a verdict from both. So
every rate below names the denominator it was computed over:

```
                                    denominator   value
IS-THERE-FOOD agreement             493/510       96.7%
LEVELLABLE agreement                436/510       85.5%
unusable, share of whole corpus     152/684       22.2%
unusable, share of both-judged      152/510       29.8%
```

The binary judgement is answered almost perfectly by independent families; the
5-level judgement never is. 152 events — empty vessels, diagrams, no discernible
food mass — are unusable, and **every number published up to v0.4 was measured
with them in the test set**. All v0.5 numbers above are on the **cleaned** test
set of 108 events (`results/v05_release/unusable_event_ids.txt`).

**Which of the two unusable percentages to quote.** The corpus-wide 22.2% is the
one that survives contact with the data: all 152 ids are present in the 684-event
pre-clean manifests (`data/manifest_v04_consensus.jsonl`) and none survive in the
532-row cleaned manifest (`data/manifest_v07_weighted.jsonl`), i.e. 684 − 152 =
532 rows were actually removed. The 29.8% figure is a *conditional* rate over the
510 both-judged events, and it is only valid if all 152 unusable events fall
inside that 510 — the aggregate sweep output does not record which ones do, and
its own counts (683 + 512 − 510 = 685) cannot all be subsets of the same 684
events. **Quote 152/684 = 22.2% for any claim about the corpus; quote 152/510 =
29.8% only for a claim explicitly conditioned on the both-judged subset, and say
so.** Reproduce both with `python3 tools/denominator_audit.py`.

Note what this says about label quality: the questions we *can* verify are
answered reliably; the 5-level question is not. That is the honest shape of the
problem.

### Finding 3 — cleaning the corpus costs performance

Removing the 152 unusable events takes the corpus from 684 to 532 rows. Training
on that cleaned 532 rather than the full 684 is more defensible and measurably
worse, because the removals fall disproportionately on the hard middle levels
(L4 96→66, L5 75→65, L7 400→305):

| model, identical cleaned test set | exact | kappa | MAE |
|---|---|---|---|
| consensus labels, 400 train rows (**released**) | 0.444–0.546 | 0.540–0.566 | 0.722–0.778 |
| consensus labels, 314 train rows (cleaned corpus) | 0.426–0.500 | 0.498–0.547 | 0.778–0.870 |

**Label set for the kappa column (H-YARDSTICK):** both rows were measured against the **v0.5 consensus** label set and both fail the pre-registered wk1 gate (weighted kappa >= 0.70) on it. The same models measured against the **v0.7 full-coverage consensus** pass it (v0.7: 0.838-0.860).

We release the 400-row model and publish the cleaned-corpus run beside it rather
than quietly dropping it. The direction of the next data effort is therefore
**not** "delete more" — it is "acquire more mid-level labels", which is what the
physically-tested specimen protocol is for.

### Gates

**Unchanged and still failing.** wk1 (κ ≥ 0.70) does not pass on any seed;
best 0.566. The calibration gate is not met. The weights stay withheld under the
2026-09-07 decision. The κ-based gate remains contested by Delgado & Tibau (2019),
*Pattern Recognition* 105:107192
(https://www.sciencedirect.com/science/article/abs/pii/S0167865518313413); we
disclose the objection rather than re-register the gate after seeing results.

**Label set for this gate result (H-YARDSTICK).** "Best 0.566" is measured against the **v0.5 consensus** label set, so this FAIL is a property of that yardstick: the same models measured against the **v0.7 full-coverage consensus** pass wk1 (v0.7: 0.838-0.860, 5 seeds, held-out n=108). The withholding decision is unchanged either way — it rests on the dangerous under-classification and calibration failures, not on the kappa alone — but a reader must not carry "0.566 < 0.70" away as a statement about the models.

### Still no ground truth

Consensus is a *more consistent* label set (74.7% cross-family vs 30.9%
within-family), which is why it trains a better model. It is **not** evidence
that the labels are *correct*. Without a physically-measured anchor we still
cannot say whether Qwen carried a family bias or the newer families share one.
The 300-specimen protocol remains the only route to ground truth and has never
been run.

## v0.6 — the gate outcome depends on the yardstick, not the model

**This is the most important result in the project, and it is a negative one
about our own reporting.**

### We could not fail the gates reliably

We measured three models against two label sets. The same models that failed
every gate we had published also pass it, on a different yardstick:

| model (5 seeds each) | vs v0.5 consensus | vs v0.7 consensus | wk1 κ≥0.70 |
|---|---|---|---|
| v0.4 | 0/5 pass (κ 0.540–0.566) | **5/5 pass** (κ 0.736–0.808) | flips |
| v0.5 | 0/5 pass (κ 0.498–0.547) | **5/5 pass** (κ 0.704–0.795) | flips |
| v0.7 | 0/5 pass (κ 0.475–0.506) | **5/5 pass** (κ 0.838–0.860) | flips |

**Nothing about the models changed between those two columns — only the labels
they were scored against.** Five versions of "we have not met the gate" were
therefore partly an artefact of measuring against a label set we ourselves had
built and then improved. A reviewer is entitled to discount every performance
claim we have made to date, including this one.

*Read that table with H-YARDSTICK (`HAZARD_LOG.md`): the gate outcome above is a
property of the label set, not of the model. The left column (v0.5 consensus)
fails wk1>=0.70 on 0/5 seeds for every model; the right column (v0.7
full-coverage consensus) passes on 5/5. Both columns are the same checkpoints.*

### v0.7 on its own labels, stated with the caveat attached

Held-out test split: **108 events** of 532 rows (314 train / 110 val / 108 test),
all web-harvested, level distribution L3 10 / L4 11 / L5 7 / L6 10 / **L7 70**.
Coverage is 78.7 – 85.2%, so 85 – 92 of the 108 events carry a prediction and
16 – 23 are abstentions. Every row below now states its denominator and its
decode rule. Reproduce with
`docker exec -w /work/lanes/an05_external iddsi-train python -I tools/benchmark_framing.py`;
outputs `lane_runs/an05_external/benchmark_framing.json`, which also asserts that
the recomputed kappa and coverage match each run's published `metrics.json`
(max abs diff 0.0 over 5 seeds).
**Label set for every kappa below (H-YARDSTICK): the v0.7 full-coverage consensus**, 5 seeds, held-out n=108, `results/v07_release/v07_seed*_metrics.json`. On this yardstick wk1 passes. The v0.4/v0.5 numbers were measured against the v0.5 consensus and fail the wk1>=0.70 gate; the same models measured against the v0.7 full-coverage consensus pass (v0.7: 0.838-0.860). The v0.7-vs-v0.7 comparison is the circular one and is discounted accordingly.
| metric | v0.7, 5 seeds, held-out n=108 |
|---|---|
| weighted kappa, covered (the wk1 quantity) | **0.838 – 0.860** |
| weighted kappa, all rows | 0.793 – 0.825 |
| **exact match, covered events, CORAL decode** | **80.90 – 84.88%** |
| **MAE in IDDSI levels, covered events, CORAL decode** | **0.244 – 0.292** |
| exact match, all 108 events, abstentions scored as misses | 0.667 – 0.704 |
| MAE in IDDSI levels, all 108 events, abstention bypassed | 0.380 – 0.444 |
| exact match, all 108 events, CORAL decode, abstention bypassed | 0.694 – 0.741 |
| exact match, covered, macro (unweighted per-class) | 0.375 – 0.480 |
| under-classification, all-N | 0.093 – 0.130 |
| **baseline: always answer L7** (the majority test class, 70/108) | **0.648 exact, MAE 0.898** |
| baseline: draw a label from the empirical test prior | 0.452 exact |

Two rows need reading against the baselines, not against each other. Against
always-L7 on the same 108 events our exact match is only **1.9 – 12.0 points
higher depending on the denominator** — 8.2 – 12.0 points on the covered basis
against the matched covered-subset baseline of 70.65 – 76.47%, and only 1.9 – 5.6
points on the all-events basis with abstentions scored as misses. On MAE the
signal is real: our covered MAE is **0.356 – 0.412 of** the matched constant
predictor's (0.600 – 0.761 levels), about 60% lower. Per-class exact match on
covered events is 100% on L7, 77.8 – 90.0% on L3, 0 – 50.0% on L4, and **0.0% on
L5 and L6 in every one of the five seeds**. This is an L7 detector with a
partial L3 detector, not a 5-level classifier. Flipping the 22 of 108 test labels
whose two labellers disagreed to the other labeller's vote moves exact match to
79.78 – 87.06% and MAE to 0.188 – 0.326 (`lane_runs/an05_external/label_noise_band.json`) — about
±3 points of our headline is a property of the yardstick, not the model.

**A correction to what this card printed before.** The v0.7 exact-match and MAE
rows in the previous revision (0.694 – 0.741 and 0.380 – 0.444) were computed
with the abstention mechanism bypassed: all 108 events forced to a CORAL level,
including the 16 – 23 the model itself called `unclear`. That is a different
quantity from the covered-events kappa printed three rows above it, and it was
placed next to an external number without saying so. All four candidate rules
were tested against the old printed ranges; only "CORAL over all 108 events,
abstention bypassed" reproduces both
(`benchmark_framing.json → summary.decode_variant_audit`). Both bases are now
printed, labelled.

### Where this sits against published work — and why the comparison is uninformative

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

The one thing that is not circular: **v0.4 and v0.5 never saw the v0.7 labels,
and they score 0.736–0.808 and 0.704–0.795 against them.** Models trained on
three different label sets all agree more with the v0.7 labels than with the
v0.5 ones. That is evidence the labels improved, not evidence they are correct.

### What changed in v0.7

A coverage-closure run gave a second independent judgement to the 314 events that
previously had only one, so **337 of the 532 cleaned events now carry two genuine
independent judgements** (251 unanimous, 86 split) instead of an unverifiable
single call. Cross-family agreement is stable and reproducible at **~74.5%**
(251/337 = 74.5%, and 166/223 = 74.4% on the earlier subset).

Two attempts to improve training this way are recorded because they failed:
dropping every disputed event kept only 31% of the corpus and destroyed the middle
classes (L4 66→7, L5 65→4, L6 72→5); weighting by agreement produced a replication
factor of 1 for every event, because agreement is all-or-nothing inside a level
rather than graded. v0.7 therefore changes only the **labels**, with the training
recipe byte-identical to every prior version.

### What this does not establish

No number here is a claim about real-world accuracy. Two independent model
families agree on an IDDSI level from a photograph about three times in four, the
"is there food" question is reliable (96.7% cross-family, 493/510 events both
families judged) while the level is not,
and **the physically-tested protocol — 300 specimens, ~10 hours, already fully
specified — remains the only route to ground truth and has never been run.** Until
it is, every gate result here is conditional on a yardstick we built ourselves.

## CORAL ordinal-decode bug and fix

A CORAL ordinal-decode bug made levels L4/L5/L6 unreachable. Argmax over sigmoid-difference class masses could never fire for interior classes. This was proven numerically from the checkpoint. Fixed with the ordinal rank rule:

```text
level = #{k : P(y>k) > 0.5}
```

| Condition | macro-F1 | coverage | ECE | temperature | Notes |
| --- | --- | --- | --- | --- | --- |
| Before fix | 0.088 | 0.70 | 0.138 | 2.83 | Interior levels unreachable |
| After fix, n=109 held out | 0.358 | 0.88 | 0.091 | 0.93 | All levels predicted |

Weighted kappa 0.42 flat in both, measured against noise.

**Label set for this kappa (H-YARDSTICK).** The 0.42 is measured against the early **Qwen tier-2 judge labels** (the pre-consensus label set of that run), not against any consensus label set, and it fails the wk1>=0.70 gate on that yardstick. The same models measured against the **v0.7 full-coverage consensus** pass it (v0.7: 0.838-0.860).

## Pre-registered gates and which fired

- wk1: weighted kappa >=0.70, else kill photo-classification.
- wk2: external macro-F1 >=0.75 plus dangerous under-classification <=5% at <=30% abstention.
- Observed dangerous under-classification is 33%.

**Label set for the wk1 gate (H-YARDSTICK).** wk1 is a single unconditional number, but the kappa it is applied to is only meaningful with its label set attached. The v0.4/v0.5 numbers were measured against the **v0.5 consensus** and fail the wk1>=0.70 gate; the same models measured against the **v0.7 full-coverage consensus** pass (v0.7: 0.838-0.860). The gate is not weakened and not re-registered — the annotation states which yardstick produced each verdict.

The model fails all of them. The wk1 kill rule FIRES. Photo-classification is killed as a product capability. This is why only code and results ship.

## Why the weights are withheld

The photo-classification head fails the pre-registered safety gates, with dangerous under-classification at 33%. A downloadable classifier would be used for verdicts whatever the card says, and a disclaimer does not cure a claim. Withholding the weights is the mitigation.

## Why the dataset is withheld

The harvested dataset is unpublishable on licence grounds. Of 587 unique events exactly 2 would ship cleanly. 592 rows carry unverified open-images/openverse licences, 7 are CC BY-SA vs the BY-NC-SA main release, 109 have unresolved redistribution status.

## Flow-test grader — the working deliverable

The flow-test grader is the part that WORKS and the genuinely useful artifact of this release. It is a deterministic CV pipeline for the 10 ml / 10 s IDDSI syringe flow test from smartphone video. MAE 0.065 ml on a 100-video bench. It grades L0-L4.

- Ships in this release: see `flowtest/`.
- This is distinct from photo classification. A still photograph is never used to grade liquid flow.

## Remedy path

Physically-tested desk specimens: 60 base foods x 5 levels = 300 specimens, one operator, protocol-following, test-filmed (docs/CAPTURE_PROTOCOL.md), with a filmed re-test subsample for intra-rater reliability. Labels will be single-operator physical tests, never clinician-validated.

## Intended use

- Research preview for reproducible methods work on IDDSI-related vision, using code and recorded eval results.
- Culinary-education use only, with confirmation by the applicable physical test.
- Research plus culinary-education use only. No other use is in scope.

## Out of scope

- No safe or unsafe verdicts. The release never issues a verdict.
- No dietary, clinical, or patient claims. Not for assessing swallowing, aspiration risk, or suitability for any person. Not for prescribing or modifying a texture.
- Not IDDSI-endorsed. This project is independent of IDDSI and of every prior-art group named below.
- An IDDSI-eating-advisor would be SaMD (HK TR-007; NMPA class II-III; EU MDR Rule 11). This release is engineered to stay OUT of that class by never issuing a verdict.

## How to reproduce — repo structure map

- `harvest/` — harvesting of CC-licence-candidate images, licence tracking, and two-tier VLM weak labelling.
- `trainer/` — SigLIP-2 CORAL head training, eval metrics, and calibrated abstention.
- `flowtest/` — working smartphone-video grader for the 10 ml / 10 s IDDSI syringe flow test. Grades L0-L4.
- `synth/` — cloud-generated development images. Validated cloud-generated images supported the pipeline; synthetic rows are never eval ground truth.
- `dataschema/` — manifests, dataset card, and datasheet wording.
- `crowd/` — capture app used for guided capture work.
- `ops/` — release checks and runbooks, including the fail-closed licence verification and its regression test.
- `paper/` — write-up of the methods, the negative finding, and the remedy path.

## Honest claims

- The ambition was "first publicly downloadable IDDSI-specific smartphone dataset + fine-tuned model under explicit reuse licences". v0 does NOT meet it: the dataset is withheld on licence grounds and the weights are withheld on safety gates.
- The flow-test grader claim stands: "first open-source smartphone grader for the 10 ml IDDSI Flow Test".

## Never-claims

- No novelty claim over prior IDDSI vision work. Prior art: PolyU ViscoCam (2021), HKUST OptiTexture (91.96% with an optical sensor), EdUHK Frontiers (2026, 61.74%), Safe Plate (AU).
- No safe or unsafe verdicts, no dietary, clinical, or patient claims.
- Not IDDSI-endorsed. Research plus culinary-education use only.

## Citation

IDDSI-Open research preview, Hugging Face org `lingualeap`, release shape decided 2026-09-07. Code licence apache-2.0. Weights and harvested image dataset withheld for the safety and licence reasons stated above.
