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

## Pre-registered gates and which fired

- wk1: weighted kappa >=0.70, else kill photo-classification.
- wk2: external macro-F1 >=0.75 plus dangerous under-classification <=5% at <=30% abstention.
- Observed dangerous under-classification is 33%.

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
