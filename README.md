# IDDSI-Open research preview — HONEST release card

> Research preview. Release shape decided 2026-09-07, locked. Published on Hugging Face under org `lingualeap` as the public face of this release.


> Also on Hugging Face: [lingualeap/iddsi-open-siglip2-v0](https://huggingface.co/lingualeap/iddsi-open-siglip2-v0) (same release; the two PDFs live here on GitHub).

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

## Headline negative finding

The two VLM judges agree on only 22.6% of the same images, which is chance level for 5 classes. THEREFORE every label-derived metric below is measured against label noise and means nothing about real-world accuracy.

This 22.6% inter-judge agreement IS the headline negative finding: web photos without a physical scale fiducial cannot be weakly labelled for IDDSI texture levels, even by strong VLMs, because middle levels (L4/L5/L6) require absolute size and flow information a photograph doesn't carry.

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
