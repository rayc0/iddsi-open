---
pretty_name: IDDSI-Open
license: cc-by-nc-sa-4.0
task_categories:
  - image-classification
  - video-classification
language:
  - en
  - zh
tags:
  - iddsi
  - food-texture
  - physical-test-ground-truth
  - abstention
size_categories:
  - [REPLACE_AFTER_COUNTING]
---

# IDDSI-Open [VERSION]

> Draft template. Replace every bracketed field and remove unverified metadata before release. No accuracy or dataset-size number belongs here until produced by the locked evaluation/validated manifest.

## Summary

IDDSI-Open is a smartphone image/video research and food-service QA dataset pairing guided food-test or 10 mL/10-second syringe-flow captures with contemporaneous physical-test records. Real release labels are independently assessed by an RD and SLP, with third-person adjudication for disagreement. Synthetic development records are explicitly tagged and excluded from physically-tested release claims.

## Intended use and mandatory disclaimer

> Research demonstration for culinary education. Estimates visual similarity to IDDSI descriptors; does not perform official IDDSI tests, assess swallowing, determine suitability for any person, or decide whether food is safe to consume.

Not the official IDDSI website. Not IDDSI-endorsed. Never use this dataset or a derived model to prescribe a texture, output “safe/unsafe/pass,” estimate aspiration risk, or replace clinician-prescribed texture and physical testing.

## Dataset structure

- Schema: [`schema/event.schema.json`](../schema/event.schema.json)
- Capture protocol: [`docs/CAPTURE_PROTOCOL.md`](CAPTURE_PROTOCOL.md)
- Records: one row per independent preparation/test event
- Media: relative paths to a 45° plate image and guided clips for food, or side-view syringe-flow video for liquid
- Splits: `train`, `val`, `external_test`, grouped by recipe, batch, and kitchen
- Sources: `real` or `synthetic`; synthetic records cannot enter `external_test`
- Capture conditions: pseudonymous operator, lighting/plate category, serving temperature, and cuisine tags for coverage reporting

### Counts

[INSERT MANIFEST-DERIVED TABLES BY SPLIT, SOURCE, SAMPLE KIND, ADJUDICATED LEVEL, KITCHEN, PHONE, CUISINE, AND CAPTURE CONDITION. REPORT MISSINGNESS.]

## Ground truth

[DESCRIBE CURRENT OFFICIAL TEST PROCEDURE USED, SERVING CONDITION, INDEPENDENT RD/SLP ASSESSMENT, BLINDING, ADJUDICATION, EXCLUSIONS, AND AUDIT.] Declared levels are not ground truth. Model outputs are not ground truth.

## Limitations

Static RGB does not directly measure flow, hardness, adhesiveness, cohesiveness, moisture separation, or swallowing suitability. Performance may change across recipes, boundaries, cuisines, kitchens, phones, lighting, plates, temperatures, and operators. An “unclear—perform physical test” abstention is a required system behavior, not an optional UI detail.

[ADD MEASURED REPRESENTATION GAPS, LABEL UNCERTAINTY, FAILURE CASES, AND KNOWN BIASES.]

## Evaluation and claims

[INSERT ONLY LOCKED, REPRODUCIBLE RESULTS: per-level confusion, macro-F1, weighted kappa, dangerous-direction under-classification, calibration/risk-coverage, abstention, flow residual-volume error/non-boundary agreement, 95% confidence intervals, and failures. Do not compare scores across non-equivalent private datasets.]

Permitted dated claim, only after verifying the stated artifacts and prior-art search:

> To our knowledge, as of 31 August 2026, the first publicly downloadable IDDSI-specific smartphone-food dataset and fine-tuned model released under explicit reuse licences, with physically tested Chinese/Cantonese soft-meal labels.

Separate flow claim, only after release of the implementation:

> First open-source smartphone grader for the 10 ml/10-second IDDSI Flow Test.

Do not claim first IDDSI AI, first image classifier, first dataset/benchmark, clinical validation, diagnosis, medical-grade performance, or food/swallowing safety.

## Licensing and citation

- Original dataset/media: CC BY-NC-SA 4.0 after per-asset provenance and consent audit.
- Code and legally eligible released weights: Apache-2.0; base-model and third-party terms still apply.
- Citation: [DOI/BIBTEX PLACEHOLDER]
- Attribution: [FINAL RELEASE ATTRIBUTION]

See [`docs/LICENSE_PLAN.md`](LICENSE_PLAN.md). IDDSI names/materials and third-party assets are not relicensed by this repository.

## Maintainers

LinguaLeap Tech Limited (HK). [PUBLIC CONTACT, GOVERNANCE, UPDATE/TAKEDOWN POLICY, AND VERSION HISTORY]
