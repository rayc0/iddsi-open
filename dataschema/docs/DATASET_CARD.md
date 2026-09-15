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
  - TBD
---

# IDDSI-Open v0.1.0 (Research Preview)

> Draft dataset card. Replace every TBD field and remove unverified metadata before release. No accuracy or dataset-size number belongs here until produced by the locked evaluation/validated manifest.

## Summary

IDDSI-Open is a smartphone image/video research and food-service QA dataset pairing guided food-test or 10 mL/10-second syringe-flow captures with contemporaneous physical-test records. Real release labels are independently assessed by an RD and SLP, with third-person adjudication for disagreement. Synthetic development records are explicitly tagged and excluded from physically-tested release claims.

**Data strategy:** synthetic-first / real-weak / real-crowd. Synthetic data is used for pipeline development only. Real-weak data (licence-eligible third-party images with visual weak labels) is used for robustness work. Real-crowd data (voluntary smartphone uploads) is quarantined and never promoted to physically tested ground truth without the full capture protocol. Real-tested release rows require the frozen capture protocol with independent RD/SLP physical tests and adjudication.

**Tests performed by LinguaLeap staff, not clinicians.** This is a research preview, not a validated safety product.

## Intended use and mandatory disclaimer

> Research demonstration for culinary education. Estimates visual similarity to IDDSI descriptors; does not perform official IDDSI tests, assess swallowing, determine suitability for any person, or decide whether food is safe to consume.

Not the official IDDSI website. Not IDDSI-endorsed. Never use this dataset or a derived model to prescribe a texture, output "safe/unsafe/pass," estimate aspiration risk, or replace clinician-prescribed texture and physical testing.

## Dataset structure

- Schema: [`schema/event.schema.json`](../schema/event.schema.json)
- Capture protocol: [`docs/CAPTURE_PROTOCOL.md`](CAPTURE_PROTOCOL.md)
- Records: one row per independent preparation/test event
- Media: relative paths to a 45° plate image and guided clips for food, or side-view syringe-flow video for liquid
- Splits: `train`, `val`, `external_test`, grouped by recipe, batch, and kitchen
- Sources: `real` or `synthetic`; synthetic records cannot enter `external_test`
- Capture conditions: pseudonymous operator, lighting/plate category, serving temperature, and cuisine tags for coverage reporting

### Counts

TBD — manifest-derived tables by split, source, sample kind, adjudicated level, kitchen, phone, cuisine, and capture condition. Report missingness.

### Real-weak pilot provenance snapshot

The current real-weak pilot is a pre-release provenance snapshot, not a released or physically tested dataset. Its locally inspected manifests contain 100 events, 100 matching attribution rows, and 100 referenced media files:

| Provider | Events | Licence recorded in the per-asset manifest |
|---|---:|---|
| Nutrition5k | 95 | CC BY 4.0 × 95 |
| Wikimedia Commons | 5 | CC BY-SA × 4 (CC BY-SA 2.5 × 1; CC BY-SA 4.0 × 3); CC0 × 1 |

All 100 events are tagged `source=real-weak`. Labelling is deferred for every row (`weak_label_record.rule=deferred` and `model_id=deferred`): `level_weak` and `level_adjudicated` are null, and `is_ground_truth=false`. These images therefore have no IDDSI ground-truth labels and must not be counted as physically tested data or used for accuracy claims. Per-asset attribution and licence terms remain controlling; this summary does not relicense third-party media.

## Ground truth

TBD — describe current official test procedure used, serving condition, staff-performed on-camera tests, blinding, exclusions, and audit. Declared levels are not ground truth. Model outputs are not ground truth.

## Limitations

Static RGB does not directly measure flow, hardness, adhesiveness, cohesiveness, moisture separation, or swallowing suitability. Performance may change across recipes, boundaries, cuisines, kitchens, phones, lighting, plates, temperatures, and operators. An "unclear—perform physical test" abstention is a required system behavior, not an optional UI detail.

**Synthetic-first pilot evidence:** the first audited Level 4 generation pilot requested 24 accepted candidates and exhausted 72 attempts with 0/24 accepted. The judge correctly rejected audited examples showing failure modes including retained rice-grain texture, added garnish, and camera-facet language rendered as a literal smartphone/device in the food scene. The prompt bank was subsequently revised, but no replacement pilot result is claimed here. Synthetic records remain pipeline-development material only and provide no evidence of real-food performance, physical-test agreement, or safety.

TBD — add measured representation gaps, label uncertainty, failure cases, and known biases.

## Evaluation and claims

TBD — insert only locked, reproducible results: per-level confusion, macro-F1, weighted kappa, dangerous-direction under-classification, calibration/risk-coverage, abstention, flow residual-volume error/non-boundary agreement, 95% confidence intervals, and failures. Do not compare scores across non-equivalent private datasets.

Permitted dated claim, only after verifying the stated artifacts and prior-art search:

> To our knowledge, as of 31 August 2026, the first publicly downloadable IDDSI-specific smartphone-food dataset and fine-tuned model released under explicit reuse licences, with physically tested Chinese/Cantonese soft-meal labels.

Separate flow claim, only after release of the implementation:

> First open-source smartphone grader for the 10 ml/10-second IDDSI Flow Test.

Do not claim first IDDSI AI, first image classifier, first dataset/benchmark, clinical validation, diagnosis, medical-grade performance, or food/swallowing safety.

## Licensing and citation

- Original dataset/media: CC BY-NC-SA 4.0 after per-asset provenance and consent audit.
- Code and legally eligible released weights: Apache-2.0; base-model and third-party terms still apply.
- Citation: TBD
- Attribution: TBD

See [`docs/LICENSE_PLAN.md`](LICENSE_PLAN.md). IDDSI names/materials and third-party assets are not relicensed by this repository.

## Maintainers

LinguaLeap Tech Limited (HK). TBD — public contact, governance, update/takedown policy, and version history.
