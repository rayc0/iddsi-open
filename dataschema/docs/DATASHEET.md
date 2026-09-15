# Datasheet for IDDSI-Open v0.1.0 (Research Preview)

> Template status: replace every TBD, remove unused prompts, and derive all counts from the validated manifest. Do not publish invented, projected, or target results.

## Motivation

- **For what purpose was the dataset created?** Open research and food-service QA benchmark, guided physical-test capture, and reproducible baselines. The dataset pairs guided smartphone media with contemporaneous physical-test records to support research on visual compatibility with IDDSI descriptors.
- **Who created/funded it?** LinguaLeap Tech Limited (HK); HKU TechUp-funded project. TBD — add grant wording approved for publication and contributor roles.
- **Who should use it?** Researchers, RDs/SLPs studying test execution, food-service QA teams, and other scoped users.
- **Who should not use it and for what?** It is not for diagnosis, patient-specific texture prescription, swallowing assessment, or deciding whether food is safe to consume.

## Composition

- Total independent events: TBD
- Real / synthetic events: TBD
- Food / liquid events and L0–L7 distribution: TBD
- Train / validation / locked external-test distribution: TBD
- Unique recipes / batches / kitchens / phones: TBD
- Cuisine, lighting, plate, phone, serving-temperature, and operator coverage: TBD
- Media per event and file formats: TBD
- Missing, unresolved, invalid, or quarantined events: TBD

One row is one independent food-preparation or syringe-flow event, never a frame. `source=synthetic` material is pipeline-development material and is excluded from claims about physically tested real-data splits.

**Data strategy:** synthetic-first / real-weak / real-crowd. Synthetic data is used for pipeline development only. Real-weak data (licence-eligible third-party images with visual weak labels) is used for robustness work. Real-crowd data (voluntary smartphone uploads) is quarantined and never promoted to physically tested ground truth without the full capture protocol.

## Collection process

- Capture dates/locations: TBD — use non-identifying location descriptions
- Capture protocol version: TBD
- Equipment and phone mix: TBD
- Sampling and assignment: TBD — how recipes, levels, kitchens, phones, plates, lighting, and splits were assigned
- Consent and no-face review: TBD — process and audit result
- Retakes, exclusions, and deviations: TBD — process and counts

**Tests performed by LinguaLeap staff, not clinicians.** Kitchen staff or interns capture the media and perform the official IDDSI test methods on camera; the recorded test itself is the label evidence. Independent RD/SLP assessment is a planned future validation tier, not part of this release.

## Ground truth and quality control

Labels for real release events come from contemporaneous physical IDDSI tests performed on camera by LinguaLeap staff (recipe-declared level recorded separately). RD/SLP adjudication is a future validation tier and is not claimed for this release.

- physical tests used by sample kind/level: TBD
- assessor qualifications and blinding: TBD
- agreement and adjudication process: TBD
- unresolved/excluded records: TBD
- quality checks and validator version: TBD — command, version, manifest checksum

Do not describe declared recipe levels, visual guesses, synthetic labels, or model predictions as physical-test ground truth.

## Preprocessing, cleaning, and derived data

- Raw-media preservation and checksum process: TBD
- Transcoding, resizing, rotation, colour processing, audio removal, metadata stripping, or redaction: TBD — every transformation, software/version, and whether raw files are released
- Frame extraction or crop generation: TBD — process and parent event linkage
- Duplicate/near-duplicate detection: TBD — process and results
- Exclusion/quarantine rules: TBD — rules and manifest-derived counts

Derived frames and crops retain the parent `event_id` and split. They are never counted as new independent preparations. Document any released raw personal or sensitive information; the expected answer is none because faces and patient/clinical data are prohibited, but verify rather than assume.

## Splits and leakage control

Splits are grouped independently by `recipe_id`, `batch_id`, and `kitchen_id`; none may cross train, validation, or external test. TBD — report phone/cuisine holdout design. The external set remains locked until the stated evaluation. Report any post-lock access or deviation.

## Uses, limitations, and risks

Appropriate uses: research, education, and food-service QA scope. Known limitations must include that static RGB cannot directly measure flow, hardness, adhesiveness, cohesiveness, or swallowing suitability; L4/L5 and L5/L6 boundaries can require physical tests; coverage may not generalize to new kitchens, cuisines, phones, lighting, temperatures, or operators; and uncertainty/abstention is mandatory.

Required regulatory disclaimer (verbatim):

> Research demonstration for culinary education. Estimates visual similarity to IDDSI descriptors; does not perform official IDDSI tests, assess swallowing, determine suitability for any person, or decide whether food is safe to consume.

Also display: Not the official IDDSI website. Not IDDSI-endorsed. Tests were performed by LinguaLeap and participating assessors; they are not IDDSI certification.

No patient profile, diagnosis, prescribed level, "safe/unsafe/pass," aspiration-risk output, or autonomous treatment advice is contained in the dataset. A disclaimer does not override intended use; obtain formal regulatory advice before patient-facing or clinical deployment.

## Honest claims

Use only after a dated prior-art review and after the release artifacts actually exist:

> To our knowledge, as of 31 August 2026, the first publicly downloadable IDDSI-specific smartphone-food dataset and fine-tuned model released under explicit reuse licences, with physically tested Chinese/Cantonese soft-meal labels.

For the flow component, use only if the released implementation supports it:

> First open-source smartphone grader for the 10 ml/10-second IDDSI Flow Test.

Never claim "first IDDSI AI," "first image classifier," "first dataset/benchmark," "clinically validated," "diagnostic," "medical-grade," "safe-to-eat," or comparable performance against an inaccessible/different dataset. Publish held-out results, confidence intervals, abstention coverage, failures, and protocol deviations only after they are measured.

**This is a research preview.** Without written HKU approval or passing gates, release only as a Y06 research preview/test assistant, not a validated safety product.

## Distribution, licensing, and maintenance

- Dataset DOI/repository/version/checksums: TBD
- Data licence: CC BY-NC-SA 4.0, subject to the release provenance audit.
- Code/eligible weights licence: Apache-2.0, subject to base-model and training-data rights review.
- Third-party assets and separate terms: TBD
- Maintainer/contact and issue process: TBD
- Update, deprecation, takedown, and consent-withdrawal process: TBD
- Security/privacy reporting: TBD
