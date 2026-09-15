# Datasheet for IDDSI-Open [VERSION]

> Template status: replace every `[PLACEHOLDER]`, remove unused prompts, and derive all counts from the validated manifest. Do not publish invented, projected, or target results.

## Motivation

- **For what purpose was the dataset created?** [Describe the open research and food-service QA benchmark, guided physical-test capture, and reproducible baselines.]
- **Who created/funded it?** LinguaLeap Tech Limited (HK); HKU TechUp-funded project. [Add grant wording approved for publication and contributor roles.]
- **Who should use it?** [Researchers, RDs/SLPs studying test execution, food-service QA teams, and other scoped users.]
- **Who should not use it and for what?** It is not for diagnosis, patient-specific texture prescription, swallowing assessment, or deciding whether food is safe to consume.

## Composition

- Total independent events: [MANIFEST-DERIVED COUNT]
- Real / synthetic events: [COUNTS]
- Food / liquid events and L0–L7 distribution: [COUNTS]
- Train / validation / locked external-test distribution: [COUNTS]
- Unique recipes / batches / kitchens / phones: [COUNTS]
- Cuisine, lighting, plate, phone, serving-temperature, and operator coverage: [OBSERVED COUNTS AND GAPS]
- Media per event and file formats: [SUMMARY]
- Missing, unresolved, invalid, or quarantined events: [COUNTS AND REASONS]

One row is one independent food-preparation or syringe-flow event, never a frame. `source=synthetic` material is pipeline-development material and is excluded from claims about physically tested real-data splits.

## Collection process

- Capture dates/locations: [RANGES; USE NON-IDENTIFYING LOCATION DESCRIPTIONS]
- Capture protocol version: [VERSION/LINK]
- Equipment and phone mix: [OBSERVED INVENTORY]
- Sampling and assignment: [HOW RECIPES, LEVELS, KITCHENS, PHONES, PLATES, LIGHTING, AND SPLITS WERE ASSIGNED]
- Consent and no-face review: [PROCESS AND AUDIT RESULT]
- Retakes, exclusions, and deviations: [PROCESS AND COUNTS]

## Ground truth and quality control

Labels for real release events must come from contemporaneous physical tests independently performed by an RD and SLP. RD/SLP disagreements require third-person adjudication. State:

- physical tests used by sample kind/level: [DETAILS]
- assessor qualifications and blinding: [DETAILS]
- agreement and adjudication process: [DETAILS]
- unresolved/excluded records: [COUNTS/REASONS]
- quality checks and validator version: [COMMAND, VERSION, MANIFEST CHECKSUM]

Do not describe declared recipe levels, visual guesses, synthetic labels, or model predictions as physical-test ground truth.

## Preprocessing, cleaning, and derived data

- Raw-media preservation and checksum process: [DETAILS]
- Transcoding, resizing, rotation, colour processing, audio removal, metadata stripping, or redaction: [EVERY TRANSFORMATION, SOFTWARE/VERSION, AND WHETHER RAW FILES ARE RELEASED]
- Frame extraction or crop generation: [PROCESS AND PARENT EVENT LINKAGE]
- Duplicate/near-duplicate detection: [PROCESS AND RESULTS]
- Exclusion/quarantine rules: [RULES AND MANIFEST-DERIVED COUNTS]

Derived frames and crops retain the parent `event_id` and split. They are never counted as new independent preparations. Document any released raw personal or sensitive information; the expected answer is none because faces and patient/clinical data are prohibited, but verify rather than assume.

## Splits and leakage control

Splits are grouped independently by `recipe_id`, `batch_id`, and `kitchen_id`; none may cross train, validation, or external test. [REPORT PHONE/CUISINE HOLDOUT DESIGN.] The external set remains locked until the stated evaluation. Report any post-lock access or deviation.

## Uses, limitations, and risks

Appropriate uses: [RESEARCH/EDUCATION/QA SCOPE]. Known limitations must include that static RGB cannot directly measure flow, hardness, adhesiveness, cohesiveness, or swallowing suitability; L4/L5 and L5/L6 boundaries can require physical tests; coverage may not generalize to new kitchens, cuisines, phones, lighting, temperatures, or operators; and uncertainty/abstention is mandatory.

Required regulatory disclaimer (verbatim):

> Research demonstration for culinary education. Estimates visual similarity to IDDSI descriptors; does not perform official IDDSI tests, assess swallowing, determine suitability for any person, or decide whether food is safe to consume.

Also display: Not the official IDDSI website. Not IDDSI-endorsed. Tests were performed by LinguaLeap and participating assessors; they are not IDDSI certification.

No patient profile, diagnosis, prescribed level, “safe/unsafe/pass,” aspiration-risk output, or autonomous treatment advice is contained in the dataset. A disclaimer does not override intended use; obtain formal regulatory advice before patient-facing or clinical deployment.

## Honest claims

Use only after a dated prior-art review and after the release artifacts actually exist:

> To our knowledge, as of 31 August 2026, the first publicly downloadable IDDSI-specific smartphone-food dataset and fine-tuned model released under explicit reuse licences, with physically tested Chinese/Cantonese soft-meal labels.

For the flow component, use only if the released implementation supports it:

> First open-source smartphone grader for the 10 ml/10-second IDDSI Flow Test.

Never claim “first IDDSI AI,” “first image classifier,” “first dataset/benchmark,” “clinically validated,” “diagnostic,” “medical-grade,” “safe-to-eat,” or comparable performance against an inaccessible/different dataset. Publish held-out results, confidence intervals, abstention coverage, failures, and protocol deviations only after they are measured.

## Distribution, licensing, and maintenance

- Dataset DOI/repository/version/checksums: [DETAILS]
- Data licence: CC BY-NC-SA 4.0, subject to the release provenance audit.
- Code/eligible weights licence: Apache-2.0, subject to base-model and training-data rights review.
- Third-party assets and separate terms: [INVENTORY OR NONE]
- Maintainer/contact and issue process: [DETAILS]
- Update, deprecation, takedown, and consent-withdrawal process: [DETAILS]
- Security/privacy reporting: [DETAILS]
