## Verdict

**Go with modifications — confidence 88%, reversibility 🟡.** Build a camera-guided IDDSI **testing and QA tool**, not a single-photo safety oracle. The dataset and milestone are strong; the proposed “first IDDSI texture AI” claim is not defensible.

### 1. Prior art and defensible novelty

Material prior art already exists:

- A 2026 Hong Kong study classifies IDDSI levels from one smartphone image. Its Qwen pipeline achieved only **61.74% exact accuracy** on 115 physically tested meals, although safety-biased false-negative performance was better ([Frontiers](https://www.frontiersin.org/journals/nutrition/articles/10.3389/fnut.2026.1829703/full)).
- HKUST’s **OptiTexture** combines RGB with a <$20 light-scattering sensor: **91.96% accuracy** across L3–L6 versus approximately **69.64% for vision alone**, on 112 samples ([ACM paper](https://dl.acm.org/doi/10.1145/3810209)).
- Australia’s commercial **Safe Plate** already advertises photo/video IDDSI analysis and corrective guidance ([Safe Plate](https://www.safeplate.au/)).
- The official IDDSI app provides descriptors and test instructions, but not AI classification ([App Store](https://apps.apple.com/gb/app/iddsi/id1145593063)).
- FoodSense is an open 66,842-image general food-texture dataset, though not IDDSI-labelled ([Hugging Face](https://huggingface.co/datasets/sababishraq/foodsense-dataset)).

Therefore, do **not** say “first IDDSI texture AI.” A potentially defensible, dated claim is:

> “To our knowledge, the first openly licensed, physically IDDSI-test-verified Cantonese/Chinese texture-food and syringe-flow video benchmark, with an open camera-guided QA demo.”

The novelty is the combination of **open data + physical-test ground truth + Chinese cuisine + video measurement + abstention**, not AI itself.

Important: indexed SeniorDeli pages already say the tool is “validated by HKU” and available for facility rollout ([current page](https://www.seniordeli.com/en/snap-to-iddsi)). Remove or substantiate these claims before presenting this as a new receipt.

### 2. Feasibility and architecture

A single photo can often separate obvious L7 from smooth L4, and sometimes reveal L5/L6 particle size when a scale reference is present. It cannot reliably measure hardness, adhesiveness, cohesiveness, moisture separation or flow. L4 versus L5 and especially L5 versus L6 require spoon/fork tests; IDDSI itself says testing methods determine compliance ([official methods](https://www.iddsi.org/images/Publications-Resources/DetailedDefnTestMethods/English/V2TestingMethodsEnglish31july2019.pdf)). Human agreement is also weak for several tests—reported κ values are only 0.23–0.36 for spoon tilt/fork pressure characteristics ([reliability study](https://doaj.org/article/51c4ce683add4c35b051eece3eab6d14)).

Use:

- **SigLIP-2 visual encoder + ordinal/multitask heads** for level compatibility, particle size and visible failure cues; calibrated confidence and mandatory abstention.
- Qwen3-VL only for multilingual explanation grounded in structured outputs.
- A deterministic flow-video pipeline: fiducial card, syringe segmentation, meniscus tracking, verticality/occlusion checks and the reading exactly 10 seconds after release—not an end-to-end VLM.

Flow grading is realistic, but require the correct 61.5 mm syringe/funnel, exact 10 ml start, serving temperature and “boundary/unclear” output near 1, 4 or 8 ml. Target residual-volume MAE ≤0.25 ml.

Minimum credible release: **3,000–5,000 independent test events**, not burst frames—about 2,000 food preparations and 1,000–1,500 flow videos, balanced by level, recipe, batch, phone, lighting, kitchen and Chinese/Western cuisine. Split by recipe/batch/kitchen to prevent leakage.

### 3. Regulation

“Not a diagnosis” does not neutralize intended use.

- Hong Kong includes software intended for diagnosis, treatment, monitoring or physiological support within its medical-device definition; MDACS remains voluntary, but classification follows claims and intended purpose ([MDD](https://www.mdd.gov.hk/en/useful-information/frequently-asked-questions/index.html)).
- NMPA treats immature AI providing clinical decision recommendations as potentially Class III and measurement/reference software as Class II; food-phone imagery creates boundary uncertainty requiring a formal classification request ([NMPA principles](https://english.nmpa.gov.cn/2021-07/08/c_660267.htm)).
- FDA general-wellness exclusion applies only when unrelated to disease. Dysphagia, aspiration prevention and patient-specific corrective advice are directly disease-related; caregiver-facing black-box recommendations also fail important non-device CDS conditions ([FDA](https://www.fda.gov/medical-devices/digital-health-center-excellence/step-6-software-function-intended-provide-clinical-decision-support)).
- EU MDR Rule 11 makes therapeutic-decision software at least IIa and potentially IIb where error may cause serious deterioration ([EU guidance](https://health.ec.europa.eu/document/download/b45335c5-1679-4c71-a91c-fc7a4d37f12b_en?filename=mdcg_2019_11_en.pdf)).

Safer demo position: **food-service QA/education**, user supplies the clinician-prescribed target; app verifies test execution and says “compatible / not verified / repeat official test.” Do not output patient suitability, “safe,” aspiration risk, or autonomous treatment level. “食品不能代替药物” is irrelevant. Add “Not the official IDDSI website/no IDDSI endorsement,” as IDDSI requires ([publishing guidance](https://www.iddsi.org/images/Publications-Resources/Guidelines/guidelines_for_creating_iddsi_web_blogs_and_product_-materials_july2020.pdf)).

### 4. Validation and receipt value

Ground truth must be physical tests independently performed by an RD and SLP, with third-person adjudication—not two visual labels. Report human photo-only baseline, weighted κ/raw agreement, macro-F1, per-level confusion matrix, calibration, risk-coverage curve, dangerous under-classification rate and 95% CIs. Reserve a locked external kitchen/phone/cuisine test set.

HKU collaboration is plausible through Professor Karen Chan’s **Swallowing Research Laboratory** and HKU Human Communication, Learning and Development unit ([HKU](https://swallow.edu.hku.hk/about-us/)). Require a written validation protocol and permission before naming HKU.

As a KOL receipt this is **7/10 with independent HKU validation; 3/10 without it**. Core audiences are SLPs/RDs, RCHE kitchens, caregivers and texture-food manufacturers; robotics interest is secondary. Realistically expect hundreds to low-thousands of HF downloads/demo users, with broader HK/CN press only if HKU formally participates. The single largest value increase is a **blinded, preregistered HKU external validation**, including publication of failures.

### Four-week gates

1. **Week 1:** Freeze intended use, claims and protocol; collect 150 paired samples. Kill definitive photo classification if expert photo-only weighted κ <0.70.
2. **Week 2:** Build baselines on 1,200 food events/400 videos. Continue only if external-kitchen macro-F1 ≥0.75 with dangerous under-classification ≤5% at ≤30% abstention.
3. **Week 3:** Scale to ≥3,000 events; locked evaluation. Flow gate: ≥95% non-boundary level agreement and MAE ≤0.25 ml.
4. **Week 4:** Blinded ≥300-event external audit, model card, datasheet, hazard log, HF Space and arXiv preprint. Without written HKU approval or passing gates, release only as a **Y06 research preview/test assistant**, not a validated safety product.