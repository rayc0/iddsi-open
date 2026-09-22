### Verdict

Feasible as a guided IDDSI test assistant—not defensible as “IDDSI from one photo.”

**Accuracy.** EdUHK’s result is verified: 115 HK-oriented images, Qwen-VL-Max at 61.74% exact match, while local open VLMs had 100% liquid-hazard FNR. [OptiTexture](https://researchportal.hkust.edu.hk/en/publications/optitexture-a-low-cost-optical-deep-learning-solution-for-objecti/) reached 91.96% on only 112 L3–L6 samples, +22.32 points over RGB-only—i.e. 69.64%. With 3–5K genuinely independent preparations, expect 78–85% macro exact-match on a recipe/session-grouped in-house test, but only 65–75% on an external kitchen/phone/food test. Claiming >85% external single-photo accuracy is implausible; add calibrated “unclear” abstention. Guided testing can plausibly exceed 90%. [EdUHK paper](https://www.frontiersin.org/journals/nutrition/articles/10.3389/fnut.2026.1829703/full)

**Why photos fail.** L3/L4 differ by flow; L4/L5 by granulation, cohesion and stickiness; L5/L6 by particle size plus softness; L6/L7 by size restrictions, chewing demand and deformation. Surface appearance cannot reveal these mechanical properties. Level 7 Regular versus Easy-to-Chew must also be distinguished.

**Capture protocol.** Take:

1. A 45° plate photo with a standard fork/15-mm reference.
2. An 8–12-second guided clip: scoop/tilt for L3–5, fork-drip for L3/4, and press a representative piece until thumbnail blanching for L5–7, showing whether it squashes and rebounds.

Extract before/after frames plus motion; reject missing scale, occlusion, mixed foods or incomplete tests. This matches the [official IDDSI test definitions](https://www.iddsi.org/images/Publications-Resources/DetailedDefnTestMethods/English/V2TestingMethodsEnglish31july2019.pdf).

**Flow grader.** Yes—this is the strongest MVP. Detect the calibrated syringe, release at t=0, and meniscus at t=10; map residue to `<1`, `1–4`, `4–8`, `≥8 mL`. L4 still requires fork-drip/spoon-tilt. Require the official 61.5-mm barrel, fixed side view, repeat test, and abstain near boundaries or with bubbles/lumps. [ViscoCam](https://research.polyu.edu.hk/en/publications/viscocam-smartphone-based-drink-viscosity-control-assistant-for-d/) proved temporal phone sensing viable: 96.52% controlled and >81% under extreme conditions, although it used sloshing and a 3D CNN—not syringe residue.

**Architecture/training.** Use [SigLIP-2 ViT-B](https://arxiv.org/abs/2502.14786) plus an ordinal five-class head as the primary still-image model; retain a small CNN baseline. Use [Qwen3-VL-2B](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct) QLoRA only for multilingual explanations and guided multi-frame reasoning—not as the safety classifier. Freeze encoder → train head → unfreeze final blocks; use ordinal/class-balanced loss, conservative thresholds, five seeds. Minimum: ≥500 preparations and ≥30 food types per level, covering boundary recipes, thickeners, temperatures, phones, lighting, plates and operators.

**Evaluation/regulation.** Lock an external-kitchen set with physical-test ground truth and two independently rating dietitians. Report per-level confusion, macro-F1, weighted κ, dangerous-direction FNR, calibration/coverage and 95% CIs; compare against human inter-rater agreement. “First IDDSI texture AI” is false; “first open HK/Cantonese guided-test dataset” may be defensible after documented searching. Disclaimers do not erase intended purpose: HK treats medical-purpose mobile software as potential SaMD, so keep the Space research/QC-only and consult MDD before patient pilots. [HK TR-007](https://www.mdd.gov.hk/filemanager/common/mdacs/TR007E.pdf)