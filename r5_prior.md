## Verdict

Proceed—but as an **open research/quality-assurance benchmark**, not as a reliable safety classifier. Static RGB cannot directly measure flow, hardness, adhesiveness or cohesiveness; mandatory abstention and physical-test confirmation are essential.

## Verifiable landscape

- **Direct automated predecessors:** ViscoCam used smartphone video plus motion sensing to classify three IDDSI liquid levels—96.5% controlled accuracy, but only >81% under extreme conditions ([PolyU](https://research.polyu.edu.hk/en/publications/viscocam-smartphone-based-drink-viscosity-control-assistant-for-d/)). OptiTexture combines RGB with a <$20 light-scattering sensor: 91.96% on 112 L3–L6 samples; its vision-only baseline was 69.64% ([ACM](https://dl.acm.org/doi/10.1145/3810209)). EdUHK’s single-image Qwen pipeline achieved only 61.74% exact match on CEIV-115; open VLMs missed every liquid hazard ([Frontiers](https://www.frontiersin.org/journals/nutrition/articles/10.3389/fnut.2026.1829703/full)). Automated video/photogrammetric measurement of the official syringe test already appeared in 2019 ([PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC6838027/)). Australia’s proprietary **Safe Plate** now advertises photo-and-video IDDSI verification ([Safe Plate](https://www.safeplate.au/)).
- **Self-preemption:** SeniorDeli already publicly markets **Snap-to-IDDSI/CareEZ**, including “validated by HKU Swallowing Research Laboratory” ([current page](https://www.seniordeli.com/en/snap-to-iddsi)). Remove or substantiate that validation claim before launch.
- **Not automated classifiers:** the official IDDSI iOS/Android app is a descriptor/test-reference library ([App Store](https://apps.apple.com/us/app/iddsi/id1145593063)); Viscgo hardware, CF-200N texture analysers, MealSuite/Dietech menu software, and manufacturer catalogs classify/test known products rather than infer an unknown plate. Major examples are Japan’s Kewpie/UDF ([Kewpie](https://www.kewpie.com/en/sustainability/dietary-lifestyle/universal/)), Korea’s Pulmuone DesignMeal and Hyundai Green Food Greating, Europe’s apetito/winVitalis ([apetito](https://www.apetito.co.uk/our-food/texture-modified)), and US Hormel/Lyons and SimplyThick. No independent mainland-Chinese plate-photo classifier was found.

## Openness and exact claim

No downloadable **IDDSI-specific** trained weights plus fully licensed image/video dataset were found. CEIV publishes examples, not the full machine-readable corpus; OptiTexture and ViscoCam expose neither dataset nor weights. Open FoodSense/PlateInsight resources concern generic perceived texture, not IDDSI ([FoodSense](https://huggingface.co/datasets/sababishraq/foodsense-dataset)).

Use:

> “To our knowledge, as of 31 August 2026, the first publicly downloadable IDDSI-specific smartphone-food dataset and fine-tuned model released under explicit reuse licences, with physically tested Chinese/Cantonese soft-meal labels.”

Separately, claim **“first open-source smartphone grader for the 10 ml/10-second IDDSI Flow Test,”** not “first video grader.”

Never claim: first IDDSI AI; first image classifier; first dataset/benchmark; clinically validated; diagnostic; medical-grade; or able to determine swallowing safety from a photograph.

A “not diagnosis” disclaimer does not override intended purpose: HK considers software intended for disease prevention/monitoring a potential medical device ([Medical Device Division](https://www.mdd.gov.hk/en/mdacs/online-tools/is-your-product-a-medical-device/index.html)). Also say “tests performed by LinguaLeap,” never IDDSI-certified/endorsed, and follow [IDDSI’s labelling rules](https://www.iddsi.org/images/Publications-Resources/ProductLabels/iddsi-product-labelling-guidelines.pdf). The KOL receipt should lead with the repository, test videos, held-out results and failure cases—not the word “first.”