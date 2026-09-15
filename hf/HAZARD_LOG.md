# Hazard log — IDDSI-Open research preview

Status: HONEST release card companion for org `lingualeap`. Release shape decided 2026-09-07, locked: CODE plus EVAL RESULTS plus the flow-test grader plus a measured NEGATIVE FINDING ship. Model WEIGHTS and harvested image DATASET are WITHHELD for the safety and licence reasons stated in the release card.

This log is not a medical-device risk-management file. No severity, probability, or acceptability finding is made here. The release never issues a verdict and is engineered to stay out of SaMD scope by never issuing a verdict.

## Control principles

- The photo model estimates visual similarity only. It does not perform a physical IDDSI test.
- The release makes no safe or unsafe determination and no swallowing-safety inference.
- `unclear — perform physical test` is a required abstention output, never a verdict.
- Training labels used here are VLM weak labels. They are not physical IDDSI tests and are not clinician-validated.
- Synthetic, weak-labelled, and physically-tested records retain distinct provenance. Weak labels are never eval ground truth.
- Photo-classification is killed as a product capability under the wk1 kill rule. Only code and results ship.

## H-CORAL — CORAL ordinal-decode bug, interior classes unreachable

- Hazard: a CORAL ordinal-decode bug made levels L4/L5/L6 unreachable. Argmax over sigmoid-difference class masses could never fire for interior classes. The failure was silent and looked like a data problem.
- How found: proven numerically from the checkpoint. The decode path was examined against the stored checkpoint and the interior classes could not fire under the old argmax rule.
- Fix: replaced with the ordinal rank rule `level = #{k : P(y>k) > 0.5}`. After fix, n=109 held out: macro-F1 0.358, coverage 0.88, ECE 0.091, temperature 0.93, all levels predicted. Before fix: macro-F1 0.088, coverage 0.70, ECE 0.138, temperature 2.83. Weighted kappa 0.42 flat in both, measured against noise.
- Residual risk: label-derived metrics remain measured against label noise and mean nothing about real-world accuracy. Photo-classification stays killed under the wk1 kill rule. Weights stay withheld.

## H-NOISE — label-noise ceiling from inter-judge disagreement

- Hazard: label-noise ceiling. Two VLM judges, Qwen3-VL-2B tier-1 vs Qwen3-VL-8B tier-2, agree on only 22.6% of the same images, which is chance level for 5 classes. Every label-derived metric is therefore measured against label noise and means nothing about real-world accuracy. Weighted kappa 0.42 is flat before and after the decode fix, measured against noise.
- How found: inter-judge agreement measured on the same images across the two VLM tiers. The 22.6% agreement IS the headline negative finding.
- Fix: published as the negative finding with its cause stated plainly. Web photos without a physical scale fiducial cannot be weakly labelled for IDDSI texture levels, even by strong VLMs, because middle levels (L4/L5/L6) require absolute size and flow information a photograph doesn't carry. The remedy path is physically-tested desk specimens: 60 base foods x 5 levels = 300 specimens, one operator, protocol-following, test-filmed (docs/CAPTURE_PROTOCOL.md), with a filmed re-test subsample for intra-rater reliability. Labels will be single-operator physical tests, never clinician-validated.
- Residual risk: all weak-labelled metrics stay non-claims for accuracy. No accuracy, safety, dietary, clinical, or patient claim may be built on them.

## H-LICENCE — licence fail-close incident on harvested rows

- Hazard: licence fail-close incident. The release builder shipped 397 unverified open-images rows until the verification check failed closed on the missing field.
- How found: the verification check failed closed on the missing field, which exposed the unverified rows in the staged release.
- Fix: fail-closed verification kept as the gate, with a regression test added under `ops/`. The harvested dataset is WITHHELD and stays unpublishable on licence grounds: of 587 unique events exactly 2 would ship cleanly. 592 rows carry unverified open-images/openverse licences, 7 are CC BY-SA vs the BY-NC-SA main release, 109 have unresolved redistribution status.
- Residual risk: the harvested image dataset does not ship in any form. Any future dataset release must re-pass the fail-closed verification from a clean manifest.

## H-L7SINK — L7 catch-all sink in the dangerous direction

- Hazard: L7 catch-all sink. Unmeasurable rows defaulted to regular food, which is the dangerous direction: predicting a SAFER texture than truth. Dangerous under-classification of this kind is observed at 33%.
- How found: the default for unmeasurable rows pointed at the regular-food end, so rows with no measurable texture evidence landed on the least restrictive level.
- Fix: fixed with scale anchors, so unmeasurable rows no longer sink to the regular-food default. The pre-registered gates confirm the boundary: wk2 requires dangerous under-classification <=5% at <=30% abstention, and the observed 33% fails it. The wk1 kill rule FIRES on weighted kappa >=0.70, and the model fails all gates.
- Residual risk: photo-classification is killed as a product capability. Weights are withheld because a downloadable classifier would be used for verdicts whatever the card says, and a disclaimer does not cure a claim.

## Retained boundary hazards

- H-PHOTO: a still photograph treated as a physical measurement. Control: release scope states plainly that middle levels (L4/L5/L6) require absolute size and flow information a photograph doesn't carry. Physical-test confirmation stays external to the release.
- H-VERDICT: a level estimate reused as a verdict on what a person should eat. Control: the release never issues a verdict, outputs no safe or unsafe determination, and makes no dietary, clinical, or patient claims. Withholding the weights is the mitigation against verdict reuse.
- H-PROV: weak labels, cloud-generated rows, or single-operator physical tests promoted to clinical ground truth. Control: provenance stays separated across `harvest/`, `synth/`, `dataschema/`, and `trainer/`. The data engine is stated exactly: 713 harvested real CC-licence-candidate images in 3 sets (realweak_pilot 117+12, harvest_d1 400, harvest_d1b 184), 701 raw plus 6 validated cloud-generated. The flow-test grader is stated exactly: deterministic CV pipeline for the 10 ml / 10 s IDDSI syringe flow test from smartphone video, MAE 0.065 ml on a 100-video bench, grades L0-L4, shipped under `flowtest/`.

## Change control

No change to intended use, prohibited uses, output wording, label handling, training sources, threshold behaviour, checkpoint status, eval split, decode path, or public claim ships without review of the release card and this log. Wording changes cannot expand the release from research plus culinary-education use into person-specific, clinical, diagnostic, or safety use.
