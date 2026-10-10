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

## H-NOISE — label-noise ceiling from inter-judge disagreement (regen 2026-09-16)

- Hazard: label-noise ceiling. Two VLM judges, Qwen3-VL-2B tier-1 vs Qwen3-VL-8B tier-2, agree on **31.0%** of the same images (n_pairs = 575, full tier-2 sample). This is only marginally above the ~20% chance level for 5 classes. Cohen's kappa is **κ_unweighted = 0.1062** and **κ_quadratic-weighted = 0.2766** (Landis-Koch: slight / fair at best). Every label-derived metric is therefore measured against label noise and means nothing about real-world accuracy. Weighted kappa 0.42 (the model-eval figure, not inter-judge) is flat before and after the decode fix, measured against this noise.
- How found: inter-judge agreement measured on the same images across the two VLM tiers. The regen was performed on the full tier-2 sample (preliminary 22.6% in v0.0 was on a smaller subset). The full table is at `results/regen_20260916/inter_judge_20260916.md`; run provenance in `tier2_resume.log` / `agreement_chain.log` on the DGX box.
- Fix: published as the negative finding with its cause stated plainly. Web photos without a physical scale fiducial cannot be weakly labelled for IDDSI texture levels, even by strong VLMs, because middle levels (L4/L5/L6) require absolute size and flow information a photograph doesn't carry. The remedy path is physically-tested desk specimens: 60 base foods x 5 levels = 300 specimens, one operator, protocol-following, test-filmed (docs/CAPTURE_PROTOCOL.md), with a filmed re-test subsample for intra-rater reliability. Labels will be single-operator physical tests, never clinician-validated.
- Residual risk: all weak-labelled metrics stay non-claims for accuracy. No accuracy, safety, dietary, clinical, or patient claim may be built on them.

## H-UNDERPRED — tier-1 systematically under-predicts difficulty (new in v0.1)

- Hazard: directional bias in the label noise. The regen cross-tab on the full tier-2 sample (n=575) shows tier-1 (Qwen3-VL-2B) does not just disagree randomly with tier-2 — it **systematically under-predicts** difficulty, dragging interior classes (L4/L5/L6) toward the regular-food end (L7). Only L3 is reliably classified (11/12 correct). The bias direction is the same dangerous direction the wk2 gate already catches: predicting a SAFER texture than truth.
- How found: cross-tab inspection of the regen agreement data. L4 calls (151) miss to L7 in 69 and to L6 in 28 (only 42 stay at L4); L5 (131) miss to L7 in 54 and to L6 in 42 (only 27 stay); L6 (155) collapse into L7 in 111 (only 30 stay). L7 (126) over-pull to L6 in 47 and to L5 in 6 (68 stay).
- Fix: documented in the v0.1 release card under "Under-prediction finding". The directional bias strengthens the case for physically-tested desk specimens (H-NOISE remedy) and for the wk2 gate (H-L7SINK). No new code change ships; the disclosure IS the fix.
- Residual risk: the directional bias means any weak-labelled training data carries systematic under-classification, on top of the random noise. The wk1 kill rule FIRES regardless. Weights stay withheld.

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

## H-ABSTBLIND — the abstention-blind under-classification metric behind the wk2 gate

- Hazard: `trainer/metrics.py` computes `dangerous_underclassification_rate_all` as `((raw_predictions < labels) & accepted).mean()`. The numerator is gated on acceptance but the denominator is all N. An under-classification the model abstains on is therefore never counted, while the row still counts in the denominator, so a model that abstains more drives this number toward 0.0. The bias has one direction and it is the flattering one: the metric always rewards declining to answer on precisely the rows where the model would have erred in the dangerous direction (predicting a SAFER texture than truth, the same direction as H-L7SINK and H-UNDERPRED). `_all` is the gate-facing name — wk2 requires dangerous under-classification <= 5% at <=30% abstention — so the quantity most likely read as the gate verdict is the quantity most flattering to a model that abstains hardest. Verified by reading the code, not by a lane verdict.
- How found: read of the metric definitions in `trainer/metrics.py` against the v0.1 and v0.2 result tables. v0.1 (n=109 held out): coverage 0.881, dangerous under-classification (covered) 0.333, and calibration `under_fnr` 0.2857 against the 0.05 target, target_met FALSE. v0.2 (5 seeds, test n=102): coverage ran 0.500-0.618, so 38% to 50% of test rows abstained, while FNR(covered) ran 0.039-0.133 with only 1 of 5 seeds (seed 29, 0.0488) under the 0.05 target. That is the regime where the blind variant and the covered variant diverge furthest: half the test set drops out of the numerator and stays in the denominator. The 33% quoted in H-L7SINK and on the release card is the covered figure, so the published numbers are unaffected by the defect.
- Fix: read `fnr_covered`, not `dangerous_underclassification_rate_all`, for every wk2 statement. `fnr_covered` restricts numerator and denominator to the covered subset, so an abstention cannot improve the rate; the cost of abstention is then carried where it belongs, by the coverage figure and by the gate's own <=30% abstention clause. Any run that wants the all-rows view must report the numerator, the denominator, and coverage together, never the rate alone. No new code change ships; the disclosure plus this naming rule ARE the fix.
- Residual risk: the `_all` name reads as the conservative all-rows number and is the opposite — it is the only under-classification quantity that improves when the model declines to answer. No gate outcome changes: the wk1 kill rule FIRES on every v0.2 seed (weighted kappa 0.455-0.539 against the >= 0.70 gate) and the weights stay withheld under the 2026-09-07 decision. But any future re-run that reads the `_all` variant can post a passing wk2 for a model that is failing it, which is the failure mode the gates exist to prevent. Photo-classification stays killed as a product capability, and every figure above remains measured against the inter-judge noise in H-NOISE (31.0% agreement, n_pairs 575, κ_unweighted 0.1062 / κ_quadratic 0.2766), so it means nothing about real-world accuracy.

## H-MACROF1 — Macro-F1 excludes the abstention class its own name promises

- Hazard: `macro_f1_with_abstentions` builds its label set from the TRUE labels only — `labels=sorted(set(int(v) for v in labels))` — so the -1 abstention class is never in it. An abstention is therefore scored purely as a missed true class; it is never a predicted class of its own. The same module's confusion matrix at `metrics.py:136` DOES treat -1 as a class, so one run emits two different pictures of the same predictions. The published macro-F1 (0.358 after the decode fix, `model_README.md:69`; quoted again at `HAZARD_LOG.md:20`) is optimistic relative to what its name says it measures.
- How found: by reading the metric code, not by a lane verdict. The macro-F1 label set is derived from the label vector alone, while the confusion matrix enumerates the prediction-side classes independently and includes -1. The gap bites where abstention is large: coverage is 0.881 on v0.1 and 0.500–0.618 across the five v0.2 seeds, so on v0.2 most predictions are abstentions that the published macro-F1 cannot score as a class.
- Fix: add -1 as a sixth class to the macro-F1 label set, so abstention carries its own precision, recall and F1. Today an abstention is charged against its true class's recall while its prediction-side false positive is counted nowhere, and -1 has no true positives because it occurs only in predictions — so the corrected macro-F1 cannot exceed the published figure and will move down on the same runs. Re-run and republish the before/after macro-F1 beside the coverage and ECE already carried at `HAZARD_LOG.md:20` and `model_README.md:68-69`. No decode path, threshold, gate, or withheld-item decision changes.
- Residual risk: the reported macro-F1 was never the metric its name describes, and any external comparison against a macro-F1 that scores abstention as a class is not like-for-like until both sides are recomputed on the same label set. The wk1 and wk2 gates are computed elsewhere and are not moved by this fix; the wk1 kill rule still fires and the weights stay withheld.

## H-KAPPACOV — coverage-filtered kappa reported against an unconditional gate

- Hazard: the wk1 gate is a single unconditional number — weighted kappa >=0.70, else kill photo-classification (`model_README.md:75`, `HAZARD_LOG.md:48`) — but the weighted kappa that is reported against it is conditional, and is not stated to be conditional at the point of reporting. `weighted_kappa_covered` calls `cohen_kappa_score(covered_true, covered_pred)` in `trainer/metrics.py`: both vectors are the covered subset only, and every abstention is dropped before the kappa is taken. The rows that leave the measurement are exactly the rows the calibrated head marked low-confidence. Every weighted kappa published in this release is therefore a selective kappa on a coverage-filtered subset: v0.1 kappa 0.4227 at coverage 0.881 (`HAZARD_LOG.md:20`, `model_README.md:71`), and the v0.2 five-seed range kappa 0.455–0.539 at coverage 0.500–0.618. The gate and the metric describe two different populations. Neither the gate text nor the results table states that the comparison is conditional on coverage.
- How found: by reading the metric code, not by a lane verdict. That the removed rows are the model's actual disagreements is NOT established and is not claimed here: abstention is gated on calibrated confidence, and no measured relationship between abstained rows and their true labels exists in this release. Confidence is a proxy for correctness, not a measure of it, so the filter is expected to remove a subset enriched for errors, but the size and direction of that enrichment are unmeasured. The same acceptance-gating family is visible in `dangerous_underclassification_rate_all` and `macro_f1_with_abstentions`.
- Fix: no code change ships; the disclosure IS the fix. Every published kappa carries its population with it — covered subset, n covered, and the coverage of that run — in the hazard log, the release card, and any results table. The wk1 line states plainly that the threshold is unconditional while the measured value is coverage-filtered, so a covered kappa is never read as a kappa. The sibling acceptance-gated metrics are labelled the same way in the same pass. This does not reopen the release shape: the release shape was decided 2026-09-07 and is locked, and the weights stay withheld.
- Residual risk: the defect does not rescue the head and does not change the kill rule. No v0.1 or v0.2 figure reaches 0.70 under the flattering reading — the covered subset — and dropping the low-confidence rows removes the hardest cases, so an unconditional kappa over all rows would be no higher, not higher. wk1 FIRES on any reading. The residual risk is comparability: a coverage-filtered kappa can be raised by abstaining more, so a future run could report a higher kappa while getting worse. That is the wrong direction for a gate meant to catch a weak head. Metric population stays labelled in every results table, and a coverage-conditional threshold is not introduced retrospectively against these runs.

## H-GROUP — the leak-control group field groups nothing

- Hazard: the training config sets `group_field: group_id` so the train/test split is leakage-safe, but in the existing manifest all 544 events carry 544 DISTINCT `group_id` values — every group has exactly one event. A one-event group separates nothing, so the configured leakage control is vacuous: any event sharing a subject, a source page, or a near-duplicate image with a held-out event can land on the opposite side of the split and the splitter would not know. Two further manifest defects compound it. 20 `event_id` values are duplicated across corpora, so `event_id` is not a unique key and cannot anchor a group. 129 events carry a NULL `level_tier1` — the rows sitting outside the n_pairs=575 measured in H-NOISE (`HAZARD_LOG.md:25`), so the two-tier provenance is not uniform across the table either.
- How found: by reading the config and the manifest, not by a lane verdict. The intent is declared in the config under `trainer/` (`model_README.md:116`); the manifest under `dataschema/` (`model_README.md:119`) does not implement it. Nothing errors on this — a splitter handed a unique-per-row key accepts it, so the control looks configured and runs. The counts also do not reconcile across artifacts: 544 grouped events, 704 labelled rows, 587 unique events in the licence accounting. The grouped population is not the labelled population, so neither number certifies the other.
- Fix: `group_id` must be a property of the capture, assigned once at capture time and carried through to the manifest, coarser than the event: dish or base-food identity (several photographs and several levels of the same food), capture session or physical specimen, and source page or licence parent, so an event and its visual relatives share a group. The 20 duplicated `event_id` values must be resolved to a surrogate primary key first — the current key cannot carry a grouping. The 129 NULL-`level_tier1` rows need an explicit rule (drop from supervised training, or train with tier-2 provenance marked as such) rather than passing as two-tier labelled. Two assertions gate every split from then on: group cardinality greater than one for the intended unit, and group-disjointness between training and held-out. Re-splitting re-randomises the eval, so no figure published in this release carries over — a corrected split is a new eval run, not a re-read of the current ones.
- Residual risk: until the split is group-disjoint on a real grouping, every metric in this log — including the ones that fired the gates — is measured on an eval set with no provenance guarantee against leakage from training, and the v0.2 seed spread (macro-F1 0.284–0.320, kappa 0.455–0.539) is seed variance on a single split, so it cannot separate split effect from seed effect. Read the gate failures as optimistic, not clean. The negative finding and the wk1 kill stand regardless: they run against the model, so a leakage correction cannot reverse them. Weights stay withheld (`HAZARD_LOG.md:21`). No severity, probability, or acceptability finding is made here (`HAZARD_LOG.md:5`).

## H-LEAK — the v0.2 evaluation split was contaminated; measured, not assumed

- Hazard: v0.2's held-out split was row-wise stratified on level, so it carried
  no leakage guarantee. Perceptual deduplication over all 684 images (dHash ≤ 6
  **and** pHash ≤ 8) found **55 near-duplicate pairs that `event_id` never
  caught**, forming **7 genuine visual families**, and **4 of those families
  spanned train, val and test simultaneously**. v0.2's reported macro-F1, kappa
  and coverage were therefore measured on an eval set containing images
  effectively present in training. Source: `results/v03_groupdisjoint/dedup_report.json`.
- How found: by running the dedup, not by a lane verdict. The 20 duplicate
  `event_id` values were already known and already removed; these 55 are a
  different failure — the same photograph re-encoded or re-cropped under a
  different identifier, which byte-hashing cannot see and `event_id` cannot
  express. The grouped union also shows the original `group_id` was not merely
  weak but vacuous: 650 groups over 684 rows, versus 544 groups over 544 rows
  (every group a single event) in v0.2.
- Fix: shipped in v0.3. Groups are the connected components of the
  near-duplicate graph unioned with the source corpus; whole groups are assigned
  to splits; the splitter **asserts** group-disjointness and per-level coverage
  and fails loudly rather than assuming them. Both assertions pass.
- Residual risk: dHash ≤ 6 with a pHash ≤ 8 confirmation is a conservative
  threshold, so visually-similar-but-not-identical images (the same dish shot
  twice) may still sit in different groups. v0.3's numbers are therefore
  *better* than v0.2's but not certified clean. The wk1 gate still fires on all
  five v0.3 seeds and the weights stay withheld, so this changes the honesty of
  the numbers, not the release decision.

## H-AUGRC — the abstention-blind metric is now measured the right way, not just disclosed

- Hazard: H-ABSTBLIND established that `dangerous_underclassification_rate_all`
  gates its numerator on acceptance while dividing by all N, so abstention
  deletes errors from the count. v0.2 disclosed it; v0.3 **fixes the
  measurement**.
- Fix: v0.3 publishes the correctly-named **all-N under-classification rate**
  (both sides of the fraction over the full test set): **0.176 – 0.282** across
  five seeds, against the covered-only figure of 0.061 – 0.147 that would be
  quoted if the defect went unfixed. It also adds **AUGRC** — joint selective
  risk integrated over thresholds (Traub, Bungert, Lüth, Baumgartner,
  Maier-Hein & Jäger, NeurIPS 2024, https://arxiv.org/abs/2407.01032) — which
  cannot be driven down by abstaining because coverage sits inside the
  integrand. Measured: **0.242 – 0.295**, every seed well below the 0.5 random
  reference. Source: `results/v03_groupdisjoint/selective_metrics.json`.
- Residual risk: AUGRC is computed post hoc from stored probabilities, not by
  `trainer/eval/metrics.py`, so the library metric and the published metric are
  still two code paths and can drift. The honest statement is that v0.3's
  selective-risk numbers come from a documented second implementation.

## H-GATE — the wk1 gate is contested by the ordinal-classification literature, and is deliberately not changed

- Hazard: the wk1 gate is weighted Cohen's kappa ≥ 0.70 on a five-level ordinal
  problem. Delgado & Tibau (2019), "Why Cohen's Kappa should be avoided as a
  performance measure in ordinal classification", *Pattern Recognition*
  105:107192, https://www.sciencedirect.com/science/article/abs/pii/S0167865518313413,
  argues the metric ranks ordinal models wrongly under class imbalance — which
  is exactly our condition (L7 is 54% of the label set).
- Decision: **the gate is NOT changed.** Re-registering a pre-registered gate
  after seeing results would make every number in this release indefensible, so
  the objection is disclosed in the release card and the gate stands as written.
- Residual risk: a reviewer may reasonably read our gate failures as a metric
  artifact rather than a model failure. We consider that acceptable: the gates
  fail, we keep weights withheld, and we state plainly that the gate itself is
  disputed. Changing it would be worse.

## H-LABELCEIL — SUPERSEDED by H-QWENBIAS; kept because the reversal is the lesson

- Original finding (2026-10-11, n=30 pilot): a third labeller from an
  independent family agreed with the Qwen-8B tier-2 labeller on **40.0%** — no
  better than the two Qwen sizes agreed with each other (**42.9%**) on the same
  sample. Read then: "a third labeller does not help; the ceiling is physical."
- **That conclusion was wrong, and it was wrong because the sample was too
  small.** Re-run over **650 events**, Step-5 vs Qwen-8B is **47.1%** against
  Qwen-2B vs Qwen-8B at **30.9%**. A 16-point gap that a 30-image pilot rendered
  invisible. The two unrelated families also agree with each other at 74.7%.
- Superseded by **H-QWENBIAS**, which carries the full evidence and the fix
  (4-way consensus training, worth about +0.12 kappa on identical ground truth).
- Why this entry stays: **a null result from an underpowered sample is not a
  result.** We nearly published "more labellers cannot help" and were about to
  stop spending on relabelling on the strength of 30 images. Pilot-scale
  agreement measurements on this task need an order of magnitude more data
  before they can be quoted, in either direction. The claims that survive the
  small sample are the ones about *coarse* judgement (is there food: 96.7%
  cross-family) rather than fine ordinal judgement.
- Residual risk: none from this entry — it makes no claim of its own. The live
  risk is the opposite one: that the 74.7% cross-family figure is read as
  evidence of correctness. It is evidence of consistency only. See H-QWENBIAS.

## H-UNUSABLE — ~30% of the corpus was never labelable, and every earlier number was measured with it

- Hazard: a labelability sweep judged all 684 events with two independent model
  families. They agree **96.7% on "is there food"** and **85.5% on "is it
  levellable"**, but **152 events (29.8%)** fail at least one test -- empty
  vessels, diagrams, no discernible food mass. Those 152 sat in the held-out
  split of v0.1 through v0.4, so every published metric was computed partly
  against images that cannot carry an IDDSI level at all. Source:
  `results/v05_release/unusable_event_ids.txt` and `labelability_sweep.json`.
- How found: by running the sweep with 32 step-5-preview lanes and 5 MiniMax
  vision workers over the whole corpus on the DGX, not by a lane verdict. The
  asymmetry is the finding: the *binary* judgement is near-perfectly reliable
  across independent families, while the *5-level* judgement is not (74.7% best
  cross-family, 30.9% within-Qwen). We can verify what is in the frame; we cannot
  reliably verify what level it is.
- Fix: v0.5 reports every metric on the **cleaned** 108-event test set. The
  152 unusable events are listed and excluded.
- Residual risk: removing them also removes training signal, because the
  removals fall on the hard middle levels (L4 96->66, L5 75->65, L7 400->305).
  A model trained on the cleaned 532 scores *worse* than the released model on
  the identical test set (exact 0.426-0.500 vs 0.444-0.546, kappa 0.498-0.547 vs
  0.540-0.566). We publish both runs. The next data effort is therefore **more
  mid-level labels, not more deletions**.

## H-QWENBIAS — two independent labellers agree with each other far better than with Qwen

- Hazard: every label before v0.4 came from Qwen3-VL-2B and Qwen3-VL-8B, two
  sizes of ONE family. Re-labelling blind with two unrelated families gives:
  Qwen-2B vs Qwen-8B **30.9%** exact (68.8% within +/-1), Step-5 vs Qwen-8B
  **47.1%** (73.2% within +/-1), and Step-5 vs MiniMax-M3.1f **74.7%**. The two
  unrelated families agree with each other at more than twice the rate either
  agrees with Qwen. Qwen-8B is also *less* consistent with its own smaller
  sibling than a foreign family is with it.
- Fix: v0.5 trains on 4-way consensus (Qwen-2B, Qwen-8B, Step-5, MiniMax-M3.1f).
  Measured on identical ground truth this is worth about **+0.12 kappa** and
  **-0.19 levels of MAE** over training on the Qwen tier-2 labels.
- Residual risk, and it is the important one: this shows the Qwen labels are
  **inconsistent with independent judgements**, not that they are **wrong**.
  Two correlated wrong families can agree with each other. Distinguishing
  "inconsistent" from "incorrect" requires a physically-measured anchor, which
  does not exist. Consensus is a lower-variance label set, not a correct one.

## Retained boundary hazards

- H-PHOTO: a still photograph treated as a physical measurement. Control: release scope states plainly that middle levels (L4/L5/L6) require absolute size and flow information a photograph doesn't carry. Physical-test confirmation stays external to the release.
- H-VERDICT: a level estimate reused as a verdict on what a person should eat. Control: the release never issues a verdict, outputs no safe or unsafe determination, and makes no dietary, clinical, or patient claims. Withholding the weights is the mitigation against verdict reuse.
- H-PROV: weak labels, cloud-generated rows, or single-operator physical tests promoted to clinical ground truth. Control: provenance stays separated across `harvest/`, `synth/`, `dataschema/`, and `trainer/`. The data engine is stated exactly: 713 harvested real CC-licence-candidate images in 3 sets (realweak_pilot 117+12, harvest_d1 400, harvest_d1b 184), 701 raw plus 6 validated cloud-generated. The flow-test grader is stated exactly: deterministic CV pipeline for the 10 ml / 10 s IDDSI syringe flow test from smartphone video, MAE 0.065 ml on a 100-video bench, grades L0-L4, shipped under `flowtest/`.

## Change control

v0.1 (2026-09-16 regen): the inter-judge agreement figure in H-NOISE is updated from the preliminary 22.6% to the regen 31.0% on n=575, with kappa (unweighted 0.1062, quadratic 0.2766) newly disclosed and the under-prediction finding added as a new section (H-UNDERPRED). Nothing else changes — intended use, prohibited uses, output wording, label handling, training sources, threshold behaviour, checkpoint status, eval split, decode path, or public claim. Wording changes cannot expand the release from research plus culinary-education use into person-specific, clinical, diagnostic, or safety use.

v0.2 (2026-10-10): four metric-and-split hazards added after a 30-lane audit of the eval code and a 5-seed retrain on a corrected manifest (684 unique events, stratified 70/15/15). H-ABSTBLIND, H-MACROF1 and H-KAPPACOV are defects in `trainer/eval/metrics.py` verified by reading the code; H-GROUP is a defect in the split configuration. No v0.1 or v0.2 figure changes sign under the corrected definitions. The wk1 kill rule still fires on every v0.2 seed (weighted kappa 0.455-0.539 against the >= 0.70 gate) and the weights stay withheld under the 2026-09-07 decision.

v0.3 (2026-10-11): four hazards added. H-LEAK records that the v0.2 split was measured to be contaminated (4 visual families spanned train/val/test) and is now fixed and asserted; H-AUGRC records that the H-ABSTBLIND defect is now measured correctly (all-N rate 0.176-0.282, AUGRC 0.242-0.295) rather than only disclosed; H-GATE records that the wk1 gate is disputed by the ordinal literature and is deliberately left unchanged; H-LABELCEIL records that a third independent labeller does not improve agreement (40.0% vs 42.9%), so the ceiling is physical, not algorithmic. The wk1 gate still fires on all five v0.3 seeds and the weights stay withheld.
