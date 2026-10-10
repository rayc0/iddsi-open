# Specimen-card audit — cards needing a fix before capture

Generated 2026-10-10 from a 32-lane audit of all 267 cards on the DGX.
Authoritative test-per-level mapping: docs/CAPTURE_PROTOCOL.md section 5.
(An earlier audit pass reported ~66% wrong test sets; that was a defect in the
AUDIT SPEC, not the cards, and has been corrected and re-run.)

| card | achievable | test set | params | single worst risk |
|---|---|---|---|---|
| bf05_cooked_instant_oats_L5 | no | no | no | the operator delivers a wet, flowing oat mash as L5, and the card's only remedy — "add 0.5 g thickener per 100 ml" — binds the free water into a gel a |
| bf09_steamed_taro_L5 | no | no | no | The operator presses the whole batch through the 4 mm sieve and photographs a smooth ≤4 mm purée as L5, so every L5 image in the set is really L4. |
| bf13_steamed_chicken_thigh_L6 | no | no | no | An operator handed a card whose PREP, IDDSI TEST, VISUAL QC and FAIL MODES are |
| bf14_steamed_egg_custard_L3 | no | yes | no | Every section after the header is "(MISSING)", so the operator improvises an egg custard from memory and labels whatever comes out of the steamer as L |
| bf28_steamed_bitter_melon_L6 | no | yes | no |  |
| bf29_steamed_eggplant_L6 | no | no | no |  |
| bf30_sweetcorn_kernels_L7 | no | no | no |  |
| bf31_banana_L4 | no | yes | no |  |
| bf31_banana_L5 | no | no | no |  |
| bf31_banana_L6 | no | no | no |  |
| bf33_papaya_L6 | no | no | no |  |
| bf35_soft_ripe_persimmon_L6 | no | no | no | The operator sieves the persimmon through the 4 mm sieve (guaranteeing ≤4 mm = L5) and then signs off the L6 label from test lines that are pre-writte |
| bf36_pear_L7 | no | no | no |  |
| bf39_orange_segments_peeled_L6 | no | no | no | the operator presses a good 15 mm orange piece, sees juice squeeze out, discards it and re-cuts smaller — repeated until the pieces drop below 8 mm an |
| bf41_plain_yogurt_L3 | no | no | no | The operator weighs the literal 30 g of thickener into 175 g of yogurt and water, producing a stiff gel that leaves the full 10 ml in the syringe, the |
| bf43_baked_egg_custard_L5 | no | no | yes | The operator sieves the fork-mashed curds, sees the "glossy puree, no visible curds" the card calls a failure, and discards good L5 mash — or, if they |
| bf44_mango_pudding_L5 | no | no | no |  |
| bf52_sweet_and_sour_pork_L6 | no | no | no |  |
| bf52_sweet_and_sour_pork_L7 | no | no | no | with no prep, no test and no QC, an operator improvises a sauce-thickened, shredded or bite-sized sweet-and-sour pork and whatever lands on the plate  |
| bf53_mapo_tofu_L6 | no | yes | no | Following the prep as written yields a loose ≤4 mm saucy mass — L5 at best — which the operator then passes against the card's own fork-pressure wordi |
| bf54_saltedegg_steamed_pork_p_L6 | no | yes | no |  |
