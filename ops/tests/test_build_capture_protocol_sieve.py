"""Regression tests for the sieve USED / FORBIDDEN distinction in
ops/build_capture_protocol.py (defect cc02_cards).

The old parser recorded `prep_params["sieve_mm"]` from the first `N mm sieve`
match in the PREP prose, so a card that FORBADE sieving was recorded with
sieve_mm 4.0. Every fixture below is a verbatim sentence from
docs/specimen_cards/, so the test fails if the parser stops reading the corpus
the way the corpus is written.

Run from repo root:
    python -m pytest ops/tests/test_build_capture_protocol_sieve.py -q
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import build_capture_protocol as bcp  # noqa: E402

# --- a sieve the card USES ---------------------------------------------------

USED = [
    # plain straining step
    ("bf32_ripe_mango_L3",
     "Pass through the 4 mm sieve; the per-level note \"blend-thin\" applies: "
     "sieve first, then thicken; do not skip the sieve."),
    # sieve as a GAUGE at L5, where the card also bars forcing the batch through
    ("bf01_steamed_white_rice_L5",
     "SIEVE AS GAUGE ONLY: scoop one 15 g sub-sample and press it through the "
     "4 mm sieve with the back of a spoon. Do NOT press the whole batch through "
     "the sieve and do NOT discard retained material."),
    # "use the 4 mm sieve as a GAUGE only" -- the aperture is named only inside
    # the prohibition that precedes it
    ("bf26_steamed_carrot_L5",
     "Do NOT press the mash through the 4 mm sieve - pressing grinds the mash "
     "finer than 4 mm. Use the 4 mm sieve as a GAUGE only: tip 20 g of the dry "
     "mash onto the sieve and tilt it 3 times without pressing."),
    # a negation that does NOT reach the verb must not silence the instruction
    ("bf05_cooked_instant_oats_L5",
     "Do not sieve the batch: at L5 the flake pieces are the texture. The 4 mm "
     "sieve is a GAUGE here, not a step: press a 1 tbsp (15 g) aliquot through "
     "it and confirm the whole aliquot passes."),
    # scope prohibition, gauge still directed
    ("bf49_doubleskin_milk_L5",
     "Sieve check (a gauge, not a processing step - L5 is not a lump-free "
     "level): weigh out 25 g of the broken curd, rub it through the 4 mm sieve "
     "with the back of a spoon. Do not sieve the specimen you serve."),
    # conditional use
    ("bf37_apple_L5",
     "Do not blend to a smooth puree and do not sieve. Work the mash through a "
     "4 mm sieve only if a first pass shows lumps >4 mm."),
    # ordering prohibition, sieve still used
    ("bf46_smooth_red_bean_paste_L3",
     "Pass through the 4 mm sieve using the back of a spoon. Do not sieve after "
     "adding thickener."),
    # hypothetical: "if no 15 mm sieve is in the kit" is not a ban
    ("bf33_papaya_L6",
     "Gauge for oversize only: pass the dice over a 15 mm sieve - or, if no "
     "15 mm sieve is in the kit, measure every piece against the 100 mm ruler."),
]

# --- a sieve the card FORBIDS ------------------------------------------------

FORBIDDEN = [
    ("bf19_steamed_shrimp_peeled_L7",
     "Do NOT blend. Do NOT sieve - no 4 mm sieve is used at L7 (it is the L5 "
     "particle gauge and cannot size a >=15 mm piece). Do NOT add thickener."),
    ("bf29_steamed_eggplant_L6",
     "No thickener, no added liquid. Do not use a blender and do not pass the "
     "specimen through the 4 mm sieve: a 4 mm sieve cannot enforce a 15 mm "
     "limit, it would reduce the batch to <=4 mm (L5)."),
    ("bf54_saltedegg_steamed_pork_p_L6",
     "Size by hand, not by sieve: every piece must measure 8-15 mm. Do NOT "
     "press the crumbles through a 4 mm sieve: a 4 mm screen caps every piece "
     "at 4 mm, which is Level 5."),
    ("bf01_steamed_white_rice_L6",
     "Do not blend. Do not sieve - a 4 mm sieve cannot enforce a 15 mm limit, "
     "and passing L6 material through one destroys the piece structure."),
    ("bf16_silken_tofu_L6",
     "Do NOT sieve - a 4 mm sieve cannot size an L6 piece and would reduce the "
     "batch to L5 or finer."),
    ("bf17_firm_tofu_L6",
     "Do not blend, do not sieve, do not add thickener, do not add water. A "
     "4 mm sieve is a gauge, not a production step, and it cannot enforce a "
     "15 mm limit. Record: sieve not used, blender not used."),
    ("bf34_dragon_fruit_L5",
     "Do NOT sieve; the 4 mm sieve is not used because the per-level note "
     "states tiny seeds pass through and must be recorded, not removed."),
    ("bf47_tofu_fa_tofu_pudding_L4",
     "Do not sieve - a 4 mm sieve will shear the curd into a pourable liquid "
     "and is incompatible with L4."),
    ("bf48_coconut_jelly_L6",
     "Do NOT sieve - the cubes must stay intact, and a 4 mm sieve cannot "
     "enforce a 15 mm limit in any case."),
    ("bf60_lor_mai_gai_glutinous_ri_L7",
     "Do not blend, do not sieve - a 4 mm sieve cannot enforce a 15 mm limit."),
]

# --- not a sieve at all ------------------------------------------------------

# "the 4 mm sieve card" is the printed sizing fiducial in the capture kit
# (tools/capture_kit.py), not a sieve. No sieve is used and none is barred.
FIDUCIAL = [
    ("bf37_apple_L6",
     "Dice by hand into pieces <= 15 mm on the 4 mm sieve card (no cube passes "
     "through the 15 mm square). Do not blend. No thickener, no water added."),
]


def _prep(sentence):
    return "Weigh the specimen on the scale. " + sentence + " Record the mass."


def test_used_sieve_records_its_aperture():
    for cid, sentence in USED:
        mm, verdict = bcp.classify_sieve(_prep(sentence))
        assert verdict == "USED", f"{cid}: verdict {verdict}"
        assert mm is not None, f"{cid}: sieve used but no aperture recorded"


def test_forbidden_sieve_records_null():
    for cid, sentence in FORBIDDEN:
        mm, verdict = bcp.classify_sieve(_prep(sentence))
        assert verdict == "FORBIDDEN", f"{cid}: verdict {verdict}"
        assert mm is None, f"{cid}: forbidden sieve recorded sieve_mm={mm}"


def test_sieve_card_fiducial_is_not_a_sieve():
    for cid, sentence in FIDUCIAL:
        mm, verdict = bcp.classify_sieve(_prep(sentence))
        assert mm is None, f"{cid}: fiducial card recorded sieve_mm={mm}"
        assert verdict in ("NONE", "FORBIDDEN"), f"{cid}: verdict {verdict}"


def test_no_sieve_mentioned_is_null():
    mm, verdict = bcp.classify_sieve("Weigh 200 g. Do not blend. Plate it.")
    assert mm is None and verdict == "NONE"


def test_used_aperture_is_the_one_directed_not_the_one_barred():
    # bf33 names a 15 mm sieve it uses and a 4 mm sieve it forbids in the same
    # paragraph; the recorded aperture must be the one it uses.
    prep = ("Gauge for oversize only: pass the dice over a 15 mm sieve. "
            "Do NOT pass the papaya through a 4 mm sieve at any point - a 4 mm "
            "mesh cannot enforce a 15 mm limit.")
    mm, verdict = bcp.classify_sieve(prep)
    assert verdict == "USED" and mm == 15.0, (mm, verdict)


def test_a_negation_that_misses_the_verb_does_not_bar_the_sieve():
    # "not a step" negates "step", not the press that follows the colon.
    prep = ("The 4 mm sieve is a GAUGE here, not a step: press a 1 tbsp (15 g) "
            "aliquot through it and confirm the whole aliquot passes.")
    mm, verdict = bcp.classify_sieve(prep)
    assert verdict == "USED" and mm == 4.0, (mm, verdict)


def test_double_negative_is_a_requirement():
    prep = "Pass through the 4 mm sieve. Do not skip the sieve."
    mm, verdict = bcp.classify_sieve(prep)
    assert verdict == "USED" and mm == 4.0, (mm, verdict)
