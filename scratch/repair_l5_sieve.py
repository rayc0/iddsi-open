#!/usr/bin/env python3
"""Repair the 21 cards that press the whole batch through the 4 mm sieve.

docs/CAPTURE_PROTOCOL.md section 2 defines L5 as "Minced and moist -- soft
particles <= 4 mm, moist throughout", evidenced by the "4 mm sieve as gauge";
section 3 lists the sieve as "particle gauge for L5; fibre/skin removal for
L4"; and section 6 step 4 says "Chop / mash to <= 4 mm particles (sieve as
gauge)".  A batch PRESSED through a 4 mm mesh comes out the far side as a
smooth puree with every particle capped at 4 mm -- that is L4 wearing an L5
label, and it is the single most common level-integrity defect in the corpus
(20 L5 cards + 1 L6 card).

The repair rewords each screening instruction into the gauge instruction the
protocol actually specifies and states the prohibition explicitly, so no
information is lost and the L5/L4 boundary is enforced rather than relaxed.

Usage:  python3 scratch/repair_l5_sieve.py [--apply] [--diff] [--only ...]
"""
import argparse
import difflib
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "ops"))

GAUGE = ("Gauge the {what} against the 4 mm sieve instead of screening it "
         "(protocol sections 2 and 6 step 4): scoop a sample onto the mesh and "
         "confirm the particles fall through the 4 mm openings under their own "
         "weight or one light tap of the {tool}. Do not press the batch "
         "through — that screens it to a smooth purée, which is L4, not L5.")

# card -> list of (old_exact_substring, new_text)
FIX = {
    "bf02_plain_congee_L5": [
        ("Press the congee through a 4 mm sieve with the back of a fork; "
         "discard what stays on the mesh. Re-weigh the sieved portion.",
         GAUGE.format(what="congee", tool="fork back")
         + " Re-chop anything held back on the mesh with the knife rather than "
           "forcing it through. Re-weigh the portion."),
        ("The per-level note requires soft grains ≤4 mm; sieving achieves this.",
         "The per-level note requires soft grains ≤4 mm; the gauge check "
         "confirms this."),
    ],
    "bf06_white_bread_L5": [
        ("Pass through the 4 mm sieve using a fork to push soft pulp through; "
         "discard any coarse residue retained on mesh — if more than ~20 % of "
         "the blend stays on the sieve, the bread is too fibrous to reach L5 as "
         "a single specimen and the operator should note this and switch to a "
         "fork-mash method instead: place 30 g torn bread in a bowl, add 30 ml "
         "water, mash with fork for 60 s until a uniform coarse paste forms, "
         "then sieve to confirm ≤4 mm.",
         "Gauge the blend against the 4 mm sieve instead of screening it "
         "(protocol sections 2 and 6 step 4): scoop a sample onto the mesh and "
         "confirm the soft pulp falls through the 4 mm openings under its own "
         "weight or one light tap of the fork; never push the batch through, "
         "because that screens it to a smooth purée, which is L4, not L5. If "
         "more than ~20 % of a sample is held back on the mesh as coarse "
         "residue, the bread is too fibrous to reach L5 as a single specimen "
         "and the operator should note this and switch to a fork-mash method "
         "instead: place 30 g torn bread in a bowl, add 30 ml water, mash with "
         "fork for 60 s until a uniform coarse paste forms, then gauge that "
         "paste against the sieve to confirm ≤4 mm."),
    ],
    "bf07_mantou_steamed_bun_L5": [
        ("Pass through the 4 mm sieve using the back of a spoon; discard "
         "retained lumps. Weigh sieved mash.",
         GAUGE.format(what="mash", tool="spoon back")
         + " Any lump held back on the mesh is re-chopped with the knife and "
           "re-mashed, never pushed through. Weigh the mash."),
    ],
    "bf11_steamed_fish_fillet_L5": [
        ("Pass through the 4 mm sieve using a spoon; discard what stays "
         "behind; record retained mass. Weigh the sieved mince.",
         GAUGE.format(what="mince", tool="spoon")
         + " Re-chop anything held back on the mesh with the knife and record "
           "the retained mass. Weigh the mince."),
    ],
    "bf15_hardboiled_egg_L5": [
        ("Press through the 4 mm sieve using the fork back; discard overs. "
         "Return the sieved mince to the bowl.",
         GAUGE.format(what="mince", tool="fork back")
         + " Re-chop any oversize piece held back on the mesh rather than "
           "forcing it through. Return the mince to the bowl."),
    ],
    "bf16_silken_tofu_L5": [
        ("Press the mash through the 4 mm sieve using the back of the fork or "
         "a spatula, discarding anything that will not pass; re-mash any "
         "retained lumps and sieve again.",
         "Gauge the mash against the 4 mm sieve instead of screening it "
         "(protocol sections 2 and 6 step 4): scoop a sample onto the mesh and "
         "confirm the curds fall through the 4 mm openings under their own "
         "weight or one light tap of the fork back. Do not press the mash "
         "through — that screens it to a smooth purée, which is L4, not L5. "
         "Re-mash any lump held back on the mesh and gauge again."),
        ("which the fork-and-sieve sequence delivers",
         "which the fork-mash and gauge sequence delivers"),
    ],
    "bf17_firm_tofu_L5": [
        ("Press through the 4 mm sieve using the back of a spoon; discard what "
         "stays behind. Return the sieved crumb to the bowl.",
         GAUGE.format(what="crumb", tool="spoon back")
         + " Re-chop whatever stays behind and gauge again. Return the crumb "
           "to the bowl."),
    ],
    "bf18_braised_pork_belly_L5": [
        ("Press through the 4 mm sieve with the back of a fork; if more than "
         "~10% stays behind, pulse the retained 10 g portion for 3 s and "
         "re-sieve.",
         GAUGE.format(what="mince", tool="fork back")
         + " If more than ~10% of a sample stays behind, pulse the retained "
           "10 g portion for 3 s and gauge again."),
        ("Mash the sieved mince with the back of the fork until visually "
         "uniform.",
         "Mash the minced portion with the back of the fork until visually "
         "uniform."),
    ],
    "bf24_braised_winter_melon_L5": [
        ("Press the melon through a 4 mm sieve using the back of a spoon, "
         "working in small batches; discard any fibrous bits that will not "
         "pass. Mash the sieved pulp with a fork until uniform.",
         "Gauge the melon against the 4 mm sieve instead of screening it "
         "(protocol sections 2 and 6 step 4): scoop a sample onto the mesh and "
         "confirm the pieces fall through the 4 mm openings under their own "
         "weight or one light tap of the spoon back, working in small batches. "
         "Do not press the batch through — that screens it to a smooth purée, "
         "which is L4, not L5. Discard any fibrous bit that will not pass the "
         "gauge. Mash the pulp with a fork until uniform."),
    ],
    "bf25_stewed_hairy_gourd_L5": [
        ("Press all material through the 4 mm sieve using the back of a spoon; "
         "discard residue that will not pass. Return sieved mash to bowl and "
         "fold gently to redistribute moisture;",
         GAUGE.format(what="mash", tool="spoon back")
         + " Re-chop and re-mash any residue that will not pass the gauge. "
           "Return the mash to the bowl and fold gently to redistribute "
           "moisture;"),
    ],
    "bf26_steamed_carrot_L6": [
        ("Pass through the 4 mm sieve using the back of a spoon; press firmly "
         "so soft material passes. Weigh the sieved mash. Record starting "
         "weight, blended weight, and any material retained on the sieve.",
         "No sieve at any point: L6 is sized against the 100 mm ruler, never "
         "through a 4 mm mesh. A 4 mm screen caps every particle at 4 mm, which "
         "is the L5 ceiling, and this card's target is 15 mm cubes (protocol "
         "sections 2 and 3). Keep the 15 mm dice whole; if the pulse broke any "
         "piece below 8 mm, re-dice from a firmer batch rather than rescuing "
         "it by sieving. Weigh the mash. Record starting weight, blended "
         "weight, and any material discarded."),
    ],
    "bf28_steamed_bitter_melon_L5": [
        ("Pass through 4 mm sieve; discard coarse residue. Mash retained "
         "fraction with fork until uniformly minced.",
         "Gauge the mince against the 4 mm sieve instead of screening it "
         "(protocol sections 2 and 6 step 4): scoop a sample onto the mesh and "
         "confirm the particles fall through the 4 mm openings under their own "
         "weight or one light tap. Do not press the batch through — that "
         "screens it to a smooth purée, which is L4, not L5. Re-chop any "
         "coarse residue held back on the mesh. Mash the mince with a fork "
         "until uniformly minced."),
    ],
    "bf35_soft_ripe_persimmon_L5": [
        ("Sieve the mash through the 4 mm sieve using the back of a spoon; "
         "discard anything that will not pass. Press retained material once "
         "more; if residue still exceeds a pea-sized clump, sieve again.",
         "Gauge the mash against the 4 mm sieve instead of screening it "
         "(protocol sections 2 and 6 step 4): scoop a sample onto the mesh and "
         "confirm the pulp falls through the 4 mm openings under its own weight "
         "or one light tap of the spoon back. Do not press the batch through — "
         "that screens it to a smooth purée, which is L4, not L5. Re-mash "
         "anything that will not pass the gauge; if a clump still exceeds 4 mm "
         "after re-mashing, discard it."),
    ],
    "bf36_pear_L5": [
        ("Pass the mash through the 4 mm sieve, pressing with the back of the "
         "spoon; discard retained fibres. Re-weigh; target 180-200 g yield. If "
         "the mash looks wet or releases pooling liquid, blend 5-10 s to "
         "homogenise, then re-sieve.",
         "Gauge the mash against the 4 mm sieve instead of screening it "
         "(protocol sections 2 and 6 step 4): scoop a sample onto the mesh and "
         "confirm the crumbs fall through the 4 mm openings under their own "
         "weight or one light tap of the spoon back. Do not press the batch "
         "through — that screens it to a smooth purée, which is L4, not L5. "
         "Discard retained fibres. Re-weigh; target 180-200 g yield. If the "
         "mash looks wet or releases pooling liquid, re-mash with the fork "
         "rather than blending: blending destroys the L5 particle structure."),
    ],
    "bf38_watermelon_L5": [
        ("if the mash looks wet, press gently against the 4 mm sieve to blot "
         "excess juice",
         "if the mash looks wet, blot excess juice against the clean underside "
         "of the 4 mm sieve mesh — the sieve is used only to blot here, never "
         "to screen the batch through, which would make it an L4 purée"),
    ],
    "bf47_tofu_fa_tofu_pudding_L5": [
        ("Sieve once through the 4 mm sieve, pressing gently with the fork; "
         "discard what stays above.",
         "Gauge the curds against the 4 mm sieve once instead of screening them "
         "(protocol sections 2 and 6 step 4): scoop a sample onto the mesh and "
         "confirm the curds fall through the 4 mm openings under their own "
         "weight or one light tap of the fork. Do not press the batch through — "
         "that screens it to a smooth purée, which is L4, not L5. Discard what "
         "stays above the gauge."),
        ("Mash the sieved curd briefly with the fork",
         "Mash the curd briefly with the fork"),
    ],
    "bf52_sweet_and_sour_pork_L5": [
        ("Press the mince through the 4 mm sieve using the back of a spoon; "
         "discard what stays behind. Return sieved mince to the bowl.",
         GAUGE.format(what="mince", tool="spoon back")
         + " Re-chop what stays behind and gauge again. Return the mince to "
           "the bowl."),
    ],
    "bf55_yangzhou_fried_rice_L5": [
        ("Tip onto the 4 mm sieve; press and discard what does not pass, "
         "noting retained mass. Return the passing portion to a bowl, drizzle "
         "25–30 ml warm stock, and mash with a fork until evenly moist and "
         "sticky.",
         "Gauge the mince against the 4 mm sieve instead of screening it "
         "(protocol sections 2 and 6 step 4): scoop a sample onto the mesh and "
         "confirm the particles fall through the 4 mm openings under their own "
         "weight or one light tap. Do not tip the batch onto the sieve and "
         "press it through — that screens it to a smooth purée, which is L4, "
         "not L5. Discard and note the retained mass of anything that does not "
         "pass the gauge. Return the passing portion to a bowl, drizzle 25–30 "
         "ml warm stock, and mash with a fork until evenly moist and sticky."),
    ],
    "bf57_braised_beef_brisket_noo_L5": [
        ("Tip onto 4 mm sieve; press through with the back of a spoon, discard "
         "retained chunks.",
         "Gauge the mince against the 4 mm sieve instead of screening it "
         "(protocol sections 2 and 6 step 4): scoop a sample onto the mesh and "
         "confirm the particles fall through the 4 mm openings under their own "
         "weight or one light tap of the spoon back. Do not tip the batch onto "
         "the sieve and press it through — that screens it to a smooth purée, "
         "which is L4, not L5. Discard retained chunks that will not pass the "
         "gauge."),
    ],
    "bf58_tomato_egg_stirfry_L5": [
        ("Press through a 4 mm sieve, pushing soft solids through with the back "
         "of a spoon and reserving any seeds or skins that stay behind.",
         "Gauge the stir-fry against the 4 mm sieve instead of screening it "
         "(protocol sections 2 and 6 step 4): scoop a sample onto the mesh and "
         "confirm the soft solids fall through the 4 mm openings under their "
         "own weight or one light tap of the spoon back. Do not press the batch "
         "through — that screens it to a smooth purée, which is L4, not L5. "
         "Reserve any seeds or skins held back on the mesh; they are discarded, "
         "not forced through."),
        ("Weigh the sieved portion on the digital scale.",
         "Weigh the portion on the digital scale."),
    ],
    "bf59_steamed_chicken_with_mus_L5": [
        ("Press the mash through a 4 mm sieve, discarding anything retained "
         "above.",
         GAUGE.format(what="mash", tool="fork")
         + " Discard anything retained above the gauge."),
    ],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cards", default="docs/specimen_cards")
    ap.add_argument("--only", default=None)
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--diff", action="store_true")
    a = ap.parse_args()
    d = pathlib.Path(a.cards)
    only = set(a.only.split(",")) if a.only else None

    missing, done = [], []
    for card, pairs in sorted(FIX.items()):
        if only and card not in only:
            continue
        p = d / (card + ".md")
        if not p.exists():
            missing.append((card, "file not found"))
            continue
        txt = p.read_text(encoding="utf-8")
        new = txt
        for old, rep in pairs:
            if old not in new:
                missing.append((card, old[:60]))
                continue
            new = new.replace(old, rep, 1)
        if new == txt:
            missing.append((card, "no change"))
            continue
        done.append((card, txt, new))
        if a.apply:
            p.write_text(new, encoding="utf-8")

    print(f"cards repaired           : {len(done)}")
    for card, _, _ in done:
        print(f"  {card}")
    if missing:
        print(f"UNMATCHED ({len(missing)}):")
        for card, frag in missing:
            print(f"  {card}: {frag!r}")
    if a.diff:
        for card, old, new in done:
            print("\n===== " + card)
            for line in difflib.unified_diff(old.splitlines(), new.splitlines(),
                                             lineterm="", n=0):
                print(line)


if __name__ == "__main__":
    main()
