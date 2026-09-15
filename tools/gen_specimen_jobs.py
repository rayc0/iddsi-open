#!/usr/bin/env python3
"""gen_specimen_jobs.py — emit one MiniMax job per (base food × level) operator card.

Reads docs/BASE_FOOD_LIST.md (60 foods, per-level notes, N/A cells) and
docs/SHOT_LIST.md (the shot spec), writes ~/mmqueue_jobs/bf<NN>_L<n>.md prompt
files for the DGX mmqueue runner. N/A cells are skipped. Run AFTER the capture
protocol pack exists. Usage: python3 gen_specimen_jobs.py [--repo ~/Projects/iddsi-open] [--out ~/mmqueue_jobs]
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

LEVELS = {
    3: ("Liquidised / Moderately thick",
        "thinned with thickener so it pours; drips slowly through fork tines; "
        "IDDSI flow test leaves 8-10 ml in the syringe after 10 s"),
    4: ("Pureed / Extremely thick",
        "blended smooth; holds shape on a spoon; does NOT flow through fork tines; "
        "no separate thin liquid; fork-drip test: sits in a mound above the tines"),
    5: ("Minced & moist",
        "particles <=4 mm, moist, no separate thin liquid; mashed with a fork; "
        "fork-pressure test: squashes and does not recover"),
    6: ("Soft & bite-sized",
        "soft pieces <=15 mm; fork-pressure test: thumbnail blanches white, "
        "piece squashes and does not recover"),
    7: ("Regular",
        "whole food as normally served; no processing; reference specimen"),
}

CARD_TEMPLATE = """You are writing ONE operator card for the IDDSI desk-specimen capture run
(single operator, desk setup: consumer blender, fork, 4 mm sieve, thickener,
digital scale, smartphone, A4 capture kit with 15 mm checkerboard + 100 mm ruler).

FOOD: {food_en} ({food_zh}) — bought as: {form}
LEVEL: L{n} — {level_name}
TARGET SPECIMEN: {level_desc}
PER-LEVEL NOTE FROM THE FOOD LIST: {note}

Write the card with exactly these sections, imperative voice, <=300 words total:
1. PREP — exact steps with quantities: starting amount (g), blender seconds if
   blended, sieve size if sieved, thickener grams per 100 ml if L3 (state it is
   the calibration starting point and the operator records the actual), water/liquid
   ml if any. Respect the per-level note; if it says a step is impossible, say so.
2. IDDSI TEST — the test(s) that prove THIS specimen is L{n}: fork pressure,
   spoon tilt, fork drip, and for L3 the 10 ml/10 s syringe flow test. Give the
   pass criterion for each.
3. VISUAL QC — what a correct specimen looks like on the fork/spoon (2-3 cues an
   operator can check in seconds).
4. SHOTS — the media checklist for this specimen per the project shot list:
   3 stills (45-degree hero with ruler+checkerboard in frame, top-down with
   checkerboard, fork-press close-up mid-press) + 2 clips (8-12 s fork-press /
   spoon-tilt; the IDDSI test film). Naming: {slug}_L{n}_take1.jpg / .mp4.
5. FAIL MODES — the 2-3 most likely ways this food fails at L{n} and the re-make rule.

Constraints: no medical/clinical wording, no patient references, no brand names
(say "thickener", brand recorded separately), English, no headers beyond the 5
section titles above."""


def parse_foods(text: str) -> list[dict]:
    foods = []
    # real table shape: | BF02 | Plain congee 白粥 | bought form | L7 | L6 | L5 | L4 | L3 | allergen |
    for line in text.splitlines():
        if not line.strip().startswith("| BF"):
            continue
        parts = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(parts) < 8:
            continue
        bf, name, form = parts[0], parts[1], parts[2]
        cells = {7: parts[3], 6: parts[4], 5: parts[5], 4: parts[6], 3: parts[7]}
        foods.append({"id": bf, "name": name, "form": form, "cells": cells})
    return foods


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=str(Path.home() / "Projects/iddsi-open"))
    ap.add_argument("--out", default=str(Path.home() / "mmqueue_jobs"))
    args = ap.parse_args()

    text = (Path(args.repo) / "docs/BASE_FOOD_LIST.md").read_text()
    foods = parse_foods(text)
    if len(foods) < 50:
        print(f"⚠️ parsed only {len(foods)} foods — check the table format before firing")
        return 1

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    n = skipped = 0
    for f in foods:
        ascii_name = re.sub(r"[^A-Za-z0-9 ]", "", f["name"]).strip()
        slug = re.sub(r"[^a-z0-9]+", "_", ascii_name.lower()).strip("_")[:24] or f["id"].lower()
        for lvl in (3, 4, 5, 6, 7):
            note = f["cells"][lvl]
            if not note or note.upper().startswith("N/A"):
                skipped += 1
                continue
            name, desc = LEVELS[lvl]
            prompt = CARD_TEMPLATE.format(food_en=f["name"], food_zh="", form=f["form"],
                                          n=lvl, level_name=name, level_desc=desc,
                                          note=note, slug=slug)
            (out / f"{f['id'].lower()}_{slug}_L{lvl}.md").write_text(prompt)
            n += 1
    print(f"{n} jobs written to {out} ({skipped} N/A cells skipped, {len(foods)} foods)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
