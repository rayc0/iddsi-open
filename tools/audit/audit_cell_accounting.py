#!/usr/bin/env python3
"""audit_cell_accounting.py — measure the 300-cell plan vs the parsed specimen cards.

Reproduces, from the repo files alone, the claim "60 base foods x 5 levels = 300
cells, 267 cards parsed, 33 cells absent". Reports exactly WHICH cells are absent
and whether each one is explained by an N/A cell in docs/BASE_FOOD_LIST.md.

Deliberately mirrors the N/A detection used by tools/gen_specimen_jobs.py
(`note.upper().startswith("N/A")`) so the two agree, and additionally flags
cells whose note merely MENTIONS "N/A" without starting with it, so a silently
dropped cell cannot hide behind a partial mention.

Writes a JSON report next to itself and prints a human summary.
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
FOOD_LIST = REPO / "docs" / "BASE_FOOD_LIST.md"
CARDS_DIR = REPO / "docs" / "specimen_cards"
OUT = Path(__file__).resolve().parent / "cell_accounting.json"

LEVELS = (3, 4, 5, 6, 7)
# BASE_FOOD_LIST.md table column order is L7, L6, L5, L4, L3.
TABLE_COL_TO_LEVEL = {3: 7, 4: 6, 5: 5, 6: 4, 7: 3}

CARD_RE = re.compile(r"^bf(\d{2})_.*_L([3-7])\.md$")


def parse_foods(text: str) -> list[dict]:
    foods = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        if not line.strip().startswith("| BF"):
            continue
        parts = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(parts) < 8:
            continue
        bf = parts[0]
        foods.append({
            "id": bf,
            "n": int(bf[2:]),
            "name": parts[1],
            "line": lineno,
            "cells": {lvl: parts[TABLE_COL_TO_LEVEL[lvl]] for lvl in LEVELS},
        })
    return foods


def main() -> int:
    foods = parse_foods(FOOD_LIST.read_text())
    if len(foods) != 60:
        print(f"ERROR: parsed {len(foods)} food rows, expected 60", file=sys.stderr)
        return 1

    ids = [f["id"] for f in foods]
    dup_rows = [i for i, c in Counter(ids).items() if c > 1]
    nums = sorted(f["n"] for f in foods)
    gaps = [n for n in range(1, 61) if n not in set(nums)]

    # 1. The planned grid: 60 foods x 5 levels.
    grid = {(f["id"], lvl) for f in foods for lvl in LEVELS}

    # 2. N/A cells, per the generator's rule.
    na_cells = {(f["id"], lvl): f["cells"][lvl]
                for f in foods for lvl in LEVELS
                if f["cells"][lvl].upper().startswith("N/A")}
    # 3. Cells that mention N/A but do not START with it (possible silent drops).
    soft_na = {(f["id"], lvl): f["cells"][lvl]
               for f in foods for lvl in LEVELS
               if not f["cells"][lvl].upper().startswith("N/A")
               and re.search(r"\bN/A\b", f["cells"][lvl], re.I)}

    # 4. Cards on disk.
    cards = {}
    bad_names = []
    for p in sorted(CARDS_DIR.glob("bf*_L*.md")):
        m = CARD_RE.match(p.name)
        if not m:
            bad_names.append(p.name)
            continue
        key = (f"BF{int(m.group(1)):02d}", int(m.group(2)))
        if key in cards:
            bad_names.append(f"{p.name} (duplicates {cards[key].name})")
        cards[key] = p
    extra_files = [p.name for p in sorted(CARDS_DIR.iterdir())
                   if p.is_file() and not CARD_RE.match(p.name)]

    missing = sorted(grid - set(cards))
    unexpected = sorted(set(cards) - grid)
    unexplained = [c for c in missing if c not in na_cells]
    missing_na = [c for c in missing if c in na_cells]
    card_for_na = sorted(set(cards) & set(na_cells))

    # 5. Duplication checks: same (food, level) twice, and byte-identical cards.
    import hashlib
    hashes: dict[str, list[str]] = {}
    for key, p in cards.items():
        hashes.setdefault(hashlib.sha256(p.read_bytes()).hexdigest(), []).append(p.name)
    identical_groups = [sorted(v) for v in hashes.values() if len(v) > 1]
    name_collisions = bad_names  # already holds "X (duplicates Y)" entries

    per_food_missing = Counter(f for f, _ in missing)
    per_level_missing = Counter(l for _, l in missing)
    full_coverage_foods = [f["id"] for f in foods if f["id"] not in per_food_missing]

    report = {
        "inputs": {
            "food_list": str(FOOD_LIST.relative_to(REPO)),
            "cards_dir": str(CARDS_DIR.relative_to(REPO)),
        },
        "planned_cells": len(grid),
        "food_rows_parsed": len(foods),
        "food_rows_duplicated": dup_rows,
        "food_numbering_gaps": gaps,
        "cards_on_disk": len(cards),
        "card_filename_anomalies": bad_names,
        "non_card_files_in_dir": extra_files,
        "absent_cells": len(missing),
        "absent_cells_explained_by_na": len(missing_na),
        "absent_cells_unexplained": len(unexplained),
        "absent_cell_list": [
            {"id": f, "level": l, "na_note": na_cells.get((f, l))} for f, l in missing
        ],
        "unexplained_cell_list": [
            {"id": f, "level": l, "note": next(x["cells"][l] for x in foods if x["id"] == f)}
            for f, l in unexplained
        ],
        "cards_present_for_na_cells": [{"id": f, "level": l} for f, l in card_for_na],
        "duplicate_card_filenames": name_collisions,
        "byte_identical_card_groups": identical_groups,
        "foods_with_all_five_cards": len(full_coverage_foods),
        "foods_with_absent_cells": len(per_food_missing),
        "cells_mentioning_na_without_leading_na": [
            {"id": f, "level": l, "note": v} for (f, l), v in sorted(soft_na.items())
        ],
        "unexpected_cards": [{"id": f, "level": l} for f, l in unexpected],
        "absent_cells_per_food": {k: v for k, v in sorted(per_food_missing.items())},
        "absent_cells_per_level": {str(k): v for k, v in sorted(per_level_missing.items())},
        "index_md_claim": (CARDS_DIR / "INDEX.md").read_text().splitlines()[0],
    }
    OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")

    print(f"food rows parsed                : {len(foods)} (gaps: {gaps}, dups: {dup_rows})")
    print(f"planned cells (60 x 5)         : {len(grid)}")
    print(f"cards on disk                   : {len(cards)}")
    print(f"absent cells                    : {len(missing)}")
    print(f"  explained by an N/A cell      : {len(missing_na)}")
    print(f"  UNEXPLAINED                   : {len(unexplained)}")
    print(f"cards written for an N/A cell   : {len(card_for_na)}")
    print(f"cards outside the planned grid  : {len(unexpected)}")
    print(f"absent cells per level          : {dict(sorted(per_level_missing.items()))}")
    print(f"foods with absent cells        : {len(per_food_missing)}")
    print(f"soft-N/A cells (mention only)   : {len(soft_na)}")
    print(f"duplicate (food, level) pairs   : {len(name_collisions)}")
    print(f"byte-identical card groups      : {len(identical_groups)}")
    print(f"foods with all five cards       : {len(full_coverage_foods)} of {len(foods)}")
    if unexplained:
        print("UNEXPLAINED DETAIL:")
        for f, l in unexplained:
            note = next(x["cells"][l] for x in foods if x["id"] == f)
            print(f"  {f} L{l}: {note}")
    if card_for_na:
        print("CARD EXISTS FOR AN N/A CELL:")
        for f, l in card_for_na:
            print(f"  {f} L{l}: {na_cells[(f, l)]}")
    print(f"report -> {OUT}")

    if "--markdown" in sys.argv:
        print()
        print(render_markdown(foods, report))
    return 0


# Reason classes, keyed on the level the row cannot reach. The row text is the
# evidence; the class is just a label so the 33 cells can be counted in groups.
LEVEL_REASON = {
    7: "no whole/regular form as bought and served",
    6: "cannot form soft bite-size pieces <= 15 mm",
    5: "no discrete particles <= 4 mm to mince",
    4: "cannot hold shape without a separate thin liquid",
    3: "cannot meet the L3 flow window honestly",
}


def reason_class(level: int) -> str:
    return LEVEL_REASON[level]


def render_markdown(foods: list[dict], report: dict) -> str:
    by_id = {f["id"]: f for f in foods}
    absent = [(c["id"], c["level"]) for c in report["absent_cell_list"]]
    classes: dict[str, list[str]] = {}
    for f, l in absent:
        classes.setdefault(reason_class(l), []).append(f"{f} L{l}")

    out = []
    out.append("| Planned | Parsed | Absent | Absent cells with an N/A reason | Unexplained |")
    out.append("|---|---|---|---|---|")
    out.append(f"| {report['planned_cells']} | {report['cards_on_disk']} | "
               f"{report['absent_cells']} | {report['absent_cells_explained_by_na']} | "
               f"{report['absent_cells_unexplained']} |")
    out.append("")
    out.append("| Integrity check | Result |")
    out.append("|---|---|")
    out.append(f"| food rows parsed (BF01–BF60, no gaps, no duplicate rows) | "
               f"{report['food_rows_parsed']} / gaps {report['food_numbering_gaps'] or 'none'} / "
               f"dups {report['food_rows_duplicated'] or 'none'} |")
    out.append(f"| cards written for a cell the list marks N/A | {len(report['cards_present_for_na_cells'])} |")
    out.append(f"| cards outside the planned 300-cell grid | {len(report['unexpected_cards'])} |")
    out.append(f"| duplicate (food, level) card filenames | {len(report['duplicate_card_filenames'])} |")
    out.append(f"| byte-identical card files | {len(report['byte_identical_card_groups'])} |")
    out.append(f"| foods with all five cards | {report['foods_with_all_five_cards']} of {report['food_rows_parsed']} |")
    out.append("")
    out.append("| Level | Absent cells | Share of the 33 |")
    out.append("|---|---|---|")
    for lvl in (3, 4, 5, 6, 7):
        n = report["absent_cells_per_level"].get(str(lvl), 0)
        out.append(f"| L{lvl} | {n} | {n / 33:.0%} |")
    out.append("")
    out.append("| Reason class (level the row cannot reach) | Cells |")
    out.append("|---|---|")
    for lvl in (7, 6, 5, 4, 3):
        label = reason_class(lvl)
        if label in classes:
            out.append(f"| L{lvl} — {label} | {', '.join(sorted(classes[label]))} |")
    out.append("")
    out.append("| Food | Cells absent | Cards parsed |")
    out.append("|---|---|---|")
    for f, n in sorted(report["absent_cells_per_food"].items(), key=lambda kv: int(kv[0][2:])):
        lv = ", ".join(f"L{l}" for i, l in absent if i == f)
        out.append(f"| {f} {by_id[f]['name']} | {lv} | {5 - n} |")
    return "\n".join(out)


if __name__ == "__main__":
    raise SystemExit(main())
