#!/usr/bin/env python3
"""Build a machine-executable capture protocol from the 268 specimen cards.

The cards are complete prose written for a human to read. That is exactly why a
10-hour run is error-prone: at specimen 200 nobody re-reads section 2 carefully,
and a mis-recorded fork_pressure silently becomes a wrong ground-truth label.

This turns each card into structured fields so a capture tool can present it,
record the result, and emit schema-conformant rows without transcription. The
prose is preserved verbatim alongside the parsed fields so nothing is lost.

Output: capture/protocol.json  (regenerate after any card edit)
"""
import json, pathlib, re, sys, collections

REPO = pathlib.Path(__file__).resolve().parent.parent
CARDS = REPO / "docs/specimen_cards"
OUT = REPO / "capture"
OUT.mkdir(exist_ok=True)

# Two card formats exist in the corpus: numbered "1. PREP — text" (a minority)
# and a bare "PREP" header on its own line (the majority). Accept both.
SECTION_NUM = re.compile(r"^\s*\d\.\s*(PREP|IDDSI TEST|VISUAL QC|SHOTS|FAIL MODES)\s*[—–-]?\s*(.*)$")
SECTION_BARE = re.compile(r"^\s*(PREP|IDDSI TEST|VISUAL QC|SHOTS|FAIL MODES)\s*$")

# Which tests apply at which level, per docs/CAPTURE_PROTOCOL.md section 5 --
# NOT per intuition. 5.1 fork-pressure: L4, L6, L7 (supporting check for L5);
# 5.2 spoon-tilt: L3, L4, L5; 5.3 fork-drip: L3, L4; 5.4 syringe: every L3.
# An earlier version of this map wrongly included spoon-tilt at L6/L7 and
# omitted it at L3, which made a card audit report ~66% of the corpus as having a
# wrong test set when the CARDS were right and the AUDIT SPEC was wrong.
APPLICABLE = {
    "L3": ["spoon_tilt", "fork_drip", "flow_ml_remaining"],
    "L4": ["fork_pressure", "spoon_tilt", "fork_drip"],
    "L5": ["sieve_4mm_gauge", "spoon_tilt", "fork_pressure"],
    "L6": ["fork_pressure"],
    "L7": ["fork_pressure"],
}

name_re = re.compile(r"^(bf\d{2})_(.+)_L([3-7])$")


def parse_card(path):
    txt = path.read_text(encoding="utf-8", errors="replace")
    m = name_re.match(path.stem)
    if not m:
        return None
    bf, slug, lvl_digit = m.group(1), m.group(2), m.group(3)
    lvl = "L" + lvl_digit          # group 3 is the DIGIT, not "L4"

    secs = {}
    cur, buf = None, []
    for line in txt.splitlines():
        s = SECTION_NUM.match(line)
        if s:
            if cur:
                secs[cur] = " ".join(buf).strip()
            cur, buf = s.group(1), [s.group(2).strip()]
            continue
        b = SECTION_BARE.match(line)
        if b:
            if cur:
                secs[cur] = " ".join(buf).strip()
            cur, buf = b.group(1), []
            continue
        if cur:
            buf.append(line.strip())
    if cur:
        secs[cur] = " ".join(buf).strip()

    # pull machine-checkable parameters out of the prep prose
    prep = secs.get("PREP", "")
    params = {}
    g = re.search(r"([\d.]+)\s*g\s*thickener", prep)
    if g:
        params["thickener_g"] = float(g.group(1))
    g = re.search(r"add\s+([\d.]+)\s*ml", prep, re.I)
    if g:
        params["liquid_ml"] = float(g.group(1))
    g = re.search(r"blend\s+(\d+)\s*s", prep, re.I)
    if g:
        params["blend_seconds"] = int(g.group(1))
    g = re.search(r"(\d+)\s*mm\s*sieve", prep, re.I)
    if g:
        params["sieve_mm"] = float(g.group(1))

    tests = secs.get("IDDSI TEST", "")
    media = re.findall(r"[\w\-.]+\.(?:jpg|mp4)", secs.get("SHOTS", ""))

    return {
        "specimen_id": path.stem,
        "base_food_id": bf.upper().replace("BF", "BF"),
        "food_slug": slug,
        "level": int(lvl[1]),
        "level_tag": lvl,
        "applicable_tests": APPLICABLE[lvl],
        "prep": prep,
        "iddsi_test": tests,
        "visual_qc": secs.get("VISUAL QC", ""),
        "shots": secs.get("SHOTS", ""),
        "fail_modes": secs.get("FAIL MODES", ""),
        "prep_params": params,
        "media_expected": media,
        "card_path": str(path.relative_to(REPO)),
    }


cards, bad = [], []
for p in sorted(CARDS.glob("*.md")):
    r = parse_card(p)
    if r:
        cards.append(r)
    else:
        bad.append(p.name)

by_level = collections.Counter(c["level"] for c in cards)
by_food = collections.Counter(c["base_food_id"] for c in cards)
missing_sections = [c["specimen_id"] for c in cards
                    if not c["iddsi_test"] or not c["prep"]]

protocol = {
    "version": "v1",
    "generated_from": "docs/specimen_cards/",
    "count": len(cards),
    "levels": dict(sorted(by_level.items())),
    "base_foods": len(by_food),
    "unparsed_cards": bad,
    "cards_missing_required_sections": missing_sections,
    "specimens": cards,
}
(OUT / "protocol.json").write_text(json.dumps(protocol, indent=2, ensure_ascii=False))

print(f"cards parsed      : {len(cards)}")
print(f"unparsed          : {len(bad)} {bad[:5]}")
print(f"cards per level   : {dict(sorted(by_level.items()))}")
print(f"base foods        : {by_food and len(by_food)}")
print(f"missing required sections: {len(missing_sections)} {missing_sections[:5]}")
withparams = sum(1 for c in cards if c["prep_params"])
print(f"cards with parsed prep params: {withparams}")
print(f"wrote {OUT/'protocol.json'} ({(OUT/'protocol.json').stat().st_size} bytes)")

# flag level coverage imbalance -- L7 is thinnest
if by_level:
    lo, hi = min(by_level.values()), max(by_level.values())
    print(f"\nlevel coverage spread: {lo}-{hi} cards; thinnest = "
          f"L{min(by_level, key=by_level.get)}")
