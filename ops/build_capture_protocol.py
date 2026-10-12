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
import collections
import json
import pathlib
import re
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent
CARDS = REPO / "docs/specimen_cards"
OUT = REPO / "capture"

# Four header shapes occur, or have occurred, in this corpus. All four are
# section boundaries to a human reader:
#   numbered, prose inline    "4. SHOTS — 3 stills + 2 clips: ..."
#   numbered, prose below     "4. SHOTS"  then the prose on the following lines
#   bare, prose below         "SHOTS"     then the prose on the following lines
#   bare, prose inline        "SHOTS — 3 stills + 2 clips: ..."
# The fourth shape is the one that was silently swallowed. It matches neither
# SECTION_NUM (no leading "4.") nor SECTION_BARE (the line does not end after the
# name), so no boundary fired and the entire SHOTS section -- every media
# filename the operator has to capture -- was folded into the FAIL MODES buffer.
# bf52_sweet_and_sour_pork_L7 shipped shots="" and media_expected=[] until the
# card itself was rewritten (commit a1f4851); the parser hole stayed open, and
# any card written in that shape would have lost its media list the same way.
# Found by qa03_capture (2026-10-12); regression test in
# ops/tests/test_build_capture_protocol.py.
SECTION_NUM = re.compile(r"^\s*\d\.\s*(PREP|IDDSI TEST|VISUAL QC|SHOTS|FAIL MODES)\s*[—–:-]?\s*(.*)$")
SECTION_BARE = re.compile(r"^\s*(PREP|IDDSI TEST|VISUAL QC|SHOTS|FAIL MODES)\s*$")
SECTION_INLINE = re.compile(r"^\s*(PREP|IDDSI TEST|VISUAL QC|SHOTS|FAIL MODES)\s*[—–:-]+\s*(.*)$")

# Anything else a human would still read as a section header ("4) SHOTS",
# "SHOTS - ...", "SHOTS:"). These do NOT start a section; they are recorded per
# card and reported, because the failure being fixed is silence. A header that
# matched nothing used to vanish into the previous section with no signal at
# all. Measured over the 267 cards: zero such lines today, so this fires only on
# a genuinely new shape.
ORPHAN_HEADER = re.compile(
    r"^\s*(?:\d+\s*[.)\-–—:]?\s*)?"
    r"(PREP|IDDSI TEST|VISUAL QC|SHOTS|FAIL MODES)"
    r"\s*(?:[—–:-].*)?$"
)

# Every section a card must carry. The parser used to check only PREP and IDDSI
# TEST, which is why an empty SHOTS -- the section whose whole job is the media
# checklist -- was not an error.
REQUIRED_SECTIONS = ("PREP", "IDDSI TEST", "VISUAL QC", "SHOTS", "FAIL MODES")
SECTION_KEY = {
    "PREP": "prep",
    "IDDSI TEST": "iddsi_test",
    "VISUAL QC": "visual_qc",
    "SHOTS": "shots",
    "FAIL MODES": "fail_modes",
}

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
    orphans = []
    for lineno, line in enumerate(txt.splitlines(), 1):
        s = SECTION_NUM.match(line) or SECTION_INLINE.match(line)
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
        if ORPHAN_HEADER.match(line):
            orphans.append({"line": lineno, "text": line.strip()})
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
        "orphan_header_lines": orphans,
        # Relative when the card lives in the repo (always, in production); the
        # absolute path for a fixture card outside it, so the parser stays
        # callable from a test.
        "card_path": (str(path.relative_to(REPO)) if REPO in path.parents
                      else str(path)),
    }


def build(cards_dir=CARDS):
    """Parse every card under cards_dir and return the protocol dict."""
    cards, bad = [], []
    for p in sorted(pathlib.Path(cards_dir).glob("*.md")):
        r = parse_card(p)
        if r:
            cards.append(r)
        else:
            bad.append(p.name)

    by_level = collections.Counter(c["level"] for c in cards)
    by_food = collections.Counter(c["base_food_id"] for c in cards)
    missing = [c for c in cards
               if any(not c[SECTION_KEY[s]] for s in REQUIRED_SECTIONS)]
    orphan_headers = [(c["specimen_id"], o) for c in cards
                      for o in c["orphan_header_lines"]]

    return {
        "version": "v1",
        "generated_from": "docs/specimen_cards/",
        "count": len(cards),
        "levels": dict(sorted(by_level.items())),
        "base_foods": len(by_food),
        "unparsed_cards": bad,
        "cards_missing_required_sections": [c["specimen_id"] for c in missing],
        "cards_missing_which_sections": {
            c["specimen_id"]: [s for s in REQUIRED_SECTIONS if not c[SECTION_KEY[s]]]
            for c in missing},
        "orphan_header_lines": orphan_headers,
        "specimens": cards,
    }


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    cards_dir = pathlib.Path(argv[0]) if argv else CARDS
    if not cards_dir.is_dir():
        print(f"no card directory at {cards_dir}", file=sys.stderr)
        return 1

    protocol = build(cards_dir)
    cards = protocol["specimens"]
    bad = protocol["unparsed_cards"]
    missing_sections = protocol["cards_missing_required_sections"]
    missing_detail = protocol["cards_missing_which_sections"]
    orphan_headers = protocol["orphan_header_lines"]

    OUT.mkdir(exist_ok=True)
    dest = OUT / "protocol.json"
    dest.write_text(json.dumps(protocol, indent=2, ensure_ascii=False))

    print(f"cards parsed      : {len(cards)}")
    print(f"unparsed          : {len(bad)} {bad[:5]}")
    print(f"cards per level   : {protocol['levels']}")
    print(f"base foods        : {protocol['base_foods']}")
    print(f"missing required sections: {len(missing_sections)} {missing_sections[:5]}")
    withparams = sum(1 for c in cards if c["prep_params"])
    print(f"cards with parsed prep params: {withparams}")
    nomed = [c["specimen_id"] for c in cards if not c["media_expected"]]
    print(f"cards with no media in SHOTS: {len(nomed)} {nomed[:5]}")
    print(f"wrote {dest} ({dest.stat().st_size} bytes)")

    # Loud, not silent. A header that matches no pattern is exactly how
    # bf52_sweet_and_sour_pork_L7 lost its whole media list (qa03_capture,
    # 2026-10-12): the line fell into the previous section and nothing reported
    # it. The exit status is deliberately unchanged -- regenerating the protocol
    # must not be blocked -- but the condition is written into protocol.json and
    # gated by ops/tests/test_build_capture_protocol.py, which fails the suite.
    if orphan_headers or missing_sections:
        print("\n" + "!" * 78, file=sys.stderr)
        print("WARNING: card sections the parser could not place.", file=sys.stderr)
        for stem, o in orphan_headers:
            print(f"  ORPHAN HEADER  {stem} line {o['line']}: {o['text'][:90]}",
                  file=sys.stderr)
        for stem, secs_missing in missing_detail.items():
            print(f"  MISSING SECTION {stem}: {', '.join(secs_missing)}", file=sys.stderr)
        print("  A line like this is folded into the previous section and its "
              "content is lost.\n  Give the card one of these headers: '4. SHOTS', "
              "'SHOTS', 'SHOTS - <prose>'.", file=sys.stderr)
        print("!" * 78 + "\n", file=sys.stderr)

    # flag level coverage imbalance -- L7 is thinnest
    levels = protocol["levels"]
    if levels:
        lo, hi = min(levels.values()), max(levels.values())
        print(f"\nlevel coverage spread: {lo}-{hi} cards; thinnest = "
              f"L{min(levels, key=levels.get)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
