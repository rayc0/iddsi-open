#!/usr/bin/env python3
"""audit_card_tests.py -- check every specimen card against the test-per-level map.

`docs/CAPTURE_PROTOCOL.md` section 5 is the authority on which IDDSI test
methods apply at which level:

    5.1 fork pressure : L4, L6, L7            (supporting check for L5)
    5.2 spoon tilt    : L3, L4, L5
    5.3 fork drip     : L3, L4
    5.4 syringe flow  : every L3 specimen (required)

A card is wrong in one of two ways, and only the first is a defect:

  * a test that APPLIES at the card's level is missing from the card's IDDSI
    TEST section (the operator has no way to pass the level honestly), or
  * a test that does NOT apply is presented as ACTIVE -- i.e. it is listed with
    a pass criterion and is not explicitly excluded. Naming a non-applicable
    test in order to exclude it ("Fork pressure is not an L3 test") is correct
    practice, not a defect.

Usage:
    python3 tools/audit/audit_card_tests.py [--cards DIR] [--out PATH]

Nothing outside the repo is read. The report is written to --out (default
stdout) and a JSON copy to <out>.json when --json is given.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

# docs/CAPTURE_PROTOCOL.md section 5 -- the authoritative map.
APPLIES: dict[int, set[str]] = {
    3: {"spoon_tilt", "fork_drip", "syringe_flow"},
    4: {"fork_pressure", "spoon_tilt", "fork_drip"},
    5: {"spoon_tilt", "fork_pressure"},   # fork pressure is a supporting check
    6: {"fork_pressure"},
    7: {"fork_pressure"},
}
REQUIRED: dict[int, set[str]] = {
    3: {"spoon_tilt", "fork_drip", "syringe_flow"},
    4: {"fork_pressure", "spoon_tilt", "fork_drip"},
    5: {"spoon_tilt"},
    6: {"fork_pressure"},
    7: {"fork_pressure"},
}

# How each test method is named in card prose. Deliberately broad: a card that
# mentions "syringe" in its IDDSI TEST section is talking about the syringe test.
PATTERNS: dict[str, re.Pattern[str]] = {
    "fork_pressure": re.compile(r"fork[ -]?press|thumbnail blan", re.I),
    "spoon_tilt": re.compile(r"spoon[ -]?tilt|tilt(?:ed|ing)? (?:the )?spoon|tilt to \d+", re.I),
    "fork_drip": re.compile(r"fork[ -]?drip|drips? (?:slowly )?through the (?:fork )?tines", re.I),
    "syringe_flow": re.compile(r"syringe", re.I),
}

# Phrases that mark a mention as an EXCLUSION rather than an active test. The
# repaired cards use several phrasings ("No fork drip at L5", "SKIP - 5.2",
# "applies at L3/L4 only", "is NOT in the L5 set", "5.3 is scoped to L3/L4",
# "not applicable", "must not be run"). A mention is only "active" when no
# exclusion phrase sits within 160 characters of it.
EXCLUSION = re.compile(
    r"\bSKIP\b"
    r"|\bno\s+(?:\d+\s*ml\s*/\s*\d+\s*s\s*)?[\w /-]{0,24}"
    r"(?:fork[ -]?drip|spoon[ -]?tilt|syringe|fork[ -]?press)"
    r"|is\s+not\s+(?:in|an?|applicable|performed|required|used)"
    r"|not\s+in\s+the\s+L\d\s+set"
    r"|applies\s+at\s+L[\d/L, and]*only"
    r"|is\s+scoped\s+to"
    r"|do\s+not\s+apply"
    r"|not\s+applicable"
    r"|not\s+evidence\s+for"
    r"|not\s+an?\s+L\d"
    r"|must\s+not\s+be\s+(?:run|recorded|performed|filmed|used|applied)"
    r"|do\s+not\s+(?:run|record|perform|film|use|apply)"
    r"|not\s+(?:performed|recorded|run|required|used|carried)",
    re.I,
)

# Two layouts ship in docs/specimen_cards: numbered ("2. IDDSI TEST", the format
# the 2026-10-12 repairs were written in) and unnumbered bare headers
# ("IDDSI TEST", the original generated format). Both are matched.
HEADER = r"(?:PREP|IDDSI\s*TESTS?|VISUAL\s*QC|SHOTS|FAIL\s*MODES)"
NUMBERED = re.compile(rf"(?m)^\s*[1-5][.)]\s*({HEADER})\b", re.I)
BARE = re.compile(rf"(?m)^\s*({HEADER})\s*$", re.I)


def _canon(name: str) -> str:
    n = re.sub(r"\s+", " ", name).strip().upper()
    if n.startswith("IDDSI"):
        return "IDDSI TEST"
    return n


def split_sections(text: str) -> dict[str, str]:
    """Return {SECTION NAME: body} for either shipped card layout.

    Some cards put the whole section on one line ("1. PREP - Weigh 200 g ..."),
    so the body starts at the end of the header match, not at the next newline.
    """
    hits = [(m.start(), m.end(), _canon(m.group(1))) for m in NUMBERED.finditer(text)]
    if not hits:
        hits = [(m.start(), m.end(), _canon(m.group(1))) for m in BARE.finditer(text)]
    hits.sort()
    out: dict[str, str] = {}
    for i, (pos, hdr_end, name) in enumerate(hits):
        end = hits[i + 1][0] if i + 1 < len(hits) else len(text)
        body = text[hdr_end:end].strip()
        # a later section name must not win over an earlier one
        if name not in out:
            out[name] = body
    return out


def card_level(path: Path) -> int:
    m = re.search(r"_L(\d)\.md$", path.name)
    if not m:
        raise ValueError(f"cannot read level from {path.name}")
    return int(m.group(1))


def classify(body: str, test: str) -> tuple[str, str]:
    """('absent'|'excluded'|'active', evidence) for one test method in one section.

    A mention counts as an exclusion only when an exclusion phrase sits within
    160 characters of it, so "No fork drip at L5" clears the mention next to it
    while a later independent mention of fork drip stays a live pass criterion.
    """
    pat = PATTERNS[test]
    hits = list(pat.finditer(body))
    if not hits:
        return "absent", ""
    for m in hits:
        window = body[max(0, m.start() - 160): m.end() + 160]
        ex = EXCLUSION.search(window)
        if ex:
            continue
        return "active", m.group(0)[:60]
    return "excluded", ""


def audit_card(path: Path) -> dict:
    level = card_level(path)
    text = path.read_text()
    sections = split_sections(text)
    test_body = sections.get("IDDSI TEST", "")
    states, evidence = {}, {}
    for t in PATTERNS:
        states[t], evidence[t] = classify(test_body, t)
    missing_required = sorted(t for t in REQUIRED[level] if states[t] == "absent")
    # a non-applicable test that is neither absent nor explicitly excluded
    wrongly_active = sorted(
        t for t in PATTERNS
        if t not in APPLIES[level] and states[t] == "active"
    )
    sections_present = sorted(k for k in ("PREP", "IDDSI TEST", "VISUAL QC", "SHOTS", "FAIL MODES") if sections.get(k, "").strip())
    empty_sections = [k for k in ("PREP", "IDDSI TEST", "VISUAL QC", "SHOTS", "FAIL MODES")
                      if k not in sections_present]
    return {
        "card": path.name,
        "level": level,
        "states": states,
        "evidence": {t: e for t, e in evidence.items() if e},
        "missing_required": missing_required,
        "non_applicable_active": wrongly_active,
        "empty_sections": empty_sections,
        "defect": bool(missing_required or wrongly_active or empty_sections),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cards", default="docs/specimen_cards")
    ap.add_argument("--out", default=None)
    ap.add_argument("--json", default=None)
    ap.add_argument("--only", default=None, help="comma-separated card stems to audit")
    args = ap.parse_args()

    cards_dir = Path(args.cards)
    paths = sorted(cards_dir.glob("bf*_L*.md"))
    if args.only:
        wanted = {s.strip() for s in args.only.split(",") if s.strip()}
        paths = [p for p in paths if p.stem in wanted or p.name in wanted]

    results = [audit_card(p) for p in paths]
    lines = []
    lines.append(f"# specimen-card test-set audit against docs/CAPTURE_PROTOCOL.md section 5")
    lines.append(f"cards audited: {len(results)}")
    defective = [r for r in results if r["defect"]]
    lines.append(f"cards with a defect: {len(defective)}")
    lines.append("")
    lines.append("| card | L | required present | non-applicable ACTIVE | empty sections |")
    lines.append("|---|---|---|---|---|")
    for r in results:
        req = ", ".join(sorted(APPLIES[r["level"]])) or "-"
        bad = ", ".join(r["non_applicable_active"]) or "-"
        empty = ", ".join(r["empty_sections"]) or "-"
        flag = "  <-- DEFECT" if r["defect"] else ""
        lines.append(f"| {r['card']} | L{r['level']} | {req} | {bad} | {empty} |{flag}")
    text = "\n".join(lines) + "\n"

    if args.out:
        Path(args.out).write_text(text)
    if args.json:
        Path(args.json).write_text(json.dumps(results, indent=1))
    print(text)
    if defective:
        print(f"DEFECTS: {len(defective)} of {len(results)} cards", file=sys.stderr)
        return 1
    print(f"CLEAN: 0 of {len(results)} cards carry a test-set defect", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
