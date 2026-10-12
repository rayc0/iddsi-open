#!/usr/bin/env python3
"""Specimen-card consistency checker -- CORRECTED exclusion handling.

Checks every card in docs/specimen_cards against the AUTHORITATIVE
test-per-level mapping in docs/CAPTURE_PROTOCOL.md section 5:

  5.1 fork-pressure : L4, L6, L7  (supporting check for L5)
  5.2 spoon-tilt    : L3, L4, L5
  5.3 fork-drip     : L3, L4
  5.4 syringe flow  : every L3

plus level-integrity rules that follow from section 2 (the desk definitions)
and section 6 (the workflow):
  - an L6/L7 specimen must never be sized through a 4 mm sieve (a 4 mm screen
    caps every particle at 4 mm = L5);
  - an L5 specimen must not be pressed through the 4 mm sieve into a smooth
    puree (that is L4); the sieve is a GAUGE at L5, not a screen.

WHY THIS FILE EXISTS
--------------------
The two first-wave checkers (lane_runs/cc06_cards/audit_cards.py and
lane_runs/cc02_cards/verify_cards.py) both answer the question "does this card
present a test as ACTIVE?" with a substring search for exclusion wording.  Both
are wrong in BOTH directions, and both errors were measured on this corpus:

  (a) MISSED EXCLUSIONS -- the vocabulary was too narrow.  "The L3 10 ml/10 s
      syringe flow test is not the L4 criterion and is not used here",
      "Syringe flow test is N/A at L6", "L4 does not require the syringe flow
      test" and "Do not perform the syringe flow test for L4" all name a test
      in order to exclude it, yet none matched.  Measured: 13 of 192 unflagged
      exclusion clauses on the 2026-10-12 corpus.

  (b) FALSE EXCLUSIONS -- two patterns were far too broad.  r"not run" matched
      "the specimen does not run between tines" (a physical behaviour, not a
      statement about running the test), and r"no (fork|spoon|syringe|flow|...)"
      matched "no flow, no slump" inside a live pass criterion.  Measured: 4
      clauses on the 2026-10-12 corpus were silenced by this, e.g.
      bf56_wonton_soup_small_wonton_L6 "Fork-drip test (on the broth only): a
      fork lifted from the broth should drip steadily through the prongs, not
      run in a continuous pour" -- an ACTIVE fork-drip test at L6, waved
      through.

Both errors inflate/deflate the headline count, so the first-wave totals
(cc06 149/267 cards, cc02 125/267 for CHECK A) are not a usable denominator.
This checker replaces the vocabulary with the documented list below and keeps
the rule set byte-for-byte identical otherwise.  It is intentionally STRICTER
than cc06 on (b): a clause that only looks like an exclusion because the
specimen "does not run" is reported as an active test.

A "test is present" means the card states it as something the operator runs and
records.  Naming a test in order to exclude it is good practice and is what the
repaired cards in this repo do, so it is never a defect.

Usage: python3 ops/audit_specimen_cards.py [--cards DIR] [--json OUT]
       [--only STEM,STEM] [--list-excluded]
"""
import argparse
import json
import pathlib
import re
import sys

# ---------------------------------------------------------------------------
# section 5, verbatim scope
# ---------------------------------------------------------------------------
APPLIES = {
    "fork_pressure": {"L4", "L6", "L7", "L5-supporting"},
    "spoon_tilt": {"L3", "L4", "L5"},
    "fork_drip": {"L3", "L4"},
    "flow_ml_remaining": {"L3"},
}
APPLICABLE = {
    "L3": ["spoon_tilt", "fork_drip", "flow_ml_remaining"],
    "L4": ["fork_pressure", "spoon_tilt", "fork_drip"],
    "L5": ["sieve_4mm_gauge", "spoon_tilt", "fork_pressure"],
    "L6": ["fork_pressure"],
    "L7": ["fork_pressure"],
}
SECTION_OF = {
    "fork_pressure": "5.1",
    "spoon_tilt": "5.2",
    "fork_drip": "5.3",
    "flow_ml_remaining": "5.4",
    "sieve_4mm_gauge": "2 (4 mm sieve as gauge)",
}
LEVELS_OF = {
    "fork_pressure": "L4/L6/L7 (supporting check at L5)",
    "spoon_tilt": "L3/L4/L5",
    "fork_drip": "L3/L4",
    "flow_ml_remaining": "every L3",
}
PRETTY = {
    "fork_pressure": "fork pressure",
    "spoon_tilt": "spoon tilt",
    "fork_drip": "fork drip",
    "flow_ml_remaining": "syringe flow",
    "sieve_4mm_gauge": "4 mm sieve gauge",
}

TEST_WORDS = {
    "fork_pressure": [r"fork[-\s]?press", r"fork\s+pressure"],
    "spoon_tilt": [r"spoon[-\s]?tilt"],
    "fork_drip": [r"fork[-\s]?drip"],
    "flow_ml_remaining": [r"syringe", r"flow\s+test", r"10\s*ml\s*/\s*10\s*s",
                          r"8[–\-]10\s*ml"],
    # L5 only, per section 2 ("4 mm sieve as gauge") -- a prep/gauge step,
    # not one of the four filmed tests in section 5.
    "sieve_4mm_gauge": [r"4\s*mm\s*sieve"],
}

# ---------------------------------------------------------------------------
# Exclusion vocabulary.
#
# A clause that names a test is an EXPLICIT EXCLUSION when it negates that
# test's applicability or its performance at THIS card's level.  Every pattern
# below is such a negation; nothing here matches a physical description of the
# food.  Patterns are grouped by what they negate, and each group notes the
# first-wave bug it fixes.
# ---------------------------------------------------------------------------

# 1. "applies" negated.  Fixes cc06's missed "N/A at L6".
_NEG_APPLIES = [
    r"not\s+applicable", r"\bN/?A\b", r"does\s+not\s+apply",
    r"do\s+not\s+apply", r"\bnot\s+applied\b", r"\bnot\s+in\s+the\s+L[3-7]\s+set\b",
    r"\bnot\s+an?\s+L[3-7]\s+test", r"\bis\s+not\s+an?\s+L[3-7]\b",
]

# 2. The test is not performed.  cc06's r"not run" also matched "the specimen
#    does not run between tines", so every pattern here requires either the
#    passive ("is/are/was/be not run") or an explicit object ("do not run it").
_NEG_PERFORMED = [
    r"\b(?:is|are|was|were|be|been)\s+not\s+(?:run|filmed|recorded|performed|"
    r"used|scored|required|applied)\b",
    r"\b(?:do|does|did)\s+not\s+(?:run|film|record|perform|use|apply|score)\s+"
    r"(?:it|them|this|that|these|the)\b",
    r"\bmust\s+not\s+be\s+(?:run|filmed|recorded|required|used|performed|applied)\b",
    r"\bnever\s+(?:run|film|record|use|perform)\b",
    r"\bnot\s+(?:run|filmed|recorded|performed|used|scored)\s+as\b",
    r"\bskip\b", r"\bomitt?ed\b", r"\bexcluded\b", r"withdrawn",
    # the test is physically impossible on this specimen
    r"\bcannot\s+be\s+(?:drawn|loaded|run|performed|used|filmed|recorded|"
    r"applied|scooped|filled|poured|pressed|tilted|lifted)\b",
    r"\bwill\s+not\s+(?:enter|leave|pass|fit)\b",
]

# 3. The test is not required / not needed.  Fixes cc06's missed
#    "L4 does not require the syringe flow test".
_NEG_REQUIRED = [
    r"\bnot\s+required\b", r"\bnot\s+needed\b", r"does\s+not\s+require",
    r"\bno\s+\S+\s+required\b", r"\bno\s+test\s+required\b",
]

# 4. "No <test>" said as a bare statement.  cc06's r"no (fork|spoon|syringe|
#    flow|sieve|10 ml)" matched "no flow, no slump" inside a live criterion, so
#    the test's own name (or the word "test") is now required.
_NO_TEST = [
    r"\bno\s+(?:[A-Za-z0-9/–.\-]+\s+){0,2}fork[-\s]?drip(?:\s+test)?\b",
    r"\bno\s+(?:[A-Za-z0-9/–.\-]+\s+){0,2}spoon[-\s]?tilt(?:\s+test)?\b",
    r"\bno\s+(?:[A-Za-z0-9/–.\-]+\s+){0,2}fork[-\s]?press(?:ure)?(?:\s+test)?\b",
    r"\bno\s+(?:[A-Za-z0-9/–.\-]+\s+){0,2}syringe(?:\s+(?:flow\s+)?test)?\b",
    r"\bno\s+(?:[A-Za-z0-9/–.\-]+\s+){0,2}flow\s+test\b",
    r"\bno\s+(?:[A-Za-z0-9/–.\-]+\s+){0,2}sieve\b",
    r"\bno\s+10\s*ml\s*(?:/\s*10\s*s\s*)?(?:syringe\s+)?(?:flow\s+)?test\b",
    r"\bno\s+.{0,40}?\s+test\s+(?:at|for|above|beyond|in)\s+L[3-7]\b",
]

# 6. The clause explicitly says the named thing is not a test.
_NOT_A_TEST = [
    r"\bnot\s+an?\s+[\w.\s]{0,40}?\s*test\b",
    r"\bis\s+a\s+[\w.\s]{0,40}?\s*check\b",
    r"\bcarries\s+no\s+information\b", r"\bno\s+pass\s+window\b",
]

# 7. A hypothetical reference ("reject if it would require the syringe test")
#    is not an instruction to run the test.
_CONDITIONAL = [
    r"\bif\b[^.]{0,160}?\b(?:require|would|needs?|needed)\b",
    r"\bapplies\s+only\s+if\b", r"\bwould\s+require\b",
    r"\bno\s+\S{0,24}\s+applies\b",
]

EXCLUDE_WORDS = (_NEG_APPLIES + _NEG_PERFORMED + _NEG_REQUIRED + _NO_TEST
                 + _NOT_A_TEST + _CONDITIONAL)


# 5. The clause scopes the test to a set of levels that does not include this
#    card's level ("5.2 covers L3/L4/L5 only" on an L6 card, "applies at
#    L4/L6/L7" on an L3 card, "are L3–L5 tests" on an L7 card).  These are
#    level-aware and handled in _scope_excluded(), not by a bare regex, because
#    "L4 only" on an L4 card is a statement about the card's own level.
_SEP = r"(?:\s*(?:/|,|&|and)\s*|\s*[–\-]\s*)"
_SCOPE_RE = [
    re.compile(r"\bL([3-7])(?:" + _SEP + r"L[3-7])*\s+only\b", re.I),
    re.compile(r"\bapplies\s+(?:at|to)\s+L[3-7](?:" + _SEP + r"L[3-7])*", re.I),
    re.compile(r"\bscope[sd]?\s+to\s+L[3-7](?:" + _SEP + r"L[3-7])*", re.I),
    re.compile(r"\bare\s+L[3-7][–\-]L[3-7]\s+tests?\b", re.I),
    re.compile(r"\bis\s+an?\s+L[3-7][–\-]L[3-7]\s+(?:test|method)\b", re.I),
    re.compile(r"\bL([3-7])\s+test\s+set\b", re.I),
]
_CRITERION_RE = re.compile(r"\bL([3-7])\s+criterion\b", re.I)


def _scope_excluded(clause, card_level):
    """True when the clause places the test outside this card's level."""
    here = card_level[1]
    for rx in _SCOPE_RE:
        for m in rx.finditer(clause):
            listed = set(re.findall(r"L([3-7])", m.group(0)))
            if listed and here not in listed:
                return True
    for m in _CRITERION_RE.finditer(clause):
        if m.group(1) != here:
            return True
    return False


def is_excluded(clause, card_level):
    """True when `clause` names a test only in order to exclude it."""
    if mentioned(clause, EXCLUDE_WORDS):
        return True
    return _scope_excluded(clause, card_level)

SECTION_RE = re.compile(
    r"^\s*(?:\d\.\s*)?(PREP|IDDSI TEST|VISUAL QC|SHOTS|FAIL MODES)\s*[—–-]?\s*(.*)$")
NAME_RE = re.compile(r"^(bf\d{2})_(.+)_L([3-7])$")
MEDIA_RE = re.compile(r"[\w\-.]+\.(?:jpg|mp4)")
# a trailing "Files: ..." index line restates the canonical filenames
MEDIA_INDEX_RE = re.compile(r"^\s*(?:files?|media)\s*:", re.I)


def split_sections(txt):
    """Section bodies.  Accepts '1. PREP — text' and a bare 'PREP' line; the
    trailing capture group is empty for the bare form, which is the majority
    format in this corpus (68 of 267 cards)."""
    secs, cur, buf = {}, None, []
    for line in txt.splitlines():
        m = SECTION_RE.match(line)
        if m:
            if cur:
                secs[cur] = "\n".join(buf).strip()
            cur, buf = m.group(1), [m.group(2).strip()]
            continue
        if cur:
            buf.append(line)
    if cur:
        secs[cur] = "\n".join(buf).strip()
    return secs


def clauses(text):
    """Split a card section into operator-readable clauses.

    Cards are written as bullets / labelled lines ("- SKIP - 5.2 spoon tilt.
    Applies at L3/L4/L5 only; ..."), so the clause, not the sentence, is the
    unit that carries the verdict. Periods inside "5.2" are protected first.
    """
    protected = re.sub(r"(\d)\.(\d)", lambda m: m.group(1) + "\x00" + m.group(2),
                       text)
    parts = re.split(r"[\n\r]+|\.\s+", protected)
    return [p.replace("\x00", ".") for p in parts if p.strip()]


def mentioned(text, patterns):
    return any(re.search(p, text, re.I) for p in patterns)


def is_excluded(clause, card_level):
    """True when `clause` names a test only in order to exclude it."""
    if mentioned(clause, EXCLUDE_WORDS):
        return True
    return _scope_excluded(clause, card_level)


def excluded_near(text, test, card_level):
    """True when every clause that names `test` also excludes it."""
    pats = TEST_WORDS[test]
    for cl in clauses(text):
        if not mentioned(cl, pats):
            continue
        if not is_excluded(cl, card_level):
            return False
    return True


def check_card(path, collect=False):
    txt = path.read_text(encoding="utf-8", errors="replace")
    m = NAME_RE.match(path.stem)
    if not m:
        return None
    lvl = "L" + m.group(3)
    secs = split_sections(txt)
    test_sec = secs.get("IDDSI TEST", "")
    prep = secs.get("PREP", "")
    qc = secs.get("VISUAL QC", "")
    fail = secs.get("FAIL MODES", "")
    shots = secs.get("SHOTS", "")

    problems = []
    active = []          # clauses judged ACTIVE, for --list-excluded auditing

    # 1. missing / stub sections
    for s in ("PREP", "IDDSI TEST", "VISUAL QC", "SHOTS", "FAIL MODES"):
        body = secs.get(s, "")
        if not body or "(MISSING)" in body.upper() or len(body) < 40:
            problems.append(("missing_section", s, body[:60]))

    # 2. non-applicable test stated as an active test in the IDDSI TEST section
    for test in ("fork_pressure", "spoon_tilt", "fork_drip", "flow_ml_remaining"):
        if test in APPLICABLE[lvl]:
            continue
        if test == "fork_pressure" and lvl == "L5":
            continue  # supporting check, allowed
        for cl in clauses(test_sec):
            if not mentioned(cl, TEST_WORDS[test]):
                continue
            if not is_excluded(cl, lvl):
                problems.append(("non_applicable_test", test, lvl))
                if collect:
                    active.append((test, cl))
                break

    # 3. applicable test absent from the card (the L5 4 mm gauge is written in
    #    PREP on most cards, so search the whole card, not just the test block)
    whole = "\n".join([prep, test_sec, qc, fail, shots])
    for test in APPLICABLE[lvl]:
        if not mentioned(whole, TEST_WORDS[test]):
            problems.append(("applicable_test_absent", test, lvl))

    # 4. 4 mm sieve used to SIZE an L6/L7 specimen
    if lvl in ("L6", "L7") and mentioned(prep, [r"4\s*mm\s*sieve"]):
        # allowed only if the card forbids it
        if not mentioned(prep, [r"do\s+not.*sieve", r"no\s+sieve", r"never.*sieve",
                                r"not.*sieve"]):
            problems.append(("sieve_sizes_L6_L7", "4 mm sieve in PREP", lvl))

    # 5. L5 pressed through the sieve into a puree
    if lvl == "L5" and mentioned(prep, [r"(press|pass|push).{0,60}4\s*mm\s*sieve",
                                        r"4\s*mm\s*sieve.{0,60}(press|back of)"]):
        if not mentioned(prep, [r"gauge", r"do\s+not.*(purée|puree|smooth)"]):
            problems.append(("L5_sieved_to_puree", "4 mm sieve used as a screen", lvl))

    # 6. duplicate media filenames.  Only SHOT SLOTS count: a trailing index
    #    line ("Files: x_take1.jpg / .mp4.") repeats the canonical filenames
    #    and is not a second slot, so it cannot overwrite anything.  cc06
    #    documented bf43_baked_egg_custard_L5 as exactly this case and left the
    #    card byte-identical; counting it would be a false positive.
    names = []
    for line in shots.splitlines():
        if MEDIA_INDEX_RE.match(line):
            continue
        names.extend(MEDIA_RE.findall(line))
    dupes = sorted({n for n in names if names.count(n) > 1})
    if dupes:
        problems.append(("duplicate_media", ",".join(dupes), len(dupes)))

    out = {
        "card": path.stem,
        "level": lvl,
        "applicable": APPLICABLE[lvl],
        "problems": problems,
    }
    if collect:
        out["active_clauses"] = active
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cards", default="docs/specimen_cards")
    ap.add_argument("--json", default=None)
    ap.add_argument("--only", default=None, help="comma-separated card stems")
    ap.add_argument("--list-excluded", action="store_true",
                    help="print the clause judged ACTIVE for each flag")
    a = ap.parse_args()

    d = pathlib.Path(a.cards)
    only = set(a.only.split(",")) if a.only else None
    rows = []
    for p in sorted(d.glob("*.md")):
        if p.stem == "INDEX":
            continue
        r = check_card(p, collect=a.list_excluded)
        if r and (only is None or r["card"] in only):
            rows.append(r)

    flagged = [r for r in rows if r["problems"]]
    print(f"cards checked            : {len(rows)}")
    print(f"cards with >=1 problem   : {len(flagged)}")
    by_kind = {}
    for r in flagged:
        for k, _, _ in r["problems"]:
            by_kind[k] = by_kind.get(k, 0) + 1
    print(f"problems by kind         : {json.dumps(by_kind, sort_keys=True)}")
    print()
    for r in flagged:
        print(f"{r['card']}  ({r['level']})")
        for k, v, extra in r["problems"]:
            print(f"    - {k}: {v}   [{extra}]")
        for test, cl in r.get("active_clauses", []):
            print(f"      ACTIVE {test}: {cl}")

    if a.json:
        pathlib.Path(a.json).write_text(json.dumps(rows, indent=1))
        print(f"\nwrote {a.json}")


if __name__ == "__main__":
    main()
