#!/usr/bin/env python3
"""Repair specimen cards that state a test as ACTIVE which
docs/CAPTURE_PROTOCOL.md section 5 does not map to the card's level.

For every "test statement" in a card's IDDSI TEST section that runs a
non-applicable test, this script makes two edits:

  1. IDDSI TEST -- the statement is replaced by the house-style exclusion the
     wave-1 repaired cards already use (bf30_sweetcorn_kernels_L7,
     bf31_banana_L6, bf39_orange_segments_peeled_L6, bf41_plain_yogurt_L3):

         Fork drip: not applicable at L7 (5.3 covers L3/L4 only) and is not run.

  2. VISUAL QC -- the observation the statement used to carry is not thrown
     away.  It is appended, explicitly framed as a by-eye check and NOT a
     section 5 test, so no information is lost and no gate is weakened:

         By-eye check only (not a section 5.3 test at L7): lift the fork;
         nothing drips or runs off.

A "test statement" is a labelled clause plus any immediately following
pass/fail clause, because these cards are written as
"Fork pressure: <instruction>. Pass: <criterion>." and replacing only the
first sentence would orphan the criterion.

Statements this script will not touch on its own are reported as MANUAL and
left byte-identical: whole-set summaries ("All three must pass"), criteria
continued by an unlabelled clause, and labels that appear at the end of a
sentence.  Those are hand-repaired.

Usage:  python3 scratch/repair_active_tests.py [--apply] [--diff]
        [--cards DIR] [--only STEM,STEM]
"""
import argparse
import difflib
import json
import pathlib
import re
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "ops"))
import audit_specimen_cards as A  # noqa: E402

NAMES = {
    "fork_pressure": r"fork[-\s]?press(?:ure)?(?:\s+test)?",
    "spoon_tilt": r"spoon[-\s]?tilt(?:\s+test)?",
    "fork_drip": r"fork[-\s]?drip(?:\s+test)?",
    "flow_ml_remaining": (r"(?:10\s*ml\s*/\s*10\s*s\s*)?(?:syringe(?:\s+flow)?"
                          r"(?:\s+test)?|flow\s+test|flow-test)"),
}
# "<bullet / '2)' / '5.1'> <test name> <(paren)> <':' or em-dash>"
LABEL = {t: re.compile(
    r"^(?P<pre>[\s\-–—•*(]*\d{0,2}[).:\s]*)(?:5\.\d\s+)?(?P<name>" + n
    + r")(?P<paren>\s*(?:\([^)]*\)|[\w\s'-]{0,32}?))?(?P<sep>\s*(?::|—|–))\s*",
    re.I)
    for t, n in NAMES.items()}

# a clause that continues the statement above it
CONT = re.compile(r"^\s*(pass|fail|reject|accept|record|note|result|qc)\b", re.I)
# a clause that is a whole-set summary and must be reworded by hand
SUMMARY = re.compile(r"^\s*(all\s+(three|\w+\s+must|three\s+pass)|if\s+any\s+of|"
                     r"repeat\s+each)\b", re.I)
HEADER_RE = re.compile(r"^\s*(?:\d\.\s*)?IDDSI TEST\s*[—–-]?\s*")

SEC = A.SECTION_OF
LEVELS = A.LEVELS_OF


def clause_offsets(body):
    """[(start, end, text)] for clauses(body), offsets into `body`."""
    out, pos = [], 0
    for cl in A.clauses(body):
        i = body.find(cl, pos)
        if i < 0:
            i = pos
        out.append((i, i + len(cl), cl))
        pos = i + len(cl)
    return out


def label_of(clause):
    for t, rx in LABEL.items():
        if rx.match(clause):
            return t, rx.match(clause)
    return None, None


def tidy(text):
    t = re.sub(r"\s+", " ", text).strip()
    t = re.sub(r"\.(\s+|$)", "; ", t).strip()
    return t.rstrip(".;, ").strip()


def statements(body):
    """Group clause offsets into labelled statements.

    Returns (list_of_(start,end,test,clause_list), manual_reasons).
    """
    cls = clause_offsets(body)
    groups, manual = [], []
    i = 0
    while i < len(cls):
        s, e, cl = cls[i]
        t, m = label_of(cl)
        if t is None:
            # unlabelled clause: only a problem if it names a non-applicable
            # test without excluding it -- the caller decides
            groups.append((s, e, None, [cl], "unlabelled"))
            i += 1
            continue
        members = [cl]
        j = i + 1
        while j < len(cls) and CONT.match(cls[j][2]):
            members.append(cls[j][2])
            j += 1
        groups.append((s, cls[j - 1][1], t, members, "labelled"))
        i = j
    return groups, manual


def analyse(path):
    """Return a repair plan for one card, or None if nothing to do."""
    txt = path.read_text(encoding="utf-8", errors="replace")
    r = A.check_card(path)
    if not r:
        return None
    lvl = r["level"]
    flagged = {v for k, v, _ in r["problems"] if k == "non_applicable_test"}
    if not flagged:
        return None

    lines = txt.split("\n")
    heads = []
    for i, ln in enumerate(lines):
        m = A.SECTION_RE.match(ln)
        if m:
            heads.append((i, m.group(1), m.group(2)))
    spans = {}
    for j, (i, name, _rest) in enumerate(heads):
        end = heads[j + 1][0] if j + 1 < len(heads) else len(lines)
        spans[name] = (i, end)
    if "IDDSI TEST" not in spans or "VISUAL QC" not in spans:
        return None
    t0, t1 = spans["IDDSI TEST"]
    qc0, qc1 = spans["VISUAL QC"]

    head_line = lines[t0]
    m = HEADER_RE.match(head_line)
    # "2. IDDSI TEST — <first statement>" keeps its first statement on the
    # header line; a bare "IDDSI TEST" line has none.
    head_keep = head_line[:m.end()] if m else head_line
    rest = "\n".join(lines[t0 + 1:t1])
    # a numbered header carries its first statement inline; a bare header
    # line is followed by the body (and usually a blank line) as-is
    if m and head_line[m.end():].strip():
        body = head_line[m.end():] + (("\n" + rest) if rest else "")
    else:
        body = rest

    groups, _ = statements(body)
    replaces = []                   # (start, end, test, new_text, body_text)
    manual = []
    for (s, e, t, members, kind) in groups:
        text = " ".join(members)
        # which non-applicable tests does this statement name unexcluded?
        bad = [x for x in flagged
               if A.mentioned(text, A.TEST_WORDS[x]) and not A.is_excluded(text, lvl)]
        if not bad:
            continue
        if t is None:
            manual.append((bad[0], text))
            continue
        if t not in bad:
            # statement leads with an applicable test but also names a
            # non-applicable one -- needs a human
            manual.append((bad[0], text))
            continue
        m = LABEL[t].match(members[0])
        head = members[0][:m.end()]
        rest = tidy(members[0][m.end():])
        for extra in members[1:]:
            extra = tidy(extra)
            rest = (rest + "; " + extra[0].lower() + extra[1:]).strip("; ")
        if not rest:
            manual.append((t, text))
            continue
        # the clause separator (". " or newline) that follows the statement is
        # left in place, so the replacement must not add its own full stop
        base = (head.rstrip() + " not applicable at " + lvl + " (" + SEC[t]
                + " covers " + LEVELS[t] + " only) and is not run, filmed or "
                "recorded as " + lvl + " evidence")
        new = base + ("" if body[e:e + 1] == "." else ".")
        replaces.append((s, e, t, new, rest[:240]))

    if not replaces and not manual:
        return None

    # ---- apply replacements to the IDDSI TEST body -------------------------
    new_body = body
    for s, e, t, new, rest in sorted(replaces, key=lambda x: -x[0]):
        new_body = new_body[:s] + new + new_body[e:]
    if new_body != body and not new_body.endswith((".", "\n")):
        new_body += "."

    # ---- VISUAL QC gains the retained by-eye observations ------------------
    notes = [f"By-eye check only (not a section {SEC[t]} test at {lvl}): {b}."
             for _, _, t, _, b in replaces if b]
    qc_region = "\n".join(lines[qc0:qc1])
    if notes:
        stripped = qc_region.rstrip("\n")
        trail = qc_region[len(stripped):]        # keep the blank line(s)
        if stripped == "":
            qc_region = "\n".join(notes) + trail
        else:
            qc_region = stripped + " " + " ".join(notes) + trail

    inline = bool(m and head_line[m.end():].strip())
    head_lines = ([head_keep + new_body] if inline
                  else [head_keep] + new_body.split("\n"))
    out_lines = (lines[:t0] + head_lines
                 + lines[t1:qc0] + qc_region.split("\n") + lines[qc1:])
    return {
        "card": path.stem,
        "level": lvl,
        "replaces": replaces,
        "manual": manual,
        "old": txt,
        "new": "\n".join(out_lines),
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
    rows = []
    for p in sorted(d.glob("bf*.md")):
        if only and p.stem not in only:
            continue
        res = analyse(p)
        if res:
            rows.append(res)

    auto = [r for r in rows if r["replaces"]]
    man = [r for r in rows if r["manual"]]
    print(f"cards with a repairable statement : {len(auto)}")
    print(f"clauses auto-repaired             : {sum(len(r['replaces']) for r in auto)}")
    print(f"cards left for a human            : {len(man)}")
    print(f"clauses left for a human          : {sum(len(r['manual']) for r in man)}")
    for r in man:
        for t, text in r["manual"]:
            print(f"  MANUAL {r['card']:40s} {r['level']} {t:18s} {text[:90]}")
    if a.diff:
        for r in auto:
            print("\n===== " + r["card"])
            for line in difflib.unified_diff(
                    r["old"].splitlines(), r["new"].splitlines(),
                    lineterm="", n=0):
                print(line)
    if a.apply:
        for r in auto:
            pathlib.Path(d / (r["card"] + ".md")).write_text(r["new"],
                                                             encoding="utf-8")
        print(f"\nrewrote {len(auto)} cards")


if __name__ == "__main__":
    main()
