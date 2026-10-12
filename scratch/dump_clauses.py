#!/usr/bin/env python3
"""Diagnostic: dump the IDDSI TEST clauses that name a non-applicable test."""
import pathlib
import re
import sys

sys.path.insert(0, "/home/tun/iddsi/lane_runs/cc06_cards")
import audit_cards as A  # noqa: E402

d = pathlib.Path("docs/specimen_cards")
rows = []
for p in sorted(d.glob("bf*.md")):
    r = A.check_card(p)
    if not r:
        continue
    for k, v, e in r["problems"]:
        if k == "non_applicable_test":
            rows.append((p.stem, r["level"], v))

seen = set()
for stem, lvl, test in rows:
    txt = (d / (stem + ".md")).read_text(encoding="utf-8", errors="replace")
    secs = A.split_sections(txt)
    tsec = secs.get("IDDSI TEST", "")
    for cl in A.clauses(tsec):
        if A.mentioned(cl, A.TEST_WORDS[test]):
            print(f"--- {stem} {lvl} {test}")
            print(f"    CLAUSE: {cl}")
            print(f"    excl={A.mentioned(cl, A.EXCLUDE_WORDS)}")
