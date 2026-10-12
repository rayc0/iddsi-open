#!/usr/bin/env python3
"""Diagnostic: for EVERY card, list every IDDSI-TEST clause that names a
non-applicable test, and whether cc06's exclusion regex catches it."""
import pathlib
import re
import sys
import collections

sys.path.insert(0, "/home/tun/iddsi/lane_runs/cc06_cards")
import audit_cards as A  # noqa: E402

d = pathlib.Path("/home/tun/iddsi/lanes2/def05_cards_remaining/docs/specimen_cards")
out = []
for p in sorted(d.glob("bf*.md")):
    r = A.check_card(p)
    if not r:
        continue
    txt = p.read_text(encoding="utf-8", errors="replace")
    tsec = A.split_sections(txt).get("IDDSI TEST", "")
    lvl = r["level"]
    for test in ("fork_pressure", "spoon_tilt", "fork_drip", "flow_ml_remaining"):
        if test in A.APPLICABLE[lvl]:
            continue
        if test == "fork_pressure" and lvl == "L5":
            continue
        for cl in A.clauses(tsec):
            if A.mentioned(cl, A.TEST_WORDS[test]):
                out.append((p.stem, lvl, test, cl, A.mentioned(cl, A.EXCLUDE_WORDS)))

print(f"total non-applicable test mentions in IDDSI TEST sections: {len(out)}")
print(f"caught as excluded by cc06 regexes: {sum(1 for o in out if o[4])}")
print(f"NOT caught (= cc06 flags): {sum(1 for o in out if not o[4])}")
print()
cnt = collections.Counter()
for stem, lvl, test, cl, ex in out:
    if not ex:
        cnt[(lvl, test)] += 1
print("uncaught by (level,test):", dict(sorted(cnt.items())))
print()
with open("/tmp/all_clauses.txt", "w") as f:
    for stem, lvl, test, cl, ex in out:
        f.write(f"{'EXCL' if ex else 'ACTV'}\t{stem}\t{lvl}\t{test}\t{cl}\n")
