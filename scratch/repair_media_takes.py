#!/usr/bin/env python3
"""Renumber the SHOTS section of cards where every media slot is named _take1.

19 cards list all five slots (3 stills + 2 clips) as `<slug>_L<n>_take1.<ext>`,
so the three stills overwrite each other on the card and the two clips
overwrite each other.  docs/SHOT_LIST.md section 2 fixes the take number "from
1 within the specimen and shot type", so the repair renumbers in order of
appearance: stills take1/take2/take3.jpg, clips take1/take2.mp4 -- the
convention 110 of the 267 cards already use, and the one the wave-1 repaired
cards (bf30_sweetcorn_kernels_L7, bf31_banana_L6, bf43_baked_egg_custard_L5)
follow.

Only the SHOTS section is touched, and only the take number.

Usage:  python3 scratch/repair_media_takes.py [--apply] [--diff] [--only ...]
"""
import argparse
import difflib
import pathlib
import re
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "ops"))
import audit_specimen_cards as A  # noqa: E402

MEDIA = re.compile(r"([\w\-.]+?_take)(\d+)(\.(?:jpg|mp4))")
HEADER_RE = re.compile(r"^\s*(?:\d\.\s*)?SHOTS\s*[—–-]?\s*(.*)$")


def renumber(shots_body):
    """Renumber media in order of appearance: jpg 1..3, mp4 1..2."""
    n_jpg = n_mp4 = 0
    out, pos = [], 0
    for m in MEDIA.finditer(shots_body):
        out.append(shots_body[pos:m.start()])
        stem, ext = m.group(1), m.group(3)
        if ext == ".jpg":
            n_jpg += 1
            n = n_jpg
        else:
            n_mp4 += 1
            n = n_mp4
        out.append(f"{stem}{n}{ext}")
        pos = m.end()
    out.append(shots_body[pos:])
    return "".join(out)


def analyse(path):
    txt = path.read_text(encoding="utf-8", errors="replace")
    r = A.check_card(path)
    if not r or not any(k == "duplicate_media" for k, _, _ in r["problems"]):
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
    if "SHOTS" not in spans:
        return None
    s0, s1 = spans["SHOTS"]
    head_line = lines[s0]
    m = HEADER_RE.match(head_line)
    inline = bool(m and m.group(1).strip())
    head_keep = head_line[:m.start(1)] if m else head_line
    rest = "\n".join(lines[s0 + 1:s1])
    body = (m.group(1) + (("\n" + rest) if rest else "")) if inline else rest
    new_body = renumber(body)
    if new_body == body:
        return None
    head_lines = ([head_keep + new_body] if inline
                  else [head_keep] + new_body.split("\n"))
    return {
        "card": path.stem,
        "old": txt,
        "new": "\n".join(lines[:s0] + head_lines + lines[s1:]),
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
    print(f"cards to renumber        : {len(rows)}")
    for r in rows:
        print(f"  {r['card']}")
    if a.diff:
        for r in rows:
            print("\n===== " + r["card"])
            for line in difflib.unified_diff(
                    r["old"].splitlines(), r["new"].splitlines(),
                    lineterm="", n=0):
                print(line)
    if a.apply:
        for r in rows:
            pathlib.Path(d / (r["card"] + ".md")).write_text(r["new"],
                                                             encoding="utf-8")
        print(f"\nrewrote {len(rows)} cards")


if __name__ == "__main__":
    main()
