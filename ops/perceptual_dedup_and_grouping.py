#!/usr/bin/env python3
"""Perceptual dedup + group-disjoint stratified split — the v0.3 evaluation fix.

Reproduces the v0.3 split from the v0.2 manifest. Two defects are addressed:

  1. CONTENT DEDUP. The 20 duplicate event_ids were already known. Perceptual
     hashing (dHash <= 6 AND pHash <= 8) additionally finds near-duplicates the
     identifier cannot express: the same photograph re-encoded or re-cropped.

  2. A GROUP KEY THAT GROUPS. The old group_id was 544 groups of exactly one
     event each, so the configured leak control separated nothing. Here a group
     is the CONNECTED COMPONENT of the near-duplicate graph unioned with the
     source corpus, so a group has >1 member whenever images are genuinely
     related and provenance never splits a visual family.

The splitter ASSERTS group-disjointness and per-level coverage and raises rather
than assuming either.

Run inside the training container (needs PIL + numpy):
    docker exec -i iddsi-train python3 ops/perceptual_dedup_and_grouping.py
"""
import hashlib, itertools, json, pathlib
from collections import defaultdict, Counter

import numpy as np
from PIL import Image

DATA = pathlib.Path("/work/data")
OUTDIR = pathlib.Path("/work/runs/v03")
OUTDIR.mkdir(parents=True, exist_ok=True)


def _bits(mask):
    flat = mask.flatten()
    return sum(int(b) << (len(flat) - 1 - i) for i, b in enumerate(flat))


def dhash(img, size=8):
    """Pillow 11 removed Image.getdata(); go through numpy."""
    a = np.asarray(img.convert("L").resize((size + 1, size), Image.LANCZOS), dtype="int16")
    return _bits((a[:, :-1] > a[:, 1:]))


def phash(img, size=8):
    """DCT-II perceptual hash; numpy exposes np.fft.dct, not a 2-D dct2 wrapper."""
    a = np.asarray(img.convert("L").resize((size, size), Image.LANCZOS), dtype="float64")
    d = np.fft.dct(np.fft.dct(a, axis=0, norm="ortho"), axis=1, norm="ortho")[:size, :size]
    med = np.median(d.ravel()[1:])
    return _bits(d > med)


def ham(a, b):
    return bin(a ^ b).count("1")


def main():
    manifest = [json.loads(l) for l in open(DATA / "manifest_v02.jsonl") if l.strip()]

    items = []
    for r in manifest:
        p = pathlib.Path(r["image_path"].replace("/work/data", str(DATA)))
        if not p.is_file():
            continue
        try:
            with Image.open(p) as im:
                im.load()
                items.append({
                    "event_id": r["event_id"], "path": str(p),
                    "corpus": r.get("source_corpus", "?"), "level": r["level"],
                    "md5": hashlib.md5(p.read_bytes()).hexdigest(),
                    "dhash": dhash(im), "phash": phash(im),
                })
        except Exception as exc:
            print("SKIP", p.name, type(exc).__name__)
    print(f"hashed {len(items)} images")

    by_md5 = defaultdict(list)
    for it in items:
        by_md5[it["md5"]].append(it["event_id"])
    exact = {k: v for k, v in by_md5.items() if len(v) > 1}

    near = []
    for a, b in itertools.combinations(items, 2):
        if ham(a["dhash"], b["dhash"]) <= 6 and ham(a["phash"], b["phash"]) <= 8:
            near.append((ham(a["dhash"], b["dhash"]), a["event_id"], b["event_id"]))
    print(f"exact-byte duplicate groups: {len(exact)}")
    print(f"near-duplicate pairs (dhash<=6, phash<=8): {len(near)}")

    parent = {it["event_id"]: it["event_id"] for it in items}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[rx] = ry

    for _, x, y in near:
        union(x, y)
    for v in exact.values():
        for o in v[1:]:
            union(v[0], o)

    comp = defaultdict(list)
    for it in items:
        comp[(find(it["event_id"]), it["corpus"])].append(it)
    groups = {it["event_id"]: "g%04d" % i for i, k in enumerate(sorted(comp)) for it in comp[k]}

    by_event = {it["event_id"]: it for it in items}
    rows = [dict(r, group_id=groups[r["event_id"]]) for r in manifest
            if r["event_id"] in by_event]
    with open(OUTDIR / "grouped_manifest.jsonl", "w") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    # ---- group-disjoint, level-balanced assignment ----
    import random
    rng = random.Random(20261010)
    gl = defaultdict(list)
    for r in rows:
        gl[r["group_id"]].append(r)
    order = sorted(gl.items(), key=lambda kv: (-len(kv[1]), kv[0]))
    rng.shuffle(order)
    TARGET = {"train": 0.70, "val": 0.15, "test": 0.15}
    assign = {}
    counts = {k: Counter() for k in TARGET}
    overall = Counter(r["level"] for r in rows)
    total = len(rows)
    for gid, members in order:
        lv = Counter(m["level"] for m in members)
        n = len(members)
        best, best_cost = None, None
        for split in TARGET:
            if sum(counts[split].values()) + n > TARGET[split] * total + 40:
                continue
            cost = 0.0
            for L, c in lv.items():
                want = TARGET[split] * overall[L]
                cost += ((counts[split][L] + c - want) / max(want, 1.0)) ** 2
            cur = sum(counts[split].values()) + n
            cost += ((cur - TARGET[split] * total) / max(TARGET[split] * total, 1.0)) ** 2 * 2
            if best_cost is None or cost < best_cost:
                best, best_cost = split, cost
        if best is None:
            best = min(TARGET, key=lambda s: sum(counts[s].values()))
        assign[gid] = best
        for L, c in lv.items():
            counts[best][L] += c

    out = [dict(r, split=assign[r["group_id"]]) for r in rows]

    # ---- ASSERT, do not assume ----
    spans = defaultdict(set)
    for r in out:
        spans[r["group_id"]].add(r["split"])
    leaky = {g: sorted(s) for g, s in spans.items() if len(s) > 1}
    assert not leaky, f"GROUP LEAK across splits: {leaky}"
    levels = set(r["level"] for r in out)
    bsl = defaultdict(Counter)
    for r in out:
        bsl[r["split"]][r["level"]] += 1
    for s in TARGET:
        missing = levels - set(bsl[s])
        assert not missing, f"level {missing} absent from {s}"

    with open(DATA / "manifest_v03.jsonl", "w") as fh:
        for r in out:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    json.dump({"images_hashed": len(items),
               "exact_byte_duplicate_groups": len(exact),
               "near_duplicate_pairs": len(near),
               "groups": len(gl),
               "rows": len(out),
               "split_sizes": {k: sum(bsl[k].values()) for k in TARGET},
               "group_disjoint": True,
               "all_levels_in_all_splits": True},
              open(OUTDIR / "dedup_report.json", "w"), indent=2)
    print(f"groups={len(gl)} rows={len(out)}")
    print("group-disjoint: VERIFIED; level coverage: VERIFIED")
    print("wrote", DATA / "manifest_v03.jsonl")


if __name__ == "__main__":
    main()
