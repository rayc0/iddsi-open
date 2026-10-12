#!/usr/bin/env python3
"""Verify the def04 derived artifacts. Run in the iddsi-train container:
  docker exec -w /work/lanes2/def04_manifest/trainer iddsi-train \
    python -I /work/lanes2/def04_manifest/tools/def04_verify.py
Asserts: derived manifest passes validate_manifest; D1 fixed (0 mismatches);
original field values preserved (only group_size changed, new cols appended);
D2 events deduplicated with the conflict annotated; D3 phash_reproduces count.
"""
import collections
import json
import sys

sys.path.insert(0, "/work/lanes2/def04_manifest")
from train.data import load_manifest, validate_manifest  # noqa: E402

DM = "/work/lane_runs/def04_manifest/manifest_v07_weighted.def04.jsonl"
SM = "/work/data/manifest_v07_weighted.jsonl"
EV = "/work/lane_runs/def04_manifest/realweak_pilot.events.def04.jsonl"
CONFLICT = "RW_nutrition5k_8e68a80fd452f62af95f"

der = load_manifest(DM)
shp = load_manifest(SM)

validate_manifest(der)
print("validate_manifest(derived): PASS")

cnt = collections.Counter(str(r["group_id"]) for r in der)
bad = [r["group_id"] for r in der if r["group_size"] != cnt[str(r["group_id"])]]
print("D1 derived group_size mismatches:", len(bad))
assert not bad

shp_by = {str(r["event_id"]): r for r in shp}
changed = collections.Counter()
added = collections.Counter()
for r in der:
    o = shp_by[str(r["event_id"])]
    for k, v in o.items():
        if r.get(k) != v:
            changed[k] += 1
    for k in r:
        if k not in o:
            added[k] += 1
print("original fields changed:", dict(changed))
print("new columns added:", dict(added))
assert set(changed) <= {"group_size"}, changed

assert {str(r["event_id"]) for r in der} == {str(r["event_id"]) for r in shp}
assert len(der) == len(shp) == 532
tr = sum(1 for r in der if r.get("phash_reproduces") is True)
print(f"D3 phash_reproduces True: {tr}/{len(der)}")
assert tr == 513, tr

ev = [json.loads(l) for l in open(EV, encoding="utf-8") if l.strip()]
ids = [str(r["event_id"]) for r in ev]
dups = [e for e, n in collections.Counter(ids).items() if n > 1]
print(f"D2 derived events: {len(ev)} rows, {len(set(ids))} distinct, {len(dups)} dups")
assert len(ev) == 117 and not dups
conf = [r for r in ev if str(r["event_id"]) == CONFLICT][0]
print("D2 conflict kept level_weak:", conf["level_weak"],
      "annotated:", "def04_source_event_conflict" in conf)
assert conf["level_weak"] == 6 and "def04_source_event_conflict" in conf

# the manifest row for the conflict event is annotated too
mconf = [r for r in der if str(r["event_id"]) == CONFLICT][0]
assert "def04_source_event_conflict" in mconf
print("D2 manifest conflict annotation present:", mconf["def04_source_event_conflict"]["values"])
print("ALL DERIVED-ARTIFACT CHECKS PASS")
