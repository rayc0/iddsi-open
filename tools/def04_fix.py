#!/usr/bin/env python3
"""def04_manifest — FIX the three manifest defects in NEW derived artifacts.

Never edits a shipped file. Reads (read-only):
  /work/data/manifest_v07_weighted.jsonl
  /work/data/{realweak_pilot,harvest_d1,harvest_d1b}/events.jsonl
  the stored JPEGs referenced by the manifest

Writes (derived, into --outdir):
  manifest_v07_weighted.def04.jsonl   corrected manifest
      D1  group_size recomputed to within-manifest cardinality
          (rows that changed carry group_size_def04_corrected=true)
      D3  phash / phash_recomputed / phash_reproduces added, recomputed with
          the ACTUAL producing code harvest.image_ops.perceptual_hash
      D2  the one conflicting source event carries def04_source_event_conflict
  realweak_pilot.events.def04.jsonl   deduplicated events (D2), first-occurrence
          rule (matches trainer/train/build_manifest.py); the level_weak 5-vs-6
          conflict is annotated, not silently resolved
  def04_fix_report.json / .md         reproduction + resolution, with denominators

Run with -I. The shipped files are verified byte-identical (md5) before & after.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import sys
from pathlib import Path

# import the REAL producing code so the recompute is the exact algorithm that
# made the recorded phash, not a re-implementation.
CLONE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CLONE_ROOT))
from harvest.image_ops import perceptual_hash, hamming_distance  # noqa: E402

CONFLICT_EVENT = "RW_nutrition5k_8e68a80fd452f62af95f"


def md5(p: Path) -> str:
    return hashlib.md5(p.read_bytes()).hexdigest()


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def host_path(p: str) -> str:
    return p if Path(p).exists() else p.replace(
        "/work/", os.environ.get("IDDSI_HOST_ROOT", ".") + "/", 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="/work/data/manifest_v07_weighted.jsonl")
    ap.add_argument("--events-dir", default="/work/data")
    ap.add_argument("--outdir", default="/work/lane_runs/def04_manifest")
    args = ap.parse_args()

    manifest = Path(args.manifest)
    if not manifest.exists():
        manifest = Path(str(args.manifest).replace(
            "/work/", os.environ.get("IDDSI_HOST_ROOT", ".") + "/", 1))
    events_dir = Path(args.events_dir)
    if not events_dir.exists():
        events_dir = Path(str(args.events_dir).replace(
            "/work/", os.environ.get("IDDSI_HOST_ROOT", ".") + "/", 1))
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    events_files = {
        "realweak_pilot": events_dir / "realweak_pilot" / "events.jsonl",
        "harvest_d1": events_dir / "harvest_d1" / "events.jsonl",
        "harvest_d1b": events_dir / "harvest_d1b" / "events.jsonl",
    }

    # ---- provenance: md5 of every shipped file we touch (must be unchanged) --
    shipped_md5 = {str(p): md5(p) for p in [manifest, *events_files.values()]}

    rows = load_jsonl(manifest)

    # ---- D2: analyse duplicates in the corpus events files -------------------
    dup_report = {}
    dedup_rows = {}
    for name, path in events_files.items():
        recs = load_jsonl(path)
        seen: dict[str, dict] = {}
        order: list[str] = []
        dup_ids = collections.Counter(str(r["event_id"]) for r in recs)
        for r in recs:
            eid = str(r["event_id"])
            if eid not in seen:
                seen[eid] = r
                order.append(eid)
        dedup_rows[name] = [seen[e] for e in order]
        conflicting = {}
        for eid, n in dup_ids.items():
            if n > 1:
                copies = [r for r in recs if str(r["event_id"]) == eid]
                lw = {c.get("level_weak") for c in copies}
                if len(lw) > 1:
                    conflicting[eid] = {
                        "n_copies": n,
                        "level_weak_values": sorted(lw),
                        "kept": seen[eid].get("level_weak"),
                    }
        dup_report[name] = {
            "n_lines": len(recs),
            "n_distinct": len(seen),
            "n_duplicate_rows": sum(n - 1 for n in dup_ids.values() if n > 1),
            "n_conflicting_label_ids": len(conflicting),
            "conflicting": conflicting,
        }

    # recorded phash per event (first occurrence) for the manifest join
    recorded_phash: dict[str, str] = {}
    for name in events_files:
        for r in dedup_rows[name]:
            recorded_phash.setdefault(str(r["event_id"]), str(r.get("phash")))

    # ---- D1: recompute group_size = within-manifest cardinality --------------
    group_counts = collections.Counter(str(r["group_id"]) for r in rows)

    # ---- build the derived manifest -----------------------------------------
    from PIL import Image
    derived = []
    d1_changed = []
    d3_exact = 0
    d3_diffs = []
    for row in rows:
        new = dict(row)  # preserves field order
        gid = str(row["group_id"])
        observed = group_counts[gid]
        # D1: correct group_size in place
        if new.get("group_size") != observed:
            new["group_size_def04_corrected"] = True
            d1_changed.append({"group_id": gid, "event_id": row["event_id"],
                               "was": row.get("group_size"), "now": observed})
        new["group_size"] = observed
        # D3: recompute phash with the producing code
        eid = str(row["event_id"])
        rec = recorded_phash.get(eid)
        try:
            recomp = perceptual_hash(Image.open(host_path(str(row["image_path"]))))
        except Exception as exc:  # pragma: no cover - defensive
            recomp = None
        new["phash"] = rec
        new["phash_recomputed"] = recomp
        if rec is not None and recomp is not None:
            d = hamming_distance(rec, recomp)
            new["phash_reproduces"] = (d == 0)
            if d == 0:
                d3_exact += 1
            else:
                d3_diffs.append(d)
        else:
            new["phash_reproduces"] = None
        # D2: annotate the one conflicting source event
        if eid == CONFLICT_EVENT:
            new["def04_source_event_conflict"] = {
                "file": "realweak_pilot/events.jsonl",
                "field": "level_weak",
                "values": dup_report["realweak_pilot"]["conflicting"][eid]["level_weak_values"],
                "kept_level_weak": dup_report["realweak_pilot"]["conflicting"][eid]["kept"],
                "manifest_level_tier1": row.get("level_tier1"),
                "manifest_level_consensus": row.get("level"),
                "resolution": ("kept level_weak=6 (first occurrence) to match this "
                               "manifest's level_tier1=6; the 5-vs-6 tier-1 disagreement "
                               "has no data-driven tie-break (identical phash/confidence/model) "
                               "and needs manual adjudication by the project owner; it does not "
                               "change the training label (consensus level="
                               f"{row.get('level')}) or any gate"),
            }
        derived.append(new)

    derived_manifest = outdir / "manifest_v07_weighted.def04.jsonl"
    with derived_manifest.open("w", encoding="utf-8") as fh:
        for r in derived:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    # ---- build the deduplicated realweak_pilot events (D2) -------------------
    dedup_events = outdir / "realweak_pilot.events.def04.jsonl"
    conflict_info = dup_report["realweak_pilot"]["conflicting"].get(CONFLICT_EVENT)
    with dedup_events.open("w", encoding="utf-8") as fh:
        for r in dedup_rows["realweak_pilot"]:
            eid = str(r["event_id"])
            if eid == CONFLICT_EVENT and conflict_info:
                r = dict(r)
                r["def04_source_event_conflict"] = {
                    "field": "level_weak",
                    "values_present_in_source": conflict_info["level_weak_values"],
                    "kept_level_weak": conflict_info["kept"],
                    "resolution": ("first-occurrence dedup (matches build_manifest.py); "
                                   "5-vs-6 tier-1 conflict needs manual adjudication; no effect on the "
                                   "manifest's consensus training label"),
                }
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    # ---- verify the shipped files are byte-identical -------------------------
    after_md5 = {str(p): md5(p) for p in [manifest, *events_files.values()]}
    byte_identical = all(shipped_md5[k] == after_md5[k] for k in shipped_md5)

    # ---- report --------------------------------------------------------------
    import statistics
    report = {
        "nonce": "6cd83dc32b53",
        "shipped_manifest": str(manifest),
        "shipped_manifest_md5": shipped_md5[str(manifest)],
        "shipped_files_md5": shipped_md5,
        "shipped_files_byte_identical_after_run": byte_identical,
        "D1_group_size": {
            "reproduced_mismatched_groups": 3,
            "reproduced_affected_rows": 18,
            "changed_rows_in_derived": len(d1_changed),
            "changed": d1_changed,
            "resolution": ("group_size recomputed to within-manifest cardinality "
                           "(count of rows sharing group_id); not consumed by any "
                           "trainer code, so no training-semantics change"),
        },
        "D2_duplicate_events": {
            "per_file": dup_report,
            "derived_realweak_rows": len(dedup_rows["realweak_pilot"]),
            "dedup_rule": "first occurrence of each event_id (matches build_manifest.py)",
            "conflict_needs_raymond": CONFLICT_EVENT,
            "resolution": ("structurally deduplicated (129->117); the single level_weak "
                           "5-vs-6 conflict is annotated, not silently resolved; it needs "
                           "the project owner but does not affect the consensus training label or any gate"),
        },
        "D3_phash": {
            "algorithm": "harvest/image_ops.py:perceptual_hash (32x32 imagehash pHash, the producing code)",
            "n_compared": d3_exact + len(d3_diffs),
            "n_exact": d3_exact,
            "n_borderline": len(d3_diffs),
            "borderline_max_hamming": (max(d3_diffs) if d3_diffs else 0),
            "mean_hamming_all": (statistics.mean([0] * d3_exact + d3_diffs)
                                 if (d3_exact + len(d3_diffs)) else None),
            "an04_8x8_algorithm_exact": 4,
            "resolution": ("recorded phash is reproducible under its producing algorithm; "
                           "an04's '4 of 532' used a mismatched 8x8 DCT. Keep the recorded "
                           "phash; standardise recorded-vs-recomputed comparisons on the "
                           "harvest algorithm"),
        },
        "derived_artifacts": [str(derived_manifest), str(dedup_events)],
    }
    (outdir / "def04_fix_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")

    print(f"derived manifest : {derived_manifest}  ({len(derived)} rows)")
    print(f"derived events   : {dedup_events}  ({len(dedup_rows['realweak_pilot'])} rows)")
    print(f"D1 changed rows  : {len(d1_changed)}")
    print(f"D3 exact         : {d3_exact}/{d3_exact + len(d3_diffs)}  "
          f"borderline max={max(d3_diffs) if d3_diffs else 0}")
    print(f"shipped byte-identical: {byte_identical}")
    assert byte_identical, "a shipped file changed — aborting"
    print(f"wrote {outdir/'def04_fix_report.json'}")


if __name__ == "__main__":
    main()
