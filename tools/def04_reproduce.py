#!/usr/bin/env python3 -I
"""def04_manifest — reproduce the three manifest defects an04_groups recorded.

Reproduction ONLY: this script measures the three defects and writes a JSON +
text report. It never writes to, or mutates, any shipped file. Run with `-I`
(isolated) so no module is picked up from the data directories.

The three defects (hf/HAZARD_LOG.md H-GROUP measured status; HGROUP_VERIFICATION.md
section D; lane_runs/an04_groups/report*.json):

  D1  `group_size` is stale for 3 groups / 18 rows in the v0.7 manifest.
  D2  realweak_pilot/events.jsonl carries 12 duplicated event rows, one
      conflicting on `level_weak` (5 vs 6).
  D3  the recorded `phash` is not reproducible from the stored JPEGs
      (4 of 532 exact).

Usage:
  python3 -I tools/def04_reproduce.py \
      --manifest /work/data/manifest_v07_weighted.jsonl \
      --events /work/data/realweak_pilot/events.jsonl \
               /work/data/harvest_d1/events.jsonl \
               /work/data/harvest_d1b/events.jsonl \
      --outdir /work/lane_runs/def04_manifest
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def host_path(p: str) -> str:
    """Manifest image_path values are container paths (/work/...). Map to host."""
    if Path(p).exists():
        return p
    return p.replace("/work/", os.environ.get("IDDSI_HOST_DATA_ROOT", ".") + "/", 1)


# ---------------------------------------------------------------------------
# D1: group_size stale
# ---------------------------------------------------------------------------
def check_group_size(rows: list[dict]) -> dict:
    per_group: dict[str, list[dict]] = collections.defaultdict(list)
    for row in rows:
        per_group[str(row["group_id"])].append(row)

    mismatched = {}
    for g, grp in per_group.items():
        declared = grp[0].get("group_size")
        observed = len(grp)
        # every row in the group must carry the same declared value
        declared_set = {r.get("group_size") for r in grp}
        if declared != observed or len(declared_set) != 1:
            mismatched[g] = {
                "declared_group_size": declared,
                "declared_values_in_group": sorted(str(x) for x in declared_set),
                "observed_rows": observed,
                "event_ids": [r["event_id"] for r in grp],
                "splits": sorted({r["split"] for r in grp}),
            }
    affected_rows = sum(m["observed_rows"] for m in mismatched.values())
    hist = collections.Counter(len(g) for g in per_group.values())
    return {
        "n_rows": len(rows),
        "n_groups": len(per_group),
        "cardinality_histogram": {str(k): v for k, v in sorted(hist.items())},
        "n_singleton_groups": hist.get(1, 0),
        "n_multirow_groups": sum(v for k, v in hist.items() if k > 1),
        "n_mismatched_groups": len(mismatched),
        "n_affected_rows": affected_rows,
        "mismatched": dict(sorted(mismatched.items())),
    }


# ---------------------------------------------------------------------------
# D2: duplicated event rows in the corpus events files
# ---------------------------------------------------------------------------
def check_duplicate_events(events_paths: list[Path]) -> dict:
    out = {"by_file": {}, "total_duplicate_rows": 0, "conflicts": []}
    for path in events_paths:
        raw_lines = [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
        records = [json.loads(ln) for ln in raw_lines]
        by_id: dict[str, list[int]] = collections.defaultdict(list)
        for i, rec in enumerate(records):
            by_id[str(rec["event_id"])].append(i)

        dup_ids = {eid: idxs for eid, idxs in by_id.items() if len(idxs) > 1}
        byte_identical = 0
        conflicting = []
        for eid, idxs in dup_ids.items():
            # canonical byte form of each duplicate row
            canon = [json.dumps(records[i], sort_keys=True, ensure_ascii=False) for i in idxs]
            distinct = sorted(set(canon))
            if len(distinct) == 1:
                byte_identical += 1
            else:
                # find which fields differ
                fields = collections.defaultdict(set)
                for i in idxs:
                    for k, v in records[i].items():
                        fields[k].add(json.dumps(v, sort_keys=True, ensure_ascii=False))
                diff_fields = {k: sorted(v) for k, v in fields.items() if len(v) > 1}
                conflicting.append({
                    "event_id": eid,
                    "n_copies": len(idxs),
                    "line_numbers": [i + 1 for i in idxs],
                    "differing_fields": diff_fields,
                    "levels_weak": sorted({records[i].get("level_weak") for i in idxs}),
                })
        out["by_file"][str(path)] = {
            "n_lines": len(raw_lines),
            "n_distinct_event_ids": len(by_id),
            "n_duplicate_event_ids": len(dup_ids),
            "n_duplicate_rows": sum(len(idxs) - 1 for idxs in dup_ids.values()),
            "n_byte_identical_dup_ids": byte_identical,
            "n_conflicting_dup_ids": len(conflicting),
            "conflicting": conflicting,
        }
        out["total_duplicate_rows"] += sum(len(idxs) - 1 for idxs in dup_ids.values())
        out["conflicts"].extend(conflicting)
    return out


# ---------------------------------------------------------------------------
# D3: recorded phash vs recomputed-from-JPEG
# ---------------------------------------------------------------------------
def check_phash(rows: list[dict], events_paths: list[Path]) -> dict:
    from PIL import Image
    import numpy as np

    def dhash(path, size=8):
        g = Image.open(path).convert("L").resize((size + 1, size), Image.LANCZOS)
        a = np.asarray(g, dtype="int16")
        return "".join("1" if b else "0" for b in (a[:, :-1] > a[:, 1:]).flatten())

    def phash(path, size=8):
        a = np.asarray(Image.open(path).convert("L").resize((size, size), Image.LANCZOS),
                       dtype="float64")
        n = np.arange(size)
        k = n.reshape(-1, 1)
        C = np.cos(np.pi * (2 * n + 1) * k / (2 * size)) * np.sqrt(2.0 / size)
        C[0] /= np.sqrt(2.0)
        d = C @ a @ C.T
        med = np.median(d.ravel()[1:])
        return "".join("1" if b else "0" for b in (d > med).flatten())

    def ham(a, b):
        return sum(x != y for x, y in zip(a, b))

    # recorded phash per event_id, from whichever events file carries it
    recorded: dict[str, str] = {}
    for path in events_paths:
        for rec in load_jsonl(path):
            eid = str(rec["event_id"])
            ph = rec.get("phash")
            if ph is not None and eid not in recorded:
                recorded[eid] = str(ph)

    exact = 0
    diffs = []
    missing_recorded = 0
    missing_image = 0
    recomputed_hex: dict[str, str] = {}
    for i, row in enumerate(rows):
        eid = str(row["event_id"])
        p = host_path(str(row["image_path"]))
        if not Path(p).exists():
            missing_image += 1
            continue
        dh = dhash(p)
        ph = phash(p)
        recomputed_hex[eid] = f"{int(ph, 2):016x}"
        rec = recorded.get(eid)
        if rec is None:
            missing_recorded += 1
            continue
        # recorded is a 16-hex string; convert to 64-bit binary for compare
        try:
            rec_bits = f"{int(rec, 16):064b}"
        except ValueError:
            rec_bits = None
        if rec_bits is not None and rec_bits == ph:
            exact += 1
        elif rec_bits is not None:
            diffs.append(ham(rec_bits, ph))
        if (i + 1) % 100 == 0:
            print(f"    hashed {i+1}/{len(rows)}")

    import statistics
    return {
        "n_rows": len(rows),
        "n_images_found": len(rows) - missing_image,
        "n_missing_image": missing_image,
        "n_missing_recorded_phash": missing_recorded,
        "n_compared": exact + len(diffs),
        "n_exact_match": exact,
        "mean_hamming_when_diff": (statistics.mean(diffs) if diffs else None),
        "mean_hamming_all_compared": (statistics.mean(
            [0] * exact + diffs) if (exact + len(diffs)) else None),
        "min_hamming_when_diff": (min(diffs) if diffs else None),
        "max_hamming_when_diff": (max(diffs) if diffs else None),
        "recomputed_phash_hex": recomputed_hex,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--events", nargs="+", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--skip-phash", action="store_true",
                    help="skip the (slower) image-hashing reproduction of D3")
    args = ap.parse_args()

    manifest = Path(args.manifest)
    events = [Path(e) for e in args.events]
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    rows = load_jsonl(manifest)
    print(f"manifest: {manifest}  rows={len(rows)}  "
          f"md5={hashlib.md5(manifest.read_bytes()).hexdigest()}")

    report = {
        "manifest": str(manifest),
        "manifest_md5": hashlib.md5(manifest.read_bytes()).hexdigest(),
        "manifest_bytes": manifest.stat().st_size,
        "n_rows": len(rows),
    }

    print("D1 group_size ...")
    report["D1_group_size_stale"] = check_group_size(rows)
    d1 = report["D1_group_size_stale"]
    print(f"    groups={d1['n_groups']} mismatched={d1['n_mismatched_groups']} "
          f"affected_rows={d1['n_affected_rows']}")

    print("D2 duplicate event rows ...")
    report["D2_duplicate_events"] = check_duplicate_events(events)
    d2 = report["D2_duplicate_events"]
    print(f"    total_duplicate_rows={d2['total_duplicate_rows']} "
          f"conflicts={len(d2['conflicts'])}")

    if not args.skip_phash:
        print("D3 recorded phash vs recomputed (image hashing) ...")
        rep3 = check_phash(rows, events)
        # keep the full hex map out of the printed summary but in the file
        report["D3_phash_irreproducible"] = {
            k: v for k, v in rep3.items() if k != "recomputed_phash_hex"}
        report["D3_phash_irreproducible"]["n_recomputed"] = len(rep3["recomputed_phash_hex"])
        d3 = report["D3_phash_irreproducible"]
        print(f"    compared={d3['n_compared']} exact={d3['n_exact_match']} "
              f"mean_hamming_all={d3['mean_hamming_all_compared']}")

    (outdir / "reproduce_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(f"wrote {outdir/'reproduce_report.json'}")


if __name__ == "__main__":
    main()
