#!/usr/bin/env python3
"""Re-measure the cross-labeller agreement figures the release card quotes, from
the stored per-event votes in
  /work/lanes/rl04_modelcard/results/v04_groupdisjoint_consensus_manifest.jsonl

The card quotes 30.9% / 47.1% / 74.7% exact and 68.8% / 73.2% within one level
(HAZARD_LOG.md H-QWENBIAS). Those numbers had no machine-readable source, so they
are recomputed here from the vote records themselves.

Also re-measures the two-judgement agreement inside the v0.7 manifest.

Writes /work/lane_runs/rl04_modelcard/labeller_agreement_measured.json.
"""
from __future__ import annotations

import itertools
import json
import pathlib
from collections import Counter

OUT = pathlib.Path("/work/lane_runs/rl04_modelcard/labeller_agreement_measured.json")
VOTES = "/work/lanes/rl04_modelcard/results/v04_groupdisjoint_consensus_manifest.jsonl"
V07 = "/work/data/manifest_v07_weighted.jsonl"


def pairwise(rows, a, b):
    pairs = [(r["votes"][a], r["votes"][b]) for r in rows if a in r.get("votes", {}) and b in r.get("votes", {})]
    if not pairs:
        return None
    exact = sum(1 for x, y in pairs if x == y) / len(pairs)
    within1 = sum(1 for x, y in pairs if abs(x - y) <= 1) / len(pairs)
    return {"n_pairs": len(pairs), "exact": exact, "within_1_level": within1}


def main() -> None:
    rows = [json.loads(l) for l in open(VOTES) if l.strip()]
    labellers = sorted({k for r in rows for k in r.get("votes", {})})
    report = {
        "source_file": VOTES,
        "events_with_votes": sum(1 for r in rows if r.get("votes")),
        "labeller_sets": {str(k): v for k, v in Counter(
            tuple(sorted(r["labellers"])) for r in rows if r.get("labellers")).items()},
        "pairwise": {},
    }
    for a, b in itertools.combinations(labellers, 2):
        report["pairwise"][f"{a}_vs_{b}"] = pairwise(rows, a, b)

    # ---- agreement inside the v0.7 manifest, for the events that carry two
    #      genuine independent judgements ----
    v07 = [json.loads(l) for l in open(V07) if l.strip()]
    two = [r for r in v07 if r["n_labellers"] >= 2]
    report["v07_events_with_two_judgements"] = len(two)
    report["v07_unanimous"] = sum(1 for r in two if r["agreement"] == "2/2")
    report["v07_split"] = sum(1 for r in two if r["agreement"] == "1/2")
    report["v07_unanimous_fraction"] = report["v07_unanimous"] / len(two)

    OUT.write_text(json.dumps(report, indent=1))
    print(f"wrote {OUT}\n")
    print("pairwise agreement over the stored votes (levels 3-7):")
    for k, v in report["pairwise"].items():
        if v:
            print(f"  {k:<18} n={v['n_pairs']:<4} exact {v['exact']:.3f}   within +/-1 {v['within_1_level']:.3f}")
    print()
    print(f"v0.7 manifest: {report['v07_events_with_two_judgements']} events with two judgements, "
          f"{report['v07_unanimous']} unanimous ({report['v07_unanimous_fraction']:.4f}), "
          f"{report['v07_split']} split")


if __name__ == "__main__":
    main()
