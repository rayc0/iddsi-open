#!/usr/bin/env python3
"""denominator_audit.py -- recompute every corpus-hygiene percentage from source.

Hazard H-UNUSABLE (`hf/HAZARD_LOG.md`) and Finding 2 of the model card
(`hf/model_README.md`) both quote a single count -- 152 unusable events -- over
two different denominators, and one of them did the arithmetic wrong. This script
recomputes every figure in that family from the files that produced it, so each
document can state its own denominator instead of restating a percentage.

Inputs (read-only, all inside this repo or the shared /work/data mount):
  results/v05_release/labelability_sweep.json   aggregate sweep counts
  results/v05_release/unusable_event_ids.txt    the 152 excluded event ids
  data/manifest_v0{2,3,4}_*.jsonl               pre-clean corpus (684 events)
  data/manifest_v0{5_clean,7_weighted}.jsonl    post-clean corpus (532 events)

Nothing is written. Usage:
  python3 tools/denominator_audit.py [--repo PATH] [--data PATH]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

PRE_CLEAN = ["manifest_v02.jsonl", "manifest_v03.jsonl", "manifest_v04_consensus.jsonl"]
POST_CLEAN = ["manifest_v05_clean.jsonl", "manifest_v07_weighted.jsonl"]


def data_candidates(repo: Path, override: str | None) -> list[Path]:
    """Where the shared manifests live: the container mount, or the host checkout."""
    if override:
        return [Path(override)]
    return [
        Path("/work/data"),           # inside the iddsi-train container
        repo.parents[1] / "data",     # host: ~/iddsi/data next to lanes/
        repo / "data",
    ]


def resolve_data(repo: Path, override: str | None) -> Path:
    for cand in data_candidates(repo, override):
        if (cand / "manifest_v04_consensus.jsonl").exists():
            return cand
    tried = ", ".join(str(c) for c in data_candidates(repo, override))
    raise SystemExit(f"no manifest directory found (tried: {tried}); pass --data PATH")


def read_ids(path: Path) -> list[str]:
    with path.open() as fh:
        return [ln.strip() for ln in fh if ln.strip()]


def read_manifest_ids(path: Path) -> list[str]:
    ids = []
    with path.open() as fh:
        for ln in fh:
            ln = ln.strip()
            if ln:
                ids.append(json.loads(ln)["event_id"])
    return ids


def pct(n: int, d: int) -> float:
    return 100.0 * n / d if d else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=str(Path(__file__).resolve().parent.parent))
    ap.add_argument("--data", default=None,
                    help="directory holding the manifests; default: auto-detect")
    args = ap.parse_args()
    repo = Path(args.repo).resolve()
    data = resolve_data(repo, args.data)

    sweep = json.loads((repo / "results/v05_release/labelability_sweep.json").read_text())
    unusable = read_ids(repo / "results/v05_release/unusable_event_ids.txt")
    n_unusable = len(set(unusable))
    print(f"# manifests read from {data}")

    print("== sweep aggregate (results/v05_release/labelability_sweep.json) ==")
    for key in sorted(sweep):
        print(f"  {key:18s} {sweep[key]}")
    both = sweep["both"]
    print()
    print("== denominators ==")
    print(f"  unusable ids (unique)          {n_unusable}   "
          f"[{len(unusable)} lines, {len(unusable) - n_unusable} duplicates]")

    pre: dict[str, list[str]] = {}
    for name in PRE_CLEAN:
        pre[name] = read_manifest_ids(data / name)
        ids = pre[name]
        print(f"  pre-clean  {name:32s} {len(ids)} rows, {len(set(ids))} unique")
    corpus = max(len(set(v)) for v in pre.values())
    print(f"  pre-clean corpus (largest)     {corpus}")

    post_rows = {}
    for name in POST_CLEAN:
        ids = read_manifest_ids(data / name)
        post_rows[name] = (len(ids), len(set(ids)))
        print(f"  post-clean {name:32s} {len(ids)} rows, {len(set(ids))} unique")

    # Is the 152 really drawn from the 684, and really removed from the 532?
    for name, ids in pre.items():
        hit = len(set(unusable) & set(ids))
        print(f"  unusable ids found in {name:28s} {hit}/{n_unusable}")
    for name, (n_rows, _n_uniq) in post_rows.items():
        hit = len(set(unusable) & set(read_manifest_ids(data / name)))
        print(f"  unusable ids remaining in {name:22s} {hit}/{n_rows}")

    print()
    print("== rates, each over the denominator it belongs to ==")
    print(f"  unusable / whole corpus        {n_unusable}/{corpus} = {pct(n_unusable, corpus):.1f}%")
    print(f"  unusable / both-judged subset  {n_unusable}/{both} = {pct(n_unusable, both):.1f}%")
    print(f"  IS-THERE-FOOD agreement        {sweep['cross_food_agree']}/{both} "
          f"= {pct(sweep['cross_food_agree'], both):.1f}%")
    print(f"  LEVELLABLE agreement           {sweep['cross_lev_agree']}/{both} "
          f"= {pct(sweep['cross_lev_agree'], both):.1f}%")
    print(f"  corpus after cleaning          {corpus} - {n_unusable} = {corpus - n_unusable}")

    # Consistency check the aggregate cannot pass on its own: the two per-family
    # counts and the "both" count must be able to come from one set of 684.
    union = sweep["step_events"] + sweep["mm_events"] - both
    print()
    print("== internal consistency of the sweep aggregate ==")
    print(f"  step {sweep['step_events']} + mm {sweep['mm_events']} - both {both} "
          f"=> union {union}, but the corpus has only {corpus} events")
    print(f"  => 'both = {both}' is NOT the plain intersection of two subsets of {corpus}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
