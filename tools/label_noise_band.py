#!/usr/bin/env python3
"""Label-noise band on the v0.7 headline exact-match / MAE.

The v0.7 test labels are machine consensus, not clinician-verified. For the 22
of 108 test events whose two independent labellers disagreed (agreement "1/2"),
the manifest records BOTH votes in `votes`. The consensus row keeps one of them
and the manifest does not record which rule chose it.

This script re-scores the five v0.7 seeds with the split-event labels flipped to
the OTHER labeller's vote - an equally defensible resolution of the same
evidence - and reports how far exact match and MAE move. That spread is the part
of our headline number that is a property of the yardstick rather than of the
model, and it is the quantity the EdUHK anchor (clinician-verified labels) does
not carry.

Output: <root>/lane_runs/an05_external/label_noise_band.json   (CPU-only)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

SEEDS = [17, 29, 43, 71, 101]
LEVELS = np.array([3, 4, 5, 6, 7], dtype=np.int64)

# The workspace root comes from the environment (or the current directory), so no
# host path is baked into the shipped tool.
_candidates = [Path(os.environ["IDDSI_ROOT"])] if os.environ.get("IDDSI_ROOT") else []
_candidates += [Path.cwd(), Path.cwd().parent]
ROOT = next((p for p in _candidates if (p / "data/manifest_v07_weighted.jsonl").exists()), None)
if ROOT is None:
    raise SystemExit("cannot locate the v0.7 manifest — set IDDSI_ROOT")
MANIFEST = ROOT / "data/manifest_v07_weighted.jsonl"
RUNS = ROOT / "runs/v07"
OUTDIR = ROOT / "lane_runs/an05_external"
OUTDIR.mkdir(parents=True, exist_ok=True)
OUT = OUTDIR / "label_noise_band.json"


def coral(probs: np.ndarray) -> np.ndarray:
    cum = np.cumsum(probs, axis=1)[:, :-1]
    return LEVELS[(1.0 - cum > 0.5).sum(axis=1)]


def main() -> None:
    rows = [json.loads(line) for line in MANIFEST.read_text().splitlines() if line.strip()]
    test = {r["event_id"]: r for r in rows if r["split"] == "test"}

    # alternate label for every event whose two labellers disagreed
    alt = {}
    for eid, r in test.items():
        if r.get("agreement") == "1/2":
            votes = r["votes"]
            others = [int(v) for k, v in votes.items() if int(v) != int(r["level"])]
            if others:
                alt[eid] = others[0]
    n_split = len(alt)

    per_seed = []
    for seed in SEEDS:
        run = RUNS / f"siglip2_v07_fullcoverage_seed{seed}"
        npz = np.load(run / "test_predictions.npz", allow_pickle=False)
        structured = {s["event_id"]: s for s in json.loads((run / "test_structured_outputs.json").read_text())}
        ids = [str(e) for e in npz["event_ids"]]
        base = npz["labels"].astype(np.int64) + 3
        flipped = np.array([alt.get(e, int(b)) for e, b in zip(ids, base)], dtype=np.int64)
        probs = npz["probabilities"].astype(float)
        pred = coral(probs)
        conf = np.array([structured[e]["confidence"] for e in ids])
        thr = np.array([structured[e]["calibrated_threshold"] for e in ids])[0]
        cov = conf >= thr

        def score(truth):
            return {
                "exact_match_pct_covered": 100.0 * float((pred[cov] == truth[cov]).mean()),
                "mae_covered_levels": float(np.abs(pred[cov] - truth[cov]).mean()),
            }

        a, b = score(base), score(flipped)
        per_seed.append(
            {
                "seed": seed,
                "n_covered": int(cov.sum()),
                "as_published_labels": a,
                "split_events_flipped": b,
                "delta_exact_match_pp": b["exact_match_pct_covered"] - a["exact_match_pct_covered"],
                "delta_mae_levels": b["mae_covered_levels"] - a["mae_covered_levels"],
            }
        )

    def rng(key, sub):
        v = [s[sub][key] for s in per_seed]
        return [min(v), max(v)]

    band = {
        "n_test_events": len(test),
        "n_split_test_events_relabelled": n_split,
        "n_split_test_events_pct": 100.0 * n_split / len(test),
        "exact_match_pct_covered_as_published": rng("exact_match_pct_covered", "as_published_labels"),
        "exact_match_pct_covered_flipped": rng("exact_match_pct_covered", "split_events_flipped"),
        "mae_covered_levels_as_published": rng("mae_covered_levels", "as_published_labels"),
        "mae_covered_levels_flipped": rng("mae_covered_levels", "split_events_flipped"),
        "per_seed": per_seed,
        "caveat": (
            "The manifest records both votes for split events but not the rule that "
            "resolved them; the flipped column is one equally defensible resolution, "
            "not a bound. It is reported to show how much of the headline moves with "
            "the yardstick alone."
        ),
    }
    OUT.write_text(json.dumps(band, indent=2) + "\n")
    print(json.dumps({k: v for k, v in band.items() if k != "per_seed"}, indent=2))


if __name__ == "__main__":
    main()
