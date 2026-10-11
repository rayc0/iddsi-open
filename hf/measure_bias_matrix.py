#!/usr/bin/env python3
"""Re-measure the model-by-yardstick bias matrix, so the card's only
non-circular claim is a measured one.

The previously published claim ("models trained on three DIFFERENT label sets
all agree more with the v0.7 labels (0.70-0.86) than with the v0.5 ones
(0.48-0.57)", hf/HAZARD_LOG.md H-YARDSTICK) is checked here against
results/v07_release/bias_matrix.json, which does not support it for v0.3.

Every model version's stored test predictions are scored against both
consensus label sets on the events the two label sets share.

Writes /work/lane_runs/rl04_modelcard/bias_matrix_measured.json.
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
from sklearn.metrics import cohen_kappa_score

sys.path.insert(0, "/work/lanes/rl04_modelcard/trainer")
from train.model import coral_decode  # noqa: E402

OUT = pathlib.Path("/work/lane_runs/rl04_modelcard/bias_matrix_measured.json")
RUNS = {
    "v0.3 (trained: Qwen tier-2 labels)": "/work/runs/v03",
    "v0.4 (trained: 4-way consensus, partial coverage)": "/work/runs/v04",
    "v0.5 (trained: consensus, cleaned corpus)": "/work/runs/v05",
    "v0.7 (trained: consensus, full labeller coverage)": "/work/runs/v07",
}
TRUTHS = {
    "v0.5 consensus": "/work/data/manifest_v05_clean.jsonl",
    "v0.7 consensus": "/work/data/manifest_v07_weighted.jsonl",
}


def load_truth(path):
    rows = [json.loads(l) for l in open(path) if l.strip()]
    return {r["event_id"]: int(r["level"]) - 3 for r in rows if r.get("split") == "test"}


def score(run_dir, truth):
    npz = np.load(run_dir / "test_predictions.npz", allow_pickle=True)
    probs = npz["probabilities"].astype(np.float64)
    probs = probs / probs.sum(axis=1, keepdims=True)
    raw, _ = coral_decode(probs)
    raw = np.asarray(raw).astype(int)
    eids = [str(x) for x in npz["event_ids"]]
    keep = [i for i, e in enumerate(eids) if e in truth]
    if len(keep) < 20:
        return None
    p = np.array([raw[i] for i in keep])
    t = np.array([truth[eids[i]] for i in keep])
    return {
        "n": len(keep),
        "kappa_all_quadratic": float(cohen_kappa_score(t, p, weights="quadratic")),
        "exact": float((p == t).mean()),
        "mae": float(np.abs(p - t).mean()),
    }


def main() -> None:
    truths = {k: load_truth(v) for k, v in TRUTHS.items()}
    res = {}
    for model, run in RUNS.items():
        res[model] = {}
        for tname, truth in truths.items():
            per_seed = {}
            for d in sorted(pathlib.Path(run).glob("*_seed*")):
                if not d.is_dir():
                    continue
                s = score(d, truth)
                if s:
                    per_seed[d.name.split("_seed")[-1]] = s
            if per_seed:
                ks = [v["kappa_all_quadratic"] for v in per_seed.values()]
                res[model][tname] = {
                    "n_seeds": len(per_seed),
                    "n_events": sorted({v["n"] for v in per_seed.values()}),
                    "kappa_mean": float(np.mean(ks)),
                    "kappa_min": min(ks),
                    "kappa_max": max(ks),
                    "per_seed": per_seed,
                }

    # the claim under test: does every model score better on the v0.7 labels?
    better = {
        m: (res[m]["v0.7 consensus"]["kappa_mean"] > res[m]["v0.5 consensus"]["kappa_mean"])
        for m in res
        if "v0.7 consensus" in res[m] and "v0.5 consensus" in res[m]
    }
    published = json.load(open("/work/lanes/rl04_modelcard/results/v07_release/bias_matrix.json"))

    OUT.write_text(json.dumps(
        {"measured": res, "v07_better_than_v05": better, "published_bias_matrix": published},
        indent=1))
    print(f"wrote {OUT}\n")
    print(f"{'model':<52}{'vs v0.5 labels':<26}{'vs v0.7 labels':<26}")
    print("-" * 104)
    for m, d in res.items():
        a = d.get("v0.5 consensus")
        b = d.get("v0.7 consensus")
        fa = f"{a['kappa_mean']:.3f} (n={a['n_events'][0]})" if a else "n/a"
        fb = f"{b['kappa_mean']:.3f} (n={b['n_events'][0]})" if b else "n/a"
        print(f"{m:<52}{fa:<26}{fb:<26}")
    print("\ndoes each model score better on the v0.7 labels?")
    for m, v in better.items():
        print(f"  {m:<52} {'YES' if v else 'NO'}")


if __name__ == "__main__":
    main()
