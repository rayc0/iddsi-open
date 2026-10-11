#!/usr/bin/env python
"""Deeper structural checks for lane an02_errors.

1. Which predicted levels the shipped CORAL rank decode can actually emit
   (does the model ever decide L5 or L6?), across all 5 seeds.
2. Where the abstained events would have landed under argmax (i.e. what the
   model was torn about) -- abstention is a non-decision, not a correct answer.
3. The harvest "L6 candidate" bucket is not the same thing as a consensus L6
   label; quantify that disagreement.
4. Bootstrap CI on the headline error rates (denominators stated).
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.metrics import confusion_matrix

import sys
sys.path.insert(0, "/work/lanes/an02_errors/trainer")
from train.model import coral_decode  # noqa: E402

ART = Path("/work/lane_runs/an02_errors/artifacts")
OUT = Path("/work/lane_runs/an02_errors/analysis")
SEEDS = [17, 29, 43, 71, 101]
LEVELS = [3, 4, 5, 6, 7]
BEST = 43


def load(seed):
    with np.load(ART / f"v07_seed{seed}_test_predictions.npz", allow_pickle=False) as a:
        probs = a["probabilities"].astype(np.float64)
        labels = a["labels"].astype(np.int64)
        ids = [str(e) for e in a["event_ids"]]
    th = json.loads((ART / f"v07_seed{seed}_calibration.json").read_text())["threshold"]
    probs = probs / probs.sum(axis=1, keepdims=True)
    pred, conf = coral_decode(probs)
    return probs, labels, ids, th, pred, conf


def main() -> None:
    out = {}

    # ---- 1. emitted-level census -----------------------------------------
    census = {}
    for seed in SEEDS:
        probs, labels, ids, th, pred, conf = load(seed)
        decided = conf >= th
        emitted = Counter(int(p) for p in pred[decided])
        argmax_all = Counter(int(a) for a in probs.argmax(axis=1))
        census[str(seed)] = {
            "decided_n": int(decided.sum()),
            "emitted_level_counts": {f"L{LEVELS[k]}": v for k, v in sorted(emitted.items())},
            "levels_never_emitted": [f"L{LEVELS[k]}" for k in range(5) if k not in emitted],
            "argmax_counts_all_108": {f"L{LEVELS[k]}": v for k, v in sorted(argmax_all.items())},
            "argmax_levels_never_top": [f"L{LEVELS[k]}" for k in range(5) if k not in argmax_all],
        }
    out["emitted_level_census"] = census

    # ---- 2. where abstained events would have landed ---------------------
    probs, labels, ids, th, pred, conf = load(BEST)
    decided = conf >= th
    abst = ~decided
    land = Counter(int(a) for a in probs[abst].argmax(axis=1))
    out["abstention_analysis_seed43"] = {
        "abstained_n": int(abst.sum()),
        "abstained_by_true_level": {
            f"L{LEVELS[k]}": int((labels == k).sum() - ((labels == k) & decided).sum())
            for k in range(5)
        },
        "abstention_rate_by_true_level": {
            f"L{LEVELS[k]}": round(float(1 - ((labels == k) & decided).sum() / (labels == k).sum()), 4)
            for k in range(5)
        },
        "argmax_would_have_landed_on": {f"L{LEVELS[k]}": v for k, v in sorted(land.items())},
        "argmax_would_have_been_correct": int(
            (probs[abst].argmax(axis=1) == labels[abst]).sum()
        ),
        "argmax_would_have_been_under": int(
            (probs[abst].argmax(axis=1) < labels[abst]).sum()
        ),
        "argmax_would_have_been_over": int(
            (probs[abst].argmax(axis=1) > labels[abst]).sum()
        ),
        "note": (
            "argmax shown only to characterise what the model was torn about; "
            "the shipped decision path is CORAL rank decode + abstention, never argmax"
        ),
    }

    # ---- 3. harvest L6 bucket vs consensus label -------------------------
    manifest = {}
    with open("/work/data/manifest_v07_weighted.jsonl") as fh:
        for line in fh:
            r = json.loads(line)
            manifest[r["event_id"]] = r
    l6bucket = [
        (eid, int(labels[i]), int(pred[i]), bool(decided[i]))
        for i, eid in enumerate(ids)
        if manifest[eid]["weak_rule"] == "visible_pieces_about_15mm_l6_candidate"
    ]
    out["harvest_l6_bucket_seed43"] = {
        "n": len(l6bucket),
        "consensus_label_distribution": {
            f"L{LEVELS[k]}": sum(1 for _, t, _, _ in l6bucket if t == k) for k in range(5)
        },
        "share_actually_labelled_L6": round(
            sum(1 for _, t, _, _ in l6bucket if t == 3) / len(l6bucket), 4
        ),
        "true_to_pred_counts": {
            f"true_L{LEVELS[t]}_pred_L{LEVELS[p]}": n
            for (t, p), n in sorted(
                Counter((t, p) for _, t, p, _ in l6bucket).items()
            )
        },
        "wrong_level_decisions": sum(1 for _, t, p, d in l6bucket if d and t != p),
        "abstained": sum(1 for _, _, _, d in l6bucket if not d),
    }

    # ---- 4. bootstrap CIs on headline error rates ------------------------
    rng = np.random.default_rng(20261012)
    idx = np.arange(len(labels))
    stats = {k: [] for k in ("wrong_level_rate", "over_rate", "under_rate_all",
                             "adjacent_slip_rate", "L6_to_L7_rate")}
    for _ in range(4000):
        s = rng.choice(idx, len(idx), replace=True)
        p, l, i2, t2, pr2, cf2 = None, None, None, None, None, None
        # re-decode on the resample (cheap: reuse cached decode)
        d = decided[s]
        pr = pred[s]
        lb = labels[s]
        wrong = int(((pr != lb) & d).sum())
        stats["wrong_level_rate"].append(wrong / len(s))
        stats["over_rate"].append(int(((pr > lb) & d).sum()) / len(s))
        stats["under_rate_all"].append(int(((pr < lb) & d).sum()) / len(s))
        stats["adjacent_slip_rate"].append(int(((np.abs(pr - lb) == 1) & d).sum()) / len(s))
        stats["L6_to_L7_rate"].append(int(((lb == 3) & (pr == 4) & d).sum()) / len(s))
    out["bootstrap_ci_seed43"] = {
        "method": "4000 non-stratified resamples of the 108 test events, seed 20261012",
        "denominator": "all 108 test events (rates are per test event, not per decision)",
        **{
            k: {
                "point": round(
                    float(
                        {
                            "wrong_level_rate": ((pred != labels) & decided).mean(),
                            "over_rate": ((pred > labels) & decided).mean(),
                            "under_rate_all": ((pred < labels) & decided).mean(),
                            "adjacent_slip_rate": ((np.abs(pred - labels) == 1) & decided).mean(),
                            "L6_to_L7_rate": ((labels == 3) & (pred == 4) & decided).mean(),
                        }[k]
                    ),
                    4,
                ),
                "lo": round(float(np.percentile(v, 2.5)), 4),
                "hi": round(float(np.percentile(v, 97.5)), 4),
            }
            for k, v in stats.items()
        },
    }

    # ---- 5. reproduce the published headline metric from the NPZ ---------
    pred_col = np.where(decided, pred, -1)
    published = json.loads((ART / f"v07_seed{BEST}_metrics.published.json").read_text())
    macro_f1 = float(
        __import__("sklearn.metrics", fromlist=["f1_score"]).f1_score(
            labels, pred_col, labels=sorted(set(int(v) for v in labels)),
            average="macro", zero_division=0,
        )
    )
    kappa = float(
        __import__("sklearn.metrics", fromlist=["cohen_kappa_score"]).cohen_kappa_score(
            labels[decided], pred[decided], weights="quadratic"
        )
    )
    out["headline_metric_reproduction_seed43"] = {
        "macro_f1_with_abstentions_recomputed": round(macro_f1, 6),
        "macro_f1_with_abstentions_published": round(
            published["metrics"]["macro_f1_with_abstentions"], 6
        ),
        "match": abs(macro_f1 - published["metrics"]["macro_f1_with_abstentions"]) < 1e-9,
        "weighted_kappa_covered_recomputed": round(kappa, 6),
        "weighted_kappa_covered_published": round(published["metrics"]["weighted_kappa_covered"], 6),
        "kappa_match": abs(kappa - published["metrics"]["weighted_kappa_covered"]) < 1e-9,
        "per_level_recall_decided": {
            f"L{LEVELS[k]}": round(
                float(((labels == k) & (pred == k) & decided).sum() / ((labels == k) & decided).sum()), 4
            )
            if ((labels == k) & decided).any() else None
            for k in range(5)
        },
        "per_level_precision_decided": {
            f"L{LEVELS[k]}": (
                round(
                    float(((labels == k) & (pred == k) & decided).sum() / ((pred == k) & decided).sum()), 4
                )
                if ((pred == k) & decided).any()
                else "undefined: level never emitted as a decision"
            )
            for k in range(5)
        },
        "distinct_levels_emitted_as_decisions": sorted(
            {f"L{LEVELS[int(p)]}" for p, d in zip(pred, decided) if d}
        ),
    }

    (OUT / "structural_seed43.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
