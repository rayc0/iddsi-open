#!/usr/bin/env python3
"""Measure the v0.7 release numbers from the stored artefacts, so the model card
quotes nothing it cannot reproduce.

Everything below is recomputed from:
  * /work/data/manifest_v07_weighted.jsonl      (v0.7 consensus labels, test split)
  * /work/data/manifest_v05_clean.jsonl         (v0.5 consensus labels, test split)
  * /work/runs/v07/siglip2_v07_fullcoverage_seed*/test_predictions.npz
  * /work/runs/v07/siglip2_v07_fullcoverage_seed*/calibration.json

and cross-checked against the already-published
  * results/v07_release/v07_seed*_metrics.json  (trainer/eval/metrics.py output)
  * results/v07_release/bias_matrix.json
  * results/v07_release/head_to_head2.json

CPU only. Writes /work/lane_runs/rl04_modelcard/v07_measured.json.
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
from sklearn.metrics import cohen_kappa_score, f1_score

sys.path.insert(0, "/work/lanes/rl04_modelcard/trainer")
from train.model import coral_decode  # noqa: E402  (the shipped decode rule)

OUT = pathlib.Path("/work/lane_runs/rl04_modelcard/v07_measured.json")
RUNS = pathlib.Path("/work/runs/v07")
LANE_RESULTS = pathlib.Path("/work/lanes/rl04_modelcard/results/v07_release")

MANIFESTS = {
    "v0.7 consensus (full labeller coverage)": "/work/data/manifest_v07_weighted.jsonl",
    "v0.5 consensus (cleaned corpus)": "/work/data/manifest_v05_clean.jsonl",
}


def load_truth(path: str) -> dict[str, int]:
    rows = [json.loads(line) for line in open(path) if line.strip()]
    return {r["event_id"]: int(r["level"]) - 3 for r in rows if r.get("split") == "test"}


def augrc(conf: np.ndarray, correct: np.ndarray) -> float:
    order = np.argsort(-conf)
    c = correct[order].astype(float)
    n = len(c)
    ks = np.arange(1, n + 1)
    risk = np.cumsum(1.0 - c) / ks
    integrate = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    return float(integrate(risk, ks / n))


def measure(seed_dir: pathlib.Path, truth: dict[str, int]) -> dict:
    npz = np.load(seed_dir / "test_predictions.npz", allow_pickle=True)
    probs = npz["probabilities"].astype(np.float64)
    probs = probs / probs.sum(axis=1, keepdims=True)
    labels = npz["labels"].astype(int)
    eids = [str(x) for x in npz["event_ids"]]
    cal = json.load(open(seed_dir / "calibration.json"))
    threshold = float(cal["threshold"])

    raw, conf = coral_decode(probs)
    raw = np.asarray(raw).astype(int)
    conf = np.asarray(conf).astype(float)
    acc = conf >= threshold

    keep = [i for i, e in enumerate(eids) if e in truth]
    pred = raw[keep]
    tru = np.array([truth[eids[i]] for i in keep])
    cf = conf[keep]
    ca = acc[keep]
    n = len(keep)

    correct = (pred == tru).astype(float)
    out = {
        "n": n,
        "coverage": float(ca.mean()),
        "abstention_rate": float(1.0 - ca.mean()),
        "kappa_all_quadratic": float(cohen_kappa_score(tru, pred, weights="quadratic")),
        "kappa_covered_quadratic": float(
            cohen_kappa_score(tru[ca], pred[ca], weights="quadratic")
        ) if ca.any() else float("nan"),
        "exact_all": float(correct.mean()),
        "exact_covered": float((pred[ca] == tru[ca]).mean()) if ca.any() else float("nan"),
        "mae_all_levels": float(np.abs(pred - tru).mean()),
        "mae_covered_levels": float(np.abs(pred[ca] - tru[ca]).mean()) if ca.any() else float("nan"),
        "fnr_covered": float((pred[ca] < tru[ca]).mean()) if ca.any() else float("nan"),
        "underclassification_allN": float((pred < tru).mean()),
        # exactly as trainer/eval/metrics.py::_point_metrics computes it:
        # abstentions become class -1, but the label set is built from the TRUE
        # labels only, so -1 is never scored as a class (hazard H-MACROF1).
        "macro_f1_published_labelset": float(
            f1_score(tru, np.where(ca, pred, -1),
                     labels=sorted(set(int(v) for v in tru)),
                     average="macro", zero_division=0)
        ),
        "macro_f1_abstention_as_class": float(
            f1_score(
                tru,
                np.where(ca, pred, -1),
                labels=[-1, 0, 1, 2, 3, 4],
                average="macro",
                zero_division=0,
            )
        ),
        "augrc": augrc(cf, correct),
        "threshold": threshold,
        "temperature": float(cal.get("temperature", float("nan"))),
        "calibration_target_met": bool(cal.get("target_met")),
        "calibration_under_fnr": float(cal.get("under_fnr", float("nan"))),
        "calibration_coverage": float(cal.get("coverage", float("nan"))),
    }
    # labels are identical in the npz and in the manifest test split: assert it,
    # otherwise every number above is measured against a different population.
    stored = {e: int(l) for e, l in zip(eids, labels)}
    mismatched = [e for e in stored if e in truth and stored[e] != truth[e]]
    out["label_mismatches_vs_manifest"] = len(mismatched)
    out["events_in_both"] = n
    return out


def rng(values: list[float], nd: int = 3) -> str:
    v = [x for x in values if np.isfinite(x)]
    if not v:
        return "n/a"
    return f"{min(v):.{nd}f}-{max(v):.{nd}f}"


def main() -> None:
    truths = {name: load_truth(p) for name, p in MANIFESTS.items()}
    report: dict = {"test_split_sizes": {k: len(v) for k, v in truths.items()}}

    seed_dirs = sorted(d for d in RUNS.glob("*_seed*") if d.is_dir())
    report["seeds"] = [d.name.split("_seed")[-1] for d in seed_dirs]

    per_seed = {}
    for yardstick, truth in truths.items():
        per_seed[yardstick] = {d.name.split("_seed")[-1]: measure(d, truth) for d in seed_dirs}
    report["per_seed"] = per_seed

    v07 = per_seed["v0.7 consensus (full labeller coverage)"]
    keys = [
        "coverage", "kappa_all_quadratic", "kappa_covered_quadratic", "exact_all",
        "exact_covered", "mae_all_levels", "mae_covered_levels", "fnr_covered",
        "underclassification_allN", "macro_f1_published_labelset",
        "macro_f1_abstention_as_class", "augrc",
    ]
    report["v07_own_labels_ranges"] = {k: rng([s[k] for s in v07.values()]) for k in keys}
    report["v07_own_labels_per_seed"] = {
        s: {k: round(v07[s][k], 4) for k in keys} for s in v07
    }

    # ---- bias matrix: same models, two yardsticks ----
    bias = {}
    for yardstick in truths:
        bias[yardstick] = {
            "mean_kappa_all_quadratic": float(
                np.mean([s["kappa_all_quadratic"] for s in per_seed[yardstick].values()])
            ),
            "min": min(s["kappa_all_quadratic"] for s in per_seed[yardstick].values()),
            "max": max(s["kappa_all_quadratic"] for s in per_seed[yardstick].values()),
        }
    report["v07_model_both_yardsticks"] = bias

    # ---- reproduce the already-published files, to prove the card's numbers
    #      are the same measurement, not a new one ----
    published = {}
    for f in sorted(LANE_RESULTS.glob("v07_seed*_metrics.json")):
        seed = f.name.split("_seed")[-1].split("_")[0]
        published[seed] = json.load(open(f))["metrics"]
    report["published_metrics_files"] = published
    checks = []
    for seed, pub in published.items():
        mine = v07.get(seed, {})
        for pubkey, mykey in [
            ("weighted_kappa_covered", "kappa_covered_quadratic"),
            ("macro_f1_with_abstentions", "macro_f1_published_labelset"),
            ("dangerous_underclassification_fnr_covered", "fnr_covered"),
            ("coverage", "coverage"),
            ("abstention_rate", "abstention_rate"),
        ]:
            checks.append(
                {
                    "seed": seed,
                    "metric": pubkey,
                    "published": pub[pubkey],
                    "remeasured": mine.get(mykey),
                    "abs_diff": abs(pub[pubkey] - mine[mykey])
                    if mine.get(mykey) is not None and np.isfinite(mine[mykey])
                    else None,
                }
            )
    report["reproduction_checks"] = checks
    report["reproduction_max_abs_diff"] = max(
        c["abs_diff"] for c in checks if c["abs_diff"] is not None
    )

    published_bias = json.load(open(LANE_RESULTS / "bias_matrix.json"))
    report["published_bias_matrix"] = published_bias

    # ---- pre-registered gates, evaluated on every yardstick that exists ----
    # wk1: weighted kappa >= 0.70, else kill photo-classification.
    # wk2: external macro-F1 >= 0.75 AND dangerous under-classification <= 5%
    #      at <= 30% abstention.
    gates = {}
    for yardstick, seeds in per_seed.items():
        cov_k = [s["kappa_covered_quadratic"] for s in seeds.values()]
        all_k = [s["kappa_all_quadratic"] for s in seeds.values()]
        f1 = [s["macro_f1_published_labelset"] for s in seeds.values()]
        fnr = [s["fnr_covered"] for s in seeds.values()]
        abst = [s["abstention_rate"] for s in seeds.values()]
        gates[yardstick] = {
            "n_seeds": len(seeds),
            "wk1_kappa_ge_0.70_covered": sum(1 for k in cov_k if k >= 0.70),
            "wk1_kappa_ge_0.70_all_rows": sum(1 for k in all_k if k >= 0.70),
            "wk1_covered_range": rng(cov_k),
            "wk1_all_rows_range": rng(all_k),
            "wk2_macro_f1_ge_0.75": sum(1 for v in f1 if v >= 0.75),
            "wk2_macro_f1_range": rng(f1),
            "wk2_fnr_le_0.05": sum(1 for v in fnr if v <= 0.05),
            "wk2_fnr_range": rng(fnr),
            "wk2_abstention_le_0.30": sum(1 for v in abst if v <= 0.30),
            "wk2_abstention_range": rng(abst),
            "wk2_met": sum(
                1 for f, n, a in zip(f1, fnr, abst) if f >= 0.75 and n <= 0.05 and a <= 0.30
            ),
        }
    report["gates"] = gates

    # ---- how much do the two yardsticks actually disagree, and in which
    #      direction?  Same 108 event ids, two consensus label sets. ----
    v07_rows = [json.loads(l) for l in open("/work/data/manifest_v07_weighted.jsonl") if l.strip()]
    v05_rows = [json.loads(l) for l in open("/work/data/manifest_v05_clean.jsonl") if l.strip()]
    t7 = {r["event_id"]: int(r["level"]) for r in v07_rows if r["split"] == "test"}
    t5 = {r["event_id"]: int(r["level"]) for r in v05_rows if r["split"] == "test"}
    shared = sorted(set(t7) & set(t5))
    diff = [e for e in shared if t7[e] != t5[e]]
    report["label_set_diff"] = {
        "shared_test_events": len(shared),
        "identical_ids": set(t7) == set(t5),
        "differing_labels": len(diff),
        "differing_fraction": len(diff) / len(shared),
        "v07_higher_than_v05": sum(1 for e in diff if t7[e] > t5[e]),
        "v07_lower_than_v05": sum(1 for e in diff if t7[e] < t5[e]),
        "mean_abs_shift_levels": sum(abs(t7[e] - t5[e]) for e in shared) / len(shared),
    }

    OUT.write_text(json.dumps(report, indent=1))
    print(f"wrote {OUT}")
    print("\n=== v0.7 on its own labels (5 seeds, held-out test) ===")
    for k, v in report["v07_own_labels_ranges"].items():
        print(f"  {k:<34} {v}")
    print("\n=== reproduction check vs results/v07_release/v07_seed*_metrics.json ===")
    print(f"  max abs diff = {report['reproduction_max_abs_diff']:.2e}")
    print("\n=== v0.7 model, both yardsticks (quadratic kappa, all rows) ===")
    for y, v in bias.items():
        print(f"  {y:<44} mean {v['mean_kappa_all_quadratic']:.3f}  ({v['min']:.3f}-{v['max']:.3f})")
    print("\n=== pre-registered gates, per yardstick ===")
    for y, g in gates.items():
        print(f"  {y}")
        print(f"    wk1 kappa>=0.70   : covered {g['wk1_kappa_ge_0.70_covered']}/{g['n_seeds']}"
              f" ({g['wk1_covered_range']})   all-rows {g['wk1_kappa_ge_0.70_all_rows']}/{g['n_seeds']}"
              f" ({g['wk1_all_rows_range']})")
        print(f"    wk2 macro-F1>=0.75: {g['wk2_macro_f1_ge_0.75']}/{g['n_seeds']}"
              f" ({g['wk2_macro_f1_range']})")
        print(f"    wk2 FNR<=0.05     : {g['wk2_fnr_le_0.05']}/{g['n_seeds']}"
              f" ({g['wk2_fnr_range']})")
        print(f"    wk2 abstention<=0.30: {g['wk2_abstention_le_0.30']}/{g['n_seeds']}"
              f" ({g['wk2_abstention_range']})")
        print(f"    wk2 met on every clause: {g['wk2_met']}/{g['n_seeds']}")


if __name__ == "__main__":
    main()
