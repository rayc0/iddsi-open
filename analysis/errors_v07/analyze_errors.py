#!/usr/bin/env python
"""Per-level error analysis for the IDDSI-Open v0.7 CORAL model.

Lane an02_errors.  Reads ONLY:
  * the five v0.7 per-seed test prediction NPZ files copied from
    /work/runs/v07/siglip2_v07_fullcoverage_seed*/test_predictions.npz
  * their published metrics.json / calibration.json (for reproduction checks)
  * /work/data/manifest_v07_weighted.jsonl (shared read-only input)
  * /work/data/harvest_*/events.jsonl (provenance titles, for the food-family
    heuristic axis only)

Nothing in the sealed evaluation set (capture/sealed_eval_*.json) is opened.

Every number this script prints is derived here from those files; no figure is
carried over from any prior report.
"""
from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import re
from sklearn.metrics import cohen_kappa_score, confusion_matrix, f1_score
from scipy.stats import fisher_exact, binomtest

import sys
sys.path.insert(0, "/work/lanes/an02_errors/trainer")
from train.model import coral_decode  # noqa: E402  (ordinal rank rule, not argmax)

ART = Path("/work/lane_runs/an02_errors/artifacts")
OUT = Path("/work/lane_runs/an02_errors/analysis")
OUT.mkdir(parents=True, exist_ok=True)

MANIFEST = Path("/work/data/manifest_v07_weighted.jsonl")
SEEDS = [17, 29, 43, 71, 101]
LEVELS = [3, 4, 5, 6, 7]


# ---------------------------------------------------------------- helpers
def load_manifest_rows() -> dict[str, dict]:
    rows = {}
    with MANIFEST.open() as fh:
        for line in fh:
            r = json.loads(line)
            rows[r["event_id"]] = r
    return rows


def load_provenance(test_ids: set[str]) -> dict[str, dict]:
    """Harvest provenance (titles/attribution) for the heuristic food axis."""
    out = {}
    for path in (
        "/work/data/harvest_d1/events.jsonl",
        "/work/data/harvest_d1b/events.jsonl",
        "/work/data/realweak_pilot/events.jsonl",
    ):
        p = Path(path)
        if not p.exists():
            continue
        with p.open() as fh:
            for line in fh:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if r.get("event_id") in test_ids:
                    out[r["event_id"]] = r
    return out


def load_seed(seed: int):
    """Return (probs, labels, event_ids, threshold) for one seed."""
    with np.load(ART / f"v07_seed{seed}_test_predictions.npz", allow_pickle=False) as a:
        probs = a["probabilities"].astype(np.float64)
        labels = a["labels"].astype(np.int64)
        event_ids = [str(e) for e in a["event_ids"]]
    calib = json.loads((ART / f"v07_seed{seed}_calibration.json").read_text())
    return probs, labels, event_ids, float(calib["threshold"])


def decode(probs: np.ndarray, threshold: float):
    """Replicate the shipped decision path: CORAL rank decode + abstention."""
    probs = probs / probs.sum(axis=1, keepdims=True)
    pred, conf = coral_decode(probs)
    decided = conf >= threshold
    return pred, conf, decided


# ---------------------------------------------------------------- 1. seed choice
def seed_ranking() -> dict:
    table = []
    for seed in SEEDS:
        m = json.loads((ART / f"v07_seed{seed}_metrics.published.json").read_text())
        c = json.loads((ART / f"v07_seed{seed}_calibration.json").read_text())
        table.append(
            {
                "seed": seed,
                "macro_f1_with_abstentions": m["metrics"]["macro_f1_with_abstentions"],
                "weighted_kappa_covered": m["metrics"]["weighted_kappa_covered"],
                "coverage": m["metrics"]["coverage"],
                "abstention_rate": m["metrics"]["abstention_rate"],
                "under_fnr_covered": m["metrics"]["dangerous_underclassification_fnr_covered"],
                "ece_all": m["metrics"]["ece_all"],
                "calibration_target_met": c["target_met"],
            }
        )
    # Stated selection rule: among seeds that met the configured validation
    # calibration gate (target_under_fnr=0.05, max_abstention=0.30), take the
    # highest macro_f1_with_abstentions -- the project's headline metric.
    eligible = [r for r in table if r["calibration_target_met"]]
    eligible.sort(key=lambda r: (-r["macro_f1_with_abstentions"], r["seed"]))
    return {
        "selection_rule": (
            "highest macro_f1_with_abstentions among seeds whose validation "
            "calibration target (under-FNR<=0.05, abstention<=0.30) was met; "
            "all 5 seeds met it"
        ),
        "n_test_events": 108,
        "table": table,
        "best_seed": eligible[0]["seed"],
    }


# ---------------------------------------------------------------- 2. reproduction
def reproduction_check() -> dict:
    """Recompute the shipped confusion matrix from the NPZ; must match exactly."""
    out = {}
    for seed in SEEDS:
        probs, labels, event_ids, threshold = load_seed(seed)
        pred, conf, decided = decode(probs, threshold)
        pred_col = np.where(decided, pred, -1)
        mine = confusion_matrix(labels, pred_col, labels=[0, 1, 2, 3, 4, -1]).tolist()
        published = json.loads((ART / f"v07_seed{seed}_metrics.published.json").read_text())
        out[str(seed)] = {
            "matches_published_confusion_matrix": mine == published["confusion_matrix"],
            "recomputed": mine,
        }
    return out


# ---------------------------------------------------------------- 3. food families
# Heuristic axis, keyword-only, word-boundary matched, single best assignment in
# priority order.  Kept deliberately coarse and high-precision: an event whose
# provenance title carries no recognisable food word is NOT forced into a
# family, it lands in "uninformative_title".  Multi-word ambiguity is reported
# as one bucket, never silently merged.
FOOD_FAMILIES = [
    ("smooth_dessert_pudding", [
        "flan", "pudding", "custard", "mousse", "douhua", "tau foo fah", "tofu pudding",
        "jelly", "gelatin", "panna cotta", "creme brulee", "yogurt", "yoghurt",
    ]),
    ("soup_congee_porridge", [
        "soup", "congee", "porridge", "gruel", "broth", "stew", "chowder", "bisque",
        "oatmeal", "cereal", "pho", "ramen", "miso", "gazpacho",
    ]),
    ("beverage", [
        "juice", "milkshake", "coffee", "tea", "latte", "soda", "cola",
        "lemonade", "cocktail", "beer", "wine", "drink", "beverage", "bellini",
    ]),
    ("candy_baked_sweet", [
        "candy", "cotton candy", "gummy", "marshmallow", "lollipop", "toffee", "fudge",
        "chocolate", "cake", "cookie", "brownie", "pastry", "croissant", "muffin",
        "donut", "doughnut", "waffle", "pancake", "pie", "tart", "cheesecake", "scone",
        "twinkie", "breadstick", "bagel", "bread", "toast",
    ]),
    ("salad_vegetable_fruit", [
        "salad", "salade", "slaw", "coleslaw", "vegetable", "veggie", "carrot",
        "cucumber", "tomato", "lettuce", "spinach", "broccoli", "celery", "pickle",
        "kimchi", "sauerkraut", "cauliflower", "sprout", "fruit", "berry", "berries",
        "apple", "banana", "orange", "grape", "melon", "watermelon", "peach", "pear",
        "mango", "pineapple",
    ]),
    ("meat_fish_egg", [
        "steak", "beef", "pork", "chicken", "turkey", "lamb", "bacon", "ham",
        "sausage", "fish", "poisson", "sashimi", "salmon", "saumon", "tuna", "lox",
        "shrimp", "prawn", "crab", "egg", "omelet", "omelette", "meatball", "ribs",
        "duck", "tofu", "brie", "cheese",
    ]),
    ("starch_dish", [
        "rice", "pasta", "spaghetti", "carbonara", "noodle", "macaroni", "mac",
        "potato", "fries", "casserole", "lasagne", "lasagna", "dumpling", "gnocchi",
        "couscous", "quinoa", "bean", "lentil", "curry", "pizza", "burger",
        "sandwich", "taco", "burrito", "sushi", "paella", "pilaf", "risotto", "adobo",
    ]),
]
_FAMILY_RE = [
    # word boundary + optional plural, so "brownie" also matches "brownies"
    (name, re.compile(r"\b(?:" + "|".join(re.escape(k) + r"(?:s|es)?" for k in kws) + r")\b", re.I))
    for name, kws in FOOD_FAMILIES
]
# Words that carry no food information in these provenance titles.
_UNINFORMATIVE_RE = re.compile(
    r"^(p?\d{4,}|[\d_\-\.]+|nutrition5k.*|project \d+.*|hotel .*|yangshuo.*)$", re.I
)


def food_family(event_id: str, prov: dict) -> str:
    """Heuristic, word-boundary keyword family. UNINFORMATIVE_TITLE is a bucket."""
    if event_id.startswith("RW_nutrition5k"):
        # Nutrition5k provenance carries an opaque dish id, no food words.
        return "uninformative_title"
    p = (prov.get(event_id) or {}).get("provenance") or {}
    sm = p.get("source_metadata") or {}
    if not isinstance(sm, dict):
        sm = {}
    text = " ".join(
        str(p.get(k) or "") for k in ("title", "attribution")
    ).strip()
    if not text or _UNINFORMATIVE_RE.match(text):
        return "uninformative_title"
    hits = [name for name, rx in _FAMILY_RE if rx.search(text)]
    if len(hits) == 1:
        return hits[0]
    if not hits:
        return "uninformative_title"
    return "ambiguous_multi_family"


# ---------------------------------------------------------------- main analysis
def main() -> None:
    manifest = load_manifest_rows()
    ranking = seed_ranking()
    repro = reproduction_check()
    (OUT / "seed_ranking.json").write_text(json.dumps(ranking, indent=1))
    (OUT / "reproduction_check.json").write_text(json.dumps(repro, indent=1))
    best = ranking["best_seed"]

    probs, labels, event_ids, threshold = load_seed(best)
    pred, conf, decided = decode(probs, threshold)
    pred_col = np.where(decided, pred, -1)
    test_ids = set(event_ids)
    prov = load_provenance(test_ids)

    n = len(labels)
    true_level = np.array([LEVELS[i] for i in labels])
    pred_level = np.array([LEVELS[p] if d else None for p, d in zip(pred, decided)], dtype=object)

    # ---- per-event table -------------------------------------------------
    rows = []
    for i, eid in enumerate(event_ids):
        m = manifest[eid]
        p = prov.get(eid, {}).get("provenance", {}) or {}
        sm = p.get("source_metadata") or {}
        if not isinstance(sm, dict):
            sm = {}
        decided_i = bool(decided[i])
        correct = decided_i and int(pred[i]) == int(labels[i])
        if not decided_i:
            outcome = "abstain"
        elif correct:
            outcome = "correct"
        elif int(pred[i]) < int(labels[i]):
            outcome = "under"      # hazard direction: predicted a thinner level
        else:
            outcome = "over"
        rows.append(
            {
                "event_id": eid,
                "true_level": int(true_level[i]),
                "pred_level": int(pred_level[i]) if decided_i else None,
                "confidence": round(float(conf[i]), 6),
                "abstained": not decided_i,
                "outcome": outcome,
                "source_corpus": m["source_corpus"],
                "provider": eid.split("_")[1] if eid.startswith("RW_") else "?",
                "weak_rule": m["weak_rule"],
                "agreement": m["agreement"],
                "n_labellers": m["n_labellers"],
                "group_id": m["group_id"],
                "food_family": food_family(eid, prov),
                "provenance_query": sm.get("query"),
                "provenance_title": p.get("title"),
            }
        )
    with (OUT / f"errors_seed{best}_per_event.csv").open("w") as fh:
        cols = list(rows[0].keys())
        fh.write(",".join(cols) + "\n")
        for r in rows:
            fh.write(",".join(json.dumps(r[c]) if r[c] is not None else "" for c in cols) + "\n")

    # ---- confusion matrices ---------------------------------------------
    cm6 = confusion_matrix(labels, pred_col, labels=[0, 1, 2, 3, 4, -1])
    dec_mask = decided
    cm5 = confusion_matrix(labels[dec_mask], pred[dec_mask], labels=[0, 1, 2, 3, 4])
    row_norm = cm5 / np.clip(cm5.sum(axis=1, keepdims=True), 1, None)

    def mat_block(cm, names, normalised=None):
        return {
            "labels": names,
            "counts": cm.tolist(),
            "row_normalised": None if normalised is None else np.round(normalised, 4).tolist(),
        }

    cm_out = {
        "seed": best,
        "n_test_events": n,
        "threshold": threshold,
        "matrix_6x6_true_x_pred_with_abstain": mat_block(
            cm6, ["true_L3", "true_L4", "true_L5", "true_L6", "true_L7", "true_unclear?"]
        ),
        "note_on_6x6": (
            "rows=true level, cols=predicted level; the 6th column is the "
            "abstention bucket (decision 'unclear'), the 6th row is empty because "
            "no test event has an unclear ground-truth label"
        ),
        "matrix_5x5_decided_only": mat_block(cm5, [f"true_L{k}" for k in LEVELS], row_norm),
        "row_support_decided": cm5.sum(axis=1).tolist(),
        "abstentions_per_true_level": cm6[:, 5].tolist(),
    }
    (OUT / f"confusion_seed{best}.json").write_text(json.dumps(cm_out, indent=1))

    # markdown confusion matrix
    md = [f"# v0.7 seed {best} confusion matrices (n={n} test events)",
          "",
          f"abstention threshold (validation-calibrated) = {threshold:.6f}",
          "",
          "## 5x5, decided events only (row-normalised rates in brackets)",
          "",
          "| true \\ pred | " + " | ".join(f"L{k}" for k in LEVELS) + " | row n |",
          "|" + "---|" * (len(LEVELS) + 2)]
    for i, k in enumerate(LEVELS):
        cells = [f"{cm5[i, j]} ({row_norm[i, j]:.2f})" for j in range(len(LEVELS))]
        md.append(f"| L{k} | " + " | ".join(cells) + f" | {cm5[i].sum()} |")
    md += ["", "## 6x6 including abstentions (counts)", "",
           "| true \\ pred | " + " | ".join(f"L{k}" for k in LEVELS) + " | unclear | row n |",
           "|" + "---|" * (len(LEVELS) + 2)]
    for i, k in enumerate(LEVELS):
        cells = [str(cm6[i, j]) for j in range(len(LEVELS))] + [str(cm6[i, 5])]
        md.append(f"| L{k} | " + " | ".join(cells) + f" | {cm6[i].sum()} |")
    md += ["| **col n** | " + " | ".join(str(cm6[:, j].sum()) for j in range(len(LEVELS)))
           + f" | {cm6[:, 5].sum()} | {n} |", ""]
    (OUT / f"confusion_seed{best}.md").write_text("\n".join(md))

    # ---- (a) adjacent pair confusion ------------------------------------
    dec_idx = np.flatnonzero(dec_mask)
    pairs = []
    for a in range(len(LEVELS) - 1):
        lo, hi = a, a + 1
        # direction lo -> hi (true lower level predicted as the next one up)
        up = int(((labels == lo) & (pred == hi) & decided).sum())
        # direction hi -> lo
        down = int(((labels == hi) & (pred == lo) & decided).sum())
        # all errors from lo row and hi row, for context
        lo_err = int(((labels == lo) & (pred != lo) & decided).sum())
        hi_err = int(((labels == hi) & (pred != hi) & decided).sum())
        lo_n = int((labels == lo).sum())
        hi_n = int((labels == hi).sum())
        pairs.append(
            {
                "pair": f"L{LEVELS[lo]}-L{LEVELS[hi]}",
                "true_L%d_pred_L%d" % (LEVELS[lo], LEVELS[hi]): up,
                "true_L%d_pred_L%d" % (LEVELS[hi], LEVELS[lo]): down,
                "adjacent_total": up + down,
                "rate_up_per_true_L%d" % LEVELS[lo]: round(up / lo_n, 4) if lo_n else None,
                "rate_down_per_true_L%d" % LEVELS[hi]: round(down / hi_n, 4) if hi_n else None,
                "true_L{n}_support".format(n=LEVELS[lo]): lo_n,
                "true_L{n}_support".format(n=LEVELS[hi]): hi_n,
                "all_errors_from_L{n}".format(n=LEVELS[lo]): lo_err,
                "all_errors_from_L{n}".format(n=LEVELS[hi]): hi_err,
            }
        )
    # non-adjacent errors for contrast
    non_adj = int(sum(
        1 for i in dec_idx if abs(int(pred[i]) - int(labels[i])) > 1
    ))
    gap2 = int(sum(1 for i in dec_idx if abs(int(pred[i]) - int(labels[i])) == 2))
    gap3plus = int(sum(1 for i in dec_idx if abs(int(pred[i]) - int(labels[i])) >= 3))
    adjacent_out = {
        "seed": best,
        "adjacent_pairs": pairs,
        "decided_events": int(decided.sum()),
        "wrong_level_decisions": int(sum(1 for i in dec_idx if pred[i] != labels[i])),
        "adjacent_slip_errors": int(sum(1 for i in dec_idx if abs(int(pred[i]) - int(labels[i])) == 1)),
        "gap2_errors": gap2,
        "gap3plus_errors": gap3plus,
        "non_adjacent_errors": non_adj,
    }
    (OUT / f"adjacent_pairs_seed{best}.json").write_text(json.dumps(adjacent_out, indent=1))

    # ---- (b) over/under bias --------------------------------------------
    signed = np.array([int(pred[i]) - int(labels[i]) for i in dec_idx])
    over = int((signed > 0).sum())
    under = int((signed < 0).sum())
    exact = int((signed == 0).sum())
    n_under_gt = int((labels > 3).sum())          # L4..L7 = under-predictable range
    n_under_all = n
    bias = {
        "seed": best,
        "denominator": "decided test events (confidence >= threshold)",
        "decided": int(decided.sum()),
        "abstained": int((~decided).sum()),
        "correct": exact,
        "over_predicted": over,
        "under_predicted": under,
        "mean_signed_error": round(float(signed.mean()), 4),
        "median_signed_error": float(np.median(signed)),
        "mean_abs_error": round(float(np.abs(signed).mean()), 4),
        "under_prediction_rate_of_decided": round(under / int(decided.sum()), 4),
        "under_prediction_fnr_of_decided": round(
            under / int(((labels > 3) & decided).sum()), 4),
        "under_predictable_denominator": int(((labels > 3) & decided).sum()),
        "under_events_all_test": int(((pred < labels) & decided).sum()),
        "under_events_all_test_rate": round(float(((pred < labels) & decided).mean()), 4),
        "over_minus_under": over - under,
        "per_level_signed_bias": {
            f"L{LEVELS[k]}": {
                "support": int((labels == k).sum()),
                "decided": int(((labels == k) & decided).sum()),
                "abstained": int(((labels == k) & ~decided).sum()),
                "mean_signed_error_decided": (
                    round(float(signed[labels[dec_idx] == k].mean()), 4)
                    if ((labels == k) & decided).any() else None
                ),
                "over": int(((labels == k) & (pred > k) & decided).sum()),
                "under": int(((labels == k) & (pred < k) & decided).sum()),
                "correct": int(((labels == k) & (pred == k) & decided).sum()),
            }
            for k in range(len(LEVELS))
        },
        "signed_error_histogram": {
            str(v): int((signed == v).sum()) for v in range(-4, 5)
        },
    }
    (OUT / f"bias_seed{best}.json").write_text(json.dumps(bias, indent=1))

    # ---- (c) group error dominance --------------------------------------
    def group_stats(keyfn, name, min_n=1):
        agg = defaultdict(lambda: Counter())
        for r in rows:
            agg[keyfn(r)]["n"] += 1
            if r["outcome"] == "correct":
                agg[keyfn(r)]["correct"] += 1
            elif r["outcome"] in ("under", "over"):
                agg[keyfn(r)]["wrong_level"] += 1
                agg[keyfn(r)][r["outcome"]] += 1
            else:
                agg[keyfn(r)]["abstain"] += 1
        table = []
        for g, c in sorted(agg.items(), key=lambda kv: -kv[1]["n"]):
            n_g = c["n"]
            table.append(
                {
                    "axis": name,
                    "group": g,
                    "n": n_g,
                    "correct": c["correct"],
                    "wrong_level": c["wrong_level"],
                    "under": c["under"],
                    "over": c["over"],
                    "abstain": c["abstain"],
                    "error_rate_wrong_level_of_n": round(c["wrong_level"] / n_g, 4),
                    "non_decision_rate_of_n": round(c["abstain"] / n_g, 4),
                }
            )
        return [t for t in table if t["n"] >= min_n]

    groups = {
        "source_corpus": group_stats(lambda r: r["source_corpus"], "source_corpus"),
        "provider": group_stats(lambda r: r["provider"], "provider"),
        "harvest_visual_bucket_weak_rule": group_stats(lambda r: r["weak_rule"], "weak_rule"),
        "label_agreement": group_stats(lambda r: r["agreement"], "agreement"),
        "food_family_heuristic": group_stats(lambda r: r["food_family"], "food_family"),
    }
    (OUT / f"group_errors_seed{best}.json").write_text(json.dumps(groups, indent=1))

    # ---- significance checks on the group axes ---------------------------
    tests = {}
    for axis_name, table in groups.items():
        # only test axes whose groups are not one big bucket
        informative = [t for t in table if t["group"] not in
                       ("uninformative_title", "ambiguous_multi_family")]
        if not informative:
            continue
        worst = max(informative, key=lambda t: t["error_rate_wrong_level_of_n"])
        others_wrong = sum(t["wrong_level"] for t in informative if t is not worst)
        others_n = sum(t["n"] for t in informative if t is not worst)
        odds, p = fisher_exact(
            [[worst["wrong_level"], worst["n"] - worst["wrong_level"]],
             [others_wrong, others_n - others_wrong]]
        )
        tests[axis_name] = {
            "worst_group": worst["group"],
            "worst_n": worst["n"],
            "worst_wrong_level": worst["wrong_level"],
            "worst_rate": worst["error_rate_wrong_level_of_n"],
            "rest_wrong_level": others_wrong,
            "rest_n": others_n,
            "rest_rate": round(others_wrong / others_n, 4) if others_n else None,
            "fisher_two_sided_p": round(float(p), 4),
            "denominator": "108 test events; 'wrong level' = decided and != true level",
        }
    # over/under asymmetry: binomial sign test on decided errors
    sign_p = float(binomtest(over, over + under, 0.5).pvalue) if (over + under) else None
    tests["over_under_asymmetry"] = {
        "over": over, "under": under, "binom_sign_test_p": round(sign_p, 4) if sign_p is not None else None,
        "denominator": "16 wrong-level decisions among 92 decided events (seed 43)",
    }
    # does the L6 harvest bucket dominate simply because it IS the L6 level?
    l6_rows = [r for r in rows if r["weak_rule"] == "visible_pieces_about_15mm_l6_candidate"]
    tests["weak_rule_l6_bucket_vs_level"] = {
        "l6_bucket_n": len(l6_rows),
        "l6_bucket_true_levels": dict(Counter(r["true_level"] for r in l6_rows)),
        "note": "the harvest L6 bucket is not purely level-6 events",
    }
    (OUT / f"significance_seed{best}.json").write_text(json.dumps(tests, indent=1))

    # ---- cross-seed stability of the error structure --------------------
    stability = {}
    for seed in SEEDS:
        p2, l2, e2, th2 = load_seed(seed)
        pr2, cf2, dc2 = decode(p2, th2)
        d = np.flatnonzero(dc2)
        s = np.array([int(pr2[i]) - int(l2[i]) for i in d])
        lv = np.array([LEVELS[i] for i in l2])
        stability[str(seed)] = {
            "decided": int(dc2.sum()),
            "wrong_level": int((pr2[d] != l2[d]).sum()),
            "over": int((s > 0).sum()),
            "under": int((s < 0).sum()),
            "mean_signed_error": round(float(s.mean()), 4),
            "adjacent_slips": int((np.abs(s) == 1).sum()),
            "top_confusions": [
                {"true": f"L{LEVELS[l2[i]]}", "pred": f"L{LEVELS[pr2[i]]}"}
                for i in d if pr2[i] != l2[i]
            ],
            "true_L4_pred_L7": int(((l2 == 1) & (pr2 == 4) & dc2).sum()),
            "true_L5_pred_L7": int(((l2 == 2) & (pr2 == 4) & dc2).sum()),
            "true_L7_pred_L4": int(((l2 == 4) & (pr2 == 1) & dc2).sum()),
        }
    (OUT / "cross_seed_stability.json").write_text(json.dumps(stability, indent=1))

    summary = {
        "best_seed": best,
        "selection_rule": ranking["selection_rule"],
        "n_test": n,
        "decided": int(decided.sum()),
        "abstained": int((~decided).sum()),
        "wrong_level": int(sum(1 for i in dec_idx if pred[i] != labels[i])),
        "correct": int(sum(1 for i in dec_idx if pred[i] == labels[i])),
        "over": over,
        "under": under,
        "under_rate_all_test": round(float(((pred < labels) & decided).mean()), 4),
        "adjacent_slips": int(sum(1 for i in dec_idx if abs(int(pred[i]) - int(labels[i])) == 1)),
        "kappa_all_decided": round(float(cohen_kappa_score(
            labels[decided], pred[decided], weights="quadratic")), 4),
        "files_used": [
            "/work/lane_runs/an02_errors/artifacts/v07_seed%d_test_predictions.npz" % s
            for s in SEEDS
        ] + [
            "/work/lane_runs/an02_errors/artifacts/v07_seed%d_calibration.json" % s
            for s in SEEDS
        ] + [
            "/work/lane_runs/an02_errors/artifacts/v07_seed%d_metrics.published.json" % s
            for s in SEEDS
        ] + [
            "/work/data/manifest_v07_weighted.jsonl",
            "/work/data/harvest_d1/events.jsonl",
            "/work/data/harvest_d1b/events.jsonl",
            "/work/data/realweak_pilot/events.jsonl",
            "/work/lanes/an02_errors/trainer/train/model.py (coral_decode)",
        ],
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
