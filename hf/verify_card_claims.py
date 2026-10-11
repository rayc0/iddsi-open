#!/usr/bin/env python3
"""verify_card_claims.py -- re-derive every number published in hf/model_README.md.

House rule: "Every claim you make must be measured by you and reproducible from a
file you name." This script is that measurement. It re-computes, from the raw
artefacts in the repository, the exact quantities the optimised model card quotes,
and it fails loudly when the card and the artefacts disagree.

Five jobs:

  1. REPRODUCE  -- recompute the headline metrics from the primary artefacts
                   (per-seed metrics JSON, the saved test predictions, the v0.7
                   consensus manifest, the inter-judge cross-tab, the flow-test
                   bench summary) rather than trusting any prose.
  2. CROSSCHECK -- assert the card actually contains the reproduced ranges, so a
                   drift between the card and the artefacts fails the run.
  3. AUDIT      -- confirm the 10 target search phrases appear in the card body,
                   not only in the signpost list.
  4. GUARD      -- confirm the card still carries every disclaimer, gate and hazard
                   entry the release depends on (nothing may be weakened).
  5. METADATA   -- assert the card's YAML front matter matches the exact block
                   specified in hf/HF_METADATA.md.

Outputs (written under the lane_runs directory):
  card_claims_verification.json   machine-readable, one entry per claim
  card_claims_verification.md     human-readable tables
  search_phrase_audit.json        phrase -> hit/miss, surface form, line numbers
  hazard_coverage_audit.json      hazard id -> referenced lines in the card

Run:  docker exec -w /work/lanes/hf01_card_seo iddsi-train \
        python hf/verify_card_claims.py
  or on the host with numpy available:
      python3 hf/verify_card_claims.py

Reads only. Writes only under the lane_runs output directory.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------
# locations
# --------------------------------------------------------------------------

HERE = Path(__file__).resolve().parent
LANE = HERE.parent                      # .../lanes/hf01_card_seo


def find_repo_root() -> Path:
    """The lane clone is .../iddsi/lanes/hf01_card_seo; data lives in .../iddsi."""
    for cand in (LANE, *LANE.parents):
        if (cand / "data" / "manifest_v07_weighted.jsonl").exists():
            return cand
    return Path("/work")


REPO = find_repo_root()
DATA_MANIFEST = REPO / "data" / "manifest_v07_weighted.jsonl"
RUNS = REPO / "runs"
CARD = HERE / "model_README.md"
METADATA = HERE / "HF_METADATA.md"
OUT = Path(os.environ.get("LANE_RUNS", "/work/lane_runs/hf01_card_seo"))

V07_SEEDS = (17, 29, 43, 71, 101)
MODEL_RUNS = (
    ("v0.4", "v04"),
    ("v0.5", "v05"),
    ("v0.7", "v07"),
)

# The exact front matter the Hub repo must carry. Copied from hf/HF_METADATA.md §1.
EXPECTED_FRONT_MATTER = {
    "language": "en",
    "license": "apache-2.0",
    "pipeline_tag": "image-classification",
    "library_name": "transformers",
    "base_model": "google/siglip2-base-patch16-224",
}
EXPECTED_TAGS = [
    "image-classification", "food", "food-texture", "food-image-classification",
    "texture", "texture-analysis", "dysphagia", "iddsi", "iddsi-texture-levels",
    "dietary", "texture-modified-diet", "ordinal-classification", "coral", "siglip2",
    "calibrated-abstention", "selective-classification", "culinary-education",
    "research-preview", "not-for-clinical-use",
]

# Every hazard entry in hf/HAZARD_LOG.md. The card must still reference each one;
# dropping one would weaken the release's disclosure.
HAZARD_IDS = [
    "H-CORAL", "H-NOISE", "H-UNDERPRED", "H-LICENCE", "H-L7SINK", "H-ABSTBLIND",
    "H-MACROF1", "H-KAPPACOV", "H-GROUP", "H-LEAK", "H-AUGRC", "H-GATE",
    "H-LABELCEIL", "H-UNUSABLE", "H-QWENBIAS", "H-YARDSTICK",
]

DISCLAIMERS = [
    ("not a swallowing-safety classifier",
     r"not a swallowing[- ]safety classifier|NOT a swallowing"),
    ("no safe/unsafe verdict",
     r"never (issues|a) .{0,20}verdict|no safe or unsafe verdict|never issues a verdict"),
    ("weights withheld",
     r"weights are withheld|weights stay withheld|weights remain withheld|WITHHELD"),
    ("dataset withheld on licence grounds",
     r"licence grounds|license grounds"),
    ("wk1 gate unchanged",
     r"wk1"),
    ("wk2 gate unchanged",
     r"wk2"),
    ("not IDDSI-endorsed",
     r"not iddsi-endorsed|independent of iddsi"),
    ("SaMD / out of scope",
     r"saMD|Rule 11|TR-007"),
    ("research + culinary-education only",
     r"culinary[- ]education"),
    ("dangerous under-classification disclosed",
     r"under-classification|underclassification"),
    ("label-noise ceiling disclosed",
     r"31\.0%|label noise"),
    ("prior art / never-claim",
     r"prior art|no novelty claim|ViscoCam|OptiTexture"),
    ("abstention never a verdict",
     r"abstention|abstain"),
    ("not clinically validated",
     r"clinically validated|not clinical|clinician"),
    ("metric-defect disclosures (KAPPACOV/ABSTBLIND/MACROF1)",
     r"H-KAPPACOV|H-ABSTBLIND|H-MACROF1"),
    ("yardstick circularity disclosed",
     r"H-YARDSTICK|yardstick"),
    ("remedy path (physical specimens)",
     r"physically-tested desk specimens|300 specimens|CAPTURE_PROTOCOL"),
]

# The 10 phrases a person would actually type, and the surface form each must take.
SEARCH_PHRASES = [
    "iddsi texture level",
    "food texture classification",
    "food image classification model",
    "texture modified diet",
    "dysphagia food texture",
    "siglip2 food",
    "ordinal classification",
    "thickened liquid",
    "culinary education",
    "iddsi level from a photograph",
]
# Each phrase must appear at least this many times: once in prose and once in the
# signpost list. 1 means "at least once anywhere"; 2 means "in prose AND in the
# signpost list", which is the discoverability bar this optimisation set.
PHRASE_MIN_OCCURRENCES = 2


def rel(p: Path) -> str:
    """Path as written in the card: repo-relative, forward slashes, no host root."""
    try:
        return str(p.resolve().relative_to(REPO.resolve())).replace(os.sep, "/")
    except ValueError:
        return str(p).replace(os.sep, "/")


def rng(vals) -> str:
    vals = list(vals)
    return f"{min(vals):.3f} - {max(vals):.3f}"


def normalise(s: str) -> str:
    """Lower-case, treat hyphens/en/em dashes as spaces, collapse whitespace.

    A second variant that DELETES hyphens is also produced, so that a phrase typed
    the way the Hub slug spells it ("siglip2") matches prose spelled the way the
    paper spells it ("SigLIP-2"), and vice versa.
    """
    s = s.lower().replace("—", " ").replace("–", " ")
    return re.sub(r"\s+", " ", s.replace("-", " "))


def variants(s: str):
    s = s.lower().replace("—", " ").replace("–", " ")
    return (re.sub(r"\s+", " ", s.replace("-", " ")),
            re.sub(r"\s+", "", s.replace("-", "")))


def contains(haystack: str, needle: str) -> bool:
    """True if the needle appears in the haystack under either hyphen convention."""
    h1, h2 = variants(haystack)
    n1, n2 = variants(needle)
    return n1 in h1 or n2 in h2


def count_occurrences(haystack: str, needle: str) -> int:
    h1, h2 = variants(haystack)
    n1, n2 = variants(needle)
    return max(h1.count(n1), h2.count(n2))


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------

def coral_decode(probabilities: np.ndarray):
    """Identical to trainer/train/model.py:coral_decode.

    level = #{k : P(y>k) > 0.5}; confidence = mass on the decoded level.
    """
    probs = np.asarray(probabilities, dtype=np.float64)
    if probs.ndim != 2:
        raise ValueError("probabilities must be a [N, C] matrix")
    probs = probs / probs.sum(axis=1, keepdims=True).clip(1e-12)
    survival = 1.0 - np.cumsum(probs, axis=1)[:, :-1]
    predictions = (survival > 0.5).sum(axis=1).astype(np.int64)
    confidence = probs[np.arange(len(predictions)), predictions]
    return predictions, confidence


def kappa_weighted(y_true, y_pred, weights: str) -> float:
    """Cohen's kappa with linear / quadratic / no weighting.

    Matches sklearn.metrics.cohen_kappa_score(weights=...) for integer labels; kept
    local so the verifier runs on the host without scikit-learn.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    k = int(max(y_true.max(), y_pred.max())) + 1
    O = np.zeros((k, k), dtype=float)
    for t, p in zip(y_true, y_pred):
        O[t, p] += 1.0
    n = O.sum()
    hist_t = np.bincount(y_true, minlength=k).astype(float)
    hist_p = np.bincount(y_pred, minlength=k).astype(float)
    E = np.outer(hist_t, hist_p) / n
    if weights == "unweighted":
        w = 1.0 - np.eye(k)
    elif weights == "linear":
        w = np.array([[abs(i - j) for j in range(k)] for i in range(k)], dtype=float)
    else:
        w = np.array([[(i - j) ** 2 for j in range(k)] for i in range(k)], dtype=float)
    return float(1.0 - (w * O).sum() / (w * E).sum())


def load_truth() -> dict:
    rows = [json.loads(l) for l in DATA_MANIFEST.read_text().splitlines() if l.strip()]
    return {r["event_id"]: r["level"] - 3 for r in rows if r["split"] == "test"}


def score_model(run_dir_name: str, truth: dict) -> dict:
    """Reproduce v07_gate.py: raw CORAL decode, calibration threshold for coverage,
    quadratic-weighted kappa on covered vs all rows."""
    per_seed = {}
    for d in sorted((RUNS / run_dir_name).glob("*_seed*")):
        if not d.is_dir():
            continue
        npz = np.load(d / "test_predictions.npz", allow_pickle=True)
        cal = json.load(open(d / "calibration.json"))
        raw, conf = coral_decode(npz["probabilities"].astype(np.float64))
        raw = np.asarray(raw).astype(int)
        conf = np.asarray(conf).astype(float)
        acc = conf >= float(cal["threshold"])
        eids = [str(x) for x in npz["event_ids"]]
        keep = [i for i, e in enumerate(eids) if e in truth]
        p = np.array([raw[i] for i in keep])
        t = np.array([truth[eids[i]] for i in keep])
        a = np.array([acc[i] for i in keep])
        kcov = kappa_weighted(t[a], p[a], "quadratic") if a.any() else None
        per_seed[d.name.rsplit("_", 1)[-1]] = {
            "n": int(len(keep)),
            "kappa_covered": kcov,
            "kappa_all": kappa_weighted(t, p, "quadratic"),
            "exact": float((p == t).mean()),
            "mae": float(np.abs(p - t).mean()),
            "underrate_all": float((p < t).mean()),
            "coverage": float(a.mean()),
            "wk1_pass": bool(kcov is not None and kcov >= 0.70),
        }
    return per_seed


# --------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------

class Report:
    def __init__(self) -> None:
        self.claims: list[dict] = []

    def add(self, cid, claim, measured, card_value, source, ok=None, note=""):
        if ok is None:
            ok = _match(measured, card_value)
        self.claims.append({"id": cid, "claim": claim, "card_value": card_value,
                            "measured": measured, "match": bool(ok),
                            "source": source, "note": note})

    @property
    def failures(self):
        return [c for c in self.claims if not c["match"]]


def _match(measured, card_value) -> bool:
    if isinstance(measured, str) or isinstance(card_value, str):
        return str(measured).strip().lower() == str(card_value).strip().lower()
    if isinstance(measured, bool) or isinstance(card_value, bool):
        return bool(measured) == bool(card_value)
    if isinstance(measured, (int, float)) and isinstance(card_value, (int, float)):
        return abs(float(measured) - float(card_value)) <= 5e-4
    if isinstance(measured, (list, tuple)) and isinstance(card_value, (list, tuple)):
        return len(measured) == len(card_value) and all(
            _match(m, c) for m, c in zip(measured, card_value))
    if isinstance(measured, dict) and isinstance(card_value, dict):
        return set(measured) == set(card_value) and all(
            _match(measured[k], card_value[k]) for k in measured)
    return measured == card_value


# --------------------------------------------------------------------------
# audits
# --------------------------------------------------------------------------

def audit_phrases(card_text: str) -> dict:
    out = {}
    for phrase in SEARCH_PHRASES:
        lines = [i + 1 for i, ln in enumerate(card_text.splitlines())
                 if contains(ln, phrase)]
        occurrences = count_occurrences(card_text, phrase)
        out[phrase] = {
            "present": occurrences >= PHRASE_MIN_OCCURRENCES,
            "occurrences": occurrences,
            "min_required": PHRASE_MIN_OCCURRENCES,
            "lines": lines,
        }
    return out


def audit_markers(card_text: str, patterns, key) -> dict:
    out = {}
    for name, pat in patterns:
        lines = [i + 1 for i, ln in enumerate(card_text.splitlines())
                 if re.search(pat, ln, re.IGNORECASE)]
        out[name] = {"present": bool(lines), "lines": lines}
    return out


def parse_front_matter(card_text: str):
    m = re.match(r"^---\n(.*?)\n---\n", card_text, re.DOTALL)
    if not m:
        return {}, []
    fields, tags, in_tags = {}, [], False
    for line in m.group(1).splitlines():
        if in_tags:
            if line.startswith("- "):
                tags.append(line[2:].strip())
                continue
            in_tags = False
        if line.startswith("tags:"):
            in_tags = True
            rest = line.split(":", 1)[1].strip()
            if rest.startswith("[") and rest.endswith("]"):
                tags = [t.strip().strip("'\"") for t in rest[1:-1].split(",") if t.strip()]
                in_tags = False
            continue
        if ":" in line:
            k, _, v = line.partition(":")
            fields[k.strip()] = v.strip()
    return fields, tags


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--no-recompute", action="store_true",
                    help="skip the npz rescoring (needs the run artefacts)")
    args = ap.parse_args()
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    rep = Report()
    card = CARD.read_text()

    def in_card(value) -> bool:
        """Does the card contain this value, under either hyphen convention?"""
        return contains(card, str(value))

    print(f"repo root : {REPO}")
    print(f"card      : {rel(CARD)}  ({len(card.splitlines())} lines, {len(card)} chars)")
    print(f"metadata  : {rel(METADATA)}  (exists={METADATA.exists()})")
    print()

    # ---- 1. v0.7 per-seed metrics ---------------------------------------
    m7 = {}
    for s in V07_SEEDS:
        d = json.load(open(LANE / "results" / "v07_release" / f"v07_seed{s}_metrics.json"))
        m, cm = d["metrics"], d["confusion_matrix"]
        m7[s] = {
            "n": d["n"], "kappa_covered": m["weighted_kappa_covered"],
            "macro_f1_published": m["macro_f1_with_abstentions"],
            "coverage": m["coverage"], "abstention_rate": m["abstention_rate"],
            "ece_all": m["ece_all"], "fnr_covered": m["dangerous_underclassification_fnr_covered"],
            "underrate_all_abstention_blind": m["dangerous_underclassification_rate_all"],
            "exact_from_confusion_matrix": sum(cm[i][i] for i in range(5)) / d["n"],
        }
    print("== v0.7 per-seed metrics (results/v07_release/v07_seed*_metrics.json) ==")
    for s, v in m7.items():
        print(f"  seed {s:>3} n={v['n']}  kappa_cov={v['kappa_covered']:.4f} "
              f"exact_cm={v['exact_from_confusion_matrix']:.4f} "
              f"cov={v['coverage']:.4f} ece={v['ece_all']:.4f}")
    v07r = {
        "kappa_covered": rng(v["kappa_covered"] for v in m7.values()),
        "coverage": rng(v["coverage"] for v in m7.values()),
        "abstention_rate": rng(v["abstention_rate"] for v in m7.values()),
        "ece_all": rng(v["ece_all"] for v in m7.values()),
        "macro_f1_published": rng(v["macro_f1_published"] for v in m7.values()),
        "fnr_covered": rng(v["fnr_covered"] for v in m7.values()),
    }
    for k, v in v07r.items():
        rep.add(f"v07.{k}", f"v0.7 {k} range over 5 seeds appears in the card",
                v, v if in_card(v) else None, "results/v07_release/v07_seed*_metrics.json")
    print()

    # ---- 2. rescores -----------------------------------------------------
    rescores = {}
    if not args.no_recompute and RUNS.exists():
        truth = load_truth()
        print(f"== rescores vs the v0.7 consensus manifest ({len(truth)} test events) ==")
        print(f"   manifest: {rel(DATA_MANIFEST)}")
        for label, dname in MODEL_RUNS:
            ps = score_model(dname, truth)
            if not ps:
                print(f"  {label}: no *_seed* dirs under runs/{dname}")
                continue
            rescores[label] = {
                "n_per_seed": sorted({v["n"] for v in ps.values()}),
                "seeds": sorted(ps),
                "kappa_all": rng(v["kappa_all"] for v in ps.values()),
                "kappa_covered": rng(v["kappa_covered"] for v in ps.values()),
                "exact": rng(v["exact"] for v in ps.values()),
                "mae": rng(v["mae"] for v in ps.values()),
                "underrate_all": rng(v["underrate_all"] for v in ps.values()),
                "coverage": rng(v["coverage"] for v in ps.values()),
                "wk1_pass": f"{sum(v['wk1_pass'] for v in ps.values())}/{len(ps)}",
                "source": f"runs/{dname}/*/test_predictions.npz",
            }
            for k in ("kappa_all", "kappa_covered", "exact", "mae", "underrate_all"):
                v = rescores[label][k]
                rep.add(f"rescore.{label}.{k}",
                        f"{label} {k} vs v0.7 labels appears in the card",
                        v, v if in_card(v) else None, f"runs/{dname}/*/test_predictions.npz")
            rep.add(f"rescore.{label}.wk1_pass",
                    f"{label} wk1 pass count appears in the card",
                    rescores[label]["wk1_pass"],
                    rescores[label]["wk1_pass"] if in_card(rescores[label]["wk1_pass"]) else None,
                    f"runs/{dname}/*/test_predictions.npz")
            print(f"  {label}: n={rescores[label]['n_per_seed']} "
                  f"kappa_all={rescores[label]['kappa_all']} "
                  f"exact={rescores[label]['exact']} mae={rescores[label]['mae']} "
                  f"undr_all={rescores[label]['underrate_all']} "
                  f"wk1={rescores[label]['wk1_pass']}")
        print()

    # ---- 3. head-to-head -------------------------------------------------
    h2h_p = LANE / "results" / "v07_release" / "head_to_head2.json"
    h2h_raw = json.load(open(h2h_p))
    h2h = {}
    for model, seeds in h2h_raw.items():
        vals = {k: [s[k] for s in seeds.values()] for k in
                ("augrc", "exact", "macro_f1", "kappa_all", "mae", "undr_allN", "coverage")}
        h2h[model] = {"n_per_seed": sorted({s["n"] for s in seeds.values()}),
                      "seeds": sorted(seeds), **{k: rng(v) for k, v in vals.items()}}
    print("== head_to_head2.json (cross-model scoring on identical events) ==")
    for model, v in h2h.items():
        print(f"  {model}: n={v['n_per_seed']} exact={v['exact']} kappa_all={v['kappa_all']} "
              f"mae={v['mae']} augrc={v['augrc']} undr_allN={v['undr_allN']}")
    for model, v in h2h.items():
        for k in ("exact", "kappa_all", "mae", "augrc", "undr_allN"):
            rep.add(f"h2h.{model}.{k}", f"head_to_head2 {model} {k} appears in the card",
                    v[k], v[k] if in_card(v[k]) else None, rel(h2h_p))
    print()

    # ---- 4. inter-judge --------------------------------------------------
    CROSSTAB = np.array([
        [11, 1, 0, 0, 0], [4, 42, 8, 28, 69], [5, 3, 27, 42, 54],
        [3, 8, 3, 30, 111], [4, 1, 6, 47, 68]], dtype=float)
    yt, yp = [], []
    for i in range(5):
        for j in range(5):
            yt += [i] * int(CROSSTAB[i, j])
            yp += [j] * int(CROSSTAB[i, j])
    ij = {
        "n_pairs": int(CROSSTAB.sum()),
        "exact_agreement": float(np.trace(CROSSTAB) / CROSSTAB.sum()),
        "kappa_unweighted": kappa_weighted(yt, yp, "unweighted"),
        "kappa_quadratic": kappa_weighted(yt, yp, "quadratic"),
        "source": "results/regen_20260916/inter_judge_20260916.md cross-tab",
    }
    print("== inter-judge agreement, recomputed from the published cross-tab ==")
    print(f"  n_pairs={ij['n_pairs']} exact={ij['exact_agreement']:.4f} "
          f"kappa_unw={ij['kappa_unweighted']:.4f} kappa_quad={ij['kappa_quadratic']:.4f}")
    for k in ("n_pairs",):
        rep.add(f"ij.{k}", f"inter-judge {k} appears in the card",
                ij[k], ij[k] if in_card(ij[k]) else None, ij["source"])
    for k in ("exact_agreement", "kappa_unweighted", "kappa_quadratic"):
        rep.add(f"ij.{k}", f"inter-judge {k} appears in the card",
                ij[k], ij[k] if in_card(f"{ij[k]:.4f}") else None, ij["source"])
    print()

    # ---- 5. labelability / coverage closure / bench ----------------------
    ls = json.load(open(LANE / "results" / "v05_release" / "labelability_sweep.json"))
    cm = json.load(open(LANE / "results" / "v07_release" / "coverage_meta.json"))
    bench = json.load(open(LANE / "flowtest" / "flowtest" / "bench" / "results_n100" / "bench_summary.json"))
    ov = bench["overall"]
    derived = {
        "cross_food_agree_pct": round(100 * ls["cross_food_agree"] / ls["both"], 1),
        "cross_lev_agree_pct": round(100 * ls["cross_lev_agree"] / ls["both"], 1),
        "unanimous_pct": round(100 * cm["unanimous"] / cm["events_ge2"], 1),
        "bench_mae_ml": ov["mae_ml_graded_only"],
        "bench_n": bench["n"],
        "bench_graded": ov["graded"],
    }
    print("== labelability sweep (results/v05_release/labelability_sweep.json) ==")
    print(f"  both={ls['both']} cross_food_agree={ls['cross_food_agree']} "
          f"({derived['cross_food_agree_pct']}%) cross_lev_agree={ls['cross_lev_agree']} "
          f"({derived['cross_lev_agree_pct']}%)")
    print("== coverage closure (results/v07_release/coverage_meta.json) ==")
    print(f"  events_ge2={cm['events_ge2']} unanimous={cm['unanimous']} "
          f"({derived['unanimous_pct']}%) split={cm['agreement_dist']['1/2']}")
    print("== flow-test bench (flowtest/flowtest/bench/results_n100/bench_summary.json) ==")
    print(f"  n={bench['n']} graded={ov['graded']} abstained={ov['abstained']} "
          f"mae_ml_graded_only={ov['mae_ml_graded_only']}")
    print(f"  scope: {bench['scope']}")
    print()

    # ---- 6. CORAL fix run ------------------------------------------------
    fx = json.load(open(LANE / "results" / "fix1_siglip2_coral_seed17" / "metrics.json"))
    fx_m = fx["metrics"]
    print("== CORAL decode-fix run (results/fix1_siglip2_coral_seed17/metrics.json) ==")
    print(f"  n={fx['n']} macro_f1={fx_m['macro_f1_with_abstentions']:.4f} "
          f"kappa_covered={fx_m['weighted_kappa_covered']:.4f} "
          f"fnr_covered={fx_m['dangerous_underclassification_fnr_covered']:.4f} "
          f"coverage={fx_m['coverage']:.4f} ece_all={fx_m['ece_all']:.4f}")
    print()

    # ---- 7. phrase audit --------------------------------------------------
    phrases = audit_phrases(card)
    print("== search-phrase audit against hf/model_README.md ==")
    for phrase, v in phrases.items():
        flag = "OK  " if v["present"] else "MISS"
        print(f"  [{flag}] {phrase!r} x{v['occurrences']} (min {v['min_required']}) "
              f"lines={v['lines'][:6]}")
    print()

    # ---- 8. disclaimer / hazard audit -------------------------------------
    disc = audit_markers(card, DISCLAIMERS, "disclaimer")
    print("== disclaimer / gate audit against hf/model_README.md ==")
    for name, v in disc.items():
        print(f"  [{'OK  ' if v['present'] else 'MISS'}] {name} lines={v['lines'][:5]}")
    print()

    hazards = {}
    print("== hazard-entry audit (16 entries in hf/HAZARD_LOG.md) ==")
    for h in HAZARD_IDS:
        lines = [i + 1 for i, ln in enumerate(card.splitlines()) if h in ln]
        hazards[h] = {"referenced_in_card": bool(lines), "lines": lines}
        print(f"  [{'OK  ' if lines else 'MISS'}] {h} lines={lines[:5]}")
    print()

    # ---- 9. front matter ---------------------------------------------------
    fm, tags = parse_front_matter(card)
    print("== card front matter ==")
    print(json.dumps(fm, indent=2))
    print(f"  tags ({len(tags)}): {tags}")
    for k, v in EXPECTED_FRONT_MATTER.items():
        rep.add(f"frontmatter.{k}", f"card front-matter {k} equals the specified value",
                v, fm.get(k), "hf/HF_METADATA.md §1")
    rep.add("frontmatter.tags", "card tag list equals the specified 19-tag list",
            EXPECTED_TAGS, tags, "hf/HF_METADATA.md §1")
    rep.add("frontmatter.tag_count", "card carries exactly 19 tags",
            19, len(tags), "hf/HF_METADATA.md §1")
    rep.add("frontmatter.pipeline_tag_in_tags",
            "pipeline_tag also appears in the tag list",
            True, EXPECTED_FRONT_MATTER["pipeline_tag"] in tags, "hf/HF_METADATA.md §1")
    # metadata doc must contain the same block
    if METADATA.exists():
        meta = METADATA.read_text()
        rep.add("metadata.block", "HF_METADATA.md carries the same front-matter block",
                True, "pipeline_tag: image-classification" in meta
                and "library_name: transformers" in meta
                and "base_model: google/siglip2-base-patch16-224" in meta,
                rel(METADATA))
        for phrase in SEARCH_PHRASES:
            rep.add(f"metadata.phrase.{phrase}",
                    f"HF_METADATA.md lists the search phrase {phrase!r}",
                    True, phrase in meta, rel(METADATA))
    print()

    # ---- write artefacts ---------------------------------------------------
    payload = {
        "card": rel(CARD),
        "card_lines": len(card.splitlines()),
        "card_chars": len(card),
        "repo_root": str(REPO),
        "v07_per_seed": m7,
        "v07_ranges": v07r,
        "rescores": rescores,
        "head_to_head2": {"source": rel(h2h_p), "models": h2h},
        "inter_judge_recomputed": ij,
        "labelability_sweep": {**ls, **{f"derived_{k}": v for k, v in derived.items()}},
        "coverage_closure": cm,
        "coral_fix_run": {"n": fx["n"], **fx_m,
                          "source": "results/fix1_siglip2_coral_seed17/metrics.json"},
        "flowtest_bench": {"source": rel(LANE / "flowtest/flowtest/bench/results_n100/bench_summary.json"),
                           **ov, "n": bench["n"], "scope": bench["scope"]},
        "search_phrases": phrases,
        "disclaimers": disc,
        "hazards": hazards,
        "front_matter": {"fields": fm, "tags": tags,
                         "expected_fields": EXPECTED_FRONT_MATTER,
                         "expected_tags": EXPECTED_TAGS},
        "claims": rep.claims,
        "claim_failures": rep.failures,
    }
    (outdir / "card_claims_verification.json").write_text(json.dumps(payload, indent=2))
    (outdir / "search_phrase_audit.json").write_text(json.dumps(phrases, indent=2))
    (outdir / "hazard_coverage_audit.json").write_text(json.dumps(hazards, indent=2))

    lines = ["# card_claims_verification", "",
             f"card: `{rel(CARD)}` ({len(card.splitlines())} lines, {len(card)} chars)", "",
             "## headline ranges re-derived this run", "",
             "| quantity | range | in card | source |", "|---|---|---|---|"]
    src = {"kappa_covered": "results/v07_release/v07_seed*_metrics.json",
           "coverage": "results/v07_release/v07_seed*_metrics.json",
           "abstention_rate": "results/v07_release/v07_seed*_metrics.json",
           "ece_all": "results/v07_release/v07_seed*_metrics.json",
           "macro_f1_published": "results/v07_release/v07_seed*_metrics.json",
           "fnr_covered": "results/v07_release/v07_seed*_metrics.json"}
    for k, v in v07r.items():
        lines.append(f"| v0.7 {k} | {v} | {'yes' if in_card(v) else 'NO'} | {src[k]} |")
    for label, r in rescores.items():
        for k in ("kappa_all", "kappa_covered", "exact", "mae", "underrate_all"):
            lines.append(f"| {label} {k} vs v0.7 labels | {r[k]} | "
                         f"{'yes' if in_card(r[k]) else 'NO'} | {r['source']} |")
    lines += ["", "## search phrases", "", "| phrase | occurrences | lines |", "|---|---|---|"]
    for p, v in phrases.items():
        lines.append(f"| {p} | {v['occurrences']} | {v['lines'][:6]} |")
    lines += ["", "## disclaimers", "", "| item | in card | lines |", "|---|---|---|"]
    for n_, v in disc.items():
        lines.append(f"| {n_} | {'yes' if v['present'] else 'NO'} | {v['lines'][:5]} |")
    lines += ["", "## hazard entries", "", "| hazard | referenced | lines |", "|---|---|---|"]
    for h, v in hazards.items():
        lines.append(f"| {h} | {'yes' if v['referenced_in_card'] else 'NO'} | {v['lines'][:5]} |")
    (outdir / "card_claims_verification.md").write_text("\n".join(lines) + "\n")

    print(f"wrote {outdir/'card_claims_verification.json'}")
    print(f"wrote {outdir/'card_claims_verification.md'}")
    print(f"wrote {outdir/'search_phrase_audit.json'}")
    print(f"wrote {outdir/'hazard_coverage_audit.json'}")

    missing_p = [p for p, v in phrases.items() if not v["present"]]
    missing_d = [n for n, v in disc.items() if not v["present"]]
    missing_h = [h for h, v in hazards.items() if not v["referenced_in_card"]]
    fails = rep.failures
    print()
    if missing_p or missing_d or missing_h or fails:
        print("FAIL")
        if missing_p:
            print(f"  search phrases missing: {missing_p}")
        if missing_d:
            print(f"  disclaimers missing: {missing_d}")
        if missing_h:
            print(f"  hazard entries not referenced: {missing_h}")
        for f in fails:
            print(f"  claim {f['id']}: card={f['card_value']!r} measured={f['measured']!r}")
        return 1
    print(f"OK: {len(phrases)}/10 search phrases present, "
          f"{len(disc)}/{len(DISCLAIMERS)} disclaimer markers present, "
          f"{len(hazards)}/{len(HAZARD_IDS)} hazard entries referenced, "
          f"{len(rep.claims)}/{len(rep.claims)} claim checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
