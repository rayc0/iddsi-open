#!/usr/bin/env python3
"""Benchmark framing for the EdUHK / Frontiers in Nutrition external anchor
(doi 10.3389/fnut.2026.1829703; cited in this repo at hf/model_README.md:149-155
and paper/paper.md:77,134-135).

Reproduces, from files on disk, OUR exact-match % and MAE on the SAME BASIS the
external anchor reports - exact level match over all covered test events - for
every v0.7 seed, alongside:

  * the alternative denominators (all-N, abstentions scored as misses),
  * the decode variants, which is how the number currently printed in the
    release card against this anchor was actually produced,
  * the class-prior baselines on the identical 108 events, which is the only
    fair way to read any exact-match number off a 5-level, 65%-L7 test set.

Inputs (read-only):
  <root>/data/manifest_v07_weighted.jsonl                            ground truth
  <root>/runs/v07/siglip2_v07_fullcoverage_seed*/test_predictions.npz
  <root>/runs/v07/siglip2_v07_fullcoverage_seed*/test_structured_outputs.json
  <root>/runs/v07/siglip2_v07_fullcoverage_seed*/metrics.json        (self-check)

Output: <root>/lane_runs/an05_external/benchmark_framing.json
CPU-only. No GPU, no network, no model weights.
"""
from __future__ import annotations

import os
import json
from pathlib import Path

import numpy as np

SEEDS = [17, 29, 43, 71, 101]
LEVELS = np.array([3, 4, 5, 6, 7], dtype=np.int64)
BOOT = 10000
BOOT_SEED = 20261012

# /work is the container mount of the host's ~/iddsi. Resolve either so the
# script runs on the host (CPU-only) or inside `iddsi-train`.
def _find_root() -> Path | None:
    """Locate the workspace root from the environment or the working directory.

    No host path is baked in: the shipped tool runs wherever IDDSI_ROOT points,
    or from a checkout that contains data/manifest_v07_weighted.jsonl.
    """
    candidates = []
    if os.environ.get("IDDSI_ROOT"):
        candidates.append(Path(os.environ["IDDSI_ROOT"]))
    candidates += [Path.cwd(), Path.cwd().parent]
    return next((p for p in candidates
                 if (p / "data/manifest_v07_weighted.jsonl").exists()), None)


ROOT = _find_root()
if ROOT is None:
    raise SystemExit("cannot locate the v0.7 manifest — set IDDSI_ROOT")
MANIFEST = ROOT / "data/manifest_v07_weighted.jsonl"
RUNS = ROOT / "runs/v07"
OUTDIR = ROOT / "lane_runs/an05_external"
OUTDIR.mkdir(parents=True, exist_ok=True)
OUT = OUTDIR / "benchmark_framing.json"


def load_manifest() -> tuple[list[dict], dict[str, int]]:
    rows = [json.loads(line) for line in MANIFEST.read_text().splitlines() if line.strip()]
    test = [r for r in rows if r["split"] == "test"]
    return rows, {r["event_id"]: int(r["level"]) for r in test}


def coral_decode(probs: np.ndarray) -> np.ndarray:
    """level = #{k : P(y > k) > 0.5}, the ordinal rank rule (trainer/train/model.py)."""
    cum = np.cumsum(probs, axis=1)[:, :-1]  # P(y <= k), k = 0..3
    return LEVELS[(1.0 - cum > 0.5).sum(axis=1)]


def boot_mean_ci(values, alpha: float = 0.05) -> list[float]:
    arr = np.asarray(values, dtype=float)
    if arr.size < 2:
        return [float("nan"), float("nan")]
    rng = np.random.default_rng(BOOT_SEED)
    idx = rng.integers(0, arr.size, size=(BOOT, arr.size))
    lo, hi = np.quantile(arr[idx].mean(axis=1), [alpha / 2, 1 - alpha / 2])
    return [float(lo), float(hi)]


def boot_prop_ci(successes: np.ndarray, alpha: float = 0.05) -> list[float]:
    """Percentile bootstrap over Bernoulli events (the exact-match rate)."""
    successes = np.asarray(successes, dtype=float)
    if successes.size < 2:
        return [float("nan"), float("nan")]
    rng = np.random.default_rng(BOOT_SEED)
    idx = rng.integers(0, successes.size, size=(BOOT, successes.size))
    lo, hi = np.quantile(successes[idx].mean(axis=1) * 100.0, [alpha / 2, 1 - alpha / 2])
    return [float(lo), float(hi)]


def main() -> None:
    rows, truth_map = load_manifest()
    test_rows = [r for r in rows if r["split"] == "test"]
    train_rows = [r for r in rows if r["split"] == "train"]
    n_test = len(test_rows)
    truth_all = np.array([truth_map[r["event_id"]] for r in test_rows], dtype=np.int64)

    per_seed, self_checks = [], []
    for seed in SEEDS:
        run = RUNS / f"siglip2_v07_fullcoverage_seed{seed}"
        npz = np.load(run / "test_predictions.npz", allow_pickle=False)
        structured = {s["event_id"]: s for s in json.loads((run / "test_structured_outputs.json").read_text())}
        published = json.loads((run / "metrics.json").read_text())

        event_ids = [str(e) for e in npz["event_ids"]]
        # npz stores 0-indexed class indices (0..4); manifest/structured carry real
        # IDDSI levels (3..7). An off-by-three here silently inverts every level,
        # so it is asserted rather than assumed.
        labels = npz["labels"].astype(np.int64) + 3
        probs = npz["probabilities"].astype(float)

        assert set(event_ids) == set(truth_map), f"seed {seed}: event-id mismatch vs manifest"
        assert len(event_ids) == n_test, f"seed {seed}: n != {n_test}"
        for eid, lab in zip(event_ids, labels):
            assert truth_map[eid] == int(lab), f"seed {seed}: npz label != manifest label for {eid}"
        assert set(structured) == set(event_ids), f"seed {seed}: structured/npz id mismatch"

        conf = np.array([structured[e]["confidence"] for e in event_ids], dtype=float)
        thresholds = {structured[e]["calibrated_threshold"] for e in event_ids}
        assert len(thresholds) == 1, f"seed {seed}: threshold varies within a seed"
        threshold = float(thresholds.pop())
        covered = conf >= threshold
        assert int(covered.sum()) == sum(1 for s in structured.values() if s["decision"] != "unclear"), (
            f"seed {seed}: covered mask disagrees with the decision field"
        )

        pred_coral = coral_decode(probs)
        pred_argmax = LEVELS[probs.argmax(axis=1)]
        truth = labels

        correct_cov = pred_coral[covered] == truth[covered]

        # ---- class-prior baselines, evaluated on the identical covered subset ----
        cov_truth = truth[covered]
        prior_counts = np.array([(cov_truth == lv).sum() for lv in LEVELS], dtype=float)
        prior_baseline_cov = float(prior_counts.max() / prior_counts.sum())  # always-L7 on covered
        prior_sample_cov = float(((prior_counts / prior_counts.sum()) ** 2).sum())  # draw from the prior
        mae_prior_cov = float(np.abs(cov_truth.max() - cov_truth).mean())

        # ---- self-check: our recomputation must reproduce the published metrics ----
        from sklearn.metrics import cohen_kappa_score

        kappa = float(cohen_kappa_score(truth[covered], pred_coral[covered], weights="quadratic"))
        self_checks.append(
            {
                "seed": seed,
                "recomputed_weighted_kappa_covered": kappa,
                "published_weighted_kappa_covered": published["metrics"]["weighted_kappa_covered"],
                "abs_diff": abs(kappa - published["metrics"]["weighted_kappa_covered"]),
                "recomputed_coverage": float(covered.mean()),
                "published_coverage": published["metrics"]["coverage"],
            }
        )

        per_class = {}
        for lv in LEVELS:
            m = covered & (truth == lv)
            per_class[str(lv)] = {
                "n_covered": int(m.sum()),
                "exact": float((pred_coral[m] == truth[m]).mean()) if m.any() else None,
            }
        macro_exact_cov = float(np.mean([v["exact"] for v in per_class.values() if v["exact"] is not None]))

        per_seed.append(
            {
                "seed": seed,
                "n_test_events": n_test,
                "threshold": threshold,
                "n_covered": int(covered.sum()),
                "n_abstained": int((~covered).sum()),
                "coverage": float(covered.mean()),
                # ===== the basis the external anchor reports =====
                "exact_match_pct_covered": 100.0 * float(correct_cov.mean()),
                "exact_match_pct_covered_ci95": boot_prop_ci(correct_cov),
                "mae_covered_levels": float(np.abs(pred_coral[covered] - truth[covered]).mean()),
                # ===== the same events under the other honest denominators =====
                "exact_match_pct_allN_abstain_as_miss": 100.0 * float(correct_cov.sum()) / n_test,
                "exact_match_pct_allN_abstain_as_miss_ci95": boot_prop_ci(
                    np.concatenate([correct_cov, np.zeros(int((~covered).sum()))])
                ),
                "mae_allN_abstain_as_miss_levels": float(np.abs(pred_coral[covered] - truth[covered]).mean()),
                # ===== decode variants: which rule produced the printed number =====
                "exact_match_pct_argmax_allN": 100.0 * float((pred_argmax == truth).mean()),
                "mae_argmax_allN_levels": float(np.abs(pred_argmax - truth).mean()),
                "exact_match_pct_coral_allN": 100.0 * float((pred_coral == truth).mean()),
                "mae_coral_allN_levels": float(np.abs(pred_coral - truth).mean()),
                # ===== imbalance-robust view =====
                "macro_exact_match_pct_covered": 100.0 * macro_exact_cov,
                "per_class_covered": per_class,
                # ===== baselines on the identical covered subset =====
                "baseline_always_majority_covered": {
                    "exact_match_pct": 100.0 * prior_baseline_cov,
                    "mae_levels": mae_prior_cov,
                },
                "baseline_sample_from_covered_prior_pct": 100.0 * prior_sample_cov,
            }
        )

    # ---------------- baselines over the full 108-event test split ----------------
    def const_baseline(level: int) -> dict:
        pred = np.full(n_test, level, dtype=np.int64)
        return {
            "constant_level": int(level),
            "exact_match_pct": 100.0 * float((pred == truth_all).mean()),
            "mae_levels": float(np.abs(pred - truth_all).mean()),
        }

    train_counts = {int(lv): sum(1 for r in train_rows if int(r["level"]) == lv) for lv in LEVELS}
    modal_train = max(train_counts, key=train_counts.get)
    test_counts = {int(lv): int((truth_all == lv).sum()) for lv in LEVELS}
    prior_probs = np.array([test_counts[int(lv)] for lv in LEVELS], dtype=float) / n_test

    baselines_allN = {f"always_L{lv}": const_baseline(int(lv)) for lv in LEVELS}
    baselines_allN["modal_train_level"] = const_baseline(int(modal_train))
    baselines_allN["sample_from_test_prior"] = {
        "exact_match_pct": 100.0 * float((prior_probs**2).sum()),
        "mae_levels": float("nan"),
        "note": "expected exact match of a sampler that draws labels from the empirical test prior",
    }

    # ---------------- pooled across all 5 seeds ----------------
    pc = pn = 0
    pa = 0.0
    for seed in SEEDS:
        run = RUNS / f"siglip2_v07_fullcoverage_seed{seed}"
        npz = np.load(run / "test_predictions.npz", allow_pickle=False)
        structured = {s["event_id"]: s for s in json.loads((run / "test_structured_outputs.json").read_text())}
        ids = [str(e) for e in npz["event_ids"]]
        lab = npz["labels"].astype(np.int64) + 3
        pr = coral_decode(npz["probabilities"].astype(float))
        cf = np.array([structured[e]["confidence"] for e in ids])
        th = np.array([structured[e]["calibrated_threshold"] for e in ids])[0]
        cv = cf >= th
        pc += int((pr[cv] == lab[cv]).sum())
        pn += int(cv.sum())
        pa += float(np.abs(pr[cv] - lab[cv]).sum())
    pooled = {
        "n_covered_events_across_5_seeds": pn,
        "exact_match_pct_covered": 100.0 * pc / pn,
        "exact_match_pct_covered_ci95": boot_prop_ci(
            np.concatenate([[1.0] * pc, [0.0] * (pn - pc)])
        ),
        "mae_covered_levels": pa / pn,
    }

    def rng(vals):
        v = list(vals)
        return [min(v), max(v)]

    # ---- audit: which decode rule produced the row the release card prints ----
    # hf/model_README.md:321-322 prints, for v0.7, "exact match 0.694 - 0.741" and
    # "MAE in IDDSI levels 0.380 - 0.444" next to the EdUHK anchor. Those ranges
    # are matched here against every candidate rule so the provenance of the
    # printed number is a measurement rather than an assumption.
    printed = {"exact_match_pct": [69.4, 74.1], "mae_levels": [0.380, 0.444]}
    candidates = {
        "coral_over_all_108_events_abstention_bypassed": {
            "exact_match_pct": rng([s["exact_match_pct_coral_allN"] for s in per_seed]),
            "mae_levels": rng([s["mae_coral_allN_levels"] for s in per_seed]),
        },
        "argmax_over_all_108_events_abstention_bypassed": {
            "exact_match_pct": rng([s["exact_match_pct_argmax_allN"] for s in per_seed]),
            "mae_levels": rng([s["mae_argmax_allN_levels"] for s in per_seed]),
        },
        "coral_over_covered_events_only": {
            "exact_match_pct": rng([s["exact_match_pct_covered"] for s in per_seed]),
            "mae_levels": rng([s["mae_covered_levels"] for s in per_seed]),
        },
        "coral_over_all_108_events_abstention_scored_as_miss": {
            "exact_match_pct": rng([s["exact_match_pct_allN_abstain_as_miss"] for s in per_seed]),
            "mae_levels": rng([s["mae_covered_levels"] for s in per_seed]),
        },
    }

    # the release card prints fractions rounded to 3 dp, i.e. 0.1 pp
    def matches(lo_hi, target, tol=0.05):
        return abs(lo_hi[0] - target[0]) <= tol and abs(lo_hi[1] - target[1]) <= tol

    decode_audit = {
        "printed_in_release_card_hf_model_README_md_lines_321_322": printed,
        "candidates": candidates,
        "verdict": {
            name: {
                "exact_match_matches_printed": matches(v["exact_match_pct"], printed["exact_match_pct"]),
                "mae_matches_printed": matches(v["mae_levels"], printed["mae_levels"]),
            }
            for name, v in candidates.items()
        },
    }

    # ---- is the mislabelled basis systemic, or only in the v0.7 row? ----
    # hf/model_README.md Finding 3 prints, for the v0.5 model trained on the
    # cleaned 314-row corpus, "exact 0.426-0.500 | kappa 0.498-0.547 |
    # MAE 0.778-0.870" on the identical 108-event test set. Those runs are still
    # on disk, so the same audit can be run on them.
    v05 = {}
    for seed in SEEDS:
        run = ROOT / "runs/v05" / f"siglip2_v05_clean_seed{seed}"
        if not (run / "test_predictions.npz").exists():
            continue
        npz = np.load(run / "test_predictions.npz", allow_pickle=False)
        lab = npz["labels"].astype(np.int64) + 3
        pr = coral_decode(npz["probabilities"].astype(float))
        am = LEVELS[npz["probabilities"].astype(float).argmax(axis=1)]
        v05[seed] = {
            "n": int(len(lab)),
            "exact_coral_allN_pct": 100.0 * float((pr == lab).mean()),
            "mae_coral_allN_levels": float(np.abs(pr - lab).mean()),
            "exact_argmax_allN_pct": 100.0 * float((am == lab).mean()),
            "mae_argmax_allN_levels": float(np.abs(am - lab).mean()),
        }
    if v05:
        decode_audit["v05_cleaned_314row_model_cross_check"] = {
            "printed_in_release_card": {"exact_match_pct": [42.6, 50.0], "mae_levels": [0.778, 0.870]},
            "measured": v05,
            "verdict": {
                "coral_allN_matches_printed": matches(
                    rng([v["exact_coral_allN_pct"] for v in v05.values()]), [42.6, 50.0], tol=0.06
                )
                and matches(rng([v["mae_coral_allN_levels"] for v in v05.values()]), [0.778, 0.870], tol=0.0006),
                "argmax_allN_matches_printed": matches(
                    rng([v["exact_argmax_allN_pct"] for v in v05.values()]), [42.6, 50.0], tol=0.06
                ),
            },
        }

    summary = {
        "n_test_events": n_test,
        "test_level_distribution": {str(k): v for k, v in test_counts.items()},
        "train_level_distribution": {str(k): v for k, v in train_counts.items()},
        "external_anchor_as_cited_in_this_repo": {
            "source_file": "hf/model_README.md:149-155",
            "doi": "10.3389/fnut.2026.1829703",
            "exact_match_pct_qwen": 61.74,
            "exact_match_pct_gpt41": 70.43,
            "mae_levels": 0.809,
            "fnr_pct": 9.2,
            "n_images": 115,
            "n_levels": 8,
            "n_vlms": 14,
            "n_web_sourced_images": 50,
        },
        "ours_same_basis": {
            "definition": "exact IDDSI level match over all covered (non-abstained) test events",
            "exact_match_pct_covered": rng([s["exact_match_pct_covered"] for s in per_seed]),
            "mae_covered_levels": rng([s["mae_covered_levels"] for s in per_seed]),
            "coverage_pct": rng([100.0 * s["coverage"] for s in per_seed]),
            "n_covered_range": rng([s["n_covered"] for s in per_seed]),
        },
        "ours_alternative_denominators": {
            "exact_match_pct_allN_abstain_as_miss": rng(
                [s["exact_match_pct_allN_abstain_as_miss"] for s in per_seed]
            ),
            "exact_match_pct_argmax_allN": rng([s["exact_match_pct_argmax_allN"] for s in per_seed]),
            "mae_argmax_allN_levels": rng([s["mae_argmax_allN_levels"] for s in per_seed]),
            "macro_exact_match_pct_covered": rng([s["macro_exact_match_pct_covered"] for s in per_seed]),
        },
        "pooled_all_5_seeds": pooled,
        "baselines_on_the_same_108_events": baselines_allN,
        "decode_variant_audit": decode_audit,
    }

    OUT.write_text(
        json.dumps(
            {
                "provenance": {
                    "manifest": str(MANIFEST),
                    "runs": str(RUNS),
                    "seeds": SEEDS,
                    "decode": "CORAL ordinal rule level = #{k : P(y>k) > 0.5} over LEVELS=[3,4,5,6,7]",
                    "covered": "confidence >= calibrated_threshold, else the event is an abstention",
                    "bootstrap": f"{BOOT} resamples, seed {BOOT_SEED}, percentile 95%",
                    "gpu": "none - CPU only",
                },
                "self_checks": self_checks,
                "per_seed": per_seed,
                "summary": summary,
            },
            indent=2,
        )
        + "\n"
    )
    print(json.dumps(summary, indent=2))
    print("\nself-check max |recomputed - published| weighted kappa:",
          max(c["abs_diff"] for c in self_checks))


if __name__ == "__main__":
    main()
