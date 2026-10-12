"""Step 1 of the decode-rule comparison: characterise what the shipped head emits.

Answers, on the VALIDATION split only (n=110, L3 14 / L4 8 / L5 6 / L6 9 / L7 73):
  - the per-seed temperature refit on this lane's own fp32 CPU predictions,
  - the CORAL interior mass ceiling at that temperature,
  - the pre-abstention decode census under the shipped rank rule,
  - the pre-abstention decode census under argmax,
  - the per-cut-point survival distribution (where the score actually lands).

Also cross-checks the shipped test artifacts: the rank decode must be unchanged.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from lane_common import (
    DISCLAIMER,
    GATE,
    LEVELS,
    PRED_DIR,
    RUN_DIR,
    SEEDS,
    coral_decode,
    level_counts,
    load_seed,
    shipped_calibration,
    survival_from_probs,
    temperature_scale,
)
from eval.calibration import fit_coral_temperature
from train.model import coral_interior_mass_ceiling

OUT = Path("/work/lane_runs/fix01_decode/analysis")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    report = {
        "lane": "fix01_decode",
        "step": "1_characterise_head",
        "disclaimer": DISCLAIMER,
        "gate": GATE,
        "seeds": SEEDS,
        "validation": {},
        "test_crosscheck": {},
        "notes": [],
    }

    val_support = None
    for seed in SEEDS:
        val = load_seed(seed, "val")
        labels = val["labels"]
        support = {f"L{int(LEVELS[l])}": int((labels == l).sum()) for l in range(5)}
        if val_support is None:
            val_support = support
        temperature, _ = fit_coral_temperature(val["logits"], labels)
        shipped = shipped_calibration(seed)
        probs = val["probabilities"]
        assert np.allclose(
            probs, temperature_scale(val["logits"], float(shipped["temperature"])), atol=1e-6, rtol=0
        ), "stored probabilities are not the shipped-temperature reconstruction"
        ceiling = coral_interior_mass_ceiling(probs)
        rank_pred, rank_conf = coral_decode(probs)
        argmax_pred = probs.argmax(axis=1)
        survival = survival_from_probs(probs)
        gaps = np.log(
            np.clip(survival, 1e-12, 1 - 1e-12)
        ) - np.log(np.clip(1.0 - survival, 1e-12, 1 - 1e-12))
        gap_medians = np.median(gaps, axis=0)

        report["validation"][str(seed)] = {
            "n": int(len(labels)),
            "support": support,
            "refit_temperature": float(temperature),
            "shipped_temperature": float(shipped["temperature"]),
            "shipped_threshold": float(shipped["threshold"]),
            "interior_mass_ceiling": float(ceiling),
            "cutpoint_gap_medians_temperature_scaled": [float(v) for v in gap_medians],
            "rank_decode_counts_pre_abstention": level_counts(rank_pred),
            "argmax_counts_pre_abstention": level_counts(argmax_pred),
            "rank_vs_argmax_disagree": int((rank_pred != argmax_pred).sum()),
            "survival_quantiles": {
                f"P(y>L{int(LEVELS[k])})": [round(float(v), 4) for v in np.quantile(survival[:, k], [0, .1, .25, .5, .75, .9, 1])]
                for k in range(survival.shape[1])
            },
            "rank_confidence_quantiles": [round(float(v), 4) for v in np.quantile(rank_conf, [0, .1, .25, .5, .75, .9, 1])],
            "max_class_prob_quantiles": [round(float(v), 4) for v in np.quantile(probs.max(axis=1), [0, .1, .25, .5, .75, .9, 1])],
        }

        # Cross-check: this lane's fp32 CPU dump must decode identically to the
        # shipped GPU/AMP artifact on all 108 test events.
        test = load_seed(seed, "test")
        shipped_npz = RUN_DIR / f"siglip2_v07_fullcoverage_seed{seed}" / "test_predictions.npz"
        shipped_probs = np.load(shipped_npz, allow_pickle=True)["probabilities"]
        mine_pred, mine_conf = coral_decode(test["probabilities"])
        ship_pred, ship_conf = coral_decode(shipped_probs)
        report["test_crosscheck"][str(seed)] = {
            "n": int(len(test["labels"])),
            "rank_decode_disagreements_vs_shipped": int((mine_pred != ship_pred).sum()),
            "probabilities_max_abs_diff_vs_shipped": float(np.abs(test["probabilities"] - shipped_probs).max()),
            "shipped_threshold": float(shipped["threshold"]),
            "decided_at_shipped_threshold": int((mine_conf >= shipped["threshold"]).sum()),
            "decided_at_shipped_threshold_in_shipped_artifact": int((ship_conf >= shipped["threshold"]).sum()),
        }

    report["validation_support_note"] = (
        "L5/L6 support on the validation split actually used: "
        f"L5 {val_support['L5']} of 110, L6 {val_support['L6']} of 110 "
        f"(L3 {val_support['L3']}, L4 {val_support['L4']}, L7 {val_support['L7']})."
    )
    (OUT / "step1_head_characterisation.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    summary = {
        "validation_support": val_support,
        "per_seed": {
            seed: {
                "refit_temperature": v["refit_temperature"],
                "shipped_temperature": v["shipped_temperature"],
                "shipped_threshold": v["shipped_threshold"],
                "interior_mass_ceiling": v["interior_mass_ceiling"],
                "rank_decode_counts_pre_abstention": v["rank_decode_counts_pre_abstention"],
                "argmax_counts_pre_abstention": v["argmax_counts_pre_abstention"],
                "rank_vs_argmax_disagree": v["rank_vs_argmax_disagree"],
            }
            for seed, v in report["validation"].items()
        },
        "test_crosscheck": report["test_crosscheck"],
        "validation_support_note": report["validation_support_note"],
    }
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
