"""Assert every claim in ``threshold_sweep_val.md`` against the JSON it came from.

Modelled on ``analysis/errors_v07/verify_findings.py`` from wave 1. Each check
reads a value out of ``threshold_sweep_val.json`` / ``threshold_sweep_val_answer.json``
/ the per-seed val NPZ and compares it with the number written in the report.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

LANE_ROOT = Path(__file__).resolve().parents[2]
TRAINER = LANE_ROOT / "trainer"
if str(TRAINER) not in sys.path:
    sys.path.insert(0, str(TRAINER))

from train.model import coral_decode  # noqa: E402

OUT = Path("/work/lane_runs/fix02_threshold")
REPORT = OUT / "threshold_sweep_val.md"
SWEEP = OUT / "threshold_sweep_val.json"
ANSWER = OUT / "threshold_sweep_val_answer.json"
SEEDS = (17, 29, 43, 71, 101)

checks: list[tuple[str, bool, str]] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    checks.append((name, bool(condition), detail))


def main() -> None:
    sweep = json.loads(SWEEP.read_text())
    answer = json.loads(ANSWER.read_text())
    report = REPORT.read_text()
    seeds = {seed["seed"]: seed for seed in sweep["seeds"]}
    answers = {entry["seed"]: entry for entry in answer["seeds"]}

    # ---- denominators -------------------------------------------------
    for seed in SEEDS:
        check(f"seed {seed}: n_val == 110", seeds[seed]["n_val"] == 110)
        check(
            f"seed {seed}: val support L3/L4/L5/L6/L7 == 14/8/6/9/73",
            [seeds[seed]["val_level_support"][k] for k in ("L3", "L4", "L5", "L6", "L7")]
            == [14, 8, 6, 9, 73],
        )
    check("report states 110 validation events", "**110**" in report)
    check("report states L5 support 6", "**L5 6**" in report)
    check("report states L6 support 9", "**L6 9**" in report)
    check("report states the test split is never loaded", "never loaded" in report)

    # ---- section 1: pre-abstention census ------------------------------
    for seed in SEEDS:
        npz = np.load(OUT / f"val_predictions_seed{seed}_cpu.npz", allow_pickle=True)
        probabilities = npz["probabilities"].astype(np.float64)
        labels = npz["labels"].astype(np.int64)
        predictions, _ = coral_decode(probabilities)
        census = {f"L{3 + i}": int((predictions == i).sum()) for i in range(5)}
        check(f"seed {seed}: census matches the npz", census == seeds[seed]["decoded_level_counts_pre_abstention"])
        check(f"seed {seed}: L5 decoded pre-abstention > 0", census["L5"] > 0, str(census))
        check(f"seed {seed}: L6 decoded pre-abstention > 0", census["L6"] > 0, str(census))
        # the interior total must equal L4 + L5 + L6 decoded
        check(
            f"seed {seed}: interior_decodes_total == L4+L5+L6",
            seeds[seed]["interior_decodes_total"] == census["L4"] + census["L5"] + census["L6"],
        )

    # ---- section 1b: no true-L5 event decodes as L5 ---------------------
    for seed in SEEDS:
        npz = np.load(OUT / f"val_predictions_seed{seed}_cpu.npz", allow_pickle=True)
        probabilities = npz["probabilities"].astype(np.float64)
        labels = npz["labels"].astype(np.int64)
        predictions, _ = coral_decode(probabilities)
        true_l5 = labels == 2
        check(
            f"seed {seed}: zero true-L5 val events decode as L5",
            int((predictions[true_l5] == 2).sum()) == 0,
        )
        check(f"seed {seed}: six true-L5 val events", int(true_l5.sum()) == 6)

    # ---- section 2: shipped threshold above the ceiling -----------------
    for seed in SEEDS:
        row = seeds[seed]["shipped_threshold_row"]
        check(
            f"seed {seed}: shipped threshold exceeds the interior mass ceiling",
            row["exceeds_interior_mass_ceiling"] is True,
            f"threshold {row['threshold']} vs ceiling {seeds[seed]['interior_mass_ceiling']}",
        )
    check(
        "seeds 29 and 101 accept zero interior decodes at the shipped threshold",
        all(
            seeds[s]["shipped_threshold_row"]["interior_decodes_accepted"] == 0 for s in (29, 101)
        ),
    )
    check(
        "seeds 29 and 101 shipped thresholds are > 2x their ceiling",
        all(
            seeds[s]["shipped_threshold"] > 2 * seeds[s]["interior_mass_ceiling"] for s in (29, 101)
        ),
    )

    # ---- section 3/4/5: coverage is not the cost ------------------------
    for seed in SEEDS:
        boundary = [
            row
            for row in seeds[seed]["rows"]
            if row["emitted_level_counts"].get("L5", 0) > 0
            and row["emitted_level_counts"].get("L6", 0) > 0
        ]
        check(f"seed {seed}: some threshold emits both L5 and L6", bool(boundary))
        best = max(boundary, key=lambda row: row["threshold"])
        shipped = seeds[seed]["shipped_threshold_row"]
        check(
            f"seed {seed}: reachability boundary coverage >= shipped coverage",
            best["coverage"] >= shipped["coverage"],
            f"{best['coverage']} vs {shipped['coverage']}",
        )
        check(
            f"seed {seed}: boundary covered FNR is above the 0.05 gate unless seed 43",
            (best["dangerous_underclassification_fnr_covered"] > 0.05)
            or (seed == 43 and best["dangerous_underclassification_fnr_covered"] <= 0.05),
        )
    check(
        "threshold 0 gives coverage 1.0 for every seed",
        all(seeds[s]["rows"][0]["coverage"] == 1.0 for s in SEEDS),
    )
    check(
        "threshold 0 emits L5 and L6 for every seed",
        all(
            seeds[s]["rows"][0]["emitted_level_counts"].get("L5", 0) > 0
            and seeds[s]["rows"][0]["emitted_level_counts"].get("L6", 0) > 0
            for s in SEEDS
        ),
    )

    # ---- section 7: L5 hits are identically zero ------------------------
    for seed in SEEDS:
        hits = {row["per_level_recall_covered"]["L5"]["hits"] for row in seeds[seed]["rows"]}
        check(f"seed {seed}: L5 hit set == {{0}} across the whole grid", hits == {0}, str(hits))

    # ---- section 4b: gate-meeting interior-emitting thresholds ----------
    for seed in SEEDS:
        entry = answers[seed]
        cheapest = entry["answer"]["cheapest_fnr_covered_among_L5_or_L6_emitting"]
        check(
            f"seed {seed}: cheapest covered FNR among L5/L6-emitting rows matches the report",
            cheapest is not None
            and abs(cheapest["dangerous_underclassification_fnr_covered"]
                    - {17: 0.0581, 29: 0.0659, 43: 0.0455, 71: 0.0722, 101: 0.0833}[seed]) < 5e-5,
            str(cheapest["dangerous_underclassification_fnr_covered"]) if cheapest else "none",
        )
    check(
        "only seed 43 has a gate-meeting interior-emitting threshold",
        [answers[s]["answer"]["n_L5_or_L6_emitting_thresholds_meeting_the_0_05_gate"] for s in SEEDS]
        == [0, 0, 3, 0, 0],
    )

    # ---- section 9: temperature invariance ------------------------------
    import torch  # noqa: PLC0415

    from train.model import coral_logits_to_probs  # noqa: PLC0415

    for seed in SEEDS:
        npz = np.load(OUT / f"val_predictions_seed{seed}_cpu.npz", allow_pickle=True)
        logits = npz["cutpoint_logits"].astype(np.float64)
        temperature = float(npz["temperature"])
        uncalibrated = coral_logits_to_probs(torch.from_numpy(logits)).numpy()
        calibrated = coral_logits_to_probs(torch.from_numpy(logits / temperature)).numpy()
        raw_pred, _ = coral_decode(uncalibrated)
        cal_pred, _ = coral_decode(calibrated)
        check(f"seed {seed}: decode is temperature-invariant", bool((raw_pred == cal_pred).all()))

    # ---- section 10: reproduction fidelity ------------------------------
    reference = np.load("/work/lane_runs/an03_calibration/val_predictions_seed43.npz", allow_pickle=True)
    mine = np.load(OUT / "val_predictions_seed43_cpu.npz", allow_pickle=True)
    check(
        "cut-point logits are bit-identical to the GPU reference",
        np.array_equal(
            np.asarray(reference["cutpoint_logits"]), np.asarray(mine["cutpoint_logits"])
        ),
    )
    ref_pred, _ = coral_decode(np.asarray(reference["probabilities"], dtype=np.float64))
    my_pred, _ = coral_decode(np.asarray(mine["probabilities"], dtype=np.float64))
    check("decoded levels identical to the GPU reference", bool((ref_pred == my_pred).all()))
    check(
        "labels identical to the GPU reference",
        np.array_equal(np.asarray(reference["labels"]), np.asarray(mine["labels"])),
    )
    for seed, shipped_coverage, recomputed in (
        (17, 0.7727272727272727, 0.7727272727272727),
        (29, 0.7727272727272727, 0.7636363636363637),
        (43, 0.8, 0.7909090909090909),
        (71, 0.8, 0.8),
        (101, 0.7545454545454545, 0.7545454545454545),
    ):
        check(
            f"seed {seed}: recomputed coverage at the shipped threshold is {recomputed:.4f}",
            abs(seeds[seed]["shipped_threshold_row"]["coverage"] - recomputed) < 1e-9,
        )
        check(
            f"seed {seed}: shipped calibration.json coverage is {shipped_coverage:.4f}",
            abs(
                json.loads(
                    (
                        Path(f"/work/runs/v07/siglip2_v07_fullcoverage_seed{seed}/calibration.json")
                    ).read_text()
                )["coverage"]
                - shipped_coverage
            )
            < 1e-9,
        )

    # ---- the shipped selector with the ceiling clip ---------------------
    from eval.calibration import calibrate_abstention_threshold  # noqa: PLC0415

    expected = {17: 0.230243, 29: 0.222890, 43: 0.237198, 71: 0.222451, 101: 0.223129}
    for seed in SEEDS:
        npz = np.load(OUT / f"val_predictions_seed{seed}_cpu.npz", allow_pickle=True)
        probabilities = npz["probabilities"].astype(np.float64)
        labels = npz["labels"].astype(np.int64)
        reselect = calibrate_abstention_threshold(probabilities, labels, 0.05, 0.30)
        check(
            f"seed {seed}: clipped selector moves the threshold to {expected[seed]:.6f}",
            abs(float(reselect["threshold"]) - expected[seed]) < 1e-6,
            str(reselect["threshold"]),
        )
        check(f"seed {seed}: clipped selector reports target_met false", reselect["target_met"] is False)

    # ---- grid exhaustiveness -------------------------------------------
    for seed in SEEDS:
        rows = seeds[seed]["rows"]
        thresholds = [row["threshold"] for row in rows]
        check(f"seed {seed}: candidate grid is sorted and unique", thresholds == sorted(set(thresholds)))
        check(
            f"seed {seed}: grid starts at 0 and ends above every confidence",
            rows[0]["threshold"] == 0.0 and rows[-1]["threshold"] > seeds[seed]["confidence_max"],
        )
        check(
            f"seed {seed}: lowest threshold accepts everything",
            rows[0]["accepted"] == 110 and rows[0]["coverage"] == 1.0,
        )
        check(
            f"seed {seed}: highest threshold accepts nothing",
            rows[-1]["accepted"] == 0 and rows[-1]["coverage"] == 0.0,
        )

    # ---- report content -------------------------------------------------
    for phrase in (
        "## Answer, in one line",
        "Yes for reachability, no for recall",
        "not one of the six",
        "Research demonstration for culinary education",
        "does not perform official IDDSI tests",
        "never replaces",
    ):
        check(f"report contains {phrase!r}", phrase.lower() in report.lower())

    failed = [entry for entry in checks if not entry[1]]
    for name, ok, detail in checks:
        print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"   [{detail}]" if detail and not ok else ""))
    print(f"\n{len(checks) - len(failed)}/{len(checks)} checks passed")
    if failed:
        print(f"{len(failed)} FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
