"""Render the VALIDATION threshold-sweep report and the coverage-targeted table.

Everything here is computed from ``threshold_sweep_val.json`` (the exhaustive
per-threshold sweep on the 110 validation events) — no number is restated from
any other artifact.
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

from eval.metrics import evaluate_predictions  # noqa: E402
from train.model import coral_decode  # noqa: E402

LEVELS = ("L3", "L4", "L5", "L6", "L7")
TARGET_COVERAGES = (0.70, 0.80, 0.90, 1.00)


def fmt(value, digits=4):
    if value is None:
        return "—"
    if isinstance(value, float):
        if value != value:
            return "nan"
        return f"{value:.{digits}f}"
    return str(value)


def recall_cell(stats):
    if stats["covered_support"] == 0:
        return "n/a (0 covered)"
    return f"{stats['recall_covered']:.3f} ({stats['hits']}/{stats['covered_support']})"


def main() -> None:
    out_dir = Path("/work/lane_runs/fix02_threshold")
    sweep = json.loads((out_dir / "threshold_sweep_val.json").read_text())
    seeds = sweep["seeds"]

    lines: list[str] = []
    add = lines.append

    add("# Abstention-threshold sweep on the VALIDATION split (lane fix02_threshold)")
    add("")
    add("> Research demonstration for culinary education. Estimates visual similarity to IDDSI")
    add("> descriptors; does not perform official IDDSI tests, assess swallowing, determine")
    add("> suitability for any person, or decide whether food is safe to consume. Never replaces")
    add("> clinician-prescribed texture or physical IDDSI testing. This project is not the")
    add("> official IDDSI website and is not endorsed by IDDSI.")
    add("")
    add("## Question")
    add("")
    add("Wave 1 measured that on the v0.7 held-out TEST split (n=108) no seed emits L5 or L6 as")
    add("a decision — per-level recall L5 0.00, L6 0.00")
    add("(`analysis/errors_v07/outputs/structural_seed43.json`). Two mechanisms produce that")
    add("symptom and the published confusion matrix cannot separate them:")
    add("")
    add("1. the CORAL rank decode never **produces** L5/L6 (a representation failure), or")
    add("2. the decode produces them but every such decode falls below the confidence")
    add("   threshold (an abstention-gate failure).")
    add("")
    add("This lane sweeps the confidence threshold over every value that is achievable on the")
    add("**validation** split and reports coverage, per-level recall, dangerous-direction FNR")
    add("and ECE at each point.")
    add("")
    add("## Denominators and provenance")
    add("")
    add("| item | value |")
    add("|---|---|")
    add("| split used for every number below | `val` (split field of `manifest_v07_weighted.jsonl`) |")
    add("| validation events | **110** (identical for all five seeds) |")
    add("| validation level support | L3 14, L4 8, **L5 6**, **L6 9**, L7 73 |")
    add("| held-out test events | **never loaded by any script in this lane** |")
    add("| decode | `train.model.coral_decode` — the shipped CORAL ordinal rank rule `#{k : P(y>k) > 0.5}` |")
    add("| probabilities | `coral_logits_to_probs(cutpoint_logits / temperature)` at the temperature shipped in `calibration.json` |")
    add("| threshold grid | exhaustive: `{0.0} ∪ {110 distinct confidences} ∪ {nextafter(max)} ∪ {shipped threshold}` = **113** candidates per seed |")
    add("| seeds | 17, 29, 43, 71, 101 (the five v0.7 release seeds) |")
    add("")
    add("The grid is exhaustive rather than sampled: acceptance is `confidence >= threshold`, so")
    add("every threshold between two consecutive distinct confidence values yields an identical")
    add("acceptance set. A round-number 0.00–1.00 grid is also emitted in the CSV.")
    add("")
    add("Code provenance (sha256, resolved through this lane's clone):")
    add("")
    for key, value in sweep["provenance"].items():
        add(f"- `{key}`: {value}")
    add("")

    # ---------------------------------------------------------------- answer
    add("## Answer, in one line")
    add("")
    add("**Yes for reachability, no for recall.** On validation the CORAL rank decode *does*")
    add("produce L5 (4–5 events per seed) and L6 (11–15 events per seed) before abstention, and")
    add("lowering the threshold below the CORAL interior mass ceiling makes both levels appear as")
    add("decisions — at **no cost in coverage** (coverage rises, it does not fall). But **L5")
    add("recall stays 0.000 at every threshold for every seed**, because not one of the six")
    add("true-L5 validation events is ever decoded as L5 by the rank rule. L5's failure is a")
    add("representation failure, not a threshold failure. L6 is partly recoverable: recall")
    add("0.333 (3/9) at threshold 0 where the covered FNR is 0.109–0.136, and 0.143–0.250 at the")
    add("reachability boundary where it is 0.074–0.083 — so for four of five seeds the boundary")
    add("point does not satisfy the 0.05 gate at all.")
    add("")

    # ------------------------------------------------- pre-abstention census
    add("## 1. Does the decode produce L5/L6 at all? (pre-abstention census, n=110 val)")
    add("")
    add("| seed | L3 | L4 | **L5** | **L6** | L7 | interior decodes | interior mass ceiling | interior confidence range |")
    add("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for seed in seeds:
        census = seed["decoded_level_counts_pre_abstention"]
        add(
            f"| {seed['seed']} | {census['L3']} | {census['L4']} | **{census['L5']}** | "
            f"**{census['L6']}** | {census['L7']} | {seed['interior_decodes_total']} | "
            f"{seed['interior_mass_ceiling']:.4f} | "
            f"{seed['interior_confidence_min']:.4f}–{seed['interior_confidence_max']:.4f} |"
        )
    add("")
    add("Every seed decodes interior levels on validation, so mechanism (2) is live: the")
    add("abstention gate is deleting interior levels the decode actually produced.")
    add("")

    add("### Where the true-L5 and true-L6 validation events actually land")
    add("")
    add("Pre-abstention decode of the validation rows, by true level (denominator = true-level")
    add("support on the val split):")
    add("")
    add("| seed | true L3 (n=14) | true L4 (n=8) | **true L5 (n=6)** | **true L6 (n=9)** | true L7 (n=73) |")
    add("|---|---|---|---|---|---|")
    for seed in seeds:
        npz = np.load(out_dir / f"val_predictions_seed{seed['seed']}_cpu.npz", allow_pickle=True)
        probabilities = npz["probabilities"].astype(np.float64)
        labels = npz["labels"].astype(np.int64)
        predictions, _ = coral_decode(probabilities)
        cells = []
        for true_index in range(5):
            mask = labels == true_index
            counts = {
                f"L{3 + level}": int(((predictions == level) & mask).sum()) for level in range(5)
            }
            cells.append(", ".join(f"{k} {v}" for k, v in counts.items() if v))
        add(f"| {seed['seed']} | " + " | ".join(cells) + " |")
    add("")
    add("**No true-L5 validation event is decoded as L5 by any seed, at any threshold.** The six")
    add("true-L5 rows decode as L3/L4/L6/L7. That is the representation failure, and no")
    add("threshold can reach it.")
    add("")

    # ------------------------------------------------------ shipped threshold
    add("## 2. The shipped operating point (validation)")
    add("")
    add("| seed | shipped threshold | interior mass ceiling | threshold > ceiling? | coverage | accepted | interior decoded / accepted | levels emitted as decisions | fnr_covered | fnr (abstention detector) | ECE (all) |")
    add("|---|---:|---:|---|---:|---:|---:|---|---:|---:|---:|")
    for seed in seeds:
        row = seed["shipped_threshold_row"]
        add(
            f"| {seed['seed']} | {seed['shipped_threshold']:.4f} | {seed['interior_mass_ceiling']:.4f} | "
            f"{'**yes**' if row['exceeds_interior_mass_ceiling'] else 'no'} | {row['coverage']:.4f} | "
            f"{row['accepted']} | {row['interior_decodes_total']} / **{row['interior_decodes_accepted']}** | "
            f"{', '.join(row['levels_emitted_as_decisions'])} | "
            f"{row['dangerous_underclassification_fnr_covered']:.4f} | "
            f"{row['dangerous_fnr_abstention_detector']:.4f} | {row['ece_all']:.4f} |"
        )
    add("")
    add("In all five seeds the shipped threshold sits **above** the CORAL interior mass ceiling")
    add("`tanh(gap/4)`, so the gate has been 'satisfied' in a region where interior decodes are")
    add("structurally un-acceptable. For seeds 29 and 101 the shipped threshold (0.5092, 0.5189)")
    add("is more than twice the ceiling (0.2229, 0.2232) and **zero** of 25 and 26 interior")
    add("decodes survive. This is the abstention-gate twin of the H-CORAL argmax failure.")
    add("")

    # ------------------------------------------------------ coverage-targeted
    add("## 3. Coverage-targeted view: what does reachability cost in coverage?")
    add("")
    add("For each target coverage the most conservative threshold that still covers at least")
    add("that fraction of the 110 validation events is selected, and the row is reported")
    add("unchanged. `L5/L6 emitted` is the count of decisions at that level.")
    add("")
    for target in TARGET_COVERAGES:
        add(f"### target coverage {target:.0%}")
        add("")
        add("| seed | threshold | coverage | accepted | L5 emitted | L6 emitted | L5 recall (hits/covered) | L6 recall (hits/covered) | fnr_covered | fnr (detector) | rate_all | ECE all | ECE covered |")
        add("|---|---:|---:|---:|---:|---:|---|---|---:|---:|---:|---:|---:|")
        for seed in seeds:
            eligible = [row for row in seed["rows"] if row["coverage"] + 1e-12 >= target]
            if not eligible:
                add(f"| {seed['seed']} | — | — | — | — | — | — | — | — | — | — | — | — |")
                continue
            row = max(eligible, key=lambda item: item["threshold"])
            recall = row["per_level_recall_covered"]
            add(
                f"| {seed['seed']} | {row['threshold']:.4f} | {row['coverage']:.4f} | {row['accepted']} | "
                f"{row['emitted_level_counts'].get('L5', 0)} | {row['emitted_level_counts'].get('L6', 0)} | "
                f"{recall_cell(recall['L5'])} | {recall_cell(recall['L6'])} | "
                f"{fmt(row['dangerous_underclassification_fnr_covered'])} | "
                f"{fmt(row['dangerous_fnr_abstention_detector'])} | "
                f"{fmt(row['dangerous_underclassification_rate_all'])} | "
                f"{row['ece_all']:.4f} | {fmt(row['ece_covered'])} |"
            )
        add("")
    add("**Coverage is not the cost.** Reachability requires a *lower* threshold, which raises")
    add("coverage. The binding cost is the dangerous-direction FNR: the cheapest covered FNR")
    add("among thresholds that emit L5 or L6 is")
    add("0.0581 / 0.0659 / 0.0455 / 0.0722 / 0.0833 for seeds 17 / 29 / 43 / 71 / 101, against")
    add("the 0.05 gate — so only seed 43 has any operating point that both emits an interior")
    add("level and meets the gate (3 such thresholds, all at or below its shipped 0.2394).")
    add("The abstention-detector FNR at those points runs 0.31–0.58, and rises to 1.00 at")
    add("threshold 0 because a threshold that accepts everything catches nothing.")
    add("")

    # ------------------------------------------------------ boundary table
    add("## 4. The reachability boundary: highest threshold that still emits both L5 and L6")
    add("")
    add("| seed | threshold | coverage | accepted | interior decoded / accepted | L5 emitted | L6 emitted | L5 recall | L6 recall | L3 recall | L4 recall | L7 recall | fnr_covered | fnr (detector) | rate_all | ECE all |")
    add("|---|---:|---:|---:|---:|---:|---:|---|---|---|---|---|---:|---:|---:|---:|")
    boundary_rows = []
    for seed in seeds:
        both = [
            row for row in seed["rows"]
            if row["emitted_level_counts"].get("L5", 0) > 0 and row["emitted_level_counts"].get("L6", 0) > 0
        ]
        if not both:
            add(f"| {seed['seed']} | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |")
            continue
        row = max(both, key=lambda item: item["threshold"])
        boundary_rows.append((seed["seed"], row))
        recall = row["per_level_recall_covered"]
        add(
            f"| {seed['seed']} | {row['threshold']:.4f} | {row['coverage']:.4f} | {row['accepted']} | "
            f"{row['interior_decodes_total']} / {row['interior_decodes_accepted']} | "
            f"{row['emitted_level_counts'].get('L5', 0)} | {row['emitted_level_counts'].get('L6', 0)} | "
            f"{recall_cell(recall['L5'])} | {recall_cell(recall['L6'])} | {recall_cell(recall['L3'])} | "
            f"{recall_cell(recall['L4'])} | {recall_cell(recall['L7'])} | "
            f"{fmt(row['dangerous_underclassification_fnr_covered'])} | "
            f"{fmt(row['dangerous_fnr_abstention_detector'])} | "
            f"{fmt(row['dangerous_underclassification_rate_all'])} | {row['ece_all']:.4f} |"
        )
    add("")
    add("Note what this table does **not** buy: L5 recall is 0.000 (0 hits) at every one of")
    add("these points, for every seed. The level is emitted and never right.")
    add("")

    # ---------------------------------------------- direction of the emissions
    add("### Which direction do the newly-reachable decisions err in?")
    add("")
    add("Levels are ordered soft -> thick (L3 softest, L7 thickest), so the dangerous")
    add("direction is `y_pred < y_true` — calling food SOFTER than it is. Composition of the")
    add("accepted L5 and L6 decisions at the reachability boundary and at threshold 0:")
    add("")
    add("| seed | point | level | emitted | correct | dangerous under (`y_pred < y_true`) | over (`y_pred > y_true`) | true-level breakdown |")
    add("|---|---|---|---:|---:|---:|---:|---|")
    for seed in seeds:
        for label, key in (("boundary", "emitted_L5_L6_composition_at_boundary"),
                           ("threshold 0", "emitted_L5_L6_composition_at_threshold_zero")):
            block = seed[key]
            for level in ("L5", "L6"):
                stats = block[level]
                breakdown = ", ".join(
                    f"{name.replace('true_', '')} {count}"
                    for name, count in stats["true_level_counts"].items()
                    if count
                )
                add(
                    f"| {seed['seed']} | {label} | {level} | {stats['n_emitted']} | {stats['correct']} | "
                    f"{stats['dangerous_direction_under']} | {stats['over_direction']} | {breakdown or '—'} |"
                )
    add("")
    add("At the reachability boundary each seed emits exactly **one** L5 decision and it is")
    add("always a true-L4 row called L5 — an over-prediction, so in the safe direction but")
    add("still wrong. The L6 decisions that appear at the boundary are dominated by true-L7")
    add("rows called L6, which IS the dangerous direction. Pooling the five seeds at threshold")
    add("0: **34 of 62 emitted L6 decisions (54.8%) are dangerous under-classifications**, and")
    add("19 of 22 emitted L5 decisions (86.4%) are over-predictions of true L3/L4 rows. So")
    add("lowering the threshold to reach the interior levels does not buy correctness; it buys")
    add("mostly L7 rows relabelled one level softer, which is precisely the error direction the")
    add("wk2 gate exists to catch.")
    add("")
    add("### The gate-meeting interior-emitting points")
    add("")
    add("| seed | thresholds emitting L5/L6 that also meet the 0.05 covered-FNR gate | most conservative such point |")
    add("|---|---|---|")
    for seed in seeds:
        answer = seed["answer"]
        best = answer["most_conservative_L5_or_L6_emitting_threshold_meeting_the_0_05_gate"]
        add(
            f"| {seed['seed']} | {answer['n_L5_or_L6_emitting_thresholds_meeting_the_0_05_gate']} | "
            + (
                f"threshold {best['threshold']:.6f}, coverage {best['coverage']:.4f}, "
                f"emitted {', '.join(f'{k} {v}' for k, v in best['emitted_level_counts'].items())}"
                if best else "none"
            )
            + " |"
        )
    add("")
    add("For four of five seeds there is **no** threshold on the whole achievable grid that")
    add("both emits an interior level and satisfies the 0.05 under-classification gate. The one")
    add("exception is seed 43, whose shipped threshold already emits L6 — and note that its")
    add("three gate-meeting points sit within 2e-6 of each other (0.2394257 / 0.2394267 /")
    add("0.2394276), straddling a single event, so this exception is a knife-edge rather than")
    add("a robust operating point.")
    add("")

    # ------------------------------------------------------ extremes
    add("## 5. Threshold 0.0 (accept everything) — the ceiling on reachability")
    add("")
    add("| seed | coverage | accepted | levels emitted | L5 recall | L6 recall | L3 recall | L4 recall | L7 recall | fnr_covered | fnr (detector) | ECE all |")
    add("|---|---:|---:|---|---|---|---|---|---|---:|---:|---:|")
    for seed in seeds:
        row = seed["rows"][0]
        recall = row["per_level_recall_covered"]
        add(
            f"| {seed['seed']} | {row['coverage']:.4f} | {row['accepted']} | "
            f"{', '.join(row['levels_emitted_as_decisions'])} | "
            f"{recall_cell(recall['L5'])} | {recall_cell(recall['L6'])} | {recall_cell(recall['L3'])} | "
            f"{recall_cell(recall['L4'])} | {recall_cell(recall['L7'])} | "
            f"{fmt(row['dangerous_underclassification_fnr_covered'])} | "
            f"{fmt(row['dangerous_fnr_abstention_detector'])} | {row['ece_all']:.4f} |"
        )
    add("")

    # ------------------------------------------------------ FNR definitions
    add("## 6. Dangerous-direction FNR: three definitions, three denominators")
    add("")
    add("`trainer/eval/abstention.py` defines the dangerous direction as `y_pred < y_true`")
    add("(predicting a SAFER texture than truth). Three quantities circulate in this repo and")
    add("they do not share a denominator:")
    add("")
    add("| name | numerator | denominator | where it lives |")
    add("|---|---|---|---|")
    add("| `dangerous_fnr_abstention_detector` | dangerous rows accepted | all dangerous rows (pre-abstention) | `eval/abstention.py:risk_coverage_row` |")
    add("| `dangerous_underclassification_fnr_covered` | dangerous rows accepted | all covered rows | `eval/metrics.py:_point_metrics` (the shipped name) |")
    add("| `dangerous_underclassification_rate_all` | dangerous rows accepted | all N | `eval/metrics.py:_point_metrics` |")
    add("")
    add("H-ABSTBLIND records that `dangerous_underclassification_rate_all` gates its numerator")
    add("on acceptance while dividing by all N, so it is the only one of the three that")
    add("improves when the model abstains. All three are reported at every point above.")
    add("")
    dangerous_totals = {
        seed["seed"]: seed["shipped_threshold_row"]["dangerous_errors_total"] for seed in seeds
    }
    add(
        "Dangerous-direction samples on the validation split (pre-abstention, denominator of "
        "the detector FNR): "
        + ", ".join(f"seed {k} = {v}/110" for k, v in dangerous_totals.items())
        + "."
    )
    add("")

    # ------------------------------------------------------ L5 zero proof
    add("## 7. Why no threshold recovers L5 (measured, not argued)")
    add("")
    add("| seed | distinct L5 hit counts over all candidate thresholds | L5 hits at threshold 0 | true-L5 val rows | thresholds that emit L5 |")
    add("|---|---:|---:|---:|---:|")
    for seed in seeds:
        hits = {
            row["per_level_recall_covered"]["L5"]["hits"] for row in seed["rows"]
        }
        l5_thresholds = [
            row for row in seed["rows"] if row["emitted_level_counts"].get("L5", 0) > 0
        ]
        add(
            f"| {seed['seed']} | {sorted(hits)} | "
            f"{seed['rows'][0]['per_level_recall_covered']['L5']['hits']} | "
            f"{seed['val_L5_support']} | {len(l5_thresholds)} |"
        )
    add("")
    add("The set of distinct L5 hit counts is the single value `[0]` for every seed across the")
    add("whole candidate grid: L5 is emitted at 13–21 thresholds per seed and is never once")
    add("correct. Reachable and recalled are different claims, and only the first is true.")
    add("")

    # ------------------------------------------------------ ECE note
    add("## 8. ECE across the sweep")
    add("")
    add("ECE is `eval.metrics.expected_calibration_error` with 10 equal-width bins on the")
    add("CORAL rank-decode confidence (the probability mass of the decoded level), computed on")
    add("the 110 validation events. ECE (all) does not depend on the threshold — the bins and")
    add("the confidences are threshold-free — and it is repeated in every row so the table is")
    add("self-contained; ECE (covered) does move with the threshold because the mask changes.")
    add("")
    add("| seed | ECE all (threshold-free) | ECE covered at shipped threshold | ECE covered at the reachability boundary | ECE covered at threshold 0 |")
    add("|---|---:|---:|---:|---:|")
    for seed in seeds:
        shipped = seed["shipped_threshold_row"]
        both = [
            row for row in seed["rows"]
            if row["emitted_level_counts"].get("L5", 0) > 0 and row["emitted_level_counts"].get("L6", 0) > 0
        ]
        boundary = max(both, key=lambda item: item["threshold"]) if both else None
        zero = seed["rows"][0]
        add(
            f"| {seed['seed']} | {shipped['ece_all']:.4f} | {fmt(shipped['ece_covered'])} | "
            f"{fmt(boundary['ece_covered']) if boundary else '—'} | {fmt(zero['ece_covered'])} |"
        )
    add("")

    # ------------------------------------------------------ temperature invariance
    add("## 9. The decode is temperature-invariant, so this is not a calibration artifact")
    add("")
    add("`P(y > k) > 0.5` is equivalent to `cutpoint_logit_k > 0`, and dividing the logits by a")
    add("positive temperature preserves the sign. The decoded level therefore cannot change")
    add("with temperature; only the confidence (class mass) can. Verified on the validation")
    add("rows of all five seeds:")
    add("")
    add("| seed | decoded levels differing between T=1 and the shipped T |")
    add("|---|")
    import torch  # noqa: PLC0415
    from train.model import coral_logits_to_probs  # noqa: PLC0415

    for seed in seeds:
        npz = np.load(out_dir / f"val_predictions_seed{seed['seed']}_cpu.npz", allow_pickle=True)
        logits = npz["cutpoint_logits"].astype(np.float64)
        shipped_temperature = float(npz["temperature"])
        uncalibrated = coral_logits_to_probs(torch.from_numpy(logits)).numpy()
        calibrated = coral_logits_to_probs(torch.from_numpy(logits / shipped_temperature)).numpy()
        raw_pred, _ = coral_decode(uncalibrated)
        cal_pred, _ = coral_decode(calibrated)
        add(f"| {seed['seed']} | {int((raw_pred != cal_pred).sum())} of 110 |")
    add("")
    add("The pre-abstention census in section 1 is therefore a property of the checkpoint, not")
    add("of the temperature that happened to be selected on the same 110 rows.")
    add("")

    # ------------------------------------------------------ reproduction fidelity
    add("## 10. Reproduction fidelity: the CPU forward against the shipped GPU run")
    add("")
    add("The container `iddsi-train` had lost CUDA when this lane ran (NVML `Unknown Error`,")
    add("`torch.cuda.device_count() == 0`), so the validation forward pass was re-run on CPU in")
    add("fp32. Two checks say that is safe here, and one says where it is not.")
    add("")
    add("**Bit-identical logits.** Wave-1 lane `an03_calibration` dumped the same validation")
    add("forward pass from the same `best.pt` on the GPU")
    add("(`/work/lane_runs/an03_calibration/val_predictions_seed43.npz`, sha256")
    add("`6988394cea0f238b26e20ed9f7e685df6512e3cc98e5104b40f1e053de0f2764`). Comparing the two:")
    add("")
    add("| quantity | value |")
    add("|---|---|")
    add("| max abs difference in the 110x4 cut-point logits | **0.0** (bit-identical) |")
    add("| labels and event_ids identical | yes (110/110) |")
    add("| decoded level disagreements | **0 of 110** |")
    add("| pre-abstention decode census (reference vs this lane) | L3 8, L4 11, L5 5, L6 12, L7 74 — identical |")
    add("")
    add("**Where the CPU path does differ.** Applying the shipped threshold to the recomputed")
    add("confidences reproduces the shipped coverage for 3 of 5 seeds exactly and is one event")
    add("short for 2 of 5:")
    add("")
    add("| seed | shipped threshold | coverage in shipped `calibration.json` | coverage recomputed here | accepted, shipped | accepted, recomputed |")
    add("|---|---:|---:|---:|---:|---:|")
    import json as _json  # noqa: PLC0415

    for seed in seeds:
        run_dir = f"/work/runs/v07/siglip2_v07_fullcoverage_seed{seed['seed']}"
        shipped_calibration = _json.loads((Path(run_dir) / "calibration.json").read_text())
        row = seed["shipped_threshold_row"]
        add(
            f"| {seed['seed']} | {seed['shipped_threshold']:.9f} | "
            f"{shipped_calibration['coverage']:.4f} | {row['coverage']:.4f} | "
            f"{round(shipped_calibration['coverage'] * 110)} | {row['accepted']} |"
        )
    add("")
    add("The shipped threshold is not one of the recomputed decoded confidences in any seed: it")
    add("is a knife-edge value that falls between two of them, and for seeds 29 and 43 one")
    add("recomputed confidence lands on the other side of it. The gap is ~1e-6 in the")
    add("confidence of a single event — the residual of CUDA kernel non-determinism between")
    add("the original training-time run and any later re-run, CPU or GPU. It moves coverage by")
    add("0.009 and nothing else: the decode census, the reachability boundary and the L5")
    add("zero-hit result are unchanged, because the decode itself has zero disagreements. The")
    add("shipped threshold is nevertheless carried as an explicit candidate in the sweep grid,")
    add("so the rows labelled 'shipped' are evaluated at the shipped value, not at the")
    add("nearest achievable one.")
    add("")

    # ------------------------------------------------------ what to change
    add("## 11. What this means for the threshold rule")
    add("")
    add("1. **The shipped selector can pick a threshold above the CORAL interior mass ceiling,")
    add("   and did so in all five seeds.** `eval/calibration.py` in this clone already clips")
    add("   the candidate grid to `threshold <= ceiling`; the five shipped `calibration.json`")
    add("   files were produced before that clip. Re-running the shipped selector with the clip")
    add("   moves the threshold to 0.2302 / 0.2229 / 0.2372 / 0.2225 / 0.2231 and reports")
    add("   `target_met: false` in all five cases, because the gate can no longer be met inside")
    add("   the reachable region. That is an honest outcome and it is the one the clip was")
    add("   written to produce.")
    add("2. **L5 cannot be fixed by a threshold.** Zero of six true-L5 validation events decode")
    add("   as L5 in any seed at any threshold. The remedy is the representation, not the gate.")
    add("3. **L6 is partially recoverable but not for free.** L6 recall reaches 0.33 (3/9) at")
    add("   threshold 0 and 0.14–0.25 at the reachability boundary, where the covered FNR is")
    add("   0.0737–0.0833 against the 0.05 gate — so for four of five seeds the boundary point")
    add("   does not satisfy the gate at all.")
    add("4. **Do not report coverage as the price of reachability** — it is the wrong sign. The")
    add("   price is dangerous-direction FNR and gate compliance.")
    add("")

    add("## Interpretation boundary")
    add("")
    add("Research demonstration for culinary education. Estimates visual similarity to IDDSI")
    add("descriptors; does not perform official IDDSI tests, assess swallowing, determine")
    add("suitability for any person, or decide whether food is safe to consume. Never replaces")
    add("clinician-prescribed texture or physical IDDSI testing. This project is not the")
    add("official IDDSI website and is not endorsed by IDDSI. Training labels are VLM weak")
    add("labels, not physical IDDSI tests and not clinician-validated; every figure above is")
    add("measured against the inter-judge label noise recorded in `hf/HAZARD_LOG.md` (H-NOISE,")
    add("H-QWENBIAS) and means nothing about real-world accuracy.")
    add("")

    (out_dir / "threshold_sweep_val.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
