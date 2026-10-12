"""Assemble the fix03_retrain headline table from the two analysis artifacts.

Reads (both written by this lane):
  /work/lane_runs/fix03_retrain/analysis/decode_comparison.json
      -- the two retrained runs (v0.7 recipe, seeds 17 and 43), each evaluated
         under BOTH decode rules with the threshold frozen on validation.
  /work/lane_runs/fix03_retrain/analysis/v07_checkpoint_recalibration.json
      -- the five SHIPPED v0.7 checkpoints re-predicted and re-calibrated with
         the current trainer code, again under both decode rules.
  /work/lane_runs/fix03_retrain/analysis/baseline_v07_reproduction.json
      -- the published v0.7 metrics recomputed from the wave-1 artifacts, i.e.
         the v0.7 release baseline as it was actually published.

Writes /work/lane_runs/fix03_retrain/analysis/headline_table.json and prints the
table. Every number below is read from one of those three files.
"""

from __future__ import annotations

import json
from pathlib import Path

ANALYSIS = Path("/work/lane_runs/fix03_retrain/analysis")


def _recall_row(per_level: dict[str, dict[str, float | int]]) -> dict[str, dict[str, float | int]]:
    return {
        key: {
            "hits": int(value["hits"]),
            "support": int(value["support"]),
            "recall": (value["hits"] / value["support"]) if value["support"] else None,
        }
        for key, value in per_level.items()
    }


def _decided_recall(confusion: list[list[int]]) -> dict[str, dict[str, float | int]]:
    """Per-level recall over DECIDED events only (the wave-1 published basis).

    ``confusion`` is the 6x6 matrix from ``evaluate_predictions``: rows are true
    L3..L7 then true-unclear, columns are pred L3..L7 then unclear. The decided
    denominator for a true level is that row's first five entries, and the hit
    is its diagonal entry.
    """
    out: dict[str, dict[str, float | int]] = {}
    for index in range(5):
        decided = sum(confusion[index][:5])
        hits = confusion[index][index]
        out[f"L{index + 3}"] = {
            "decided_support": int(decided),
            "hits": int(hits),
            "recall_decided": (hits / decided) if decided else None,
        }
    return out


def _row(
    *,
    label: str,
    weights: str,
    decode: str,
    threshold: float | None,
    metrics: dict[str, float],
    per_level: dict[str, dict[str, float | int]],
    emitted: dict[str, int],
    interior_total: int,
    interior_accepted: int,
    n: int,
    confusion: list[list[int]] | None = None,
) -> dict[str, object]:
    def metric(*names: str) -> float:
        for name in names:
            if name in metrics:
                return float(metrics[name])
        raise KeyError(f"none of {names} in {sorted(metrics)}")

    return {
        "label": label,
        "weights": weights,
        "decode": decode,
        "threshold": threshold,
        "n_test_events": n,
        "macro_f1_with_abstentions": metric(
            "macro_f1_with_abstentions", "macro_f1_with_abstentions_shipped_definition"
        ),
        "weighted_kappa_covered": metric("weighted_kappa_covered"),
        "dangerous_fnr_covered": metric(
            "dangerous_underclassification_fnr_covered", "dangerous_fnr_covered"
        ),
        "coverage": metric("coverage"),
        "per_level_recall_all_events": _recall_row(per_level),
        "per_level_recall_decided_only": (
            _decided_recall(confusion) if confusion is not None else None
        ),
        "emitted_level_counts": emitted,
        "interior_decodes_total": interior_total,
        "interior_decodes_accepted": interior_accepted,
    }


def main() -> None:
    comparison = json.loads((ANALYSIS / "decode_comparison.json").read_text())
    recalibration = json.loads((ANALYSIS / "v07_checkpoint_recalibration.json").read_text())
    baseline = json.loads((ANALYSIS / "baseline_v07_reproduction.json").read_text())

    rows: list[dict[str, object]] = []

    # --- the v0.7 release baseline, as published -----------------------------
    for seed, entry in baseline["seeds"].items():
        rank = entry["per_level_recall_all_events_rank_decode"]
        rows.append(
            _row(
                label=f"v0.7 published, seed {seed}",
                weights="v0.7 release (shipped)",
                decode="rank (no ceiling clip)",
                threshold=entry["threshold"],
                metrics=entry["recomputed"],
                per_level=rank,
                emitted=entry["rank_emitted_level_counts"],
                confusion=entry["recomputed"]["confusion_matrix"],
                interior_total=entry["diagnostics"]["interior_decodes_total"],
                interior_accepted=entry["diagnostics"]["interior_decodes_accepted"],
                n=108,
            )
        )

    # --- shipped v0.7 weights, re-calibrated with the current code -----------
    for seed, entry in recalibration["seeds"].items():
        for rule in ("rank", "argmax"):
            cell = entry["rules"][rule]
            rows.append(
                _row(
                    label=f"v0.7 shipped weights re-calibrated, seed {seed}",
                    weights="v0.7 release (shipped)",
                    decode=rule,
                    threshold=cell["threshold"],
                    metrics=cell["metrics"],
                    per_level=cell["per_level_recall_all_events"],
                    emitted=cell["emitted_level_counts"],
                    confusion=cell["confusion_matrix"],
                    interior_total=cell["interior_decodes_total"],
                    interior_accepted=cell["interior_decodes_accepted"],
                    n=recalibration["n_test"],
                )
            )

    # --- this lane's retrains, both decodes on identical weights -------------
    for run in comparison["runs"]:
        seed = run["run_dir"].rsplit("_seed", 1)[1]
        for rule in ("rank", "argmax"):
            cell = run["rules"][rule]
            test = cell["test"]
            rows.append(
                _row(
                    label=f"fix03_retrain retrain (candidate c), seed {seed}",
                    weights="retrained by this lane (v0.7 recipe)",
                    decode=rule,
                    threshold=cell["threshold_selected_on_val"],
                    metrics={
                        "macro_f1_with_abstentions": test[
                            "macro_f1_with_abstentions_shipped_definition"
                        ],
                        "weighted_kappa_covered": test["weighted_kappa_covered"],
                        "dangerous_underclassification_fnr_covered": test["dangerous_fnr_covered"],
                        "coverage": test["coverage"],
                    },
                    per_level=test["per_level_recall_all_events"],
                    emitted=test["emitted_level_counts"],
                    confusion=test["confusion_matrix"],
                    interior_total=test["interior_decodes_total"],
                    interior_accepted=test["interior_decodes_accepted"],
                    n=run["n_test"],
                )
            )

    payload = {
        "denominators": {
            "test_events": 108,
            "val_events": 110,
            "val_level_support": recalibration["val_level_support"],
            "test_level_support": recalibration["test_level_support"],
            "recall_definition": (
                "per-level recall = hits / ALL events whose true label is that level, so an "
                "abstention is a miss. The wave-1 published recall used decided events only."
            ),
            "kappa_definition": (
                "weighted_kappa_covered = quadratic Cohen kappa over the covered subset only "
                "(H-KAPPACOV: abstentions are dropped before the kappa is taken)."
            ),
            "macro_f1_definition": (
                "macro_f1_with_abstentions is the SHIPPED definition (label set from true "
                "labels only, so -1 is not a class -- H-MACROF1). The variant with -1 as a "
                "sixth class is reported separately in decode_comparison.json."
            ),
        },
        "rows": rows,
    }
    (ANALYSIS / "headline_table.json").write_text(json.dumps(payload, indent=1), encoding="utf-8")

    header = (
        f"{'configuration':<46} {'decode':<22} {'thr':>7} {'macroF1':>8} "
        f"{'kappa':>7} {'cov':>6} {'fnrCov':>7} {'intAcc/Tot':>11}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row['label']:<46} {row['decode']:<22} "
            f"{(row['threshold'] if row['threshold'] is not None else float('nan')):>7.4f} "
            f"{row['macro_f1_with_abstentions']:>8.4f} "
            f"{row['weighted_kappa_covered']:>7.4f} "
            f"{row['coverage']:>6.3f} "
            f"{row['dangerous_fnr_covered']:>7.4f} "
            f"{row['interior_decodes_accepted']:>5}/{row['interior_decodes_total']:<5}"
        )
    print()
    print(f"{'configuration':<46} {'L3':>12} {'L4':>12} {'L5':>12} {'L6':>12} {'L7':>12}  emitted")
    print("-" * 130)
    for row in rows:
        recall = row["per_level_recall_all_events"]
        cells = " ".join(
            f"{recall[f'L{level}']['hits']}/{recall[f'L{level}']['support']}".rjust(12)
            for level in range(3, 8)
        )
        emitted = ",".join(
            f"{level}={count}" for level, count in row["emitted_level_counts"].items() if count
        )
        print(f"{row['label']:<46} {cells}  {emitted}")
    print()
    print("decided-only recall (wave-1 published basis):")
    for row in rows:
        decided = row["per_level_recall_decided_only"]
        if decided is None:
            continue
        cells = " ".join(
            f"{decided[f'L{level}']['hits']}/{decided[f'L{level}']['decided_support']}".rjust(9)
            for level in range(3, 8)
        )
        print(f"  {row['label']:<46} {row['decode']:<22} {cells}")
    print(f"\nwrote {ANALYSIS / 'headline_table.json'}")


if __name__ == "__main__":
    main()
