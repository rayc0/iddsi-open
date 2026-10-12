#!/usr/bin/env python3
"""Collect the fix04_classbalance arms into one table with denominators stated.

Reads the per-seed artifacts the (patched) engine writes under
``/work/lane_runs/fix04_classbalance/<arm>/<name>_seed<N>/`` and emits
``classbalance_summary.json`` plus a markdown table.

Every number printed here is read from a file named in the same row; nothing is
recomputed from a checkpoint and nothing is carried over from a prior run.

Two splits are reported for every arm and they are NOT interchangeable:

* ``val``  -- the split the temperature and the abstention threshold were both
  fitted on.  This is the only split any rule in the run was chosen on, so it is
  the split the per-level recall question belongs to.  L5 support is 6 and L6
  support is 9 there, so those recalls are over 6 and 9 events.
* ``test`` -- the held-out split, read once at the policy fixed on val.  L5
  support is 7 and L6 support is 10.

Per-level recall is reported twice, because the two denominators disagree in the
direction that matters (H-ABSTBLIND / H-KAPPACOV):

* ``recall_all``      -- denominator is every event of that true level; an
  abstention is a miss.
* ``recall_covered``  -- denominator is only the accepted events of that level,
  i.e. the population the covered kappa is computed on.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

RUNS = Path("/home/tun/iddsi/lane_runs/fix04_classbalance")
LEVELS = ["L3", "L4", "L5", "L6", "L7"]

ARMS = [
    ("a_beta999", "fix04_a_beta999", "(a) shipped beta = 0.999, natural loader"),
    ("b_beta99999", "fix04_b_beta99999", "(b) stronger beta = 0.99999, natural loader"),
    ("c_equalised", "fix04_c_equalised", "(c) shipped beta = 0.999, balanced sampler"),
    ("d_beta0_nobalance", "fix04_d_beta0_nobalance", "(d) EXTENSION beta = 0.0, no balance"),
]


def _load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _fmt(value, digits=2):
    if value is None:
        return "NA"
    if isinstance(value, float) and value != value:
        return "NA"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _recall_cell(recall: dict, level: str, key: str) -> str:
    entry = recall.get(level, {})
    if key == "recall_all":
        return f"{_fmt(entry.get('recall_all'))} ({entry.get('support', '?')})"
    return f"{_fmt(entry.get('recall_covered'))} ({entry.get('covered_support', '?')})"


def collect() -> dict:
    rows: list[dict] = []
    for slug, name, label in ARMS:
        arm_dir = RUNS / slug
        seeds = sorted(int(p.name.rsplit("_seed", 1)[1]) for p in arm_dir.glob(f"{name}_seed*") if p.is_dir())
        for seed in seeds:
            run_dir = arm_dir / f"{name}_seed{seed}"
            required = ["val_metrics.json", "metrics.json", "calibration.json", "class_balance.json"]
            missing = [item for item in required if not (run_dir / item).exists()]
            if missing:
                print(f"WARNING {run_dir} missing {missing}", file=sys.stderr)
                continue
            val = _load(run_dir / "val_metrics.json")
            test = _load(run_dir / "metrics.json")
            calib = _load(run_dir / "calibration.json")
            balance = _load(run_dir / "class_balance.json")
            sampler_path = run_dir / "train_sampler.json"
            sampler = _load(sampler_path) if sampler_path.exists() else {"sampler": "natural"}
            rows.append(
                {
                    "arm": slug,
                    "arm_label": label,
                    "seed": seed,
                    "run_dir": str(run_dir),
                    "class_balance_beta": balance["beta"],
                    "class_weights": balance["class_weights"],
                    "max_over_min_weight_ratio": balance["max_over_min_weight_ratio"],
                    "train_level_counts": balance["train_level_counts"],
                    "sampler": sampler.get("sampler"),
                    "realised_first_epoch_draws": sampler.get("realised_first_epoch_draws"),
                    "val": {
                        "n": val["n"],
                        "level_support": val["level_support"],
                        "threshold": val["threshold"],
                        "temperature": calib["temperature"],
                        "target_met": calib["target_met"],
                        "coverage": val["metrics"]["coverage"],
                        "weighted_kappa_covered": val["metrics"]["weighted_kappa_covered"],
                        "per_level_recall": val["per_level_recall"],
                        "decoded_level_counts_pre_abstention": val["decoded_level_counts_pre_abstention"],
                        "argmax_level_counts": val["argmax_level_counts"],
                        "interior_decodes_total": val["interior_decodes_total"],
                        "interior_decodes_accepted": val["interior_decodes_accepted"],
                        "interior_levels_reachable": val["interior_levels_reachable"],
                    },
                    "test": {
                        "n": test["n"],
                        "level_support": test["level_support"],
                        "threshold": test["threshold"],
                        "coverage": test["metrics"]["coverage"],
                        "weighted_kappa_covered": test["metrics"]["weighted_kappa_covered"],
                        "per_level_recall": test["per_level_recall"],
                        "decoded_level_counts_pre_abstention": test["decoded_level_counts_pre_abstention"],
                        "argmax_level_counts": test["argmax_level_counts"],
                        "interior_decodes_total": test["interior_decodes_total"],
                        "interior_decodes_accepted": test["interior_decodes_accepted"],
                        "interior_levels_reachable": test["interior_levels_reachable"],
                    },
                }
            )
    return {"arms": rows}


def markdown(payload: dict) -> str:
    lines = [
        "# fix04_classbalance — per-level recall by arm",
        "",
        "Every cell is `value (denominator)`. Val recall denominators are the",
        "val level supports; test recall denominators are the test level supports.",
        "Threshold and temperature were fitted on val only; test is the held-out read.",
        "",
        "## VALIDATION split (the split every rule was chosen on)",
        "",
        "| arm | seed | beta | sampler | L3 recall_all | L4 | L5 | L6 | L7 |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for row in payload["arms"]:
        recall = row["val"]["per_level_recall"]
        cells = " | ".join(_recall_cell(recall, level, "recall_all") for level in LEVELS)
        lines.append(
            f"| {row['arm_label']} | {row['seed']} | {row['class_balance_beta']} | "
            f"{row['sampler']} | {cells} |"
        )
    lines += [
        "",
        "## VALIDATION split — covered-subset recall (abstentions excluded)",
        "",
        "| arm | seed | L3 recall_covered | L4 | L5 | L6 | L7 |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in payload["arms"]:
        recall = row["val"]["per_level_recall"]
        cells = " | ".join(_recall_cell(recall, level, "recall_covered") for level in LEVELS)
        lines.append(f"| {row['arm_label']} | {row['seed']} | {cells} |")

    lines += [
        "",
        "## VALIDATION split — is the level reachable at all?",
        "",
        "| arm | seed | rank decode L3/L4/L5/L6/L7 | argmax decode L3/L4/L5/L6/L7 | interior decodes | interior accepted | threshold |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in payload["arms"]:
        val = row["val"]
        rank = "/".join(str(val["decoded_level_counts_pre_abstention"][level]) for level in LEVELS)
        amax = "/".join(str(val["argmax_level_counts"][level]) for level in LEVELS)
        lines.append(
            f"| {row['arm_label']} | {row['seed']} | {rank} | {amax} | "
            f"{val['interior_decodes_total']} | {val['interior_decodes_accepted']} | "
            f"{_fmt(val['threshold'], 4)} |"
        )

    lines += [
        "",
        "## TEST split (held-out read of the policy fixed on val)",
        "",
        "| arm | seed | L3 recall_all | L4 | L5 | L6 | L7 | coverage | kappa_covered |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for row in payload["arms"]:
        recall = row["test"]["per_level_recall"]
        cells = " | ".join(_recall_cell(recall, level, "recall_all") for level in LEVELS)
        lines.append(
            f"| {row['arm_label']} | {row['seed']} | {cells} | "
            f"{_fmt(row['test']['coverage'], 3)} | {_fmt(row['test']['weighted_kappa_covered'], 4)} |"
        )

    lines += [
        "",
        "## Class-balance actually applied",
        "",
        "| arm | seed | beta | weights L3/L4/L5/L6/L7 | max/min | realised first-epoch draws L3/L4/L5/L6/L7 |",
        "|---|---|---|---|---|---|",
    ]
    for row in payload["arms"]:
        weights = "/".join(_fmt(row["class_weights"][level], 3) for level in LEVELS)
        draws = row.get("realised_first_epoch_draws")
        draw_text = "/".join(str(draws[level]) for level in LEVELS) if draws else "natural (28/37/13/35/201)"
        lines.append(
            f"| {row['arm_label']} | {row['seed']} | {row['class_balance_beta']} | {weights} | "
            f"{_fmt(row['max_over_min_weight_ratio'], 2)} | {draw_text} |"
        )
    lines += [
        "",
        "## Support in the splits used",
        "",
        "train (314 rows): L3 28 / L4 37 / L5 13 / L6 35 / L7 201",
        "val   (110 rows): L3 14 / L4  8 / L5  6 / L6  9 / L7 73",
        "test  (108 rows): L3 10 / L4 11 / L5  7 / L6 10 / L7 70",
        "",
        "Per-seed artifacts: the `run_dir` column of `classbalance_summary.json`.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    payload = collect()
    out_dir = RUNS / "summary"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "classbalance_summary.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    (out_dir / "classbalance_summary.md").write_text(markdown(payload), encoding="utf-8")
    print(markdown(payload))
    if not payload["arms"]:
        print("NO RUNS COLLECTED", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
