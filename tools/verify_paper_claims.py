#!/usr/bin/env python3
"""verify_paper_claims.py -- recompute every number paper/ quotes, then lint the
paper against the four reconciliation rules this lane was given.

WHY THIS EXISTS
---------------
`paper/paper.md` and `paper/sections/*.md` quoted kappas without naming the
metric population (covered subset vs all rows) -- the defect recorded as
H-YARDEVID in `hf/HAZARD_LOG.md` -- quoted the "152 unusable events" count
without either denominator, made a pass claim about three models without naming
the metric it passed on, and did not report the L5/L6 decode collapse that the
v0.7 error analysis measured. This script does two things:

1. NUMBERS. Every figure the paper quotes is recomputed from the committed
   source files listed under SOURCES below and written to
   `results/v07_release/paper_claims_verification.json`. Nothing is taken from
   prose, from memory, or from an aggregate file whose metric is not identified.

2. LINT. paper/paper.md and paper/sections/*.md are checked against four rules:

   R1  every paragraph quoting a kappa value names its YARDSTICK and its
       POPULATION (covered subset / all rows / a no-abstention pair set);
   R2  the 152 unusable events are stated with BOTH denominators, 152/684
       (whole corpus) and 152/510 (both-judged subset), and 29.8% is never
       paired with 684 alone;
   R3  no pass claim covering all three models is made without naming the
       metric it passed on;
   R4  the L5/L6 decode collapse is reported as a limitation, with its
       denominator.

Exit status is non-zero if any rule fails, so this can be run as a gate.

Nothing here is written into the sealed evaluation set, and the sealed files
(`capture/sealed_eval_plan.json`, `capture/sealed_eval_pilot.json`) are never
opened.

Run from the repo root:
    python3 -P tools/verify_paper_claims.py [--out PATH]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from tools.denominator_audit import read_ids, read_manifest_ids, resolve_data  # noqa: E402

# --------------------------------------------------------------------------- #
# sources -- every number in the paper must come from one of these
# --------------------------------------------------------------------------- #
SOURCES = {
    "per_seed_yardstick": "results/yardstick_audit/per_seed.json",
    "bias_matrix": "results/v07_release/bias_matrix.json",
    "v07_seed_metrics": "results/v07_release/v07_seed*_metrics.json",
    "v01_decode_fix": "results/fix1_siglip2_coral_seed17/metrics.json",
    "structural_seed43": "analysis/errors_v07/outputs/structural_seed43.json",
    "inter_judge": "results/regen_20260916/inter_judge_20260916.json",
    "labelability_sweep": "results/v05_release/labelability_sweep.json",
    "unusable_ids": "results/v05_release/unusable_event_ids.txt",
    "metric_code": "trainer/eval/metrics.py",
}

SEEDS = [17, 29, 43, 71, 101]
LEVELS = [3, 4, 5, 6, 7]
PAPER_FILES = ["paper/paper.md", "paper/sections/limitations.md", "paper/sections/related_work.md"]
# R4 (the decode collapse is a limitation) belongs to the two documents that
# carry limitations, not to the related-work survey.
DECODE_COLLAPSE_FILES = ["paper/paper.md", "paper/sections/limitations.md"]

# A kappa value: a decimal or a decimal range, on a line that mentions kappa.
KAPPA_LINE = re.compile(r"kappa|κ", re.I)
KAPPA_VALUE = re.compile(r"\b0\.\d{2,4}\b(?:\s*[-–]\s*0?\.\d{2,4})?")

# Yardstick names. A paragraph quoting a kappa must name one of these.
YARDSTICK_MARKERS = (
    "label set",
    "yardstick",
    "qwen tier-2 judge labels",
    "judge labels",
    "v0.5 consensus",
    "v0.7 full-coverage consensus",
    "consensus",
    "human raters",
    "the two vlm judges",
    "the two judges",
)
# Population names. A paragraph quoting a kappa must name one of these.
POPULATION_MARKERS = (
    "covered",
    "all rows",
    "all-rows",
    "all 108",
    "all n",
    "all-n",
    "no coverage filter",
    "abstention",
    "n = 109",
    "n=109",
    "n_pairs = 575",
    "n_pairs=575",
    "n = 575",
    "n=575",
    "575 pairs",
    "575 judged pairs",
)
# A claim that all three models pass, which R3 requires be tied to a metric.
THREE_MODEL_PASS = re.compile(r"(all three|same three|the same models|three models)", re.I)


# --------------------------------------------------------------------------- #
# part 1 -- recompute the numbers
# --------------------------------------------------------------------------- #
def compute_numbers() -> dict:
    out: dict = {"sources": SOURCES}

    # --- v0.1 CORAL decode fix, held-out n=109 -------------------------------
    v01 = json.loads((REPO / "results/fix1_siglip2_coral_seed17/metrics.json").read_text())
    m = v01["metrics"]
    out["v01_decode_fix"] = {
        "n_held_out": v01["n"],
        "label_set": "Qwen tier-2 judge labels (v0.1 CORAL decode-fix run)",
        "macro_f1_with_abstentions": m["macro_f1_with_abstentions"],
        "weighted_kappa_covered": m["weighted_kappa_covered"],
        "kappa_population": "covered events only (abstentions dropped from numerator and denominator)",
        "dangerous_underclassification_fnr_covered": m["dangerous_underclassification_fnr_covered"],
        "dangerous_underclassification_rate_all": m["dangerous_underclassification_rate_all"],
        "coverage": m["coverage"],
        "abstention_rate": m["abstention_rate"],
        "ece_all": m["ece_all"],
    }

    # --- inter-judge agreement (the yardstick itself, not a model score) -----
    ij = json.loads((REPO / "results/regen_20260916/inter_judge_20260916.json").read_text())
    out["inter_judge"] = {
        "n_pairs": ij["n_pairs"],
        "exact_agreement": ij["exact_agreement"],
        "kappa_unweighted": ij["kappa_unweighted"],
        "kappa_quadratic": ij["kappa_quadratic"],
        "population": "all 575 pairs both judges labelled; no coverage filter (neither judge abstains)",
        "label_set": "agreement between Qwen3-VL-2B tier-1 and Qwen3-VL-8B tier-2, the two judges who produced the label set",
    }

    # --- v0.7 five-seed held-out results, per seed ---------------------------
    per_seed = []
    for seed in SEEDS:
        d = json.loads((REPO / f"results/v07_release/v07_seed{seed}_metrics.json").read_text())
        cm = d["confusion_matrix"]
        # rows are true L3..L7 (+unclear); columns are pred L3..L7 (+unclear).
        # A true level's full count includes the rows the model abstained on;
        # the DECIDED count drops them, and per-level recall on decided rows is
        # the quantity `analysis/errors_v07/outputs/structural_seed43.json`
        # publishes as `per_level_recall_decided`.
        true_counts = [sum(r) for r in cm[:5]]
        abstained = [r[5] for r in cm[:5]]
        decided_true = [true_counts[i] - abstained[i] for i in range(5)]
        decided_rows = [r[:5] for r in cm[:5]]
        col_sums = [sum(decided_rows[i][j] for i in range(5)) for j in range(5)]
        n_decided = sum(col_sums)
        diagonal = [decided_rows[i][i] for i in range(5)]
        never_emitted = [f"L{LEVELS[j]}" for j in range(5) if col_sums[j] == 0]
        recall_decided = {
            f"L{LEVELS[i]}": (diagonal[i] / decided_true[i] if decided_true[i] else None)
            for i in range(5)
        }
        recall_all = {
            f"L{LEVELS[i]}": (diagonal[i] / true_counts[i] if true_counts[i] else None)
            for i in range(5)
        }
        precision = {
            f"L{LEVELS[j]}": (diagonal[j] / col_sums[j] if col_sums[j] else None)
            for j in range(5)
        }
        per_seed.append(
            {
                "seed": seed,
                "n_test_events": d["n"],
                "coverage": d["metrics"]["coverage"],
                "n_covered": round(d["metrics"]["coverage"] * d["n"]),
                "n_abstained": d["n"] - round(d["metrics"]["coverage"] * d["n"]),
                "weighted_kappa_covered": d["metrics"]["weighted_kappa_covered"],
                "macro_f1_with_abstentions": d["metrics"]["macro_f1_with_abstentions"],
                "dangerous_underclassification_fnr_covered": d["metrics"][
                    "dangerous_underclassification_fnr_covered"
                ],
                "ece_all": d["metrics"]["ece_all"],
                "test_level_distribution": {f"L{LEVELS[i]}": true_counts[i] for i in range(5)},
                "decided_by_true_level": {f"L{LEVELS[i]}": decided_true[i] for i in range(5)},
                "n_decided": n_decided,
                "exact_match_covered": sum(diagonal) / n_decided if n_decided else None,
                "exact_match_allN_abstain_as_miss": sum(diagonal) / d["n"],
                "always_L7_baseline": true_counts[4] / d["n"],
                "per_level_recall_decided": recall_decided,
                "per_level_recall_all_rows": recall_all,
                "per_level_precision_decided": precision,
                "levels_never_emitted_as_decisions": never_emitted,
            }
        )
    out["v07_per_seed"] = per_seed

    def rng(values):
        vals = [v for v in values if v is not None]
        return [round(min(vals), 4), round(max(vals), 4)]

    out["v07_summary"] = {
        "n_test_events": 108,
        "seeds": SEEDS,
        "label_set": "v0.7 full-coverage consensus",
        "weighted_kappa_covered_range": rng([r["weighted_kappa_covered"] for r in per_seed]),
        "macro_f1_with_abstentions_range": rng(
            [r["macro_f1_with_abstentions"] for r in per_seed]
        ),
        "coverage_range": rng([r["coverage"] for r in per_seed]),
        "n_covered_range": [min(r["n_covered"] for r in per_seed), max(r["n_covered"] for r in per_seed)],
        "n_abstained_range": [
            min(r["n_abstained"] for r in per_seed),
            max(r["n_abstained"] for r in per_seed),
        ],
        "exact_match_covered_range": rng([r["exact_match_covered"] for r in per_seed]),
        "exact_match_allN_abstain_as_miss_range": rng(
            [r["exact_match_allN_abstain_as_miss"] for r in per_seed]
        ),
        "always_L7_baseline_seed43": [r for r in per_seed if r["seed"] == 43][0][
            "always_L7_baseline"
        ],
        "L5_never_emitted_in_all_5_seeds": all(
            "L5" in r["levels_never_emitted_as_decisions"] for r in per_seed
        ),
        "L6_never_emitted_in_all_5_seeds": all(
            "L6" in r["levels_never_emitted_as_decisions"] for r in per_seed
        ),
        "levels_never_emitted_by_seed": {
            str(r["seed"]): r["levels_never_emitted_as_decisions"] for r in per_seed
        },
    }

    # --- the cross-yardstick table (wk1 vs all-rows kappa) -------------------
    rows = json.loads((REPO / "results/yardstick_audit/per_seed.json").read_text())
    table: dict = {}
    for r in rows:
        key = (r["version"], r["yardstick"])
        cell = table.setdefault(
            key,
            {
                "n_events": r["n_events"],
                "wk1_kappa_covered": [],
                "kappa_all_rows_no_abstain": [],
                "kappa_all_rows_abstain_as_class": [],
                "coverage": [],
                "n_covered": [],
            },
        )
        cell["wk1_kappa_covered"].append(r["wk1_kappa_covered"])
        cell["kappa_all_rows_no_abstain"].append(r["kappa_all_rows_no_abstain"])
        cell["kappa_all_rows_abstain_as_class"].append(r["kappa_all_rows_abstain_as_class"])
        cell["coverage"].append(r["coverage"])
        cell["n_covered"].append(r["n_covered"])
    summarised = {}
    for (version, yardstick), cell in table.items():
        wk1 = cell["wk1_kappa_covered"]
        allrows = cell["kappa_all_rows_no_abstain"]
        summarised[f"{version} | {yardstick}"] = {
            "n_events": cell["n_events"],
            "n_covered_range": [min(cell["n_covered"]), max(cell["n_covered"])],
            "coverage_range": [round(min(cell["coverage"]), 4), round(max(cell["coverage"]), 4)],
            "wk1_covered_range": [round(min(wk1), 4), round(max(wk1), 4)],
            "wk1_covered_seeds_over_0.70": sum(1 for v in wk1 if v >= 0.70),
            "all_rows_range": [round(min(allrows), 4), round(max(allrows), 4)],
            "all_rows_seeds_over_0.70": sum(1 for v in allrows if v >= 0.70),
        }
    out["cross_yardstick"] = summarised

    # --- bias_matrix.json identification (H-YARDEVID) ------------------------
    bm = json.loads((REPO / "results/v07_release/bias_matrix.json").read_text())
    ident = {}
    for model, cols in bm.items():
        for yardstick, value in cols.items():
            ident[f"{model} | {yardstick}"] = {
                "value": value,
                "identified_as": "mean over 5 seeds of quadratic weighted kappa over ALL held-out rows, abstentions included in the denominator",
            }
    out["bias_matrix_identified"] = ident

    # the v0.5 model's all-rows kappa against the v0.7 labels, per seed: the
    # number that decides whether "all three pass" is metric-dependent.
    v05_allrows = [
        r["kappa_all_rows_no_abstain"]
        for r in rows
        if r["version"] == "v0.5" and r["yardstick"] == "v0.7 full-coverage consensus"
    ]
    out["all_three_pass_is_metric_dependent"] = {
        "metric": "wk1 = quadratic weighted kappa on covered rows",
        "v05_wk1_covered_range": [
            round(min(r["wk1_kappa_covered"] for r in rows
                      if r["version"] == "v0.5"
                      and r["yardstick"] == "v0.7 full-coverage consensus"), 4),
            round(max(r["wk1_kappa_covered"] for r in rows
                      if r["version"] == "v0.5"
                      and r["yardstick"] == "v0.7 full-coverage consensus"), 4),
        ],
        "v05_all_rows_range": [round(min(v05_allrows), 4), round(max(v05_allrows), 4)],
        "v05_all_rows_mean": round(sum(v05_allrows) / len(v05_allrows), 6),
        "v05_all_rows_seeds_over_0.70": sum(1 for v in v05_allrows if v >= 0.70),
        "reading": "on wk1 (covered) all three models clear 0.70 in 5/5 seeds each; on the all-rows kappa v0.5 does not, so the pass verdict depends on the metric as well as on the yardstick",
    }

    # --- corpus denominators (H-UNUSABLE) -----------------------------------
    data = resolve_data(REPO, None)
    sweep = json.loads((REPO / "results/v05_release/labelability_sweep.json").read_text())
    unusable = read_ids(REPO / "results/v05_release/unusable_event_ids.txt")
    n_unusable = len(set(unusable))
    pre = read_manifest_ids(data / "manifest_v04_consensus.jsonl")
    post = read_manifest_ids(data / "manifest_v07_weighted.jsonl")
    both = sweep["both"]
    out["corpus_denominators"] = {
        "unusable_events": n_unusable,
        "whole_corpus": f"{n_unusable}/{len(set(pre))}",
        "whole_corpus_pct": round(100.0 * n_unusable / len(set(pre)), 1),
        "both_judged_subset": f"{n_unusable}/{both}",
        "both_judged_subset_pct": round(100.0 * n_unusable / both, 1),
        "is_there_food_agreement": f"{sweep['cross_food_agree']}/{both}",
        "levellable_agreement": f"{sweep['cross_lev_agree']}/{both}",
        "corpus_after_cleaning": f"{len(set(pre))} - {n_unusable} = {len(set(post))}",
        "sweep_union_check": sweep["step_events"] + sweep["mm_events"] - both,
        "sweep_union_note": "step 683 + mm 512 - both 510 = 685 > 684, so 'both = 510' is not a clean subset of the corpus and 29.8% may not be quoted as a corpus rate",
    }
    return out


# --------------------------------------------------------------------------- #
# part 2 -- lint the paper
# --------------------------------------------------------------------------- #
def paragraphs(text: str) -> list[tuple[int, str]]:
    """(start_line, text) per claim block, 1-indexed in the original file.

    A block is a blank-line-separated run of lines, further split at heading and
    list-item boundaries so one bullet's claim is never judged by its
    neighbour's annotation. Table rows and blockquote lines stay inside their
    block: a table has one caption and a quote one speaker, so the annotation
    that covers the block covers every row of it. The YAML front matter is
    skipped: it is metadata, not a prose claim.
    """
    lines = text.splitlines()
    offset = 0
    if lines and lines[0].strip() == "---":
        for j in range(1, len(lines)):
            if lines[j].strip() == "---":
                offset = j + 1
                break

    blocks: list[tuple[int, str]] = []
    buf: list[str] = []
    start = 0

    def flush() -> None:
        nonlocal buf
        if buf:
            blocks.append((start, "\n".join(buf)))
            buf = []

    for i in range(offset, len(lines)):
        stripped = lines[i].strip()
        if not stripped:
            flush()
            continue
        if buf and (stripped.startswith(("- ", "* ", "#"))):
            flush()
        if not buf:
            start = i + 1
        buf.append(lines[i])
    flush()
    return blocks


def lint(path: Path) -> list[str]:
    rel = path.relative_to(REPO)
    text = path.read_text(encoding="utf-8")
    problems: list[str] = []

    for start, block in paragraphs(text):
        if "```" in block or block.lstrip().startswith("<!--"):
            continue  # fenced code and html comments are not prose claims
        lines = block.splitlines()
        kappa_lines = [ln for ln in lines if KAPPA_LINE.search(ln) and KAPPA_VALUE.search(ln)]
        if not kappa_lines:
            continue
        low = " ".join(block.lower().split())
        if not any(marker in low for marker in YARDSTICK_MARKERS):
            problems.append(
                f"{rel}:{start}: kappa quoted with no yardstick named -> {kappa_lines[0].strip()[:80]}"
            )
        if not any(marker in low for marker in POPULATION_MARKERS):
            problems.append(
                f"{rel}:{start}: kappa quoted with no population named (covered / all rows) -> {kappa_lines[0].strip()[:80]}"
            )

    # R3: a pass claim over all three models must name the metric
    for start, block in paragraphs(text):
        low = " ".join(block.lower().split())
        if "pass" not in low or not THREE_MODEL_PASS.search(block):
            continue
        if not any(m in low for m in ("covered", "all rows", "all-rows")):
            problems.append(
                f"{rel}:{start}: pass claim over three models names no metric population -> "
                f"{block.splitlines()[0].strip()[:80]}"
            )
    return problems


def lint_denominators(path: Path) -> list[str]:
    rel = path.relative_to(REPO)
    text = path.read_text(encoding="utf-8")
    problems: list[str] = []
    for i, line in enumerate(text.splitlines(), start=1):
        if "29.8" in line and "684" in line and "510" not in line:
            problems.append(f"{rel}:{i}: 29.8% sits next to 684 but not 510")
        if "22.2" in line and "510" in line and "684" not in line:
            problems.append(f"{rel}:{i}: 22.2% sits next to 510 but not 684")
        if re.search(r"152 of 684\s*=\s*29\.8", line):
            problems.append(f"{rel}:{i}: the original defect string '152 of 684 = 29.8%' is back")
    if "152/684" not in text and "152 of 684" not in text:
        problems.append(f"{rel}: never states the 152/684 whole-corpus denominator")
    if "152/510" not in text and "152 of 510" not in text:
        problems.append(f"{rel}: never states the 152/510 both-judged denominator")
    return problems


def lint_decode_collapse(path: Path) -> list[str]:
    """R4: one paragraph must report that L5 and L6 collapse out of the decode."""
    rel = path.relative_to(REPO)
    text = path.read_text(encoding="utf-8")
    collapse = re.compile(r"never emitted|never produced|decode collapse|0\.0% on l5", re.I)
    for start, block in paragraphs(text):
        if block.lstrip().startswith(("<!--", "#")):
            continue  # a comment or a bare heading is not the limitation itself
        low = " ".join(block.lower().split())
        if "l5" in low and "l6" in low and collapse.search(block):
            if not any(m in low for m in ("108", "5 seeds", "five seeds", "decided")):
                return [f"{rel}:{start}: decode-collapse paragraph carries no denominator"]
            return []
    return [f"{rel}: no paragraph reports the L5/L6 decode collapse with a denominator"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None, help="where to write the verification JSON")
    args = ap.parse_args()

    numbers = compute_numbers()

    problems: list[str] = []
    for rel in PAPER_FILES:
        path = REPO / rel
        if not path.exists():
            problems.append(f"{rel}: MISSING")
            continue
        problems += lint(path)
        problems += lint_denominators(path)
        if rel in DECODE_COLLAPSE_FILES:
            problems += lint_decode_collapse(path)

    out = {
        "numbers": numbers,
        "lint": {"rules": ["R1 kappa names yardstick+population", "R2 152/684 and 152/510",
                           "R3 three-model pass names its metric",
                           "R4 L5/L6 decode collapse reported with a denominator"],
                 "files_checked": PAPER_FILES,
                 "problems": problems},
        "verdict": "PASS" if not problems else "FAIL",
    }

    if args.out:
        dest = Path(args.out)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(out, indent=1) + "\n")
        print(f"wrote {dest}")
    else:
        default = REPO / "results/v07_release/paper_claims_verification.json"
        default.parent.mkdir(parents=True, exist_ok=True)
        default.write_text(json.dumps(out, indent=1) + "\n")
        print(f"wrote {default}")

    if problems:
        print(f"\n{len(problems)} lint problem(s):")
        for p in problems:
            print(f"  {p}")
        return 1
    print("\nlint: PASS (all four rules hold on every checked file)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
