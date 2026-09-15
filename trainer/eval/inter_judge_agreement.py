#!/usr/bin/env python3
"""inter_judge_agreement.py — regenerate the headline inter-judge agreement stat.

Compares tier-1 weak labels (events.jsonl `level_weak`, Qwen3-VL-2B) against the
tier-2 sidecar (tier2_labels.jsonl `level_tier2`, Qwen3-VL-8B) over the event_ids
present in BOTH. Emits JSON + Markdown: n, exact agreement, Cohen's kappa
(unweighted + quadratic-weighted), per-level cross-tab. This is the evidence
behind the release's negative finding — every published number must come from
this script's output, never from memory.

Usage: python3 inter_judge_agreement.py <dataset_dir> [<dataset_dir> ...] --out <prefix>
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

LEVELS = [3, 4, 5, 6, 7]


def _level(v):
    try:
        i = int(v)
        return i if i in LEVELS else None
    except (TypeError, ValueError):
        return None


def load_pairs(root: Path) -> list[tuple[str, int, int]]:
    tier1: dict[str, int] = {}
    with (root / "events.jsonl").open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue
            lv = _level(ev.get("level_weak"))
            if lv is not None:
                tier1[str(ev.get("event_id"))] = lv
    pairs = []
    sc = root / "tier2_labels.jsonl"
    if not sc.is_file():
        return pairs
    with sc.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            eid = str(row.get("event_id"))
            lv2 = _level(row.get("level_tier2"))
            if eid in tier1 and lv2 is not None:
                pairs.append((eid, tier1[eid], lv2))
    return pairs


def kappa(pairs: list[tuple[str, int, int]], weighted: bool) -> float:
    n = len(pairs)
    if n == 0:
        return float("nan")
    c1 = Counter(t1 for _, t1, _ in pairs)
    c2 = Counter(t2 for _, _, t2 in pairs)
    if weighted:
        num = sum((t1 - t2) ** 2 for _, t1, t2 in pairs) / n
        den = sum(c1[a] * c2[b] * (a - b) ** 2 for a in LEVELS for b in LEVELS) / (n * n)
        return 1.0 - num / den if den else float("nan")
    po = sum(1 for _, t1, t2 in pairs if t1 == t2) / n
    pe = sum(c1[k] * c2[k] for k in LEVELS) / (n * n)
    return (po - pe) / (1 - pe) if pe < 1 else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("datasets", nargs="+")
    ap.add_argument("--out", required=True, help="output prefix (.json/.md appended)")
    args = ap.parse_args()

    pairs: list[tuple[str, int, int]] = []
    per_ds = {}
    for d in args.datasets:
        p = load_pairs(Path(d))
        per_ds[Path(d).name] = len(p)
        pairs.extend(p)

    n = len(pairs)
    exact = sum(1 for _, a, b in pairs if a == b) / n if n else float("nan")
    tab = {a: {b: 0 for b in LEVELS} for a in LEVELS}
    for _, a, b in pairs:
        tab[a][b] += 1

    result = {
        "n_pairs": n,
        "per_dataset": per_ds,
        "exact_agreement": round(exact, 4) if n else None,
        "kappa_unweighted": round(kappa(pairs, False), 4) if n else None,
        "kappa_quadratic": round(kappa(pairs, True), 4) if n else None,
        "crosstab_tier1_rows_tier2_cols": {str(a): {str(b): tab[a][b] for b in LEVELS} for a in LEVELS},
        "judges": {"tier1": "Qwen3-VL-2B (level_weak in events.jsonl)",
                    "tier2": "Qwen3-VL-8B-Instruct (tier2_labels.jsonl)"},
    }

    out = Path(args.out)
    out.with_suffix(".json").write_text(json.dumps(result, indent=2))
    lines = [
        "# Inter-judge agreement (tier-1 Qwen3-VL-2B vs tier-2 Qwen3-VL-8B)",
        "",
        f"- pairs judged by both: **{n}** ({', '.join(f'{k}: {v}' for k, v in per_ds.items())})",
        f"- exact agreement: **{exact:.1%}**" if n else "- exact agreement: n/a",
        f"- Cohen's kappa (unweighted): **{result['kappa_unweighted']}**",
        f"- Cohen's kappa (quadratic-weighted): **{result['kappa_quadratic']}**",
        "",
        "Cross-tab (rows = tier-1 level, cols = tier-2 level):",
        "",
        "| t1\\t2 | " + " | ".join(f"L{l}" for l in LEVELS) + " |",
        "|---|" + "---:|" * len(LEVELS),
    ]
    for a in LEVELS:
        lines.append(f"| L{a} | " + " | ".join(str(tab[a][b]) for b in LEVELS) + " |")
    lines += ["", "Chance exact-agreement for 5 classes ≈ 20% (uniform) — see release card for interpretation.", ""]
    out.with_suffix(".md").write_text("\n".join(lines))
    print(json.dumps({k: result[k] for k in ("n_pairs", "exact_agreement", "kappa_unweighted", "kappa_quadratic")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
