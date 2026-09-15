"""Synthetic flow-test benchmark: ``python -m flowtest.bench.run_bench``.

Generates ``--n`` varied synthetic syringe-flow videos over a deterministic
parameter grid, grades each with ``flowtest.pipeline.grade_video``, and writes
a JSON + markdown summary (overall and per-bucket MAE, abstention rates,
level confusion counts, abstention-reason counts).

Run from the ``flowtest`` package directory so ``flowtest.*`` imports resolve::

    python -m flowtest.bench.run_bench --n 100 --out results_n100

Synthetic fixtures are pipeline-development checks only; this benchmark is
NOT evidence of performance on real liquids or real phone videos, and it is
not a safety benchmark.
"""

from __future__ import annotations

import argparse
import itertools
import json
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from flowtest.pipeline import FlowTestConfig, _level_for_volume, grade_video
from flowtest.synthetic import SyntheticSpec, generate_video

#: Documented parameter grid. Enumerated deterministically via
#: ``itertools.product`` in the listed order; case ``i`` uses combo
#: ``i % len(grid)`` with seed ``1000 + i``.
GRID: dict[str, list[Any]] = {
    "residual_ml": [0.5, 2.5, 4.25, 6.0, 8.4, 10.0],
    "tilt_deg": [0.0, 2.0, 4.0],
    "bubbles": [False, True],
    "lighting_factor": [1.0, 0.75, 1.15],
}
#: Fixed capture parameters (kept constant to bound runtime; documented here).
FIXED_PARAMS: dict[str, Any] = {"fps": 8.0, "width": 640, "height": 480}
#: Level boundaries of the grader; used for the near/far volume bucket.
BOUNDARY_ML = (1.0, 4.0, 8.0)
BOUNDARY_MARGIN_ML = 0.5
LEVELS = ("L0", "L1", "L2", "L3", "L4")


def build_grid() -> list[dict[str, Any]]:
    """Deterministic cartesian product of GRID (108 combos in fixed order)."""

    keys = list(GRID)
    return [dict(zip(keys, combo)) for combo in itertools.product(*(GRID[key] for key in keys))]


def _apply_lighting(src: Path, dst: Path, factor: float) -> Path:
    """Re-encode a video with a per-frame brightness scale factor.

    factor 1.0 returns the source untouched; <1 darkens, >1 brightens
    (clipped). Deterministic: pure per-pixel scaling plus mp4v re-encode.
    """

    if factor == 1.0:
        return src
    reader = cv2.VideoCapture(str(src))
    if not reader.isOpened():
        raise RuntimeError(f"cannot open {src} for lighting transform")
    fps = reader.get(cv2.CAP_PROP_FPS) or 8.0
    width = int(reader.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(str(dst), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        reader.release()
        raise RuntimeError("OpenCV could not create an mp4v video")
    try:
        while True:
            ok, frame = reader.read()
            if not ok:
                break
            writer.write(np.clip(frame.astype(np.float32) * factor, 0, 255).astype(np.uint8))
    finally:
        reader.release()
        writer.release()
    return dst


def _volume_bucket(residual_ml: float) -> str:
    distance = min(abs(residual_ml - boundary) for boundary in BOUNDARY_ML)
    return "near_boundary" if distance <= BOUNDARY_MARGIN_ML else "far_from_boundary"


def _ground_truth_level(residual_ml: float, config: FlowTestConfig) -> str:
    flowed = max(0.0, 10.0 - residual_ml)
    return _level_for_volume(residual_ml, flowed, config)


def _bucket_stats(cases: list[dict[str, Any]]) -> dict[str, Any]:
    errors = [case["absolute_error_ml"] for case in cases if case["absolute_error_ml"] is not None]
    graded_errors = [
        case["absolute_error_ml"] for case in cases if case["absolute_error_ml"] is not None and case["status"] == "graded"
    ]
    abstained = sum(1 for case in cases if case["status"] == "abstain")
    return {
        "n": len(cases),
        "graded": len(cases) - abstained,
        "abstained": abstained,
        "abstention_rate": round(abstained / len(cases), 4) if cases else None,
        "mae_ml": round(float(np.mean(errors)), 4) if errors else None,
        "mae_ml_graded_only": round(float(np.mean(graded_errors)), 4) if graded_errors else None,
        "n_with_residual_estimate": len(errors),
    }


def _confusion_counts(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counts = Counter((case["ground_truth_level"], case["predicted_level"] or "abstain") for case in cases)
    return [{"ground_truth": gt, "predicted": pred, "count": count} for (gt, pred), count in sorted(counts.items())]


def run_bench(n: int, out_dir: str | Path) -> dict[str, Any]:
    """Generate, grade, and summarise ``n`` synthetic videos."""

    if n < 1:
        raise ValueError("n must be >= 1")
    grid = build_grid()
    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)
    cases: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="flowtest_bench_") as tmp:
        workdir = Path(tmp)
        for index in range(n):
            combo = grid[index % len(grid)]
            spec = SyntheticSpec(
                residual_ml=combo["residual_ml"],
                fps=FIXED_PARAMS["fps"],
                width=FIXED_PARAMS["width"],
                height=FIXED_PARAMS["height"],
                tilt_deg=combo["tilt_deg"],
                bubbles=combo["bubbles"],
                seed=1000 + index,
            )
            raw_path = workdir / f"case_{index:03d}_raw.mp4"
            generate_video(raw_path, spec)
            video_path = raw_path
            if combo["lighting_factor"] != 1.0:
                video_path = workdir / f"case_{index:03d}.mp4"
                _apply_lighting(raw_path, video_path, combo["lighting_factor"])
            result = grade_video(video_path)
            gt_level = _ground_truth_level(combo["residual_ml"], FlowTestConfig())
            estimate = result.estimated_residual_ml
            error = abs(float(estimate) - float(combo["residual_ml"])) if estimate is not None else None
            cases.append(
                {
                    "case_id": f"case_{index:03d}",
                    "spec": {**combo, "seed": 1000 + index, **FIXED_PARAMS},
                    "volume_bucket": _volume_bucket(combo["residual_ml"]),
                    "ground_truth_level": gt_level,
                    "status": result.status,
                    "predicted_level": result.iddsi_level,
                    "estimated_residual_ml": estimate,
                    "absolute_error_ml": round(error, 4) if error is not None else None,
                    "boundary_distance_ml": result.boundary_distance_ml,
                    "abstention_reasons": list(result.abstention_reasons),
                }
            )

    per_bucket: dict[str, dict[str, Any]] = {}
    for dimension in ("residual_ml", "tilt_deg", "bubbles", "lighting_factor", "volume_bucket"):
        groups: dict[str, list[dict[str, Any]]] = {}
        for case in cases:
            key = case["volume_bucket"] if dimension == "volume_bucket" else case["spec"][dimension]
            groups.setdefault(str(key), []).append(case)
        per_bucket[dimension] = {key: _bucket_stats(group) for key, group in sorted(groups.items())}
    reason_counts = Counter(
        reason for case in cases for reason in case["abstention_reasons"]
    )

    report: dict[str, Any] = {
        "scope": "synthetic pipeline-development benchmark only; not real-video validation and not a safety benchmark",
        "n": n,
        "grid": GRID,
        "fixed_params": FIXED_PARAMS,
        "seed_rule": "case i uses grid combo i % len(grid) with SyntheticSpec seed 1000 + i",
        "overall": _bucket_stats(cases),
        "per_bucket": per_bucket,
        "confusion_counts": _confusion_counts(cases),
        "abstention_reasons": dict(sorted(reason_counts.items())),
        "cases": cases,
    }
    (output / "bench_summary.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output / "bench_summary.md").write_text(render_markdown(report), encoding="utf-8")
    return report


def _md_row(cells: list[Any]) -> str:
    return "| " + " | ".join(str(cell) for cell in cells) + " |"


def render_markdown(report: dict[str, Any]) -> str:
    lines: list[str] = []
    overall = report["overall"]
    lines.append(f"# Flow-Test synthetic bench (n={report['n']})")
    lines.append("")
    lines.append(report["scope"])
    lines.append("")
    lines.append("Grid: " + json.dumps(report["grid"]) + "; fixed: " + json.dumps(report["fixed_params"]) + "; " + report["seed_rule"])
    lines.append("")
    lines.append(_md_row(["n", "graded", "abstained", "abstention rate", "MAE mL (all estimates)", "MAE mL (graded only)", "n with estimate"]))
    lines.append(_md_row(["---"] * 7))
    lines.append(
        _md_row(
            [
                overall["n"],
                overall["graded"],
                overall["abstained"],
                overall["abstention_rate"],
                overall["mae_ml"],
                overall["mae_ml_graded_only"],
                overall["n_with_residual_estimate"],
            ]
        )
    )
    for dimension in ("volume_bucket", "residual_ml", "tilt_deg", "bubbles", "lighting_factor"):
        lines.append("")
        lines.append(f"## Per bucket: {dimension}")
        lines.append(_md_row(["bucket", "n", "graded", "abstained", "abstention rate", "MAE mL", "MAE mL (graded only)"]))
        lines.append(_md_row(["---"] * 7))
        for key, stats in report["per_bucket"][dimension].items():
            lines.append(
                _md_row(
                    [
                        key,
                        stats["n"],
                        stats["graded"],
                        stats["abstained"],
                        stats["abstention_rate"],
                        stats["mae_ml"],
                        stats["mae_ml_graded_only"],
                    ]
                )
            )
    lines.append("")
    lines.append("## Confusion (rows = ground truth, columns = prediction; last column = grader abstained)")
    columns = list(LEVELS) + ["abstain"]
    lines.append(_md_row(["gt \\ pred"] + columns))
    lines.append(_md_row(["---"] * (len(columns) + 1)))
    matrix = {gt: {pred: 0 for pred in columns} for gt in LEVELS}
    for entry in report["confusion_counts"]:
        matrix.setdefault(entry["ground_truth"], {}).setdefault(entry["predicted"], 0)
        matrix[entry["ground_truth"]][entry["predicted"]] = entry["count"]
    for gt in LEVELS:
        if gt in matrix:
            lines.append(_md_row([gt] + [matrix[gt].get(pred, 0) for pred in columns]))
    if report["abstention_reasons"]:
        lines.append("")
        lines.append("## Abstention reasons")
        lines.append(_md_row(["code", "count"]))
        lines.append(_md_row(["---", "---"]))
        for code, count in report["abstention_reasons"].items():
            lines.append(_md_row([code, count]))
    lines.append("")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Synthetic flow-test benchmark for the IDDSI grader")
    parser.add_argument("--n", type=int, default=100, help="number of synthetic videos (default: 100)")
    parser.add_argument("--out", type=Path, default=Path("bench_out"), help="output directory (default: bench_out)")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = run_bench(args.n, args.out)
    print(render_markdown(report))
    print(f"\n[bench] wrote {args.out / 'bench_summary.json'} and {args.out / 'bench_summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
