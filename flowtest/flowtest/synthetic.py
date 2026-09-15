"""Synthetic fixtures for deterministic, end-to-end pipeline testing.

Synthetic measurements are pipeline-development checks only. They are not
evidence of performance on real liquids or real phone videos.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .pipeline import grade_video


@dataclass(frozen=True)
class SyntheticSpec:
    residual_ml: float
    fps: float = 10.0
    release_s: float = 1.0
    post_release_s: float = 10.5
    width: int = 640
    height: int = 480
    pixels_per_mm: float = 2.5
    barrel_length_mm: float = 61.5
    tilt_deg: float = 0.0
    bubbles: bool = False
    occlusion: bool = False
    fiducial: bool = True
    seed: int = 7


def _draw_fiducial(frame: np.ndarray, pixels_per_mm: float) -> None:
    card_w = int(round(85.60 * pixels_per_mm))
    card_h = int(round(53.98 * pixels_per_mm))
    x0, y0 = 38, frame.shape[0] - card_h - 28
    cv2.rectangle(frame, (x0, y0), (x0 + card_w, y0 + card_h), (25, 25, 25), 4)
    # Asymmetric interior marks make the object visibly a calibration card but
    # do not affect the outer metric rectangle detector.
    cv2.rectangle(frame, (x0 + 12, y0 + 12), (x0 + 27, y0 + 27), (25, 25, 25), -1)
    cv2.putText(frame, "85.60 x 53.98 mm", (x0 + 38, y0 + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (30, 30, 30), 1, cv2.LINE_AA)


def _syringe_layer(spec: SyntheticSpec, volume_ml: float, before_release: bool) -> np.ndarray:
    layer = np.full((spec.height, spec.width, 3), 255, dtype=np.uint8)
    length = int(round(spec.barrel_length_mm * spec.pixels_per_mm))
    x0, x1 = 375, 435
    y0, y1 = 74, 74 + length
    cv2.rectangle(layer, (x0, y0), (x1, y1), (18, 18, 18), 4)
    meniscus_y = int(round(y1 - np.clip(volume_ml, 0, 10) / 10.0 * (y1 - y0)))
    if volume_ml > 0:
        cv2.rectangle(layer, (x0 + 4, meniscus_y), (x1 - 4, y1 - 4), (205, 125, 45), -1)
        cv2.line(layer, (x0 + 5, meniscus_y), (x1 - 5, meniscus_y), (135, 65, 20), 2)
    for mark in range(11):
        y = int(round(y1 - mark / 10 * (y1 - y0)))
        tick = 17 if mark % 5 == 0 else 10
        cv2.line(layer, (x1, y), (x1 + tick, y), (20, 20, 20), 2)
    cv2.rectangle(layer, (x0 + 22, y1), (x1 - 22, y1 + 26), (25, 25, 25), 2)
    cv2.line(layer, ((x0 + x1) // 2, y1 + 26), ((x0 + x1) // 2, y1 + 40), (25, 25, 25), 3)
    if before_release:
        cv2.circle(layer, ((x0 + x1) // 2 + 3, y1 + 31), 17, (80, 145, 225), -1)
    if spec.bubbles and volume_ml > 0:
        rng = np.random.default_rng(spec.seed)
        low = max(meniscus_y + 9, y0 + 9)
        high = max(low + 1, y1 - 9)
        for _ in range(7):
            center = (int(rng.integers(x0 + 12, x1 - 12)), int(rng.integers(low, high)))
            cv2.circle(layer, center, 4, (245, 245, 245), -1)
            cv2.circle(layer, center, 4, (65, 65, 65), 1)
    if spec.occlusion and not before_release:
        cv2.rectangle(layer, (x0 - 9, meniscus_y - 28), (x1 + 9, meniscus_y + 62), (118, 118, 118), -1)

    if spec.tilt_deg:
        matrix = cv2.getRotationMatrix2D(((x0 + x1) / 2, (y0 + y1) / 2), -spec.tilt_deg, 1.0)
        layer = cv2.warpAffine(layer, matrix, (spec.width, spec.height), borderValue=(255, 255, 255))
    return layer


def generate_video(path: str | Path, spec: SyntheticSpec) -> Path:
    """Write one deterministic synthetic MP4 and return its path."""

    if not 0 <= spec.residual_ml <= 10:
        raise ValueError("residual_ml must be between 0 and 10")
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output),
        cv2.VideoWriter_fourcc(*"mp4v"),
        spec.fps,
        (spec.width, spec.height),
    )
    if not writer.isOpened():
        raise RuntimeError("OpenCV could not create an mp4v video")
    total_s = spec.release_s + spec.post_release_s
    frame_count = int(round(total_s * spec.fps)) + 1
    try:
        for index in range(frame_count):
            time_s = index / spec.fps
            if time_s < spec.release_s:
                volume = 10.0
            else:
                progress = min(1.0, (time_s - spec.release_s) / 10.0)
                # Smooth deterministic descent with endpoints fixed exactly.
                eased = progress * progress * (3.0 - 2.0 * progress)
                volume = 10.0 + (spec.residual_ml - 10.0) * eased
            frame = np.full((spec.height, spec.width, 3), 247, dtype=np.uint8)
            if spec.fiducial:
                _draw_fiducial(frame, spec.pixels_per_mm)
            layer = _syringe_layer(spec, volume, time_s < spec.release_s)
            mask = np.any(layer < 250, axis=2)
            frame[mask] = layer[mask]
            writer.write(frame)
    finally:
        writer.release()
    return output


def evaluate_synthetic(output_dir: Path, residuals: list[float] | None = None) -> dict[str, object]:
    residuals = residuals or [0.25, 2.5, 6.0, 9.0, 10.0]
    rows: list[dict[str, object]] = []
    absolute_errors: list[float] = []
    for residual in residuals:
        path = output_dir / f"synthetic_{residual:.2f}.mp4"
        generate_video(path, SyntheticSpec(residual_ml=residual))
        result = grade_video(path)
        estimate = result.estimated_residual_ml
        error = abs(float(estimate) - residual) if estimate is not None else None
        if error is not None:
            absolute_errors.append(error)
        rows.append(
            {
                "ground_truth_residual_ml": residual,
                "estimated_residual_ml": estimate,
                "absolute_error_ml": round(error, 4) if error is not None else None,
                "status": result.status,
                "iddsi_level": result.iddsi_level,
                "abstention_reasons": result.abstention_reasons,
            }
        )
    return {
        "scope": "synthetic pipeline-development fixtures only; not real-video validation",
        "count": len(rows),
        "mae_ml": round(float(np.mean(absolute_errors)), 4) if absolute_errors else None,
        "cases": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate deterministic synthetic Flow-Test videos")
    parser.add_argument("--out", type=Path, help="single output MP4")
    parser.add_argument("--residual-ml", type=float, default=6.0)
    parser.add_argument("--evaluate", type=Path, help="directory for a synthetic MAE suite")
    args = parser.parse_args()
    if args.evaluate:
        report = evaluate_synthetic(args.evaluate)
        report_path = args.evaluate / "synthetic_mae.json"
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0
    if args.out is None:
        parser.error("--out is required unless --evaluate is used")
    generated = generate_video(args.out, SyntheticSpec(residual_ml=args.residual_ml))
    print(generated)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
