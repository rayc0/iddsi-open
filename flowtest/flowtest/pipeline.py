"""Deterministic OpenCV implementation of the syringe Flow-Test grader.

The supported v0 capture protocol uses a standard ID-1 card-sized fiducial
(85.60 x 53.98 mm) in the same plane as the syringe. No learned models or
network services are used.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .abstention import CHECK_FAILURE_CODES, describe_all
from .models import Check, GradeResult
from .video_io import cleanup_source, open_video


@dataclass(frozen=True)
class FlowTestConfig:
    barrel_length_mm: float = 61.5
    barrel_tolerance_mm: float = 3.0
    fiducial_width_mm: float = 85.60
    fiducial_height_mm: float = 53.98
    max_vertical_deviation_deg: float = 5.0
    boundary_margin_ml: float = 0.5
    measurement_delay_s: float = 10.0
    min_detectable_flow_ml: float = 0.20
    max_occluded_fraction: float = 0.20
    min_meniscus_confidence: float = 0.35
    max_artifact_fraction: float = 0.025
    normalized_width_px: int = 120
    normalized_height_px: int = 308
    # When False, a missing fiducial no longer forces abstention: the grader
    # runs in uncalibrated mode (pixel scale unverified, syringe size not
    # checked) and flags the result with the "uncalibrated_scale_mode" code.
    fiducial_required: bool = True
    # --- W18 robustness knobs (all deterministic) ---
    # Minimum barrel-detection confidence to accept a geometry instead of
    # abstaining with "syringe_not_detected" BEFORE normalisation (W4 §6.1).
    min_barrel_confidence: float = 0.30
    # Re-detect the barrel every this many seconds of video (tracked in
    # between); per-frame tracking replaces the W4 frame-0 freeze (W4 §6.2).
    redetect_interval_s: float = 0.5
    # Maximum allowed centre jump (fraction of barrel length) between the
    # running tracked geometry and a fresh detection before the detection is
    # rejected as a tracking outlier.
    max_center_jump_fraction: float = 0.60
    # If more than this fraction of re-detection attempts fail or are
    # rejected, tracking is considered lost and the grader abstains with
    # "syringe_tracking_lost" (W4 §6.2).
    max_tracking_loss_fraction: float = 0.50
    # Rolling-reference window (seconds) for nozzle-release change detection,
    # replacing the frame-0 fixed reference (W4 §6.3).
    release_reference_window_s: float = 0.75
    # Volume-trace outlier rejection: a point further than this many robust
    # (MAD-based) standard deviations from the rolling median is replaced by
    # the rolling median before baseline/release/residual maths (W4 §6.2).
    volume_outlier_mad_k: float = 4.0


@dataclass(frozen=True)
class Geometry:
    center: np.ndarray
    vertical: np.ndarray
    horizontal: np.ndarray
    length_px: float
    width_px: float
    vertical_deviation_deg: float
    pixels_per_mm: float | None
    fiducial_size_px: tuple[float, float] | None

    @property
    def barrel_length_mm(self) -> float | None:
        if self.pixels_per_mm is None:
            return None
        return self.length_px / self.pixels_per_mm


def _detect_fiducial(frame: np.ndarray, config: FlowTestConfig) -> tuple[float | None, tuple[float, float] | None]:
    """Find a dark-bordered ID-1 rectangle and return pixels/mm.

    A plain, high-contrast, landscape rectangle is deliberately used instead
    of ArUco so opencv-contrib is not required. The rectangle must be printed
    at 85.60 x 53.98 mm and placed in the syringe plane.
    """

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, dark = cv2.threshold(blurred, 90, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(dark, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    target_ratio = config.fiducial_width_mm / config.fiducial_height_mm
    candidates: list[tuple[float, float, float]] = []
    frame_area = frame.shape[0] * frame.shape[1]
    for contour in contours:
        area = abs(cv2.contourArea(contour))
        if area < frame_area * 0.01 or area > frame_area * 0.45:
            continue
        rect = cv2.minAreaRect(contour)
        a, b = rect[1]
        if min(a, b) < 30:
            continue
        long_px, short_px = max(a, b), min(a, b)
        ratio_error = abs(long_px / short_px - target_ratio)
        if ratio_error <= 0.16:
            candidates.append((ratio_error, long_px, short_px))
    if not candidates:
        return None, None
    _, long_px, short_px = min(candidates, key=lambda item: item[0])
    px_per_mm = 0.5 * (
        long_px / config.fiducial_width_mm
        + short_px / config.fiducial_height_mm
    )
    return float(px_per_mm), (float(long_px), float(short_px))


def _line_angle_from_vertical(line: np.ndarray) -> float:
    x1, y1, x2, y2 = (float(v) for v in line)
    # Hough segment endpoint order is arbitrary. Directing every segment from
    # image top to bottom makes the signed tilt comparable across both sides.
    if y2 < y1:
        x1, y1, x2, y2 = x2, y2, x1, y1
    return float(np.degrees(np.arctan2(x2 - x1, y2 - y1)))


def _edge_support(edges: np.ndarray, line: np.ndarray) -> float:
    """Fraction of sampled points along a line that land on/near an edge pixel.

    A genuine barrel side has continuous Canny support along its full length;
    a background line-pair that merely fits the aspect filter usually does not
    (W4 §5a: the frame-0 detector locked onto background edges). Sampling is
    deterministic: 24 evenly spaced points, 3x3 neighbourhood tolerance.
    """

    x1, y1, x2, y2 = (float(v) for v in line)
    samples = 24
    height, width = edges.shape[:2]
    hits = 0
    for t in np.linspace(0.0, 1.0, samples):
        x = int(round(x1 + (x2 - x1) * t))
        y = int(round(y1 + (y2 - y1) * t))
        x0, x1c = max(0, x - 1), min(width, x + 2)
        y0, y1c = max(0, y - 1), min(height, y + 2)
        if edges[y0:y1c, x0:x1c].max() > 0:
            hits += 1
    return hits / samples


def _flange_tip_evidence(gray: np.ndarray, geometry: "Geometry") -> float:
    """Score 0..1 for syringe prior features near the barrel ends (W4 §6.1).

    Looks for (a) a dark horizontal flange/plunger band just above the barrel
    top and (b) a dark tip/cone structure just below the barrel bottom, both
    centred on the barrel axis. Either feature present is evidence the line
    pair is a real syringe rather than scene edges. Fully deterministic.
    """

    height, width = gray.shape[:2]
    center, vertical, horizontal = geometry.center, geometry.vertical, geometry.horizontal
    half_len = geometry.length_px / 2
    band_depth = max(3.0, geometry.length_px * 0.10)
    half_w = geometry.width_px * 0.55
    score = 0.0
    for sign in (-1.0, 1.0):  # -1: above top (flange), +1: below bottom (tip)
        band_center = center + vertical * (half_len + band_depth * 0.5) * sign
        corners = np.array(
            [
                band_center - vertical * band_depth / 2 - horizontal * half_w,
                band_center - vertical * band_depth / 2 + horizontal * half_w,
                band_center + vertical * band_depth / 2 + horizontal * half_w,
                band_center + vertical * band_depth / 2 - horizontal * half_w,
            ],
            dtype=np.float32,
        )
        x, y, w, h = cv2.boundingRect(corners.astype(np.int32))
        x, y = max(0, x), max(0, y)
        roi = gray[y : min(height, y + h), x : min(width, x + w)]
        if roi.size < 25:
            continue
        # A flange/tip is darker than the frame's bright background.
        if float(np.median(roi)) < 0.55 * float(np.median(gray)) + 40:
            score += 0.5
    return score


def _barrel_template(width_px: int, length_px: int) -> np.ndarray:
    """Deterministic edge template of an idealised syringe barrel.

    Two long parallel sides plus short flange/tip ticks — the minimal syringe
    prior from W4 §6.1. Rendered with cv2 only; no assets, no randomness.
    """

    pad = max(4, width_px // 4)
    h, w = length_px + 2 * pad, width_px + 2 * pad
    template = np.zeros((h, w), dtype=np.uint8)
    x0, x1 = pad, pad + width_px
    y0, y1 = pad, pad + length_px
    cv2.line(template, (x0, y0), (x0, y1), 255, 2)
    cv2.line(template, (x1, y0), (x1, y1), 255, 2)
    tick = max(3, width_px // 3)
    cv2.line(template, (x0 - tick, y0), (x1 + tick, y0), 255, 2)  # flange
    cv2.line(template, (x0 + tick, y1), (x1 - tick, y1), 255, 2)  # tip cone
    return template


def _detect_barrel_template(
    edges: np.ndarray, px_per_mm: float | None, config: FlowTestConfig
) -> tuple[float, "Geometry | None"]:
    """Template-matching fallback when the Hough line-pair finds nothing.

    Slides the deterministic barrel edge template over the Canny map at a
    small set of scales derived from the expected barrel size (or the frame
    height when no scale is available). Returns (best_score, Geometry|None).
    """

    height, width = edges.shape[:2]
    if px_per_mm:
        base_length = config.barrel_length_mm * px_per_mm
        scales = (0.9, 1.0, 1.1)
    else:
        base_length = height * 0.42
        scales = (0.85, 1.0, 1.15)
    best_score, best_geometry = 0.0, None
    for scale in scales:
        length_px = int(round(base_length * scale))
        width_px = int(round(length_px / 3.8))  # syringe prior aspect ~3.8:1
        if length_px < 40 or width_px < 10 or length_px >= height or width_px >= width:
            continue
        template = _barrel_template(width_px, length_px)
        if template.shape[0] >= height or template.shape[1] >= width:
            continue
        response = cv2.matchTemplate(edges, template, cv2.TM_CCOEFF_NORMED)
        _, peak, _, peak_loc = cv2.minMaxLoc(response)
        if peak <= best_score:
            continue
        px, py = peak_loc
        pad = max(4, width_px // 4)
        cx, cy = px + pad + width_px / 2, py + pad + length_px / 2
        geometry = Geometry(
            center=np.array([cx, cy]),
            vertical=np.array([0.0, 1.0]),
            horizontal=np.array([1.0, 0.0]),
            length_px=float(length_px),
            width_px=float(width_px),
            vertical_deviation_deg=0.0,
            pixels_per_mm=px_per_mm,
            fiducial_size_px=None,
        )
        best_score, best_geometry = float(peak), geometry
    return best_score, best_geometry


def _detect_barrel(frame: np.ndarray, px_per_mm: float | None, config: FlowTestConfig) -> Geometry | None:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 45, 140)
    min_length = max(50, int(frame.shape[0] * 0.14))
    raw = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 360,
        threshold=35,
        minLineLength=min_length,
        maxLineGap=12,
    )
    if raw is None:
        return None
    # OpenCV builds have returned both (N, 1, 4) and (1, N, 4).
    lines = [row.astype(float) for row in np.asarray(raw).reshape(-1, 4)]
    lines = [line for line in lines if abs(_line_angle_from_vertical(line)) <= 20]
    pairs: list[tuple[float, Geometry]] = []
    for index, first in enumerate(lines):
        for second in lines[index + 1 :]:
            angle_a = _line_angle_from_vertical(first)
            angle_b = _line_angle_from_vertical(second)
            if abs(angle_a - angle_b) > 3.0:
                continue
            pa = np.array([(first[0] + first[2]) / 2, (first[1] + first[3]) / 2])
            pb = np.array([(second[0] + second[2]) / 2, (second[1] + second[3]) / 2])
            angle = np.radians((angle_a + angle_b) / 2)
            vertical = np.array([np.sin(angle), np.cos(angle)])
            if vertical[1] < 0:
                vertical *= -1
            horizontal = np.array([vertical[1], -vertical[0]])
            if horizontal[0] < 0:
                horizontal *= -1
            separation = abs(float(np.dot(pb - pa, horizontal)))
            if separation < 18 or separation > frame.shape[1] * 0.28:
                continue
            lengths = [float(np.hypot(line[2] - line[0], line[3] - line[1])) for line in (first, second)]
            length = min(lengths)
            aspect = length / separation
            if not 1.7 <= aspect <= 6.0:
                continue
            projected_centers = [float(np.dot(point, vertical)) for point in (pa, pb)]
            if abs(projected_centers[0] - projected_centers[1]) > length * 0.12:
                continue
            center = (pa + pb) / 2
            geometry = Geometry(
                center=center,
                vertical=vertical,
                horizontal=horizontal,
                length_px=length,
                width_px=separation,
                vertical_deviation_deg=abs(float(np.degrees(angle))),
                pixels_per_mm=px_per_mm,
                fiducial_size_px=None,
            )
            expected_px = config.barrel_length_mm * px_per_mm if px_per_mm else length
            size_penalty = abs(length - expected_px) / max(expected_px, 1)
            score = length - 2.5 * abs(lengths[0] - lengths[1]) - 100 * size_penalty
            pairs.append((score, geometry))
    if not pairs:
        return None
    return max(pairs, key=lambda item: item[0])[1]


def _barrel_corners(geometry: Geometry, inset_fraction: float = -0.06) -> np.ndarray:
    half_length = geometry.length_px / 2
    half_width = geometry.width_px * (0.5 - inset_fraction)
    top = geometry.center - geometry.vertical * half_length
    bottom = geometry.center + geometry.vertical * half_length
    return np.float32(
        [
            top - geometry.horizontal * half_width,
            top + geometry.horizontal * half_width,
            bottom - geometry.horizontal * half_width,
            bottom + geometry.horizontal * half_width,
        ]
    )


def _normalise_barrel(frame: np.ndarray, geometry: Geometry, config: FlowTestConfig) -> np.ndarray:
    src = _barrel_corners(geometry)
    width, height = config.normalized_width_px, config.normalized_height_px
    dst = np.float32([[0, 0], [width - 1, 0], [0, height - 1], [width - 1, height - 1]])
    transform = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(frame, transform, (width, height))


def _nozzle_roi(frame: np.ndarray, geometry: Geometry) -> np.ndarray:
    bottom = geometry.center + geometry.vertical * geometry.length_px / 2
    center = bottom + geometry.vertical * geometry.length_px * 0.11
    half_h = geometry.length_px * 0.12
    half_w = geometry.width_px * 0.48
    points = np.array(
        [
            center - geometry.vertical * half_h - geometry.horizontal * half_w,
            center - geometry.vertical * half_h + geometry.horizontal * half_w,
            center + geometry.vertical * half_h + geometry.horizontal * half_w,
            center + geometry.vertical * half_h - geometry.horizontal * half_w,
        ],
        dtype=np.float32,
    )
    x, y, w, h = cv2.boundingRect(points.astype(np.int32))
    x, y = max(0, x), max(0, y)
    return frame[y : min(frame.shape[0], y + h), x : min(frame.shape[1], x + w)]


def _meniscus(normalized: np.ndarray) -> tuple[float, float, float]:
    """Return (volume mL, confidence 0..1, artifact fraction)."""

    height, width = normalized.shape[:2]
    x0, x1 = int(width * 0.14), int(width * 0.86)
    roi = normalized[:, x0:x1]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    saturation_profile = np.mean(hsv[:, :, 1] > 35, axis=1).astype(np.float32)
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY).astype(np.float32)
    intensity_profile = np.median(gray, axis=1)
    intensity_profile = cv2.GaussianBlur(intensity_profile[:, None], (1, 9), 0).ravel()
    gradient = np.abs(np.gradient(intensity_profile))

    # Ignore warped top/bottom outlines. Codec chroma bleed can make a black
    # border appear highly saturated even when the air column is colourless.
    margin = max(5, int(round(height * 0.026)))
    liquid_rows = saturation_profile > 0.48
    core_liquid_rows = liquid_rows[margin : height - margin]
    if np.mean(core_liquid_rows) > 0.96:
        row = 0.0
        confidence = min(1.0, float(np.mean(saturation_profile[margin : height - margin])))
    elif np.mean(core_liquid_rows) < 0.01:
        row = float(height - 1)
        confidence = 0.45 if float(np.max(gradient)) > 8 else 0.30
    else:
        transitions = np.where(core_liquid_rows)[0] + margin
        sat_row = float(transitions[0])
        lo, hi = max(1, int(sat_row) - 8), min(height - 2, int(sat_row) + 8)
        grad_row = float(lo + np.argmax(gradient[lo : hi + 1]))
        row = 0.75 * sat_row + 0.25 * grad_row
        before = float(np.mean(saturation_profile[max(0, int(row) - 6) : max(1, int(row) - 1)]))
        after = float(np.mean(saturation_profile[min(height - 1, int(row) + 1) : min(height, int(row) + 7)]))
        contrast = max(0.0, after - before)
        confidence = min(1.0, 0.65 * contrast + 0.35 * min(1.0, float(gradient[int(round(grad_row))]) / 35))

    volume = float(np.clip(10.0 * (height - 1 - row) / (height - 1), 0.0, 10.0))

    # Bubble/lump heuristic: unusually different connected regions within the
    # liquid, excluding a narrow band around the meniscus and barrel edges.
    # Leave codec-bleed margins around the horizontal meniscus and the dark
    # syringe base so neither is mislabelled as a bubble/lump.
    liquid_start = min(height - 1, int(row) + 10)
    artifact_bottom_margin = max(12, int(round(height * 0.05)))
    liquid_gray = gray[liquid_start : height - artifact_bottom_margin, 4:-4]
    artifact_fraction = 0.0
    if liquid_gray.size > 100:
        median = float(np.median(liquid_gray))
        mad = float(np.median(np.abs(liquid_gray - median)))
        threshold = max(28.0, 4.5 * mad)
        unusual = (np.abs(liquid_gray - median) > threshold).astype(np.uint8)
        count, _, stats, _ = cv2.connectedComponentsWithStats(unusual, 8)
        artifact_area = sum(int(stats[i, cv2.CC_STAT_AREA]) for i in range(1, count) if stats[i, cv2.CC_STAT_AREA] >= 10)
        artifact_fraction = artifact_area / float(liquid_gray.size)
    return volume, float(confidence), float(artifact_fraction)


def _border_occlusion(normalized: np.ndarray) -> float:
    gray = cv2.cvtColor(normalized, cv2.COLOR_BGR2GRAY)
    search_width = max(8, normalized.shape[1] // 5)
    left_columns = np.mean(gray[:, :search_width] < 92, axis=0)
    right_columns = np.mean(gray[:, -search_width:] < 92, axis=0)
    visibility = min(float(np.max(left_columns)), float(np.max(right_columns)))
    return float(np.clip(1.0 - visibility / 0.80, 0.0, 1.0))


def _sustained_first(values: np.ndarray, threshold: float, count: int = 2) -> int | None:
    above = values >= threshold
    if len(above) < count:
        return None
    runs = np.convolve(above.astype(np.int16), np.ones(count, dtype=np.int16), mode="valid")
    hits = np.where(runs == count)[0]
    return int(hits[0]) if len(hits) else None


def _level_for_volume(residual_ml: float, flowed_ml: float, config: FlowTestConfig) -> str:
    if residual_ml < 1.0:
        return "L0"
    if residual_ml < 4.0:
        return "L1"
    if residual_ml < 8.0:
        return "L2"
    if flowed_ml <= config.min_detectable_flow_ml:
        return "L4"
    return "L3"


def grade_video(path: str | Path, config: FlowTestConfig | None = None) -> GradeResult:
    """Grade one video and return a JSON-serialisable result object."""

    config = config or FlowTestConfig()
    video_path = Path(path)
    result = GradeResult(video=str(video_path))
    source = open_video(video_path)
    fps, metadata_frames = source.fps, source.metadata_frames
    try:
        frames = list(source.frames())
    finally:
        cleanup_source(source)
    if not frames:
        raise ValueError("video contains no decodable frames")
    duration_s = (len(frames) - 1) / fps

    px_per_mm, fiducial_size = _detect_fiducial(frames[0], config)
    geometry = _detect_barrel(frames[0], px_per_mm, config)
    result.diagnostics.update(
        {
            "fps": round(fps, 6),
            "decoded_frames": len(frames),
            "metadata_frames": metadata_frames,
            "duration_s": round(duration_s, 4),
            "deterministic_pipeline": True,
            "rotation_deg": source.rotation_deg,
            "rotation_applied": bool(source.rotation_applied and source.rotation_deg),
            "ascii_path_fallback": source.used_ascii_fallback,
            "fiducial_required": config.fiducial_required,
        }
    )
    if geometry is None:
        result.quality_checks["syringe_detected"] = Check(False, None, "required", "No supported barrel geometry found.")
        result.abstention_reasons.append("syringe_not_detected")
        _finalize(result)
        return result
    result.quality_checks["syringe_detected"] = Check(True, True, "required", "Paired barrel sides detected.")

    length_mm = geometry.barrel_length_mm
    scale_ok = px_per_mm is not None
    uncalibrated = not scale_ok and not config.fiducial_required
    result.quality_checks["scale_fiducial"] = Check(
        scale_ok or uncalibrated,
        round(px_per_mm, 4) if px_per_mm else None,
        "85.60 x 53.98 mm ID-1 rectangle required" if config.fiducial_required else "ID-1 rectangle optional (uncalibrated mode)",
        (
            "Printed fiducial detected in syringe plane."
            if scale_ok
            else (
                "No fiducial present; grading uncalibrated (pixel scale unverified)."
                if uncalibrated
                else "Absolute syringe size cannot be verified without the fiducial."
            )
        ),
    )
    correct_syringe = length_mm is not None and abs(length_mm - config.barrel_length_mm) <= config.barrel_tolerance_mm
    result.quality_checks["correct_syringe"] = Check(
        correct_syringe or uncalibrated,
        round(length_mm, 3) if length_mm is not None else None,
        f"{config.barrel_length_mm} +/- {config.barrel_tolerance_mm} mm",
        (
            "Measured calibrated barrel length."
            if length_mm is not None
            else ("Not measurable without scale; syringe not verified in uncalibrated mode." if uncalibrated else "Not measurable without scale.")
        ),
    )
    vertical_ok = geometry.vertical_deviation_deg <= config.max_vertical_deviation_deg
    result.quality_checks["verticality"] = Check(
        vertical_ok,
        round(geometry.vertical_deviation_deg, 3),
        f"<= {config.max_vertical_deviation_deg} degrees",
        "Deviation of barrel axis from image vertical.",
    )
    result.diagnostics["fiducial_size_px"] = [round(v, 2) for v in fiducial_size] if fiducial_size else None
    result.diagnostics["barrel_length_px"] = round(geometry.length_px, 3)

    normalized = [_normalise_barrel(frame, geometry, config) for frame in frames]
    measurements = np.array([_meniscus(view) for view in normalized], dtype=float)
    volumes, confidences, artifact_fractions = measurements.T
    occluded_fractions = np.array([_border_occlusion(view) for view in normalized])

    nozzle_reference = cv2.cvtColor(_nozzle_roi(frames[0], geometry), cv2.COLOR_BGR2GRAY)
    nozzle_changes: list[float] = []
    for frame in frames:
        roi = _nozzle_roi(frame, geometry)
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        if gray.shape != nozzle_reference.shape or gray.size == 0:
            nozzle_changes.append(0.0)
        else:
            nozzle_changes.append(float(np.mean(cv2.absdiff(gray, nozzle_reference))))
    nozzle_changes_array = np.asarray(nozzle_changes)
    nozzle_index = _sustained_first(nozzle_changes_array, threshold=12.0, count=2)

    baseline_count = max(2, min(len(frames) // 5, int(round(fps * 0.75))))
    baseline_volume = float(np.median(volumes[:baseline_count]))
    displacement = baseline_volume - volumes
    liquid_index = _sustained_first(displacement, threshold=0.12, count=2)
    if nozzle_index is not None and (liquid_index is None or nozzle_index <= liquid_index):
        release_index, release_method = nozzle_index, "nozzle_release"
    elif liquid_index is not None:
        release_index, release_method = max(0, liquid_index - 1), "liquid_start"
    else:
        release_index, release_method = None, None

    if release_index is None:
        result.quality_checks["release_detected"] = Check(False, None, "required", "No release cue or liquid start was detected.")
        result.abstention_reasons.append("release_not_detected")
        target_index = None
    else:
        release_time = release_index / fps
        target_index = int(round(release_index + config.measurement_delay_s * fps))
        result.release_time_s = round(release_time, 4)
        result.release_method = release_method
        result.quality_checks["release_detected"] = Check(True, result.release_time_s, "required", f"Detected from {release_method}.")

    enough_duration = target_index is not None and target_index < len(frames)
    result.quality_checks["ten_second_capture"] = Check(
        enough_duration,
        round(duration_s - (result.release_time_s or 0.0), 3),
        f">= {config.measurement_delay_s} s after release",
        "Video contains the requested measurement frame." if enough_duration else "Video ends before the 10 s reading.",
    )

    if enough_duration and target_index is not None:
        window = max(1, int(round(fps * 0.10)))
        lo, hi = max(0, target_index - window), min(len(frames), target_index + window + 1)
        residual = float(np.median(volumes[lo:hi]))
        confidence = float(np.median(confidences[lo:hi]))
        artifact_fraction = float(np.max(artifact_fractions[lo:hi]))
        occluded_fraction = float(np.max(occluded_fractions[lo:hi]))
        flowed = max(0.0, baseline_volume - residual)
        result.estimated_residual_ml = round(residual, 3)
        result.measurement_time_s = round(target_index / fps, 4)
        result.diagnostics.update(
            {
                "initial_volume_ml": round(baseline_volume, 3),
                "flowed_volume_ml": round(flowed, 3),
                "meniscus_confidence": round(confidence, 4),
            }
        )
        meniscus_ok = confidence >= config.min_meniscus_confidence
        result.quality_checks["meniscus"] = Check(
            meniscus_ok,
            round(confidence, 4),
            f">= {config.min_meniscus_confidence}",
            "Contrast-based confidence at the 10 s reading.",
        )
        occlusion_ok = occluded_fraction <= config.max_occluded_fraction
        result.quality_checks["occlusion"] = Check(
            occlusion_ok,
            round(occluded_fraction, 4),
            f"<= {config.max_occluded_fraction}",
            "Estimated missing fraction of both barrel edges.",
        )
        artifacts_ok = artifact_fraction <= config.max_artifact_fraction
        result.quality_checks["bubbles_or_lumps"] = Check(
            artifacts_ok,
            round(artifact_fraction, 5),
            f"<= {config.max_artifact_fraction}",
            "Heuristic contrasting-region area within residual liquid.",
        )
        boundary_distance = min(abs(residual - boundary) for boundary in (1.0, 4.0, 8.0))
        result.boundary_distance_ml = round(boundary_distance, 3)
        boundary_ok = boundary_distance > config.boundary_margin_ml
        result.quality_checks["boundary"] = Check(
            boundary_ok,
            result.boundary_distance_ml,
            f"> {config.boundary_margin_ml} mL from 1/4/8 mL",
            "Boundary readings require repeat physical testing.",
        )
        result.iddsi_level = _level_for_volume(residual, flowed, config)

    if uncalibrated:
        # Informational: the result is graded without verified scale. This code
        # is attached for audit but does not by itself force abstention.
        result.diagnostics["scale_mode"] = "uncalibrated"
    else:
        result.diagnostics["scale_mode"] = "calibrated" if scale_ok else "unavailable"

    for name, check in result.quality_checks.items():
        if not check.passed:
            reason = CHECK_FAILURE_CODES.get(name)
            if reason and reason not in result.abstention_reasons:
                result.abstention_reasons.append(reason)
    if uncalibrated and "uncalibrated_scale_mode" not in result.abstention_reasons:
        result.abstention_reasons.append("uncalibrated_scale_mode")
    _finalize(result)
    return result


def _finalize(result: GradeResult) -> None:
    """Set final status and attach documented abstention-code descriptions."""

    # "uncalibrated_scale_mode" is an audit flag, not a disqualifier.
    disqualifying = [code for code in result.abstention_reasons if code != "uncalibrated_scale_mode"]
    if not disqualifying and result.iddsi_level is not None:
        result.status = "graded"
    else:
        result.status = "abstain"
        result.iddsi_level = None
    result.diagnostics["abstention_code_descriptions"] = describe_all(result.abstention_reasons)
