"""Guided-capture pre-flight checks for the IDDSI Flow-Test capture app.

Pure functions on single BGR frames (the numpy arrays cv2 yields): no I/O, no
network, no learned models, no state.  Each check returns a :class:`QCResult`
carrying an actionable bilingual (EN + 繁中) user message so the app can tell
the user what to fix BEFORE recording starts ("tilt the phone", "move closer",
...).

Detector reuse: verticality, syringe presence and fiducial presence reuse the
grader's own deterministic detectors (Hough line-pair barrel geometry and the
ID-1 rectangle finder in :mod:`flowtest.pipeline`), so the pre-check and the
grader agree on what a valid capture looks like.  Lighting is a plain
luminance/clipping measurement.  Occlusion measures how much of the two barrel
side edges is actually visible in the Canny edge map — the same idea as the
grader's normalised-view occlusion estimate, applied pre-capture.

What each check actually measures (honest heuristics):

- ``lighting``     — mean grey luminance + fraction of near-saturated pixels.
- ``fiducial``     — presence of an ID-1 aspect dark-bordered rectangle.
- ``syringe``      — paired near-vertical Hough segments with barrel aspect.
- ``verticality``  — deviation of the detected barrel axis from image vertical.
- ``occlusion``    — (a) detected barrel length vs the fiducial-calibrated
                     expected length (a blocker hiding part of the barrel
                     shortens the detected sides), (b) sampled Canny edge
                     support along both detected barrel sides, and (c) a large
                     near-uniform, low-saturation connected blob differing
                     from the background level (finger over the lens /
                     extreme close-range).  Known false-positive risks: (a)
                     barrel foreshortened by camera angle; (c) a large dark
                     uniform surface such as a tabletop entering the frame.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import cv2
import numpy as np

from .pipeline import (
    FlowTestConfig,
    Geometry,
    _detect_barrel,
    _detect_fiducial,
    _edge_support,
)

__all__ = [
    "QCResult",
    "QCFrameResult",
    "check_lighting",
    "check_fiducial",
    "check_syringe",
    "check_verticality",
    "check_occlusion",
    "qc_checks",
    "qc_frame",
]

# --- tunables (deterministic thresholds, no learned components) -------------
DARK_MEAN_ERROR: float = 55.0  # mean grey below this -> blocking "too dark"
DARK_MEAN_WARN: float = 75.0  # mean grey below this -> advisory only
CLIP_FRACTION_ERROR: float = 0.35  # near-saturated fraction above this -> blocking
CLIP_FRACTION_WARN: float = 0.20  # advisory glare band
MIN_SIDE_EDGE_SUPPORT: float = 0.70  # barrel side visibility for occlusion test
MIN_BARREL_FRAME_FRACTION: float = 0.20  # barrel length vs frame height ("move closer")
MIN_DETECTED_LENGTH_FRACTION: float = 0.80  # detected vs calibrated barrel length
OCCLUDER_MIN_AREA_FRACTION: float = 0.25  # uniform-blob occlusion area gate
OCCLUDER_BG_DIFF: float = 40.0  # blob intensity must differ from background
OCCLUDER_MAX_SATURATION: float = 60.0  # blobs: grey/near-grey only (skip liquid)
OCCLUDER_MAX_STD: float = 40.0  # blob internal uniformity


@dataclass(frozen=True)
class QCResult:
    """Outcome of one pre-capture check with bilingual user guidance.

    ``passed`` is True only when the check is clean AND assessable.  Severity:
    ``"ok"`` clean · ``"warning"`` assessable but advisory (capture still
    usable) · ``"error"`` blocking problem with a concrete fix ·
    ``"unknown"`` cannot assess (typically: syringe not found yet).
    """

    check_id: str
    passed: bool
    severity: str
    value: Any
    limit: str
    message_en: str
    message_zh: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class QCFrameResult:
    """Aggregated pre-capture verdict over all checks for one frame."""

    checks: tuple[QCResult, ...]

    @property
    def passed(self) -> bool:
        return all(check.passed for check in self.checks)

    @property
    def failures(self) -> tuple[QCResult, ...]:
        """Failing checks, errors first (most actionable ordering)."""

        order = {"error": 0, "unknown": 1, "warning": 2, "ok": 3}
        failed = [check for check in self.checks if not check.passed]
        return tuple(sorted(failed, key=lambda check: order.get(check.severity, 4)))

    @property
    def messages_en(self) -> list[str]:
        return [check.message_en for check in self.failures]

    @property
    def messages_zh(self) -> list[str]:
        return [check.message_zh for check in self.failures]

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "messages_en": self.messages_en,
            "messages_zh": self.messages_zh,
            "checks": [check.to_dict() for check in self.checks],
        }


def _gray(frame: np.ndarray) -> np.ndarray:
    if frame.ndim == 2:
        return frame
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------


def check_lighting(frame: np.ndarray, config: FlowTestConfig | None = None) -> QCResult:
    """Mean luminance + clipping fraction -> too dark / too bright guidance."""
    del config  # thresholds are module-level tunables; keeps signature uniform
    gray = _gray(frame)
    mean = float(np.mean(gray))
    clipped = float(np.mean(gray >= 250))
    limit = f"mean {DARK_MEAN_WARN}-{int(255 - 5)} or clipped < {CLIP_FRACTION_WARN:.2f}"
    if mean < DARK_MEAN_ERROR:
        return QCResult(
            check_id="lighting",
            passed=False,
            severity="error",
            value=round(mean, 1),
            limit=limit,
            message_en="Too dark - turn on more light or move to a brighter spot.",
            message_zh="光線太暗——請開燈或移到較光亮的位置。",
        )
    if clipped > CLIP_FRACTION_ERROR:
        return QCResult(
            check_id="lighting",
            passed=False,
            severity="error",
            value=round(clipped, 3),
            limit=limit,
            message_en="Too bright - avoid backlight and strong glare on the syringe.",
            message_zh="光線太亮——請避免背光及針筒上的強烈反光。",
        )
    if mean < DARK_MEAN_WARN:
        return QCResult(
            check_id="lighting",
            passed=True,
            severity="warning",
            value=round(mean, 1),
            limit=limit,
            message_en="A little dim - brighter, even lighting will improve the reading.",
            message_zh="光線略暗——較均勻的光線可提高讀數準確度。",
        )
    if clipped > CLIP_FRACTION_WARN:
        return QCResult(
            check_id="lighting",
            passed=True,
            severity="warning",
            value=round(clipped, 3),
            limit=limit,
            message_en="Some glare - tilting slightly away from the light helps.",
            message_zh="有反光——稍為避開光源會更清晰。",
        )
    return QCResult(
        check_id="lighting",
        passed=True,
        severity="ok",
        value=round(mean, 1),
        limit=limit,
        message_en="Lighting looks good.",
        message_zh="光線正常。",
    )


def check_fiducial(frame: np.ndarray, config: FlowTestConfig | None = None) -> QCResult:
    """Reuse the grader's ID-1 rectangle detector (85.60 x 53.98 mm card)."""
    config = config or FlowTestConfig()
    px_per_mm, _size = _detect_fiducial(frame, config)
    limit = "ID-1 card 85.60 x 53.98 mm in syringe plane"
    if px_per_mm is None:
        return QCResult(
            check_id="fiducial",
            passed=False,
            severity="error",
            value=None,
            limit=limit,
            message_en="Show the printed calibration card (85.6 x 54 mm) in the frame.",
            message_zh="請將校準卡（85.6 x 54 毫米）放入畫面。",
        )
    return QCResult(
        check_id="fiducial",
        passed=True,
        severity="ok",
        value=round(px_per_mm, 2),
        limit=limit,
        message_en="Calibration card detected.",
        message_zh="已偵測到校準卡。",
    )


def check_syringe(
    frame: np.ndarray,
    config: FlowTestConfig | None = None,
    geometry: Geometry | None = None,
    px_per_mm: float | None = None,
) -> QCResult:
    """Syringe presence via the grader's Hough line-pair detector + size advisory.

    ``geometry``/``px_per_mm`` may be supplied by :func:`qc_checks` to avoid
    re-running detection; standalone callers leave them as ``None``.
    """
    config = config or FlowTestConfig()
    if px_per_mm is None:
        px_per_mm, _size = _detect_fiducial(frame, config)
    if geometry is None:
        geometry = _detect_barrel(frame, px_per_mm, config)
    if geometry is None:
        if px_per_mm is not None:
            # The fiducial is visible but no barrel: most often the phone is
            # too far back or the syringe is out of frame.
            message_en = "Syringe not detected - move closer until the whole syringe is in frame."
            message_zh = "偵測不到針筒——請移近一些，令整支針筒完整入鏡。"
        else:
            message_en = "Syringe not detected - show the syringe upright in the frame."
            message_zh = "偵測不到針筒——請將針筒垂直放入畫面。"
        return QCResult(
            check_id="syringe",
            passed=False,
            severity="error",
            value=None,
            limit="paired barrel sides with barrel aspect",
            message_en=message_en,
            message_zh=message_zh,
        )
    frame_height = frame.shape[0]
    length_fraction = geometry.length_px / frame_height
    value = round(float(geometry.length_px), 1)
    if length_fraction < MIN_BARREL_FRAME_FRACTION:
        return QCResult(
            check_id="syringe",
            passed=True,
            severity="warning",
            value=value,
            limit=f">= {MIN_BARREL_FRAME_FRACTION:.0%} of frame height",
            message_en="Move closer - the syringe is too small in the frame.",
            message_zh="請移近一些——針筒在畫面中太小。",
        )
    return QCResult(
        check_id="syringe",
        passed=True,
        severity="ok",
        value=value,
        limit=f">= {MIN_BARREL_FRAME_FRACTION:.0%} of frame height",
        message_en="Syringe detected.",
        message_zh="已偵測到針筒。",
    )


def check_verticality(
    frame: np.ndarray,
    config: FlowTestConfig | None = None,
    geometry: Geometry | None = None,
) -> QCResult:
    """Barrel-axis tilt vs image vertical, using the grader's line-pair detector."""
    config = config or FlowTestConfig()
    if geometry is None:
        px_per_mm, _size = _detect_fiducial(frame, config)
        geometry = _detect_barrel(frame, px_per_mm, config)
    limit = f"<= {config.max_vertical_deviation_deg} degrees from vertical"
    if geometry is None:
        return QCResult(
            check_id="verticality",
            passed=False,
            severity="unknown",
            value=None,
            limit=limit,
            message_en="Cannot check tilt - hold the full syringe upright in frame.",
            message_zh="無法檢查傾斜——請垂直握住整支針筒入鏡。",
        )
    deviation = float(geometry.vertical_deviation_deg)
    if deviation > config.max_vertical_deviation_deg:
        return QCResult(
            check_id="verticality",
            passed=False,
            severity="error",
            value=round(deviation, 2),
            limit=limit,
            message_en=f"Tilt the phone upright - the syringe is {deviation:.1f} degrees off vertical.",
            message_zh=f"請將手機垂直握正——針筒偏斜約 {deviation:.1f} 度。",
        )
    return QCResult(
        check_id="verticality",
        passed=True,
        severity="ok",
        value=round(deviation, 2),
        limit=limit,
        message_en="Syringe is vertical.",
        message_zh="針筒垂直。",
    )


def _barrel_side_lines(geometry: Geometry) -> tuple[np.ndarray, np.ndarray]:
    """(left, right) side segments of the detected barrel as 4-float lines."""

    half_length = geometry.length_px / 2
    half_width = geometry.width_px / 2
    top = geometry.center - geometry.vertical * half_length
    bottom = geometry.center + geometry.vertical * half_length
    lines = []
    for sign in (-1.0, 1.0):
        p = top + geometry.horizontal * half_width * sign
        q = bottom + geometry.horizontal * half_width * sign
        lines.append(np.array([p[0], p[1], q[0], q[1]], dtype=float))
    return lines[0], lines[1]


def _occluder_blob_fraction(frame: np.ndarray) -> float:
    """Fraction of the frame covered by the largest near-uniform, low-saturation
    connected region that differs strongly from the background level.

    Targets a finger/hand over the lens or an extreme close-range obstruction;
    the low-saturation gate keeps the (orange) test liquid out of the mask.
    Uniformity is measured over the component's own pixels (not its bounding
    box) so a small dark object touching the blob (e.g. the fiducial border)
    does not break the measurement.
    """
    gray = _gray(frame)
    height, width = gray.shape
    border = np.concatenate([gray[0, :], gray[-1, :], gray[:, 0], gray[:, -1]])
    background = float(np.median(border))
    saturation = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)[:, :, 1]
    mask = (
        (np.abs(gray.astype(np.float32) - background) > OCCLUDER_BG_DIFF)
        & (saturation < OCCLUDER_MAX_SATURATION)
    ).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    labels, stats = cv2.connectedComponentsWithStats(mask, 8)[1:3]
    worst = 0.0
    for index in range(1, len(stats)):
        x, y, w, h, area = (int(v) for v in stats[index])
        if area < OCCLUDER_MIN_AREA_FRACTION * height * width:
            continue
        component = gray[y : y + h, x : x + w][labels[y : y + h, x : x + w] == index]
        if float(np.std(component)) > OCCLUDER_MAX_STD:
            continue  # textured regions are not uniform occluders
        worst = max(worst, area / float(height * width))
    return worst


def check_occlusion(
    frame: np.ndarray,
    config: FlowTestConfig | None = None,
    geometry: Geometry | None = None,
    px_per_mm: float | None = None,
) -> QCResult:
    """Occlusion via (a) calibrated barrel length vs detected length,
    (b) edge support along both barrel sides, and (c) a large uniform
    non-background blob (lens coverage / extreme close range)."""
    config = config or FlowTestConfig()
    if px_per_mm is None:
        px_per_mm, _size = _detect_fiducial(frame, config)
    if geometry is None:
        geometry = _detect_barrel(frame, px_per_mm, config)
    limit = (
        f"barrel length >= {MIN_DETECTED_LENGTH_FRACTION:.0%} of calibrated; "
        f"side edge support >= {MIN_SIDE_EDGE_SUPPORT:.2f}; "
        f"uniform blob < {OCCLUDER_MIN_AREA_FRACTION:.0%}"
    )
    blob_fraction = _occluder_blob_fraction(frame)
    if blob_fraction >= OCCLUDER_MIN_AREA_FRACTION:
        return QCResult(
            check_id="occlusion",
            passed=False,
            severity="error",
            value=round(float(blob_fraction), 3),
            limit=limit,
            message_en="Move your hand away from the camera - the view is blocked.",
            message_zh="鏡頭被遮擋——請移開手或後退一些。",
        )
    if geometry is None:
        return QCResult(
            check_id="occlusion",
            passed=False,
            severity="unknown",
            value=None,
            limit=limit,
            message_en="Cannot check for blocking - show the whole syringe.",
            message_zh="無法檢查遮擋——請令整支針筒入鏡。",
        )
    expected_px = config.barrel_length_mm * px_per_mm if px_per_mm else None
    if expected_px and geometry.length_px < MIN_DETECTED_LENGTH_FRACTION * expected_px:
        # A blocker hiding part of the barrel shortens the detected sides.
        ratio = geometry.length_px / expected_px
        return QCResult(
            check_id="occlusion",
            passed=False,
            severity="error",
            value=round(float(ratio), 3),
            limit=limit,
            message_en="Something is blocking the syringe - move your hand away.",
            message_zh="針筒被遮擋——請移開手或其他物件。",
        )
    edges = cv2.Canny(_gray(frame), 45, 140)
    left_line, right_line = _barrel_side_lines(geometry)
    support = min(
        _edge_support(edges, left_line),
        _edge_support(edges, right_line),
    )
    if support < MIN_SIDE_EDGE_SUPPORT:
        return QCResult(
            check_id="occlusion",
            passed=False,
            severity="error",
            value=round(float(support), 3),
            limit=limit,
            message_en="Something is blocking the syringe - move your hand away.",
            message_zh="針筒被遮擋——請移開手或其他物件。",
        )
    return QCResult(
        check_id="occlusion",
        passed=True,
        severity="ok",
        value=round(float(support), 3),
        limit=limit,
        message_en="Syringe is clearly visible.",
        message_zh="針筒清晰可見。",
    )


# ---------------------------------------------------------------------------
# Aggregates
# ---------------------------------------------------------------------------


def qc_checks(frame: np.ndarray, config: FlowTestConfig | None = None) -> list[QCResult]:
    """Run every pre-capture check on one BGR frame, sharing detections."""
    config = config or FlowTestConfig()
    px_per_mm, _size = _detect_fiducial(frame, config)
    geometry = _detect_barrel(frame, px_per_mm, config)
    return [
        check_lighting(frame),
        check_fiducial(frame, config),
        check_syringe(frame, config, geometry=geometry, px_per_mm=px_per_mm),
        check_verticality(frame, config, geometry=geometry),
        check_occlusion(frame, config, geometry=geometry, px_per_mm=px_per_mm),
    ]


def qc_frame(frame: np.ndarray, config: FlowTestConfig | None = None) -> QCFrameResult:
    """One-call aggregate: all checks + combined bilingual guidance."""
    return QCFrameResult(checks=tuple(qc_checks(frame, config)))
