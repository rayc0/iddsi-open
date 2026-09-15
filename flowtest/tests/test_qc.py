"""Tests for flowtest.qc: guided-capture pre-flight checks on single frames.

Frames are built in memory from the same deterministic synthetic primitives the
pipeline tests use (``flowtest.synthetic`` drawing helpers), so these tests run
CPU-fast with no I/O.  They verify each check's pass AND fail direction plus
the presence of actionable bilingual (EN + 繁中) user messages.
"""

from __future__ import annotations

import json

import cv2
import numpy as np

from flowtest.qc import (
    QCFrameResult,
    QCResult,
    check_fiducial,
    check_lighting,
    check_occlusion,
    check_syringe,
    check_verticality,
    qc_checks,
    qc_frame,
)
from flowtest.synthetic import SyntheticSpec, _draw_fiducial, _syringe_layer

WIDTH, HEIGHT = 640, 480


def make_frame(**spec_kwargs: float | bool | int) -> np.ndarray:
    """One deterministic synthetic capture frame, no I/O (mirrors generate_video)."""

    spec = SyntheticSpec(residual_ml=6.0, **spec_kwargs)  # type: ignore[arg-type]
    frame = np.full((spec.height, spec.width, 3), 247, dtype=np.uint8)
    if spec.fiducial:
        _draw_fiducial(frame, spec.pixels_per_mm)
    layer = _syringe_layer(spec, volume_ml=6.0, before_release=True)
    mask = np.any(layer < 250, axis=2)
    frame[mask] = layer[mask]
    return frame


def scaled_down_frame() -> np.ndarray:
    """Good frame at half size: syringe detectable but small in frame."""
    small = np.full((HEIGHT, WIDTH, 3), 247, dtype=np.uint8)
    resized = cv2.resize(make_frame(), (WIDTH // 2, HEIGHT // 2), interpolation=cv2.INTER_AREA)
    small[HEIGHT // 4 : HEIGHT * 3 // 4, WIDTH // 4 : WIDTH * 3 // 4] = resized
    return small


def partially_blocked_frame() -> np.ndarray:
    """Gray blocker hiding the top third of the barrel (calibrated-length path)."""
    frame = make_frame()
    cv2.rectangle(frame, (355, 74), (455, 115), (118, 118, 118), -1)
    return frame


def lens_covered_frame() -> np.ndarray:
    """Large uniform gray blob over the syringe (lens-coverage path)."""
    frame = make_frame()
    cv2.rectangle(frame, (250, 40), (560, 400), (118, 118, 118), -1)
    return frame


def fiducial_only_frame() -> np.ndarray:
    frame = np.full((HEIGHT, WIDTH, 3), 247, dtype=np.uint8)
    _draw_fiducial(frame, 2.5)
    return frame


def by_id(results: list[QCResult] | tuple[QCResult, ...], check_id: str) -> QCResult:
    matches = [result for result in results if result.check_id == check_id]
    assert len(matches) == 1, f"expected exactly one {check_id} check"
    return matches[0]


def assert_bilingual(result: QCResult) -> None:
    assert isinstance(result.message_en, str) and result.message_en.strip()
    assert isinstance(result.message_zh, str) and result.message_zh.strip()
    assert any("一" <= character <= "鿿" for character in result.message_zh)


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_good_frame_all_checks_pass() -> None:
    result = qc_frame(make_frame())
    assert result.passed
    assert [check.check_id for check in result.checks] == [
        "lighting",
        "fiducial",
        "syringe",
        "verticality",
        "occlusion",
    ]
    for check in result.checks:
        assert check.passed, check.check_id
        assert check.severity == "ok"
        assert_bilingual(check)


def test_all_checks_carry_bilingual_messages_in_every_state() -> None:
    for frame in (
        make_frame(),
        make_frame(tilt_deg=10.0),
        make_frame(fiducial=False),
        make_frame() // 5,
        np.full((HEIGHT, WIDTH, 3), 255, dtype=np.uint8),
        scaled_down_frame(),
        partially_blocked_frame(),
        lens_covered_frame(),
        fiducial_only_frame(),
    ):
        for check in qc_checks(frame):
            assert check.severity in {"ok", "warning", "error", "unknown"}
            assert check.passed == (check.severity in {"ok", "warning"})
            assert_bilingual(check)


def test_checks_are_deterministic() -> None:
    frame = make_frame()
    first = qc_frame(frame)
    second = qc_frame(frame)
    assert first.checks == second.checks
    assert first.passed == second.passed


def test_each_check_function_is_callable_standalone() -> None:
    frame = make_frame()
    assert check_lighting(frame).passed
    assert check_fiducial(frame).passed
    assert check_syringe(frame).passed
    assert check_verticality(frame).passed
    assert check_occlusion(frame).passed


# ---------------------------------------------------------------------------
# Lighting
# ---------------------------------------------------------------------------


def test_lighting_too_dark() -> None:
    result = check_lighting(make_frame() // 5)
    assert not result.passed
    assert result.severity == "error"
    assert "dark" in result.message_en.lower()
    assert "光線太暗" in result.message_zh


def test_lighting_too_bright() -> None:
    result = check_lighting(np.full((HEIGHT, WIDTH, 3), 255, dtype=np.uint8))
    assert not result.passed
    assert result.severity == "error"
    assert "bright" in result.message_en.lower()
    assert "光線太亮" in result.message_zh


def test_lighting_dim_warning_is_advisory() -> None:
    result = check_lighting(np.full((HEIGHT, WIDTH, 3), 65, dtype=np.uint8))
    assert result.passed
    assert result.severity == "warning"
    assert "dim" in result.message_en.lower()


def test_lighting_glare_warning_is_advisory() -> None:
    frame = np.full((HEIGHT, WIDTH, 3), 247, dtype=np.uint8)
    frame[: int(HEIGHT * 0.30)] = 255  # 30% blown-out band: warning, not error
    result = check_lighting(frame)
    assert result.passed
    assert result.severity == "warning"
    assert "glare" in result.message_en.lower()


# ---------------------------------------------------------------------------
# Fiducial
# ---------------------------------------------------------------------------


def test_fiducial_missing() -> None:
    result = check_fiducial(make_frame(fiducial=False))
    assert not result.passed
    assert result.severity == "error"
    assert result.value is None
    assert "calibration card" in result.message_en.lower()
    assert "校準卡" in result.message_zh


def test_fiducial_detected_reports_scale() -> None:
    result = check_fiducial(make_frame())
    assert result.passed
    assert isinstance(result.value, float)
    assert 1.5 < result.value < 3.5  # synthetic card drawn at 2.5 px/mm


# ---------------------------------------------------------------------------
# Syringe presence
# ---------------------------------------------------------------------------


def test_syringe_detected() -> None:
    result = check_syringe(make_frame())
    assert result.passed
    assert result.severity == "ok"
    assert isinstance(result.value, float) and result.value > 0


def test_syringe_missing_but_fiducial_visible_says_move_closer() -> None:
    result = check_syringe(fiducial_only_frame())
    assert not result.passed
    assert result.severity == "error"
    assert "not detected" in result.message_en.lower()
    assert "移近" in result.message_zh


def test_syringe_small_in_frame_warns_move_closer() -> None:
    result = check_syringe(scaled_down_frame())
    assert result.passed  # advisory: capture usable but framing is poor
    assert result.severity == "warning"
    assert result.message_en.startswith("Move closer")
    assert "請移近一些" in result.message_zh


# ---------------------------------------------------------------------------
# Verticality
# ---------------------------------------------------------------------------


def test_verticality_pass_on_upright_frame() -> None:
    result = check_verticality(make_frame())
    assert result.passed
    assert result.value is not None and result.value <= 5.0


def test_verticality_fail_on_tilted_syringe() -> None:
    result = check_verticality(make_frame(tilt_deg=10.0))
    assert not result.passed
    assert result.severity == "error"
    assert result.value is not None and result.value > 5.0
    assert result.message_en.startswith("Tilt the phone upright")
    assert "請將手機垂直握正" in result.message_zh


def test_verticality_unknown_without_syringe() -> None:
    result = check_verticality(fiducial_only_frame())
    assert not result.passed
    assert result.severity == "unknown"


# ---------------------------------------------------------------------------
# Occlusion
# ---------------------------------------------------------------------------


def test_occlusion_ok_on_clear_frame() -> None:
    result = check_occlusion(make_frame())
    assert result.passed
    assert result.severity == "ok"
    assert isinstance(result.value, float) and result.value >= 0.70


def test_occlusion_partial_blocker_over_barrel() -> None:
    result = check_occlusion(partially_blocked_frame())
    assert not result.passed
    assert result.severity == "error"
    assert "blocking the syringe" in result.message_en.lower()
    assert "針筒被遮擋" in result.message_zh


def test_occlusion_lens_covered_by_large_blob() -> None:
    result = check_occlusion(lens_covered_frame())
    assert not result.passed
    assert result.severity == "error"
    assert "move your hand away" in result.message_en.lower()
    assert "鏡頭被遮擋" in result.message_zh


def test_occlusion_unknown_without_syringe() -> None:
    result = check_occlusion(fiducial_only_frame())
    assert not result.passed
    assert result.severity == "unknown"


# ---------------------------------------------------------------------------
# Aggregates
# ---------------------------------------------------------------------------


def test_qc_frame_aggregates_failures_with_messages() -> None:
    result = qc_frame(make_frame(tilt_deg=10.0))
    assert isinstance(result, QCFrameResult)
    assert not result.passed
    tilt = by_id(result.checks, "verticality")
    assert tilt.message_en in result.messages_en
    assert tilt.message_zh in result.messages_zh
    # The only failing check is the tilt; nothing else should be surfaced.
    assert result.messages_en == [tilt.message_en]


def test_qc_frame_failures_order_errors_first() -> None:
    result = qc_frame(fiducial_only_frame())
    severities = [check.severity for check in result.failures]
    assert severities == sorted(severities, key={"error": 0, "unknown": 1, "warning": 2, "ok": 3}.get)
    assert "error" in severities and "unknown" in severities


def test_to_dict_is_json_serialisable() -> None:
    payload = qc_frame(partially_blocked_frame()).to_dict()
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    restored = json.loads(text)
    assert restored["passed"] is False
    assert {check["check_id"] for check in restored["checks"]} == {
        "lighting",
        "fiducial",
        "syringe",
        "verticality",
        "occlusion",
    }
    assert any("針筒被遮擋" in message for message in restored["messages_zh"])
