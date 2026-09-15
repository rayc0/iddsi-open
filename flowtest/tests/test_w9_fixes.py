"""Tests for W9 fixes: rotation handling, ASCII-safe paths, fiducial-optional
mode, and documented abstention codes."""

from __future__ import annotations

import shutil
import subprocess

import cv2
import numpy as np
import pytest

from flowtest import video_io
from flowtest.abstention import ABSTENTION_CODES, CHECK_FAILURE_CODES, describe, describe_all
from flowtest.pipeline import FlowTestConfig, grade_video
from flowtest.synthetic import SyntheticSpec, generate_video


# ---------------------------------------------------------------------------
# Abstention codes (fix 4)
# ---------------------------------------------------------------------------


def test_every_emitted_code_is_documented() -> None:
    """Every code the pipeline can emit must exist in the registry."""
    pipeline_codes = set(CHECK_FAILURE_CODES.values()) | {"syringe_not_detected", "release_not_detected", "uncalibrated_scale_mode"}
    assert pipeline_codes <= set(ABSTENTION_CODES)
    for code, (description, action) in ABSTENTION_CODES.items():
        assert description.strip(), code
        assert action.strip(), code


def test_describe_unknown_code_is_explicit() -> None:
    assert "Undocumented" in describe("not_a_real_code")
    assert describe("meniscus_unclear") == ABSTENTION_CODES["meniscus_unclear"][0]
    assert set(describe_all(["meniscus_unclear", "bogus"])) == {"meniscus_unclear", "bogus"}


def test_result_carries_code_descriptions(tmp_path) -> None:
    video = generate_video(tmp_path / "desc.mp4", SyntheticSpec(residual_ml=4.25, fps=8.0))
    result = grade_video(video)
    descriptions = result.diagnostics["abstention_code_descriptions"]
    assert set(descriptions) == set(result.abstention_reasons)
    assert "near_level_boundary" in descriptions


# ---------------------------------------------------------------------------
# Rotation / orientation handling (fix 1)
# ---------------------------------------------------------------------------


class _StubCapture:
    """Minimal VideoCapture stand-in for rotation-logic tests."""

    def __init__(self, rotation: float, auto_ok: bool, frames: list[np.ndarray]):
        self._rotation = rotation
        self._auto_ok = auto_ok
        self._frames = frames
        self.released = False

    def isOpened(self) -> bool:  # noqa: N802 - mirrors cv2 API
        return True

    def get(self, prop: int) -> float:
        if prop == cv2.CAP_PROP_FPS:
            return 8.0
        if prop == cv2.CAP_PROP_FRAME_COUNT:
            return float(len(self._frames))
        if prop == video_io._CAP_PROP_ORIENTATION_META:
            return self._rotation
        return 0.0

    def set(self, prop: int, value: float) -> bool:
        if prop == video_io._CAP_PROP_ORIENTATION_AUTO:
            return self._auto_ok
        return False

    def read(self):
        if self._frames:
            return True, self._frames.pop(0)
        return False, None

    def release(self) -> None:
        self.released = True


def _portrait_frame() -> np.ndarray:
    frame = np.full((40, 20, 3), 255, np.uint8)
    cv2.line(frame, (5, 2), (5, 38), (0, 0, 0), 1)
    return frame


def test_rotation_snapping_rules() -> None:
    for raw, expected in [(0.0, 0), (89.6, 90), (90.0, 90), (179.9, 180), (-90.0, 270), (271.0, 270), (float("nan"), 0)]:
        capture = _StubCapture(raw, auto_ok=False, frames=[])
        assert video_io._read_rotation_deg(capture) == expected


@pytest.mark.parametrize(
    ("probe_payload", "expected"),
    [
        ('{"streams":[{"tags":{"rotate":"90"}}]}', 90),
        ('{"streams":[{"side_data_list":[{"rotation":-90}]}]}', 90),
    ],
)
def test_ffprobe_rotation_fallback(monkeypatch, tmp_path, probe_payload: str, expected: int) -> None:
    """Legacy tags and modern display matrices use different sign conventions."""
    completed = subprocess.CompletedProcess(["ffprobe"], 0, stdout=probe_payload, stderr="")
    monkeypatch.setattr(video_io.shutil, "which", lambda command: "/usr/bin/ffprobe")
    monkeypatch.setattr(video_io.subprocess, "run", lambda *args, **kwargs: completed)
    assert video_io._probe_rotation_deg(tmp_path / "video.mp4") == expected


def test_manual_rotation_applied_when_auto_unsupported() -> None:
    # A stored 90-degree rotation on a portrait frame must yield a landscape frame.
    capture = _StubCapture(90.0, auto_ok=False, frames=[_portrait_frame()])
    source = video_io.VideoSource(
        capture=capture, fps=8.0, metadata_frames=1,
        rotation_deg=90, rotation_applied=True, opened_path="x", used_ascii_fallback=False,
    )
    (frame,) = list(source.frames())
    assert frame.shape == (20, 40, 3)
    assert capture.released


def test_no_double_rotation_when_auto_supported() -> None:
    capture = _StubCapture(90.0, auto_ok=True, frames=[_portrait_frame()])
    source = video_io.VideoSource(
        capture=capture, fps=8.0, metadata_frames=1,
        rotation_deg=90, rotation_applied=False, opened_path="x", used_ascii_fallback=False,
    )
    (frame,) = list(source.frames())
    assert frame.shape == (40, 20, 3)  # untouched; OpenCV handles it


def _ffmpeg_functional() -> bool:
    if shutil.which("ffmpeg") is None:
        return False
    try:
        probe = subprocess.run(
            ["ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i", "color=black:s=16x16:d=0.1", "-f", "null", "-"],
            capture_output=True, timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return probe.returncode == 0


@pytest.mark.skipif(not _ffmpeg_functional(), reason="functional ffmpeg not available on this host")
def test_rotated_video_grades_upright(tmp_path) -> None:
    """A 90-degree-rotated capture of a valid scene must still grade (metadata honoured)."""
    upright = generate_video(tmp_path / "upright.mp4", SyntheticSpec(residual_ml=6.0, fps=8.0))
    sideways = tmp_path / "sideways.mp4"
    rotated = tmp_path / "rotated.mp4"
    # Store the scene 90 degrees counter-clockwise so a clockwise display
    # rotation restores the original upright pixels during decode.
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(upright), "-vf", "transpose=2", str(sideways)],
        check=True, capture_output=True, timeout=120,
    )
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(sideways), "-c", "copy", "-metadata:s:v", "rotate=90", str(rotated)],
        check=True, capture_output=True, timeout=120,
    )
    # FFmpeg 9 accepts the legacy rotate-tag command but silently emits no
    # tag/matrix.  Its input-side display_rotation option writes real display
    # matrix side data; use it only when the legacy output is metadata-free.
    capture = cv2.VideoCapture(str(rotated))
    legacy_rotation = video_io._read_rotation_deg(capture)
    capture.release()
    if not legacy_rotation and not video_io._probe_rotation_deg(rotated):
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-display_rotation", "-90", "-i", str(sideways), "-c", "copy", str(rotated)],
            check=True, capture_output=True, timeout=120,
        )
    result = grade_video(rotated)
    assert result.diagnostics["rotation_deg"] == 90
    assert result.status == "graded", result.to_dict()
    assert result.iddsi_level == "L2"


# ---------------------------------------------------------------------------
# ASCII-safe paths (fix 2)
# ---------------------------------------------------------------------------


def test_non_ascii_filename_grades(tmp_path) -> None:
    """A video named 增稠剂.mp4 must grade without crashing."""
    video = generate_video(tmp_path / "增稠剂.mp4", SyntheticSpec(residual_ml=6.0, fps=8.0))
    assert not str(video).isascii()
    result = grade_video(video)
    assert result.status == "graded", result.to_dict()
    assert result.iddsi_level == "L2"
    assert video.exists()  # original untouched


def test_ascii_fallback_opens_temp_copy(monkeypatch, tmp_path) -> None:
    """When OpenCV rejects a non-ASCII path, a temp ASCII copy is used and cleaned up."""
    video = generate_video(tmp_path / "增稠剂.mp4", SyntheticSpec(residual_ml=6.0, fps=8.0))
    real_capture = cv2.VideoCapture
    created: list[str] = []

    def picky_capture(path: str):
        if not str(path).isascii():
            class _Dead:
                def isOpened(self): return False
                def release(self): pass
            return _Dead()
        created.append(path)
        return real_capture(path)

    monkeypatch.setattr(cv2, "VideoCapture", picky_capture)
    result = grade_video(video)
    assert result.status == "graded", result.to_dict()
    assert result.diagnostics["ascii_path_fallback"] is True
    assert created and str(created[0]).isascii()
    assert video.exists()


def test_ascii_path_failure_still_raises(monkeypatch, tmp_path) -> None:
    """An ASCII path that cannot be opened raises the plain error (no fallback attempted)."""
    missing = tmp_path / "missing.mp4"
    with pytest.raises(ValueError, match="could not open video"):
        grade_video(missing)


def test_cleanup_only_removes_temp_copy(tmp_path) -> None:
    original = tmp_path / "orig.mp4"
    original.write_bytes(b"not a real video")
    source = video_io.VideoSource(
        capture=None, fps=1.0, metadata_frames=0, rotation_deg=0,
        rotation_applied=False, opened_path=original, used_ascii_fallback=False,
    )
    video_io.cleanup_source(source)
    assert original.exists()


# ---------------------------------------------------------------------------
# Fiducial-optional mode (fix 3)
# ---------------------------------------------------------------------------


def _no_fiducial_spec(**kwargs) -> SyntheticSpec:
    return SyntheticSpec(residual_ml=6.0, fps=8.0, fiducial=False, **kwargs)


def test_missing_fiducial_abstains_by_default(tmp_path) -> None:
    video = generate_video(tmp_path / "nofid.mp4", _no_fiducial_spec())
    result = grade_video(video)
    assert result.quality_checks["scale_fiducial"].value is None  # no card in scene
    assert result.status == "abstain"
    assert "scale_not_verified" in result.abstention_reasons
    assert result.diagnostics["scale_mode"] == "unavailable"


def test_fiducial_optional_mode_grades_uncalibrated(tmp_path) -> None:
    video = generate_video(tmp_path / "nofid_opt.mp4", _no_fiducial_spec())
    config = FlowTestConfig(fiducial_required=False)
    result = grade_video(video, config=config)
    assert result.quality_checks["scale_fiducial"].value is None  # no card in scene
    # Uncalibrated mode must not abstain on scale/syringe-verification grounds.
    assert "scale_not_verified" not in result.abstention_reasons
    assert "wrong_or_unverified_syringe" not in result.abstention_reasons
    assert "uncalibrated_scale_mode" in result.abstention_reasons
    assert result.diagnostics["scale_mode"] == "uncalibrated"
    assert result.diagnostics["fiducial_required"] is False
    assert result.status == "graded", result.to_dict()
    assert result.iddsi_level is not None


def test_fiducial_optional_still_calibrated_when_card_present(tmp_path) -> None:
    video = generate_video(tmp_path / "withfid.mp4", SyntheticSpec(residual_ml=6.0, fps=8.0))
    result = grade_video(video, config=FlowTestConfig(fiducial_required=False))
    assert result.status == "graded", result.to_dict()
    assert result.diagnostics["scale_mode"] == "calibrated"
    assert "uncalibrated_scale_mode" not in result.abstention_reasons


def test_cli_fiducial_optional_flag(tmp_path, monkeypatch) -> None:
    from flowtest.grade import main
    import json

    video = generate_video(tmp_path / "cli_nofid.mp4", _no_fiducial_spec())
    output = tmp_path / "result.json"
    monkeypatch.setattr(
        "sys.argv",
        ["flowtest.grade", str(video), "--out", str(output), "--fiducial-optional"],
    )
    assert main() == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["diagnostics"]["fiducial_required"] is False
    assert payload["quality_checks"]["scale_fiducial"]["value"] is None
    assert "uncalibrated_scale_mode" in payload["abstention_reasons"]
