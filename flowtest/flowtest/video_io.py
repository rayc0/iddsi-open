"""Video decoding helpers: rotation/orientation handling and ASCII-safe paths.

Two real-world failure modes are handled here:

1. **Rotation metadata.** Phone videos commonly store a display rotation
   (90/180/270) as container metadata instead of rotating pixels. Some
   OpenCV/FFmpeg builds expose it (``CAP_PROP_ORIENTATION_META``) but do not
   apply it, so frames arrive sideways and the barrel detector — which only
   accepts near-vertical line pairs — fails. ``open_video`` reads the
   metadata rotation and physically rotates decoded frames so downstream
   code always sees upright frames. If the build supports
   ``CAP_PROP_ORIENTATION_AUTO`` we let OpenCV auto-rotate instead and do
   not double-rotate.

2. **Non-ASCII paths.** Some OpenCV/FFmpeg builds cannot open paths with
   non-ASCII characters (e.g. ``增稠剂.mp4``) even though the file exists.
   ``open_video`` first tries the path directly; on failure, and only when
   the path contains non-ASCII characters, it copies the file to a temporary
   ASCII-named file (the original is never modified or deleted) and opens
   the copy.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

# Present in OpenCV >= 4.6 (FFmpeg backend). Guarded for older builds.
_CAP_PROP_ORIENTATION_META = getattr(cv2, "CAP_PROP_ORIENTATION_META", 48)
_CAP_PROP_ORIENTATION_AUTO = getattr(cv2, "CAP_PROP_ORIENTATION_AUTO", 49)

_ROTATE_CODES = {
    90: getattr(cv2, "ROTATE_90_CLOCKWISE", 0),
    180: getattr(cv2, "ROTATE_180", 1),
    270: getattr(cv2, "ROTATE_90_COUNTERCLOCKWISE", 2),
}


def _snap_rotation_deg(raw: object) -> int:
    """Normalise a rotation value to a clockwise display rotation."""

    try:
        value = float(raw)
    except (TypeError, ValueError):
        return 0
    if not np.isfinite(value):
        return 0
    degrees = int(round(value)) % 360
    snapped = int(round(degrees / 90.0)) * 90 % 360
    return snapped if snapped in _ROTATE_CODES or snapped == 0 else 0


def _read_rotation_deg(capture: cv2.VideoCapture) -> int:
    """Return the metadata display rotation in {0, 90, 180, 270}."""

    try:
        raw = capture.get(_CAP_PROP_ORIENTATION_META)
    except cv2.error:
        return 0
    return _snap_rotation_deg(raw)


def _probe_rotation_deg(path: Path) -> int:
    """Read rotation with ffprobe when OpenCV does not expose it.

    Legacy MOV/MP4 files can carry a clockwise ``rotate`` stream tag.  Newer
    FFmpeg versions expose the equivalent display matrix as side data whose
    ``rotation`` value uses the opposite (counter-clockwise) sign convention.
    ffprobe is an optional fallback: absent/broken binaries simply yield zero.
    """

    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        return 0
    try:
        completed = subprocess.run(
            [
                ffprobe,
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream_tags=rotate:stream_side_data=rotation",
                "-of",
                "json",
                str(path),
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
        if completed.returncode != 0:
            return 0
        streams = json.loads(completed.stdout).get("streams", [])
        if not streams:
            return 0
        stream = streams[0]
        tagged = _snap_rotation_deg(stream.get("tags", {}).get("rotate"))
        if tagged:
            return tagged
        for side_data in stream.get("side_data_list", []):
            if "rotation" in side_data:
                return _snap_rotation_deg(-float(side_data["rotation"]))
    except (OSError, subprocess.SubprocessError, ValueError, TypeError, json.JSONDecodeError):
        return 0
    return 0


def _try_open_auto_rotate(capture: cv2.VideoCapture) -> bool:
    """Ask OpenCV to apply orientation itself; True if the build accepted it."""

    try:
        return bool(capture.set(_CAP_PROP_ORIENTATION_AUTO, 1))
    except cv2.error:
        return False


@dataclass
class VideoSource:
    """An opened video plus the decoding context downstream code needs."""

    capture: cv2.VideoCapture
    fps: float
    metadata_frames: int
    rotation_deg: int
    rotation_applied: bool
    opened_path: Path  # the path actually handed to OpenCV (temp copy if used)
    used_ascii_fallback: bool

    def frames(self) -> Iterator[np.ndarray]:
        rotate_code = _ROTATE_CODES.get(self.rotation_deg) if self.rotation_applied else None
        try:
            while True:
                ok, frame = self.capture.read()
                if not ok:
                    break
                if rotate_code is not None:
                    frame = cv2.rotate(frame, rotate_code)
                yield frame
        finally:
            self.capture.release()


def _open_capture(path: Path) -> tuple[cv2.VideoCapture, Path, bool]:
    """Open ``path``; fall back to a temp ASCII-named copy for non-ASCII paths."""

    capture = cv2.VideoCapture(str(path))
    if capture.isOpened():
        return capture, path, False
    capture.release()
    if str(path).isascii():
        raise ValueError(f"could not open video: {path}")
    # Non-ASCII path this OpenCV build cannot handle: copy to an ASCII temp
    # name. The source file is only read, never modified or deleted.
    suffix = path.suffix if path.suffix.isascii() else ".mp4"
    temp = tempfile.NamedTemporaryFile(prefix="flowtest_", suffix=suffix, delete=False)
    temp_path = Path(temp.name)
    temp.close()
    try:
        shutil.copyfile(path, temp_path)
        capture = cv2.VideoCapture(str(temp_path))
        if not capture.isOpened():
            capture.release()
            raise ValueError(f"could not open video: {path} (ASCII-path fallback also failed)")
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise
    return capture, temp_path, True


def open_video(path: str | Path) -> VideoSource:
    """Open a video, honouring rotation metadata and non-ASCII paths."""

    video_path = Path(path)
    capture, opened_path, used_fallback = _open_capture(video_path)
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    metadata_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if not np.isfinite(fps) or fps <= 0:
        capture.release()
        if used_fallback:
            opened_path.unlink(missing_ok=True)
        raise ValueError("video has missing or invalid FPS metadata")

    rotation_deg = _read_rotation_deg(capture)
    rotation_from_probe = False
    if not rotation_deg:
        rotation_deg = _probe_rotation_deg(opened_path)
        rotation_from_probe = bool(rotation_deg)
    rotation_applied = False
    if rotation_deg:
        if rotation_from_probe:
            # A backend which returned no ORIENTATION_META cannot be trusted
            # to apply the metadata merely because set(AUTO, 1) succeeds.
            # Disable any implicit transform and rotate deterministically.
            try:
                capture.set(_CAP_PROP_ORIENTATION_AUTO, 0)
            except cv2.error:
                pass
            rotation_applied = True
        else:
            # Prefer OpenCV's own auto-rotation; only rotate manually when the
            # build cannot, so frames are never double-rotated.
            rotation_applied = not _try_open_auto_rotate(capture)

    return VideoSource(
        capture=capture,
        fps=fps,
        metadata_frames=metadata_frames,
        rotation_deg=rotation_deg,
        rotation_applied=rotation_applied,
        opened_path=opened_path,
        used_ascii_fallback=used_fallback,
    )


def cleanup_source(source: VideoSource) -> None:
    """Remove the temporary ASCII copy if one was created. Never touches the original."""

    if source.used_ascii_fallback:
        source.opened_path.unlink(missing_ok=True)
