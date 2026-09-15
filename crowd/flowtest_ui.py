"""Flow-Test video grading wiring for the CROWD-CAPTURE research preview.

This module connects the Gradio UI to the sibling ``flowtest/`` package's
deterministic grader (``flowtest.grade_video`` / ``FlowTestConfig`` /
``ABSTENTION_CODES``).  The flowtest package lives in the sibling
``flowtest/`` directory and is importable as the top-level ``flowtest``
package once that directory is on ``sys.path`` (the same pattern
``crowd.inference`` uses for ``trainer/``).

The grader is deterministic OpenCV — no ML, no GPU, no network.  This wrapper
adds nothing to its measurement; it only normalises the ``GradeResult`` into a
JSON-serialisable dict for the UI and surfaces abstention reasons verbatim.
No accuracy, safety, or IDDSI-compliance claim is made or implied by the
output — the result is a research/culinary-QA measurement aid only (see
``app.DISCLAIMER`` and the grader's own ``notice``).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

# The flowtest package lives in the sibling ``flowtest/`` directory.  The
# *inner* ``flowtest/flowtest/`` package is what exposes ``grade_video``; the
# outer directory shadows it when the repo root is on sys.path, so we insert
# the outer directory explicitly (idempotent).
_REPO_ROOT = Path(__file__).resolve().parent.parent
_FLOWTEST_DIR = _REPO_ROOT / "flowtest"


def _ensure_flowtest_importable() -> None:
    if str(_FLOWTEST_DIR) not in sys.path:
        sys.path.insert(0, str(_FLOWTEST_DIR))


def grade_flow_test(
    video: Any,
    *,
    fiducial_required: bool | None = None,
) -> dict[str, Any]:
    """Grade one syringe Flow-Test video and return a JSON-serialisable dict.

    ``video`` may be a filepath (str / Path) or a Gradio-style dict/object
    carrying a ``path``/``name`` — it is normalised the same way
    ``app._as_path`` normalises upload values.

    Returns the grader's own ``GradeResult.to_dict()`` payload, which always
    contains ``status`` (``"ok"`` / ``"abstain"`` / ``"error"``),
    ``abstention_reasons`` (machine-readable codes), and
    ``diagnostics.abstention_code_descriptions`` (human explanations).  On an
    unreadable/undecodable input the grader raises ``OSError``/``ValueError``;
    this wrapper converts that into ``status: "error"`` so the UI can render
    it without a traceback.

    ``fiducial_required`` defaults to the ``CROWD_FLOWTEST_FIDUCIAL_REQUIRED``
    env var (default ``true``); pass ``False`` for the uncalibrated
    fiducial-optional mode (flagged ``uncalibrated_scale_mode`` by the grader).
    """

    _ensure_flowtest_importable()
    # Imported lazily so the module stays importable in minimal environments
    # where cv2 (a flowtest dependency) is absent.
    from flowtest import ABSTENTION_CODES  # noqa: PLC0415
    from flowtest.grade import FlowTestConfig, grade_video  # noqa: PLC0415

    path = _normalise_video_path(video)
    if path is None:
        return {
            "video": None,
            "status": "error",
            "error": "No video was provided.",
            "abstention_reasons": [],
            "diagnostics": {"abstention_code_descriptions": {}},
        }

    if fiducial_required is None:
        fiducial_required = os.getenv("CROWD_FLOWTEST_FIDUCIAL_REQUIRED", "true").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
    config = FlowTestConfig(fiducial_required=fiducial_required)

    try:
        result = grade_video(path, config=config)
    except (OSError, ValueError) as exc:
        return {
            "video": str(path),
            "status": "error",
            "error": str(exc),
            "abstention_reasons": [],
            "diagnostics": {"abstention_code_descriptions": {}},
        }

    payload = result.to_dict()
    # Guarantee the human-readable descriptions are present even if a grader
    # version omits them, so the UI never has to guess at a code's meaning.
    diagnostics = payload.setdefault("diagnostics", {})
    descriptions = diagnostics.get("abstention_code_descriptions")
    if not descriptions:
        diagnostics["abstention_code_descriptions"] = {
            code: ABSTENTION_CODES.get(code, (f"Undocumented abstention code: {code}", ""))[0]
            for code in payload.get("abstention_reasons", [])
        }
    return payload


def _normalise_video_path(video: Any) -> Path | None:
    """Normalise a Gradio upload value to a Path without trusting filenames."""

    if video is None or video == "":
        return None
    if isinstance(video, (str, os.PathLike)):
        return Path(video)
    if isinstance(video, dict):
        candidate = video.get("path") or video.get("name")
        return Path(candidate) if candidate else None
    candidate = getattr(video, "path", None) or getattr(video, "name", None)
    return Path(candidate) if candidate else None
