"""Named, documented abstention codes for the Flow-Test grader.

Every ``abstain`` result carries one or more of these codes in
``GradeResult.abstention_reasons``. The registry below is the single source
of truth: codes are stable identifiers (safe to match programmatically) and
each maps to a human-readable description plus the corrective capture action.

``status: "abstain"`` always means "unclear — repeat the official physical
test"; it is never a pass and never a level.
"""

from __future__ import annotations

# code -> (short description, corrective action for the operator)
ABSTENTION_CODES: dict[str, tuple[str, str]] = {
    "syringe_not_detected": (
        "No supported syringe barrel geometry was found in the first frame.",
        "Re-capture with the full calibrated barrel and nozzle in frame against a plain contrasting background.",
    ),
    "scale_not_verified": (
        "No ID-1 fiducial (85.60 x 53.98 mm) was detected, so absolute scale is unverified.",
        "Place the printed ID-1 card in the syringe plane, or run with fiducial-optional mode for an uncalibrated reading.",
    ),
    "wrong_or_unverified_syringe": (
        "The calibrated barrel length does not match 61.5 +/- 3.0 mm, or scale is unavailable to verify it.",
        "Use the official 10 mL syringe with a 61.5 mm calibrated barrel and the fiducial card in frame.",
    ),
    "syringe_not_vertical": (
        "The barrel axis deviates from image vertical by more than the configured limit (default 5 degrees).",
        "Mount the syringe vertically and correct any camera roll before filming.",
    ),
    "release_not_detected": (
        "Neither a nozzle-release cue nor the start of liquid movement was detected.",
        "Start filming before the outlet is released and keep the nozzle region visible and unobstructed.",
    ),
    "insufficient_duration": (
        "The video ends before the 10-second post-release measurement frame.",
        "Continue filming for more than 10 seconds after release.",
    ),
    "meniscus_unclear": (
        "Meniscus contrast at the 10-second reading is below the confidence limit.",
        "Use diffuse lighting, a plain background, and a liquid with visible contrast; avoid glare on the barrel.",
    ),
    "syringe_occluded": (
        "One or both barrel edges are estimated to be obscured beyond the allowed fraction.",
        "Keep hands and other objects away from the barrel during the measurement window.",
    ),
    "bubbles_or_lumps_detected": (
        "Contrasting regions inside the residual liquid exceed the allowed fraction (bubble/lump heuristic).",
        "Remix or rest the sample to remove bubbles and lumps, then repeat the physical test.",
    ),
    "near_level_boundary": (
        "The reading is within the boundary margin (default 0.5 mL) of the 1/4/8 mL level boundaries.",
        "Repeat the official physical test; boundary readings cannot be assigned a level by this tool.",
    ),
    "uncalibrated_scale_mode": (
        "Graded without a fiducial: pixel scale is unverified and the syringe size could not be checked.",
        "For a calibrated result, re-capture with the ID-1 fiducial card in the syringe plane.",
    ),
}

#: Mapping from quality-check name to the abstention code emitted when it fails.
CHECK_FAILURE_CODES: dict[str, str] = {
    "scale_fiducial": "scale_not_verified",
    "correct_syringe": "wrong_or_unverified_syringe",
    "verticality": "syringe_not_vertical",
    "ten_second_capture": "insufficient_duration",
    "meniscus": "meniscus_unclear",
    "occlusion": "syringe_occluded",
    "bubbles_or_lumps": "bubbles_or_lumps_detected",
    "boundary": "near_level_boundary",
}


def describe(code: str) -> str:
    """Return the human description for an abstention code."""

    entry = ABSTENTION_CODES.get(code)
    return entry[0] if entry else f"Undocumented abstention code: {code}"


def describe_all(codes: list[str]) -> dict[str, str]:
    """Return ``{code: description}`` for a list of codes (unknown codes included)."""

    return {code: describe(code) for code in codes}
