from __future__ import annotations

import re
from dataclasses import dataclass

from .types import Candidate


_CC_RE = re.compile(r"^CC\s+BY(?P<sa>-SA)?\s+(?P<version>2\.0|2\.5|3\.0|4\.0)$", re.I)


@dataclass(frozen=True)
class LicenseDecision:
    allowed: bool
    canonical_name: str | None
    reason: str


def evaluate_license(name: str, url: str, creator: str) -> LicenseDecision:
    clean_name = " ".join(name.replace("Creative Commons", "CC").split()).strip()
    clean_url = url.strip().lower().replace("http://", "https://")
    if not creator.strip():
        return LicenseDecision(False, None, "missing_creator_attribution")

    if clean_name.upper() in {"CC0", "CC0 1.0"}:
        if "/publicdomain/zero/1.0" not in clean_url:
            return LicenseDecision(False, None, "cc0_url_missing_or_mismatched")
        return LicenseDecision(True, "CC0 1.0", "explicit_cc0")

    match = _CC_RE.match(clean_name)
    if not match:
        return LicenseDecision(False, None, "license_not_allowlisted")
    version = match.group("version")
    family = "by-sa" if match.group("sa") else "by"
    if f"/licenses/{family}/{version}" not in clean_url:
        return LicenseDecision(False, None, "license_name_url_mismatch")
    canonical = f"CC BY{'-SA' if family == 'by-sa' else ''} {version}"
    return LicenseDecision(True, canonical, "explicit_allowlisted_cc_license")


def candidate_license_decision(candidate: Candidate) -> LicenseDecision:
    return evaluate_license(candidate.license_name, candidate.license_url, candidate.creator)
