from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Candidate:
    provider: str
    source_id: str
    download_url: str
    landing_url: str
    license_name: str
    license_url: str
    creator: str
    title: str
    attribution: str
    captured_at: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)
    verification_status: str = "verified"


@dataclass(frozen=True)
class PrivacyResult:
    contains_face: bool
    contains_text: bool
    face_engine: str
    text_engine: str
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class WeakLabel:
    level_weak: int
    confidence: float
    rule: str
    level_relation: str
    observations: dict[str, Any]
    model_id: str
    prompt_version: str
    contains_face: bool = False
    contains_text: bool = False
    rationale: str = ""
