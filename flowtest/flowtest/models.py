"""Small serialisable data models used by the grading pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class Check:
    passed: bool
    value: Any
    limit: Any
    detail: str


@dataclass
class GradeResult:
    video: str
    status: str = "abstain"
    iddsi_level: str | None = None
    estimated_residual_ml: float | None = None
    boundary_distance_ml: float | None = None
    release_time_s: float | None = None
    measurement_time_s: float | None = None
    release_method: str | None = None
    quality_checks: dict[str, Check] = field(default_factory=dict)
    abstention_reasons: list[str] = field(default_factory=list)
    diagnostics: dict[str, Any] = field(default_factory=dict)
    notice: str = (
        "Research/culinary-QA measurement aid only. Not IDDSI-endorsed; does "
        "not assess swallowing safety or replace the official physical test."
    )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
