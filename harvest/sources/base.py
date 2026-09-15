from __future__ import annotations

from typing import Iterable, Protocol

from ..types import Candidate


class CandidateSource(Protocol):
    name: str

    def discover(self, max_hint: int) -> Iterable[Candidate]: ...
