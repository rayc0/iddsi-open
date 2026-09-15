from __future__ import annotations

from typing import Any, Iterable

from ..http import HTTPClient
from ..types import Candidate


API_URL = "https://api.openverse.org/v1/images"
QUERIES = (
    "congee",
    "juk",
    "okayu",
    "chao",
    "arroz caldo",
    "chawanmushi",
    "gyeran-jjim",
    "douhua",
    "tofu pudding",
    "sesame paste",
    "mashed yam",
    "pureed pumpkin",
    "mashed carrot",
)
ALLOWED_LICENSES = {"cc0", "by"}
PAGE_SIZE = 20


def _canonical_license(item: dict[str, Any]) -> tuple[str, str] | None:
    code = str(item.get("license") or "").strip().casefold()
    url = str(item.get("license_url") or "").strip()
    if code not in ALLOWED_LICENSES or not url:
        return None
    version = str(item.get("license_version") or "").strip()
    if code == "cc0":
        return "CC0 1.0", url
    if not version:
        marker = "/licenses/by/"
        lower_url = url.casefold()
        if marker in lower_url:
            version = lower_url.split(marker, 1)[1].strip("/").split("/", 1)[0]
    if not version:
        return None
    return f"CC BY {version}", url


class OpenverseSource:
    """Discover candidates through Openverse; verify rights on the foreign landing page."""

    name = "openverse"

    def __init__(self, http: HTTPClient, *, queries: tuple[str, ...] = QUERIES) -> None:
        self.http = http
        self.queries = queries

    @staticmethod
    def _candidate(item: dict[str, Any], query: str) -> Candidate | None:
        license_fields = _canonical_license(item)
        if license_fields is None:
            return None
        license_name, license_url = license_fields
        source_id = str(item.get("id") or "").strip()
        download_url = str(item.get("url") or item.get("thumbnail") or "").strip()
        landing_url = str(item.get("foreign_landing_url") or "").strip()
        creator = str(item.get("creator") or "").strip()
        title = str(item.get("title") or f"Openverse image {source_id}").strip()
        if not all((source_id, download_url, landing_url, creator, title)):
            return None
        attribution = (
            f"{title} — {creator}; {license_name}; licence verification required at "
            f"the source landing page: {landing_url}"
        )
        return Candidate(
            provider="openverse",
            source_id=source_id,
            download_url=download_url,
            landing_url=landing_url,
            license_name=license_name,
            license_url=license_url,
            creator=creator,
            title=title,
            attribution=attribution,
            verification_status="unverified",
            extra={
                "query": query,
                "foreign_landing_url": landing_url,
                "openverse_license": item.get("license"),
                "openverse_license_version": item.get("license_version"),
                "openverse_license_url": item.get("license_url"),
                "creator_url": item.get("creator_url"),
                "source": item.get("source"),
                "verification_status": "unverified",
                "license_verification_caveat": (
                    "Openverse is discovery-only; verify the per-image licence on the "
                    "foreign landing page before release."
                ),
                "license_verification_landing_url": landing_url,
            },
        )

    def discover(self, max_hint: int) -> Iterable[Candidate]:
        if max_hint < 1:
            return
        emitted = 0
        seen: set[str] = set()
        pages = {query: 1 for query in self.queries}
        active = list(self.queries)
        while active and emitted < max_hint:
            next_active: list[str] = []
            for query in active:
                page = pages[query]
                payload = self.http.get_json(
                    API_URL,
                    {
                        "q": query,
                        "license": "cc0,by",
                        "page": str(page),
                        "page_size": str(PAGE_SIZE),
                    },
                )
                results = payload.get("results", [])
                if not isinstance(results, list):
                    results = []
                new_ids = 0
                for item in results:
                    if not isinstance(item, dict):
                        continue
                    candidate = self._candidate(item, query)
                    if candidate is None or candidate.source_id in seen:
                        continue
                    seen.add(candidate.source_id)
                    new_ids += 1
                    emitted += 1
                    yield candidate
                    if emitted >= max_hint:
                        return

                page_count = payload.get("page_count")
                has_next = bool(payload.get("next"))
                if isinstance(page_count, int):
                    has_next = has_next or page < page_count
                elif len(results) == PAGE_SIZE:
                    has_next = True
                if has_next and new_ids:
                    pages[query] = page + 1
                    next_active.append(query)
            active = next_active
