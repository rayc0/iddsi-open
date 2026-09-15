from __future__ import annotations

import html
import re
import urllib.parse
from datetime import datetime
from typing import Any, Iterable, Iterator

from ..http import HTTPClient
from ..types import Candidate


API_URL = "https://commons.wikimedia.org/w/api.php"
SEARCH_TERMS = (
    "pureed food",
    "congee",
    "porridge",
    "thickened drinks",
    "soft diet",
    "minced meat",
    "tofu",
    "steamed egg",
    "baby food",
    "hospital meals",
    "dysphagia diet",
)
CATEGORIES = (
    "Congee",
    "Porridges",
    "Puréed food",
    "Puréed vegetables",
    "Baby food",
    "Tofu dishes",
    "Steamed eggs",
    "Mashed potato dishes",
)
_TAG_RE = re.compile(r"<[^>]+>")


def _metadata_value(metadata: dict[str, Any], key: str) -> str:
    raw = metadata.get(key, {})
    value = raw.get("value", "") if isinstance(raw, dict) else ""
    return html.unescape(_TAG_RE.sub(" ", str(value))).strip()


def _captured_datetime(metadata: dict[str, Any]) -> str | None:
    value = _metadata_value(metadata, "DateTimeOriginal")
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return value if "T" in value and parsed.utcoffset() is not None else None


class WikimediaCommonsSource:
    name = "wikimedia"

    def __init__(self, http: HTTPClient, *, search_terms: tuple[str, ...] = SEARCH_TERMS) -> None:
        self.http = http
        self.search_terms = search_terms
        self.stats: dict[str, Any] = {}
        self._failed_imageinfo_batches: list[tuple[str, list[str]]] = []
        self._queued_imageinfo_batches: set[tuple[str, ...]] = set()
        self._reset_stats()

    def _reset_stats(self) -> None:
        self.stats.clear()
        self.stats.update(
            {
                "category_pages_requested": 0,
                "category_pages_succeeded": 0,
                "imageinfo_batches_requested": 0,
                "imageinfo_batches_succeeded": 0,
                "imageinfo_batches_requeued": 0,
                "imageinfo_requeue_batches_succeeded": 0,
                "search_pages_requested": 0,
                "search_pages_succeeded": 0,
                "candidates_skipped_no_download_url": 0,
                "errors": [],
            }
        )
        self._failed_imageinfo_batches.clear()
        self._queued_imageinfo_batches.clear()

    def _record_error(self, stage: str, exc: Exception, **context: Any) -> None:
        self.stats["errors"].append(
            {
                "stage": stage,
                **context,
                "error_type": type(exc).__name__,
                "error": str(exc)[:500],
            }
        )

    @staticmethod
    def _base_params() -> dict[str, str]:
        return {
            "action": "query",
            "format": "json",
            "formatversion": "2",
            "prop": "imageinfo",
            "iiprop": "url|mime|size|sha1|extmetadata|timestamp",
            "iiurlwidth": "1600",
            "iiextmetadatalanguage": "en",
            "iiextmetadatafilter": (
                "LicenseShortName|LicenseUrl|Artist|Credit|ObjectName|"
                "ImageDescription|DateTimeOriginal|Categories"
            ),
        }

    def _search_queries(self) -> Iterator[dict[str, str]]:
        for term in self.search_terms:
            yield {
                "generator": "search",
                "gsrsearch": f'"{term}" filetype:bitmap',
                "gsrnamespace": "6",
                "gsrlimit": "50",
            }

    def _category_pages(
        self,
        category_title: str,
        *,
        include_subcategories: bool,
    ) -> Iterator[dict[str, Any]]:
        category_params = {
            "action": "query",
            "format": "json",
            "formatversion": "2",
            "list": "categorymembers",
            "cmtitle": category_title,
            "cmnamespace": "6|14" if include_subcategories else "6",
            "cmtype": "file|subcat" if include_subcategories else "file",
            "cmlimit": "max",
        }
        continuation: dict[str, str] = {}
        seen_continuations: set[str] = set()
        page_number = 1
        while True:
            self.stats["category_pages_requested"] += 1
            try:
                payload = self.http.get_json(
                    API_URL,
                    {**category_params, **continuation},
                )
            except Exception as exc:
                self._record_error(
                    "categorymembers",
                    exc,
                    category=category_title,
                    page=page_number,
                    cmcontinue=continuation.get("cmcontinue"),
                )
                return
            self.stats["category_pages_succeeded"] += 1
            members_value = payload.get("query", {}).get("categorymembers", [])
            if isinstance(members_value, list):
                members = members_value
            else:
                members = []
            for member in members:
                if isinstance(member, dict):
                    yield member
            next_values = payload.get("continue")
            if not isinstance(next_values, dict):
                return
            next_cmcontinue = next_values.get("cmcontinue")
            if not isinstance(next_cmcontinue, str) or not next_cmcontinue:
                return
            if next_cmcontinue in seen_continuations:
                self._record_error(
                    "categorymembers",
                    RuntimeError("repeated cmcontinue token"),
                    category=category_title,
                    page=page_number,
                    cmcontinue=next_cmcontinue,
                )
                return
            seen_continuations.add(next_cmcontinue)
            continuation = {"cmcontinue": next_cmcontinue}
            generic_continue = next_values.get("continue")
            if isinstance(generic_continue, str):
                continuation["continue"] = generic_continue
            page_number += 1

    def _resolved_category_pages(
        self,
        titles: list[str],
        query_name: str,
        *,
        requeue_on_failure: bool = True,
    ) -> Iterator[dict[str, Any]]:
        for offset in range(0, len(titles), 50):
            batch = titles[offset : offset + 50]
            params = {
                **self._base_params(),
                "iiprop": "url|extmetadata",
                "titles": "|".join(batch),
            }
            self.stats["imageinfo_batches_requested"] += 1
            try:
                payload = self.http.get_json(API_URL, params)
            except Exception as exc:
                self._record_error(
                    "imageinfo",
                    exc,
                    batch_offset=offset,
                    batch_size=len(batch),
                    first_title=batch[0],
                    requeue_attempt=not requeue_on_failure,
                )
                batch_key = tuple(batch)
                if requeue_on_failure and batch_key not in self._queued_imageinfo_batches:
                    self._queued_imageinfo_batches.add(batch_key)
                    self._failed_imageinfo_batches.append((query_name, batch))
                    self.stats["imageinfo_batches_requeued"] += 1
                continue
            self.stats["imageinfo_batches_succeeded"] += 1
            if not requeue_on_failure:
                self.stats["imageinfo_requeue_batches_succeeded"] += 1
            pages_value = payload.get("query", {}).get("pages", [])
            if isinstance(pages_value, dict):
                pages = list(pages_value.values())
            elif isinstance(pages_value, list):
                pages = pages_value
            else:
                pages = []
            for page in pages:
                if isinstance(page, dict):
                    yield page

    def _search_pages(self, generator_params: dict[str, str]) -> Iterator[dict[str, Any]]:
        continuation: dict[str, str] = {}
        page_number = 1
        while True:
            self.stats["search_pages_requested"] += 1
            try:
                payload = self.http.get_json(
                    API_URL,
                    {**self._base_params(), **generator_params, **continuation},
                )
            except Exception as exc:
                self._record_error(
                    "search",
                    exc,
                    query=generator_params.get("gsrsearch", ""),
                    page=page_number,
                )
                return
            self.stats["search_pages_succeeded"] += 1
            pages_value = payload.get("query", {}).get("pages", [])
            if isinstance(pages_value, dict):
                pages = list(pages_value.values())
            elif isinstance(pages_value, list):
                pages = pages_value
            else:
                pages = []
            for page in pages:
                if isinstance(page, dict):
                    yield page
            next_values = payload.get("continue")
            if not isinstance(next_values, dict):
                return
            continuation = {str(key): str(value) for key, value in next_values.items()}
            page_number += 1

    def _candidate(
        self,
        page: dict[str, Any],
        query_name: str,
        *,
        require_mime: bool = True,
    ) -> Candidate | None:
        infos = page.get("imageinfo")
        if not isinstance(infos, list) or not infos or not isinstance(infos[0], dict):
            self.stats["candidates_skipped_no_download_url"] += 1
            return None
        info = infos[0]
        # Prefer the ORIGINAL file URL (upload.wikimedia.org, allowlisted) over thumburl
        # (thumb.wikimedia.org is not an approved download host).
        raw_download_url = str(info.get("url") or info.get("thumburl") or "").strip()
        if not raw_download_url:
            self.stats["candidates_skipped_no_download_url"] += 1
            return None
        if require_mime and info.get("mime") not in {
            "image/jpeg",
            "image/png",
            "image/webp",
        }:
            return None
        metadata = info.get("extmetadata", {})
        if not isinstance(metadata, dict):
            return None
        title = _metadata_value(metadata, "ObjectName") or str(page.get("title", "Untitled"))
        creator = _metadata_value(metadata, "Artist") or _metadata_value(metadata, "Credit")
        license_name = _metadata_value(metadata, "LicenseShortName")
        license_url = _metadata_value(metadata, "LicenseUrl")
        landing_url = str(info.get("descriptionurl", ""))
        parsed_download_url = urllib.parse.urlparse(raw_download_url)
        download_url = urllib.parse.urlunparse(
            parsed_download_url._replace(query="", fragment="")
        )
        if not all((download_url, landing_url, license_name, license_url)):
            return None
        source_id = str(page.get("pageid") or page.get("title") or download_url)
        attribution = f"{title} — {creator}; {license_name}; {landing_url}"
        return Candidate(
            provider=self.name,
            source_id=source_id,
            download_url=download_url,
            landing_url=landing_url,
            license_name=license_name,
            license_url=license_url,
            creator=creator,
            title=title,
            attribution=attribution,
            captured_at=_captured_datetime(metadata),
            extra={
                "commons_file_title": page.get("title"),
                "query": query_name,
                "description": _metadata_value(metadata, "ImageDescription"),
                "categories": _metadata_value(metadata, "Categories"),
                "source_sha1": info.get("sha1"),
                "source_bytes": info.get("size"),
                "source_upload_timestamp": info.get("timestamp"),
                "date_time_original_raw": _metadata_value(metadata, "DateTimeOriginal"),
            },
        )

    def discover(self, max_hint: int) -> Iterable[Candidate]:
        self._reset_stats()
        emitted = 0
        seen: set[str] = set()
        for category in CATEGORIES:
            root_title = f"Category:{category}"
            subcategories: list[str] = []
            file_titles: list[str] = []
            for page in self._category_pages(root_title, include_subcategories=True):
                title = str(page.get("title") or "")
                if page.get("ns") == 14 or title.startswith("Category:"):
                    if title and title != root_title and title not in subcategories:
                        subcategories.append(title)
                    continue
                if title:
                    file_titles.append(title)
            for page in self._resolved_category_pages(file_titles, root_title):
                candidate = self._candidate(page, root_title, require_mime=False)
                if candidate is None:
                    continue
                if not candidate.download_url:
                    self.stats["candidates_skipped_no_download_url"] += 1
                    continue
                if candidate.source_id in seen:
                    continue
                seen.add(candidate.source_id)
                emitted += 1
                yield candidate
                if emitted >= max_hint:
                    return
            for subcategory_title in subcategories:
                file_titles = []
                for page in self._category_pages(
                    subcategory_title,
                    include_subcategories=False,
                ):
                    title = str(page.get("title") or "")
                    if title:
                        file_titles.append(title)
                for page in self._resolved_category_pages(file_titles, subcategory_title):
                    candidate = self._candidate(
                        page,
                        subcategory_title,
                        require_mime=False,
                    )
                    if candidate is None:
                        continue
                    if not candidate.download_url:
                        self.stats["candidates_skipped_no_download_url"] += 1
                        continue
                    if candidate.source_id in seen:
                        continue
                    seen.add(candidate.source_id)
                    emitted += 1
                    yield candidate
                    if emitted >= max_hint:
                        return

        for generator_params in self._search_queries():
            query_name = generator_params.get("gsrsearch", "")
            for page in self._search_pages(generator_params):
                candidate = self._candidate(page, query_name)
                if candidate is None:
                    continue
                if not candidate.download_url:
                    self.stats["candidates_skipped_no_download_url"] += 1
                    continue
                if candidate.source_id in seen:
                    continue
                seen.add(candidate.source_id)
                emitted += 1
                yield candidate
                if emitted >= max_hint:
                    return

        failed_batches = list(self._failed_imageinfo_batches)
        self._failed_imageinfo_batches.clear()
        for query_name, titles in failed_batches:
            for page in self._resolved_category_pages(
                titles,
                query_name,
                requeue_on_failure=False,
            ):
                candidate = self._candidate(page, query_name, require_mime=False)
                if candidate is None:
                    continue
                if not candidate.download_url:
                    self.stats["candidates_skipped_no_download_url"] += 1
                    continue
                if candidate.source_id in seen:
                    continue
                seen.add(candidate.source_id)
                emitted += 1
                yield candidate
                if emitted >= max_hint:
                    return
