from __future__ import annotations

import json
import math
import os
import shutil
import urllib.parse
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

from .image_ops import PHashIndex, decode_image, normalized_jpeg, perceptual_hash, sha256_hex
from .licenses import candidate_license_decision
from .manifest import (
    append_jsonl,
    append_jsonl_pair,
    attribution_row,
    build_event,
    build_realweak_schema,
    event_id_for,
    read_jsonl,
    utc_now,
    validate_realweak_event,
)
from .screen import PrivacyScreener
from .sources.base import CandidateSource
from .types import Candidate


@dataclass(frozen=True)
class HarvestSummary:
    accepted: int
    rejected: int
    existing: int
    candidates_examined: int
    target: int
    stopped_reason: str
    http_stats: dict[str, dict[str, int]] = field(default_factory=dict)


def _round_robin(
    sources: Sequence[tuple[str, Iterable[Candidate]]],
) -> Iterator[tuple[Candidate | None, tuple[str, Exception] | None]]:
    active = [(name, iter(item)) for name, item in sources]
    while active:
        next_active: list[tuple[str, Iterator[Candidate]]] = []
        for name, iterator in active:
            try:
                yield next(iterator), None
                next_active.append((name, iterator))
            except StopIteration:
                continue
            except Exception as exc:
                yield None, (name, exc)
        active = next_active


def _safe_download_url(candidate: Candidate) -> bool:
    parsed = urllib.parse.urlparse(candidate.download_url)
    if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password:
        return False
    host = parsed.hostname.lower()
    if candidate.provider == "wikimedia":
        return host == "upload.wikimedia.org"
    if candidate.provider == "open-images":
        return host == "storage.googleapis.com" or host.endswith(".staticflickr.com")
    if candidate.provider == "nutrition5k":
        return host == "storage.googleapis.com"
    if candidate.provider == "openverse":
        # Openverse is an index: its result URL intentionally remains on the
        # original provider's HTTPS host rather than an Openverse CDN.
        return True
    return host == "example.test"  # injectable test-only source


class Harvester:
    def __init__(
        self,
        *,
        http: Any,
        sources: Sequence[CandidateSource],
        labeler: Any,
        screener: PrivacyScreener | Any | None = None,
        phash_threshold: int = 8,
        max_download_bytes: int = 25_000_000,
        candidate_multiplier: int = 10,
        provider_cap: float = 0.4,
    ) -> None:
        self.http = http
        self.sources = sources
        self.labeler = labeler
        self.screener = screener or PrivacyScreener()
        self.phash_threshold = phash_threshold
        self.max_download_bytes = max_download_bytes
        self.candidate_multiplier = candidate_multiplier
        self.provider_cap = provider_cap

    @staticmethod
    def _reject(path: Path, candidate: Candidate, reason: str, **details: Any) -> None:
        append_jsonl(
            path,
            {
                "provider": candidate.provider,
                "source_id": candidate.source_id,
                "download_url": candidate.download_url,
                "landing_url": candidate.landing_url,
                "license_name": candidate.license_name,
                "license_url": candidate.license_url,
                "verification_status": candidate.verification_status,
                "reason": reason,
                "details": details,
                "rejected_at": utc_now(),
            },
        )

    def run(self, *, maximum: int, output: Path) -> HarvestSummary:
        if maximum < 1:
            raise ValueError("--max must be at least 1")
        if not 0.0 < self.provider_cap <= 1.0:
            raise ValueError("--provider-cap must be greater than 0 and at most 1")
        output.mkdir(parents=True, exist_ok=True)
        media_root = output / "media" / "realweak"
        media_root.mkdir(parents=True, exist_ok=True)
        events_path = output / "events.jsonl"
        attribution_path = output / "attribution.jsonl"
        rejected_path = output / "rejected.jsonl"

        package_root = Path(__file__).resolve().parent
        base_schema = package_root.parent / "dataschema" / "schema" / "event.schema.json"
        schema = build_realweak_schema(base_schema)
        (output / "realweak-event.schema.json").write_text(
            json.dumps(schema, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        shutil.copyfile(package_root / "source_licenses.json", output / "source_licenses.json")

        existing_rows = read_jsonl(events_path)
        existing_attribution = read_jsonl(attribution_path)
        event_ids = [row.get("event_id") for row in existing_rows]
        attribution_ids = [row.get("event_id") for row in existing_attribution]
        if len(event_ids) != len(set(event_ids)):
            raise ValueError("existing events.jsonl contains duplicate event_id values")
        if len(attribution_ids) != len(set(attribution_ids)):
            raise ValueError("existing attribution.jsonl contains duplicate event_id values")
        if set(event_ids) != set(attribution_ids):
            raise ValueError(
                "existing events.jsonl and attribution.jsonl do not contain the same event IDs"
            )
        for row in existing_rows:
            validation_errors = validate_realweak_event(row, base_schema)
            if validation_errors:
                raise ValueError(
                    "existing real-weak manifest row is invalid: " + "; ".join(validation_errors)
                )
            for media in row.get("media_files", []):
                relative = media.get("path") if isinstance(media, dict) else None
                if not isinstance(relative, str):
                    continue
                media_path = (output / relative).resolve()
                try:
                    media_path.relative_to(output.resolve())
                except ValueError as exc:
                    raise ValueError(f"existing media path escapes output root: {relative}") from exc
                if not media_path.is_file():
                    raise ValueError(f"existing manifest media is missing: {relative}")
        seen_source = {
            (row.get("provenance", {}).get("provider"), row.get("provenance", {}).get("source_id"))
            for row in existing_rows
        }
        accepted_by_provider = Counter(
            row.get("provenance", {}).get("provider")
            for row in existing_rows
            if isinstance(row.get("provenance"), dict)
            and isinstance(row.get("provenance", {}).get("provider"), str)
        )
        hashes = PHashIndex(self.phash_threshold)
        exact_hashes: dict[str, str] = {}
        for row in existing_rows:
            if isinstance(row.get("phash"), str):
                hashes.add(str(row.get("event_id")), row["phash"])
            for media in row.get("media_files", []):
                if isinstance(media, dict) and isinstance(media.get("sha256"), str):
                    exact_hashes[media["sha256"]] = str(row.get("event_id"))

        existing = len(existing_rows)
        if existing >= maximum:
            return HarvestSummary(
                0,
                0,
                existing,
                0,
                maximum,
                "target_already_present",
                self._http_stats(),
            )

        accepted = rejected = examined = 0
        candidate_cap = max(maximum * self.candidate_multiplier, maximum)
        provider_limit = max(1, math.floor(maximum * self.provider_cap))
        source_iterables = [
            (source.name, source.discover(candidate_cap)) for source in self.sources
        ]
        for candidate, source_error in _round_robin(source_iterables):
            if existing + accepted >= maximum or examined >= candidate_cap:
                break
            examined += 1
            if source_error is not None:
                rejected += 1
                source_name, exc = source_error
                append_jsonl(
                    rejected_path,
                    {
                        "provider": source_name,
                        "source_id": "__source_discovery__",
                        "landing_url": "",
                        "license_name": "",
                        "license_url": "",
                        "reason": "source_discovery_error",
                        "details": {
                            "error_type": type(exc).__name__,
                            "error": str(exc)[:500],
                        },
                        "rejected_at": utc_now(),
                    },
                )
                continue
            assert candidate is not None
            source_key = (candidate.provider, candidate.source_id)
            if source_key in seen_source:
                continue
            seen_source.add(source_key)
            if accepted_by_provider[candidate.provider] >= provider_limit:
                rejected += 1
                self._reject(
                    rejected_path,
                    candidate,
                    "provider_cap_reached",
                    provider_cap=self.provider_cap,
                    provider_limit=provider_limit,
                    provider_accepted=accepted_by_provider[candidate.provider],
                )
                continue
            decision = candidate_license_decision(candidate)
            if not decision.allowed:
                rejected += 1
                self._reject(rejected_path, candidate, decision.reason)
                continue
            if not _safe_download_url(candidate):
                rejected += 1
                self._reject(rejected_path, candidate, "unsafe_or_unapproved_download_host")
                continue

            destination: Path | None = None
            try:
                source_bytes = self.http.get_bytes(
                    candidate.download_url,
                    max_bytes=self.max_download_bytes,
                )
                image = decode_image(source_bytes)
                normalized = normalized_jpeg(image)
                digest = sha256_hex(normalized)
                if digest in exact_hashes:
                    rejected += 1
                    self._reject(
                        rejected_path,
                        candidate,
                        "exact_duplicate",
                        duplicate_of=exact_hashes[digest],
                    )
                    continue
                phash = perceptual_hash(image)
                duplicate = hashes.nearest_duplicate(phash)
                if duplicate is not None:
                    rejected += 1
                    self._reject(
                        rejected_path,
                        candidate,
                        "perceptual_duplicate",
                        duplicate_of=duplicate[0],
                        hamming_distance=duplicate[1],
                    )
                    continue

                privacy = self.screener.screen(image)
                if privacy.contains_face or privacy.contains_text:
                    rejected += 1
                    reason = "face_detected" if privacy.contains_face else "text_detected"
                    self._reject(rejected_path, candidate, reason, screening=privacy.details)
                    continue
                weak = self.labeler.label(image)
                if weak.contains_face or weak.contains_text:
                    rejected += 1
                    reason = "face_detected_by_vlm" if weak.contains_face else "text_detected_by_vlm"
                    self._reject(rejected_path, candidate, reason)
                    continue

                event_id = event_id_for(candidate)
                provider_dir = media_root / candidate.provider
                provider_dir.mkdir(parents=True, exist_ok=True)
                destination = provider_dir / f"{event_id}.jpg"
                temporary = destination.with_suffix(".jpg.part")
                temporary.write_bytes(normalized)
                os.replace(temporary, destination)
                relative = destination.relative_to(output).as_posix()
                harvested_at = utc_now()
                event = build_event(
                    candidate=candidate,
                    media_path=relative,
                    media_sha256=digest,
                    media_bytes=len(normalized),
                    phash=phash,
                    weak=weak,
                    privacy=privacy,
                    harvested_at=harvested_at,
                )
                validation_errors = validate_realweak_event(event, base_schema)
                if validation_errors:
                    destination.unlink(missing_ok=True)
                    raise ValueError("real-weak manifest validation failed: " + "; ".join(validation_errors))
                append_jsonl_pair(events_path, event, attribution_path, attribution_row(event))
                hashes.add(event_id, phash)
                exact_hashes[digest] = event_id
                accepted += 1
                accepted_by_provider[candidate.provider] += 1
            except Exception as exc:
                if destination is not None:
                    destination.unlink(missing_ok=True)
                    destination.with_suffix(".jpg.part").unlink(missing_ok=True)
                rejected += 1
                self._reject(
                    rejected_path,
                    candidate,
                    "processing_error",
                    error_type=type(exc).__name__,
                    error=str(exc)[:500],
                )

        reason = "target_reached" if existing + accepted >= maximum else "candidate_budget_exhausted"
        return HarvestSummary(
            accepted,
            rejected,
            existing,
            examined,
            maximum,
            reason,
            self._http_stats(),
        )

    def _http_stats(self) -> dict[str, dict[str, int]]:
        snapshot = getattr(self.http, "stats_snapshot", None)
        return snapshot() if callable(snapshot) else {}
