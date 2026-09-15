from __future__ import annotations

import hashlib
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .types import Candidate, PrivacyResult, WeakLabel


EXTENSION_FIELDS = {
    "level_weak",
    "weak_confidence",
    "weak_label_record",
    "provenance",
    "privacy_screening",
    "phash",
}
_SAFE = re.compile(r"[^A-Za-z0-9._-]+")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _safe(value: str, *, fallback: str, limit: int = 63) -> str:
    cleaned = _SAFE.sub("-", value).strip("-._")
    return (cleaned or fallback)[:limit]


def event_id_for(candidate: Candidate) -> str:
    digest = hashlib.sha256(f"{candidate.provider}\0{candidate.source_id}".encode()).hexdigest()[:20]
    provider = _safe(candidate.provider, fallback="source", limit=24)
    return f"RW_{provider}_{digest}"


def build_event(
    *,
    candidate: Candidate,
    media_path: str,
    media_sha256: str,
    media_bytes: int,
    phash: str,
    weak: WeakLabel,
    privacy: PrivacyResult,
    harvested_at: str,
) -> dict[str, Any]:
    event_id = event_id_for(candidate)
    provider = _safe(candidate.provider, fallback="source", limit=32)
    source_key = hashlib.sha256(candidate.source_id.encode()).hexdigest()[:16]
    captured_at = candidate.captured_at or harvested_at
    captured_at_basis = "source_timestamp" if candidate.captured_at else "harvest_time_fallback"
    sample_kind = "liquid" if weak.level_relation == "at_most" else "food"
    return {
        "schema_version": "1.0.0",
        "event_id": event_id,
        "recipe_id": f"rwrecipe_{source_key}",
        "batch_id": f"rwbatch_{source_key}",
        "kitchen_id": f"RW_{provider}",
        "phone_id": "unknown_source_camera",
        "captured_at": captured_at,
        "sample_kind": sample_kind,
        "level_declared": weak.level_weak,
        "level_tested_RD": None,
        "level_tested_SLP": None,
        "level_adjudicated": None,
        "media_files": [
            {
                "path": media_path,
                "media_type": "image",
                "capture_role": "plate_photo",
                "sha256": media_sha256,
                "bytes": media_bytes,
            }
        ],
        "test_type": ["plate_photo"],
        "capture_conditions": {
            "operator_id": "autonomous_harvester",
            "lighting_category": "other",
            "lighting_notes": "unknown; third-party source image",
            "plate_category": "other",
            "plate_notes": "unknown; third-party source image",
            "serving_temperature_c": None,
            "cuisine_tags": ["unknown_source_cuisine"],
        },
        "ground_truth_record": {
            "rd_assessment_id": None,
            "slp_assessment_id": None,
            "adjudication_status": "not_applicable",
            "adjudicator_code": None,
            "adjudication_notes": "No physical test; visual weak label only.",
        },
        "notes": (
            "Autonomously harvested third-party image; weak visual candidate only, not "
            f"ground truth. captured_at_basis={captured_at_basis}."
        ),
        "split": "train",
        "source": "real-weak",
        "consent_recorded": False,
        "contains_face": False,
        "level_weak": weak.level_weak,
        "weak_confidence": weak.confidence,
        "weak_label_record": {
            "model_id": weak.model_id,
            "prompt_version": weak.prompt_version,
            "rule": weak.rule,
            "level_relation": weak.level_relation,
            "observations": weak.observations,
            "rationale": weak.rationale,
            "is_ground_truth": False,
        },
        "provenance": {
            "provider": candidate.provider,
            "source_id": candidate.source_id,
            "landing_url": candidate.landing_url,
            "download_url": candidate.download_url,
            "title": candidate.title,
            "creator": candidate.creator,
            "license_name": candidate.license_name,
            "license_url": candidate.license_url,
            "verification_status": candidate.verification_status,
            "attribution": candidate.attribution,
            "captured_at_basis": captured_at_basis,
            "harvested_at": harvested_at,
            "transformations": ["EXIF orientation applied", "converted to RGB JPEG", "metadata stripped"],
            "source_metadata": candidate.extra,
        },
        "privacy_screening": {
            "contains_face": False,
            "contains_text": False,
            "face_engine": privacy.face_engine,
            "text_engine": privacy.text_engine,
            "details": privacy.details,
            "vlm_face_check": weak.contains_face,
            "vlm_text_check": weak.contains_text,
        },
        "phash": phash,
    }


def attribution_row(event: dict[str, Any]) -> dict[str, Any]:
    provenance = event["provenance"]
    media = event["media_files"][0]
    return {
        "event_id": event["event_id"],
        "media_path": media["path"],
        "media_sha256": media["sha256"],
        "provider": provenance["provider"],
        "source_id": provenance["source_id"],
        "title": provenance["title"],
        "creator": provenance["creator"],
        "landing_url": provenance["landing_url"],
        "license_name": provenance["license_name"],
        "license_url": provenance["license_url"],
        "verification_status": provenance.get("verification_status", "verified"),
        "attribution": provenance["attribution"],
        "transformations": provenance["transformations"],
    }


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
        handle.write("\n")
        handle.flush()


def append_jsonl_pair(
    first_path: Path,
    first_row: dict[str, Any],
    second_path: Path,
    second_row: dict[str, Any],
) -> None:
    """Append manifest and attribution together, rolling both back on an I/O failure."""
    first_path.parent.mkdir(parents=True, exist_ok=True)
    second_path.parent.mkdir(parents=True, exist_ok=True)
    first_text = json.dumps(first_row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
    second_text = json.dumps(second_row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
    with first_path.open("a+", encoding="utf-8") as first, second_path.open(
        "a+", encoding="utf-8"
    ) as second:
        first.seek(0, os.SEEK_END)
        second.seek(0, os.SEEK_END)
        first_start = first.tell()
        second_start = second.tell()
        try:
            first.write(first_text)
            second.write(second_text)
            first.flush()
            second.flush()
            os.fsync(first.fileno())
            os.fsync(second.fileno())
        except Exception:
            first.seek(first_start)
            first.truncate()
            second.seek(second_start)
            second.truncate()
            raise


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{number} is not a JSON object")
            rows.append(value)
    return rows


def build_realweak_schema(base_schema_path: Path) -> dict[str, Any]:
    """Derive the local extension without editing the frozen parent contract."""
    schema = json.loads(base_schema_path.read_text(encoding="utf-8"))
    schema["$id"] = "https://github.com/LinguaLeap/iddsi-open/harvest/realweak-event.schema.json"
    schema["title"] = "IDDSI-Open real-weak harvest event extension"
    schema["description"] = (
        "Frozen event schema fields plus explicit real-weak provenance and model-label fields; "
        "these rows have no physical-test ground truth."
    )
    schema["properties"]["source"] = {"const": "real-weak"}
    schema["properties"].update(
        {
            "level_weak": {
                "anyOf": [
                    {"type": "integer", "minimum": 3, "maximum": 7},
                    {"type": "null"},
                ]
            },
            "weak_confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "weak_label_record": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "model_id",
                    "prompt_version",
                    "rule",
                    "level_relation",
                    "observations",
                    "rationale",
                    "is_ground_truth",
                ],
                "properties": {
                    "model_id": {"type": "string", "minLength": 1},
                    "prompt_version": {"type": "string", "minLength": 1},
                    "rule": {"type": "string", "minLength": 1},
                    "level_relation": {"enum": ["candidate", "at_most"]},
                    "observations": {"type": "object"},
                    "rationale": {"type": "string", "maxLength": 500},
                    "is_ground_truth": {"const": False},
                },
            },
            "provenance": {
                "type": "object",
                "required": [
                    "provider",
                    "source_id",
                    "landing_url",
                    "download_url",
                    "title",
                    "creator",
                    "license_name",
                    "license_url",
                    "attribution",
                    "captured_at_basis",
                    "harvested_at",
                    "transformations",
                    "source_metadata",
                ],
                "properties": {
                    "provider": {"type": "string", "minLength": 1},
                    "source_id": {"type": "string", "minLength": 1},
                    "landing_url": {"type": "string", "minLength": 1},
                    "download_url": {"type": "string", "minLength": 1},
                    "title": {"type": "string", "minLength": 1},
                    "creator": {"type": "string", "minLength": 1},
                    "license_name": {"type": "string", "minLength": 1},
                    "license_url": {"type": "string", "minLength": 1},
                    "verification_status": {
                        "enum": ["verified", "unverified"],
                    },
                    "attribution": {"type": "string", "minLength": 1},
                    "captured_at_basis": {"enum": ["source_timestamp", "harvest_time_fallback"]},
                    "harvested_at": {"type": "string", "format": "date-time"},
                    "transformations": {
                        "type": "array",
                        "minItems": 1,
                        "items": {"type": "string"},
                    },
                    "source_metadata": {"type": "object"},
                },
            },
            "privacy_screening": {
                "type": "object",
                "required": [
                    "contains_face",
                    "contains_text",
                    "face_engine",
                    "text_engine",
                    "details",
                    "vlm_face_check",
                    "vlm_text_check",
                ],
                "properties": {
                    "contains_face": {"const": False},
                    "contains_text": {"const": False},
                    "face_engine": {"type": "string"},
                    "text_engine": {"type": "string"},
                    "details": {"type": "object"},
                    "vlm_face_check": {"const": False},
                    "vlm_text_check": {"const": False},
                },
            },
            "phash": {"type": "string", "pattern": "^[a-f0-9]{16}$"},
        }
    )
    schema["required"] = [*schema["required"], *sorted(EXTENSION_FIELDS)]
    return schema


def validate_realweak_event(event: dict[str, Any], base_schema_path: Path) -> list[str]:
    """Validate structure with the frozen validator plus real-weak safety invariants."""
    schema = build_realweak_schema(base_schema_path)
    try:
        from dataschema.tools.validate_dataset import _validate_schema_value
    except ImportError as exc:  # pragma: no cover - repository layout invariant
        raise RuntimeError("cannot import frozen dataschema validator") from exc
    errors = [str(item) for item in _validate_schema_value(event, schema, schema, 1, "$")]
    if event.get("source") != "real-weak":
        errors.append("source must be real-weak")
    if event.get("split") != "train":
        errors.append("real-weak rows are development-only and must use split=train")
    if any(event.get(field) is not None for field in ("level_tested_RD", "level_tested_SLP", "level_adjudicated")):
        errors.append("real-weak rows cannot carry physical-test label fields")
    if event.get("level_declared") != event.get("level_weak"):
        errors.append("level_declared must mirror level_weak in the extension")
    if event.get("contains_face") is not False:
        errors.append("contains_face must be false")
    if event.get("weak_label_record", {}).get("is_ground_truth") is not False:
        errors.append("weak label must explicitly say is_ground_truth=false")
    provenance = event.get("provenance", {})
    for field in ("provider", "source_id", "landing_url", "creator", "license_name", "license_url", "attribution"):
        if not isinstance(provenance.get(field), str) or not provenance[field].strip():
            errors.append(f"provenance.{field} is required")
    return errors
