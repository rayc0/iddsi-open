#!/usr/bin/env python3
"""Validate an IDDSI-Open dataset folder, JSONL manifest, or Parquet dataset.

The JSONL path has no third-party dependencies. Parquet validation requires pyarrow.
Validation is strict by default: real events need independent RD and SLP labels plus
an adjudicated release label. Use --allow-incomplete-ground-truth only for capture
work in progress, never to qualify a public physically-tested release.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Sequence


SCHEMA_VERSION = "1.0.0"
GROUP_FIELDS = ("recipe_id", "batch_id", "kitchen_id")
EXPECTED_FOOD_TESTS = {
    3: {"plate_photo", "spoon_tilt", "fork_drip"},
    4: {"plate_photo", "spoon_tilt", "fork_drip"},
    5: {"plate_photo", "spoon_tilt", "fork_press"},
    6: {"plate_photo", "fork_press"},
    7: {"plate_photo", "fork_press"},
}


@dataclass(frozen=True)
class ValidationError:
    row: int | None
    field: str
    message: str

    def __str__(self) -> str:
        location = "dataset" if self.row is None else f"row {self.row}"
        return f"{location}: {self.field}: {self.message}"


def _json_type_matches(value: Any, expected: str) -> bool:
    if expected == "null":
        return value is None
    if expected == "object":
        return isinstance(value, dict)
    if expected == "array":
        return isinstance(value, list)
    if expected == "string":
        return isinstance(value, str)
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected == "boolean":
        return isinstance(value, bool)
    return False


def _resolve_ref(root_schema: dict[str, Any], ref: str) -> dict[str, Any]:
    if not ref.startswith("#/"):
        raise ValueError(f"only local schema references are supported, got {ref!r}")
    value: Any = root_schema
    for component in ref[2:].split("/"):
        component = component.replace("~1", "/").replace("~0", "~")
        value = value[component]
    return value


def _valid_datetime(value: str) -> bool:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return False
    return "T" in value and parsed.utcoffset() is not None


def _validate_schema_value(
    value: Any,
    schema: dict[str, Any],
    root_schema: dict[str, Any],
    row: int,
    path: str,
) -> list[ValidationError]:
    """Validate the Draft 2020-12 features used by event.schema.json."""
    if "$ref" in schema:
        schema = _resolve_ref(root_schema, schema["$ref"])

    errors: list[ValidationError] = []
    if "const" in schema and json.dumps(value, sort_keys=True) != json.dumps(schema["const"], sort_keys=True):
        errors.append(ValidationError(row, path, f"must equal {schema['const']!r}"))
        return errors
    if "enum" in schema and json.dumps(value, sort_keys=True) not in {
        json.dumps(item, sort_keys=True) for item in schema["enum"]
    }:
        errors.append(ValidationError(row, path, f"must be one of {schema['enum']!r}"))
        return errors

    expected = schema.get("type")
    if expected is not None:
        expected_types = [expected] if isinstance(expected, str) else expected
        if not any(_json_type_matches(value, item) for item in expected_types):
            errors.append(ValidationError(row, path, f"expected {expected!r}, got {type(value).__name__}"))
            return errors

    if isinstance(value, dict):
        required = schema.get("required", [])
        for key in required:
            if key not in value:
                errors.append(ValidationError(row, f"{path}.{key}", "required property is missing"))
        properties = schema.get("properties", {})
        if schema.get("additionalProperties") is False:
            for key in sorted(value.keys() - properties.keys()):
                errors.append(ValidationError(row, f"{path}.{key}", "additional property is not allowed"))
        for key, child_schema in properties.items():
            if key in value:
                errors.extend(
                    _validate_schema_value(value[key], child_schema, root_schema, row, f"{path}.{key}")
                )

    if isinstance(value, list):
        if len(value) < schema.get("minItems", 0):
            errors.append(ValidationError(row, path, f"requires at least {schema['minItems']} item(s)"))
        if schema.get("uniqueItems"):
            serialized = [json.dumps(item, sort_keys=True) for item in value]
            if len(serialized) != len(set(serialized)):
                errors.append(ValidationError(row, path, "items must be unique"))
        item_schema = schema.get("items")
        if item_schema:
            for index, item in enumerate(value):
                errors.extend(
                    _validate_schema_value(item, item_schema, root_schema, row, f"{path}[{index}]")
                )

    if isinstance(value, str):
        if len(value) < schema.get("minLength", 0):
            errors.append(ValidationError(row, path, f"must contain at least {schema['minLength']} character(s)"))
        if "maxLength" in schema and len(value) > schema["maxLength"]:
            errors.append(ValidationError(row, path, f"must contain at most {schema['maxLength']} character(s)"))
        if "pattern" in schema and re.search(schema["pattern"], value) is None:
            errors.append(ValidationError(row, path, "does not match the required pattern"))
        if schema.get("format") == "date-time" and not _valid_datetime(value):
            errors.append(ValidationError(row, path, "must be an ISO 8601 date-time"))

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"]:
            errors.append(ValidationError(row, path, f"must be >= {schema['minimum']}"))
        if "maximum" in schema and value > schema["maximum"]:
            errors.append(ValidationError(row, path, f"must be <= {schema['maximum']}"))

    return errors


def _normalize_parquet_value(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, dict):
        return {key: _normalize_parquet_value(child) for key, child in value.items()}
    if isinstance(value, list):
        return [_normalize_parquet_value(child) for child in value]
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc.msg}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: each JSONL line must be an object")
            rows.append(value)
    return rows


def _read_parquet(path: Path) -> list[dict[str, Any]]:
    try:
        import pyarrow.dataset as ds
    except ImportError as exc:  # pragma: no cover - depends on target image
        raise RuntimeError("Parquet input requires pyarrow; JSONL validation has no dependencies") from exc
    dataset = ds.dataset(path, format="parquet", partitioning="hive" if path.is_dir() else None)
    return [_normalize_parquet_value(row) for row in dataset.to_table().to_pylist()]


def _discover_input(target: Path) -> tuple[Path, Path]:
    """Return (manifest_or_parquet, dataset_root)."""
    if target.is_file():
        if target.suffix not in {".jsonl", ".parquet"}:
            raise ValueError("input file must end in .jsonl or .parquet")
        return target, target.parent
    if not target.is_dir():
        raise ValueError(f"dataset path does not exist: {target}")
    candidates = [item for item in (target / "events.jsonl", target / "events.parquet", target / "events") if item.exists()]
    if not candidates:
        raise ValueError("dataset folder must contain events.jsonl, events.parquet, or an events/ Parquet directory")
    if len(candidates) > 1:
        names = ", ".join(item.name for item in candidates)
        raise ValueError(f"ambiguous dataset folder; found multiple event stores: {names}")
    return candidates[0], target


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _semantic_event_errors(
    event: dict[str, Any],
    row: int,
    dataset_root: Path,
    allow_incomplete_ground_truth: bool,
    verify_hashes: bool,
) -> list[ValidationError]:
    errors: list[ValidationError] = []
    # Stop semantic checks when structural essentials are absent or malformed.
    if not all(key in event for key in ("sample_kind", "level_declared", "test_type", "media_files", "source", "split")):
        return errors
    if not isinstance(event["test_type"], list) or not isinstance(event["media_files"], list):
        return errors

    if not all(isinstance(item, str) for item in event["test_type"]):
        return errors
    tests = set(event["test_type"])
    level = event["level_declared"]
    if event["sample_kind"] == "food":
        if not isinstance(level, int) or isinstance(level, bool) or level not in EXPECTED_FOOD_TESTS:
            errors.append(ValidationError(row, "$.level_declared", "food captures require a declared level from L3 through L7"))
        elif tests != EXPECTED_FOOD_TESTS[level]:
            errors.append(
                ValidationError(row, "$.test_type", f"L{level} food requires exactly {sorted(EXPECTED_FOOD_TESTS[level])!r}")
            )
        for field in ("level_tested_RD", "level_tested_SLP", "level_adjudicated"):
            tested = event.get(field)
            if tested is not None and (
                not isinstance(tested, int)
                or isinstance(tested, bool)
                or tested not in EXPECTED_FOOD_TESTS
            ):
                errors.append(ValidationError(row, f"$.{field}", "food physical-test labels must be from L3 through L7"))
    elif event["sample_kind"] == "liquid":
        if level not in range(0, 5):
            errors.append(ValidationError(row, "$.level_declared", "liquid captures require a declared level from L0 through L4"))
        if tests != {"syringe_flow"}:
            errors.append(ValidationError(row, "$.test_type", "liquid captures require exactly ['syringe_flow']"))
        for field in ("level_tested_RD", "level_tested_SLP", "level_adjudicated"):
            tested = event.get(field)
            if tested is not None and tested not in range(0, 5):
                errors.append(ValidationError(row, f"$.{field}", "liquid physical-test labels must be from L0 through L4"))

    media_roles: set[str] = set()
    for index, media in enumerate(event["media_files"]):
        if not isinstance(media, dict):
            continue
        role = media.get("capture_role")
        media_type = media.get("media_type")
        if isinstance(role, str):
            media_roles.add(role)
            expected_type = "image" if role == "plate_photo" else "video"
            if media_type != expected_type:
                errors.append(ValidationError(row, f"$.media_files[{index}].media_type", f"{role} must use {expected_type}"))
        relative = media.get("path")
        if not isinstance(relative, str):
            continue
        media_path = dataset_root / relative
        try:
            resolved = media_path.resolve()
            resolved.relative_to(dataset_root.resolve())
        except (OSError, ValueError):
            errors.append(ValidationError(row, f"$.media_files[{index}].path", "must resolve inside the dataset root"))
            continue
        if not resolved.is_file():
            errors.append(ValidationError(row, f"$.media_files[{index}].path", f"file does not exist: {relative}"))
        elif verify_hashes and media.get("sha256") is not None and _sha256(resolved) != media["sha256"]:
            errors.append(ValidationError(row, f"$.media_files[{index}].sha256", "does not match the media file"))
        if resolved.is_file() and media.get("bytes") is not None and resolved.stat().st_size != media["bytes"]:
            errors.append(ValidationError(row, f"$.media_files[{index}].bytes", "does not match the media file size"))
    if media_roles != tests:
        errors.append(ValidationError(row, "$.media_files", "capture_role values must exactly cover test_type"))

    source = event["source"]
    gt = event.get("ground_truth_record")
    labels = [event.get("level_tested_RD"), event.get("level_tested_SLP"), event.get("level_adjudicated")]
    if source == "synthetic":
        if event["split"] == "external_test":
            errors.append(ValidationError(row, "$.split", "synthetic events are development-only and cannot enter external_test"))
        if any(label is not None for label in labels):
            errors.append(ValidationError(row, "$.source", "synthetic events cannot carry physically-tested label fields"))
        if isinstance(gt, dict):
            expected_null = (gt.get("rd_assessment_id"), gt.get("slp_assessment_id"), gt.get("adjudicator_code"))
            if gt.get("adjudication_status") != "not_applicable" or any(value is not None for value in expected_null):
                errors.append(ValidationError(row, "$.ground_truth_record", "synthetic events require not_applicable status and null assessment/adjudicator IDs"))
    elif source == "real":
        if event.get("consent_recorded") is not True:
            errors.append(ValidationError(row, "$.consent_recorded", "real capture requires recorded consent"))
        if not allow_incomplete_ground_truth:
            rd, slp, adjudicated = labels
            if any(label is None for label in labels):
                errors.append(ValidationError(row, "$.level_adjudicated", "real release events require RD, SLP, and adjudicated labels"))
            if isinstance(gt, dict):
                rd_id = gt.get("rd_assessment_id")
                slp_id = gt.get("slp_assessment_id")
                status = gt.get("adjudication_status")
                adjudicator = gt.get("adjudicator_code")
                if rd_id is None or slp_id is None:
                    errors.append(ValidationError(row, "$.ground_truth_record", "real release events require RD and SLP assessment IDs"))
                elif rd_id == slp_id:
                    errors.append(ValidationError(row, "$.ground_truth_record", "RD and SLP assessment IDs must identify independent assessments"))
                if rd is not None and slp is not None and adjudicated is not None:
                    if rd == slp and adjudicated != rd:
                        errors.append(ValidationError(row, "$.level_adjudicated", "when RD and SLP agree, the adjudicated label must match"))
                    if rd != slp and (status != "adjudicated" or adjudicator is None):
                        errors.append(ValidationError(row, "$.ground_truth_record", "RD/SLP disagreement requires third-person adjudication"))
                    if status == "adjudicated" and adjudicator is None:
                        errors.append(ValidationError(row, "$.ground_truth_record.adjudicator_code", "adjudicated status requires an adjudicator code"))
                    if status not in {"not_required", "adjudicated"}:
                        errors.append(ValidationError(row, "$.ground_truth_record.adjudication_status", "real release status must be not_required or adjudicated"))

    return errors


def validate_rows(
    rows: Sequence[dict[str, Any]],
    schema: dict[str, Any],
    dataset_root: Path,
    *,
    allow_incomplete_ground_truth: bool = False,
    verify_hashes: bool = False,
    extra_group_fields: Iterable[str] = (),
) -> list[ValidationError]:
    errors: list[ValidationError] = []
    seen_ids: dict[str, int] = {}
    grouped_splits: dict[str, dict[str, set[str]]] = {
        field: defaultdict(set) for field in (*GROUP_FIELDS, *extra_group_fields)
    }

    for row_number, event in enumerate(rows, start=1):
        errors.extend(_validate_schema_value(event, schema, schema, row_number, "$"))
        errors.extend(
            _semantic_event_errors(
                event, row_number, dataset_root, allow_incomplete_ground_truth, verify_hashes
            )
        )
        event_id = event.get("event_id")
        if isinstance(event_id, str):
            if event_id in seen_ids:
                errors.append(ValidationError(row_number, "$.event_id", f"duplicates row {seen_ids[event_id]}"))
            else:
                seen_ids[event_id] = row_number
        split = event.get("split")
        if isinstance(split, str):
            for field, groups in grouped_splits.items():
                group = event.get(field)
                if isinstance(group, str):
                    groups[group].add(split)

    if not rows:
        errors.append(ValidationError(None, "$", "event store is empty"))
    for field, groups in grouped_splits.items():
        for group, splits in sorted(groups.items()):
            if len(splits) > 1:
                errors.append(
                    ValidationError(None, field, f"group {group!r} leaks across splits {sorted(splits)!r}")
                )
    return errors


def validate_dataset(
    target: Path,
    schema_path: Path,
    *,
    allow_incomplete_ground_truth: bool = False,
    verify_hashes: bool = False,
    extra_group_fields: Iterable[str] = (),
) -> tuple[list[ValidationError], list[dict[str, Any]], Path]:
    store, dataset_root = _discover_input(target)
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    rows = _read_jsonl(store) if store.suffix == ".jsonl" else _read_parquet(store)
    errors = validate_rows(
        rows,
        schema,
        dataset_root,
        allow_incomplete_ground_truth=allow_incomplete_ground_truth,
        verify_hashes=verify_hashes,
        extra_group_fields=extra_group_fields,
    )
    return errors, rows, store


def _parser() -> argparse.ArgumentParser:
    default_schema = Path(__file__).resolve().parents[1] / "schema" / "event.schema.json"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path, help="dataset folder, events.jsonl, events.parquet, or events/ directory")
    parser.add_argument("--schema", type=Path, default=default_schema, help="event JSON Schema")
    parser.add_argument(
        "--allow-incomplete-ground-truth",
        action="store_true",
        help="allow real capture rows before RD/SLP/adjudication is complete",
    )
    parser.add_argument("--verify-hashes", action="store_true", help="calculate and check non-null SHA-256 values")
    parser.add_argument(
        "--extra-group-field",
        action="append",
        default=[],
        help="also prevent this field from crossing splits (for example phone_id)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        errors, rows, store = validate_dataset(
            args.dataset,
            args.schema,
            allow_incomplete_ground_truth=args.allow_incomplete_ground_truth,
            verify_hashes=args.verify_hashes,
            extra_group_fields=args.extra_group_field,
        )
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        print(f"FAILED: {len(errors)} error(s) in {len(rows)} event(s) from {store}", file=sys.stderr)
        return 1

    splits = Counter(row["split"] for row in rows)
    sources = Counter(row["source"] for row in rows)
    print(
        f"OK: {len(rows)} event(s) from {store}; "
        f"splits={dict(sorted(splits.items()))}; sources={dict(sorted(sources.items()))}; "
        f"schema={SCHEMA_VERSION}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
