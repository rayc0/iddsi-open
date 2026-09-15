"""Tier-2 judge: re-label an existing weak-labelled dataset with a stronger VLM.

Reads a dataset directory with ``events.jsonl`` plus media (as produced by the
tier-1 weak labeler, e.g. Qwen3-VL-2B via :mod:`harvest.label`), re-judges each
event's image with a tier-2 model (default ``Qwen/Qwen3-VL-8B-Instruct``), and
writes a SEPARATE sidecar file ``tier2_labels.jsonl`` next to the manifest,
keyed by ``event_id``. ``events.jsonl`` is opened read-only and never modified.

Each sidecar row records:
- ``event_id``: the event key;
- ``level_tier2``: the tier-2 weak level (3-7, or null when unmeasurable);
- ``confidence_tier2``: the tier-2 confidence in [0, 1];
- ``level_weak``: the existing tier-1 level copied from the event row;
- ``agrees``: whether tier-2 agrees with the existing ``level_weak``;
- ``rule`` / ``model_id`` / ``prompt_version``: tier-2 provenance.

A summary (overall agreement rate, per-level agreement, and the tier-1 x tier-2
confusion matrix) is printed to stdout and optionally written as JSON.

The model-loading/inference half lives behind :class:`Tier2Labeler` (a thin
subclass of the tier-1 :class:`harvest.label.Qwen3VLLabeler`, so both tiers
speak the same prompt/output format). :class:`StubTier2Labeler` implements the
same ``label(image) -> WeakLabel`` interface deterministically with no model
load, so ``--dry-run`` and the test suite need no GPU and no model download.
The real model is loaded lazily on first ``label()`` call, never at import.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from .image_ops import decode_image
from .label import Qwen3VLLabeler, apply_descriptor_rules
from .manifest import read_jsonl
from .relabel import _resolve_media
from .types import WeakLabel

SIDECAR_NAME = "tier2_labels.jsonl"
DEFAULT_TIER2_MODEL = "Qwen/Qwen3-VL-8B-Instruct"
LEVELS = (3, 4, 5, 6, 7)


class Tier2Labeler(Qwen3VLLabeler):
    """Tier-2 judge labeler: tier-1 machinery with a stronger default model.

    Inherits the lazy loader from :class:`harvest.label.Qwen3VLLabeler`, so
    transformers/torch are only imported and the weights only read from
    ``HF_HOME`` when ``label()`` is first called — never at import time.
    Reuses the exact tier-1 prompt and descriptor-rule parsing, so tier-1 and
    tier-2 rows are directly comparable.
    """

    def __init__(
        self,
        model_id: str = DEFAULT_TIER2_MODEL,
        *,
        device: str = "auto",
        max_new_tokens: int = 384,
    ) -> None:
        super().__init__(model_id, device=device, max_new_tokens=max_new_tokens)


class StubTier2Labeler:
    """Deterministic no-model labeler for ``--dry-run`` and tests.

    Implements the same ``label(image) -> WeakLabel`` interface as
    :class:`Tier2Labeler`. The returned level is derived from the image's
    average red channel so different synthetic fixtures can exercise both the
    agree and disagree paths without any weights on disk.
    """

    model_id = "stub-tier2"
    prompt_version = "stub"

    def label(self, image: Any) -> WeakLabel:
        red = 0
        if image is not None:
            try:
                small = image.convert("RGB").resize((1, 1))
                red = small.getpixel((0, 0))[0]
            except Exception:  # pragma: no cover - defensive only
                red = 0
        # Map the red channel onto the L4/L6/L7 buckets deterministically.
        if red < 85:
            observations: dict[str, Any] = {
                "uniform_smooth": True,
                "visible_particles": False,
                "visible_pieces": False,
            }
        elif red < 170:
            observations = {"visible_pieces": True, "typical_piece_mm": 15.0}
        else:
            observations = {"visible_particles": True, "max_particle_mm": 12.0}
        observations.update(
            {
                "contains_face": False,
                "contains_legible_text": False,
                "confidence": 0.9,
                "rationale": "stub tier-2 judgement",
            }
        )
        return apply_descriptor_rules(observations, model_id=self.model_id)


@dataclass
class Tier2Summary:
    total: int
    judged: int
    errors: int
    compared: int
    agree: int
    agreement_rate: float | None
    per_level: dict[str, dict[str, Any]] = field(default_factory=dict)
    confusion: dict[str, dict[str, int]] = field(default_factory=dict)
    sidecar_path: str = ""
    events_path: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "total": self.total,
            "judged": self.judged,
            "errors": self.errors,
            "compared": self.compared,
            "agree": self.agree,
            "agreement_rate": self.agreement_rate,
            "per_level": self.per_level,
            "confusion": self.confusion,
            "sidecar_path": self.sidecar_path,
            "events_path": self.events_path,
        }


def judge_dataset(
    dataset: Path,
    labeler: Any,
    *,
    limit: int | None = None,
) -> Tier2Summary:
    """Re-judge ``dataset`` with ``labeler`` and write the tier-2 sidecar.

    ``events.jsonl`` is only ever read. The sidecar is written atomically
    (temp file + ``os.replace``) so an interrupted run cannot leave a
    half-written ``tier2_labels.jsonl``.
    """
    import os

    dataset = dataset.resolve()
    events_path = dataset / "events.jsonl"
    if not events_path.is_file():
        raise ValueError(f"dataset has no events.jsonl: {dataset}")
    sidecar_path = dataset / SIDECAR_NAME

    rows = read_jsonl(events_path)
    if limit is not None:
        rows = rows[:limit]

    out_rows: list[dict[str, Any]] = []
    judged = 0
    errors = 0
    compared = 0
    agree = 0
    per_level: dict[str, dict[str, Any]] = {}
    confusion: dict[str, dict[str, int]] = {}

    for row in rows:
        event_id = row.get("event_id")
        level_weak = row.get("level_weak")
        record: dict[str, Any] = {
            "event_id": event_id,
            "level_weak": level_weak,
        }
        try:
            media_path = _resolve_media(dataset, row)
            image = decode_image(media_path.read_bytes())
            weak = labeler.label(image)
        except (OSError, ValueError, RuntimeError) as exc:
            errors += 1
            record.update({"error": str(exc), "agrees": None})
            out_rows.append(record)
            continue

        judged += 1
        level_tier2 = weak.level_weak
        agrees: bool | None = None
        if level_weak is not None and level_tier2 is not None:
            agrees = int(level_tier2) == int(level_weak)
            compared += 1
            if agrees:
                agree += 1
            key = str(level_weak)
            bucket = per_level.setdefault(key, {"compared": 0, "agree": 0})
            bucket["compared"] += 1
            bucket["agree"] += int(agrees)
            row_key = f"tier1_{level_weak}"
            col_key = f"tier2_{level_tier2}"
            matrix_row = confusion.setdefault(row_key, {})
            matrix_row[col_key] = matrix_row.get(col_key, 0) + 1
        record.update(
            {
                "level_tier2": level_tier2,
                "confidence_tier2": weak.confidence,
                "agrees": agrees,
                "rule": weak.rule,
                "model_id": weak.model_id,
                "prompt_version": weak.prompt_version,
            }
        )
        out_rows.append(record)

    for bucket in per_level.values():
        bucket["agreement_rate"] = (
            bucket["agree"] / bucket["compared"] if bucket["compared"] else None
        )

    temporary = sidecar_path.with_name(f"{sidecar_path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in out_rows:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
            handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, sidecar_path)

    return Tier2Summary(
        total=len(rows),
        judged=judged,
        errors=errors,
        compared=compared,
        agree=agree,
        agreement_rate=(agree / compared) if compared else None,
        per_level=per_level,
        confusion=confusion,
        sidecar_path=str(sidecar_path),
        events_path=str(events_path),
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Re-judge a weak-labelled dataset with the tier-2 model "
            f"({DEFAULT_TIER2_MODEL}) and write {SIDECAR_NAME} next to "
            "events.jsonl (which is never modified)."
        )
    )
    parser.add_argument(
        "--dataset", type=Path, required=True, help="dataset directory with events.jsonl"
    )
    parser.add_argument("--model", default=DEFAULT_TIER2_MODEL)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--max-new-tokens", type=int, default=384)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="judge at most N events (cheap smoke runs)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="use the deterministic stub labeler; no model load, no GPU",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=None,
        help="optional path to also write the summary as JSON",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.dry_run:
        labeler: Any = StubTier2Labeler()
    else:
        labeler = Tier2Labeler(
            args.model, device=args.device, max_new_tokens=args.max_new_tokens
        )
    try:
        summary = judge_dataset(args.dataset, labeler, limit=args.limit)
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    payload = json.dumps(summary.as_dict(), indent=2, sort_keys=True)
    print(payload)
    if args.summary_json is not None:
        args.summary_json.write_text(payload + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
