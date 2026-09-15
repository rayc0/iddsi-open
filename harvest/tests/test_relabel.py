from __future__ import annotations

import io
import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from harvest.label import DeferLabeler, _extract_json, apply_descriptor_rules
from harvest.manifest import read_jsonl, validate_realweak_event
from harvest.pipeline import Harvester
from harvest.relabel import apply_weak_label, is_deferred, relabel_dataset
from harvest.types import Candidate, PrivacyResult


ROOT = Path(__file__).resolve().parents[2]


def image_bytes(color: tuple[int, int, int] = (120, 80, 40)) -> bytes:
    output = io.BytesIO()
    Image.new("RGB", (96, 96), color).save(output, format="PNG")
    return output.getvalue()


class FakeHTTP:
    def __init__(self) -> None:
        self.byte_routes: dict[str, bytes] = {}

    def get_bytes(self, url: str, params=None, **kwargs):
        return self.byte_routes[url]


class FakeSource:
    name = "wikimedia"

    def __init__(self, candidates: list[Candidate]) -> None:
        self.candidates = candidates

    def discover(self, max_hint: int):
        yield from self.candidates[:max_hint]


class ClearScreener:
    def screen(self, image):
        return PrivacyResult(False, False, "test-face", "test-text", {})


class SmoothLabeler:
    """CPU-safe fake VLM: deterministic smooth-L4 observations, no model load."""

    def label(self, image):
        return apply_descriptor_rules(
            {
                "visible_liquid_flow": False,
                "uniform_smooth": True,
                "visible_particles": False,
                "max_particle_mm": None,
                "visible_pieces": False,
                "typical_piece_mm": None,
                "contains_face": False,
                "contains_legible_text": False,
                "confidence": 0.77,
                "rationale": "uniform smooth appearance",
            },
            model_id="fake-qwen",
        )


class FaceFlagLabeler:
    """Fake VLM that reports a face: relabel must fail closed and keep deferred."""

    def label(self, image):
        return apply_descriptor_rules(
            {
                "uniform_smooth": True,
                "contains_face": True,
                "contains_legible_text": False,
                "confidence": 0.5,
                "rationale": "face visible",
            },
            model_id="fake-qwen",
        )


def candidate(source_id: str, url: str) -> Candidate:
    return Candidate(
        provider="wikimedia",
        source_id=source_id,
        download_url=url,
        landing_url=f"https://commons.wikimedia.org/wiki/File:{source_id}.jpg",
        license_name="CC BY 4.0",
        license_url="https://creativecommons.org/licenses/by/4.0/",
        creator="Test Creator",
        title=f"Test {source_id}",
        attribution=f"Test {source_id} by Test Creator, CC BY 4.0",
        captured_at="2026-01-02T03:04:05Z",
    )


def build_deferred_dataset(root: Path, count: int = 2) -> Path:
    http = FakeHTTP()
    candidates = []
    for index in range(count):
        url = f"https://upload.wikimedia.org/{index}.png"
        http.byte_routes[url] = image_bytes(((index * 97) % 256, (index * 53) % 256, 40))
        candidates.append(candidate(f"img{index}", url))
    harvester = Harvester(
        http=http,
        sources=[FakeSource(candidates)],
        labeler=DeferLabeler(),
        screener=ClearScreener(),
        provider_cap=1.0,
    )
    output = root / "dataset"
    summary = harvester.run(maximum=count, output=output)
    assert summary.accepted == count, summary
    return output


class RelabelTests(unittest.TestCase):
    def test_is_deferred_detects_rule_field(self) -> None:
        self.assertTrue(is_deferred({"weak_label_record": {"rule": "deferred"}}))
        self.assertFalse(is_deferred({"weak_label_record": {"rule": "uniform_smooth_l4_candidate"}}))
        self.assertFalse(is_deferred({"weak_label_record": {}}))
        self.assertFalse(is_deferred({}))

    def test_apply_weak_label_mirrors_build_event_wiring(self) -> None:
        weak = SmoothLabeler().label(None)
        event = {
            "level_weak": None,
            "level_declared": None,
            "weak_confidence": 0.0,
            "sample_kind": "food",
            "weak_label_record": {
                "model_id": "deferred",
                "prompt_version": "none",
                "rule": "deferred",
                "level_relation": "candidate",
                "observations": {},
                "rationale": "labelling deferred to GPU pass",
                "is_ground_truth": False,
            },
            "privacy_screening": {"vlm_face_check": False, "vlm_text_check": False},
        }
        updated = apply_weak_label(event, weak)
        self.assertEqual(updated["level_weak"], 4)
        self.assertEqual(updated["level_declared"], 4)
        self.assertEqual(updated["weak_confidence"], 0.77)
        self.assertEqual(updated["sample_kind"], "food")
        record = updated["weak_label_record"]
        self.assertEqual(record["model_id"], "fake-qwen")
        self.assertEqual(record["rule"], "uniform_smooth_l4_candidate")
        self.assertFalse(record["is_ground_truth"])
        # original event dict is not mutated
        self.assertIsNone(event["level_weak"])

    def test_liquid_relation_flips_sample_kind(self) -> None:
        weak = apply_descriptor_rules(
            {"visible_liquid_flow": True, "confidence": 0.5}, model_id="fake-qwen"
        )
        updated = apply_weak_label({"privacy_screening": {}}, weak)
        self.assertEqual(updated["sample_kind"], "liquid")
        self.assertEqual(updated["level_weak"], 3)

    def test_relabel_rewrites_deferred_rows_atomically_with_backup(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            dataset = build_deferred_dataset(Path(tmp), count=2)
            events_path = dataset / "events.jsonl"
            attribution_before = (dataset / "attribution.jsonl").read_bytes()
            original_lines = events_path.read_text(encoding="utf-8").splitlines()

            summary = relabel_dataset(dataset, SmoothLabeler())

            self.assertEqual(summary.total, 2)
            self.assertEqual(summary.deferred, 2)
            self.assertEqual(summary.relabelled, 2)
            self.assertEqual(summary.vlm_flagged, 0)
            self.assertEqual(summary.already_labelled, 0)

            # backup holds the exact original bytes
            backup = Path(summary.backup_path)
            self.assertTrue(backup.is_file())
            self.assertEqual(backup.read_text(encoding="utf-8").splitlines(), original_lines)
            self.assertTrue(backup.name.startswith("events.jsonl.bak-"))

            # no temp file left behind
            self.assertFalse((dataset / "events.jsonl.relab.tmp").exists())

            rows = read_jsonl(events_path)
            self.assertEqual(len(rows), 2)
            for row in rows:
                self.assertEqual(row["level_weak"], 4)
                self.assertEqual(row["level_declared"], 4)
                self.assertEqual(row["weak_label_record"]["model_id"], "fake-qwen")
                self.assertEqual(
                    row["weak_label_record"]["rule"], "uniform_smooth_l4_candidate"
                )
                errors = validate_realweak_event(
                    row, ROOT / "dataschema" / "schema" / "event.schema.json"
                )
                self.assertEqual(errors, [])

            # attribution ledger untouched
            self.assertEqual((dataset / "attribution.jsonl").read_bytes(), attribution_before)

    def test_relabel_is_idempotent_when_nothing_deferred(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            dataset = build_deferred_dataset(Path(tmp), count=1)
            first = relabel_dataset(dataset, SmoothLabeler())
            self.assertEqual(first.relabelled, 1)
            after_first = (dataset / "events.jsonl").read_bytes()

            second = relabel_dataset(dataset, SmoothLabeler())
            self.assertEqual(second.total, 1)
            self.assertEqual(second.deferred, 0)
            self.assertEqual(second.relabelled, 0)
            self.assertEqual(second.already_labelled, 1)
            self.assertEqual(second.backup_path, "")
            # no second backup, file untouched
            self.assertEqual((dataset / "events.jsonl").read_bytes(), after_first)
            backups = list(dataset.glob("events.jsonl.bak-*"))
            self.assertEqual(len(backups), 1)

    def test_vlm_face_flag_keeps_row_deferred_and_writes_nothing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            dataset = build_deferred_dataset(Path(tmp), count=1)
            before = (dataset / "events.jsonl").read_bytes()
            summary = relabel_dataset(dataset, FaceFlagLabeler())
            self.assertEqual(summary.relabelled, 0)
            self.assertEqual(summary.vlm_flagged, 1)
            self.assertEqual(summary.backup_path, "")
            self.assertEqual((dataset / "events.jsonl").read_bytes(), before)
            self.assertEqual(list(dataset.glob("events.jsonl.bak-*")), [])

    def test_missing_media_raises_and_leaves_manifest_untouched(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            dataset = build_deferred_dataset(Path(tmp), count=1)
            row = read_jsonl(dataset / "events.jsonl")[0]
            media = dataset / row["media_files"][0]["path"]
            media.rename(dataset / "moved.jpg")
            before = (dataset / "events.jsonl").read_bytes()
            with self.assertRaises(ValueError):
                relabel_dataset(dataset, SmoothLabeler())
            self.assertEqual((dataset / "events.jsonl").read_bytes(), before)
            self.assertEqual(list(dataset.glob("events.jsonl.bak-*")), [])

    def test_missing_events_file_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                relabel_dataset(Path(tmp), SmoothLabeler())


if __name__ == "__main__":
    unittest.main()


class ExtractJsonTruncationTests(unittest.TestCase):
    """A VLM that runs out of tokens mid-rationale must still yield its label.

    Regression for the 2026-09-06 relabel failure: Qwen3-VL-2B emitted a fenced
    JSON object but hit max_new_tokens inside `rationale`, so neither the fence
    nor the closing brace arrived and every image raised "VLM did not return a
    JSON object" -- producing zero labels on three consecutive nights.
    """

    TRUNCATED = (
        '```json\n{\n  "visible_liquid_flow": false,\n  "uniform_smooth": true,\n'
        '  "visible_particles": false,\n  "max_particle_mm": null,\n'
        '  "visible_pieces": true,\n  "typical_piece_mm": 15,\n'
        '  "contains_face": false,\n  "contains_legible_text": false,\n'
        '  "confidence": 0.85,\n  "rationale": "The image shows a white plate with'
    )

    def test_recovers_fields_before_truncation(self) -> None:
        parsed = _extract_json(self.TRUNCATED)
        self.assertEqual(parsed["typical_piece_mm"], 15)
        self.assertTrue(parsed["visible_pieces"])
        self.assertEqual(parsed["confidence"], 0.85)

    def test_complete_json_still_parses(self) -> None:
        parsed = _extract_json('```json\n{"uniform_smooth": true}\n```')
        self.assertTrue(parsed["uniform_smooth"])

    def test_garbage_still_raises(self) -> None:
        with self.assertRaises(ValueError):
            _extract_json("I cannot analyse this image.")
