from __future__ import annotations

import io
import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from harvest.manifest import read_jsonl
from harvest.tier2_judge import (
    SIDECAR_NAME,
    StubTier2Labeler,
    Tier2Labeler,
    judge_dataset,
    main,
)


def _png_bytes(color: tuple[int, int, int]) -> bytes:
    output = io.BytesIO()
    Image.new("RGB", (96, 96), color).save(output, format="PNG")
    return output.getvalue()


def _write_event(dataset: Path, event_id: str, level_weak, color: tuple[int, int, int]) -> None:
    media_rel = f"media/{event_id}.png"
    media_path = dataset / media_rel
    media_path.parent.mkdir(parents=True, exist_ok=True)
    media_path.write_bytes(_png_bytes(color))
    row = {
        "event_id": event_id,
        "level_weak": level_weak,
        "media_files": [{"media_type": "image", "path": media_rel}],
    }
    with (dataset / "events.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


def build_dataset(root: Path) -> Path:
    """Three events whose stub tier-2 levels are 4, 6, 7 (by red channel)."""
    dataset = root / "dataset"
    dataset.mkdir(parents=True)
    (dataset / "events.jsonl").write_text("", encoding="utf-8")
    _write_event(dataset, "ev_agree4", 4, (10, 0, 0))      # stub -> L4, agrees
    _write_event(dataset, "ev_disagree", 4, (120, 0, 0))   # stub -> L6, disagrees
    _write_event(dataset, "ev_agree7", 7, (250, 0, 0))     # stub -> L7, agrees
    return dataset


class StubLabelerTests(unittest.TestCase):
    def test_stub_levels_are_deterministic(self) -> None:
        stub = StubTier2Labeler()
        self.assertEqual(stub.label(Image.new("RGB", (4, 4), (10, 0, 0))).level_weak, 4)
        self.assertEqual(stub.label(Image.new("RGB", (4, 4), (120, 0, 0))).level_weak, 6)
        self.assertEqual(stub.label(Image.new("RGB", (4, 4), (250, 0, 0))).level_weak, 7)

    def test_tier2_labeler_defaults_to_8b_without_loading(self) -> None:
        labeler = Tier2Labeler()
        self.assertEqual(labeler.model_id, "Qwen/Qwen3-VL-8B-Instruct")
        # lazy: nothing loaded at construction time
        self.assertIsNone(labeler._model)
        self.assertIsNone(labeler._processor)


class JudgeDatasetTests(unittest.TestCase):
    def test_sidecar_written_and_events_untouched(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            dataset = build_dataset(Path(tmp))
            events_path = dataset / "events.jsonl"
            before = events_path.read_bytes()

            summary = judge_dataset(dataset, StubTier2Labeler())

            self.assertEqual(events_path.read_bytes(), before)
            sidecar = dataset / SIDECAR_NAME
            self.assertTrue(sidecar.is_file())
            self.assertEqual(summary.sidecar_path, str(sidecar.resolve()))
            self.assertFalse((dataset / f"{SIDECAR_NAME}.tmp").exists())

            rows = read_jsonl(sidecar)
            self.assertEqual(len(rows), 3)
            by_id = {row["event_id"]: row for row in rows}
            self.assertEqual(by_id["ev_agree4"]["level_tier2"], 4)
            self.assertTrue(by_id["ev_agree4"]["agrees"])
            self.assertEqual(by_id["ev_disagree"]["level_tier2"], 6)
            self.assertFalse(by_id["ev_disagree"]["agrees"])
            self.assertEqual(by_id["ev_disagree"]["level_weak"], 4)
            self.assertTrue(by_id["ev_agree7"]["agrees"])
            for row in rows:
                self.assertEqual(row["model_id"], "stub-tier2")
                self.assertIn("confidence_tier2", row)

    def test_summary_math(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            dataset = build_dataset(Path(tmp))
            summary = judge_dataset(dataset, StubTier2Labeler())

            self.assertEqual(summary.total, 3)
            self.assertEqual(summary.judged, 3)
            self.assertEqual(summary.errors, 0)
            self.assertEqual(summary.compared, 3)
            self.assertEqual(summary.agree, 2)
            self.assertAlmostEqual(summary.agreement_rate, 2 / 3)
            self.assertEqual(
                summary.per_level["4"],
                {"compared": 2, "agree": 1, "agreement_rate": 0.5},
            )
            self.assertEqual(
                summary.per_level["7"],
                {"compared": 1, "agree": 1, "agreement_rate": 1.0},
            )
            self.assertEqual(summary.confusion["tier1_4"], {"tier2_4": 1, "tier2_6": 1})
            self.assertEqual(summary.confusion["tier1_7"], {"tier2_7": 1})

    def test_limit_restricts_judged_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            dataset = build_dataset(Path(tmp))
            summary = judge_dataset(dataset, StubTier2Labeler(), limit=1)
            self.assertEqual(summary.total, 1)
            self.assertEqual(summary.judged, 1)
            rows = read_jsonl(dataset / SIDECAR_NAME)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["event_id"], "ev_agree4")

    def test_missing_media_records_error_row(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            dataset = build_dataset(Path(tmp))
            (dataset / "media" / "ev_disagree.png").unlink()
            summary = judge_dataset(dataset, StubTier2Labeler())
            self.assertEqual(summary.errors, 1)
            self.assertEqual(summary.judged, 2)
            rows = read_jsonl(dataset / SIDECAR_NAME)
            by_id = {row["event_id"]: row for row in rows}
            self.assertIn("error", by_id["ev_disagree"])
            self.assertIsNone(by_id["ev_disagree"]["agrees"])

    def test_missing_events_file_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                judge_dataset(Path(tmp), StubTier2Labeler())


class CliTests(unittest.TestCase):
    def test_dry_run_cli_writes_sidecar_and_summary_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            dataset = build_dataset(Path(tmp))
            before = (dataset / "events.jsonl").read_bytes()
            summary_path = Path(tmp) / "summary.json"
            rc = main(
                [
                    "--dataset",
                    str(dataset),
                    "--dry-run",
                    "--summary-json",
                    str(summary_path),
                ]
            )
            self.assertEqual(rc, 0)
            self.assertEqual((dataset / "events.jsonl").read_bytes(), before)
            self.assertTrue((dataset / SIDECAR_NAME).is_file())
            payload = json.loads(summary_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["judged"], 3)
            self.assertEqual(payload["agree"], 2)

    def test_dry_run_cli_limit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            dataset = build_dataset(Path(tmp))
            rc = main(["--dataset", str(dataset), "--dry-run", "--limit", "2"])
            self.assertEqual(rc, 0)
            rows = read_jsonl(dataset / SIDECAR_NAME)
            self.assertEqual(len(rows), 2)

    def test_cli_missing_dataset_returns_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            rc = main(["--dataset", str(Path(tmp) / "nope"), "--dry-run"])
            self.assertEqual(rc, 1)


if __name__ == "__main__":
    unittest.main()
