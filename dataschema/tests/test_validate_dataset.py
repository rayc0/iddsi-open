from __future__ import annotations

import copy
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.validate_dataset import validate_dataset  # noqa: E402


class ValidateDatasetTests(unittest.TestCase):
    def setUp(self) -> None:
        self.schema = ROOT / "schema" / "event.schema.json"
        self.fixture = ROOT / "tests" / "fixtures" / "valid_dataset"

    def _copy_fixture(self, destination: Path) -> list[dict]:
        shutil.copytree(self.fixture, destination)
        return [json.loads(line) for line in (destination / "events.jsonl").read_text().splitlines()]

    @staticmethod
    def _write_rows(destination: Path, rows: list[dict]) -> None:
        text = "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows)
        (destination / "events.jsonl").write_text(text, encoding="utf-8")

    def test_tiny_synthetic_fixture_is_valid(self) -> None:
        errors, rows, _ = validate_dataset(self.fixture, self.schema)
        self.assertEqual(errors, [])
        self.assertEqual(len(rows), 2)

    def test_recipe_cannot_cross_splits(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            dataset = Path(tmp) / "dataset"
            rows = self._copy_fixture(dataset)
            rows[1]["recipe_id"] = rows[0]["recipe_id"]
            self._write_rows(dataset, rows)
            errors, _, _ = validate_dataset(dataset, self.schema)
            self.assertTrue(any(error.field == "recipe_id" and "leaks" in error.message for error in errors))

    def test_missing_media_and_face_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            dataset = Path(tmp) / "dataset"
            rows = self._copy_fixture(dataset)
            # JSON booleans and integers are distinct; 0 must not satisfy const:false.
            rows[0]["contains_face"] = 0
            rows[0]["media_files"][0]["path"] = "media/K01/not-there.jpg"
            self._write_rows(dataset, rows)
            errors, _, _ = validate_dataset(dataset, self.schema)
            messages = "\n".join(str(error) for error in errors)
            self.assertIn("must equal False", messages)
            self.assertIn("file does not exist", messages)

    def test_synthetic_event_cannot_enter_external_test(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            dataset = Path(tmp) / "dataset"
            rows = self._copy_fixture(dataset)
            rows[1]["split"] = "external_test"
            self._write_rows(dataset, rows)
            errors, _, _ = validate_dataset(dataset, self.schema)
            self.assertTrue(any("development-only" in error.message for error in errors))

    def test_real_disagreement_requires_third_person_adjudication(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            dataset = Path(tmp) / "dataset"
            rows = self._copy_fixture(dataset)
            event = copy.deepcopy(rows[0])
            event.update(
                source="real",
                consent_recorded=True,
                level_tested_RD=5,
                level_tested_SLP=6,
                level_adjudicated=5,
            )
            event["ground_truth_record"] = {
                "rd_assessment_id": "rd-a01",
                "slp_assessment_id": "slp-a01",
                "adjudication_status": "not_required",
                "adjudicator_code": None,
                "adjudication_notes": "",
            }
            rows[0] = event
            self._write_rows(dataset, rows)
            errors, _, _ = validate_dataset(dataset, self.schema)
            self.assertTrue(any("third-person adjudication" in error.message for error in errors))


if __name__ == "__main__":
    unittest.main()
