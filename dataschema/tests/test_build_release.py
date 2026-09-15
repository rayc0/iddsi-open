from __future__ import annotations

import base64
import contextlib
import copy
import hashlib
import io
import json
import sys
import tempfile
import unittest
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.build_release import GROUP_FIELDS, _parser, main  # noqa: E402


# Valid 1x1 RGB JPEG generated once with Pillow and embedded so the tests do
# not acquire a Pillow dependency merely to create their temporary fixtures.
TINY_JPEG = base64.b64decode(
    "/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAoHBwgHBgoICAgLCgoLDhgQDg0NDh0VFhEY"
    "Ix8lJCIfIiEmKzcvJik0KSEiMEExNDk7Pj4+JS5ESUM8SDc9Pjv/2wBDAQoLCw4NDhwQ"
    "EBw7KCIoOzs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Oz"
    "s7Ozv/wAARCAABAAEDASIAAhEBAxEB/8QAFQABAQAAAAAAAAAAAAAAAAAAAAf/xAAUEA"
    "EAAAAAAAAAAAAAAAAAAAAA/8QAFAEBAAAAAAAAAAAAAAAAAAAABP/EABQRAQAAAAAAAA"
    "AAAAAAAAAAAAD/2gAMAwEAAhEDEQA/AJqAUK//2Q=="
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )


def _media(path: Path, role: str, media_type: str) -> dict:
    data = path.read_bytes()
    return {
        "path": path.relative_to(path.parents[1]).as_posix(),
        "media_type": media_type,
        "capture_role": role,
        "sha256": hashlib.sha256(data).hexdigest(),
        "bytes": len(data),
    }


def _base_event(event_id: str, index: int) -> dict:
    return {
        "schema_version": "1.0.0",
        "event_id": event_id,
        "recipe_id": f"recipe_{index}",
        "batch_id": f"batch_{index}",
        "kitchen_id": f"kitchen_{index}",
        "phone_id": f"phone_{index}",
        "captured_at": f"2026-09-02T09:0{index}:00+08:00",
        "sample_kind": "food",
        "level_declared": 6,
        "level_tested_RD": None,
        "level_tested_SLP": None,
        "level_adjudicated": None,
        "media_files": [],
        "test_type": ["plate_photo", "fork_press"],
        "capture_conditions": {
            "operator_id": "fixture_operator",
            "lighting_category": "controlled",
            "lighting_notes": "Generated test fixture.",
            "plate_category": "light_plain",
            "plate_notes": "Generated test fixture.",
            "serving_temperature_c": None,
            "cuisine_tags": ["fixture"],
        },
        "ground_truth_record": {
            "rd_assessment_id": None,
            "slp_assessment_id": None,
            "adjudication_status": "not_applicable",
            "adjudicator_code": None,
            "adjudication_notes": "No physical test; generated fixture only.",
        },
        "notes": "Generated release-builder fixture.",
        "split": "train",
        "source": "synthetic",
        "consent_recorded": False,
        "contains_face": False,
    }


def _make_sources(root: Path) -> tuple[Path, Path]:
    synthetic = root / "synthetic"
    realweak = root / "realweak"
    synthetic_media = synthetic / "media"
    realweak_media = realweak / "media"
    synthetic_media.mkdir(parents=True)
    realweak_media.mkdir(parents=True)

    # Each source has exactly two tiny generated JPEGs. Synthetic food events
    # also need a fork-press video role to satisfy the frozen event contract;
    # tiny placeholder MP4 blobs are sufficient because this builder verifies
    # paths, hashes and byte counts rather than decoding media.
    for index in range(2):
        (synthetic_media / f"synthetic_{index}.jpg").write_bytes(TINY_JPEG)
        (synthetic_media / f"synthetic_{index}.mp4").write_bytes(b"fixture video\n")
        (realweak_media / f"realweak_{index}.jpg").write_bytes(TINY_JPEG)

    synthetic_rows = []
    for index in range(2):
        row = _base_event(f"SYN_EVENT_{index}", index)
        row["media_files"] = [
            _media(synthetic_media / f"synthetic_{index}.jpg", "plate_photo", "image"),
            _media(synthetic_media / f"synthetic_{index}.mp4", "fork_press", "video"),
        ]
        row["split"] = "train" if index == 0 else "val"
        synthetic_rows.append(row)
    synthetic_rows[1]["recipe_id"] = synthetic_rows[0]["recipe_id"]

    realweak_rows = []
    for index in range(2):
        row = copy.deepcopy(_base_event(f"RW_EVENT_{index}", index + 2))
        row.update(
            source="real-weak",
            split="train",
            consent_recorded=False,
            level_declared=None,
            level_weak=None,
            weak_confidence=0.0,
            phash=f"{index + 1:016x}",
        )
        row["test_type"] = ["plate_photo"]
        row["media_files"] = [
            _media(realweak_media / f"realweak_{index}.jpg", "plate_photo", "image")
        ]
        row["weak_label_record"] = {
            "model_id": "deferred",
            "prompt_version": "none",
            "rule": "deferred",
            "level_relation": "candidate",
            "observations": {},
            "rationale": "Labelling deferred in generated fixture.",
            "is_ground_truth": False,
        }
        row["privacy_screening"] = {
            "contains_face": False,
            "contains_text": False,
            "face_engine": "fixture",
            "text_engine": "fixture",
            "details": {},
            "vlm_face_check": False,
            "vlm_text_check": False,
        }
        realweak_rows.append(row)

    synthetic_attribution = [{"event_id": row["event_id"]} for row in synthetic_rows]
    realweak_attribution = [
        {
            "event_id": "RW_EVENT_0",
            "attribution": "Fixture author; CC BY 4.0; generated test image.",
            "creator": "Fixture author",
            "license_name": "CC BY 4.0",
            "license_url": "https://creativecommons.org/licenses/by/4.0/",
            "landing_url": "https://example.invalid/by",
            "media_path": "media/realweak_0.jpg",
            "media_sha256": hashlib.sha256(TINY_JPEG).hexdigest(),
            "provider": "fixture",
            "verification_status": "verified",
        },
        {
            "event_id": "RW_EVENT_1",
            "attribution": "Fixture author; CC BY-SA 4.0; generated test image.",
            "creator": "Fixture author",
            "license_name": "CC BY-SA 4.0",
            "license_url": "https://creativecommons.org/licenses/by-sa/4.0/",
            "landing_url": "https://example.invalid/by-sa",
            "media_path": "media/realweak_1.jpg",
            "media_sha256": hashlib.sha256(TINY_JPEG).hexdigest(),
            "provider": "fixture",
            "verification_status": "verified",
        },
    ]
    for index, (row, attribution) in enumerate(zip(realweak_rows, realweak_attribution)):
        row["provenance"] = {
            **attribution,
            "source_id": f"fixture-{index}",
            "download_url": f"https://example.invalid/media/{index}",
            "title": f"Generated fixture image {index}",
            "captured_at_basis": "harvest_time_fallback",
            "harvested_at": "2026-09-02T09:00:00+08:00",
            "transformations": ["generated as a 1x1 RGB JPEG for testing"],
            "source_metadata": {},
        }

    _write_jsonl(synthetic / "events.jsonl", synthetic_rows)
    _write_jsonl(synthetic / "attribution.jsonl", synthetic_attribution)
    _write_jsonl(realweak / "events.jsonl", realweak_rows)
    _write_jsonl(realweak / "attribution.jsonl", realweak_attribution)
    (realweak / "source_licenses.json").write_text(
        json.dumps(
            {"allowed_licence_families": ["CC BY 4.0", "CC BY-SA 4.0"]},
            separators=(",", ":"),
        )
        + "\n",
        encoding="utf-8",
    )
    return synthetic, realweak


def _read_parquet_rows(path: Path) -> list[dict]:
    try:
        import pyarrow.parquet as pq
    except ImportError:
        import polars as pl

        return pl.read_parquet(path).to_dicts()
    return pq.read_table(path).to_pylist()


class BuildReleaseTests(unittest.TestCase):
    def test_cli_builds_manifests_grouped_splits_and_excludes_by_sa(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            synthetic, realweak = _make_sources(root)
            out = root / "release"

            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                result = main(
                    [
                        "--source",
                        f"synthetic={synthetic}",
                        "--source",
                        f"realweak={realweak}",
                        "--out",
                        str(out),
                        "--seed",
                        "23",
                        "--val-fraction",
                        "0.3",
                        "--test-fraction",
                        "0.3",
                    ]
                )

            self.assertEqual(result, 0, stdout.getvalue())
            required = {
                "events.parquet",
                "media_manifest.jsonl",
                "source_counts.json",
                "licence_manifest.json",
            }
            self.assertTrue(required.issubset({path.name for path in out.iterdir()}))

            rows = _read_parquet_rows(out / "events.parquet")
            self.assertEqual({row["event_id"] for row in rows}, {"SYN_EVENT_0", "SYN_EVENT_1", "RW_EVENT_0"})
            self.assertTrue(all(row["licence_partition"] == "main" for row in rows))
            self.assertNotIn("RW_EVENT_1", {row["event_id"] for row in rows})

            for field in GROUP_FIELDS:
                splits_by_group: dict[str, set[str]] = defaultdict(set)
                for row in rows:
                    splits_by_group[row[field]].add(row["split"])
                self.assertTrue(all(len(splits) == 1 for splits in splits_by_group.values()))
            # The two synthetic rows entered with conflicting train/val values
            # but share this recipe, so they must leave in one assigned split.
            shared = [row for row in rows if row["recipe_id"] == "recipe_0"]
            self.assertEqual(len(shared), 2)
            self.assertEqual(len({row["split"] for row in shared}), 1)
            self.assertTrue(
                all(
                    row["split"] != "external_test"
                    for row in rows
                    if row["release_source"] == "synthetic"
                )
            )

            source_counts = json.loads((out / "source_counts.json").read_text(encoding="utf-8"))
            self.assertEqual(source_counts["per_source"], {"real-weak": 1, "synthetic": 2})
            self.assertEqual(source_counts["per_source_input"], {"real-weak": 2, "synthetic": 2})
            self.assertEqual(source_counts["excluded_from_main"]["per_source"], {"real-weak": 1})

            media_manifest = [
                json.loads(line)
                for line in (out / "media_manifest.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            self.assertNotIn("RW_EVENT_1", {entry["event_id"] for entry in media_manifest})
            self.assertTrue(all(entry["licence_partition"] == "main" for entry in media_manifest))

            licence_manifest = json.loads(
                (out / "licence_manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(licence_manifest["dataset_licence"], "CC BY-NC-SA 4.0")
            self.assertEqual(licence_manifest["licence_counts_total"], {"CC BY 4.0": 1})
            excluded = licence_manifest["excluded_from_main"]
            self.assertEqual(excluded["licence_counts"], {"CC BY-SA 4.0": 1})
            self.assertEqual([record["event_id"] for record in excluded["records"]], ["RW_EVENT_1"])
            self.assertIn("excluded from CC BY-NC-SA main release: 1", stdout.getvalue())

            help_text = _parser().format_help()
            self.assertIn("Any row attributed under CC BY-SA is excluded", help_text)

    def test_cli_refuses_realweak_row_without_attribution_and_writes_nothing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            synthetic, realweak = _make_sources(root)
            records = [
                json.loads(line)
                for line in (realweak / "attribution.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            _write_jsonl(
                realweak / "attribution.jsonl",
                [record for record in records if record["event_id"] != "RW_EVENT_0"],
            )
            events = [
                json.loads(line)
                for line in (realweak / "events.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            events[0].pop("provenance")
            _write_jsonl(realweak / "events.jsonl", events)
            out = root / "refused-release"

            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr):
                result = main(
                    [
                        "--source",
                        f"synthetic={synthetic}",
                        "--source",
                        f"realweak={realweak}",
                        "--out",
                        str(out),
                    ]
                )

            self.assertEqual(result, 1)
            self.assertIn("no attribution record", stderr.getvalue())
            self.assertIn("refusing release", stderr.getvalue())
            self.assertFalse(out.exists())

    def test_unverified_per_image_licence_is_excluded_from_release(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            synthetic, realweak = _make_sources(root)
            events = [
                json.loads(line)
                for line in (realweak / "events.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            attribution = [
                json.loads(line)
                for line in (realweak / "attribution.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            events[1]["provenance"].update(
                license_name="CC BY 4.0",
                license_url="https://creativecommons.org/licenses/by/4.0/",
                verification_status="unverified",
            )
            attribution[1].update(
                license_name="CC BY 4.0",
                license_url="https://creativecommons.org/licenses/by/4.0/",
                verification_status="unverified",
            )
            _write_jsonl(realweak / "events.jsonl", events)
            _write_jsonl(realweak / "attribution.jsonl", attribution)
            out = root / "release"

            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                result = main(
                    [
                        "--source",
                        f"synthetic={synthetic}",
                        "--source",
                        f"realweak={realweak}",
                        "--out",
                        str(out),
                    ]
                )

            self.assertEqual(result, 0, stdout.getvalue())
            rows = _read_parquet_rows(out / "events.parquet")
            self.assertNotIn("RW_EVENT_1", {row["event_id"] for row in rows})
            manifest = json.loads((out / "licence_manifest.json").read_text(encoding="utf-8"))
            excluded = manifest["excluded_from_main"]["records"]
            record = next(item for item in excluded if item["event_id"] == "RW_EVENT_1")
            self.assertEqual(record["verification_status"], "unverified")
            self.assertIn("before release", record["reason"])

    def test_absent_verification_status_is_excluded_fail_closed(self) -> None:
        # Legacy harvest runs (e.g. the first open-images batch) predate the
        # verification_status field entirely. An absent field means the
        # per-image verification state is unknown, so the row must not ship.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            synthetic, realweak = _make_sources(root)
            events = [
                json.loads(line)
                for line in (realweak / "events.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            attribution = [
                json.loads(line)
                for line in (realweak / "attribution.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            for row in events:
                row["provenance"].pop("verification_status", None)
            for record in attribution:
                record.pop("verification_status", None)
            attribution[1].update(
                license_name="CC BY 4.0",
                license_url="https://creativecommons.org/licenses/by/4.0/",
            )
            events[1]["provenance"].update(
                license_name="CC BY 4.0",
                license_url="https://creativecommons.org/licenses/by/4.0/",
            )
            _write_jsonl(realweak / "events.jsonl", events)
            _write_jsonl(realweak / "attribution.jsonl", attribution)
            out = root / "release"

            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                result = main(
                    [
                        "--source",
                        f"synthetic={synthetic}",
                        "--source",
                        f"realweak={realweak}",
                        "--out",
                        str(out),
                    ]
                )

            self.assertEqual(result, 0, stdout.getvalue())
            rows = _read_parquet_rows(out / "events.parquet")
            self.assertEqual({row["event_id"] for row in rows}, {"SYN_EVENT_0", "SYN_EVENT_1"})
            manifest = json.loads((out / "licence_manifest.json").read_text(encoding="utf-8"))
            excluded = manifest["excluded_from_main"]
            self.assertEqual(excluded["total_events"], 2)
            for record in excluded["records"]:
                self.assertEqual(record["verification_status"], "unverified")
                self.assertIn("before release", record["reason"])

    def test_real_corpus_shape_release_counts(self) -> None:
        # Pin the counts for a fixture shaped like the harvested corpus:
        # 412 open-images + 180 openverse (all verification_status=unverified),
        # 109 nutrition5k (CC BY 4.0, verified), and 12 wikimedia (7 CC BY-SA,
        # 5 verified CC BY/CC0). Only the nutrition5k and non-sharealike
        # wikimedia rows may ship in the CC BY-NC-SA main release.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            realweak = root / "realweak"
            media_dir = realweak / "media"
            media_dir.mkdir(parents=True)
            (media_dir / "img.jpg").write_bytes(TINY_JPEG)
            media_entry = _media(media_dir / "img.jpg", "plate_photo", "image")

            def make_row(index: int, provider: str, licence: str, status: str) -> dict:
                row = copy.deepcopy(_base_event(f"RW_{provider}_{index}", index % 10))
                row.update(source="real-weak", split="train", level_declared=None)
                row["test_type"] = ["plate_photo"]
                row["media_files"] = [dict(media_entry)]
                row["kitchen_id"] = f"RW_{provider}"
                row["provenance"] = {
                    "event_id": row["event_id"],
                    "attribution": f"Fixture {provider}; {licence}; https://example.invalid/{index}",
                    "creator": "Fixture author",
                    "license_name": licence,
                    "license_url": "https://example.invalid/licence",
                    "landing_url": f"https://example.invalid/{index}",
                    "provider": provider,
                    "verification_status": status,
                }
                return row

            rows = []
            for index in range(412):
                rows.append(make_row(index, "open-images", "CC BY 2.0", "unverified"))
            for index in range(180):
                rows.append(make_row(412 + index, "openverse", "CC BY 2.0", "unverified"))
            for index in range(109):
                rows.append(make_row(592 + index, "nutrition5k", "CC BY 4.0", "verified"))
            wikimedia_licences = (
                ["CC BY-SA 4.0"] * 7 + ["CC BY 4.0"] * 4 + ["CC0 1.0"]
            )
            for index, licence in enumerate(wikimedia_licences):
                rows.append(make_row(701 + index, "wikimedia", licence, "verified"))

            attribution = [dict(row["provenance"]) for row in rows]
            _write_jsonl(realweak / "events.jsonl", rows)
            _write_jsonl(realweak / "attribution.jsonl", attribution)
            (realweak / "source_licenses.json").write_text(
                json.dumps(
                    {
                        "allowed_licence_families": [
                            "CC0 1.0",
                            "CC BY 2.0",
                            "CC BY 4.0",
                            "CC BY-SA 4.0",
                        ]
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            out = root / "release"

            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                result = main(["--source", f"realweak={realweak}", "--out", str(out)])

            self.assertEqual(result, 0, stdout.getvalue())
            parquet_rows = _read_parquet_rows(out / "events.parquet")
            self.assertEqual(len(parquet_rows), 114)  # 109 nutrition5k + 5 wikimedia
            shipped = Counter(row["kitchen_id"] for row in parquet_rows)
            self.assertEqual(shipped, {"RW_nutrition5k": 109, "RW_wikimedia": 5})

            source_counts = json.loads((out / "source_counts.json").read_text(encoding="utf-8"))
            self.assertEqual(source_counts["input_events"], 713)
            self.assertEqual(source_counts["total_events"], 114)
            self.assertEqual(source_counts["excluded_from_main"]["total_events"], 599)

            manifest = json.loads((out / "licence_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(
                manifest["licence_counts_total"], {"CC BY 4.0": 113, "CC0 1.0": 1}
            )
            excluded = manifest["excluded_from_main"]
            self.assertEqual(
                excluded["licence_counts"], {"CC BY 2.0": 592, "CC BY-SA 4.0": 7}
            )
            reasons = Counter(record["reason"] for record in excluded["records"])
            self.assertEqual(sum(reasons.values()), 599)
            unverified_reasons = [r for r in reasons if "unverified" in r]
            sharealike_reasons = [r for r in reasons if "CC BY-SA" in r]
            self.assertEqual(sum(reasons[r] for r in unverified_reasons), 592)
            self.assertEqual(sum(reasons[r] for r in sharealike_reasons), 7)


if __name__ == "__main__":
    unittest.main()
