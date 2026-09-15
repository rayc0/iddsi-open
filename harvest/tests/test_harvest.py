from __future__ import annotations

import io
import json
import tempfile
import unittest
from pathlib import Path
from typing import Any

from PIL import Image

from harvest.label import apply_descriptor_rules
from harvest.licenses import evaluate_license
from harvest.manifest import read_jsonl, validate_realweak_event
from harvest.pipeline import Harvester
from harvest.run import SOURCE_NAMES, _parser
from harvest.sources.nutrition5k import LIST_URL, Nutrition5kSource
from harvest.sources.open_images import CLASS_URL, LABEL_URL, METADATA_URL, OpenImagesSource
from harvest.sources.openverse import API_URL as OPENVERSE_API_URL, QUERIES, OpenverseSource
from harvest.sources.wikimedia import API_URL, CATEGORIES, WikimediaCommonsSource
from harvest.types import Candidate, PrivacyResult


ROOT = Path(__file__).resolve().parents[2]


def image_bytes(color: tuple[int, int, int] = (120, 80, 40)) -> bytes:
    output = io.BytesIO()
    Image.new("RGB", (96, 96), color).save(output, format="PNG")
    return output.getvalue()


class FakeHTTP:
    def __init__(self) -> None:
        self.json_routes: dict[str, Any] = {}
        self.text_routes: dict[str, str] = {}
        self.byte_routes: dict[str, bytes] = {}
        self.calls: list[tuple[str, str, Any]] = []

    def get_json(self, url: str, params=None, **kwargs):
        self.calls.append(("json", url, params))
        route = self.json_routes[url]
        return route(params) if callable(route) else route

    def get_text(self, url: str, params=None, **kwargs):
        self.calls.append(("text", url, params))
        return self.text_routes[url]

    def get_bytes(self, url: str, params=None, **kwargs):
        self.calls.append(("bytes", url, params))
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


class FaceScreener:
    def screen(self, image):
        return PrivacyResult(True, False, "test-face", "test-text", {"face_regions": 1})


class SmoothLabeler:
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


class ExplodingSource:
    name = "broken-source"

    def discover(self, max_hint: int):
        raise RuntimeError("mock discovery failure")
        yield  # pragma: no cover - keeps this a generator


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


class LicenseTests(unittest.TestCase):
    def test_allowlist_and_missing_creator(self) -> None:
        self.assertTrue(
            evaluate_license(
                "CC BY-SA 4.0", "https://creativecommons.org/licenses/by-sa/4.0/", "A"
            ).allowed
        )
        self.assertFalse(
            evaluate_license(
                "CC BY 4.0", "https://creativecommons.org/licenses/by/4.0/", ""
            ).allowed
        )
        self.assertFalse(
            evaluate_license("Public domain", "https://example.test/pd", "A").allowed
        )


class RuleTests(unittest.TestCase):
    def test_descriptor_precedence_and_confidence_cap(self) -> None:
        weak = apply_descriptor_rules(
            {
                "visible_liquid_flow": True,
                "uniform_smooth": True,
                "visible_particles": True,
                "max_particle_mm": 2,
                "visible_pieces": True,
                "typical_piece_mm": 15,
                "confidence": 1.0,
            },
            model_id="test",
        )
        self.assertEqual((weak.level_weak, weak.level_relation), (3, "at_most"))
        self.assertEqual(weak.confidence, 0.95)

    def test_particle_piece_and_fallback_rules(self) -> None:
        particle = apply_descriptor_rules(
            {"visible_particles": True, "max_particle_mm": 4, "confidence": 0.5},
            model_id="test",
        )
        piece = apply_descriptor_rules(
            {"visible_pieces": True, "typical_piece_mm": 15, "confidence": 0.5},
            model_id="test",
        )
        fallback = apply_descriptor_rules({"confidence": 0.1}, model_id="test")
        self.assertEqual((particle.level_weak, piece.level_weak, fallback.level_weak), (5, 6, 7))


class SourceTests(unittest.TestCase):
    def test_wikimedia_api_metadata_becomes_candidate(self) -> None:
        http = FakeHTTP()
        def route(params):
            if params.get("list") == "categorymembers":
                return {"query": {"categorymembers": []}}
            return {
                "query": {
                    "pages": [
                        {
                            "pageid": 7,
                            "title": "File:Congee.jpg",
                            "imageinfo": [
                                {
                                    "mime": "image/jpeg",
                                    "thumburl": "https://upload.wikimedia.org/congee.jpg",
                                    "descriptionurl": (
                                        "https://commons.wikimedia.org/wiki/File:Congee.jpg"
                                    ),
                                    "timestamp": "2025-01-01T00:00:00Z",
                                    "extmetadata": {
                                        "ObjectName": {"value": "Congee"},
                                        "Artist": {"value": "<b>Alice</b>"},
                                        "LicenseShortName": {"value": "CC BY-SA 4.0"},
                                        "LicenseUrl": {
                                            "value": (
                                                "https://creativecommons.org/licenses/by-sa/4.0/"
                                            )
                                        },
                                    },
                                }
                            ],
                        }
                    ]
                }
            }

        http.json_routes[API_URL] = route
        result = list(WikimediaCommonsSource(http).discover(1))
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].creator, "Alice")
        self.assertEqual(http.calls[-1][2]["generator"], "search")

    def test_wikimedia_crawls_requested_categories_to_depth_one(self) -> None:
        http = FakeHTTP()

        def route(params):
            title = params.get("cmtitle")
            if title == "Category:Congee":
                return {
                    "query": {
                        "categorymembers": [
                            {"pageid": 70, "ns": 14, "title": "Category:Rice porridges"}
                        ]
                    }
                }
            if title == "Category:Rice porridges":
                return {
                    "query": {
                        "categorymembers": [
                            {"pageid": 71, "ns": 6, "title": "File:Juk.jpg"}
                        ]
                    }
                }
            if params.get("titles") == "File:Juk.jpg":
                return {
                    "query": {
                        "pages": [
                            {
                                "pageid": 71,
                                "title": "File:Juk.jpg",
                                "imageinfo": [
                                    {
                                        "thumburl": "https://upload.wikimedia.org/juk.jpg",
                                        "descriptionurl": (
                                            "https://commons.wikimedia.org/wiki/File:Juk.jpg"
                                        ),
                                        "extmetadata": {
                                            "ObjectName": {"value": "Juk"},
                                            "Artist": {"value": "Alice"},
                                            "LicenseShortName": {"value": "CC BY 4.0"},
                                            "LicenseUrl": {
                                                "value": (
                                                    "https://creativecommons.org/licenses/by/4.0/"
                                                )
                                            },
                                        },
                                    }
                                ],
                            }
                        ]
                    }
                }
            return {"query": {"categorymembers": []}}

        http.json_routes[API_URL] = route
        result = list(WikimediaCommonsSource(http).discover(1))
        self.assertEqual(CATEGORIES, (
            "Congee",
            "Porridges",
            "Puréed food",
            "Puréed vegetables",
            "Baby food",
            "Tofu dishes",
            "Steamed eggs",
            "Mashed potato dishes",
        ))
        self.assertEqual([item.source_id for item in result], ["71"])
        self.assertEqual(result[0].download_url, "https://upload.wikimedia.org/juk.jpg")
        self.assertEqual(result[0].landing_url, "https://commons.wikimedia.org/wiki/File:Juk.jpg")
        self.assertEqual(result[0].license_name, "CC BY 4.0")
        self.assertEqual(
            result[0].license_url,
            "https://creativecommons.org/licenses/by/4.0/",
        )
        self.assertEqual((result[0].creator, result[0].title), ("Alice", "Juk"))
        category_calls = [call[2] for call in http.calls]
        self.assertEqual(category_calls[0]["cmtype"], "file|subcat")
        self.assertEqual(category_calls[1]["cmtitle"], "Category:Rice porridges")
        self.assertEqual(category_calls[1]["cmtype"], "file")
        self.assertEqual(category_calls[2]["prop"], "imageinfo")
        self.assertEqual(category_calls[2]["iiprop"], "url|extmetadata")

    def test_wikimedia_follows_real_category_continue_and_strips_image_url_query(self) -> None:
        http = FakeHTTP()
        first_page_titles = [f"File:Missing-{index}.jpg" for index in range(50)]
        unicode_title = "File:Chiu Chow oyster congee 潮州粥.jpg"
        cmcontinue = "file|434849552043484f57204f595354455220434f4e4745452e4a5047|12345"

        def route(params):
            if params.get("cmtitle") == "Category:Congee":
                if "cmcontinue" not in params:
                    return {
                        "continue": {"cmcontinue": cmcontinue, "continue": "-||"},
                        "query": {
                            "categorymembers": [
                                {"pageid": index + 1, "ns": 6, "title": title}
                                for index, title in enumerate(first_page_titles)
                            ]
                        },
                    }
                self.assertEqual(params["cmcontinue"], cmcontinue)
                self.assertEqual(params["continue"], "-||")
                return {
                    "query": {
                        "categorymembers": [
                            {"pageid": 51, "ns": 6, "title": unicode_title}
                        ]
                    }
                }
            if "titles" in params:
                requested = params["titles"].split("|")
                if requested == first_page_titles:
                    return {
                        "query": {
                            "pages": [
                                {"pageid": index + 1, "title": title, "missing": True}
                                for index, title in enumerate(requested)
                            ]
                        }
                    }
                return {
                    "query": {
                        "pages": [
                            {
                                "pageid": 51,
                                "title": unicode_title,
                                "imageinfo": [
                                    {
                                        "url": (
                                            "https://upload.wikimedia.org/wikipedia/commons/9/9b/"
                                            "Chiu_Chow_oyster_congee.jpg?utm_source="
                                            "commons.wikimedia.org&utm_campaign=imageinfo&"
                                            "utm_content=original"
                                        ),
                                        "descriptionurl": (
                                            "https://commons.wikimedia.org/wiki/"
                                            "File:Chiu_Chow_oyster_congee.jpg"
                                        ),
                                        "extmetadata": {
                                            "ObjectName": {"value": "Chiu Chow oyster congee"},
                                            "Artist": {"value": "Bob"},
                                            "LicenseShortName": {"value": "CC BY-SA 2.0"},
                                            "LicenseUrl": {
                                                "value": (
                                                    "https://creativecommons.org/licenses/by-sa/2.0/"
                                                )
                                            },
                                        },
                                    }
                                ],
                            }
                        ]
                    }
                }
            return {"query": {"categorymembers": []}}

        http.json_routes[API_URL] = route
        source = WikimediaCommonsSource(http)
        result = list(source.discover(1))
        self.assertEqual([item.source_id for item in result], ["51"])
        self.assertEqual(
            result[0].download_url,
            "https://upload.wikimedia.org/wikipedia/commons/9/9b/"
            "Chiu_Chow_oyster_congee.jpg",
        )
        self.assertEqual(result[0].license_name, "CC BY-SA 2.0")
        resolution_calls = [
            call[2] for call in http.calls if call[0] == "json" and "titles" in call[2]
        ]
        self.assertEqual([len(call["titles"].split("|")) for call in resolution_calls], [50, 1])
        self.assertEqual(source.stats["category_pages_succeeded"], 2)

    def test_wikimedia_logs_bad_continuation_page_and_continues_next_category(self) -> None:
        http = FakeHTTP()
        failed_token = "file|BROKEN|456"

        def route(params):
            title = params.get("cmtitle")
            if title == "Category:Congee" and "cmcontinue" not in params:
                return {
                    "continue": {"cmcontinue": failed_token, "continue": "-||"},
                    "query": {"categorymembers": []},
                }
            if title == "Category:Congee":
                raise RuntimeError("mock bad continuation page")
            if title == "Category:Porridges":
                return {
                    "query": {
                        "categorymembers": [
                            {"pageid": 88, "ns": 6, "title": "File:Recovered.jpg"}
                        ]
                    }
                }
            if params.get("titles") == "File:Recovered.jpg":
                return {
                    "query": {
                        "pages": [
                            {
                                "pageid": 88,
                                "title": "File:Recovered.jpg",
                                "imageinfo": [
                                    {
                                        "url": "https://upload.wikimedia.org/recovered.jpg",
                                        "descriptionurl": (
                                            "https://commons.wikimedia.org/wiki/File:Recovered.jpg"
                                        ),
                                        "extmetadata": {
                                            "ObjectName": {"value": "Recovered porridge"},
                                            "Artist": {"value": "Alice"},
                                            "LicenseShortName": {"value": "CC BY 4.0"},
                                            "LicenseUrl": {
                                                "value": (
                                                    "https://creativecommons.org/licenses/by/4.0/"
                                                )
                                            },
                                        },
                                    }
                                ],
                            }
                        ]
                    }
                }
            return {"query": {"categorymembers": []}}

        http.json_routes[API_URL] = route
        source = WikimediaCommonsSource(http)
        result = list(source.discover(1))
        self.assertEqual([item.source_id for item in result], ["88"])
        self.assertEqual(len(source.stats["errors"]), 1)
        self.assertEqual(source.stats["errors"][0]["stage"], "categorymembers")
        self.assertEqual(source.stats["errors"][0]["cmcontinue"], failed_token)

    def test_wikimedia_skips_failed_imageinfo_batch_and_resolves_later_batch(self) -> None:
        http = FakeHTTP()
        titles = [f"File:Broken-{index}.jpg" for index in range(50)] + ["File:Valid.jpg"]

        def route(params):
            if params.get("cmtitle") == "Category:Congee":
                return {
                    "query": {
                        "categorymembers": [
                            {"pageid": index + 1, "ns": 6, "title": title}
                            for index, title in enumerate(titles)
                        ]
                    }
                }
            if "titles" in params and len(params["titles"].split("|")) == 50:
                raise RuntimeError("mock bad imageinfo batch")
            if params.get("titles") == "File:Valid.jpg":
                return {
                    "query": {
                        "pages": [
                            {
                                "pageid": 51,
                                "title": "File:Valid.jpg",
                                "imageinfo": [
                                    {
                                        "url": "https://upload.wikimedia.org/valid.jpg",
                                        "descriptionurl": (
                                            "https://commons.wikimedia.org/wiki/File:Valid.jpg"
                                        ),
                                        "extmetadata": {
                                            "ObjectName": {"value": "Valid puree"},
                                            "Artist": {"value": "Bob"},
                                            "LicenseShortName": {"value": "CC BY-SA 4.0"},
                                            "LicenseUrl": {
                                                "value": (
                                                    "https://creativecommons.org/licenses/by-sa/4.0/"
                                                )
                                            },
                                        },
                                    }
                                ],
                            }
                        ]
                    }
                }
            return {"query": {"categorymembers": []}}

        http.json_routes[API_URL] = route
        source = WikimediaCommonsSource(http)
        result = list(source.discover(1))
        self.assertEqual([item.source_id for item in result], ["51"])
        self.assertEqual(source.stats["imageinfo_batches_requested"], 2)
        self.assertEqual(source.stats["imageinfo_batches_succeeded"], 1)
        self.assertEqual(source.stats["errors"][0]["stage"], "imageinfo")

    def test_wikimedia_requeues_failed_imageinfo_batch_once_at_end(self) -> None:
        http = FakeHTTP()
        imageinfo_attempts = 0

        def route(params):
            nonlocal imageinfo_attempts
            if params.get("cmtitle") == "Category:Congee":
                return {
                    "query": {
                        "categorymembers": [
                            {"pageid": 91, "ns": 6, "title": "File:Retry.jpg"}
                        ]
                    }
                }
            if params.get("titles") == "File:Retry.jpg":
                imageinfo_attempts += 1
                if imageinfo_attempts == 1:
                    raise RuntimeError("mock exhausted HTTP retries")
                return {
                    "query": {
                        "pages": [
                            {
                                "pageid": 91,
                                "title": "File:Retry.jpg",
                                "imageinfo": [
                                    {
                                        "url": "https://upload.wikimedia.org/retry.jpg",
                                        "descriptionurl": (
                                            "https://commons.wikimedia.org/wiki/File:Retry.jpg"
                                        ),
                                        "extmetadata": {
                                            "ObjectName": {"value": "Retried puree"},
                                            "Artist": {"value": "Alice"},
                                            "LicenseShortName": {"value": "CC BY 4.0"},
                                            "LicenseUrl": {
                                                "value": (
                                                    "https://creativecommons.org/licenses/by/4.0/"
                                                )
                                            },
                                        },
                                    }
                                ],
                            }
                        ]
                    }
                }
            return {"query": {"categorymembers": []}}

        http.json_routes[API_URL] = route
        source = WikimediaCommonsSource(http, search_terms=())
        result = list(source.discover(1))

        self.assertEqual([item.source_id for item in result], ["91"])
        self.assertEqual(imageinfo_attempts, 2)
        self.assertEqual(source.stats["imageinfo_batches_requeued"], 1)
        self.assertEqual(source.stats["imageinfo_requeue_batches_succeeded"], 1)

    def test_wikimedia_never_emits_candidate_without_download_url(self) -> None:
        http = FakeHTTP()

        def route(params):
            if params.get("cmtitle") == "Category:Congee":
                return {
                    "query": {
                        "categorymembers": [
                            {"pageid": 92, "ns": 6, "title": "File:No-URL.jpg"}
                        ]
                    }
                }
            if params.get("titles") == "File:No-URL.jpg":
                return {
                    "query": {
                        "pages": [
                            {
                                "pageid": 92,
                                "title": "File:No-URL.jpg",
                                "imageinfo": [
                                    {
                                        "url": None,
                                        "thumburl": None,
                                        "descriptionurl": (
                                            "https://commons.wikimedia.org/wiki/File:No-URL.jpg"
                                        ),
                                        "extmetadata": {
                                            "Artist": {"value": "Alice"},
                                            "LicenseShortName": {"value": "CC BY 4.0"},
                                            "LicenseUrl": {
                                                "value": (
                                                    "https://creativecommons.org/licenses/by/4.0/"
                                                )
                                            },
                                        },
                                    }
                                ],
                            }
                        ]
                    }
                }
            return {"query": {"categorymembers": []}}

        http.json_routes[API_URL] = route
        source = WikimediaCommonsSource(http, search_terms=())
        result = list(source.discover(5))

        self.assertEqual(result, [])
        self.assertEqual(source.stats["candidates_skipped_no_download_url"], 1)

    def test_open_images_joins_human_labels_and_per_file_metadata(self) -> None:
        http = FakeHTTP()
        http.text_routes[CLASS_URL] = "/m/food,Food\n/m/cat,Cat\n"
        http.text_routes[LABEL_URL] = (
            "ImageID,Source,LabelName,Confidence\nabc,verification,/m/food,1\n"
        )
        http.text_routes[METADATA_URL] = (
            "ImageID,Subset,OriginalURL,OriginalLandingURL,License,AuthorProfileURL,Author,Title,"
            "OriginalSize,OriginalMD5,Thumbnail300KURL,Rotation\n"
            "abc,validation,https://farm.staticflickr.com/a.jpg,https://flickr.test/a,"
            "https://creativecommons.org/licenses/by/2.0/,https://flickr.test/u,Alice,Porridge,"
            "10,md5,https://c1.staticflickr.com/thumb.jpg,0\n"
        )
        result = list(OpenImagesSource(http).discover(2))
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].license_name, "CC BY 2.0")
        self.assertEqual(result[0].verification_status, "unverified")
        self.assertIn("verify", result[0].extra["license_verification_caveat"])

    def test_openverse_discovery_records_foreign_licence_landing_page(self) -> None:
        http = FakeHTTP()
        http.json_routes[OPENVERSE_API_URL] = {
            "results": [
                {
                    "id": "ov-1",
                    "title": "Pumpkin puree",
                    "creator": "Alice",
                    "creator_url": "https://creator.example/alice",
                    "url": "https://images.example/pumpkin.jpg",
                    "foreign_landing_url": "https://source.example/pumpkin",
                    "license": "by",
                    "license_version": "4.0",
                    "license_url": "https://creativecommons.org/licenses/by/4.0/",
                    "source": "flickr",
                },
                {
                    "id": "ov-nc",
                    "title": "Not eligible",
                    "creator": "Bob",
                    "url": "https://images.example/nc.jpg",
                    "foreign_landing_url": "https://source.example/nc",
                    "license": "by-nc",
                    "license_version": "4.0",
                    "license_url": "https://creativecommons.org/licenses/by-nc/4.0/",
                },
            ],
            "page_count": 1,
        }
        result = list(OpenverseSource(http).discover(5))
        self.assertEqual(len(result), 1)
        item = result[0]
        self.assertEqual(item.license_name, "CC BY 4.0")
        self.assertEqual(item.landing_url, "https://source.example/pumpkin")
        self.assertEqual(item.verification_status, "unverified")
        self.assertEqual(item.extra["foreign_landing_url"], item.landing_url)
        self.assertIn(item.landing_url, item.attribution)
        params = http.calls[0][2]
        self.assertEqual(params["license"], "cc0,by")
        self.assertEqual(params["q"], QUERIES[0])
        self.assertEqual(
            QUERIES,
            (
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
            ),
        )

    def test_nutrition5k_lists_only_rgb_objects(self) -> None:
        http = FakeHTTP()
        http.json_routes[LIST_URL] = {
            "items": [
                {"name": "nutrition5k_dataset/imagery/realsense_overhead/dish_1/rgb.png"},
                {"name": "nutrition5k_dataset/imagery/realsense_overhead/dish_1/depth_raw.png"},
            ]
        }
        result = list(Nutrition5kSource(http).discover(2))
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].license_name, "CC BY 4.0")


class PipelineTests(unittest.TestCase):
    def test_pipeline_writes_valid_event_attribution_and_dedup_rejection(self) -> None:
        http = FakeHTTP()
        http.stats_snapshot = lambda: {
            "upload.wikimedia.org": {"requests": 2, "429s": 1, "retries": 1}
        }
        first_url = "https://upload.wikimedia.org/a.png"
        second_url = "https://upload.wikimedia.org/b.png"
        http.byte_routes[first_url] = image_bytes()
        http.byte_routes[second_url] = image_bytes()
        source = FakeSource([candidate("a", first_url), candidate("b", second_url)])
        harvester = Harvester(
            http=http,
            sources=[source],
            labeler=SmoothLabeler(),
            screener=ClearScreener(),
            candidate_multiplier=2,
            provider_cap=1.0,
        )
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "dataset"
            summary = harvester.run(maximum=2, output=output)
            events = read_jsonl(output / "events.jsonl")
            attribution = read_jsonl(output / "attribution.jsonl")
            rejected = read_jsonl(output / "rejected.jsonl")
            self.assertEqual((summary.accepted, summary.rejected), (1, 1))
            self.assertEqual(summary.http_stats["upload.wikimedia.org"]["429s"], 1)
            self.assertEqual(events[0]["source"], "real-weak")
            self.assertIsNone(events[0]["level_adjudicated"])
            self.assertEqual(events[0]["level_weak"], 4)
            self.assertEqual(len(attribution), 1)
            self.assertEqual(rejected[0]["reason"], "exact_duplicate")
            errors = validate_realweak_event(
                events[0], ROOT / "dataschema" / "schema" / "event.schema.json"
            )
            self.assertEqual(errors, [])
            self.assertTrue((output / events[0]["media_files"][0]["path"]).is_file())

    def test_provider_cap_rejects_further_candidates_from_provider(self) -> None:
        http = FakeHTTP()
        first_url = "https://upload.wikimedia.org/cap-a.png"
        second_url = "https://upload.wikimedia.org/cap-b.png"
        http.byte_routes[first_url] = image_bytes()
        http.byte_routes[second_url] = image_bytes((20, 180, 90))
        harvester = Harvester(
            http=http,
            sources=[FakeSource([candidate("cap-a", first_url), candidate("cap-b", second_url)])],
            labeler=SmoothLabeler(),
            screener=ClearScreener(),
            provider_cap=0.4,
        )
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "dataset"
            summary = harvester.run(maximum=2, output=output)
            self.assertEqual(summary.accepted, 1)
            rejection = read_jsonl(output / "rejected.jsonl")[0]
            self.assertEqual(rejection["reason"], "provider_cap_reached")
            self.assertEqual(rejection["details"]["provider_limit"], 1)

    def test_open_images_unverified_status_reaches_both_manifests(self) -> None:
        http = FakeHTTP()
        url = "https://storage.googleapis.com/openimages/test.png"
        open_images_candidate = Candidate(
            provider="open-images",
            source_id="oi-unverified",
            download_url=url,
            landing_url="https://source.example/open-image",
            license_name="CC BY 2.0",
            license_url="https://creativecommons.org/licenses/by/2.0/",
            creator="Alice",
            title="Open image",
            attribution="Open image — Alice; CC BY 2.0; https://source.example/open-image",
            verification_status="unverified",
        )
        http.byte_routes[url] = image_bytes()
        source = FakeSource([open_images_candidate])
        source.name = "open-images"
        harvester = Harvester(
            http=http,
            sources=[source],
            labeler=SmoothLabeler(),
            screener=ClearScreener(),
            provider_cap=1.0,
        )
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "dataset"
            summary = harvester.run(maximum=1, output=output)
            self.assertEqual(summary.accepted, 1)
            self.assertEqual(
                read_jsonl(output / "events.jsonl")[0]["provenance"]["verification_status"],
                "unverified",
            )
            self.assertEqual(
                read_jsonl(output / "attribution.jsonl")[0]["verification_status"],
                "unverified",
            )

    def test_face_is_rejected_before_labelling(self) -> None:
        http = FakeHTTP()
        url = "https://upload.wikimedia.org/face.png"
        http.byte_routes[url] = image_bytes()
        harvester = Harvester(
            http=http,
            sources=[FakeSource([candidate("face", url)])],
            labeler=SmoothLabeler(),
            screener=FaceScreener(),
        )
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "dataset"
            summary = harvester.run(maximum=1, output=output)
            self.assertEqual((summary.accepted, summary.rejected), (0, 1))
            self.assertFalse((output / "events.jsonl").exists())
            self.assertEqual(read_jsonl(output / "rejected.jsonl")[0]["reason"], "face_detected")

    def test_source_discovery_error_is_audited_without_crashing(self) -> None:
        harvester = Harvester(
            http=FakeHTTP(),
            sources=[ExplodingSource()],
            labeler=SmoothLabeler(),
            screener=ClearScreener(),
        )
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "dataset"
            summary = harvester.run(maximum=1, output=output)
            self.assertEqual((summary.accepted, summary.rejected), (0, 1))
            rejection = read_jsonl(output / "rejected.jsonl")[0]
            self.assertEqual(rejection["reason"], "source_discovery_error")
            self.assertEqual(rejection["provider"], "broken-source")

    def test_registry_records_blocked_food101_and_foodsense(self) -> None:
        registry = json.loads((ROOT / "harvest" / "source_licenses.json").read_text())
        statuses = {item["id"]: item["status"] for item in registry["sources"]}
        self.assertTrue(statuses["food101"].startswith("blocked"))
        self.assertTrue(statuses["sababishraq/foodsense-dataset"].startswith("blocked"))

    def test_cli_registers_openverse_and_defaults_provider_cap(self) -> None:
        args = _parser().parse_args(["--out", "/tmp/mock-output"])
        self.assertIn("openverse", SOURCE_NAMES)
        self.assertEqual(args.provider_cap, 0.4)
        self.assertEqual(args.http_min_interval, 1.0)

    def test_cli_rejects_negative_http_minimum_interval(self) -> None:
        from harvest.run import main

        self.assertEqual(
            main(["--out", "/tmp/mock-output", "--http-min-interval", "-0.1"]),
            2,
        )


if __name__ == "__main__":
    unittest.main()
