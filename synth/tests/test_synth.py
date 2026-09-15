from __future__ import annotations

import json
import hashlib
import random
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image


SYNTH_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = SYNTH_ROOT.parent
sys.path.insert(0, str(SYNTH_ROOT))
sys.path.insert(0, str(REPO_ROOT))

from dataschema.tools.validate_dataset import _validate_schema_value  # noqa: E402
from synth.generate import GenerationConfig, SyntheticGenerator  # noqa: E402
from synth.judge import JudgeDecision, JudgeThresholds, decision_from_response  # noqa: E402
from synth.prompt_bank import (  # noqa: E402
    ALL_DISHES,
    CHINESE_CANTONESE_DISHES,
    LEVEL_DESCRIPTORS,
    WESTERN_DISHES,
    build_prompt,
    crossed_dishes,
)


class StubPipeline:
    model_id = "stub/no-download"

    def __init__(self) -> None:
        self.calls = 0

    def generate(self, prompts, negative_prompts, seeds, *, width, height, short_prompts=None):
        self.calls += 1
        self.assertions = (len(prompts), len(negative_prompts), len(seeds), width, height)
        return [Image.new("RGB", (width, height), (seed % 255, 40, 80)) for seed in seeds]


class RejectFirstJudge:
    def __init__(self) -> None:
        self.calls = 0

    def judge(self, image_path, spec):
        self.calls += 1
        accepted = self.calls != 1
        return JudgeDecision(
            accepted=accepted,
            backend="stub-judge",
            descriptor_match_score=0.9 if accepted else 0.1,
            visual_quality_score=0.9,
            cue_consistency_score=0.9 if spec.cue else None,
            observed_level=spec.level,
            contains_face=False,
            contains_text=False,
            reasons=("stub decision",),
        )


class PromptBankTests(unittest.TestCase):
    def test_bank_has_requested_coverage_and_full_cross(self) -> None:
        self.assertGreaterEqual(len(CHINESE_CANTONESE_DISHES), 60)
        self.assertGreaterEqual(len(WESTERN_DISHES), 20)
        self.assertEqual(set(LEVEL_DESCRIPTORS), {3, 4, 5, 6, 7})
        self.assertEqual(len(list(crossed_dishes())), 5 * len(ALL_DISHES))

    def test_prompt_is_deterministic_and_contains_safety_boundaries(self) -> None:
        one = build_prompt(4, random.Random(123), cue_probability=1.0)
        two = build_prompt(4, random.Random(123), cue_probability=1.0)
        self.assertEqual(one, two)
        self.assertIn("IDDSI Level 4", one.prompt)
        self.assertIn("not proof of a test result", one.prompt)
        self.assertIn("faces", one.negative_prompt)
        self.assertTrue(one.cue)

    def test_l7_bank_covers_easy_to_chew_and_regular_variants(self) -> None:
        variants = {build_prompt(7, random.Random(seed)).level_variant for seed in range(20)}
        self.assertEqual(variants, {"Easy to Chew", "Regular"})


class GenerationTests(unittest.TestCase):
    def test_stub_pipeline_produces_images_manifest_and_rejection_log(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary) / "dataset"
            pipeline = StubPipeline()
            judge = RejectFirstJudge()
            summary = SyntheticGenerator(pipeline, judge).run(
                GenerationConfig(
                    level=5,
                    n=3,
                    out=out,
                    batch_size=2,
                    seed=7,
                    width=32,
                    height=32,
                    max_attempts=8,
                )
            )
            self.assertEqual(summary.accepted, 3)
            self.assertEqual(summary.rejected, 1)
            rows = [json.loads(line) for line in (out / "events.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(rows), 3)
            schema = json.loads((REPO_ROOT / "dataschema" / "schema" / "event.schema.json").read_text())
            for index, row in enumerate(rows, start=1):
                errors = _validate_schema_value(row, schema, schema, index, "$")
                self.assertEqual(errors, [])
                self.assertEqual(row["source"], "synthetic")
                self.assertEqual(row["level_declared"], 5)
                self.assertIsNone(row["level_tested_RD"])
                self.assertIsNone(row["level_tested_SLP"])
                self.assertIsNone(row["level_adjudicated"])
                media_path = out / row["media_files"][0]["path"]
                self.assertTrue(media_path.is_file())
                self.assertEqual(row["media_files"][0]["bytes"], media_path.stat().st_size)
                self.assertEqual(row["media_files"][0]["sha256"], hashlib.sha256(media_path.read_bytes()).hexdigest())
            logs = [json.loads(line) for line in (out / "generation_log.jsonl").read_text().splitlines()]
            self.assertEqual(len(logs), 4)
            self.assertTrue(all(log["source_tag"] == "synthetic" for log in logs))
            self.assertTrue(all(isinstance(log["seed"], int) for log in logs))
            self.assertEqual(sum(not log["accepted"] for log in logs), 1)

    def test_synthetic_external_test_is_forbidden(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ValueError, "never external_test"):
                GenerationConfig(level=4, n=1, out=Path(temporary), split="external_test")


class JudgeParsingTests(unittest.TestCase):
    def test_thresholds_reject_level_mismatch(self) -> None:
        spec = build_prompt(6, random.Random(1), cue_probability=0.0)
        response = json.dumps(
            {
                "descriptor_match_score": 0.95,
                "visual_quality_score": 0.9,
                "cue_consistency_score": None,
                "observed_level": 5,
                "contains_face": False,
                "contains_text": False,
                "reasons": ["looks more finely divided"],
            }
        )
        decision = decision_from_response(response, spec, JudgeThresholds(), backend="stub")
        self.assertFalse(decision.accepted)
        self.assertTrue(any("did not match" in reason for reason in decision.reasons))

    def test_invalid_response_is_logged_as_error_rejection(self) -> None:
        spec = build_prompt(3, random.Random(1))
        decision = decision_from_response("not-json", spec, JudgeThresholds(), backend="stub")
        self.assertFalse(decision.accepted)
        self.assertEqual(decision.status, "error")
        self.assertIsNone(decision.descriptor_match_score)


if __name__ == "__main__":
    unittest.main()
