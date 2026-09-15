"""Judge parser tests against recorded Qwen3-VL responses from the L4 pilot.

Fixtures in ``fixtures/judge_pilot_L4_responses.json`` are real judge
verdicts captured from the 2026-09-01 DGX Spark pilot run
(``~/iddsi/data/synth_pilot_L4/generation_log.jsonl`` on Spark, run
``20260901T151002Z_215c4edb``, backend
``transformers:Qwen/Qwen3-VL-2B-Instruct``).  The pilot accepted 0 of 72
attempts, so all five fixtures are rejections: three ``scored`` rejections
(sub-threshold scores, one with an observed_level mismatch, one with a null
observed_level, one carrying a staged fork cue) and two ``error`` rejections
whose raw responses are truncated JSON — the model repeated a sentence until
``max_new_tokens=400`` cut it off mid-object, so no closing ``}`` exists.

Each test replays the recorded ``raw_response`` through
``decision_from_response`` — the parsing/decision logic only, never a live
model — and asserts the parser reproduces the verdict fields the pilot
recorded.
"""

from __future__ import annotations

import json
import random
import sys
import unittest
from pathlib import Path


SYNTH_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SYNTH_ROOT))

from synth.judge import JudgeThresholds, decision_from_response  # noqa: E402
from synth.prompt_bank import build_prompt  # noqa: E402


FIXTURES = json.loads(
    (Path(__file__).parent / "fixtures" / "judge_pilot_L4_responses.json").read_text(encoding="utf-8")
)

PILOT_BACKEND = "transformers:Qwen/Qwen3-VL-2B-Instruct"


def _spec_for(fixture: dict):
    """Rebuild a PromptSpec matching the fixture's level and staged cue.

    The parser only reads ``spec.level`` and ``spec.cue``; the dish and
    styling fields are irrelevant to the decision logic, so any seed works
    as long as the cue presence matches the recorded attempt.
    """
    want_cue = fixture["cue"] is not None
    for seed in range(500):
        spec = build_prompt(
            fixture["level_declared"],
            random.Random(seed),
            cue_probability=1.0 if want_cue else 0.0,
        )
        if (spec.cue is not None) == want_cue:
            return spec
    raise AssertionError("could not build a spec with the required cue presence")


class PilotFixtureTests(unittest.TestCase):
    def test_fixture_file_shape(self) -> None:
        self.assertEqual(len(FIXTURES), 5)
        statuses = {f["judge"]["status"] for f in FIXTURES}
        self.assertEqual(statuses, {"scored", "error"})
        # The pilot accepted nothing; these fixtures document that reality.
        self.assertTrue(all(f["judge"]["accepted"] is False for f in FIXTURES))
        self.assertTrue(all(f["judge"]["backend"] == PILOT_BACKEND for f in FIXTURES))

    def test_scored_rejections_reproduce_recorded_verdicts(self) -> None:
        for fixture in FIXTURES:
            if fixture["judge"]["status"] != "scored":
                continue
            recorded = fixture["judge"]
            spec = _spec_for(fixture)
            decision = decision_from_response(
                recorded["raw_response"], spec, JudgeThresholds(), backend=recorded["backend"]
            )
            with self.subTest(attempt=fixture["attempt"]):
                self.assertEqual(decision.status, "scored")
                self.assertFalse(decision.accepted)
                self.assertEqual(decision.descriptor_match_score, recorded["descriptor_match_score"])
                self.assertEqual(decision.visual_quality_score, recorded["visual_quality_score"])
                self.assertEqual(decision.cue_consistency_score, recorded["cue_consistency_score"])
                self.assertEqual(decision.observed_level, recorded["observed_level"])
                self.assertEqual(decision.contains_face, recorded["contains_face"])
                self.assertEqual(decision.contains_text, recorded["contains_text"])
                # The model's own reasons are preserved ahead of threshold failures.
                for reason in recorded["reasons"]:
                    self.assertIn(reason, decision.reasons)

    def test_scored_fixture_with_level_mismatch_names_the_mismatch(self) -> None:
        fixture = next(
            f for f in FIXTURES if f["judge"]["status"] == "scored" and f["judge"]["observed_level"] == 3
        )
        spec = _spec_for(fixture)
        decision = decision_from_response(
            fixture["judge"]["raw_response"], spec, JudgeThresholds(), backend=fixture["judge"]["backend"]
        )
        self.assertFalse(decision.accepted)
        self.assertTrue(any("did not match" in reason for reason in decision.reasons))
        self.assertTrue(any("below threshold" in reason for reason in decision.reasons))

    def test_error_fixtures_are_truncated_json_and_reject_as_errors(self) -> None:
        error_fixtures = [f for f in FIXTURES if f["judge"]["status"] == "error"]
        self.assertEqual(len(error_fixtures), 2)
        for fixture in error_fixtures:
            raw = fixture["judge"]["raw_response"]
            with self.subTest(attempt=fixture["attempt"]):
                # Document the real failure mode: the pilot's raw response is
                # an unterminated JSON object (token budget exhausted), so the
                # parser must reject with status="error", never invent scores.
                self.assertIn("{", raw)
                self.assertNotIn("}", raw)
                self.assertEqual(
                    fixture["judge"]["reasons"],
                    ["Invalid judge response: judge response did not contain a JSON object"],
                )
                spec = _spec_for(fixture)
                decision = decision_from_response(
                    raw, spec, JudgeThresholds(), backend=fixture["judge"]["backend"]
                )
                self.assertFalse(decision.accepted)
                self.assertEqual(decision.status, "error")
                self.assertIsNone(decision.descriptor_match_score)
                self.assertIsNone(decision.visual_quality_score)
                self.assertIsNone(decision.observed_level)
                self.assertEqual(decision.reasons, tuple(fixture["judge"]["reasons"]))

    def test_markdown_fenced_json_is_accepted_by_parser(self) -> None:
        # Every scored pilot response was wrapped in a ```json fence despite
        # the prompt saying "No markdown"; the parser's first-{ to last-}
        # extraction tolerates it.  Pin that behaviour against a real fixture.
        fixture = next(f for f in FIXTURES if f["judge"]["status"] == "scored")
        self.assertTrue(fixture["judge"]["raw_response"].lstrip().startswith("```json"))
        spec = _spec_for(fixture)
        decision = decision_from_response(
            fixture["judge"]["raw_response"], spec, JudgeThresholds(), backend=fixture["judge"]["backend"]
        )
        self.assertEqual(decision.status, "scored")


if __name__ == "__main__":
    unittest.main()
