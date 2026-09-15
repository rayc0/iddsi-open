"""Tests for per-level judge checklists and negative signatures (W32).

Verifies that ``synth.judge_prompts`` provides, for every IDDSI level 3-7:

- a non-empty visual-signature checklist (EN and 繁中) grounded in the
  official IDDSI descriptor concepts,
- a negative signature (EN and 繁中) that includes the failure modes audited
  in the 0/24 pilot (whole grains, garnish, separated liquid, devices in the
  food, camera-facet nouns rendered as objects),
- a deterministic ``build_judge_prompt`` that composes the checklist and the
  negatives while keeping the JSON response contract of ``judge._judge_prompt``.

Pure data/function tests - no network, no models.  ``judge.py`` itself is
deliberately not imported for editing; only its prompt shape is matched.
"""

from __future__ import annotations

import random
import sys
import unittest
from pathlib import Path


SYNTH_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SYNTH_ROOT))

from synth.judge_prompts import (  # noqa: E402
    COMMON_NEGATIVE_SIGNATURES,
    COMMON_NEGATIVE_SIGNATURES_ZH,
    LEVELS,
    LEVEL_SIGNATURES,
    L7_REGULAR_CHECKLIST,
    build_judge_prompt,
    checklist_items,
    judge_checklist,
    negative_signature,
)
from synth.prompt_bank import build_prompt  # noqa: E402


# Official-descriptor concepts each level's checklist must reference.
EXPECTED_CHECKLIST_CONCEPTS: dict[int, tuple[str, ...]] = {
    3: ("smooth", "lump-free"),
    4: ("holds its own shape", "single soft mass"),
    5: ("4 mm", "mash"),
    6: ("15 mm", "break apart"),
    7: ("easy-to-chew", "natural food shapes"),
}


class JudgeChecklistDataTests(unittest.TestCase):
    def test_all_five_levels_present(self) -> None:
        self.assertEqual(sorted(LEVEL_SIGNATURES), [3, 4, 5, 6, 7])
        self.assertEqual(LEVELS, (3, 4, 5, 6, 7))

    def test_every_level_has_non_empty_checklist_en_and_zh(self) -> None:
        for level in LEVELS:
            with self.subTest(level=level):
                self.assertGreaterEqual(len(judge_checklist(level).checklist), 3)
                self.assertGreaterEqual(len(judge_checklist(level).checklist_zh), 3)
                for item in judge_checklist(level).checklist:
                    self.assertTrue(item.strip(), "blank EN checklist item")
                for item in judge_checklist(level).checklist_zh:
                    self.assertTrue(item.strip(), "blank ZH checklist item")

    def test_checklist_references_official_descriptor_concepts(self) -> None:
        for level, concepts in EXPECTED_CHECKLIST_CONCEPTS.items():
            text = " ".join(checklist_items(level)).lower()
            for concept in concepts:
                with self.subTest(level=level, concept=concept):
                    self.assertIn(concept.lower(), text)

    def test_checklist_matches_prompt_bank_level_names(self) -> None:
        from synth.prompt_bank import LEVEL_DESCRIPTORS

        for level in LEVELS:
            self.assertEqual(judge_checklist(level).name, LEVEL_DESCRIPTORS[level].name)

    def test_checklist_items_zh_toggle(self) -> None:
        for level in LEVELS:
            with self.subTest(level=level):
                self.assertEqual(
                    checklist_items(level, zh=True),
                    judge_checklist(level).checklist_zh,
                )
                self.assertEqual(checklist_items(level), judge_checklist(level).checklist)

    def test_every_level_has_negative_signature(self) -> None:
        for level in LEVELS:
            with self.subTest(level=level):
                signature = judge_checklist(level)
                self.assertGreaterEqual(len(signature.negative_signatures), 3)
                self.assertGreaterEqual(len(signature.negative_signatures_zh), 3)
                merged = negative_signature(level)
                self.assertGreaterEqual(
                    len(merged), len(COMMON_NEGATIVE_SIGNATURES) + len(signature.negative_signatures)
                )
                for item in COMMON_NEGATIVE_SIGNATURES:
                    self.assertIn(item, merged)
                for item in signature.negative_signatures:
                    self.assertIn(item, merged)
                merged_zh = negative_signature(level, zh=True)
                for item in signature.negative_signatures_zh:
                    self.assertIn(item, merged_zh)
                for item in COMMON_NEGATIVE_SIGNATURES_ZH:
                    self.assertIn(item, merged_zh)

    def test_negative_signature_includes_audited_failure_modes(self) -> None:
        # The 0/24 pilot audited these recurring failures; each level's
        # merged negative signature must carry the common ones (garnish,
        # devices in the food, camera-facet noun literalism) and the
        # grain / separated-liquid modes must appear on L3-L5.
        for level in LEVELS:
            merged = " ".join(negative_signature(level)).lower()
            with self.subTest(level=level):
                self.assertIn("garnish", merged)
                self.assertIn("sesame", merged)
                self.assertIn("scallion", merged)
                self.assertIn("device", merged)
                self.assertIn("camera-facet", merged)
        for level in (3, 4, 5):
            merged = " ".join(negative_signature(level)).lower()
            with self.subTest(level=level, mode="whole grains"):
                self.assertIn("whole grains", merged)
            with self.subTest(level=level, mode="separated liquid"):
                self.assertIn("separated", merged)
                self.assertIn("liquid", merged)

    def test_bad_level_rejected(self) -> None:
        for bad in (0, 2, 8, "4"):  # type: ignore[arg-type]
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                judge_checklist(bad)
        with self.assertRaises(ValueError):
            negative_signature(99)


class BuildJudgePromptTests(unittest.TestCase):
    def _spec(self, level: int, *, cue: bool, variant: str | None = None):
        want = 1.0 if cue else 0.0
        for seed in range(500):
            spec = build_prompt(level, random.Random(seed), cue_probability=want)
            if (spec.cue is not None) != cue:
                continue
            if variant is not None and spec.level_variant != variant:
                continue
            return spec
        raise AssertionError("could not build a spec with the required cue presence / variant")

    def test_prompt_contains_checklist_and_negatives(self) -> None:
        for level in LEVELS:
            spec = self._spec(level, cue=(level != 3), variant=("Easy to Chew" if level == 7 else None))
            prompt = build_judge_prompt(spec)
            with self.subTest(level=level):
                self.assertIn(f"IDDSI Level {level} {spec.level_variant}", prompt)
                self.assertIn("visual-signature checklist", prompt)
                for item in judge_checklist(level).checklist:
                    self.assertIn(item, prompt)
                self.assertIn("Negative signatures", prompt)
                for item in negative_signature(level):
                    self.assertIn(item, prompt)

    def test_prompt_deterministic(self) -> None:
        for level in LEVELS:
            spec = self._spec(level, cue=False)
            with self.subTest(level=level):
                self.assertEqual(build_judge_prompt(spec), build_judge_prompt(spec))

    def test_prompt_matches_judge_json_contract(self) -> None:
        spec = self._spec(4, cue=True)
        prompt = build_judge_prompt(spec)
        # Exact response contract of judge._judge_prompt - a later wiring
        # change must not alter what decision_from_response parses.
        for key in (
            "descriptor_match_score",
            "visual_quality_score",
            "cue_consistency_score",
            "observed_level",
            "contains_face",
            "contains_text",
            "reasons",
        ):
            self.assertIn(key, prompt)
        self.assertIn(
            "an image cannot prove rheology",
            prompt.lower(),
        )
        self.assertIn("No markdown.", prompt)

    def test_l7_variant_selection(self) -> None:
        easy = self._spec(7, cue=False, variant="Easy to Chew")
        self.assertEqual(easy.level_variant, "Easy to Chew")
        regular = self._spec(7, cue=False, variant="Regular")
        self.assertEqual(regular.level_variant, "Regular")
        easy_prompt = build_judge_prompt(easy)
        regular_prompt = build_judge_prompt(regular)
        for item in judge_checklist(7).checklist:
            self.assertIn(item, easy_prompt)
        for item in L7_REGULAR_CHECKLIST:
            self.assertIn(item, regular_prompt)
        self.assertIn("no texture restriction", regular_prompt)
        self.assertNotEqual(easy_prompt, regular_prompt)

    def test_cue_wording_mirrors_judge(self) -> None:
        spec = self._spec(5, cue=False)
        self.assertIn("Expected staged utensil cue: none expected.", build_judge_prompt(spec))
        spec_with_cue = self._spec(5, cue=True)
        self.assertIn(f"Expected staged utensil cue: {spec_with_cue.cue}.", build_judge_prompt(spec_with_cue))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
