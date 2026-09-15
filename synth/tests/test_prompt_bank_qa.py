"""W15 prompt-bank QA tests: per-level coverage, leakage, dedupe, negatives."""

from __future__ import annotations

import random
import sys
import unittest
from pathlib import Path

SYNTH_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = SYNTH_ROOT.parent
sys.path.insert(0, str(SYNTH_ROOT))
sys.path.insert(0, str(REPO_ROOT))

from synth.coverage import DESCRIPTOR_PHRASES, LEVEL_EXCLUSIVE_KEYWORDS, ScriptedRng  # noqa: E402
from synth.prompt_bank import (  # noqa: E402
    ALL_DISHES,
    LEVEL_DESCRIPTORS,
    LIGHTING,
    PHONE_PROFILES,
    PLATES,
    ANGLES,
    CUES,
    NEGATIVE_PROMPT,
    build_prompt,
    dedupe_prompts,
    normalize_prompt,
)

LEVELS = (3, 4, 5, 6, 7)
SAMPLES_PER_LEVEL = 200
MIN_DISTINCT_PROMPTS = 40


def _sample_prompts(level: int, count: int = SAMPLES_PER_LEVEL) -> list[str]:
    return [build_prompt(level, random.Random(seed * 100 + level)).prompt for seed in range(count)]


def _scripted_prompt(level: int, dish, with_cue: bool) -> str:
    plan: list[object] = [
        dish,
        LIGHTING[0],
        PLATES[0],
        PHONE_PROFILES[0],
        ANGLES[0],
        0.0 if with_cue else 1.0,
    ]
    if with_cue:
        plan.append(CUES[level][0])
    if level == 7:
        plan.append("Easy to Chew" if with_cue else "Regular")
    return build_prompt(level, ScriptedRng(plan)).prompt


class PerLevelCoverageTests(unittest.TestCase):
    def test_every_level_has_at_least_40_distinct_prompts(self) -> None:
        for level in LEVELS:
            with self.subTest(level=level):
                prompts = _sample_prompts(level)
                unique, stats = dedupe_prompts(prompts)
                self.assertGreaterEqual(
                    len(unique),
                    MIN_DISTINCT_PROMPTS,
                    f"L{level} has only {len(unique)} distinct prompts of {len(prompts)} sampled",
                )
                self.assertGreaterEqual(stats.kept, MIN_DISTINCT_PROMPTS)

    def test_each_level_descriptor_phrases_reach_every_prompt(self) -> None:
        for level in LEVELS:
            with self.subTest(level=level):
                for prompt in _sample_prompts(level, count=50):
                    lowered = prompt.lower()
                    for phrase in DESCRIPTOR_PHRASES[level]:
                        if level == 7 and phrase in ("easy to chew", "soft and tender"):
                            continue  # variant-specific; asserted below
                        self.assertIn(phrase, lowered, f"L{level} prompt misses descriptor {phrase!r}")

    def test_l7_variants_each_carry_their_own_descriptor_phrases(self) -> None:
        easy = [p for p in _sample_prompts(7, count=100) if "Easy to Chew" in p]
        regular = [p for p in _sample_prompts(7, count=100) if "Regular" in p]
        self.assertTrue(easy and regular)
        for prompt in easy:
            self.assertIn("easy to chew", prompt.lower())
            self.assertIn("soft and tender", prompt.lower())
        for prompt in regular:
            self.assertIn("natural food shapes", prompt.lower())


class CrossLevelLeakageTests(unittest.TestCase):
    def test_no_prompt_contains_another_levels_descriptor_vocabulary(self) -> None:
        for level in LEVELS:
            foreign = {kw for other, kws in LEVEL_EXCLUSIVE_KEYWORDS.items() if other != level for kw in kws}
            prompts = _sample_prompts(level, count=100)
            prompts += [
                _scripted_prompt(level, dish, with_cue)
                for dish in ALL_DISHES
                for with_cue in (True, False)
            ]
            with self.subTest(level=level):
                for prompt in prompts:
                    lowered = prompt.lower()
                    for keyword in foreign:
                        self.assertNotIn(
                            keyword,
                            lowered,
                            f"L{level} prompt leaks foreign descriptor {keyword!r}: {prompt[:160]}...",
                        )

    def test_dish_bank_is_level_neutral(self) -> None:
        all_keywords = [kw for kws in LEVEL_EXCLUSIVE_KEYWORDS.values() for kw in kws]
        for dish in ALL_DISHES:
            text = (dish.description_en + " " + dish.name_zh).lower()
            with self.subTest(dish=dish.slug):
                for keyword in all_keywords:
                    self.assertNotIn(
                        keyword,
                        text,
                        f"dish {dish.slug} base description embeds level vocabulary {keyword!r}; "
                        "level wording must come from the level guidance, not the dish name",
                    )

    def test_dish_bank_has_no_duplicate_entries(self) -> None:
        seen: set[tuple[str, str]] = set()
        for dish in ALL_DISHES:
            key = (dish.name_zh, dish.description_en.lower())
            self.assertNotIn(key, seen, f"duplicate dish entry: {dish.slug} / {key}")
            seen.add(key)


class NegativePromptTests(unittest.TestCase):
    def test_negative_prompt_suppresses_text_labels_watermarks(self) -> None:
        lowered = NEGATIVE_PROMPT.lower()
        for term in ("text", "labels", "watermarks"):
            self.assertIn(term, lowered)

    def test_negative_prompt_suppresses_human_faces(self) -> None:
        self.assertIn("faces", NEGATIVE_PROMPT.lower())

    def test_negative_prompt_suppresses_utensil_only_shots_without_food(self) -> None:
        lowered = NEGATIVE_PROMPT.lower()
        for term in ("utensil-only", "no food", "empty plate", "empty bowl"):
            self.assertIn(term, lowered)


class DedupeTests(unittest.TestCase):
    def test_normalize_prompt_folds_case_space_and_punctuation(self) -> None:
        self.assertEqual(normalize_prompt("IDDSI Level 4! Pureed..."), normalize_prompt("iddsi    level 4 pureed"))
        self.assertNotEqual(normalize_prompt("mango mash"), normalize_prompt("melon mash"))

    def test_dedupe_removes_exact_and_normalized_duplicates(self) -> None:
        prompts = ["Show congee A.", "Show congee A.", "show   congee a.", "Show taro B."]
        unique, stats = dedupe_prompts(prompts)
        self.assertEqual(unique, ["Show congee A.", "Show taro B."])
        self.assertEqual(stats.removed_exact, 1)
        self.assertEqual(stats.removed_normalized, 1)
        self.assertEqual(stats.removed_total, 2)
        self.assertEqual(stats.kept, 2)

    def test_full_bank_samples_have_no_near_duplicates(self) -> None:
        for level in LEVELS:
            with self.subTest(level=level):
                unique, stats = dedupe_prompts(_sample_prompts(level))
                # Exact repeats can occur when random sampling collides on the
                # same variation combo (birthday effect); only normalised
                # near-duplicates would indicate a real bank defect.
                self.assertEqual(stats.removed_normalized, 0)
                self.assertEqual(len(unique), stats.kept)


if __name__ == "__main__":
    unittest.main()
