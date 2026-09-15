"""Coverage matrix: prompt bank x IDDSI descriptor phrases.

Enumerates the full combinatorial prompt bank (dish x lighting x plate x
phone x angle x cue x level variant) with a scripted RNG, then reports,
per IDDSI level:

- how many distinct prompts the bank realises (exact and normalised),
- which official descriptor phrases each prompt carries,
- how many prompts leak another level's descriptor vocabulary.

This is prompt-level QA only.  A phrase being present means the *prompt*
carries the descriptor wording; it never means the generated image was
physically tested at that level.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass, field
from typing import Iterable, Iterator, Sequence

from .prompt_bank import (
    ALL_DISHES,
    ANGLES,
    CUES,
    Dish,
    LIGHTING,
    LEVEL_DESCRIPTORS,
    PHONE_PROFILES,
    PLATES,
    PromptSpec,
    build_prompt,
    dishes,
    normalize_prompt,
)

LEVELS: tuple[int, ...] = (3, 4, 5, 6, 7)

# Official IDDSI descriptor wording (visual, prompt-oriented subset) that
# every prompt of a level is expected to carry, cross-checked against the
# level definitions in LEVEL_DESCRIPTORS.
DESCRIPTOR_PHRASES: dict[int, tuple[str, ...]] = {
    3: ("liquidised", "smooth and lump-free", "flows slowly", "no chunks"),
    4: ("pureed", "thick and cohesive", "soft mound", "without being pourable"),
    5: ("minced & moist", "4 mm", "cohesive enough to hold together", "no hard pieces"),
    6: ("soft & bite-sized", "15 mm", "easy to break apart", "tender moist pieces"),
    7: ("easy to chew", "soft and tender", "natural food shapes"),
}

# Vocabulary that belongs to exactly one level.  A prompt for level L must
# not contain any of these words for another level - dish base names are
# kept level-neutral so the level wording always comes from the guidance.
LEVEL_EXCLUSIVE_KEYWORDS: dict[int, tuple[str, ...]] = {
    3: ("liquidised", "liquidized"),
    4: ("pureed", "puree"),
    5: ("minced", "mince"),
    6: ("bite-sized", "bite sized"),
    7: ("easy-to-chew", "easy to chew"),
}


class ScriptedRng(random.Random):
    """Deterministic RNG replaying scripted choice()/random() results."""

    def __init__(self, plan: Sequence[object]) -> None:
        super().__init__(0)
        self._plan = list(plan)
        self._index = 0

    def _next(self) -> object:
        value = self._plan[self._index]
        self._index += 1
        return value

    def choice(self, seq):  # type: ignore[override]
        return self._next()

    def random(self):  # type: ignore[override]
        return float(self._next())  # type: ignore[arg-type]


def _iter_level_specs(level: int, pool: Sequence[Dish]) -> Iterator[PromptSpec]:
    """Yield every combination of the bank's variation axes for one level."""
    cue_options: tuple[str | None, ...] = (None, *CUES[level])
    variants: tuple[str | None, ...] = ("Easy to Chew", "Regular") if level == 7 else (None,)
    for dish in pool:
        for lighting in LIGHTING:
            for plate in PLATES:
                for phone in PHONE_PROFILES:
                    for angle in ANGLES:
                        for cue in cue_options:
                            for variant in variants:
                                plan: list[object] = [
                                    dish,
                                    lighting,
                                    plate,
                                    phone,
                                    angle,
                                    0.0 if cue is not None else 1.0,
                                ]
                                if cue is not None:
                                    plan.append(cue)
                                if variant is not None:
                                    plan.append(variant)
                                yield build_prompt(level, ScriptedRng(plan))


def iter_full_bank(cuisine: str = "all") -> Iterable[PromptSpec]:
    """Yield the entire combinatorial prompt bank, level by level."""
    pool = dishes(cuisine)
    for level in LEVELS:
        yield from _iter_level_specs(level, pool)


@dataclass
class LevelReport:
    level: int
    name: str
    dish_count: int
    prompt_count: int
    distinct_exact: int
    distinct_normalized: int
    phrase_counts: dict[str, int] = field(default_factory=dict)
    leaked_prompts: int = 0
    leak_examples: list[str] = field(default_factory=list)


@dataclass
class BankReport:
    levels: list[LevelReport]
    cuisine: str

    def rows(self) -> str:
        lines = [
            f"coverage matrix - cuisine={self.cuisine!r} - "
            f"dishes={len(dishes(self.cuisine))} levels={len(self.levels)}"
        ]
        for report in self.levels:
            lines.append(
                f"L{report.level} {report.name}: dishes={report.dish_count} "
                f"prompts={report.prompt_count} distinct_exact={report.distinct_exact} "
                f"distinct_normalized={report.distinct_normalized}"
            )
            for phrase, count in report.phrase_counts.items():
                lines.append(f"    descriptor {phrase!r}: {count}/{report.prompt_count}")
            lines.append(f"    cross-level leakage: {report.leaked_prompts} prompts")
            for example in report.leak_examples:
                lines.append(f"        leak example: {example}")
        return "\n".join(lines)


def analyse(cuisine: str = "all") -> BankReport:
    """Stream the full bank and compute the coverage matrix."""
    pool = dishes(cuisine)
    reports: list[LevelReport] = []
    for level in LEVELS:
        phrases = DESCRIPTOR_PHRASES[level]
        exclusive = {other: keywords for other, keywords in LEVEL_EXCLUSIVE_KEYWORDS.items() if other != level}
        phrase_counts = {phrase: 0 for phrase in phrases}
        seen_exact: set[int] = set()
        seen_normalized: set[int] = set()
        leaked = 0
        leak_examples: list[str] = []
        prompt_count = 0
        for spec in _iter_level_specs(level, pool):
            prompt_count += 1
            lowered = spec.prompt.lower()
            seen_exact.add(hash(spec.prompt))
            seen_normalized.add(hash(normalize_prompt(spec.prompt)))
            for phrase in phrases:
                if phrase in lowered:
                    phrase_counts[phrase] += 1
            for other_level, keywords in exclusive.items():
                for keyword in keywords:
                    if keyword in lowered:
                        leaked += 1
                        if len(leak_examples) < 3:
                            leak_examples.append(
                                f"L{level} prompt contains L{other_level} keyword {keyword!r} "
                                f"(dish {spec.dish.name_zh})"
                            )
                        break
        descriptor = LEVEL_DESCRIPTORS[level]
        reports.append(
            LevelReport(
                level=level,
                name=descriptor.name,
                dish_count=len(pool),
                prompt_count=prompt_count,
                distinct_exact=len(seen_exact),
                distinct_normalized=len(seen_normalized),
                phrase_counts=phrase_counts,
                leaked_prompts=leaked,
                leak_examples=leak_examples,
            )
        )
    return BankReport(levels=reports, cuisine=cuisine)


def _normalise_for_print(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def main() -> int:
    print(analyse().rows())
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
