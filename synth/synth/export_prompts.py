"""Export sampled prompts as JSONL for review before any GPU time is spent.

This module deliberately imports nothing beyond the stdlib and
``synth.prompt_bank`` — no torch, no diffusers, no transformers — so it runs
on any machine, including one with no GPU and no model dependencies
installed.  It reproduces the exact seed-sampling logic used by
``synth.generate.SyntheticGenerator.run`` (a master ``random.Random`` seeded
with ``--seed``; each candidate draws ``master_rng.randrange(0, 2**63)`` and
builds its prompt from ``random.Random(candidate_seed)``), so the prompts
reviewed here are the prompts the generator would sample for the same seed
and level.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Sequence

from .prompt_bank import build_prompt


def export_prompts(
    *,
    level: int,
    n: int,
    out: Path,
    seed: int = 20260831,
    cuisine: str = "all",
    cue_probability: float = 0.35,
) -> int:
    """Sample ``n`` prompts exactly as the generator would and write JSONL.

    Each line carries the level, the candidate seed, the full prompt and
    negative prompt, plus the sampled variation fields (dish, lighting,
    plate, phone profile, angle, staged cue) so a reviewer can audit the
    spread without running a model.  Returns the number of rows written.
    """
    if level not in range(3, 8):
        raise ValueError("level must be from 3 through 7")
    if n <= 0:
        raise ValueError("n must be positive")
    if not 0.0 <= cue_probability <= 1.0:
        raise ValueError("cue_probability must be between 0 and 1")
    master_rng = random.Random(seed)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for index in range(1, n + 1):
            candidate_seed = master_rng.randrange(0, 2**63)
            spec = build_prompt(
                level,
                random.Random(candidate_seed),
                cuisine=cuisine,
                cue_probability=cue_probability,
            )
            row = {
                "index": index,
                "level": spec.level,
                "level_variant": spec.level_variant,
                "seed": candidate_seed,
                "dish_slug": spec.dish.slug,
                "dish_zh": spec.dish.name_zh,
                "dish_en": spec.dish.description_en,
                "cuisine_tags": list(spec.dish.cuisine_tags),
                "lighting_category": spec.lighting_category,
                "plate_category": spec.plate_category,
                "phone_profile": spec.phone_profile,
                "angle": spec.angle,
                "cue": spec.cue,
                "prompt": spec.prompt,
                "negative_prompt": spec.negative_prompt,
            }
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return n


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--level", type=int, required=True, choices=range(3, 8))
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260831)
    parser.add_argument("--cuisine", choices=("all", "chinese", "cantonese", "western"), default="all")
    parser.add_argument("--cue-probability", type=float, default=0.35)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    written = export_prompts(
        level=args.level,
        n=args.n,
        out=args.out,
        seed=args.seed,
        cuisine=args.cuisine,
        cue_probability=args.cue_probability,
    )
    print(f"wrote {written} prompts to {args.out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
