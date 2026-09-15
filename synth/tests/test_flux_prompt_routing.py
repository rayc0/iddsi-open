"""Tests for FLUX CLIP-truncation prompt routing in the diffusers adapter.

FLUX routes ``prompt`` to a CLIP text encoder with a hard 77-token limit, so
the full texture descriptor must travel via ``prompt_2`` (T5, up to 512
tokens) while CLIP receives only a short dish-and-plating caption.  SDXL has
no such dual-encoder pathway and must keep its previous behaviour exactly.

All tests use fake pipeline/torch objects: no model downloads, no GPU.
"""

from __future__ import annotations

import random
import sys
import unittest
from pathlib import Path

from PIL import Image


SYNTH_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SYNTH_ROOT))

from synth.pipelines import (  # noqa: E402
    CLIP_TOKEN_LIMIT as PROD_CLIP_LIMIT,
    DiffusersConfig,
    DiffusersImagePipeline,
    clip_token_count,
)
from synth.prompt_bank import ALL_DISHES, PLATES, build_prompt, short_caption  # noqa: E402


CLIP_TOKEN_LIMIT = 77


def _clip_token_proxy(text: str) -> int:
    """Whitespace/punctuation word count as a cheap CLIP-token upper bound.

    CLIP's BPE tokenizer emits at least one token per whitespace-separated
    word (punctuation usually splits further), so a word count at or under
    the 77-token limit is a conservative proxy that needs no tokenizer
    dependency.
    """
    return len(text.split())


class _FakeResult:
    def __init__(self, images: list[Image.Image]) -> None:
        self.images = images


class _FakePipe:
    """Records the kwargs of its last call and returns blank images."""

    def __init__(self) -> None:
        self.calls: list[dict] = []
        self.device = "cpu"

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        batch = len(kwargs["prompt"])
        return _FakeResult([Image.new("RGB", (kwargs["width"], kwargs["height"])) for _ in range(batch)])


class _FakeGenerator:
    def __init__(self, device: str) -> None:
        self.device = device
        self.seed: int | None = None

    def manual_seed(self, seed: int) -> "_FakeGenerator":
        self.seed = seed
        return self


class _FakeInferenceMode:
    def __enter__(self) -> None:
        return None

    def __exit__(self, *exc) -> bool:
        return False


class _FakeTorch:
    Generator = _FakeGenerator

    @staticmethod
    def inference_mode() -> _FakeInferenceMode:
        return _FakeInferenceMode()


def _loaded_pipeline(model_id: str) -> tuple[DiffusersImagePipeline, _FakePipe]:
    pipeline = DiffusersImagePipeline(DiffusersConfig(primary_model=model_id, allow_cpu=True))
    fake_pipe = _FakePipe()
    pipeline._pipe = fake_pipe  # bypass _ensure_loaded: no downloads, no GPU
    pipeline._torch = _FakeTorch()
    pipeline.model_id = model_id
    return pipeline, fake_pipe


class ShortCaptionTests(unittest.TestCase):
    def test_short_caption_is_deterministic_and_names_dish_and_plate(self) -> None:
        spec = build_prompt(4, random.Random(123), cue_probability=1.0)
        again = short_caption(spec.dish, spec.plate_description)
        self.assertEqual(spec.short_prompt, again)
        self.assertIn(spec.dish.description_en, spec.short_prompt)
        self.assertIn(spec.plate_description, spec.short_prompt)

    def test_short_caption_stays_under_clip_limit_for_all_levels(self) -> None:
        for level in (3, 4, 5, 6, 7):
            for seed in range(40):
                spec = build_prompt(level, random.Random(seed), cue_probability=1.0)
                self.assertLessEqual(
                    _clip_token_proxy(spec.short_prompt),
                    CLIP_TOKEN_LIMIT,
                    msg=f"L{level} seed={seed} short caption too long: {spec.short_prompt!r}",
                )
                # And the full descriptor really is the thing that overflows.
                self.assertGreater(_clip_token_proxy(spec.prompt), CLIP_TOKEN_LIMIT)


class FluxPromptRoutingTests(unittest.TestCase):
    def test_flux_sends_short_prompt_to_clip_and_full_descriptor_to_t5(self) -> None:
        pipeline, fake_pipe = _loaded_pipeline("black-forest-labs/FLUX.1-schnell")
        full = ["FULL descriptor one", "FULL descriptor two"]
        short = ["short one", "short two"]
        images = pipeline.generate(
            full,
            ["neg one", "neg two"],
            [11, 22],
            width=64,
            height=64,
            short_prompts=short,
        )
        self.assertEqual(len(images), 2)
        self.assertEqual(len(fake_pipe.calls), 1)
        kwargs = fake_pipe.calls[0]
        self.assertIn("prompt", kwargs)
        self.assertIn("prompt_2", kwargs)
        self.assertEqual(kwargs["prompt"], short)
        self.assertEqual(kwargs["prompt_2"], full)  # full descriptor unmodified
        self.assertNotIn("negative_prompt", kwargs)  # FLUX path never sent one before either
        self.assertEqual(kwargs["num_inference_steps"], pipeline.config.flux_steps)
        self.assertEqual(kwargs["max_sequence_length"], 256)

    def test_flux_without_short_prompts_keeps_legacy_single_prompt(self) -> None:
        pipeline, fake_pipe = _loaded_pipeline("black-forest-labs/FLUX.1-schnell")
        pipeline.generate(["only prompt"], ["neg"], [7], width=64, height=64)
        kwargs = fake_pipe.calls[0]
        self.assertEqual(kwargs["prompt"], ["only prompt"])
        self.assertNotIn("prompt_2", kwargs)

    def test_sdxl_path_is_unchanged(self) -> None:
        pipeline, fake_pipe = _loaded_pipeline("stabilityai/sdxl-turbo")
        full = ["FULL descriptor one", "FULL descriptor two"]
        negatives = ["neg one", "neg two"]
        pipeline.generate(
            full,
            negatives,
            [11, 22],
            width=64,
            height=64,
            short_prompts=["short one", "short two"],  # must be ignored off-FLUX
        )
        kwargs = fake_pipe.calls[0]
        self.assertEqual(kwargs["prompt"], full)  # same prompt as before
        self.assertNotIn("prompt_2", kwargs)
        self.assertEqual(kwargs["negative_prompt"], negatives)
        self.assertEqual(kwargs["num_inference_steps"], pipeline.config.sdxl_steps)
        self.assertNotIn("max_sequence_length", kwargs)

    def test_short_prompts_length_mismatch_rejected(self) -> None:
        pipeline, _ = _loaded_pipeline("black-forest-labs/FLUX.1-schnell")
        with self.assertRaisesRegex(ValueError, "short_prompts"):
            pipeline.generate(["a", "b"], ["n", "n"], [1, 2], width=64, height=64, short_prompts=["only one"])


class RealBankPromptRoutingTests(unittest.TestCase):
    """The contract against the real 99-dish bank, not synthetic strings."""

    def test_production_clip_limit_is_77(self) -> None:
        self.assertEqual(PROD_CLIP_LIMIT, 77)

    def test_long_bank_prompt_reaches_t5_untruncated(self) -> None:
        spec = build_prompt(4, random.Random(20260831))
        # The fixture premise: a real bank prompt overflows CLIP ...
        self.assertGreater(_clip_token_proxy(spec.prompt), CLIP_TOKEN_LIMIT)
        self.assertGreater(clip_token_count(spec.prompt), PROD_CLIP_LIMIT)
        pipeline, fake_pipe = _loaded_pipeline("black-forest-labs/FLUX.1-schnell")
        pipeline.generate(
            [spec.prompt],
            [spec.negative_prompt],
            [99],
            width=64,
            height=64,
            short_prompts=[spec.short_prompt],
        )
        kwargs = fake_pipe.calls[0]
        # ... yet the exact full descriptor survives to the T5 slot.
        self.assertEqual(kwargs["prompt_2"], [spec.prompt])
        self.assertEqual(kwargs["prompt"], [spec.short_prompt])

    def test_clip_core_within_limit_for_full_99_dish_bank(self) -> None:
        self.assertEqual(len(ALL_DISHES), 99)
        for level in (3, 4, 5, 6, 7):
            for index, dish in enumerate(ALL_DISHES):
                spec = build_prompt(level, random.Random(level * 1000 + index), dish=dish)
                self.assertEqual(spec.dish.slug, dish.slug)
                self.assertLessEqual(
                    clip_token_count(spec.short_prompt),
                    PROD_CLIP_LIMIT,
                    msg=f"L{level} {dish.slug} CLIP core over budget",
                )
        # short_caption depends only on dish + plate: cover every combo.
        for dish in ALL_DISHES:
            for _, plate_description in PLATES:
                caption = short_caption(dish, plate_description)
                self.assertLessEqual(
                    clip_token_count(caption),
                    PROD_CLIP_LIMIT,
                    msg=f"{dish.slug} / {plate_description[:40]}... over budget",
                )


class ClipGuardTests(unittest.TestCase):
    def test_guard_warns_on_legacy_flux_path_without_short_prompts(self) -> None:
        spec = build_prompt(4, random.Random(20260831))
        pipeline, _ = _loaded_pipeline("black-forest-labs/FLUX.1-schnell")
        with self.assertWarnsRegex(UserWarning, "77"):
            pipeline.generate([spec.prompt], ["neg"], [1], width=64, height=64)

    def test_guard_warns_on_overlong_clip_core(self) -> None:
        pipeline, _ = _loaded_pipeline("black-forest-labs/FLUX.1-schnell")
        with self.assertWarnsRegex(UserWarning, "77"):
            pipeline.generate(
                ["full descriptor"],
                ["neg"],
                [1],
                width=64,
                height=64,
                short_prompts=["word " * 100],
            )

    def test_guard_silent_for_in_budget_clip_core(self) -> None:
        import warnings

        spec = build_prompt(4, random.Random(20260831))
        pipeline, _ = _loaded_pipeline("black-forest-labs/FLUX.1-schnell")
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # any warning becomes a failure
            pipeline.generate(
                [spec.prompt],
                [spec.negative_prompt],
                [1],
                width=64,
                height=64,
                short_prompts=[spec.short_prompt],
            )


if __name__ == "__main__":
    unittest.main()
