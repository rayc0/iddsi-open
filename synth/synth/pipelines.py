"""Lazy diffusers pipeline adapter with an explicit model fallback."""

from __future__ import annotations

import gc
import warnings
from dataclasses import dataclass
from typing import Any, Protocol, Sequence

from PIL import Image


PRIMARY_MODEL = "black-forest-labs/FLUX.1-schnell"
FALLBACK_MODEL = "stabilityai/sdxl-turbo"

#: Hard context limit of the CLIP-L text encoder behind diffusers' ``prompt``
#: argument.  Anything longer is silently truncated, so the full texture
#: descriptor must travel via ``prompt_2`` (T5, up to 512 tokens) instead.
CLIP_TOKEN_LIMIT = 77


def clip_token_count(text: str) -> int:
    """Approximate the CLIP token count of ``text`` with a word count.

    CLIP's BPE tokenizer emits at least one token per whitespace-separated
    word (punctuation usually splits further), so ``len(text.split())`` is a
    conservative proxy: a prompt at or under 77 words is guaranteed to fit,
    and anything over 77 words is guaranteed to be truncated.  This needs no
    tokenizer dependency at generation time.
    """
    return len(text.split())


class ImagePipeline(Protocol):
    model_id: str

    def generate(
        self,
        prompts: Sequence[str],
        negative_prompts: Sequence[str],
        seeds: Sequence[int],
        *,
        width: int,
        height: int,
        short_prompts: Sequence[str] | None = None,
    ) -> list[Image.Image]: ...


@dataclass
class DiffusersConfig:
    primary_model: str = PRIMARY_MODEL
    fallback_model: str = FALLBACK_MODEL
    device: str = "auto"
    allow_cpu: bool = False
    flux_steps: int = 4
    sdxl_steps: int = 2


class DiffusersImagePipeline:
    """Load FLUX.1-schnell on first use, falling back to SDXL-Turbo.

    Model imports and downloads never happen at module import time.  CPU use is
    rejected by default because these release models are not CPU smoke fixtures.
    """

    def __init__(self, config: DiffusersConfig | None = None) -> None:
        self.config = config or DiffusersConfig()
        self.model_id = self.config.primary_model
        self._pipe: Any | None = None
        self._torch: Any | None = None

    def _device(self, torch: Any) -> str:
        if self.config.device != "auto":
            device = self.config.device
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
        if device == "cpu" and not self.config.allow_cpu:
            raise RuntimeError(
                "No CUDA device is available. Release-model generation is GPU-only by default; "
                "use a stub pipeline for tests or pass --allow-cpu deliberately."
            )
        return device

    def _load_one(self, model_id: str, torch: Any, device: str) -> Any:
        try:
            from diffusers import DiffusionPipeline
        except ImportError as exc:  # pragma: no cover - target-only dependency
            raise RuntimeError("diffusers is required for real image generation; install synth[models]") from exc
        on_cuda = device.startswith("cuda")
        dtype = torch.bfloat16 if on_cuda and torch.cuda.is_bf16_supported() else (
            torch.float16 if on_cuda else torch.float32
        )
        pipe = DiffusionPipeline.from_pretrained(model_id, torch_dtype=dtype)
        pipe.to(device)
        if hasattr(pipe, "set_progress_bar_config"):
            pipe.set_progress_bar_config(disable=False)
        return pipe

    def _ensure_loaded(self) -> None:
        if self._pipe is not None:
            return
        try:
            import torch
        except ImportError as exc:  # pragma: no cover - target-only dependency
            raise RuntimeError("PyTorch is required for real image generation; install synth[models]") from exc
        device = self._device(torch)
        primary_error: Exception | None = None
        try:
            self._pipe = self._load_one(self.config.primary_model, torch, device)
            self.model_id = self.config.primary_model
        except Exception as exc:  # fallback is an explicit operational requirement
            primary_error = exc
            gc.collect()
            if device.startswith("cuda"):
                torch.cuda.empty_cache()
            warnings.warn(
                f"Primary diffusion model {self.config.primary_model!r} failed to load; "
                f"trying {self.config.fallback_model!r}. Cause: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )
            try:
                self._pipe = self._load_one(self.config.fallback_model, torch, device)
                self.model_id = self.config.fallback_model
            except Exception as fallback_exc:
                raise RuntimeError(
                    f"Could not load primary model {self.config.primary_model!r} ({primary_error}) "
                    f"or fallback {self.config.fallback_model!r} ({fallback_exc})"
                ) from fallback_exc
        self._torch = torch

    def generate(
        self,
        prompts: Sequence[str],
        negative_prompts: Sequence[str],
        seeds: Sequence[int],
        *,
        width: int,
        height: int,
        short_prompts: Sequence[str] | None = None,
    ) -> list[Image.Image]:
        if not prompts or len(prompts) != len(negative_prompts) or len(prompts) != len(seeds):
            raise ValueError("prompts, negative_prompts, and seeds must have the same non-zero length")
        if short_prompts is not None and len(short_prompts) != len(prompts):
            raise ValueError("short_prompts must match prompts in length when provided")
        self._ensure_loaded()
        assert self._pipe is not None and self._torch is not None
        device = str(self._pipe.device) if hasattr(self._pipe, "device") else self._device(self._torch)
        generators = [self._torch.Generator(device=device).manual_seed(seed) for seed in seeds]
        kwargs: dict[str, Any] = {
            "prompt": list(prompts),
            "generator": generators,
            "height": height,
            "width": width,
            "guidance_scale": 0.0,
        }
        if "FLUX" in self.model_id.upper():
            # FLUX routes `prompt` to its CLIP encoder, which truncates hard at
            # 77 tokens and would silently drop the texture descriptors.  Send
            # the short dish-and-plating caption (clip_core) to CLIP and the
            # FULL descriptor (t5_full, unmodified) to the T5 encoder via
            # `prompt_2` (up to 512 tokens).
            if short_prompts is not None:
                kwargs["prompt"] = list(short_prompts)
                kwargs["prompt_2"] = list(prompts)
            for index, clip_prompt in enumerate(kwargs["prompt"]):
                if clip_token_count(clip_prompt) > CLIP_TOKEN_LIMIT:
                    if short_prompts is None:
                        detail = (
                            "no short_prompts were supplied, so the full descriptor "
                            "is going to CLIP and will be truncated"
                        )
                    else:
                        detail = "the CLIP core is over budget"
                    warnings.warn(
                        f"FLUX CLIP prompt {index} has ~{clip_token_count(clip_prompt)} tokens "
                        f"over the {CLIP_TOKEN_LIMIT}-token CLIP limit ({detail}); "
                        "CLIP will silently truncate it. Pass a compressed core of at most "
                        "77 tokens via short_prompts and keep the full descriptor in "
                        "prompt_2 (T5) so texture language survives.",
                        UserWarning,
                        stacklevel=2,
                    )
            kwargs.update(num_inference_steps=self.config.flux_steps, max_sequence_length=256)
        else:
            # SDXL has a single 77-token CLIP-family encoder: long descriptors
            # stay lossy here by design.  This path is the offline fallback
            # only, so its behaviour is intentionally unchanged (no prompt_2).
            kwargs.update(
                negative_prompt=list(negative_prompts),
                num_inference_steps=self.config.sdxl_steps,
            )
        with self._torch.inference_mode():
            result = self._pipe(**kwargs)
        images = list(result.images)
        if len(images) != len(prompts):
            raise RuntimeError(f"pipeline returned {len(images)} images for {len(prompts)} prompts")
        return images
