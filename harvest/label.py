from __future__ import annotations

import base64
import io
import json
import re
from typing import Any

from PIL import Image

from .types import WeakLabel


PROMPT_VERSION = "iddsi-visual-descriptors-v2-scale-anchored"
PROMPT = """You are inspecting a food/drink photograph for research data triage.
This is NOT an official IDDSI test and you must not infer swallowing safety.

Return one JSON object only with these keys:
- visible_liquid_flow: boolean (true only when flowing/dripping liquid is visibly captured)
- uniform_smooth: boolean (visibly uniform, smooth, lump-free appearance)
- visible_particles: boolean
- max_particle_mm: number or null (visual estimate only; null without a scale)
- visible_pieces: boolean
- typical_piece_mm: number or null (visual estimate only; null without a scale)
- contains_face: boolean
- contains_legible_text: boolean (including labels, watermarks, menus, badges, screens)
- confidence: number from 0 to 1 for the visual-form observations
- rationale: short factual visual description, AT MOST 20 WORDS, no medical or safety claim

Estimating size without a ruler: most food photographs contain tableware you can use
as a scale reference. Use whichever is visible, in this order of reliability:
a teaspoon bowl is about 25 mm wide, a dinner fork head about 25 mm wide with tines
about 3 mm apart, a tablespoon bowl about 40 mm, a standard rice bowl rim about 110 mm
across, a dinner plate rim about 260 mm. Estimate max_particle_mm and typical_piece_mm
against the nearest visible reference and report the number. Only return null when the
frame contains no tableware, hand, or utensil of any kind to judge against - a missing
number is treated as "unmeasurable", so guessing conservatively is better than null.

Descriptor context for observation only: visible liquid flow maps to at most L3; a
uniform smooth appearance is an L4 candidate; visible particles no larger than 4 mm
are an L5 candidate; pieces around 15 mm are an L6 candidate; otherwise L7. Static
appearance cannot establish flow, softness, stickiness, cohesiveness, or ground truth.
"""


def _as_bool(value: Any) -> bool:
    """Coerce a VLM-supplied truthy value to bool.

    The model returns JSON, but small VLMs routinely emit ``"true"``/``"yes"``
    or ``1`` instead of a JSON boolean. The original strict ``value is True``
    silently read every one of those as False, which pushed the sample into the
    ``else`` branch of :func:`apply_descriptor_rules` (the L7 bucket).
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value == 1
    if isinstance(value, str):
        return value.strip().lower() in {"true", "yes", "y", "1"}
    return False


def _number(value: Any) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None


def apply_descriptor_rules(observations: dict[str, Any], *, model_id: str) -> WeakLabel:
    flow = _as_bool(observations.get("visible_liquid_flow"))
    smooth = _as_bool(observations.get("uniform_smooth"))
    particles = _as_bool(observations.get("visible_particles"))
    particle_mm = _number(observations.get("max_particle_mm"))
    pieces = _as_bool(observations.get("visible_pieces"))
    piece_mm = _number(observations.get("typical_piece_mm"))

    if flow:
        level, rule, relation = 3, "visible_liquid_flow_at_most_l3", "at_most"
    elif smooth and not particles and not pieces:
        level, rule, relation = 4, "uniform_smooth_l4_candidate", "candidate"
    elif particles and particle_mm is not None and particle_mm <= 4.0:
        level, rule, relation = 5, "visible_particles_le_4mm_l5_candidate", "candidate"
    elif pieces and piece_mm is not None and 10.0 <= piece_mm <= 20.0:
        level, rule, relation = 6, "visible_pieces_about_15mm_l6_candidate", "candidate"
    else:
        level, rule, relation = 7, "other_or_unmeasurable_l7_visual_bucket", "candidate"

    confidence = _number(observations.get("confidence"))
    confidence = max(0.0, min(0.95, confidence if confidence is not None else 0.0))
    return WeakLabel(
        level_weak=level,
        confidence=confidence,
        rule=rule,
        level_relation=relation,
        observations=observations,
        model_id=model_id,
        prompt_version=PROMPT_VERSION,
        contains_face=_as_bool(observations.get("contains_face")),
        contains_text=_as_bool(observations.get("contains_legible_text")),
        rationale=str(observations.get("rationale", ""))[:500],
    )


def _salvage_truncated_json(fragment: str) -> dict[str, Any] | None:
    """Recover an object from JSON cut off mid-generation.

    A small VLM that hits ``max_new_tokens`` while writing the free-text
    ``rationale`` leaves an unterminated string and no closing brace, which
    defeats both ``json.loads`` and a greedy ``{.*}`` match. Every field the
    descriptor rules actually read is emitted *before* ``rationale``, so the
    prefix is still worth parsing: truncate back to a complete key/value pair
    and close the object.
    """
    start = fragment.find("{")
    if start < 0:
        return None
    body = fragment[start:]
    cuts = [len(body)] + [i for i in range(len(body) - 1, 0, -1) if body[i] == ","]
    for cut in cuts:
        stem = body[:cut].rstrip().rstrip(",")
        for suffix in ("}", '"}', "]}"):
            try:
                value = json.loads(stem + suffix)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                return value
    return None


def _extract_json(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.I)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, flags=re.S)
        if match:
            value = json.loads(match.group(0))
        else:
            value = _salvage_truncated_json(cleaned)
            if value is None:
                raise ValueError("VLM did not return a JSON object")
    if not isinstance(value, dict):
        raise ValueError("VLM response must be a JSON object")
    return value


class DeferLabeler:
    """No-VLM labeler: keeps the sample, defers weak labelling to a later pass
    (e.g. on the DGX where torch + the local Qwen3-VL model are available).
    Emits label_status metadata via rule="deferred"; never rejects."""

    model_id = "deferred"
    prompt_version = "none"

    def label(self, image: Any) -> "WeakLabel":
        return WeakLabel(
            level_weak=None,
            confidence=0.0,
            model_id=self.model_id,
            prompt_version=self.prompt_version,
            rule="deferred",
            level_relation="candidate",
            observations={},
            rationale="labelling deferred to GPU pass",
            contains_face=False,
            contains_text=False,
        )


class Qwen3VLLabeler:
    """Lazy local transformers loader; CUDA use is explicit and guarded."""

    def __init__(
        self,
        model_id: str = "Qwen/Qwen3-VL-2B-Instruct",
        *,
        device: str = "auto",
        max_new_tokens: int = 384,
    ) -> None:
        self.model_id = model_id
        self.device = device
        self.max_new_tokens = max_new_tokens
        self._model: Any | None = None
        self._processor: Any | None = None

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            import transformers
            from transformers import AutoProcessor
        except ImportError as exc:
            raise RuntimeError(
                "Qwen labelling requires torch and transformers in the target container"
            ) from exc

        requested = self.device
        if requested == "auto":
            requested = "cuda" if torch.cuda.is_available() else "cpu"
        if requested == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested but torch.cuda.is_available() is false")
        if requested not in {"cpu", "cuda"}:
            raise ValueError("device must be auto, cpu, or cuda")

        model_class = getattr(transformers, "AutoModelForMultimodalLM", None)
        if model_class is None:
            model_class = getattr(transformers, "AutoModelForImageTextToText", None)
        if model_class is None:
            raise RuntimeError("installed transformers has no Qwen3-VL-compatible auto model class")

        kwargs: dict[str, Any] = {"trust_remote_code": True}
        if requested == "cuda":
            kwargs.update(device_map="auto", dtype=torch.bfloat16)
        else:
            kwargs.update(dtype=torch.float32)
        self._processor = AutoProcessor.from_pretrained(self.model_id, trust_remote_code=True)
        self._model = model_class.from_pretrained(self.model_id, **kwargs)
        if requested == "cpu":
            self._model.to("cpu")
        self._model.eval()
        self.device = requested

    def label(self, image: Image.Image) -> WeakLabel:
        self._load()
        assert self._model is not None and self._processor is not None
        model_image = image.copy()
        model_image.thumbnail((2048, 2048), Image.Resampling.LANCZOS)
        encoded_image = io.BytesIO()
        model_image.save(encoded_image, format="JPEG", quality=92)
        image_base64 = base64.b64encode(encoded_image.getvalue()).decode("ascii")
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "base64": image_base64},
                    {"type": "text", "text": PROMPT},
                ],
            }
        ]
        inputs = self._processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        if self.device == "cpu":
            inputs = inputs.to("cpu")
        else:
            inputs = inputs.to(self._model.device)
        outputs = self._model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
        )
        input_length = inputs["input_ids"].shape[1]
        generated = outputs[:, input_length:]
        text = self._processor.batch_decode(
            generated,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        return apply_descriptor_rules(_extract_json(text), model_id=self.model_id)
