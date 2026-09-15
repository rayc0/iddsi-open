"""VLM consistency judges for synthetic candidate filtering."""

from __future__ import annotations

import base64
import json
import mimetypes
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

from PIL import Image

from .prompt_bank import LEVEL_DESCRIPTORS, PromptSpec


DEFAULT_QWEN_MODEL = "Qwen/Qwen3-VL-2B-Instruct"

# Default base URL for the OpenAI-compatible judge endpoint.  Port 8091 is
# reserved for the future W5 Qwen3-VL serving endpoint (vLLM/SGLang
# OpenAI-compatible server); the earlier deepseek text endpoint lives on
# 8090 and remains selectable via --endpoint.
DEFAULT_OPENAI_ENDPOINT = "http://127.0.0.1:8091/v1"


@dataclass(frozen=True)
class JudgeThresholds:
    descriptor_match: float = 0.65
    visual_quality: float = 0.50
    cue_consistency: float = 0.50
    require_observed_level: bool = True

    def __post_init__(self) -> None:
        for value in (self.descriptor_match, self.visual_quality, self.cue_consistency):
            if not 0.0 <= value <= 1.0:
                raise ValueError("judge thresholds must be between 0 and 1")


@dataclass(frozen=True)
class JudgeDecision:
    accepted: bool
    backend: str
    descriptor_match_score: float | None
    visual_quality_score: float | None
    cue_consistency_score: float | None
    observed_level: int | None
    contains_face: bool | None
    contains_text: bool | None
    reasons: tuple[str, ...]
    raw_response: str | None = None
    status: str = "scored"

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["reasons"] = list(self.reasons)
        return value


class ConsistencyJudge(Protocol):
    def judge(self, image_path: Path, spec: PromptSpec) -> JudgeDecision: ...


class DisabledJudge:
    """Explicit opt-out.  It logs missing scores instead of inventing them."""

    def judge(self, image_path: Path, spec: PromptSpec) -> JudgeDecision:
        return JudgeDecision(
            accepted=True,
            backend="disabled",
            descriptor_match_score=None,
            visual_quality_score=None,
            cue_consistency_score=None,
            observed_level=None,
            contains_face=None,
            contains_text=None,
            reasons=("VLM judge explicitly disabled; candidate was not consistency-scored.",),
            status="not_run",
        )


def _judge_prompt(spec: PromptSpec) -> str:
    descriptor = LEVEL_DESCRIPTORS[spec.level]
    cue = spec.cue or "none expected"
    return f"""You are auditing a synthetic food photograph for visual consistency only.
Target: IDDSI Level {spec.level} {spec.level_variant}.
Prompt-oriented visual descriptor: {spec.visual_guidance}.
Dish: {spec.dish.description_en}. Expected staged utensil cue: {cue}.

Important: an image cannot prove rheology, flow, hardness, adhesiveness, cohesiveness, or a physical IDDSI test. Do not claim that it does. Judge only visible compatibility, image quality, prohibited text/faces, and whether a visible cue contradicts the target.

Return exactly one JSON object with these keys:
{{"descriptor_match_score": 0.0, "visual_quality_score": 0.0, "cue_consistency_score": null, "observed_level": null, "contains_face": false, "contains_text": false, "reasons": ["brief reason"]}}
Scores are numbers from 0 to 1. cue_consistency_score is null when no cue is expected. observed_level is 3, 4, 5, 6, 7, or null. contains_face and contains_text must be JSON booleans. No markdown."""


def _extract_json(text: str) -> dict[str, Any]:
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("judge response did not contain a JSON object")
    value = json.loads(text[start : end + 1])
    if not isinstance(value, dict):
        raise ValueError("judge response JSON must be an object")
    return value


def _score(value: Any, name: str, *, nullable: bool = False) -> float | None:
    if value is None and nullable:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number from 0 to 1")
    result = float(value)
    if not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must be a number from 0 to 1")
    return result


def decision_from_response(
    text: str,
    spec: PromptSpec,
    thresholds: JudgeThresholds,
    *,
    backend: str,
) -> JudgeDecision:
    try:
        value = _extract_json(text)
        descriptor_score = _score(value.get("descriptor_match_score"), "descriptor_match_score")
        quality_score = _score(value.get("visual_quality_score"), "visual_quality_score")
        cue_score = _score(value.get("cue_consistency_score"), "cue_consistency_score", nullable=True)
        observed = value.get("observed_level")
        if observed is not None and (isinstance(observed, bool) or observed not in LEVEL_DESCRIPTORS):
            raise ValueError("observed_level must be 3, 4, 5, 6, 7, or null")
        contains_face = value.get("contains_face")
        contains_text = value.get("contains_text")
        if not isinstance(contains_face, bool) or not isinstance(contains_text, bool):
            raise ValueError("contains_face and contains_text must be booleans")
        raw_reasons = value.get("reasons", [])
        if not isinstance(raw_reasons, list) or not all(isinstance(reason, str) for reason in raw_reasons):
            raise ValueError("reasons must be an array of strings")
        failures: list[str] = []
        if descriptor_score is None or descriptor_score < thresholds.descriptor_match:
            failures.append("descriptor score below threshold")
        if quality_score is None or quality_score < thresholds.visual_quality:
            failures.append("visual quality score below threshold")
        if spec.cue is not None and (cue_score is None or cue_score < thresholds.cue_consistency):
            failures.append("cue score missing or below threshold")
        if observed is not None and observed != spec.level:
            failures.append("observed level did not match declared prompt level")
        elif thresholds.require_observed_level and observed is None:
            failures.append("observed level was null")
        if contains_face:
            failures.append("image contains a face")
        if contains_text:
            failures.append("image contains visible text or a label")
        reasons = tuple(raw_reasons + failures) or ("Scores met configured thresholds.",)
        return JudgeDecision(
            accepted=not failures,
            backend=backend,
            descriptor_match_score=descriptor_score,
            visual_quality_score=quality_score,
            cue_consistency_score=cue_score,
            observed_level=observed,
            contains_face=contains_face,
            contains_text=contains_text,
            reasons=reasons,
            raw_response=text,
        )
    except (ValueError, json.JSONDecodeError) as exc:
        return JudgeDecision(
            accepted=False,
            backend=backend,
            descriptor_match_score=None,
            visual_quality_score=None,
            cue_consistency_score=None,
            observed_level=None,
            contains_face=None,
            contains_text=None,
            reasons=(f"Invalid judge response: {exc}",),
            raw_response=text,
            status="error",
        )


class OpenAICompatibleJudge:
    def __init__(
        self,
        *,
        endpoint: str = DEFAULT_OPENAI_ENDPOINT,
        model: str = "deepseek-v4-flash",
        thresholds: JudgeThresholds | None = None,
        timeout: float = 120.0,
    ) -> None:
        self.endpoint = endpoint.rstrip("/")
        self.model = model
        self.thresholds = thresholds or JudgeThresholds()
        self.timeout = timeout

    def judge(self, image_path: Path, spec: PromptSpec) -> JudgeDecision:
        mime = mimetypes.guess_type(image_path.name)[0] or "image/jpeg"
        encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
        body = {
            "model": self.model,
            "temperature": 0,
            "max_tokens": 400,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": _judge_prompt(spec)},
                        {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{encoded}"}},
                    ],
                }
            ],
        }
        request = urllib.request.Request(
            f"{self.endpoint}/chat/completions",
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
            text = payload["choices"][0]["message"]["content"]
            if not isinstance(text, str):
                raise ValueError("endpoint message content was not text")
        except (OSError, KeyError, IndexError, ValueError, json.JSONDecodeError) as exc:
            return JudgeDecision(
                accepted=False,
                backend=f"openai:{self.model}",
                descriptor_match_score=None,
                visual_quality_score=None,
                cue_consistency_score=None,
                observed_level=None,
                contains_face=None,
                contains_text=None,
                reasons=(f"Judge endpoint failed or does not support image input: {exc}",),
                status="error",
            )
        return decision_from_response(text, spec, self.thresholds, backend=f"openai:{self.model}")


class TransformersQwenJudge:
    def __init__(
        self,
        *,
        model_id: str = DEFAULT_QWEN_MODEL,
        device: str = "auto",
        allow_cpu: bool = False,
        thresholds: JudgeThresholds | None = None,
    ) -> None:
        self.model_id = model_id
        self.device_name = device
        self.allow_cpu = allow_cpu
        self.thresholds = thresholds or JudgeThresholds()
        self._model: Any | None = None
        self._processor: Any | None = None
        self._torch: Any | None = None
        self._device: str | None = None

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            from transformers import AutoModelForImageTextToText, AutoProcessor
        except ImportError as exc:  # pragma: no cover - target-only dependency
            raise RuntimeError("transformers and torch are required for the local Qwen judge") from exc
        device = self.device_name if self.device_name != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        if device == "cpu" and not self.allow_cpu:
            raise RuntimeError("Local Qwen VLM judging requires CUDA by default; pass --allow-cpu deliberately")
        on_cuda = device.startswith("cuda")
        dtype = torch.bfloat16 if on_cuda and torch.cuda.is_bf16_supported() else (
            torch.float16 if on_cuda else torch.float32
        )
        processor = AutoProcessor.from_pretrained(self.model_id)
        model = AutoModelForImageTextToText.from_pretrained(self.model_id, torch_dtype=dtype)
        model.to(device)
        model.eval()
        self._model, self._processor, self._torch, self._device = model, processor, torch, device

    def judge(self, image_path: Path, spec: PromptSpec) -> JudgeDecision:
        try:
            self._load()
            assert self._model is not None and self._processor is not None
            assert self._torch is not None and self._device is not None
            messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": _judge_prompt(spec)}]}]
            rendered = self._processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            with Image.open(image_path) as opened:
                image = opened.convert("RGB")
                inputs = self._processor(text=[rendered], images=[image], padding=True, return_tensors="pt")
            inputs = {key: value.to(self._device) for key, value in inputs.items()}
            input_length = inputs["input_ids"].shape[1]
            with self._torch.inference_mode():
                output = self._model.generate(**inputs, max_new_tokens=400, do_sample=False)
            text = self._processor.batch_decode(output[:, input_length:], skip_special_tokens=True)[0]
        except Exception as exc:  # load/inference errors are recorded, never treated as passes
            return JudgeDecision(
                accepted=False,
                backend=f"transformers:{self.model_id}",
                descriptor_match_score=None,
                visual_quality_score=None,
                cue_consistency_score=None,
                observed_level=None,
                contains_face=None,
                contains_text=None,
                reasons=(f"Local VLM judge failed: {exc}",),
                status="error",
            )
        return decision_from_response(text, spec, self.thresholds, backend=f"transformers:{self.model_id}")
