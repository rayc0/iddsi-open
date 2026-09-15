from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

import requests


DISCLAIMERS = {
    "en": (
        "Research demonstration for culinary education. Estimates visual similarity to IDDSI descriptors; "
        "does not perform official IDDSI tests, assess swallowing, determine suitability for any person, "
        "or decide whether food is safe to consume. Never replaces clinician-prescribed texture or physical "
        "IDDSI testing. This is not the official IDDSI website and is not endorsed by IDDSI."
    ),
    "zh": (
        "本工具僅為烹飪教育研究示範，只估算食物外觀與 IDDSI 描述的相似程度；不執行官方 IDDSI 測試，"
        "不評估吞嚥能力，不判定食物是否適合任何人士，也不判定食物是否可安全食用。絕不能取代臨床人員處方的"
        "食物質地或實體 IDDSI 測試。本工具並非 IDDSI 官方網站，亦未獲 IDDSI 認可。"
    ),
    "yue": (
        "本工具只係烹飪教育研究示範，只會估算食物外觀同 IDDSI 描述有幾相似；唔會執行官方 IDDSI 測試，"
        "唔會評估吞嚥能力，唔會判斷食物適唔適合任何人，亦唔會判斷食物係咪可以安全食用。絕對唔可以取代"
        "臨床人員指定嘅食物質地或者實體 IDDSI 測試。本工具唔係 IDDSI 官方網站，亦未獲 IDDSI 認可。"
    ),
}

LANGUAGE_NAMES = {"en": "English", "zh": "Traditional Chinese", "yue": "written Cantonese (Traditional Chinese)"}


def validate_structured_output(data: dict[str, Any]) -> None:
    if not isinstance(data, dict):
        raise TypeError("structured output must be an object")
    if data.get("decision") not in {"level_compatibility", "unclear"}:
        raise ValueError("decision must be level_compatibility or unclear")
    if data["decision"] == "level_compatibility" and int(data.get("level", -1)) not in range(3, 8):
        raise ValueError("a level_compatibility decision needs level 3 through 7")
    prohibited = {"patient_profile", "diagnosis", "safe_to_eat", "aspiration_risk"} & data.keys()
    if prohibited:
        raise ValueError(f"patient/safety fields are outside this research explanation boundary: {sorted(prohibited)}")


def build_prompt(data: dict[str, Any], language: str) -> str:
    validate_structured_output(data)
    if language not in LANGUAGE_NAMES:
        raise ValueError("language must be en, zh, or yue")
    return (
        f"Write a concise explanation in {LANGUAGE_NAMES[language]} grounded only in the JSON below. "
        "Describe this as visual level compatibility, never a pass/fail, safety, diagnosis, or patient-suitability decision. "
        "Mention uncertainty and tell the reader to perform the relevant physical IDDSI test. "
        "Do not invent observations, measurements, validation, or performance. Do not add a disclaimer; the caller appends it.\n\n"
        + json.dumps(data, ensure_ascii=False, indent=2)
    )


def append_frozen_disclaimer(text: str, language: str) -> str:
    clean = text.strip()
    return f"{clean}\n\n---\n{DISCLAIMERS[language]}"


@dataclass
class ExplanationClient:
    backend: str
    model: str = "Qwen/Qwen3-VL-2B-Instruct"
    base_url: str | None = None
    api_key_env: str = "OPENAI_API_KEY"
    timeout_seconds: int = 60
    max_new_tokens: int = 300

    def generate(self, data: dict[str, Any], language: str = "yue") -> str:
        prompt = build_prompt(data, language)
        if self.backend == "openai_compatible":
            generated = self._generate_openai_compatible(prompt)
        elif self.backend == "local_qwen":
            generated = self._generate_local(prompt)
        else:
            raise ValueError("backend must be openai_compatible or local_qwen")
        return append_frozen_disclaimer(generated, language)

    def _generate_openai_compatible(self, prompt: str) -> str:
        if not self.base_url:
            raise ValueError("base_url is required for openai_compatible backend")
        api_key = os.environ.get(self.api_key_env)
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        base = self.base_url.rstrip("/")
        endpoint = base + "/chat/completions" if base.endswith("/v1") else base + "/v1/chat/completions"
        response = requests.post(
            endpoint,
            headers=headers,
            json={
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.1,
                "max_tokens": self.max_new_tokens,
            },
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        return str(response.json()["choices"][0]["message"]["content"])

    def _generate_local(self, prompt: str) -> str:
        try:
            import torch
            from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
        except ImportError as exc:
            raise ImportError("local Qwen requires a recent transformers installation") from exc
        processor = AutoProcessor.from_pretrained(self.model)
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            self.model, torch_dtype="auto", device_map="auto"
        )
        conversation = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        inputs = processor.apply_chat_template(
            conversation,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(model.device)
        inputs.pop("token_type_ids", None)
        with torch.inference_mode():
            generated_ids = model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
        new_tokens = generated_ids[:, inputs["input_ids"].shape[1]:]
        return str(processor.batch_decode(new_tokens, skip_special_tokens=True)[0])
