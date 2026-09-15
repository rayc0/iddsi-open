import pytest

from explain.client import DISCLAIMERS, ExplanationClient, append_frozen_disclaimer, build_prompt


class FakeClient(ExplanationClient):
    def _generate_openai_compatible(self, prompt: str) -> str:
        assert "never a pass/fail" in prompt
        return "Structured explanation."


def test_explanation_always_appends_controlled_disclaimer():
    client = FakeClient(backend="openai_compatible", base_url="http://unused.invalid")
    text = client.generate({"decision": "unclear", "visible_cues": [], "test_guidance": "physical test"}, "en")
    assert text.endswith(DISCLAIMERS["en"])


def test_patient_and_safety_fields_are_rejected():
    with pytest.raises(ValueError, match="outside"):
        build_prompt({"decision": "unclear", "safe_to_eat": True}, "yue")


def test_localized_disclaimer_is_static():
    assert append_frozen_disclaimer("內容", "yue").endswith(DISCLAIMERS["yue"])

