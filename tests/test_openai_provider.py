"""The OpenAI provider builds the right request and reads the reply (no network)."""

from types import SimpleNamespace

import pytest

from vlmdg.config import VLMConfig
from vlmdg.vlm.openai_provider import OpenAIVLMProvider


class _FakeCompletions:
    def __init__(self, finish_reason="stop"):
        self.request = None
        self.finish_reason = finish_reason

    def create(self, **request):
        self.request = request
        message = SimpleNamespace(content="OK", refusal=None)
        return SimpleNamespace(
            model=request["model"],
            choices=[SimpleNamespace(message=message, finish_reason=self.finish_reason)],
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=2),
        )


def _provider(completions, temperature=1.0):
    config = VLMConfig(provider="openai", model="gpt-6-luna", temperature=temperature)
    provider = OpenAIVLMProvider(config, api_key="test-key")
    provider._client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    return provider


def test_request_and_response():
    completions = _FakeCompletions()
    response = _provider(completions).generate("sys", "hi", [("image/png", b"\x89PNG")])
    request = completions.request
    assert request["model"] == "gpt-6-luna"
    assert request["messages"][0] == {"role": "system", "content": "sys"}
    parts = request["messages"][1]["content"]
    assert parts[0] == {"type": "text", "text": "hi"}
    assert parts[1]["image_url"]["url"].startswith("data:image/png;base64,")
    assert "temperature" not in request
    assert response.text == "OK"
    assert (response.usage.input_tokens, response.usage.output_tokens) == (10, 2)


def test_non_default_temperature_is_sent():
    completions = _FakeCompletions()
    _provider(completions, temperature=0.3).generate("sys", "hi")
    assert completions.request["temperature"] == 0.3


def test_truncated_reply_raises():
    with pytest.raises(RuntimeError, match="output-token limit"):
        _provider(_FakeCompletions(finish_reason="length")).generate("sys", "hi")
