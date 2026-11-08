"""TAMU adapter request and gateway error handling without network access."""

import sys
from types import SimpleNamespace

import pytest

from vlmdg.config import VLMConfig
from vlmdg.vlm.tamu_provider import TAMUVLMProvider


def _fake_requests(monkeypatch, payload):
    calls = {"posts": []}
    payloads = iter(payload if isinstance(payload, list) else [payload])

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return next(payloads)

    def post(url, **kwargs):
        calls["posts"].append({"url": url, **kwargs})
        return FakeResponse()

    fake = SimpleNamespace(
        post=post,
        exceptions=SimpleNamespace(RequestException=type("RequestException", (Exception,), {})),
    )
    monkeypatch.setitem(sys.modules, "requests", fake)
    return calls


def test_tamu_request_includes_image_and_prefix(monkeypatch):
    payload = {
        "choices": [
            {
                "message": {"content": "```python\ndef rule(design, rng):\n    return design\n```"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 100, "completion_tokens": 25},
    }
    calls = _fake_requests(monkeypatch, payload)
    provider = TAMUVLMProvider(api_key="test-key", endpoint="https://example.test/")
    result = provider.generate("system", "prompt", images=[b"png"], max_tokens=1000)

    request = calls["posts"][0]
    assert request["url"] == "https://example.test/openai/chat/completions"
    assert request["headers"]["Authorization"] == "Bearer test-key"
    body = request["json"]
    assert body["model"] == "protected.gpt-5.6-luna"
    assert body["stream"] is False
    # gpt-5.6-luna, the default model, accepts only the default temperature of 1.0
    assert body["temperature"] == 1.0
    assert body["cache"] == {"no-cache": True, "no-store": True}
    assert "max_completion_tokens" not in body
    assert body["messages"][1]["content"][1]["image_url"]["url"].startswith(
        "data:image/png;base64,"
    )
    assert result.usage.input_tokens == 100
    assert result.usage.output_tokens == 25
    assert result.usage.requests == 1


def test_tamu_rejects_http_200_gateway_error(monkeypatch):
    _fake_requests(monkeypatch, {"error": {"message": "model unavailable"}})
    provider = TAMUVLMProvider(api_key="test-key")
    with pytest.raises(RuntimeError, match="model unavailable"):
        provider.generate("system", "prompt")


def test_tamu_rejects_truncated_response(monkeypatch):
    _fake_requests(
        monkeypatch,
        {
            "choices": [{"message": {"content": "partial"}, "finish_reason": "length"}],
        },
    )
    provider = TAMUVLMProvider(api_key="test-key")
    with pytest.raises(RuntimeError, match="output-token limit"):
        provider.generate("system", "prompt")


def test_tamu_retries_transient_gateway_error(monkeypatch):
    calls = _fake_requests(
        monkeypatch,
        [
            {"error": {"code": "500", "message": "temporary error"}},
            {"choices": [{"message": {"content": "rule"}, "finish_reason": "stop"}]},
        ],
    )
    monkeypatch.setattr("vlmdg.vlm.tamu_provider.time.sleep", lambda _seconds: None)
    provider = TAMUVLMProvider(api_key="test-key")
    result = provider.generate("system", "prompt")
    assert result.text == "rule"
    assert result.usage.requests == 2
    assert len(calls["posts"]) == 2


def test_tamu_model_overrides(monkeypatch):
    calls = _fake_requests(
        monkeypatch,
        [
            {"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]},
            {"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]},
        ],
    )
    provider = TAMUVLMProvider(
        VLMConfig(provider="tamu", model="Claude-Haiku-4.5"), api_key="test-key"
    )
    provider.generate("system", "prompt", temperature=0.3)
    body = calls["posts"][0]["json"]
    assert body["temperature"] == 0.3
    assert body["thinking"] == {"type": "disabled"}

    provider = TAMUVLMProvider(VLMConfig(provider="tamu", model="gpt-5-nano"), api_key="test-key")
    provider.generate("system", "prompt", temperature=0.3)
    assert calls["posts"][1]["json"]["temperature"] == 1.0
