"""TAMU AI chat gateway adapter for runtime grammar-rule generation."""

from __future__ import annotations

import os
import time
from typing import Sequence

from vlmdg.config import VLMConfig, load_env
from vlmdg.vlm.base import ImageInput, VLMProvider, VLMResponse, VLMUsage, encode_image

DEFAULT_ENDPOINT = "https://chat-api.tamu.ai"
MODEL_PREFIX = "protected."
TRANSIENT_STATUS_CODES = {429, 500, 502, 503, 504, 524}
MAX_RETRIES = 3
RETRY_BACKOFF_SECONDS = 15
TIMEOUT_SECONDS = 300


class TAMUVLMProvider(VLMProvider):
    """Post directly to TAMU's OpenAI-compatible chat-completions route."""

    name = "tamu"

    def __init__(
        self,
        config: VLMConfig | None = None,
        api_key: str | None = None,
        endpoint: str | None = None,
    ) -> None:
        load_env()
        try:
            import requests  # pylint: disable=import-outside-toplevel
        except ImportError as exc:
            raise RuntimeError("Install the 'requests' package to use --provider tamu") from exc

        key = api_key or os.getenv("TAMUS_AI_CHAT_API_KEY")
        if not key:
            raise RuntimeError("Set TAMUS_AI_CHAT_API_KEY in vlmdg/.env or the environment")
        self.config = config or VLMConfig(provider="tamu")
        self.model = (
            self.config.model
            if self.config.model.startswith(MODEL_PREFIX)
            else MODEL_PREFIX + self.config.model
        )
        base_url = endpoint or os.getenv("TAMUS_AI_CHAT_API_ENDPOINT") or DEFAULT_ENDPOINT
        self.url = f"{base_url.rstrip('/')}/openai/chat/completions"
        self.headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
        self._requests = requests

    def _resolve_temperature(self, temperature: float | None) -> float:
        """Use the gateway's required 1.0 temperature for nano and mini models."""
        if "nano" in self.model or "mini" in self.model:
            return 1.0
        return self.config.temperature if temperature is None else temperature

    def _apply_model_overrides(self, body: dict) -> None:
        """Neither read nor write the gateway response cache; disable Claude thinking.

        ``no-cache`` skips cached answers and ``no-store`` keeps this answer out of
        the cache, so repeated rule requests always reach the model.
        """
        body["cache"] = {"no-cache": True, "no-store": True}
        if "claude" in self.model.lower():
            body["thinking"] = {"type": "disabled"}

    def generate(
        self,
        system: str,
        prompt: str,
        images: Sequence[ImageInput] = (),
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> VLMResponse:
        """Return model text; ``max_tokens`` is unused to match the supplied TAMU call."""
        content: list[dict] = [{"type": "text", "text": prompt}]
        for image in images:
            media_type, data = encode_image(image)
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{media_type};base64,{data}"},
                }
            )
        body = {
            "model": self.model,
            "stream": False,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": content if images else prompt},
            ],
            "temperature": self._resolve_temperature(temperature),
        }
        self._apply_model_overrides(body)

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                response = self._requests.post(
                    self.url, headers=self.headers, json=body, timeout=TIMEOUT_SECONDS
                )
                response.raise_for_status()
            except self._requests.exceptions.RequestException as exc:
                status = getattr(getattr(exc, "response", None), "status_code", None)
                if (
                    status is not None and status not in TRANSIENT_STATUS_CODES
                ) or attempt == MAX_RETRIES:
                    raise RuntimeError(f"TAMU API request failed: {exc}") from exc
                time.sleep(RETRY_BACKOFF_SECONDS * attempt)
                continue

            payload = response.json()
            if "error" in payload and "choices" not in payload:
                error = payload["error"]
                detail = (
                    error.get("message", "unknown error") if isinstance(error, dict) else str(error)
                )
                code = error.get("code") if isinstance(error, dict) else None
                try:
                    transient = int(code) in TRANSIENT_STATUS_CODES
                except (TypeError, ValueError):
                    transient = False
                if transient and attempt < MAX_RETRIES:
                    time.sleep(RETRY_BACKOFF_SECONDS * attempt)
                    continue
                raise RuntimeError(f"TAMU API returned no choices: {detail}")

            choices = payload.get("choices")
            if not choices:
                raise RuntimeError("TAMU response contained no choices")
            choice = choices[0]
            message = choice.get("message") or {}
            if message.get("refusal"):
                raise RuntimeError(f"TAMU response refused the request: {message['refusal']}")
            finish_reason = choice.get("finish_reason")
            if finish_reason == "length":
                raise RuntimeError("TAMU response reached the output-token limit")
            text = message.get("content") or ""
            if not isinstance(text, str) or not text.strip():
                if attempt < MAX_RETRIES:
                    time.sleep(RETRY_BACKOFF_SECONDS * attempt)
                    continue
                raise RuntimeError(
                    f"TAMU response contained no text (finish_reason={finish_reason})"
                )
            usage = payload.get("usage") or {}
            return VLMResponse(
                text=text,
                model=self.model,
                usage=VLMUsage(
                    input_tokens=int(usage.get("prompt_tokens") or 0),
                    output_tokens=int(usage.get("completion_tokens") or 0),
                    # Institutional gateway cost is not known from this response.
                    cost_usd=0.0,
                    requests=attempt,
                ),
                raw=payload,
            )

        raise RuntimeError("TAMU request failed after all retries")
