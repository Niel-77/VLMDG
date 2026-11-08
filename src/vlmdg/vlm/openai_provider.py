"""OpenAI VLM provider.

Reads ``OPENAI_API_KEY`` from the environment (``vlmdg.config.load_env`` loads
``vlmdg/.env`` first) and calls the Chat Completions API. Images are sent as
base64 data URIs. Costs are estimated from the price table in
:class:`~vlmdg.config.VLMConfig`; set those fields for the model in use.
"""

from __future__ import annotations

import os
from typing import Sequence

from vlmdg.config import VLMConfig, load_env
from vlmdg.vlm.base import ImageInput, VLMProvider, VLMResponse, VLMUsage, encode_image


class OpenAIVLMProvider(VLMProvider):
    """Call an OpenAI chat model with optional image attachments."""

    name = "openai"

    def __init__(self, config: VLMConfig | None = None, api_key: str | None = None) -> None:
        load_env()
        try:
            import openai  # pylint: disable=import-outside-toplevel
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "Install the 'openai' package to use --provider openai "
                "(`pip install openai`)"
            ) from exc

        key = api_key or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("Set OPENAI_API_KEY in vlmdg/.env or the environment")
        self.config = config or VLMConfig(provider="openai")
        self._client = openai.OpenAI(api_key=key)

    def generate(
        self,
        system: str,
        prompt: str,
        images: Sequence[ImageInput] = (),
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> VLMResponse:
        """Send one chat request and return the reply text."""
        content: list[dict] = [{"type": "text", "text": prompt}]
        for image in images:
            media_type, data = encode_image(image)
            content.append(
                {"type": "image_url", "image_url": {"url": f"data:{media_type};base64,{data}"}}
            )
        request: dict = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": content},
            ],
            "max_completion_tokens": max_tokens or self.config.max_tokens,
        }
        # Reasoning models accept only the default temperature (1.0), so send it
        # only when a different value is asked for.
        temp = self.config.temperature if temperature is None else temperature
        if temp != 1.0:
            request["temperature"] = temp

        completion = self._client.chat.completions.create(**request)

        choice = completion.choices[0]
        if getattr(choice.message, "refusal", None):
            raise RuntimeError(f"OpenAI response refused the request: {choice.message.refusal}")
        if choice.finish_reason == "length":
            raise RuntimeError("OpenAI response reached the output-token limit")
        text = choice.message.content or ""
        in_tok = int(getattr(completion.usage, "prompt_tokens", 0) or 0)
        out_tok = int(getattr(completion.usage, "completion_tokens", 0) or 0)
        usage = VLMUsage(
            input_tokens=in_tok,
            output_tokens=out_tok,
            cost_usd=self.config.estimate_cost(in_tok, out_tok),
            requests=1,
        )
        return VLMResponse(text=text, model=completion.model, usage=usage, raw=completion)
