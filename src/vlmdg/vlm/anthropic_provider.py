"""Anthropic (Claude) VLM provider.

Reads ``ANTHROPIC_API_KEY`` from the environment (``vlmdg.config.load_env`` loads
a ``.env`` first). Costs are estimated from the price table in
:class:`~vlmdg.config.VLMConfig`; override those fields for the exact model and
date in use.
"""

from __future__ import annotations

import os
from typing import Sequence

from vlmdg.config import VLMConfig
from vlmdg.vlm.base import ImageInput, VLMProvider, VLMResponse, VLMUsage


class AnthropicVLMProvider(VLMProvider):
    """Call the Claude Messages API with an optional image attachment."""

    name = "anthropic"

    def __init__(self, config: VLMConfig | None = None, api_key: str | None = None) -> None:
        try:
            import anthropic  # pylint: disable=import-outside-toplevel
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "the 'anthropic' package is required for the Anthropic provider; "
                "install it with `pip install anthropic`"
            ) from exc

        key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set. Export it or add it to a .env file, "
                "or use --provider mock for an offline run."
            )
        self.config = config or VLMConfig(provider="anthropic")
        self._client = anthropic.Anthropic(api_key=key)

    def generate(
        self,
        system: str,
        prompt: str,
        images: Sequence[ImageInput] = (),
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> VLMResponse:
        """Send one message and return the concatenated text blocks of the reply."""
        from vlmdg.vlm.base import encode_image  # pylint: disable=import-outside-toplevel

        content: list[dict] = []
        for image in images:
            media_type, data = encode_image(image)
            content.append(
                {
                    "type": "image",
                    "source": {"type": "base64", "media_type": media_type, "data": data},
                }
            )
        content.append({"type": "text", "text": prompt})

        request = {
            "model": self.config.model,
            "system": system,
            "max_tokens": max_tokens or self.config.max_tokens,
            "temperature": self.config.temperature if temperature is None else temperature,
            "messages": [{"role": "user", "content": content}],
        }
        # astroid mis-resolves the anthropic SDK overloads, so it flags valid kwargs.
        # pylint: disable-next=unexpected-keyword-arg
        message = self._client.messages.create(**request)

        text = "".join(
            block.text for block in message.content if getattr(block, "type", None) == "text"
        )
        in_tok = int(getattr(message.usage, "input_tokens", 0))
        out_tok = int(getattr(message.usage, "output_tokens", 0))
        usage = VLMUsage(
            input_tokens=in_tok,
            output_tokens=out_tok,
            cost_usd=self.config.estimate_cost(in_tok, out_tok),
            requests=1,
        )
        return VLMResponse(text=text, model=self.config.model, usage=usage, raw=message)
