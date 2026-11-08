"""Local open-weight VLM provider - declared stub.

Intended target: an OpenAI-compatible local server (Ollama, vLLM, llama.cpp).
Implement :meth:`generate` by pointing an OpenAI-style client at the local base
URL. Kept as a stub so an open-weight model can be added later.
"""

from __future__ import annotations

from typing import Sequence

from vlmdg.config import VLMConfig
from vlmdg.vlm.base import ImageInput, VLMProvider, VLMResponse

_MESSAGE = (
    "LocalVLMProvider is not implemented yet. Point it at an OpenAI-compatible "
    "local server (Ollama / vLLM) or use --provider mock / --provider anthropic."
)


class LocalVLMProvider(VLMProvider):
    """Placeholder for a locally hosted open-weight model."""

    name = "local"

    def __init__(self, config: VLMConfig | None = None, base_url: str | None = None) -> None:
        raise NotImplementedError(_MESSAGE)

    def generate(
        self,
        system: str,
        prompt: str,
        images: Sequence[ImageInput] = (),
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> VLMResponse:  # pragma: no cover - unreachable stub
        raise NotImplementedError(_MESSAGE)
