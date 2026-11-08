"""Construct a :class:`~vlmdg.vlm.base.VLMProvider` from a name."""

from __future__ import annotations

from typing import Tuple

from vlmdg.config import VLMConfig
from vlmdg.vlm.base import VLMProvider


def available_providers() -> Tuple[str, ...]:
    """Return the recognized provider names."""
    return ("mock", "anthropic", "tamu", "openai", "local")


def get_provider(name: str | None = None, config: VLMConfig | None = None) -> VLMProvider:
    """Return a provider instance for ``name`` (default ``"mock"``).

    ``local`` is a stub and raises :class:`NotImplementedError`.
    """
    key = (name or "mock").strip().lower()

    if key == "mock":
        from vlmdg.vlm.mock_provider import MockVLMProvider  # pylint: disable=import-outside-toplevel

        return MockVLMProvider(config)
    if key == "anthropic":
        from vlmdg.vlm.anthropic_provider import (  # pylint: disable=import-outside-toplevel
            AnthropicVLMProvider,
        )

        return AnthropicVLMProvider(config)
    if key == "tamu":
        from vlmdg.vlm.tamu_provider import TAMUVLMProvider  # pylint: disable=import-outside-toplevel

        return TAMUVLMProvider(config)
    if key == "openai":
        from vlmdg.vlm.openai_provider import OpenAIVLMProvider  # pylint: disable=import-outside-toplevel

        return OpenAIVLMProvider(config)
    if key == "local":
        from vlmdg.vlm.local_provider import LocalVLMProvider  # pylint: disable=import-outside-toplevel

        return LocalVLMProvider(config)

    raise ValueError(f"unknown provider {name!r}; available: {available_providers()}")
