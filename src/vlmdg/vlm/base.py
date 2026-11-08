"""Abstract VLM provider interface and shared value types."""

from __future__ import annotations

import abc
import base64
import mimetypes
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence, Tuple

ImageInput = Path | str | bytes | Tuple[str, bytes]


@dataclass
class VLMUsage:
    """Token accounting and estimated cost for one or more requests."""

    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    requests: int = 0

    def __add__(self, other: "VLMUsage") -> "VLMUsage":
        return VLMUsage(
            self.input_tokens + other.input_tokens,
            self.output_tokens + other.output_tokens,
            self.cost_usd + other.cost_usd,
            self.requests + other.requests,
        )


@dataclass
class VLMResponse:
    """A single completion returned by a provider."""

    text: str
    model: str
    usage: VLMUsage = field(default_factory=VLMUsage)
    raw: Any = None


def encode_image(image: ImageInput) -> Tuple[str, str]:
    """Return ``(media_type, base64_data)`` for a path / raw bytes / typed tuple."""
    if isinstance(image, tuple):
        media_type, data = image
        return media_type, base64.b64encode(data).decode("ascii")
    if isinstance(image, (bytes, bytearray)):
        return "image/png", base64.b64encode(bytes(image)).decode("ascii")
    path = Path(image)
    media_type = mimetypes.guess_type(path.name)[0] or "image/png"
    return media_type, base64.b64encode(path.read_bytes()).decode("ascii")


class VLMProvider(abc.ABC):
    """Minimal text+image completion interface used by the rule generator."""

    #: Short provider identifier (``"mock"``, ``"anthropic"``, ...).
    name: str = "provider"

    @abc.abstractmethod
    def generate(
        self,
        system: str,
        prompt: str,
        images: Sequence[ImageInput] = (),
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> VLMResponse:
        """Return a completion for ``prompt`` (with optional attached ``images``)."""

    def __repr__(self) -> str:
        return f"<{type(self).__name__} name={self.name!r}>"
