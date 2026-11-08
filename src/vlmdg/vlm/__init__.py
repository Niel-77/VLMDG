"""Provider-agnostic VLM interface plus concrete backends.

``get_provider(name, config)`` is the entry point. ``"mock"`` is a deterministic
offline provider that lets the whole loop run with no API key;
``"anthropic"``, ``"openai"`` and ``"tamu"`` call hosted models. ``"local"`` is
a declared stub to be filled in later.
"""

from vlmdg.vlm.base import VLMProvider, VLMResponse, VLMUsage, encode_image
from vlmdg.vlm.factory import available_providers, get_provider
from vlmdg.vlm.mock_provider import MockVLMProvider

__all__ = [
    "VLMProvider",
    "VLMResponse",
    "VLMUsage",
    "encode_image",
    "get_provider",
    "available_providers",
    "MockVLMProvider",
]
