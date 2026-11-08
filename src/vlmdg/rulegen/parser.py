"""Extract the Python rule body from a VLM response."""

from __future__ import annotations

import re

_FENCED_PY = re.compile(r"```(?:python|py)\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)
_FENCED_ANY = re.compile(r"```\s*\n(.*?)```", re.DOTALL)


def extract_code_block(text: str) -> str | None:
    """Return the first fenced code block that defines ``rule``.

    Preference order: a ```python fence, then any ``` fence, then - as a last
    resort - the whole response if it contains a ``def rule(`` line.
    """
    for pattern in (_FENCED_PY, _FENCED_ANY):
        for match in pattern.finditer(text):
            block = match.group(1).strip()
            if "def rule(" in block:
                return block
        # fall back to the first block of this kind even without the signature
        first = pattern.search(text)
        if first and first.group(1).strip():
            candidate = first.group(1).strip()
            if "def rule(" in candidate:
                return candidate

    if "def rule(" in text:
        return text.strip()
    return None
