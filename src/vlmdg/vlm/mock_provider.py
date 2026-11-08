"""Deterministic offline VLM provider.

``MockVLMProvider`` returns a fixed, well-formed rule so the full rule-generation loop -
prompt construction, response parsing, the execution gate, the admissibility
gate, rule registration and reheating - can be exercised end to end with no API
key and no network. It is *not* a model of VLM behaviour; it always emits an
admissible bracing operator.

To show that search diagnostics reach the generator, the mock inspects the
prompt: when a specific member is flagged ``VIOLATED`` it bakes that member's
joint ids into the generated code; otherwise the generated rule targets the
longest member at run time.
"""

from __future__ import annotations

import re
from typing import Sequence

from vlmdg.config import VLMConfig
from vlmdg.vlm.base import ImageInput, VLMProvider, VLMResponse, VLMUsage

_VIOLATED_RE = re.compile(r"member\s+(\d+)-(\d+)\s*:\s*g=[^\n]*VIOLATED", re.IGNORECASE)

_RULE_TEMPLATE = '''```python
def rule(design, rng):
    """Sub-triangulate the most critical member by tying its ends to a nearby joint.

    No new joint is introduced (so joint spacing cannot be violated); the rule
    adds the one or two missing members that connect the endpoints of the target
    member to the closest existing joint, creating a stiffer local triangle.
    """
    d = design  # already a fresh copy supplied by the caller
    members = d.members()
    if not members:
        return d

    target = {target_expr}
    if target is None or not d.has_member(*target):
        target = max(members, key=lambda mb: d.member_length(*mb))
    i, j = target

    (xi, yi), (xj, yj) = d.coords(i), d.coords(j)
    mx, my = (xi + xj) / 2.0, (yi + yj) / 2.0
    area = d.area(i, j)

    candidates = [n for n in d.joints() if n not in (i, j)]
    candidates.sort(key=lambda n: math.hypot(d.coords(n)[0] - mx, d.coords(n)[1] - my))
    for k in candidates:
        if d.has_member(i, k) and d.has_member(j, k):
            continue
        added = False
        if not d.has_member(i, k):
            d.add_member(i, k, area=area)
            added = True
        if not d.has_member(j, k):
            d.add_member(j, k, area=area)
            added = True
        if added:
            return d
    return d
```'''


class MockVLMProvider(VLMProvider):
    """Offline provider that always proposes an admissible bracing rule."""

    name = "mock"

    def __init__(self, config: VLMConfig | None = None) -> None:
        self.config = config or VLMConfig(provider="mock")

    def generate(
        self,
        system: str,
        prompt: str,
        images: Sequence[ImageInput] = (),
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> VLMResponse:
        """Return the canned rule, targeting a flagged member when the prompt has one."""
        match = _VIOLATED_RE.search(prompt)
        if match:
            target_expr = f"({int(match.group(1))}, {int(match.group(2))})"
        else:
            target_expr = "None"

        text = (
            "The critical member is carrying too much axial force. I propose a new "
            "rule that erects a triangulated apex over it to split the load path.\n\n"
            + _RULE_TEMPLATE.format(target_expr=target_expr)
        )
        usage = VLMUsage(
            input_tokens=len(system.split()) + len(prompt.split()),
            output_tokens=len(text.split()),
            cost_usd=0.0,
            requests=1,
        )
        return VLMResponse(text=text, model="mock", usage=usage, raw={"prompt": prompt})
