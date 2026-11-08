"""Grammar rules. A rule takes a design and returns a changed copy.

Rule is the base class of the five starting rules: find_matches finds where
the rule can apply, and apply_match makes the change there.

FunctionRule wraps a plain function f(design, rng) -> design as a rule. Rules
written by the VLM are run this way.

Rules never change their input; the design is copied first.
"""

from __future__ import annotations

import abc
import random
from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional

from vlmdg.design.truss import TrussDesign


@dataclass
class RuleApplication:
    """Record of a single rule application, for logging and rule-utility stats."""

    rule_name: str
    applied: bool
    match: Any = None
    note: str = ""


class Rule(abc.ABC):
    """Abstract grammar rule.

    Subclasses set :pyattr:`name` / :pyattr:`description` and implement
    :meth:`find_matches` and :meth:`apply_match`.
    """

    #: Short identifier, unique within a :class:`~vlmdg.grammar.registry.Grammar`.
    name: str = "rule"
    #: One-line natural-language description (also shown to the VLM).
    description: str = ""
    #: ``True`` for topology/geometry-changing rules, ``False`` for pure sizing.
    changes_topology: bool = True
    #: Provenance tag: ``"base"`` for G_0 rules, ``"vlm"`` / ``"random"`` otherwise.
    origin: str = "base"

    @abc.abstractmethod
    def find_matches(self, design: TrussDesign) -> List[Any]:
        """Return every left-hand-side match in ``design`` (opaque match tokens)."""

    @abc.abstractmethod
    def apply_match(self, design: TrussDesign, match: Any, rng: random.Random) -> TrussDesign:
        """Apply the rewrite for ``match`` to a **copy** of ``design`` and return it."""

    def is_applicable(self, design: TrussDesign) -> bool:
        """Return ``True`` when at least one LHS match exists."""
        return bool(self.find_matches(design))

    def __call__(
        self, design: TrussDesign, rng: random.Random
    ) -> Optional[TrussDesign]:
        """Pick a uniformly random match and apply it; return ``None`` if no match."""
        matches = self.find_matches(design)
        if not matches:
            return None
        match = rng.choice(matches)
        return self.apply_match(design.copy(), match, rng)

    def __repr__(self) -> str:
        return f"<Rule {self.name!r} origin={self.origin}>"


@dataclass
class FunctionRule(Rule):
    """Wrap a plain ``callable(design, rng) -> TrussDesign`` as a grammar rule.

    Parameters
    ----------
    name, description:
        Metadata (shown in diagnostics and to the VLM).
    fn:
        The operator. It receives a *fresh copy* of the design and must return a
        :class:`TrussDesign`. Returning ``None`` or raising is treated as
        "not applicable" by :meth:`__call__` but surfaced by the execution gate.
    source:
        Optional source code, retained for logging when ``fn`` was compiled from
        a VLM response.
    """

    name: str = "function_rule"
    description: str = ""
    fn: Callable[[TrussDesign, random.Random], TrussDesign] | None = None
    source: str = ""
    origin: str = "vlm"
    changes_topology: bool = True
    metadata: dict = field(default_factory=dict)

    def find_matches(self, design: TrussDesign) -> List[Any]:
        """A function rule is opaque; it always reports a single trivial match."""
        return [None]

    def apply_match(
        self, design: TrussDesign, match: Any, rng: random.Random
    ) -> TrussDesign:
        """Invoke the wrapped function on ``design`` (already a copy)."""
        if self.fn is None:
            raise RuntimeError(f"FunctionRule {self.name!r} has no callable")
        result = self.fn(design, rng)
        if not isinstance(result, TrussDesign):
            raise TypeError(
                f"rule {self.name!r} returned {type(result).__name__}, expected TrussDesign"
            )
        return result

    def __call__(
        self, design: TrussDesign, rng: random.Random
    ) -> Optional[TrussDesign]:
        return self.apply_match(design.copy(), None, rng)
