"""Compile a VLM-generated rule and run the execution-only gate.

The execution gate checks *only* that the generated
function runs and returns a well-formed design (right type, non-empty, finite
coordinates). It deliberately learns nothing about structural correctness - that
is the job of the separate admissibility gate. On failure the code is sent back
to the model for repair, up to a fixed number of attempts.

The namespace given to the generated code is restricted (no ``open``, no
arbitrary imports) but this is a guard rail, not a security sandbox: do not run
untrusted model output on a machine that matters without OS-level isolation.
"""

from __future__ import annotations

import itertools
import math
import random
import traceback
from dataclasses import dataclass, field
from typing import Callable, List, Sequence

import networkx as nx
import numpy as np

from vlmdg.design.truss import TrussDesign
from vlmdg.grammar.rule import FunctionRule

RepairFn = Callable[[str, str], "str | None"]

_ALLOWED_IMPORTS = {"math", "random", "itertools", "numpy", "networkx"}

_SAFE_BUILTIN_NAMES = (
    "abs", "all", "any", "bool", "dict", "divmod", "enumerate", "filter", "float",
    "frozenset", "getattr", "hasattr", "int", "isinstance", "issubclass", "iter",
    "len", "list", "map", "max", "min", "next", "pow", "print", "range", "repr",
    "reversed", "round", "set", "setattr", "slice", "sorted", "str", "sum", "tuple",
    "zip", "Exception", "ValueError", "TypeError", "KeyError", "IndexError",
    "RuntimeError", "ZeroDivisionError", "ArithmeticError", "StopIteration",
    "True", "False", "None",
)


class RuleCompileError(RuntimeError):
    """Raised when a generated rule cannot be compiled into a callable."""


@dataclass
class ExecutionGateResult:
    """Outcome of :func:`run_execution_gate`."""

    rule: FunctionRule | None
    attempts: int
    passed: bool
    transcript: List[dict] = field(default_factory=list)

    @property
    def final_error(self) -> str:
        """The last recorded error message, or ``""`` on success."""
        for entry in reversed(self.transcript):
            if entry.get("error"):
                return entry["error"]
        return ""


def _guarded_import(name, globals_=None, locals_=None, fromlist=(), level=0):
    """``__import__`` replacement limited to a small whitelist."""
    root = name.split(".", 1)[0]
    if root not in _ALLOWED_IMPORTS:
        raise ImportError(f"import of {name!r} is not allowed in a generated rule")
    return __import__(name, globals_, locals_, fromlist, level)


def _build_namespace() -> dict:
    """Create the restricted global namespace for exec-ing a generated rule."""
    import builtins  # pylint: disable=import-outside-toplevel

    safe_builtins = {
        n: getattr(builtins, n) for n in _SAFE_BUILTIN_NAMES if hasattr(builtins, n)
    }
    safe_builtins["__import__"] = _guarded_import
    return {
        "__builtins__": safe_builtins,
        "math": math,
        "random": random,
        "itertools": itertools,
        "np": np,
        "numpy": np,
        "nx": nx,
        "networkx": nx,
        "TrussDesign": TrussDesign,
    }


def rule_description(fn: Callable) -> str:
    """Return the first paragraph of a rule's docstring as one line.

    The description is listed in later prompts, so the model can see what each
    earlier generated rule does instead of a generic placeholder.
    """
    doc = (getattr(fn, "__doc__", None) or "").strip()
    if not doc:
        return "VLM-generated rule (no docstring)"
    first_paragraph = doc.split("\n\n", 1)[0]
    return " ".join(first_paragraph.split())


def compile_rule(code: str, name: str, description: str = "") -> FunctionRule:
    """Compile ``code`` and wrap its ``rule`` function as a :class:`FunctionRule`."""
    namespace = _build_namespace()
    try:
        compiled = compile(code, f"<vlm_rule:{name}>", "exec")
        exec(compiled, namespace)  # pylint: disable=exec-used
    except SyntaxError as exc:
        raise RuleCompileError(f"syntax error: {exc}") from exc
    except Exception as exc:  # pylint: disable=broad-except
        raise RuleCompileError(f"error while defining rule: {exc}") from exc

    fn = namespace.get("rule")
    if not callable(fn):
        raise RuleCompileError("generated code does not define a callable `rule`")

    return FunctionRule(
        name=name,
        description=description or rule_description(fn),
        fn=fn,
        source=code,
        origin="vlm",
    )


def _well_formed(design: object) -> str:
    """Return an error string if ``design`` is not a usable TrussDesign, else ``""``."""
    if not isinstance(design, TrussDesign):
        return f"rule returned {type(design).__name__}, expected TrussDesign"
    if design.n_joints == 0 or design.n_members == 0:
        return "rule returned an empty design"
    for n in design.joints():
        x, y = design.coords(n)
        if not (math.isfinite(x) and math.isfinite(y)):
            return f"joint {n} has non-finite coordinates"
    for i, j in design.members():
        if not math.isfinite(design.area(i, j)) or design.area(i, j) <= 0.0:
            return f"member ({i}, {j}) has a non-positive or non-finite area"
    return ""


def run_execution_gate(
    code: str,
    name: str,
    probes: Sequence[TrussDesign],
    *,
    max_repairs: int = 3,
    repair_fn: RepairFn | None = None,
    seed: int = 0,
) -> ExecutionGateResult:
    """Compile and run ``code`` on each probe design, repairing on failure.

    Parameters
    ----------
    code:
        The initial generated rule body.
    name:
        Name to give the compiled :class:`FunctionRule`.
    probes:
        Designs the rule must run on without error (typically the incumbent best
        plus a couple of variations).
    max_repairs:
        Maximum number of repair round-trips to ``repair_fn``.
    repair_fn:
        ``(code, error) -> new_code`` (or ``None`` to give up). Usually a thin
        wrapper around a VLM call.
    """
    transcript: List[dict] = []
    rng = random.Random(seed)
    current_code = code

    for attempt in range(max_repairs + 1):
        entry: dict = {"attempt": attempt, "code": current_code}
        error = ""
        try:
            rule = compile_rule(current_code, name)
            for idx, probe in enumerate(probes):
                result = rule(probe.copy(), rng)
                problem = _well_formed(result)
                if problem:
                    error = f"probe {idx}: {problem}"
                    break
        except RuleCompileError as exc:
            error = str(exc)
        except Exception:  # pylint: disable=broad-except
            error = "runtime error:\n" + traceback.format_exc(limit=4)

        entry["error"] = error
        transcript.append(entry)

        if not error:
            return ExecutionGateResult(
                rule=compile_rule(current_code, name),
                attempts=attempt + 1,
                passed=True,
                transcript=transcript,
            )

        if attempt >= max_repairs or repair_fn is None:
            break
        repaired = repair_fn(current_code, error)
        if not repaired:
            break
        current_code = repaired

    return ExecutionGateResult(
        rule=None, attempts=len(transcript), passed=False, transcript=transcript
    )
