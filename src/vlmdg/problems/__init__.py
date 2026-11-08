"""Test problems. Each gives a starting design, its spec, and its material."""

from typing import Callable, Dict, Tuple

from vlmdg.config import DesignSpec, Material
from vlmdg.design.truss import TrussDesign
from vlmdg.problems.bridge import build_bridge
from vlmdg.problems.cantilever import build_cantilever
from vlmdg.problems.ten_bar import build_ten_bar

Problem = Tuple[TrussDesign, DesignSpec]

#: Registry of ``name -> builder`` for the CLI and experiment runner.
PROBLEMS: Dict[str, Callable[[], Problem]] = {
    "ten_bar": build_ten_bar,
    "cantilever": build_cantilever,
    "bridge": build_bridge,
}

#: The :class:`~vlmdg.config.Material` each problem is posed in.
PROBLEM_MATERIALS: Dict[str, Callable[[], Material]] = {
    "ten_bar": Material.ten_bar,
    "cantilever": Material,
    "bridge": Material.ten_bar,
}

#: Default problem for the CLI / :class:`~vlmdg.config.RunConfig`.
DEFAULT_PROBLEM = "ten_bar"


def get_problem(name: str) -> Problem:
    """Return ``(design0, spec)`` for a registered problem name."""
    if name not in PROBLEMS:
        raise KeyError(f"unknown problem {name!r}; available: {sorted(PROBLEMS)}")
    return PROBLEMS[name]()


def get_material(name: str) -> Material:
    """Return the :class:`~vlmdg.config.Material` a registered problem is posed in."""
    if name not in PROBLEM_MATERIALS:
        raise KeyError(f"unknown problem {name!r}; available: {sorted(PROBLEM_MATERIALS)}")
    return PROBLEM_MATERIALS[name]()


__all__ = [
    "PROBLEMS",
    "PROBLEM_MATERIALS",
    "DEFAULT_PROBLEM",
    "Problem",
    "get_problem",
    "get_material",
    "build_ten_bar",
    "build_cantilever",
    "build_bridge",
]
