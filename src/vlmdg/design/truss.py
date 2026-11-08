"""Labeled-graph representation of a planar pin-jointed truss.

A :class:`TrussDesign` is a thin, mutation-friendly wrapper around a
``networkx.Graph``:

* **Nodes** are joints keyed by an integer id, with attributes
  ``x``, ``y`` (position), ``support`` (a ``(bool, bool)`` pair marking
  constrained x / y reaction DOFs) and ``load`` (an applied ``(Fx, Fy)`` force).
* **Edges** are members with attributes ``area`` (cross-sectional area) and
  ``label`` (an optional non-terminal marker used by grammar rules).

The class deliberately keeps geometry and topology in one object so that a
grammar rule is simply a function ``TrussDesign -> TrussDesign``.
"""

from __future__ import annotations

import math
from typing import Dict, Iterable, Iterator, List, Sequence, Tuple

import networkx as nx
import numpy as np

Coord = Tuple[float, float]
Dof = Tuple[bool, bool]
Member = Tuple[int, int]


def _key(i: int, j: int) -> Member:
    """Return a canonical ordered member key so ``(i, j)`` and ``(j, i)`` match."""
    return (i, j) if i <= j else (j, i)


class TrussDesign:
    """A planar truss as a labeled graph with joint geometry and boundary conditions."""

    def __init__(self, graph: nx.Graph | None = None, next_id: int | None = None) -> None:
        self.graph: nx.Graph = graph if graph is not None else nx.Graph()
        if next_id is None:
            next_id = max(self.graph.nodes, default=-1) + 1
        self._next_id: int = next_id

    # builders
    @classmethod
    def from_spec(
        cls,
        joints: Dict[int, Coord],
        members: Iterable[Member],
        supports: Dict[int, Dof] | None = None,
        loads: Dict[int, Coord] | None = None,
        areas: Dict[Member, float] | None = None,
        default_area: float = 1.0,
    ) -> "TrussDesign":
        """Build a design from plain dictionaries.

        Parameters
        ----------
        joints:
            Mapping ``joint_id -> (x, y)``.
        members:
            Iterable of ``(i, j)`` joint-id pairs.
        supports:
            Mapping ``joint_id -> (fix_x, fix_y)``; missing joints are free.
        loads:
            Mapping ``joint_id -> (Fx, Fy)``; missing joints are unloaded.
        areas:
            Mapping ``(i, j) -> area``; missing members use ``default_area``.
        """
        supports = supports or {}
        loads = loads or {}
        areas = areas or {}
        design = cls()
        for jid, (x, y) in joints.items():
            design.add_joint(
                x, y,
                support=supports.get(jid, (False, False)),
                load=loads.get(jid, (0.0, 0.0)),
                joint_id=jid,
            )
        for i, j in members:
            design.add_member(i, j, area=areas.get(_key(i, j), default_area))
        return design

    # joint access
    def add_joint(
        self,
        x: float,
        y: float,
        support: Dof = (False, False),
        load: Coord = (0.0, 0.0),
        joint_id: int | None = None,
    ) -> int:
        """Add a joint and return its id (auto-assigned when ``joint_id`` is ``None``)."""
        if joint_id is None:
            joint_id = self._next_id
        self.graph.add_node(
            joint_id,
            x=float(x),
            y=float(y),
            support=(bool(support[0]), bool(support[1])),
            load=(float(load[0]), float(load[1])),
        )
        self._next_id = max(self._next_id, joint_id + 1)
        return joint_id

    def remove_joint(self, joint_id: int) -> None:
        """Remove a joint and all incident members."""
        self.graph.remove_node(joint_id)

    def move_joint(self, joint_id: int, x: float, y: float) -> None:
        """Set the position of an existing joint."""
        self.graph.nodes[joint_id]["x"] = float(x)
        self.graph.nodes[joint_id]["y"] = float(y)

    def set_support(self, joint_id: int, support: Dof) -> None:
        """Set the constrained-DOF flags of a joint."""
        self.graph.nodes[joint_id]["support"] = (bool(support[0]), bool(support[1]))

    def set_load(self, joint_id: int, load: Coord) -> None:
        """Set the applied force vector of a joint."""
        self.graph.nodes[joint_id]["load"] = (float(load[0]), float(load[1]))

    # member access
    def add_member(self, i: int, j: int, area: float = 1.0, label: str | None = None) -> Member:
        """Add (or overwrite) a member between joints ``i`` and ``j``; return its key."""
        if i == j:
            raise ValueError(f"self-loop member requested at joint {i}")
        if i not in self.graph or j not in self.graph:
            raise KeyError(f"member ({i}, {j}) references a missing joint")
        self.graph.add_edge(i, j, area=float(area), label=label)
        return _key(i, j)

    def remove_member(self, i: int, j: int) -> None:
        """Remove the member between joints ``i`` and ``j``."""
        self.graph.remove_edge(i, j)

    def set_area(self, i: int, j: int, area: float) -> None:
        """Set the cross-sectional area of a member."""
        self.graph.edges[i, j]["area"] = float(area)

    # introspection
    @property
    def node_order(self) -> List[int]:
        """Joint ids in a stable ascending order (used for DOF indexing)."""
        return sorted(self.graph.nodes)

    @property
    def n_joints(self) -> int:
        """Number of joints ``j``."""
        return self.graph.number_of_nodes()

    @property
    def n_members(self) -> int:
        """Number of members ``b``."""
        return self.graph.number_of_edges()

    @property
    def n_reactions(self) -> int:
        """Number of constrained reaction DOFs ``r``."""
        total = 0
        for _, data in self.graph.nodes(data=True):
            fx, fy = data["support"]
            total += int(fx) + int(fy)
        return total

    def joints(self) -> List[int]:
        """Return the list of joint ids."""
        return list(self.graph.nodes)

    def members(self) -> List[Member]:
        """Return the list of canonical member keys."""
        return [_key(i, j) for i, j in self.graph.edges]

    def coords(self, joint_id: int) -> Coord:
        """Return the ``(x, y)`` position of a joint."""
        data = self.graph.nodes[joint_id]
        return (data["x"], data["y"])

    def support(self, joint_id: int) -> Dof:
        """Return the constrained-DOF flags of a joint."""
        return self.graph.nodes[joint_id]["support"]

    def load(self, joint_id: int) -> Coord:
        """Return the applied force vector of a joint."""
        return self.graph.nodes[joint_id]["load"]

    def area(self, i: int, j: int) -> float:
        """Return the cross-sectional area of a member."""
        return self.graph.edges[i, j]["area"]

    def label(self, i: int, j: int) -> str | None:
        """Return the label attribute of a member."""
        return self.graph.edges[i, j].get("label")

    def has_member(self, i: int, j: int) -> bool:
        """Return ``True`` when a member connects joints ``i`` and ``j``."""
        return self.graph.has_edge(i, j)

    def coords_array(self) -> np.ndarray:
        """Return joint positions as an ``(n_joints, 2)`` array in :pyattr:`node_order`."""
        return np.array([self.coords(n) for n in self.node_order], dtype=float)

    def member_length(self, i: int, j: int) -> float:
        """Return the Euclidean length of a member."""
        (xi, yi), (xj, yj) = self.coords(i), self.coords(j)
        return math.hypot(xj - xi, yj - yi)

    def total_length(self) -> float:
        """Return the summed length of all members."""
        return sum(self.member_length(i, j) for i, j in self.members())

    def loaded_joints(self) -> List[int]:
        """Joint ids that carry a non-zero applied force."""
        return [n for n in self.graph.nodes if any(abs(v) > 0.0 for v in self.load(n))]

    def supported_joints(self) -> List[int]:
        """Joint ids that have at least one constrained reaction DOF."""
        return [n for n in self.graph.nodes if any(self.support(n))]

    def neighbors(self, joint_id: int) -> Iterator[int]:
        """Iterate over joints directly connected to ``joint_id``."""
        return iter(self.graph.neighbors(joint_id))

    def degree(self, joint_id: int) -> int:
        """Return the number of members incident to ``joint_id``."""
        return self.graph.degree(joint_id)

    def is_connected(self) -> bool:
        """Return ``True`` when the truss forms a single connected component."""
        return self.n_joints > 0 and nx.is_connected(self.graph)

    # copying
    def copy(self) -> "TrussDesign":
        """Return a deep copy safe to mutate independently."""
        return TrussDesign(self.graph.copy(), next_id=self._next_id)

    # comparison
    def structural_key(self, tol: float = 1e-6) -> Tuple:
        """Return a hashable canonical fingerprint of geometry, topology and labels.

        Joint positions are quantized to ``tol`` and joints are matched by
        rounded position (not id), so two designs that differ only by relabeling
        compare equal.
        """
        digits = max(0, int(round(-math.log10(tol)))) if tol > 0 else 9

        def q(value: float) -> float:
            return round(value, digits) + 0.0  # normalize -0.0

        pos = {n: (q(self.coords(n)[0]), q(self.coords(n)[1])) for n in self.graph.nodes}
        nodes = tuple(sorted(
            (p, self.support(n), (q(self.load(n)[0]), q(self.load(n)[1])))
            for n, p in pos.items()
        ))
        edges = tuple(sorted(
            (
                tuple(sorted((pos[i], pos[j]))),
                q(self.area(i, j)),
                self.label(i, j),
            )
            for i, j in self.graph.edges
        ))
        return (nodes, edges)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, TrussDesign):
            return NotImplemented
        return self.structural_key() == other.structural_key()

    def __hash__(self) -> int:
        return hash(self.structural_key())

    def __repr__(self) -> str:
        return (
            f"TrussDesign(j={self.n_joints}, b={self.n_members}, "
            f"r={self.n_reactions}, L={self.total_length():.3f})"
        )

    # convenience
    def nearest_joints(
        self, x: float, y: float, k: int = 1, exclude: Sequence[int] = ()
    ) -> List[int]:
        """Return up to ``k`` joint ids closest to ``(x, y)``, nearest first."""
        excluded = set(exclude)
        ranked = sorted(
            (n for n in self.graph.nodes if n not in excluded),
            key=lambda n: math.hypot(self.coords(n)[0] - x, self.coords(n)[1] - y),
        )
        return ranked[:k]
