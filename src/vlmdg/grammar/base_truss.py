"""Base planar-truss grammar ``G_0``.

This is a documented reconstruction, in the style of the shape-annealing truss
grammar of Cagan & Mitchell (1993) and Reddy & Cagan (1993, 1995), of a minimal
set of boundary-preserving operators:

======================  =========================================================
Rule                    Effect
======================  =========================================================
``add_triangle``        Erect a new joint on a member, braced by two new members.
``split_member``        Insert a mid-joint on a member and brace it to a neighbour.
``connect_joints``      Add a member between two visible, unconnected joints.
``remove_member``       Delete a member if the truss stays connected.
``move_joint``          Parametric: relocate a free (non-support, non-load) joint.
======================  =========================================================

``add_triangle``, ``split_member`` and ``connect_joints`` enlarge the design,
``remove_member`` shrinks it, and ``move_joint`` changes only the geometry. No
rule sets member areas: the sizer sets them for every candidate. Every rule keeps
supports and loaded joints untouched, so an application that also passes the
admissibility gate yields a design in ``L(G_0)``.
"""

from __future__ import annotations

import itertools
import math
import random
from typing import Any, List, Tuple

from vlmdg.config import DesignSpec, Material
from vlmdg.design.truss import TrussDesign
from vlmdg.grammar.geometry import perpendicular_apex, segments_properly_intersect
from vlmdg.grammar.registry import Grammar
from vlmdg.grammar.rule import Rule

Member = Tuple[int, int]


class _BaseTrussRule(Rule):
    """Common constructor: every base rule needs the spec and material bounds."""

    origin = "base"

    def __init__(self, spec: DesignSpec, material: Material) -> None:
        self.spec = spec
        self.material = material

    # shared geometry predicates
    def _spacing_ok(self, design: TrussDesign, x: float, y: float, ignore: set[int]) -> bool:
        for n in design.joints():
            if n in ignore:
                continue
            px, py = design.coords(n)
            if math.hypot(px - x, py - y) < self.spec.min_joint_spacing:
                return False
        return True

    def _crosses_existing(
        self, design: TrussDesign, p: Tuple[float, float], q: Tuple[float, float],
        skip: set[Member],
    ) -> bool:
        for i, j in design.members():
            if (i, j) in skip:
                continue
            if segments_properly_intersect(p, q, design.coords(i), design.coords(j)):
                return True
        return False


class AddTriangleRule(_BaseTrussRule):
    """Erect a new apex joint over an existing member, forming a triangle."""

    name = "add_triangle"
    description = (
        "Pick a member (i, j); add a new joint k offset perpendicular to the "
        "member midpoint; connect k to i and j. Adds one joint and two members."
    )

    def find_matches(self, design: TrussDesign) -> List[Any]:
        if design.n_joints >= self.spec.max_joints:
            return []
        if design.n_members + 2 > self.spec.max_members:
            return []
        return [
            (i, j) for i, j in design.members()
            if design.member_length(i, j) >= 2.0 * self.spec.min_member_length
        ]

    def apply_match(self, design: TrussDesign, match: Any, rng: random.Random) -> TrussDesign:
        i, j = match
        (xi, yi), (xj, yj) = design.coords(i), design.coords(j)
        base_len = math.hypot(xj - xi, yj - yi)
        area = design.area(i, j)
        for offset in (0.45 * base_len, 0.3 * base_len, 1.5 * self.spec.min_joint_spacing):
            for side in rng.sample((1, -1), 2):
                ax, ay = perpendicular_apex(xi, yi, xj, yj, offset, side)
                if not self.spec.in_domain(ax, ay):
                    continue
                if not self._spacing_ok(design, ax, ay, ignore=set()):
                    continue
                k = design.add_joint(ax, ay)
                design.add_member(i, k, area=area)
                design.add_member(j, k, area=area)
                return design
        return design  # no admissible apex found -> no-op


class SplitMemberRule(_BaseTrussRule):
    """Insert a mid-joint on a member and brace it to the nearest other joint."""

    name = "split_member"
    description = (
        "Pick a member (i, j); insert a joint k at its midpoint, replacing (i, j) "
        "with (i, k) and (k, j); brace k with a member to its nearest other joint."
    )

    def find_matches(self, design: TrussDesign) -> List[Any]:
        if design.n_joints >= self.spec.max_joints:
            return []
        return [
            (i, j) for i, j in design.members()
            if design.member_length(i, j) >= 4.0 * self.spec.min_member_length
        ]

    def apply_match(self, design: TrussDesign, match: Any, rng: random.Random) -> TrussDesign:
        i, j = match
        (xi, yi), (xj, yj) = design.coords(i), design.coords(j)
        mx, my = (xi + xj) / 2.0, (yi + yj) / 2.0
        if not self.spec.in_domain(mx, my) or not self._spacing_ok(design, mx, my, set()):
            return design
        area = design.area(i, j)
        k = design.add_joint(mx, my)
        design.remove_member(i, j)
        design.add_member(i, k, area=area)
        design.add_member(k, j, area=area)
        brace = design.nearest_joints(mx, my, k=1, exclude=(i, j, k))
        if brace and not self._crosses_existing(
            design, (mx, my), design.coords(brace[0]), skip=set()
        ):
            design.add_member(k, brace[0], area=area)
        return design


class ConnectJointsRule(_BaseTrussRule):
    """Add a member between two currently unconnected, mutually visible joints."""

    name = "connect_joints"
    description = (
        "Pick two joints that are not connected and whose connecting segment does "
        "not cross an existing member; add a member between them."
    )

    def find_matches(self, design: TrussDesign) -> List[Any]:
        if design.n_members >= self.spec.max_members:
            return []
        out: List[Any] = []
        for i, k in itertools.combinations(design.node_order, 2):
            if design.has_member(i, k):
                continue
            length = math.hypot(*(
                v - w for v, w in zip(design.coords(i), design.coords(k))
            ))
            if not self.spec.min_member_length <= length <= self.spec.max_member_length:
                continue
            if self._crosses_existing(design, design.coords(i), design.coords(k), skip=set()):
                continue
            out.append((i, k))
        return out

    def apply_match(self, design: TrussDesign, match: Any, rng: random.Random) -> TrussDesign:
        i, k = match
        areas = [design.area(a, b) for a, b in design.members()] or [1.0]
        areas.sort()
        median = areas[len(areas) // 2]
        design.add_member(i, k, area=median)
        return design


class RemoveMemberRule(_BaseTrussRule):
    """Delete a member when the truss stays connected (the gate rejects mechanisms)."""

    name = "remove_member"
    description = (
        "Pick a member and delete it, provided the remaining truss is still a "
        "single connected component."
    )

    def find_matches(self, design: TrussDesign) -> List[Any]:
        matches: List[Any] = []
        for i, j in design.members():
            trial = design.copy()
            trial.remove_member(i, j)
            if trial.n_members == 0:
                continue
            if trial.is_connected() and not any(
                trial.degree(n) == 0 for n in trial.joints()
            ):
                matches.append((i, j))
        return matches

    def apply_match(self, design: TrussDesign, match: Any, rng: random.Random) -> TrussDesign:
        i, j = match
        design.remove_member(i, j)
        # Drop a joint that the removal left dangling with no members.
        for n in list(design.joints()):
            if design.degree(n) == 0 and n not in self.spec.protected_joints():
                design.remove_joint(n)
        return design


class MoveJointRule(_BaseTrussRule):
    """Parametric: relocate a free joint by a bounded random step, clamped to the domain."""

    name = "move_joint"
    description = (
        "Pick a joint that is neither a support nor loaded; translate it by a "
        "random vector, clamped to the design domain."
    )
    changes_topology = False

    def find_matches(self, design: TrussDesign) -> List[Any]:
        protected = self.spec.protected_joints()
        return [n for n in design.joints() if n not in protected]

    def apply_match(self, design: TrussDesign, match: Any, rng: random.Random) -> TrussDesign:
        n = match
        x, y = design.coords(n)
        (lo_x, hi_x), (lo_y, hi_y) = self.spec.x_bounds, self.spec.y_bounds
        step = 0.15 * math.hypot(hi_x - lo_x, hi_y - lo_y)
        for _ in range(8):
            nx = min(max(x + rng.uniform(-step, step), lo_x), hi_x)
            ny = min(max(y + rng.uniform(-step, step), lo_y), hi_y)
            if self._spacing_ok(design, nx, ny, ignore={n}):
                design.move_joint(n, nx, ny)
                return design
        return design


#: Rule classes that make up the base grammar, in registration order.
BASE_TRUSS_RULES = (
    AddTriangleRule,
    SplitMemberRule,
    ConnectJointsRule,
    RemoveMemberRule,
    MoveJointRule,
)


def base_truss_grammar(spec: DesignSpec, material: Material) -> Grammar:
    """Construct ``G_0`` bound to a specific design spec and material."""
    return Grammar([cls(spec, material) for cls in BASE_TRUSS_RULES])
