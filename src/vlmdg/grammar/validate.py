"""Admissibility gate: checks that a design produced by a rule is a valid truss.

admissibility_gate runs every check and lists the reasons a design fails:

  closure      supports and loads unchanged; joint and bar counts within limits
  connected    one connected structure
  geometry     joints inside the domain, no coincident joints, no duplicate or
               zero-length bars, bar lengths within limits, positive areas
  planar       no crossing bars and no joint on a bar it does not end
               (only if the problem requires it)
  stable       enough bars (b + r >= 2j for b bars, r support reactions, j
               joints) and a full-rank equilibrium matrix, so the truss is
               not a mechanism
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np

from vlmdg.config import DesignSpec
from vlmdg.design.truss import TrussDesign

Member = Tuple[int, int]


@dataclass
class GateReport:
    """Result of running the admissibility gate on one design."""

    passed: bool
    reasons: List[str] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)

    def __bool__(self) -> bool:
        return self.passed


# geometry
def check_connectivity(design: TrussDesign) -> List[str]:
    """Return failure reasons for the single-connected-component requirement."""
    if design.n_joints == 0:
        return ["design has no joints"]
    if not design.is_connected():
        return ["design is not a single connected component"]
    isolated = [n for n in design.joints() if design.degree(n) == 0]
    if isolated:
        return [f"isolated joints: {sorted(isolated)}"]
    return []


def check_areas(design: TrussDesign) -> List[str]:
    """Return failure reasons for member areas that are not finite and positive.

    The upper and lower area bounds are not checked here: rules copy areas from
    existing members, and the sizer sets every area within the bounds afterwards.
    """
    return [
        f"member ({i}, {j}) has a non-positive or non-finite area"
        for i, j in design.members()
        if not (math.isfinite(design.area(i, j)) and design.area(i, j) > 0.0)
    ]


def check_geometry(design: TrussDesign, spec: DesignSpec) -> List[str]:
    """Return failure reasons for domain / spacing / member-length validity."""
    reasons: List[str] = []
    coords = {n: design.coords(n) for n in design.joints()}

    for n, (x, y) in coords.items():
        if not spec.in_domain(x, y):
            reasons.append(f"joint {n} at ({x:.3g}, {y:.3g}) is outside the design domain")

    for (na, (xa, ya)), (nb, (xb, yb)) in itertools.combinations(sorted(coords.items()), 2):
        if math.hypot(xa - xb, ya - yb) < spec.min_joint_spacing:
            reasons.append(f"joints {na} and {nb} are closer than min_joint_spacing")

    seen: set[Member] = set()
    for i, j in design.members():
        if (i, j) in seen:
            reasons.append(f"duplicate member ({i}, {j})")
        seen.add((i, j))
        length = design.member_length(i, j)
        if length < spec.min_member_length:
            reasons.append(f"member ({i}, {j}) shorter than min_member_length")
        if length > spec.max_member_length:
            reasons.append(f"member ({i}, {j}) longer than max_member_length")
    return reasons


def check_language_closure(design: TrussDesign, spec: DesignSpec) -> List[str]:
    """Return failure reasons for the ``L(G)`` membership requirement."""
    reasons: List[str] = []
    tol = spec.position_tolerance

    for jid, (x, y) in spec.fixed_joints.items():
        if jid not in design.graph:
            reasons.append(f"protected joint {jid} was deleted")
            continue
        cx, cy = design.coords(jid)
        if math.hypot(cx - x, cy - y) > max(tol, 1e-6):
            reasons.append(f"protected joint {jid} was moved from its prescribed position")

    for jid, dof in spec.supports.items():
        if jid in design.graph and design.support(jid) != tuple(dof):
            reasons.append(f"support condition at joint {jid} was altered")

    for jid, force in spec.loads.items():
        if jid not in design.graph:
            reasons.append(f"loaded joint {jid} was deleted")
            continue
        lx, ly = design.load(jid)
        if math.hypot(lx - force[0], ly - force[1]) > max(tol, 1e-6):
            reasons.append(f"applied load at joint {jid} was altered")

    if design.n_joints > spec.max_joints:
        reasons.append(f"joint count {design.n_joints} exceeds max_joints {spec.max_joints}")
    if design.n_members > spec.max_members:
        reasons.append(f"member count {design.n_members} exceeds max_members {spec.max_members}")
    return reasons


# equilibrium test
def equilibrium_matrix(design: TrussDesign) -> Tuple[np.ndarray, List[int], List[Member]]:
    """Assemble the free-DOF equilibrium matrix ``A`` of the pin-jointed truss.

    ``A`` has shape ``(n_free_dof, n_members)``; column ``m`` holds the joint
    force contributions of a unit tension in member ``m`` (direction cosines,
    ``+`` on one end, ``-`` on the other), with constrained DOF rows removed.

    Returns
    -------
    A:
        The equilibrium matrix.
    free_dofs:
        Global DOF indices retained (row order of ``A``).
    members:
        Member keys in column order.
    """
    order = design.node_order
    index = {node: k for k, node in enumerate(order)}
    ndof = 2 * len(order)
    members = design.members()

    full = np.zeros((ndof, len(members)), dtype=float)
    for col, (i, j) in enumerate(members):
        (xi, yi), (xj, yj) = design.coords(i), design.coords(j)
        length = math.hypot(xj - xi, yj - yi)
        if length <= 0.0:
            continue
        c, s = (xj - xi) / length, (yj - yi) / length
        full[2 * index[i], col] += c
        full[2 * index[i] + 1, col] += s
        full[2 * index[j], col] -= c
        full[2 * index[j] + 1, col] -= s

    fixed: set[int] = set()
    for node in order:
        sx, sy = design.support(node)
        if sx:
            fixed.add(2 * index[node])
        if sy:
            fixed.add(2 * index[node] + 1)
    free_dofs = [d for d in range(ndof) if d not in fixed]
    return full[free_dofs, :], free_dofs, members


def check_stability(design: TrussDesign) -> Tuple[List[str], Dict[str, float]]:
    """Return failure reasons + metrics for kinematic stability.

    Combines Maxwell's necessary count with a numerical rank test:
    the truss is a rigid, non-mechanism structure iff
    ``rank(A_free) == n_free_dof``. The number of independent mechanisms is
    ``n_free_dof - rank`` and the degree of static indeterminacy (self-stress
    states) is ``n_members - rank``.
    """
    j = design.n_joints
    b = design.n_members
    r = design.n_reactions
    maxwell = b + r - 2 * j

    reasons: List[str] = []
    metrics = {"maxwell": float(maxwell), "b": float(b), "r": float(r), "j": float(j)}

    if r < 3:
        reasons.append(f"only {r} reaction DOFs (< 3): rigid-body motion is unconstrained")
    if maxwell < 0:
        reasons.append(f"Maxwell count b+r-2j = {maxwell} < 0: structure is a mechanism")

    a_free, free_dofs, members = equilibrium_matrix(design)
    n_free = len(free_dofs)
    if members and n_free:
        scale = max(1.0, float(np.linalg.norm(a_free, ord=1)))
        rank = int(np.linalg.matrix_rank(a_free, tol=1e-9 * scale))
    else:
        rank = 0
    n_mechanisms = n_free - rank
    n_selfstress = len(members) - rank

    metrics.update(
        {
            "n_free_dof": float(n_free),
            "equilibrium_rank": float(rank),
            "n_mechanisms": float(n_mechanisms),
            "n_selfstress": float(n_selfstress),
        }
    )
    if n_mechanisms > 0:
        reasons.append(
            f"equilibrium matrix rank {rank} < {n_free} free DOF: "
            f"{n_mechanisms} kinematic mechanism(s)"
        )
    return reasons, metrics


def check_planarity(design: TrussDesign, tol: float = 1.0e-6) -> List[str]:
    """Return reasons if members cross or a joint lies on a non-incident member.

    Members that share an end joint may touch only at that joint; collinear
    overlap of such members is reported as a crossing too.
    """
    reasons: List[str] = []
    coords = {n: design.coords(n) for n in design.joints()}
    segs = [(i, j, coords[i], coords[j]) for i, j in design.members()]

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    def on_segment(p, a, b):
        length = math.hypot(b[0] - a[0], b[1] - a[1])
        if length == 0.0:
            return False
        if abs(cross(a, b, p)) / length > tol:
            return False
        t = ((p[0] - a[0]) * (b[0] - a[0]) + (p[1] - a[1]) * (b[1] - a[1])) / length**2
        return tol / length < t < 1.0 - tol / length

    for i, j, a, b in segs:
        for n, p in coords.items():
            if n not in (i, j) and on_segment(p, a, b):
                reasons.append(f"joint {n} lies on member ({i}, {j}) without being connected")

    for (i, j, a, b), (k, m, c, d) in itertools.combinations(segs, 2):
        shared = {i, j} & {k, m}
        if shared:
            # Members sharing a joint only conflict if they overlap collinearly.
            s = shared.pop()
            far1 = b if s == i else a
            far2 = d if s == k else c
            if on_segment(far1, c, d) or on_segment(far2, a, b):
                reasons.append(f"members ({i}, {j}) and ({k}, {m}) overlap")
            continue
        d1, d2 = cross(c, d, a), cross(c, d, b)
        d3, d4 = cross(a, b, c), cross(a, b, d)
        if ((d1 > tol and d2 < -tol) or (d1 < -tol and d2 > tol)) and (
            (d3 > tol and d4 < -tol) or (d3 < -tol and d4 > tol)
        ):
            reasons.append(f"members ({i}, {j}) and ({k}, {m}) cross")
    return reasons


# gate
def admissibility_gate(design: TrussDesign, spec: DesignSpec) -> GateReport:
    """Run every check on one design and return a combined :class:`GateReport`."""
    reasons: List[str] = []
    reasons += check_language_closure(design, spec)
    reasons += check_connectivity(design)
    reasons += check_geometry(design, spec)
    reasons += check_areas(design)
    if getattr(spec, "planar", False):
        reasons += check_planarity(design)
    stab_reasons, metrics = check_stability(design)
    reasons += stab_reasons
    return GateReport(passed=not reasons, reasons=reasons, metrics=metrics)
