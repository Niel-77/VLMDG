"""Direct-stiffness linear analysis of a 2D pin-jointed truss.

Small, dependency-light (``numpy`` only) solver. It returns member axial forces
(tension positive), member stresses, nodal displacements, support reactions and
the strain-energy compliance and also returns a ``singular`` flag. The flag is set when the truss is
unstable: some joints can move without stretching or compressing any bar (for
example, a square frame with no diagonal folds into a parallelogram), so the
truss cannot carry the load. The code checksthis by counting how many independent
directions the truss resists (the matrix rank). If that count is smaller than 
the number of free joint directions, the truss is flagged as singular.

For a singular truss the solver still returns a best-fit (least-squares)
answer, but the shape-annealing objective ignores it and marks the design
infeasible.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from vlmdg.config import Material
from vlmdg.design.truss import TrussDesign

Member = Tuple[int, int]


@dataclass(frozen=True)
class FEAResult:
    """Outcome of a single linear truss analysis.

    Attributes
    ----------
    displacements:
        Mapping ``joint_id -> (ux, uy)``.
    member_forces:
        Mapping ``member_key -> axial force`` (tension positive).
    member_stress:
        Mapping ``member_key -> axial stress`` (tension positive).
    reactions:
        Mapping ``joint_id -> (Rx, Ry)`` at constrained joints.
    compliance:
        External work ``F . u`` (twice the strain energy); a stiffness proxy.
    singular:
        ``True`` when the reduced stiffness matrix is rank deficient.
    max_abs_stress:
        Convenience: largest absolute member stress (0.0 for an empty truss).
    """

    displacements: Dict[int, Tuple[float, float]]
    member_forces: Dict[Member, float]
    member_stress: Dict[Member, float]
    reactions: Dict[int, Tuple[float, float]]
    compliance: float
    singular: bool
    max_abs_stress: float


def _element_stiffness(c: float, s: float, ea_over_l: float) -> np.ndarray:
    """Return the 4x4 global-frame stiffness of one bar element."""
    k = np.array(
        [
            [c * c, c * s, -c * c, -c * s],
            [c * s, s * s, -c * s, -s * s],
            [-c * c, -c * s, c * c, c * s],
            [-c * s, -s * s, c * s, s * s],
        ],
        dtype=float,
    )
    return ea_over_l * k


def solve(design: TrussDesign, material: Material, rcond: float = 1e-12) -> FEAResult:
    """Analyze ``design`` under its own joint loads and supports.

    Parameters
    ----------
    design:
        The truss to analyze.
    material:
        Supplies the Young's modulus ``E`` (density is not used here).
    rcond:
        Relative singular-value cutoff used both for the rank test and for the
        least-squares fallback on a singular system.
    """
    order: List[int] = design.node_order
    index = {node: k for k, node in enumerate(order)}
    ndof = 2 * len(order)

    stiffness = np.zeros((ndof, ndof), dtype=float)
    geometry: Dict[Member, Tuple[float, float, float, float]] = {}

    for i, j in design.members():
        (xi, yi), (xj, yj) = design.coords(i), design.coords(j)
        length = math.hypot(xj - xi, yj - yi)
        if length <= 0.0:
            raise ValueError(f"zero-length member ({i}, {j})")
        c, s = (xj - xi) / length, (yj - yi) / length
        area = design.area(i, j)
        ke = _element_stiffness(c, s, material.young_modulus * area / length)
        dofs = [2 * index[i], 2 * index[i] + 1, 2 * index[j], 2 * index[j] + 1]
        stiffness[np.ix_(dofs, dofs)] += ke
        geometry[(i, j)] = (c, s, length, area)

    forces = np.zeros(ndof, dtype=float)
    fixed: List[int] = []
    for node in order:
        fx, fy = design.load(node)
        forces[2 * index[node]] += fx
        forces[2 * index[node] + 1] += fy
        sx, sy = design.support(node)
        if sx:
            fixed.append(2 * index[node])
        if sy:
            fixed.append(2 * index[node] + 1)

    free = [d for d in range(ndof) if d not in set(fixed)]
    disp = np.zeros(ndof, dtype=float)
    singular = False

    if free:
        k_ff = stiffness[np.ix_(free, free)]
        f_f = forces[free]
        rank = np.linalg.matrix_rank(k_ff, tol=rcond * max(1.0, np.linalg.norm(k_ff, ord=1)))
        if rank < len(free):
            singular = True
            disp[free], *_ = np.linalg.lstsq(k_ff, f_f, rcond=rcond)
        else:
            disp[free] = np.linalg.solve(k_ff, f_f)

    displacements = {
        node: (float(disp[2 * index[node]]), float(disp[2 * index[node] + 1])) for node in order
    }

    member_forces: Dict[Member, float] = {}
    member_stress: Dict[Member, float] = {}
    for (i, j), (c, s, length, area) in geometry.items():
        du = disp[2 * index[j]] - disp[2 * index[i]]
        dv = disp[2 * index[j] + 1] - disp[2 * index[i] + 1]
        axial = material.young_modulus * area / length * (c * du + s * dv)
        member_forces[(i, j)] = float(axial)
        member_stress[(i, j)] = float(axial / area)

    reaction_vec = stiffness @ disp - forces
    reactions = {
        node: (float(reaction_vec[2 * index[node]]), float(reaction_vec[2 * index[node] + 1]))
        for node in order
        if any(design.support(node))
    }

    max_abs_stress = max((abs(v) for v in member_stress.values()), default=0.0)
    compliance = float(forces @ disp)

    return FEAResult(
        displacements=displacements,
        member_forces=member_forces,
        member_stress=member_stress,
        reactions=reactions,
        compliance=compliance,
        singular=singular,
        max_abs_stress=max_abs_stress,
    )
