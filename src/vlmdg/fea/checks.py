"""Structural constraint evaluation for a truss design.

Each constraint is returned in normalized *violation* form ``g``:

* ``g <= 0``  -> satisfied,
* ``g > 0``   -> violated (the value is the fractional overshoot).

This matches the convention used by the shape-annealing penalty in
:mod:`vlmdg.optimize.objective`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from vlmdg.config import Material
from vlmdg.design.truss import TrussDesign
from vlmdg.fea.truss_fea import FEAResult

Member = Tuple[int, int]


@dataclass
class ConstraintReport:
    """Collected constraint violations for one design/analysis pair."""

    stress: Dict[Member, float] = field(default_factory=dict)
    buckling: Dict[Member, float] = field(default_factory=dict)
    singular: bool = False
    #: Members whose area is not finite, not positive, or outside the bounds.
    invalid_area: Dict[Member, float] = field(default_factory=dict)

    @property
    def violations(self) -> List[float]:
        """All violation values (positive == violated).

        A singular truss or an invalid area counts as one large violation.
        """
        values = list(self.stress.values()) + list(self.buckling.values())
        if self.singular or self.invalid_area:
            values.append(1.0e6)
        return values

    @property
    def max_violation(self) -> float:
        """Largest violation; ``<= 0`` means the design is feasible."""
        return max(self.violations, default=0.0)

    @property
    def feasible(self) -> bool:
        """``True`` when no constraint is violated."""
        return self.max_violation <= 0.0

    def active_members(self, band: float = 0.05) -> List[Tuple[str, Member, float]]:
        """Return ``(kind, member, g)`` triples that are violated or near-active.

        A constraint is "active" when ``g >= -band`` (satisfied but within
        ``band`` of the limit) - the set the optimizer is fighting against.
        """
        out: List[Tuple[str, Member, float]] = []
        for member, g in self.stress.items():
            if g >= -band:
                out.append(("stress", member, g))
        for member, g in self.buckling.items():
            if g >= -band:
                out.append(("buckling", member, g))
        out.sort(key=lambda item: item[2], reverse=True)
        return out


def invalid_areas(design: TrussDesign, material: Material) -> Dict[Member, float]:
    """Return the members whose area is not finite, not positive, or out of bounds.

    The bounds are ``[material.min_area, material.max_area]``, with a relative
    tolerance of ``1e-9`` for rounding.
    """
    lo = material.min_area * (1.0 - 1e-9)
    hi = material.max_area * (1.0 + 1e-9)
    out: Dict[Member, float] = {}
    for i, j in design.members():
        area = design.area(i, j)
        if not (math.isfinite(area) and area > 0.0 and lo <= area <= hi):
            out[(i, j)] = area
    return out


def stress_constraints(fea: FEAResult, material: Material) -> Dict[Member, float]:
    """Return ``|sigma| / sigma_allow - 1`` for every member."""
    allow = material.stress_allowable
    return {m: abs(sigma) / allow - 1.0 for m, sigma in fea.member_stress.items()}


def buckling_constraints(
    design: TrussDesign, fea: FEAResult, material: Material
) -> Dict[Member, float]:
    """Return the Euler-buckling violation for every compression member.

    The critical stress uses the solid-section model ``I = section_factor * A**2``:
    ``sigma_cr = pi**2 * E * I / (A * L**2)``. Tension members get ``g = -1``
    (no buckling risk).
    """
    out: Dict[Member, float] = {}
    for (i, j), axial in fea.member_forces.items():
        area = design.area(i, j)
        length = design.member_length(i, j)
        sigma = axial / area
        if sigma >= 0.0:
            out[(i, j)] = -1.0
            continue
        inertia = material.section_factor * area * area
        sigma_cr = math.pi ** 2 * material.young_modulus * inertia / (area * length ** 2)
        out[(i, j)] = abs(sigma) / sigma_cr - 1.0
    return out


def evaluate_constraints(
    design: TrussDesign, fea: FEAResult, material: Material
) -> ConstraintReport:
    """Bundle stress + buckling + singularity into a :class:`ConstraintReport`."""
    return ConstraintReport(
        stress=stress_constraints(fea, material),
        buckling=buckling_constraints(design, fea, material),
        singular=fea.singular,
    )


def structural_mass(design: TrussDesign, material: Material) -> float:
    """Return ``sum(rho * A * L)`` over all members."""
    return sum(
        material.density * design.area(i, j) * design.member_length(i, j)
        for i, j in design.members()
    )
