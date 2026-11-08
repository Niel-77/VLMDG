"""Penalized weight objective for shape annealing.

The search minimizes the energy

    E = m * (1 + w * sum(max(0, g_k)))

where m is the mass, g_k are the normalized stress and buckling violations
from vlmdg.fea.checks, and w is AnnealConfig.penalty_weight. A mechanism
(singular stiffness matrix) gets E = inf. Multiplying, rather than adding,
means the penalty needs no scaling to the mass units.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

from vlmdg.config import Material
from vlmdg.design.truss import TrussDesign
from vlmdg.fea import solve
from vlmdg.fea.checks import (
    ConstraintReport,
    evaluate_constraints,
    invalid_areas,
    structural_mass,
)
from vlmdg.fea.truss_fea import FEAResult


@dataclass
class Evaluation:
    """The full result of scoring one design."""

    mass: float
    penalty: float
    energy: float
    feasible: bool
    constraints: ConstraintReport
    fea: Optional[FEAResult]

    @property
    def max_violation(self) -> float:
        """Largest normalized constraint violation (``<= 0`` == feasible)."""
        return self.constraints.max_violation


class Objective:
    """Callable scorer: ``objective(design) -> Evaluation``."""

    def __init__(self, material: Material, penalty_weight: float = 10.0) -> None:
        self.material = material
        self.penalty_weight = penalty_weight

    def evaluate(self, design: TrussDesign) -> Evaluation:
        """Analyze ``design`` and return its :class:`Evaluation`.

        A design with an area that is not finite, not positive, or outside the
        material's area bounds is infeasible with infinite energy, and is not
        analyzed.
        """
        bad = invalid_areas(design, self.material)
        if bad:
            return Evaluation(
                mass=math.inf,
                penalty=math.inf,
                energy=math.inf,
                feasible=False,
                constraints=ConstraintReport(invalid_area=bad),
                fea=None,
            )

        mass = structural_mass(design, self.material)

        if design.n_members == 0:
            return Evaluation(
                mass=mass,
                penalty=math.inf,
                energy=math.inf,
                feasible=False,
                constraints=ConstraintReport(singular=True),
                fea=None,
            )

        try:
            fea = solve(design, self.material)
        except (ValueError, ZeroDivisionError):
            return Evaluation(
                mass=mass,
                penalty=math.inf,
                energy=math.inf,
                feasible=False,
                constraints=ConstraintReport(singular=True),
                fea=None,
            )

        report = evaluate_constraints(design, fea, self.material)
        if report.singular:
            return Evaluation(
                mass=mass, penalty=math.inf, energy=math.inf,
                feasible=False, constraints=report, fea=fea,
            )

        penalty = sum(max(0.0, g) for g in report.violations)
        energy = mass * (1.0 + self.penalty_weight * penalty)
        return Evaluation(
            mass=mass,
            penalty=penalty,
            energy=energy,
            feasible=report.feasible,
            constraints=report,
            fea=fea,
        )

    __call__ = evaluate
