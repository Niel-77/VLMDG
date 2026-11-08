"""Linear 2D truss finite-element analysis and structural constraint checks."""

from vlmdg.fea.checks import (
    ConstraintReport,
    buckling_constraints,
    stress_constraints,
    structural_mass,
)
from vlmdg.fea.truss_fea import FEAResult, solve

__all__ = [
    "FEAResult",
    "solve",
    "ConstraintReport",
    "stress_constraints",
    "buckling_constraints",
    "structural_mass",
]
