"""Shape-annealing search, diagnostics, and experiment baselines."""

from vlmdg.optimize.baselines import STALL_CALLBACKS, make_stall_callback
from vlmdg.optimize.diagnostics import SearchDiagnostics, summarize_diagnostics
from vlmdg.optimize.objective import Evaluation, Objective
from vlmdg.optimize.shape_annealing import AnnealResult, ShapeAnnealer
from vlmdg.optimize.sizing import (
    FullyStressedSizer,
    ShapeSizer,
    Sizer,
    SlsqpSizer,
    make_sizer,
)

__all__ = [
    "Objective",
    "Evaluation",
    "ShapeAnnealer",
    "AnnealResult",
    "SearchDiagnostics",
    "summarize_diagnostics",
    "STALL_CALLBACKS",
    "make_stall_callback",
    "FullyStressedSizer",
    "SlsqpSizer",
    "ShapeSizer",
    "Sizer",
    "make_sizer",
]
