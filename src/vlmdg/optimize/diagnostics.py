"""Search diagnostics sent to the rule generator at a stall.

SearchDiagnostics describes why the search is stuck: rule usage and success,
active constraints, and stall length. The informed mode passes it to the VLM
(vision language model); the uninformed mode does not.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

from vlmdg.design.io import describe_design
from vlmdg.design.truss import TrussDesign
from vlmdg.optimize.objective import Evaluation

Member = Tuple[int, int]


@dataclass
class SearchDiagnostics:
    """A snapshot of the search state at the moment a stall is declared."""

    iteration: int
    temperature: float
    stall_index: int
    iterations_since_improvement: int
    best_energy: float
    best_mass: float
    best_feasible: bool
    max_violation: float
    active_constraints: List[Tuple[str, Member, float]] = field(default_factory=list)
    rule_utility: List[dict] = field(default_factory=list)
    recent_accepted_rules: List[str] = field(default_factory=list)
    best_design: TrussDesign | None = None

    def to_dict(self) -> dict:
        """JSON-friendly view (drops the live design object)."""
        return {
            "iteration": self.iteration,
            "temperature": self.temperature,
            "stall_index": self.stall_index,
            "iterations_since_improvement": self.iterations_since_improvement,
            "best_energy": self.best_energy,
            "best_mass": self.best_mass,
            "best_feasible": self.best_feasible,
            "max_violation": self.max_violation,
            "active_constraints": [
                {"kind": k, "member": list(m), "violation": round(g, 5)}
                for k, m, g in self.active_constraints
            ],
            "rule_utility": self.rule_utility,
            "recent_accepted_rules": self.recent_accepted_rules,
        }


def build_diagnostics(
    *,
    iteration: int,
    temperature: float,
    stall_index: int,
    iterations_since_improvement: int,
    best_design: TrussDesign,
    best_eval: Evaluation,
    rule_utility: List[dict],
    recent_accepted_rules: List[str],
    active_band: float = 0.05,
) -> SearchDiagnostics:
    """Assemble a :class:`SearchDiagnostics` from raw annealer state."""
    return SearchDiagnostics(
        iteration=iteration,
        temperature=temperature,
        stall_index=stall_index,
        iterations_since_improvement=iterations_since_improvement,
        best_energy=best_eval.energy,
        best_mass=best_eval.mass,
        best_feasible=best_eval.feasible,
        max_violation=best_eval.max_violation,
        active_constraints=best_eval.constraints.active_members(band=active_band),
        rule_utility=rule_utility,
        recent_accepted_rules=recent_accepted_rules,
        best_design=best_design,
    )


def summarize_diagnostics(diag: SearchDiagnostics, *, include_design: bool = True) -> str:
    """Render diagnostics as prompt text for the VLM (informed mode)."""
    lines = [
        f"Search stall #{diag.stall_index} at iteration {diag.iteration}.",
        f"Temperature: {diag.temperature:.4g}. "
        f"Iterations since last improvement: {diag.iterations_since_improvement}.",
        f"Incumbent best: mass={diag.best_mass:.4g}, energy={diag.best_energy:.4g}, "
        f"feasible={diag.best_feasible}, max_constraint_violation={diag.max_violation:.3g}.",
    ]

    if diag.active_constraints:
        lines.append("Active / violated constraints (worst first):")
        for kind, member, g in diag.active_constraints[:8]:
            state = "VIOLATED" if g > 0 else "near-limit"
            lines.append(f"  - {kind} on member {member[0]}-{member[1]}: g={g:+.3f} ({state})")
    else:
        lines.append("No constraints are near their limits; the design is mass-bound.")

    if diag.rule_utility:
        lines.append("Rule utility so far (name: proposed/accepted, mean dE):")
        for row in diag.rule_utility:
            lines.append(
                f"  - {row['name']} [{row['origin']}]: "
                f"{row['proposed']}/{row['accepted']}, "
                f"mean dE={row['mean_energy_delta']:+.4g}"
            )
    if diag.recent_accepted_rules:
        lines.append("Recently accepted rules: " + ", ".join(diag.recent_accepted_rules[-10:]))

    if include_design and diag.best_design is not None:
        lines.append("")
        lines.append("Current best design:")
        lines.append(describe_design(diag.best_design))

    return "\n".join(lines)
