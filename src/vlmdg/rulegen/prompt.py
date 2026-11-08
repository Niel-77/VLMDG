"""Prompt construction for the VLM rule generator.

The informed prompt carries the full stall diagnostics (active constraints, rule
utility, incumbent design); the uninformed prompt is the baseline that withholds
those and shows only the current design and the existing rules.
"""

from __future__ import annotations

from vlmdg.design.io import describe_design
from vlmdg.grammar.registry import Grammar
from vlmdg.optimize.diagnostics import SearchDiagnostics, summarize_diagnostics

API_REFERENCE = """\
You write ONE Python function with this exact signature:

    def rule(design, rng):
        ...
        return new_design

`design` is a TrussDesign that has ALREADY been copied for you - mutate it in
place and return it (or return a further copy). `rng` is a random.Random.
`math`, `random`, `itertools`, `numpy` (as np) and `networkx` (as nx) are
available; do not read files, network, or environment.

TrussDesign API you may use:
    design.joints() -> list[int]                 joint ids
    design.members() -> list[tuple[int,int]]     canonical (i, j) pairs, i < j
    design.node_order -> list[int]               ids in ascending order
    design.n_joints, design.n_members            counts
    design.coords(i) -> (x, y)
    design.support(i) -> (bool, bool)            fixed x / y reaction DOF
    design.load(i) -> (Fx, Fy)
    design.area(i, j) -> float                   set by sizing; read-only
    design.member_length(i, j) -> float
    design.degree(i) -> int
    design.has_member(i, j) -> bool
    design.nearest_joints(x, y, k=1, exclude=()) -> list[int]
    design.add_joint(x, y) -> int                returns the new joint id
    design.add_member(i, j)
    design.remove_member(i, j)
    design.remove_joint(i)
    design.move_joint(i, x, y)

Member areas are not part of a rule: after every rule, a sizing step sets the
area of every member, so change only joints and members.

Hard invariants (a rule that breaks these is rejected by the admissibility gate):
  * never move, delete, or reload a supported joint or a loaded joint;
  * keep the truss one connected component with no isolated joints;
  * keep every joint inside the design domain and members within length limits;
  * the result must be a rigid pin-jointed truss (no mechanism): every new joint
    must be triangulated / braced, not left on a single member or two collinear
    members.

Output contract: a short rationale, then the function in ONE ```python fenced
block. No prose inside the block other than a docstring/comments.
"""

SYSTEM_PROMPT = (
    "You are a structural design assistant embedded in a shape-annealing optimizer "
    "for planar pin-jointed trusses. The search has stalled. Your job is to invent "
    "ONE new grammar rule - a general geometric/topological operator, not a fix for "
    "one design - that gives the search a new, physically sensible move so it can "
    "escape the stall and keep reducing mass while satisfying stress and Euler "
    "buckling limits.\n\n" + API_REFERENCE
)


def _grammar_digest(grammar: Grammar) -> str:
    """One line per existing rule: name, origin, and description."""
    lines = ["Existing grammar rules (do not simply duplicate one):"]
    for rule in grammar.rules:
        lines.append(f"  - {rule.name} [{rule.origin}]: {rule.description or '(no description)'}")
    return "\n".join(lines)


def problem_limits(spec, design=None) -> str:
    """State the numeric limits of the problem that the admissibility gate enforces."""
    (x0, x1), (y0, y1) = spec.x_bounds, spec.y_bounds
    lines = [
        "Problem limits (the admissibility gate rejects a result that breaks any):",
        f"  * design domain: {x0:g} <= x <= {x1:g}, {y0:g} <= y <= {y1:g}",
        f"  * joints at least {spec.min_joint_spacing:g} apart",
        f"  * member length between {spec.min_member_length:g} and {spec.max_member_length:g}",
        f"  * at most {spec.max_joints} joints and {spec.max_members} members",
    ]
    if design is not None:
        lines.append(
            f"  The current design has {design.n_joints} joints and {design.n_members} members."
        )
        if design.n_joints >= spec.max_joints:
            lines.append(
                "  It is at the joint limit, so a rule that adds a joint must also "
                "remove one; rules that only rewire members or move joints still apply."
            )
        if design.n_members >= spec.max_members:
            lines.append(
                "  It is at the member limit, so a rule that adds a member must also "
                "remove one."
            )
    if getattr(spec, "planar", False):
        lines.append(
            "  * the truss must be PLANAR: no two members may cross, and no joint may "
            "lie on a member it is not an end of. Check this before adding a member "
            "or moving a joint."
        )
    else:
        lines.append(
            "  * members MAY cross (this problem is not planar); a crossing is not a joint."
        )
    return "\n".join(lines)


def build_user_prompt(
    diag: SearchDiagnostics, grammar: Grammar, *, informed: bool, spec=None
) -> str:
    """Assemble the user-turn prompt for a rule-generation request."""
    parts: list[str] = []
    if informed:
        parts.append(summarize_diagnostics(diag, include_design=True))
    else:
        parts.append(
            "The optimizer has stalled (no search diagnostics are provided in this "
            "condition). Here is the current design only:"
        )
        if diag.best_design is not None:
            parts.append(describe_design(diag.best_design))

    parts.append("")
    parts.append(_grammar_digest(grammar))
    parts.append("")
    if spec is not None:
        parts.append(problem_limits(spec, diag.best_design))
        parts.append("")
    parts.append(
        "Propose ONE new rule as `def rule(design, rng): ...`. Prefer an operator "
        "that adds or reroutes load paths (bracing, sub-triangulation, a new "
        "member following the force flow) over a purely random perturbation."
    )
    return "\n".join(parts)


def build_repair_prompt(previous_code: str, error: str) -> str:
    """Prompt asking the model to fix a rule that failed the execution gate."""
    return (
        "The rule you provided failed to execute. Fix it and return the corrected "
        "function in ONE ```python block, same signature `def rule(design, rng):`.\n\n"
        f"Error:\n{error}\n\n"
        f"Your previous code:\n```python\n{previous_code}\n```"
    )


def build_rejection_feedback(rejected: list[tuple[str, list[str]]]) -> str:
    """Tell the model which of its proposals at this stall were rejected, and why."""
    lines = [
        "Your earlier proposal(s) at this stall were REJECTED. Propose a rule that "
        "makes a genuinely different move, not a variant of these:"
    ]
    for k, (source, reasons) in enumerate(rejected, start=1):
        lines.append(f"\nRejected proposal {k}. Reasons: {'; '.join(reasons) or 'unknown'}")
        lines.append(f"```python\n{source}\n```")
    return "\n".join(lines)
