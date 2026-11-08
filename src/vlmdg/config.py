"""Configuration dataclasses and environment loading for :mod:`vlmdg`.

All tunable quantities for a run live here so that an experiment is fully
described by a small set of serializable objects. The default problem
(``ten_bar``) is posed in SI units via :meth:`Material.ten_bar` - the exact
material of Reddy & Cagan (1995). :meth:`Material.steel` is another SI preset and
the bare :class:`Material` default is normalized (``E = sigma_allow = rho = 1``),
used by the toy ``cantilever`` problem. The Metropolis test in
:mod:`vlmdg.optimize.shape_annealing` normalizes the energy change by the
starting energy, so the annealing schedule constants are portable across units.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Dict, Optional, Tuple

try:  # optional, but part of the declared dependencies
    from dotenv import load_dotenv as _load_dotenv
except ImportError:  # pragma: no cover - dotenv is a hard dependency in practice
    _load_dotenv = None


JointId = int
Coord = Tuple[float, float]
Dof = Tuple[bool, bool]


def load_env(dotenv_path: str | None = None) -> None:
    """Load environment variables from a ``.env`` file if ``python-dotenv`` is available.

    Parameters
    ----------
    dotenv_path:
        Explicit path to a dotenv file. When ``None``, load the repository's
        ``.env`` regardless of the caller's working directory.
    """
    if _load_dotenv is not None:
        path = dotenv_path or Path(__file__).resolve().parents[2] / ".env"
        _load_dotenv(dotenv_path=path, override=False)


@dataclass(frozen=True)
class Material:
    """Isotropic bar material and the sizing bounds applied to every member.

    Attributes
    ----------
    young_modulus:
        Axial stiffness modulus ``E``.
    density:
        Mass per unit volume ``rho`` used for the mass objective.
    stress_allowable:
        Symmetric allowable axial stress used by the stress constraint.
    min_area, max_area:
        Lower / upper bound on a member cross-sectional area.
    section_factor:
        Second-moment-of-area model ``I = section_factor * A**2`` used by the
        Euler buckling check. ``1 / (4 * pi)`` corresponds to a solid circular
        section (the classic shape-annealing truss assumption).
    """

    young_modulus: float = 1.0
    density: float = 1.0
    stress_allowable: float = 1.0
    min_area: float = 0.1
    max_area: float = 10.0
    section_factor: float = 1.0 / (4.0 * 3.141592653589793)

    @classmethod
    def steel(cls) -> "Material":
        """Return a mild-steel material in SI units (N, m, kg)."""
        return cls(
            young_modulus=2.1e11,
            density=7850.0,
            stress_allowable=2.5e8,
            min_area=1.0e-4,
            max_area=5.0e-2,
        )

    @classmethod
    def ten_bar(cls) -> "Material":
        """Material for the Reddy & Cagan (1995) 10-bar truss benchmark.

        SI units (N, m). ``density`` is the *weight* density (27 kN/m^3, as
        printed in the paper) so the ``sum(rho * A * L)`` objective is the truss
        **weight in newtons** - directly comparable to the paper's figures. The
        stress limit (172 MPa) is also printed in the paper. Neither baseline
        source (the 1995a paper or EDRC report 24-101-93) states E;
        ``young_modulus`` = 68.8 GPa is the value Reddy & Cagan give for the same
        benchmark in "An Improved Shape Annealing Algorithm For Truss Topology
        Generation" (J. Mech. Des., June 1995). ``min_area`` (0.1 in^2) is the
        value commonly used for this aluminium benchmark. ``section_factor = 1 / (4 * pi)`` keeps the
        solid circular section assumed by the shape-annealing buckling check.
        """
        return cls(
            young_modulus=6.88e10,
            density=2.7e4,
            stress_allowable=1.72e8,
            min_area=6.45e-5,
            # The paper sets no upper area bound (``min_area`` is only a bar-
            # removal threshold); this generous cap keeps the NLP bounded while
            # still admitting the stout compression members Euler buckling forces
            # on the long unbraced diagonals of the initial 10-bar truss.
            max_area=2.0e-1,
        )


@dataclass(frozen=True)
class DesignSpec:
    """Immutable definition of the design problem a grammar rule must respect.

    The spec captures everything that makes a candidate design a member of the
    language ``L(G)``: the bounding box, the joints that must stay put (supports
    and their reaction DOFs), the loads that must remain applied, and the size
    caps that keep the search finite.
    """

    name: str
    x_bounds: Coord = (0.0, 3.0)
    y_bounds: Coord = (0.0, 2.0)
    fixed_joints: Dict[JointId, Coord] = field(default_factory=dict)
    supports: Dict[JointId, Dof] = field(default_factory=dict)
    loads: Dict[JointId, Coord] = field(default_factory=dict)
    min_joint_spacing: float = 0.15
    min_member_length: float = 0.15
    max_member_length: float = 4.0
    max_joints: int = 24
    max_members: int = 60
    position_tolerance: float = 1.0e-6
    #: When ``True`` the truss must be planar: no two members may cross, and
    #: no joint may lie on a member it is not an end of.
    planar: bool = False

    def in_domain(self, x: float, y: float) -> bool:
        """Return ``True`` when ``(x, y)`` lies inside the (closed) design domain."""
        lo_x, hi_x = self.x_bounds
        lo_y, hi_y = self.y_bounds
        tol = self.position_tolerance
        return (lo_x - tol) <= x <= (hi_x + tol) and (lo_y - tol) <= y <= (hi_y + tol)

    def protected_joints(self) -> set[JointId]:
        """Joint ids that a rule may neither move nor delete (supports + loads)."""
        return set(self.fixed_joints) | set(self.loads)


@dataclass(frozen=True)
class AnnealConfig:
    """Simulated-annealing schedule and stall/reheat policy for shape annealing."""

    initial_temperature: float = 1.0
    cooling_rate: float = 0.97
    steps_per_temperature: int = 20
    max_iterations: int = 4000
    min_temperature: float = 1.0e-4
    stall_window: int = 250
    max_stalls: int = 6
    reheat_fraction: float = 0.5
    #: At each stall, continue the search from the best design found so far.
    restart_from_best: bool = True
    penalty_weight: float = 10.0
    seed: int = 0

    def scaled(self, factor: float) -> "AnnealConfig":
        """Return a copy with iteration budget and windows scaled by ``factor``.

        Useful for fast smoke tests.
        """
        return replace(
            self,
            max_iterations=max(1, int(self.max_iterations * factor)),
            stall_window=max(1, int(self.stall_window * factor)),
        )


@dataclass(frozen=True)
class VLMConfig:
    """Selection and pricing of the VLM backend."""

    provider: str = "mock"
    model: str = "gpt-5.6-luna"
    max_tokens: int = 2048
    temperature: float = 1.0
    # Approximate USD per 1M tokens; override for the exact model/date in use.
    price_input_per_mtok: float = 3.0
    price_output_per_mtok: float = 15.0
    attach_image: bool = True

    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Estimate request cost in USD from token counts."""
        return (
            input_tokens / 1.0e6 * self.price_input_per_mtok
            + output_tokens / 1.0e6 * self.price_output_per_mtok
        )


@dataclass(frozen=True)
class RunConfig:
    """Top-level description of one run."""

    problem: str = "ten_bar"
    mode: str = "informed"  # one of: reheat, random, uninformed, informed
    anneal: AnnealConfig = field(default_factory=AnnealConfig)
    vlm: VLMConfig = field(default_factory=VLMConfig)
    #: Material the problem is posed in. ``None`` -> resolved from ``problem``
    #: via :func:`vlmdg.problems.get_material`.
    material: Material | None = None
    #: Sizer that sets the member areas of every candidate before it is scored:
    #: ``"auto"`` (FSD),
    #: ``"fsd"``, ``"slsqp"``, or ``"shape"`` (areas + free joint positions).
    sizer: str = "auto"
    output_dir: str = "runs"
    max_repair_attempts: int = 3
    #: Proposals per stall: a rejected rule is sent back with its reasons.
    max_proposal_attempts: int = 3
    #: Reject generated rules whose outcomes an existing rule already produces.
    check_duplicates: bool = True
    #: Generate rules only at the first this many stalls; later stalls only
    #: reheat. ``None`` = every stall.
    max_rule_stalls: Optional[int] = None
    #: JSON file of extra rules (``[{"name", "source"}, ...]``) compiled and
    #: added to the initial grammar before the search starts.
    extra_rules_path: Optional[str] = None
    #: Store the current and best designs every this many iterations (0 = off).
    snapshot_every: int = 0

    @property
    def anthropic_api_key(self) -> str | None:
        """Fetch the Anthropic API key from the environment (never stored here)."""
        return os.environ.get("ANTHROPIC_API_KEY")
