"""Bar sizing for a fixed topology.

FullyStressedSizer gives each bar the smallest area A that meets both limits,
re-analysing the truss until the areas stop changing (fully stressed design):

    stress:    A >= |N| / sigma_allow
    buckling:  A >= sqrt(|N| * L**2 / (pi**2 * E * k))   (compression only)

where N is the bar force, L its length, E the elastic modulus, and k links the
second moment of area to the area (I = k * A**2). Areas are kept within
[min_area, max_area]. On the ten-bar truss this matches the published
area-only weight within 1%.

SlsqpSizer refines the areas with SLSQP (sequential least squares
programming). ShapeSizer also moves the free joints.
"""

from __future__ import annotations

import math
from typing import Callable, List, Tuple

from vlmdg.config import DesignSpec, Material
from vlmdg.design.truss import TrussDesign
from vlmdg.fea import solve
from vlmdg.fea.checks import evaluate_constraints, structural_mass

Sizer = Callable[[TrussDesign], TrussDesign]

#: Fractional safety margin the sizers keep inside the stress / buckling limits
#: so the result reads as genuinely feasible despite iterative round-off.
SIZING_MARGIN = 0.01


class FullyStressedSizer:
    """Size bar areas by the fully stressed design rule.

    Returns a copy of the design with new areas; topology and geometry are
    unchanged. If the analysis fails (a mechanism), the input is returned as is.
    """

    def __init__(
        self,
        material: Material,
        *,
        max_iter: int = 40,
        tol: float = 1.0e-4,
        relaxation: float = 0.7,
        margin: float = SIZING_MARGIN,
    ) -> None:
        self.material = material
        self.max_iter = max_iter
        self.tol = tol
        self.relaxation = relaxation
        self.margin = margin

    def __call__(self, design: TrussDesign) -> TrussDesign:
        return self.size(design)

    def size(self, design: TrussDesign) -> TrussDesign:
        """Return a copy of ``design`` with fully-stressed bar areas."""
        if design.n_members == 0:
            return design

        mat = self.material
        k_sec = mat.section_factor
        safe_stress = mat.stress_allowable / (1.0 + self.margin)
        buckle_gain = (1.0 + self.margin) / (math.pi ** 2 * mat.young_modulus * k_sec)
        sized = design.copy()

        for _ in range(self.max_iter):
            try:
                fea = solve(sized, mat)
            except (ValueError, ZeroDivisionError):
                return design
            if fea.singular:
                return design

            max_rel_change = 0.0
            for (i, j), axial in fea.member_forces.items():
                length = sized.member_length(i, j)
                old_area = sized.area(i, j)

                need_stress = abs(axial) / safe_stress
                need_buckle = 0.0
                if axial < 0.0:  # compression -> Euler buckling
                    need_buckle = math.sqrt(abs(axial) * length * length * buckle_gain)
                target = min(max(need_stress, need_buckle, mat.min_area), mat.max_area)
                new_area = old_area + self.relaxation * (target - old_area)
                new_area = min(max(new_area, mat.min_area), mat.max_area)

                sized.set_area(i, j, new_area)
                max_rel_change = max(max_rel_change, abs(new_area - old_area) / old_area)

            if max_rel_change < self.tol:
                break

        return sized


class SlsqpSizer:
    """Minimize truss weight over the bar areas with SLSQP.

    The constraints are the stress and buckling limits, tightened by margin so
    the result stays feasible when checked again. Starts from the fully
    stressed design and returns it if SLSQP fails or does not improve on it.
    About 20 times slower than FullyStressedSizer.
    """

    def __init__(
        self,
        material: Material,
        *,
        max_members: int = 24,
        max_iter: int = 150,
        margin: float = 0.03,
    ) -> None:
        self.material = material
        self.max_members = max_members
        self.max_iter = max_iter
        self.margin = margin
        self._fsd = FullyStressedSizer(material)

    def __call__(self, design: TrussDesign) -> TrussDesign:
        return self.size(design)

    def size(self, design: TrussDesign) -> TrussDesign:
        """Return a copy of ``design`` with SQP-optimised bar areas."""
        warm = self._fsd.size(design)
        if design.n_members == 0 or design.n_members > self.max_members:
            return warm
        try:
            import numpy as np  # pylint: disable=import-outside-toplevel
            from scipy.optimize import minimize  # pylint: disable=import-outside-toplevel
        except ImportError:
            return warm

        mat = self.material
        members: List[Tuple[int, int]] = list(design.members())
        lengths = np.array([design.member_length(i, j) for i, j in members])
        rho, e_mod, k_sec = mat.density, mat.young_modulus, mat.section_factor
        safe_stress = mat.stress_allowable / (1.0 + self.margin)
        area_scale = mat.max_area
        weight_scale = float(rho * area_scale * lengths.sum())

        def forces(areas: "np.ndarray") -> "np.ndarray":
            probe = design.copy()
            for (i, j), a in zip(members, areas):
                probe.set_area(i, j, float(max(a, mat.min_area * 1e-3)))
            fea = solve(probe, mat)
            if fea.singular:
                return np.full(len(members), np.nan)
            return np.array([fea.member_forces[(i, j)] for i, j in members])

        def weight(unit_areas):
            return float(rho * np.dot(unit_areas * area_scale, lengths) / weight_scale)

        def weight_grad(unit_areas):  # pylint: disable=unused-argument
            return rho * lengths * area_scale / weight_scale

        def constraints(unit_areas: "np.ndarray") -> "np.ndarray":
            areas = unit_areas * area_scale
            axial = forces(areas)
            if not np.all(np.isfinite(axial)):
                return np.full(2 * len(members), -1.0e3)
            stress_ok = 1.0 - np.abs(axial) / (safe_stress * areas)
            compressed = np.minimum(axial, 0.0)
            crit_force = np.pi ** 2 * e_mod * k_sec * areas ** 2 / lengths ** 2
            buckle_ok = 1.0 - np.abs(compressed) * (1.0 + self.margin) / crit_force
            return np.concatenate([stress_ok, buckle_ok])

        u0 = np.array([warm.area(i, j) for i, j in members]) / area_scale
        lo = mat.min_area / area_scale
        try:
            res = minimize(
                weight, u0, jac=weight_grad, bounds=[(lo, 1.0)] * len(members),
                constraints={"type": "ineq", "fun": constraints},
                method="SLSQP", options={"maxiter": self.max_iter, "ftol": 1e-9},
            )
        except (ValueError, np.linalg.LinAlgError):
            return warm
        if not res.success:
            return warm

        cand = design.copy()
        for (i, j), a in zip(members, res.x * area_scale):
            cand.set_area(i, j, float(min(max(a, mat.min_area), mat.max_area)))

        warm_report = evaluate_constraints(warm, solve(warm, mat), mat)
        cand_report = evaluate_constraints(cand, solve(cand, mat), mat)
        cand_ok = cand_report.max_violation <= 0.0
        lighter = structural_mass(cand, mat) < structural_mass(warm, mat)
        if cand_ok and (lighter or warm_report.max_violation > 0.0):
            return cand
        return warm


class ShapeSizer:
    """Minimize truss weight over the bar areas and the free joint positions.

    An outer SLSQP moves the free joints (not supports or load points); each
    trial geometry is sized by the fully stressed design, and the best one is
    refined by SlsqpSizer. Joints stay in the design domain, bar lengths and
    joint spacing stay within limits, and the result must pass the
    admissibility gate. Otherwise, or if it is not lighter, the area-only
    design is returned. The result is a local optimum near the input geometry.
    """

    def __init__(
        self,
        material: Material,
        spec: DesignSpec,
        *,
        max_members: int = 30,
        max_iter: int = 100,
        fd_step: float = 1.0e-5,
        margin: float = SIZING_MARGIN,
    ) -> None:
        self.material = material
        self.spec = spec
        self.max_members = max_members
        self.max_iter = max_iter
        self.fd_step = fd_step
        self.margin = margin
        # A tightly converged FSD keeps the outer objective smooth enough for
        # finite-difference gradients.
        self._inner = FullyStressedSizer(material, max_iter=200, tol=1.0e-7, margin=margin)
        self._areas_only = SlsqpSizer(material, max_members=max_members, margin=margin)

    def __call__(self, design: TrussDesign) -> TrussDesign:
        return self.size(design)

    def movable_joints(self, design: TrussDesign) -> List[int]:
        """Joints the sizer may move: not protected, not supported, not loaded."""
        protected = self.spec.protected_joints()
        return [
            j for j in design.joints()
            if j not in protected and not any(design.support(j)) and design.load(j) == (0.0, 0.0)
        ]

    def size(self, design: TrussDesign) -> TrussDesign:
        """Return a copy of ``design`` with optimised joint positions and areas."""
        warm = self._areas_only.size(design)
        movable = self.movable_joints(design)
        if not movable or design.n_members == 0 or design.n_members > self.max_members:
            return warm
        try:
            import numpy as np  # pylint: disable=import-outside-toplevel
            from scipy.optimize import minimize  # pylint: disable=import-outside-toplevel
        except ImportError:
            return warm
        from vlmdg.grammar.validate import (  # pylint: disable=import-outside-toplevel
            admissibility_gate,
        )

        mat, spec = self.material, self.spec
        (x_lo, x_hi), (y_lo, y_hi) = spec.x_bounds, spec.y_bounds
        span = np.array([x_hi - x_lo, y_hi - y_lo])
        origin = np.array([x_lo, y_lo])
        joints = design.joints()
        pairs = [(a, b) for idx, a in enumerate(joints) for b in joints[idx + 1:]]
        ref_weight = structural_mass(warm, mat)

        def place(u: "np.ndarray") -> TrussDesign:
            probe = warm.copy()
            for jid, (x, y) in zip(movable, u.reshape(-1, 2) * span + origin):
                probe.move_joint(jid, float(x), float(y))
            return probe

        def fsd_weight(u: "np.ndarray") -> float:
            # SLSQP's line search may step outside the geometry constraints
            # (e.g. two joints on top of each other): score such points as heavy.
            try:
                sized = self._inner.size(place(u))
                fea = solve(sized, mat)
            except (ValueError, ZeroDivisionError, np.linalg.LinAlgError):
                return 1.0e3
            if fea.singular:
                return 1.0e3
            # A failed FSD returns its unsized input: penalise its violation.
            viol = max(evaluate_constraints(sized, fea, mat).max_violation, 0.0)
            return structural_mass(sized, mat) / ref_weight * (1.0 + 10.0 * viol)

        def geometry(u: "np.ndarray") -> "np.ndarray":
            probe = place(u)
            lengths = np.array([probe.member_length(i, j) for i, j in probe.members()])
            spacing = np.array([math.dist(probe.coords(a), probe.coords(b)) for a, b in pairs])
            return np.concatenate([
                lengths / spec.min_member_length - 1.0,
                1.0 - lengths / spec.max_member_length,
                spacing / spec.min_joint_spacing - 1.0,
            ])

        u0 = ((np.array([design.coords(j) for j in movable]) - origin) / span).ravel()
        try:
            res = minimize(
                fsd_weight, np.clip(u0, 0.0, 1.0), method="SLSQP",
                bounds=[(0.0, 1.0)] * u0.size,
                constraints={"type": "ineq", "fun": geometry},
                options={"maxiter": self.max_iter, "eps": self.fd_step},
            )
        except (ValueError, np.linalg.LinAlgError, ZeroDivisionError):
            return warm
        if not np.all(np.isfinite(res.x)):
            return warm

        cand = self._areas_only.size(place(np.clip(res.x, 0.0, 1.0)))
        try:
            fea = solve(cand, mat)
        except (ValueError, ZeroDivisionError):
            return warm
        if fea.singular or not evaluate_constraints(cand, fea, mat).feasible:
            return warm
        if not admissibility_gate(cand, spec).passed:
            return warm
        if structural_mass(cand, mat) < structural_mass(warm, mat):
            return cand
        return warm


SIZER_KINDS = ("auto", "fsd", "slsqp", "shape")


def make_sizer(material: Material, kind: str = "auto", spec: DesignSpec | None = None) -> Sizer:
    """Build a sizer by kind: "fsd", "slsqp", "shape", or "auto".

    "auto" is the fully stressed sizer. "shape" also moves the free joints and
    needs the problem spec.
    """
    if kind in ("fsd", "auto"):
        return FullyStressedSizer(material)
    if kind == "slsqp":
        return SlsqpSizer(material)
    if kind == "shape":
        if spec is None:
            raise ValueError("the 'shape' sizer needs the problem DesignSpec")
        return ShapeSizer(material, spec)
    raise ValueError(f"unknown sizer kind {kind!r}")
