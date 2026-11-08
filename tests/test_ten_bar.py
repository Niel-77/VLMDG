"""The Reddy & Cagan (1995) 10-bar truss benchmark and its sizing step."""

import dataclasses
import math

from vlmdg.config import Material
from vlmdg.fea import solve
from vlmdg.fea.checks import evaluate_constraints, structural_mass
from vlmdg.grammar.validate import admissibility_gate
from vlmdg.optimize.sizing import FullyStressedSizer, ShapeSizer, SlsqpSizer, make_sizer
from vlmdg.problems import build_ten_bar, get_material


def test_ten_bar_matches_the_paper_specification():
    design, spec = build_ten_bar()

    # Six joints on a 2 x 9.1 m grid, ten bars.
    assert design.n_joints == 6
    assert design.n_members == 10

    # Two pinned supports 9.1 m apart vertically, both reaction DOFs fixed.
    assert set(spec.fixed_joints) == {0, 1}
    assert design.coords(0) == (0.0, 9.1) and design.coords(1) == (0.0, 0.0)
    assert design.support(0) == (True, True) and design.support(1) == (True, True)
    assert design.n_reactions == 4

    # 445,000 N downward at each of the two bottom free joints.
    assert design.load(3) == (0.0, -445_000.0)
    assert design.load(5) == (0.0, -445_000.0)

    # Starting design is a valid member of the language and analysable.
    assert admissibility_gate(design, spec).passed
    assert not solve(design, Material.ten_bar()).singular


def test_get_material_ten_bar_is_the_paper_material():
    mat = get_material("ten_bar")
    assert mat.stress_allowable == 172e6          # 172 MPa, printed in the paper
    assert mat.density == 27e3                    # 27 kN/m^3, printed in the paper
    assert math.isclose(mat.section_factor, 1.0 / (4.0 * math.pi))


def test_fully_stressed_sizing_makes_the_ten_bar_feasible():
    design, _ = build_ten_bar()
    mat = Material.ten_bar()

    start = evaluate_constraints(design, solve(design, mat), mat)
    assert start.max_violation > 0.0             # unsized start is overstressed

    sized = FullyStressedSizer(mat)(design)
    report = evaluate_constraints(sized, solve(sized, mat), mat)
    assert report.feasible
    # Buckling on the long unbraced diagonals makes this a heavy design - same
    # order of magnitude as the paper's stress+buckling 10-bar weight (~24 kN).
    weight = structural_mass(sized, mat)
    assert 15_000.0 < weight < 60_000.0


def test_slsqp_sizer_is_feasible_and_no_heavier_than_fsd():
    design, _ = build_ten_bar()
    mat = Material.ten_bar()

    fsd = FullyStressedSizer(mat)(design)
    slsqp = SlsqpSizer(mat)(design)

    fsd_w = structural_mass(fsd, mat)
    slsqp_w = structural_mass(slsqp, mat)
    assert evaluate_constraints(slsqp, solve(slsqp, mat), mat).feasible
    assert slsqp_w <= fsd_w * 1.05               # never much worse than the warm start


def test_shape_sizer_matches_paper_stress_only_shape_optimum():
    """Moving joints 2 and 4 reaches the papers' 6,880 N (1995a) / 1,546 lb (1992)."""
    design, spec = build_ten_bar()
    mat = dataclasses.replace(Material.ten_bar(), section_factor=1.0e9)  # no buckling

    shaped = ShapeSizer(mat, spec)(design)
    weight = structural_mass(shaped, mat)

    assert evaluate_constraints(shaped, solve(shaped, mat), mat).feasible
    assert admissibility_gate(shaped, spec).passed
    assert abs(weight - 6880.0) / 6880.0 < 0.02
    # Supports and load points stay put; only the free top joints move.
    for joint in (0, 1, 3, 5):
        assert shaped.coords(joint) == design.coords(joint)
    assert shaped.coords(2) != design.coords(2)


def test_shape_sizer_with_buckling_is_lighter_than_area_only():
    design, spec = build_ten_bar()
    mat = Material.ten_bar()

    area_only = structural_mass(FullyStressedSizer(mat)(design), mat)
    shaped = make_sizer(mat, "shape", spec)(design)

    assert evaluate_constraints(shaped, solve(shaped, mat), mat).feasible
    assert structural_mass(shaped, mat) < 0.85 * area_only


def test_slsqp_buckling_margin_tightens_the_limit():
    """Regression: the SLSQP buckling margin used to loosen the limit by 3%."""
    design, _ = build_ten_bar()
    mat = Material.ten_bar()

    fsd_w = structural_mass(FullyStressedSizer(mat)(design), mat)
    slsqp = SlsqpSizer(mat, margin=0.01)(design)

    assert evaluate_constraints(slsqp, solve(slsqp, mat), mat).feasible
    assert structural_mass(slsqp, mat) < fsd_w
