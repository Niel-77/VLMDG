"""Invalid member areas are never scored as feasible and never crash a search."""

import math

import pytest

from vlmdg.config import AnnealConfig
from vlmdg.grammar.base_truss import base_truss_grammar
from vlmdg.grammar.rule import FunctionRule
from vlmdg.grammar.validate import admissibility_gate
from vlmdg.optimize.objective import Objective
from vlmdg.optimize.shape_annealing import ShapeAnnealer
from vlmdg.optimize.sizing import make_sizer
from vlmdg.problems import get_material, get_problem


def _sized_ten_bar():
    design, spec = get_problem("ten_bar")
    material = get_material("ten_bar")
    return make_sizer(material, "fsd", spec)(design.copy()), spec, material


def test_sized_design_is_valid_and_feasible():
    design, _, material = _sized_ten_bar()
    evaluation = Objective(material).evaluate(design)
    assert evaluation.feasible
    assert not evaluation.constraints.invalid_area


@pytest.mark.parametrize("area", [-0.01, 0.0, math.nan, math.inf, 0.5, 1e-6])
def test_invalid_area_is_infeasible(area):
    design, _, material = _sized_ten_bar()
    member = design.members()[0]
    design.set_area(*member, area)
    evaluation = Objective(material).evaluate(design)
    assert not evaluation.feasible
    assert evaluation.energy == math.inf
    assert member in evaluation.constraints.invalid_area


def test_all_negative_areas_do_not_give_negative_weight():
    design, _, material = _sized_ten_bar()
    for member in design.members():
        design.set_area(*member, -design.area(*member))
    evaluation = Objective(material).evaluate(design)
    assert not evaluation.feasible
    assert evaluation.energy == math.inf


@pytest.mark.parametrize("area", [-0.01, 0.0, math.nan, math.inf])
def test_gate_rejects_non_positive_or_non_finite_area(area):
    design, spec, _ = _sized_ten_bar()
    design.set_area(*design.members()[0], area)
    report = admissibility_gate(design, spec)
    assert not report.passed
    assert any("area" in reason for reason in report.reasons)


def test_search_survives_a_rule_that_sets_a_zero_area():
    design, spec = get_problem("ten_bar")
    material = get_material("ten_bar")

    def zero_area(d, rng):
        d.set_area(*d.members()[0], 0.0)
        return d

    grammar = base_truss_grammar(spec, material)
    grammar.add_rule(FunctionRule(name="zero_area", description="", fn=zero_area,
                                  origin="vlm", source=""))
    annealer = ShapeAnnealer(spec, Objective(material),
                             AnnealConfig(max_iterations=60, stall_window=1000))
    result = annealer.run(design.copy(), grammar)
    assert math.isfinite(result.best_eval.energy)
    assert not Objective(material).evaluate(result.best_design).constraints.invalid_area
