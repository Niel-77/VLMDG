"""A rule that cannot change the design does not use up a search iteration."""

import random

from vlmdg.config import AnnealConfig
from vlmdg.grammar.base_truss import base_truss_grammar
from vlmdg.grammar.registry import Grammar
from vlmdg.grammar.rule import FunctionRule
from vlmdg.optimize.objective import Objective
from vlmdg.optimize.shape_annealing import ShapeAnnealer
from vlmdg.problems import get_material, get_problem


def _rule(name, fn):
    return FunctionRule(name=name, description="", fn=fn, origin="vlm", source="")


def _no_op(design, rng):
    return design


def _returns_none(design, rng):
    return None


def test_no_op_rule_is_redrawn_in_the_same_iteration():
    design, spec = get_problem("ten_bar")
    material = get_material("ten_bar")
    grammar = base_truss_grammar(spec, material)
    grammar.add_rule(_rule("no_op", _no_op))
    grammar.add_rule(_rule("returns_none", _returns_none))
    for seed in range(50):
        rule, candidate = ShapeAnnealer._draw_move(design, grammar, random.Random(seed))
        assert rule.name not in ("no_op", "returns_none")
        assert candidate is not None


def test_grammar_of_only_no_op_rules_ends_the_draw():
    design, _ = get_problem("ten_bar")
    grammar = Grammar([_rule("no_op", _no_op), _rule("returns_none", _returns_none)])
    rule, candidate = ShapeAnnealer._draw_move(design, grammar, random.Random(0))
    assert candidate is None
    assert rule is not None


def test_no_op_rule_is_counted_as_proposed_but_not_applied():
    design, spec = get_problem("ten_bar")
    material = get_material("ten_bar")
    grammar = base_truss_grammar(spec, material)
    grammar.add_rule(_rule("no_op", _no_op))
    ShapeAnnealer(spec, Objective(material),
                  AnnealConfig(max_iterations=200, stall_window=1000)).run(design.copy(), grammar)
    stats = {u["name"]: u for u in grammar.utility_table()}
    assert stats["no_op"]["proposed"] > 0
    assert stats["no_op"]["applied"] == 0
    applied = sum(u["applied"] for u in stats.values())
    assert applied == 200  # every iteration made a real move
