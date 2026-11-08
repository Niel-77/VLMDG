"""Base grammar rules and the admissibility gate."""

import random

from vlmdg.config import Material
from vlmdg.grammar.base_truss import base_truss_grammar
from vlmdg.grammar.validate import admissibility_gate, equilibrium_matrix
from vlmdg.problems import build_cantilever


def test_base_grammar_has_expected_rules():
    _, spec = build_cantilever()
    grammar = base_truss_grammar(spec, Material())
    assert grammar.size == 5
    assert set(grammar.names()) == {
        "add_triangle",
        "split_member",
        "connect_joints",
        "remove_member",
        "move_joint",
    }


def test_starting_design_is_admissible():
    design, spec = build_cantilever()
    report = admissibility_gate(design, spec)
    assert report.passed, report.reasons
    assert report.metrics["n_mechanisms"] == 0.0


def test_each_applicable_rule_yields_a_truss_design():
    design, spec = build_cantilever()
    grammar = base_truss_grammar(spec, Material())
    rng = random.Random(0)
    for rule in grammar.rules:
        if rule.is_applicable(design):
            out = rule(design, rng)
            assert out is not None
            assert out.n_joints >= 3
            assert out.is_connected()


def test_removing_a_brace_creates_a_detectable_mechanism():
    design, spec = build_cantilever()
    design.remove_member(0, 3)  # drop a panel diagonal
    design.remove_member(2, 5)  # drop the other -> top panel becomes a mechanism
    report = admissibility_gate(design, spec)
    assert not report.passed
    assert any("mechanism" in r for r in report.reasons)


def test_equilibrium_matrix_shape():
    design, _ = build_cantilever()
    a_free, free_dofs, members = equilibrium_matrix(design)
    assert a_free.shape == (len(free_dofs), len(members))
    assert len(free_dofs) == 2 * design.n_joints - design.n_reactions
