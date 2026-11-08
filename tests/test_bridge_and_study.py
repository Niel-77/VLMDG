"""Tests for the planar bridge problem and the study tooling (snapshots, rule pools)."""

import json
import random

from vlmdg.config import AnnealConfig, Material
from vlmdg.design.truss import TrussDesign
from vlmdg.grammar.base_truss import base_truss_grammar
from vlmdg.grammar.validate import admissibility_gate, check_planarity
from vlmdg.optimize.diagnostics import build_diagnostics
from vlmdg.optimize.objective import Objective
from vlmdg.optimize.shape_annealing import ShapeAnnealer
from vlmdg.problems import build_cantilever, get_problem
from vlmdg.problems.bridge import reference_designs
from vlmdg.rulegen.prompt import build_user_prompt
from vlmdg.runner import load_rule_pool


def _square(members):
    joints = {0: (0.0, 0.0), 1: (1.0, 0.0), 2: (1.0, 1.0), 3: (0.0, 1.0)}
    return TrussDesign.from_spec(joints=joints, members=members, supports={}, loads={})


def test_planarity_detects_crossing_diagonals():
    braced = _square([(0, 1), (1, 2), (2, 3), (3, 0), (0, 2)])
    crossed = _square([(0, 1), (1, 2), (2, 3), (3, 0), (0, 2), (1, 3)])
    assert check_planarity(braced) == []
    assert any("cross" in r for r in check_planarity(crossed))


def test_planarity_detects_joint_on_member():
    joints = {0: (0.0, 0.0), 1: (2.0, 0.0), 2: (1.0, 0.0), 3: (1.0, 1.0)}
    design = TrussDesign.from_spec(
        joints=joints, members=[(0, 1), (2, 3), (0, 3)], supports={}, loads={}
    )
    assert any("lies on member (0, 1)" in r for r in check_planarity(design))


def test_planarity_detects_collinear_overlap():
    joints = {0: (0.0, 0.0), 1: (2.0, 0.0), 2: (1.0, 0.0)}
    design = TrussDesign.from_spec(joints=joints, members=[(0, 1), (0, 2)], supports={}, loads={})
    assert check_planarity(design)


def test_bridge_start_and_references_are_admissible():
    design0, spec = get_problem("bridge")
    assert spec.planar
    assert admissibility_gate(design0, spec).passed
    for joints, members in reference_designs().values():
        d = TrussDesign.from_spec(
            joints=joints, members=members, supports=spec.supports, loads=spec.loads
        )
        assert admissibility_gate(d, spec).passed


def test_planar_invariant_only_in_planar_prompts():
    material = Material()
    for name, expected in (("bridge", True), ("cantilever", False)):
        design, spec = get_problem(name)
        diag = build_diagnostics(
            iteration=1, temperature=0.1, stall_index=1, iterations_since_improvement=1,
            best_design=design, best_eval=Objective(material)(design),
            rule_utility=[], recent_accepted_rules=[],
        )
        grammar = base_truss_grammar(spec, material)
        prompt = build_user_prompt(diag, grammar, informed=True, spec=spec)
        assert ("PLANAR" in prompt) is expected


def test_annealer_records_snapshots():
    design0, spec = build_cantilever()
    material = Material()
    config = AnnealConfig(max_iterations=40, stall_window=1000, seed=0)
    annealer = ShapeAnnealer(spec, Objective(material), config, snapshot_every=10)
    result = annealer.run(design0, base_truss_grammar(spec, material))
    its = [s["iteration"] for s in result.snapshots]
    assert its[0] == 0 and 10 in its and 40 in its
    assert {"current", "best", "best_mass", "temperature"} <= set(result.snapshots[0])


def test_load_rule_pool_marks_rules_pooled(tmp_path):
    source = '''def rule(design, rng):
    """Remove a random member."""
    i, j = rng.choice(sorted(design.members()))
    design.remove_member(i, j)
    return design
'''
    path = tmp_path / "pool.json"
    path.write_text(json.dumps([{"name": "p0", "source": source}]))
    (rule,) = load_rule_pool(path)
    assert rule.name == "p0" and rule.origin == "pooled"
    design0, _ = build_cantilever()
    out = rule(design0.copy(), random.Random(0))
    assert out.n_members == design0.n_members - 1
