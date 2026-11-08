"""The rule screen accepts rules that meet the criteria and rejects the rest.

Each test sends one fixed rule through :class:`RuleGenerator` with a fake
provider, on a sized incumbent, and checks the decision and its reason.
"""

import dataclasses

import pytest

from vlmdg.config import VLMConfig
from vlmdg.grammar.base_truss import base_truss_grammar
from vlmdg.optimize.diagnostics import build_diagnostics
from vlmdg.optimize.objective import Objective
from vlmdg.optimize.sizing import FullyStressedSizer
from vlmdg.problems import get_material, get_problem
from vlmdg.rulegen import RuleGenerator
from vlmdg.rulegen.prompt import build_user_prompt
from vlmdg.vlm.base import VLMProvider, VLMResponse, VLMUsage

CONFIG = VLMConfig(provider="mock", attach_image=False)

# Connects the first pair of unconnected joints. The new member gets the
# default area 1.0 m^2, above the area bound, so it must be sized before scoring.
ADD_BRACE = '''
def rule(design, rng):
    js = design.joints()
    pairs = [(i, j) for k, i in enumerate(js) for j in js[k + 1:] if not design.has_member(i, j)]
    rng.shuffle(pairs)
    for i, j in pairs:
        design.add_member(i, j)
        return design
    return design
'''

DANGLING_JOINT = '''
def rule(design, rng):
    j = max(design.joints(), key=lambda n: design.coords(n)[0])
    x, y = design.coords(j)
    new = design.add_joint(x - 1.0, y + 1.0)
    design.add_member(j, new)
    return design
'''

MOVE_SUPPORT = '''
def rule(design, rng):
    for j in design.joints():
        if any(design.support(j)):
            x, y = design.coords(j)
            design.move_joint(j, x + 1.0, y)
            return design
    return design
'''

NO_OP = '''
def rule(design, rng):
    return design
'''

REMOVE_RANDOM_MEMBER = '''
def rule(design, rng):
    i, j = rng.choice(design.members())
    design.remove_member(i, j)
    return design
'''

# Inserts a joint inside a member's triangle region: adds a joint near the
# centroid of three joints and braces it to all three.
ADD_BRACED_JOINT = '''
def rule(design, rng):
    js = design.joints()
    a, b, c = rng.sample(js, 3)
    pts = [design.coords(n) for n in (a, b, c)]
    x = sum(p[0] for p in pts) / 3 + rng.uniform(-0.2, 0.2)
    y = sum(p[1] for p in pts) / 3 + rng.uniform(-0.2, 0.2)
    new = design.add_joint(x, y)
    for n in (a, b, c):
        design.add_member(n, new)
    return design
'''

# Valid on some draws only: half the draws put the joint outside the domain.
SOMETIMES_OUTSIDE = '''
def rule(design, rng):
    js = design.joints()
    a, b, c = rng.sample(js, 3)
    pts = [design.coords(n) for n in (a, b, c)]
    x = sum(p[0] for p in pts) / 3
    y = sum(p[1] for p in pts) / 3
    if rng.random() < 0.5:
        y = 1.0e3
    new = design.add_joint(x, y)
    for n in (a, b, c):
        design.add_member(n, new)
    return design
'''


class FixedProvider(VLMProvider):
    """Return the same rule for every request."""

    name = "fixed"

    def __init__(self, code: str) -> None:
        self.code = code
        self.prompts = []

    def generate(self, system, prompt, images=(), *, max_tokens=None, temperature=None):
        self.prompts.append(prompt)
        return VLMResponse(
            text=f"Rationale.\n```python\n{self.code}\n```",
            model="fixed",
            usage=VLMUsage(requests=1),
        )


def _setup(problem="ten_bar", **spec_changes):
    design, spec = get_problem(problem)
    if spec_changes:
        spec = dataclasses.replace(spec, **spec_changes)
    material = get_material(problem)
    objective = Objective(material)
    design = FullyStressedSizer(material)(design.copy())
    grammar = base_truss_grammar(spec, material)
    diag = build_diagnostics(
        iteration=100, temperature=0.05, stall_index=1, iterations_since_improvement=50,
        best_design=design, best_eval=objective(design), rule_utility=[],
        recent_accepted_rules=[],
    )
    return spec, objective, grammar, diag


def _propose(code, problem="ten_bar", sizer=None, **spec_changes):
    spec, objective, grammar, diag = _setup(problem, **spec_changes)
    provider = FixedProvider(code)
    generator = RuleGenerator(provider, spec, objective, CONFIG, max_proposal_attempts=1,
                              sizer=sizer)
    return generator.propose_rule(diag, grammar, informed=False), provider


# Accepted


def test_new_member_with_default_area_is_accepted():
    """Regression: unsized scoring rejected every new member as infinite energy."""
    proposal, _ = _propose(ADD_BRACE)
    assert proposal.accepted, proposal.reasons
    assert proposal.screen["valid"] == 1


def test_braced_new_joint_is_accepted():
    proposal, _ = _propose(ADD_BRACED_JOINT)
    assert proposal.accepted, proposal.reasons


def test_rule_valid_on_only_some_draws_is_accepted():
    proposal, _ = _propose(SOMETIMES_OUTSIDE)
    assert proposal.accepted, proposal.reasons


def test_bridge_accepts_a_planar_change():
    # A random braced joint often crosses the fan; at least one of the draws
    # must find a planar placement.
    proposal, _ = _propose(ADD_BRACED_JOINT, problem="bridge")
    assert proposal.accepted, (proposal.reasons, proposal.screen)


def test_rule_generator_uses_the_given_sizer():
    calls = {"n": 0}
    material = get_material("ten_bar")
    base = FullyStressedSizer(material)

    def spy(design):
        calls["n"] += 1
        return base(design)

    proposal, _ = _propose(ADD_BRACE, sizer=spy)
    assert proposal.accepted, proposal.reasons
    assert calls["n"] >= 1


# Rejected


def test_dangling_joint_is_rejected_as_a_mechanism():
    proposal, _ = _propose(DANGLING_JOINT)
    assert not proposal.accepted
    assert any("mechanism" in r or "Maxwell" in r for r in proposal.reasons), proposal.reasons


def test_moving_a_support_is_rejected():
    proposal, _ = _propose(MOVE_SUPPORT)
    assert not proposal.accepted
    assert any("protected joint" in r or "support" in r for r in proposal.reasons), proposal.reasons


def test_no_op_rule_is_rejected():
    proposal, _ = _propose(NO_OP)
    assert not proposal.accepted
    assert any("never changed" in r for r in proposal.reasons), proposal.reasons
    assert proposal.screen["unchanged"] == proposal.screen["draws"]


def test_duplicate_of_remove_member_is_rejected():
    proposal, _ = _propose(REMOVE_RANDOM_MEMBER)
    assert not proposal.accepted
    assert any("duplicates existing rule 'remove_member'" in r for r in proposal.reasons), \
        proposal.reasons


def test_joint_limit_is_enforced():
    proposal, _ = _propose(ADD_BRACED_JOINT, max_joints=6)
    assert not proposal.accepted
    assert any("max_joints" in r for r in proposal.reasons), proposal.reasons


def test_crossing_member_is_rejected_on_the_planar_bridge():
    # Joint at (2, 3) braced to the deck at x = 0 and x = 8: the member to
    # (8, 0) crosses the fan member from the apex (12, 4) to (4, 0) at (6, 1).
    crossing = '''
def rule(design, rng):
    left = design.nearest_joints(0.0, 0.0)[0]
    right = design.nearest_joints(8.0, 0.0)[0]
    apex = design.nearest_joints(12.0, 4.0)[0]
    new = design.add_joint(2.0, 3.0)
    for n in (left, right, apex):
        design.add_member(n, new)
    return design
'''
    proposal, _ = _propose(crossing, problem="bridge")
    assert not proposal.accepted
    assert any("cross" in r for r in proposal.reasons), proposal.reasons


def test_ten_bar_allows_crossing_members():
    # The ten-bar start design has crossed diagonals and passes the gate, and
    # a rule that keeps them is not rejected for crossing.
    proposal, _ = _propose(ADD_BRACE)
    assert proposal.accepted
    assert not any("cross" in r for r in proposal.reasons)


# Prompt


@pytest.mark.parametrize("problem, phrase", [("ten_bar", "MAY cross"), ("bridge", "PLANAR")])
def test_prompt_states_the_problem_limits(problem, phrase):
    spec, _, grammar, diag = _setup(problem)
    for informed in (False, True):
        prompt = build_user_prompt(diag, grammar, informed=informed, spec=spec)
        assert f"at most {spec.max_joints} joints" in prompt
        assert f"joints at least {spec.min_joint_spacing:g} apart" in prompt
        assert phrase in prompt
        assert "joint limit" not in prompt


def test_prompt_warns_at_the_joint_limit():
    spec, _, grammar, diag = _setup("ten_bar", max_joints=6)
    prompt = build_user_prompt(diag, grammar, informed=False, spec=spec)
    assert "at the joint limit" in prompt


def test_accepted_rules_are_numbered_without_gaps():
    spec, objective, grammar, diag = _setup()
    names = []
    for code in (NO_OP, ADD_BRACE, DANGLING_JOINT, ADD_BRACED_JOINT):
        gen_provider = FixedProvider(code)
        if not names:
            generator = RuleGenerator(gen_provider, spec, objective, CONFIG,
                                      max_proposal_attempts=1)
        generator.provider = gen_provider
        proposal = generator.propose_rule(diag, grammar, informed=False)
        if proposal.accepted:
            names.append(proposal.rule.name)
            grammar.add_rule(proposal.rule)
        elif not names:
            names.append(None)
    assert [n for n in names if n] == ["vlm_rule_1", "vlm_rule_2"]
