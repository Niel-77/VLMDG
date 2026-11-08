"""Behavioral duplicate check and re-prompting for generated rules."""

from vlmdg.config import Material, VLMConfig
from vlmdg.grammar.base_truss import base_truss_grammar
from vlmdg.grammar.novelty import check_novelty, design_signature
from vlmdg.optimize.objective import Objective
from vlmdg.problems import build_cantilever
from vlmdg.rulegen import RuleGenerator
from vlmdg.rulegen.sandbox import compile_rule
from vlmdg.vlm import get_provider
from vlmdg.vlm.base import VLMProvider, VLMResponse
from vlmdg.vlm.mock_provider import MockVLMProvider

from vlmdg.optimize.diagnostics import build_diagnostics


def _diagnostics(design, objective):
    return build_diagnostics(
        iteration=100,
        temperature=0.5,
        stall_index=1,
        iterations_since_improvement=200,
        best_design=design,
        best_eval=objective(design),
        rule_utility=[],
        recent_accepted_rules=[],
    )

NOOP_SOURCE = '''def rule(design, rng):
    """Do nothing at all.

    Second paragraph that should not appear in the description.
    """
    return design
'''

REMOVE_SOURCE = '''def rule(design, rng):
    """Remove a random member."""
    i, j = rng.choice(sorted(design.members()))
    design.remove_member(i, j)
    return design
'''


def test_description_is_first_docstring_paragraph():
    rule = compile_rule(NOOP_SOURCE, "noop")
    assert rule.description == "Do nothing at all."


def test_signature_ignores_areas():
    design, _ = build_cantilever()
    other = design.copy()
    i, j = next(iter(other.members()))
    other.set_area(i, j, other.area(i, j) * 2.0)
    assert design_signature(design) == design_signature(other)


def test_noop_rule_is_rejected():
    design, spec = build_cantilever()
    grammar = base_truss_grammar(spec, Material())
    report = check_novelty(compile_rule(NOOP_SOURCE, "noop"), grammar.rules, [design])
    assert not report.novel
    assert report.n_outcomes == 0
    assert "never changed" in report.reason


def test_copy_of_existing_rule_is_duplicate():
    design, _ = build_cantilever()
    first = compile_rule(REMOVE_SOURCE, "first")
    second = compile_rule(REMOVE_SOURCE, "second")
    report = check_novelty(second, [first], [design])
    assert not report.novel
    assert report.duplicate_of == "first"
    assert report.overlap == 1.0


def test_new_move_is_novel():
    design, _ = build_cantilever()
    report = check_novelty(compile_rule(REMOVE_SOURCE, "cand"), [], [design])
    assert report.novel and report.n_outcomes > 0


class _ScriptedProvider(VLMProvider):
    """Returns a no-op rule first, then the mock provider's working rule."""

    name = "scripted"

    def __init__(self):
        self.prompts = []
        self._mock = MockVLMProvider()

    def generate(self, system, prompt, images=(), *, max_tokens=None, temperature=None):
        self.prompts.append(prompt)
        if len(self.prompts) == 1:
            return VLMResponse(text="```python\n" + NOOP_SOURCE + "```", model="scripted")
        return self._mock.generate(system, prompt, images)


def _generator(provider, spec, objective):
    return RuleGenerator(
        provider=provider,
        spec=spec,
        objective=objective,
        vlm_config=VLMConfig(provider="mock", attach_image=False),
    )


def test_rejected_proposal_is_reprompted_with_feedback():
    design, spec = build_cantilever()
    objective = Objective(Material())
    grammar = base_truss_grammar(spec, Material())
    provider = _ScriptedProvider()
    proposal = _generator(provider, spec, objective).propose_rule(
        _diagnostics(design, objective), grammar, informed=True
    )
    assert proposal.accepted, proposal.reasons
    assert len(proposal.rejected_attempts) == 1
    assert len(provider.prompts) == 2
    assert "REJECTED" in provider.prompts[1]
    assert "never changed" in provider.prompts[1]


def test_repeated_mock_rule_is_rejected_as_duplicate():
    design, spec = build_cantilever()
    objective = Objective(Material())
    grammar = base_truss_grammar(spec, Material())
    provider = get_provider("mock", config=VLMConfig(provider="mock", attach_image=False))
    generator = _generator(provider, spec, objective)
    diag = _diagnostics(design, objective)
    first = generator.propose_rule(diag, grammar, informed=True)
    assert first.accepted
    grammar.add_rule(first.rule)
    second = generator.propose_rule(diag, grammar, informed=True)
    assert not second.accepted
    assert any("duplicates" in r for r in second.reasons)
