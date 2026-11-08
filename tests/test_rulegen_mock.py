"""The VLM rule-generation pipeline with the offline mock provider."""

from vlmdg.config import Material, VLMConfig
from vlmdg.grammar.base_truss import base_truss_grammar
from vlmdg.optimize.diagnostics import build_diagnostics
from vlmdg.optimize.objective import Objective
from vlmdg.problems import build_cantilever
from vlmdg.rulegen import RuleGenerator, extract_code_block, run_execution_gate
from vlmdg.vlm import get_provider


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


def test_extract_code_block_prefers_python_fence():
    text = "blah\n```python\ndef rule(design, rng):\n    return design\n```\ntail"
    assert extract_code_block(text).startswith("def rule(")


def test_execution_gate_repairs_then_passes():
    design, _ = build_cantilever()
    bad = "def rule(design, rng):\n    return undefined_name\n"
    good = "def rule(design, rng):\n    return design\n"
    calls = {"n": 0}

    def repair(_code, _err):
        calls["n"] += 1
        return good

    result = run_execution_gate(bad, "t", [design], max_repairs=3, repair_fn=repair)
    assert result.passed
    assert calls["n"] == 1
    assert result.attempts == 2


def test_mock_proposal_is_accepted_and_registered():
    design, spec = build_cantilever()
    objective = Objective(Material())
    grammar = base_truss_grammar(spec, Material())
    generator = RuleGenerator(
        provider=get_provider("mock", config=VLMConfig(provider="mock", attach_image=False)),
        spec=spec,
        objective=objective,
        vlm_config=VLMConfig(provider="mock", attach_image=False),
    )
    proposal = generator.propose_rule(_diagnostics(design, objective), grammar, informed=True)
    assert proposal.accepted, proposal.reasons
    assert proposal.rule is not None
    assert proposal.rule.origin == "vlm"

    name = grammar.add_rule(proposal.rule)
    assert name in grammar.names()
    assert grammar.size == 6
